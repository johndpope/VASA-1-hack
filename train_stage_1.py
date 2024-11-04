import argparse
import torch
import model
import cv2 as cv
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import models
from model import PerceptualLoss,IdentitySimilarityLoss, PairwiseTransferLoss,crop_and_warp_face, get_foreground_mask,remove_background_and_convert_to_rgb,apply_warping_field
import mediapipe as mp
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils
import time
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable

from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from lpips import LPIPS
from EmoDataset import EMODataset

import wandb
from accelerate import Accelerator
from accelerate.utils import LoggerType
import accelerate
from tqdm import tqdm
from utils import get_vasa_exp_name


from helper import log_grad_flow,consistent_sub_sample,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



# work from here https://github.com/johndpope/MegaPortrait-hack


# Function to calculate FID
def calculate_fid(real_images, fake_images):
    real_images = real_images.detach().cpu().numpy()
    fake_images = fake_images.detach().cpu().numpy()
    mu1, sigma1 = real_images.mean(axis=0), np.cov(real_images, rowvar=False)
    mu2, sigma2 = fake_images.mean(axis=0), np.cov(fake_images, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to calculate CSIM (Cosine Similarity)
def calculate_csim(real_features, fake_features):
    csim = cosine_similarity(real_features.detach().cpu().numpy(), fake_features.detach().cpu().numpy())
    return np.mean(csim)

# Function to calculate LPIPS
def calculate_lpips(real_images, fake_images):
    lpips_model = LPIPS(net='alex').cuda()  # 'alex', 'vgg', 'squeeze'
    lpips_scores = []
    for real, fake in zip(real_images, fake_images):
        real = real.unsqueeze(0).cuda()
        fake = fake.unsqueeze(0).cuda()
        lpips_score = lpips_model(real, fake)
        lpips_scores.append(lpips_score.item())
    return np.mean(lpips_scores)

# align to cyclegan
def discriminator_loss(real_pred, fake_pred, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = torch.mean((real_pred - 1)**2)
        fake_loss = torch.mean(fake_pred**2)
    elif loss_type == 'vanilla':
        real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    
    return ((real_loss + fake_loss) * 0.5).requires_grad_()


def cosine_loss(positive_pairs, negative_pairs, margin=0.5, scale=5):
    """
    Calculates the cosine loss for the positive and negative pairs.

    Args:
        positive_pairs (list): List of tuples containing positive pairs (z_i, z_j).
        negative_pairs (list): List of tuples containing negative pairs (z_i, z_j).
        margin (float): Margin value for the cosine distance (default: 0.5).
        scale (float): Scaling factor for the cosine distance (default: 5).

    Returns:
        torch.Tensor: Cosine loss value.
    """
    def cosine_distance(z_i, z_j):
        # Normalize the feature vectors
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Calculate the cosine similarity
        cos_sim = torch.sum(z_i * z_j, dim=-1)
        
        # Apply the scaling and margin
        cos_dist = scale * (cos_sim - margin)
        
        return cos_dist

    # Calculate the cosine distance for positive pairs
    pos_cos_dist = [cosine_distance(z_i, z_j) for z_i, z_j in positive_pairs]
    pos_cos_dist = torch.stack(pos_cos_dist)

    # Calculate the cosine distance for negative pairs
    neg_cos_dist = [cosine_distance(z_i, z_j) for z_i, z_j in negative_pairs]
    neg_cos_dist = torch.stack(neg_cos_dist)

    # Calculate the cosine loss
    loss = -torch.log(torch.exp(pos_cos_dist) / (torch.exp(pos_cos_dist) + torch.sum(torch.exp(neg_cos_dist))))
    
    return loss.mean().requires_grad_()

class VASAStage1Trainer:
    def __init__(self, cfg, Gbase, Dbase):
        self.cfg = cfg
        self.Gbase = Gbase
        self.Dbase = Dbase
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            log_with="wandb",
            mixed_precision="fp16"
        )
        
        experiment_name =  "test1" #get_vasa_exp_name(config)
        print(f"Generated experiment name: {experiment_name}")

        # Initialize trackers for experiment logging
        self.accelerator.init_trackers(
            project_name=cfg.training.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {
                "name":experiment_name,
                "dir": cfg.training.log_dir
            }}
        )
        
        # Initialize loss functions on correct device
        self.perceptual_loss = LPIPS(net='alex').to(self.accelerator.device)
        self.pairwise_transfer_loss = PairwiseTransferLoss()
        self.identity_loss = IdentitySimilarityLoss()
        
        # Setup optimizers
        self.optimizer_G = torch.optim.AdamW(
            self.Gbase.parameters(),
            lr=cfg.training.lr,
            betas=(0.5, 0.999),
            weight_decay=cfg.training.weight_decay
        )
        self.optimizer_D = torch.optim.AdamW(
            self.Dbase.parameters(),
            lr=cfg.training.lr,
            betas=(0.5, 0.999),
            weight_decay=cfg.training.weight_decay
        )
        
        # Setup schedulers
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G,
            T_max=cfg.training.base_epochs,
            eta_min=cfg.training.min_lr
        )
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D,
            T_max=cfg.training.base_epochs,
            eta_min=cfg.training.min_lr
        )

    def train_step(self, batch):
        """Single training step processing all frames in batch"""
        metrics = {
            'loss_G': 0,
            'loss_D': 0,
            'loss_perceptual': 0,
            'loss_gan': 0,
            'loss_pairwise': 0,
            'loss_identity': 0,
            'loss_motion': 0
        }

        # Extract frame sequences
        source_frames = batch['source_frames']
        driving_frames = batch['driving_frames']
        source_frames_star = batch['source_frames_star']
        driving_frames_star = batch['driving_frames_star']

        num_frames = len(driving_frames)
        len_source_frames = len(source_frames)
        len_driving_frames = len(driving_frames)
        len_source_frames_star = len(source_frames_star)
        len_driving_frames_star = len(driving_frames_star)

        # Process all frames in sequence
        for idx in range(num_frames):
            # Get current frames with wraparound
            source_frame = source_frames[idx % len_source_frames].to(self.accelerator.device)
            driving_frame = driving_frames[idx % len_driving_frames].to(self.accelerator.device)
            source_frame_star = source_frames_star[idx % len_source_frames_star].to(self.accelerator.device)
            driving_frame_star = driving_frames_star[idx % len_driving_frames_star].to(self.accelerator.device)

            with torch.cuda.amp.autocast():
                # Generator forward passes
                pred_frame, pred_pyramids = self.Gbase(source_frame, driving_frame)
                cross_frame, _ = self.Gbase(source_frame_star, driving_frame)

                # Compute perceptual pyramid loss
                loss_perceptual = 0
                for scale, pred in pred_pyramids.items():
                    target = F.interpolate(
                        driving_frame, 
                        size=pred.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    loss_perceptual += self.perceptual_loss(pred, target)

                # Extract motion codes
                _, _, zs = self.Gbase.motionEncoder(source_frame)
                _, _, zd = self.Gbase.motionEncoder(driving_frame)
                _, _, zs_star = self.Gbase.motionEncoder(source_frame_star)
                _, _, zd_star = self.Gbase.motionEncoder(driving_frame_star)
                _, _, z_pred = self.Gbase.motionEncoder(pred_frame)
                _, _, z_cross = self.Gbase.motionEncoder(cross_frame)

                # GAN loss
                pred_fake = self.Dbase(pred_frame, source_frame)
                loss_gan = -pred_fake.mean()

                # Get the next frame index for disentanglement
                next_idx = (idx + 20) % len_source_frames
                next_frame = source_frames[next_idx].to(self.accelerator.device)
                next_frame_star = source_frames_star[next_idx % len_source_frames_star].to(self.accelerator.device)

                # Compute disentanglement losses
                loss_pairwise = self.pairwise_transfer_loss(
                    self.Gbase,
                    source_frame,
                    next_frame
                )
                
                loss_identity = self.identity_loss(
                    source_frame_star,
                    next_frame_star
                )

                # Motion contrastive loss
                P = [(z_pred, zd), (z_cross, zd)]
                N = [(z_pred, zd_star), (z_cross, zd_star)]
                loss_motion = cosine_loss(P, N)

                # Total generator loss
                loss_G = (
                    self.cfg.loss.perceptual_weight * loss_perceptual +
                    self.cfg.loss.gan_weight * loss_gan +
                    self.cfg.loss.pairwise_weight * loss_pairwise +
                    self.cfg.loss.identity_weight * loss_identity +
                    self.cfg.loss.motion_weight * loss_motion
                )

                # Train discriminator
                pred_real = self.Dbase(driving_frame, source_frame)
                pred_fake = self.Dbase(pred_frame.detach(), source_frame)
                loss_D = discriminator_loss(pred_real, pred_fake)

            # Accumulate losses
            metrics['loss_G'] += loss_G.item()
            metrics['loss_D'] += loss_D.item()
            metrics['loss_perceptual'] += loss_perceptual.item()
            metrics['loss_gan'] += loss_gan.item()
            metrics['loss_pairwise'] += loss_pairwise.item()
            metrics['loss_identity'] += loss_identity.item()
            metrics['loss_motion'] += loss_motion.item()

            # Optional: Save sample images
            if idx == 0 and self.accelerator.is_local_main_process:  # Save only first frame
                sample_images = {
                    'source': source_frame,
                    'driving': driving_frame,
                    'predicted': pred_frame,
                    'cross_reenacted': cross_frame
                }
                vutils.save_image(
                    torch.cat([img for img in sample_images.values()]),
                    f"{self.cfg.training.sample_dir}/train_step_{self.global_step}.png",
                    nrow=len(sample_images)
                )

        # Average metrics over all frames
        for key in metrics:
            metrics[key] /= num_frames

        return metrics

    def train(self, train_loader, start_epoch=0):
        """Full training loop"""
        self.global_step = start_epoch * len(train_loader)

        # Prepare for distributed training
        (self.Gbase, self.Dbase, self.optimizer_G, self.optimizer_D, 
        self.scheduler_G, self.scheduler_D, train_loader) = self.accelerator.prepare(
            self.Gbase, self.Dbase, self.optimizer_G, self.optimizer_D,
            self.scheduler_G, self.scheduler_D, train_loader
        )

        for epoch in range(start_epoch, self.cfg.training.base_epochs):
            self.Gbase.train()
            self.Dbase.train()
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.cfg.training.base_epochs}"
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Generator update
                self.optimizer_G.zero_grad()
                
                with self.accelerator.accumulate(self.Gbase):
                    metrics = self.train_step(batch)
                    self.accelerator.backward(metrics['loss_G'])
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.Gbase.parameters(), 1.0)
                    
                    self.optimizer_G.step()

                # Discriminator update
                self.optimizer_D.zero_grad()
                
                with self.accelerator.accumulate(self.Dbase):
                    self.accelerator.backward(metrics['loss_D'])
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.Dbase.parameters(), 1.0)
                    
                    self.optimizer_D.step()

                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f"{metrics['loss_G']:.4f}",
                    'D_Loss': f"{metrics['loss_D']:.4f}"
                })

                # Log metrics
                if self.global_step % self.cfg.training.log_interval == 0:
                    self.accelerator.log(
                        {f"train/{k}": v for k, v in metrics.items()},
                        step=self.global_step
                    )

                self.global_step += 1

            # Step schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Save checkpoint
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                self.save_checkpoint(
                    f"checkpoint_epoch_{epoch+1}.pt",
                    metrics
                )

        self.accelerator.end_training()
    def log_samples(self, batch):
        """Log sample images to wandb"""
        if self.accelerator.is_local_main_process:
            with torch.no_grad():
                source_frame = batch['source_frames'][:4]  # Take first 4 samples
                driving_frame = batch['driving_frames'][:4]
                
                pred_frame, _ = self.Gbase(source_frame, driving_frame)
                
                # Create grid of images
                images = {
                    "source": self.accelerator.gather(source_frame),
                    "driving": self.accelerator.gather(driving_frame),
                    "generated": self.accelerator.gather(pred_frame)
                }
                
                self.accelerator.log({"samples": images})

    def save_checkpoint(self, filename, metrics):
        """Save training checkpoint"""
        # Unwrap models before saving
        unwrapped_Gbase = self.accelerator.unwrap_model(self.Gbase)
        unwrapped_Dbase = self.accelerator.unwrap_model(self.Dbase)
        
        checkpoint = {
            'Gbase_state_dict': unwrapped_Gbase.state_dict(),
            'Dbase_state_dict': unwrapped_Dbase.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'metrics': metrics,
            'config': self.cfg
        }
        
        self.accelerator.save(checkpoint, filename)

    

    
def load_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model_G.load_state_dict(checkpoint['model_G_state_dict'])
        model_D.load_state_dict(checkpoint['model_D_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        start_epoch = 0
    return start_epoch



def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        remove_background=True,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform,
        apply_crop_warping=True
    )


    
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)

    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator().to(device)
    
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)

    # Load checkpoint if available
    checkpoint_path = cfg.training.checkpoint_path
    start_epoch = load_checkpoint(checkpoint_path, Gbase, Dbase, optimizer_G, optimizer_D)


    # Initialize trainer
    trainer = VASAStage1Trainer(config, Gbase, Dbase)
    # Start training
    trainer.train(train_loader=dataloader,start_epoch=start_epoch)




if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)

