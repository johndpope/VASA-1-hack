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

from torch.autograd import Variable

from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from lpips import LPIPS
from VideoDataset import VideoDataset

import wandb
from accelerate import Accelerator
from accelerate.utils import LoggerType
import accelerate
from tqdm import tqdm
from utils import get_vasa_exp_name

from memory_profiler import profile
from collections import defaultdict
from helper import log_grad_flow,consistent_sub_sample,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
from rich.console import Console
from rich.traceback import install
console = Console(width=3000)
# Install Rich traceback handling
install()




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

class RunningAverage:
    """Efficient running average computation"""
    def __init__(self):
        self.avg = 0
        self.count = 0

    def update(self, value):
        self.avg = (self.avg * self.count + value) / (self.count + 1)
        self.count += 1



def custom_collate_fn(batch):
    """
    Custom collate function for batching video frames of different sizes.
    
    Args:
        batch: List of dictionaries containing video data
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Initialize empty lists for each key in the batch
    batch_dict = {
        "video_id": [],
        "source_frames": [],
        "driving_frames": [],
        "video_id_star": [],
        "source_frames_star": [],
        "driving_frames_star": []
    }
    
    # Collect all frame tensors to determine max dimensions
    all_frames = []
    for item in batch:
        all_frames.extend(item['source_frames'])
        all_frames.extend(item['driving_frames'])
        all_frames.extend(item['source_frames_star'])
        all_frames.extend(item['driving_frames_star'])
    
    # Find maximum dimensions
    max_height = max(frame.shape[1] for frame in all_frames)
    max_width = max(frame.shape[2] for frame in all_frames)
    
    def pad_and_stack_frames(frames):
        """Helper function to pad and stack frames to maximum dimensions"""
        padded_frames = []
        for frame in frames:
            # Calculate padding sizes
            pad_h = max_height - frame.shape[1]
            pad_w = max_width - frame.shape[2]
            
            # Pad frame to match maximum dimensions
            padded_frame = torch.nn.functional.pad(
                frame,
                (0, pad_w, 0, pad_h),  # padding left, right, top, bottom
                mode='constant',
                value=0
            )
            padded_frames.append(padded_frame)
        
        return torch.stack(padded_frames) if padded_frames else torch.tensor([])
    
    # Process each item in the batch
    for item in batch:
        batch_dict['video_id'].append(item['video_id'])
        batch_dict['video_id_star'].append(item['video_id_star'])
        
        # Pad and stack frames
        batch_dict['source_frames'].append(pad_and_stack_frames(item['source_frames']))
        batch_dict['driving_frames'].append(pad_and_stack_frames(item['driving_frames']))
        batch_dict['source_frames_star'].append(pad_and_stack_frames(item['source_frames_star']))
        batch_dict['driving_frames_star'].append(pad_and_stack_frames(item['driving_frames_star']))
    
    # Stack all frame tensors along batch dimension
    batch_dict['source_frames'] = torch.stack(batch_dict['source_frames'])
    batch_dict['driving_frames'] = torch.stack(batch_dict['driving_frames'])
    batch_dict['source_frames_star'] = torch.stack(batch_dict['source_frames_star'])
    batch_dict['driving_frames_star'] = torch.stack(batch_dict['driving_frames_star'])
    
    return batch_dict

class VASAStage1Trainer:
    def __init__(self, cfg, Gbase, Dbase, dataloader):
        self.cfg = cfg
        
        # Initialize accelerator first
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            log_with="wandb",
            mixed_precision=None
        )
        
        experiment_name = "megaportraits"
        print(f"Generated experiment name: {experiment_name}")

        # Initialize trackers
        self.accelerator.init_trackers(
            project_name=cfg.training.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {
                "name": experiment_name,
                "dir": cfg.training.log_dir
            }}
        )
        
        # Initialize models without moving to device
        self.Gbase = Gbase
        self.Dbase = Dbase
        
        # Initialize loss functions
        self.perceptual_loss = LPIPS(net='alex')
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
        
        # Prepare models, optimizers, and dataloader with accelerator
        (
            self.Gbase,
            self.Dbase,
            self.optimizer_G,
            self.optimizer_D,
            self.perceptual_loss,
            dataloader,
        ) = self.accelerator.prepare(
            self.Gbase,
            self.Dbase,
            self.optimizer_G,
            self.optimizer_D,
            self.perceptual_loss,
            dataloader
        )
        
        self.dataloader = dataloader  # Save prepared dataloader


 



    def _cleanup_memory(self):
        """Clean up memory by explicitly clearing cuda cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    # @profile
    def train_step(self, source_frame, driving_frame, source_frame_star, driving_frame_star):
        """Training step with accelerator-prepared models"""
        with self.accelerator.accumulate(self.Gbase):
            # Generator forward pass
            pred_frame = self.Gbase(source_frame, driving_frame)
            
            # Calculate losses
            loss_perceptual = self.perceptual_loss(pred_frame, source_frame)
            
            # Discriminator losses
            real_pred = self.Dbase(driving_frame, source_frame)
            with torch.no_grad():  # Reduce memory usage during backprop
                fake_pred = self.Dbase(pred_frame.detach(), source_frame)
            loss_D = discriminator_loss(real_pred, fake_pred, loss_type='lsgan')
            
            # Generator adversarial loss
            fake_pred = self.Dbase(pred_frame, source_frame)
            loss_G_adv = -torch.mean(fake_pred)
            
            # Feature matching loss
            loss_fm = F.mse_loss(pred_frame, driving_frame)
            
            # Cross-reenactment and cycle consistency
            cross_reenacted_image = self.Gbase(source_frame_star, driving_frame)
            
            # Calculate motion encodings with reduced memory usage
            with torch.no_grad():
                z_pred = self.Gbase.motionEncoder(pred_frame)[-1]
                zd = self.Gbase.motionEncoder(driving_frame)[-1]
                z_star_pred = self.Gbase.motionEncoder(cross_reenacted_image)[-1]
                zd_star = self.Gbase.motionEncoder(driving_frame_star)[-1]
            
            # Calculate cycle consistency loss
            P = [(z_pred, zd), (z_star_pred, zd)]
            N = [(z_pred, zd_star), (z_star_pred, zd_star)]
            loss_cycle = cosine_loss(P, N)
            
            # Total generator loss
            loss_G = (self.cfg.training.w_per * loss_perceptual +
                    self.cfg.training.w_adv * loss_G_adv +
                    self.cfg.training.w_fm * loss_fm +
                    self.cfg.training.w_cos * loss_cycle)
            
            return {
                'loss_G': loss_G,
                'loss_D': loss_D,
                'loss_perceptual': loss_perceptual,
                'loss_adversarial': loss_G_adv,
                'loss_feature_matching': loss_fm,
                'loss_cycle': loss_cycle,
                'pred_frame': pred_frame
            }


    def train(self, train_loader, start_epoch=0):
            """Training loop using accelerator-prepared components"""
            for epoch in range(start_epoch, self.cfg.training.base_epochs):
                print(f"Epoch: {epoch}")
                
                running_metrics = defaultdict(lambda: RunningAverage())
                progress_bar = tqdm(self.dataloader, desc=f'Epoch {epoch}')  # Use prepared dataloader
                
                for batch_idx, batch in enumerate(progress_bar):
                    # No need to move batch to device - accelerator handles this
                    for idx in range(len(batch['driving_frames'])):
                        source_frame = batch['source_frames'][idx % len(batch['source_frames'])]
                        driving_frame = batch['driving_frames'][idx % len(batch['driving_frames'])]
                        source_frame_star = batch['source_frames_star'][idx % len(batch['source_frames_star'])]
                        driving_frame_star = batch['driving_frames_star'][idx % len(batch['driving_frames_star'])]
                        
                        # Training step
                        metrics = self.train_step(
                            source_frame, driving_frame,
                            source_frame_star, driving_frame_star
                        )
                        
                        # Use accelerator for backwards pass
                        self.accelerator.backward(metrics['loss_G'])
                        self.optimizer_G.step()
                        self.optimizer_G.zero_grad()
                        
                        self.accelerator.backward(metrics['loss_D'])
                        self.optimizer_D.step()
                        self.optimizer_D.zero_grad()

                        
                        # Log images periodically
                        if idx == 0 and batch_idx % self.cfg.training.log_interval == 0:
                            wandb.log({
                                "source_frame": wandb.Image(source_frame),
                                "driving_frame": wandb.Image(driving_frame),
                                "predicted_frame": wandb.Image(metrics['pred_frame'])
                            })
                        
                        # Clean up memory
                        del metrics
                        torch.cuda.empty_cache()
                    
                    # Update progress bar
                    progress_bar.set_postfix({k: f"{v.avg:.4f}" for k, v in running_metrics.items()})
                
                # Log epoch metrics
                wandb.log({
                    'epoch': epoch,
                    **{k: v.avg for k, v in running_metrics.items()},
                    'learning_rate_G': self.scheduler_G.get_last_lr()[0],
                    'learning_rate_D': self.scheduler_D.get_last_lr()[0]
                })
                
                # Step schedulers
                self.scheduler_G.step()
                self.scheduler_D.step()
                
                # Save checkpoint
                if (epoch + 1) % self.cfg.training.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", running_metrics)
            
            wandb.finish()


    def save_checkpoint(self, filename, metrics):
        checkpoint = {
            'Gbase_state_dict': self.Gbase.state_dict(),
            'Dbase_state_dict': self.Dbase.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'metrics': metrics,
            'config': self.cfg
        }
        
        torch.save(checkpoint, filename)
        wandb.save(filename)



def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = VideoDataset(
        video_dir=cfg.training.video_dir,
        width=256,
        height=256,
        n_pairs=10000,
        cache_dir=cfg.training.cache_video_dir,
        transform=transform,
        remove_background=True,
        use_greenscreen=False,
        apply_warping=True,
        max_frames=100,
        duplicate_short=True,
        warp_strength=0.01
    )



    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=1,  # Reduce number of workers to manage memory better
        collate_fn=custom_collate_fn,
        pin_memory=False,  # Disable pin_memory to reduce memory usage
        persistent_workers=True,  # Keep workers alive between iterations
        prefetch_factor=2,  # Reduce prefetch factor to manage memory
    )


    
    Gbase = model.Gbase()
    Dbase = model.Discriminator()


    # Initialize trainer
    trainer = VASAStage1Trainer(config, Gbase, Dbase,dataloader)
    # Start training
    trainer.train(train_loader=dataloader,start_epoch=0)




if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)

