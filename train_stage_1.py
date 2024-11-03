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
            project_name=cfg.project_name,
            mixed_precision="fp16"
        )
        
        # Initialize trackers for experiment logging
        self.accelerator.init_trackers(
            project_name=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {
                "name": cfg.experiment_name,
                "dir": cfg.training.log_dir
            }}
        )
        
        # Initialize loss functions on correct device
        self.perceptual_loss = PerceptualLoss(
            self.accelerator.device,
            weights={
                'vgg19': cfg.loss.vgg19_weight,
                'vggface': cfg.loss.vggface_weight,
                'gaze': cfg.loss.gaze_weight,
                'lpips': cfg.loss.lpips_weight
            }
        )
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
        """Single training step"""
        # Extract frame pairs
        source_frame = batch['source_frames']
        driving_frame = batch['driving_frames']
        source_frame_2 = batch['source_frames_star']
        driving_frame_2 = batch['driving_frames_star']
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        # Forward pass main pair
        pred_frame, pred_pyramids = self.Gbase(source_frame, driving_frame)
        
        # Forward pass auxiliary pair for disentanglement
        cross_frame, _ = self.Gbase(source_frame_2, driving_frame)
        
        # Extract motion codes for loss computation
        vs1, es1 = self.Gbase.appearanceEncoder(source_frame)
        Rs1, ts1, zs1 = self.Gbase.motionEncoder(source_frame)
        
        vs2, es2 = self.Gbase.appearanceEncoder(source_frame_2)
        Rs2, ts2, zs2 = self.Gbase.motionEncoder(source_frame_2)
        
        # Compute losses
        # 1. Perceptual pyramid loss
        loss_perceptual = 0
        for scale, pred in pred_pyramids.items():
            target = F.interpolate(
                driving_frame,
                size=pred.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            loss_perceptual += self.perceptual_loss(pred, target)
        
        # 2. GAN loss
        pred_fake = self.Dbase(pred_frame, source_frame)
        loss_gan = -pred_fake.mean()
        
        # 3. Pairwise transfer loss
        loss_pairwise = self.pairwise_transfer_loss(
            self.Gbase,
            source_frame,
            source_frame_2
        )
        
        # 4. Identity preservation loss
        loss_identity = self.identity_loss(
            self.Gbase,
            source_frame,
            cross_frame
        )
        
        # 5. Motion disentanglement loss using contrastive learning
        positive_pairs = [
            (zs1, self.Gbase.motionEncoder(pred_frame)[2]),
            (zs2, self.Gbase.motionEncoder(cross_frame)[2])
        ]
        negative_pairs = [
            (zs1, zs2),
            (self.Gbase.motionEncoder(pred_frame)[2],
             self.Gbase.motionEncoder(cross_frame)[2])
        ]
        loss_motion = cosine_loss(positive_pairs, negative_pairs)
        
        # Total generator loss
        loss_G = (
            self.cfg.loss.perceptual_weight * loss_perceptual +
            self.cfg.loss.gan_weight * loss_gan +
            self.cfg.loss.pairwise_weight * loss_pairwise +
            self.cfg.loss.identity_weight * loss_identity +
            self.cfg.loss.motion_weight * loss_motion
        )
        
        # Accelerator backward pass
        self.accelerator.backward(loss_G)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.Gbase.parameters(), 1.0)
            
        self.optimizer_G.step()
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        pred_real = self.Dbase(driving_frame, source_frame)
        pred_fake = self.Dbase(pred_frame.detach(), source_frame)
        
        loss_D = (
            F.relu(1 - pred_real).mean() +
            F.relu(1 + pred_fake).mean()
        )
        
        # Accelerator backward pass
        self.accelerator.backward(loss_D)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.Dbase.parameters(), 1.0)
            
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_perceptual': loss_perceptual.item(),
            'loss_gan': loss_gan.item(),
            'loss_pairwise': loss_pairwise.item(),
            'loss_identity': loss_identity.item(),
            'loss_motion': loss_motion.item()
        }

    def train(self, train_loader, start_epoch=0):
        """Full training loop"""
        # Prepare for distributed training
        self.Gbase, self.Dbase, self.optimizer_G, self.optimizer_D, self.scheduler_G, self.scheduler_D, train_loader = \
            self.accelerator.prepare(
                self.Gbase, self.Dbase, 
                self.optimizer_G, self.optimizer_D,
                self.scheduler_G, self.scheduler_D,
                train_loader
            )
        
        # Training loop
        for epoch in range(start_epoch, self.cfg.training.base_epochs):
            self.Gbase.train()
            self.Dbase.train()
            
            progress_bar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            metrics = defaultdict(float)
            
            for batch_idx, batch in enumerate(train_loader):
                with self.accelerator.accumulate(self.Gbase, self.Dbase):
                    step_metrics = self.train_step(batch)
                    
                    # Update metrics
                    for k, v in step_metrics.items():
                        metrics[k] += v
                
                # Log samples periodically
                if batch_idx % self.cfg.training.sample_interval == 0:
                    self.log_samples(batch)
                
                progress_bar.update(1)
            
            # Average metrics
            metrics = {k: v / len(train_loader) for k, v in metrics.items()}
            metrics['epoch'] = epoch
            metrics['learning_rate'] = self.scheduler_G.get_last_lr()[0]
            
            # Log metrics
            self.accelerator.log(metrics)
            
            # Update schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", metrics)
            
            progress_bar.close()

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


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)