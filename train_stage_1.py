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
from helper import handle_training_error,resize_for_wandb,log_grad_flow,consistent_sub_sample,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
from rich.console import Console
from rich.traceback import install
from rich.progress import track

console = Console(width=3000)
# Install Rich traceback handling
# Install Rich traceback handler with custom settings
install(
    console=console,
    # show_locals=True,     # Show local variables in tracebacks
    width=None,           # Full width
    word_wrap=False,      # Disable word wrapping
    indent_guides=True,   # Show indent guides
    suppress=[           # Suppress specific modules from traceback
        "torch",
        "numpy",
        "wandb"
    ],
    max_frames=10        # Show last 10 frames
)



from losses.FeatureMatchingLoss import FeatureMatchingLoss


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
    
    return loss.mean()




def custom_collate_fn(batch):
    """
    Custom collate function for batching paired frames with consecutive frames
    
    Args:
        batch: List of dictionaries containing paired frames
        
    Returns:
        Dictionary with batched tensors
    """
    # Initialize batch dictionary with all required keys
    batch_dict = {
        "source_frame": [],
        "next_source_frame": [],
        "driving_frame": [],
        "next_driving_frame": [],
        "source_vid": [],
        "driving_vid": []
    }
    
    for item in batch:
        # Add frames
        batch_dict["source_frame"].append(item["source_frame"])
        batch_dict["next_source_frame"].append(item["next_source_frame"])
        batch_dict["driving_frame"].append(item["driving_frame"])
        batch_dict["next_driving_frame"].append(item["next_driving_frame"])
        
        # Add video IDs
        batch_dict["source_vid"].append(item["source_vid"])
        batch_dict["driving_vid"].append(item["driving_vid"])
    
    # Stack tensors
    batch_dict["source_frame"] = torch.stack(batch_dict["source_frame"])
    batch_dict["next_source_frame"] = torch.stack(batch_dict["next_source_frame"])
    batch_dict["driving_frame"] = torch.stack(batch_dict["driving_frame"])
    batch_dict["next_driving_frame"] = torch.stack(batch_dict["next_driving_frame"])
    
    return batch_dict

class RunningMetric:
    """Class to track running averages of metrics"""
    def __init__(self):
        self.avg = 0
        self.count = 0

    def update(self, value):
        self.avg = (self.avg * self.count + value) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class MetricsTracker:
    """Class to manage multiple running metrics"""
    def __init__(self):
        self.metrics = {}

    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = RunningMetric()
        self.metrics[name].update(value)

    def get_average(self, name):
        return self.metrics[name].get_value() if name in self.metrics else 0

    def get_all_averages(self):
        return {name: metric.get_value() for name, metric in self.metrics.items()}


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

class VASAStage1Trainer:
    def __init__(self, cfg, Gbase, Dbase, dataloader):
        self.cfg = cfg
        

    
        # Initialize accelerator first
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
          
            mixed_precision=None
        )
        
        experiment_name = "megaportraits"
        print(f"Generated experiment name: {experiment_name}")
        wandb.init(project="vasa", name=experiment_name,settings=wandb.Settings(save_code=False))  # Disable code saving


        # Initialize trackers
        self.accelerator.init_trackers(
            project_name=cfg.training.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
          
        )
        
        # Initialize models without moving to device
        self.Gbase = Gbase
        self.Dbase = Dbase
        
        # Initialize loss functions
        self.perceptual_loss = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})
        self.pairwise_transfer_loss = PairwiseTransferLoss()
        self.identity_loss = IdentitySimilarityLoss()
        self.hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
        self.feature_matching_loss = FeatureMatchingLoss()
      
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
        self.dataset = dataloader.dataset


 



    def _cleanup_memory(self):
        """Enhanced memory cleanup"""
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats()
            
            # Optional: force garbage collection
            import gc
            gc.collect()



    # @profile    
    def train(self, train_loader, start_epoch=0):
        """Training loop with enhanced memory management"""
        torch.cuda.empty_cache()  # Initial cache clear
        
        for epoch in range(start_epoch, self.cfg.training.base_epochs):
            metrics_tracker = MetricsTracker()
            
            # Reset memory at start of epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            for batch_idx in tqdm(range(len(self.dataloader)), desc=f'Epoch {epoch}'):
                try:
                    # Get batches with memory cleanup
                    with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision temporarily
                        current_batch = next(iter(self.dataloader))
                        next_batch = next(iter(self.dataloader))
                    
                    # Extract frames and immediately move to device
                    frames = {
                        'source_frame': current_batch['source_frame'],  # Changed from 'source'
                        'next_source': current_batch['next_source_frame'],
                        'driving_frame': current_batch['driving_frame'], # Changed from 'driving'
                        'next_driving': current_batch['next_driving_frame'],
                        'source_frame_star': next_batch['source_frame'], # Changed from 'source_star'
                        'driving_frame_star': next_batch['driving_frame'] # Changed from 'driving_star'
                    }
                    
                    # Clear unused batch data
                    del current_batch
                    del next_batch
                    torch.cuda.empty_cache()
                    
                    # Training step with memory optimization
                    with self.accelerator.accumulate(self.Gbase):
                        # Generator update
                        self.Gbase.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                        metrics = self.train_step(**frames)
                    
                    pred_frame = metrics['pred_frame']
        
                    
                    # Log with memory optimization
                    if batch_idx % self.cfg.training.log_interval == 0:
                        with torch.no_grad():
                            source_frame_cpu = frames['source_frame'][0].cpu()
                            pred_frame_cpu = pred_frame[0].cpu()
                            
                            wandb.log({
                                "source_frame": wandb.Image(resize_for_wandb(source_frame_cpu)),
                                "pred_frame": wandb.Image(resize_for_wandb(pred_frame_cpu))
                            })
                            
                            del source_frame_cpu
                            del pred_frame_cpu
                    # Explicit cleanup
                    for v in frames.values():
                        del v
                    del frames
                    del metrics
                    del pred_frame
                    
                    # Force garbage collection and cache clearing
                    self._cleanup_memory()
                    
                except Exception as e:
                    handle_training_error(
                        e, 
                        batch_idx=batch_idx,
                        extra_info=f"Error occurred during epoch {epoch}"
                    )
                    
                    # Memory cleanup after error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # You might want to break the epoch if error is severe
                    if isinstance(e, (RuntimeError, AssertionError)):
                        console.print("[bold red]Critical error detected - Breaking epoch[/bold red]")
                        break
                        
                    continue
                
            # End of epoch cleanup
            self.scheduler_G.step()
            self.scheduler_D.step()
            self.dataset.reset_pairs()
            
            # Log epoch metrics
            wandb.log({
                'epoch': epoch,
                **metrics_tracker.get_all_averages(),
                'learning_rate_G': self.scheduler_G.get_last_lr()[0],
                'learning_rate_D': self.scheduler_D.get_last_lr()[0],
                'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9
            })
            
            if (epoch + 1) % self.cfg.training.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", metrics_tracker.get_all_averages())
            
            # Epoch-end cleanup
            self._cleanup_memory()
        
        wandb.finish()

    # @profile
    def train_step(self, source_frame, next_source, driving_frame, next_driving, source_frame_star, driving_frame_star):
        """Training step with accelerator-prepared models and proper graph retention"""
        
        # Get source and driving frames
        source = source_frame
        driving = driving_frame
        
        # Cross-identity sample for cycle consistency
        source_star = source_frame_star
        
        # Generator forward pass
        pred_frame = self.Gbase(source, driving)
        pred_star = self.Gbase(source_star, driving)
        
        # Extract motion descriptors
        Rs, ts, zs = self.Gbase.motionEncoder(source)
        Rd, td, zd = self.Gbase.motionEncoder(driving)
        Rs_star, ts_star, zs_star = self.Gbase.motionEncoder(source_star)
        _, _, zd_star = self.Gbase.motionEncoder(driving_frame_star)
        # Cycle consistency - positive pairs should align with driving
        P = [(zs, zd), (zs_star, zd)]
        N = [(zs, zd_star), (zs_star, zd_star)]
        loss_cycle = cosine_loss(P, N)
        
        #feature matching
        loss_fm = self.feature_matching_loss(pred_frame, driving_frame)



        # Perceptual & adversarial losses
        loss_perceptual = self.perceptual_loss(pred_frame, driving)
        pred_fake = self.Dbase(pred_frame, source)
        loss_gan = -pred_fake.mean()
        
        # Generator total loss
        loss_G = (self.cfg.training.w_per * loss_perceptual + 
                 self.cfg.training.w_adv * loss_gan +
                 self.cfg.training.w_fm * loss_fm +
                 self.cfg.training.w_cos * loss_cycle)

        # Discriminator loss  
        pred_real = self.Dbase(driving, source)
        pred_fake = self.Dbase(pred_frame.detach(), source)
        loss_D = discriminator_loss(pred_real, pred_fake)
        
        # Optimization
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_perceptual': loss_perceptual.item(),
            'loss_cycle': loss_cycle.item(),
            'pred_frame': pred_frame
        }


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
        # wandb.save(filename)



def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = VideoDataset(
        video_dir=cfg.training.video_dir,
        width=512,
        height=512,
        initial_pairs=1, # just 1 video pair
        cache_dir=cfg.training.cache_video_dir,
        transform=transform,
        remove_background=True,
        use_greenscreen=False,
        apply_warping=False,
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

