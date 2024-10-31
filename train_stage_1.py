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




def train_base(cfg, Gbase, Dbase, dataloader, start_epoch=0):
    """
    Main training function for the base stage of VASA
    
    Args:
        cfg: Configuration object containing training parameters
        Gbase: Generator model
        Dbase: Discriminator model
        dataloader: DataLoader for training data
        start_epoch: Epoch to start training from
    """
    # Initialize accelerator and wandb
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps
    )
    
    accelerator.init_trackers(
        project_name=cfg.training.project_name,
        config={
            "learning_rate": cfg.training.lr,
            "epochs": cfg.training.base_epochs,
            "batch_size": cfg.training.batch_size,
            "architecture": "VASA-Base",
            **cfg.training
        }
    )

    # Initialize losses and compute patch size for discriminator
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()

    # Initialize optimizers and schedulers
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    # Prepare models, optimizers, etc. with accelerator
    Gbase, Dbase, optimizer_G, optimizer_D, scheduler_G, scheduler_D, dataloader = accelerator.prepare(
        Gbase, Dbase, optimizer_G, optimizer_D, scheduler_G, scheduler_D, dataloader
    )

    # Initialize loss functions
    perceptual_loss_fn = PerceptualLoss(accelerator.device, 
                                       weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0, 'lpips': 10.0})
    pairwise_transfer_loss = PairwiseTransferLoss()
    identity_similarity_loss = PerceptualLoss(accelerator.device, 
                                            weights={'vgg19': 0.0, 'vggface': 1.0, 'gaze': 0.0, 'lpips': 0.0})

    global_step = start_epoch * len(dataloader)

    # Training loop
    for epoch in range(start_epoch, cfg.training.base_epochs):
        Gbase.train()
        Dbase.train()
        
        metrics = {
            "train/loss_G": 0.0,
            "train/loss_D": 0.0,
            "train/loss_perceptual": 0.0,
            "train/loss_adversarial": 0.0,
            "train/loss_feature_matching": 0.0,
            "train/loss_cosine": 0.0,
            "train/loss_pairwise": 0.0,
            "train/loss_identity": 0.0,
        }

        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}")


        for batch_idx, batch in enumerate(dataloader):
            # Extract frames from batch
            source_frames = batch['source_frames']
            driving_frames = batch['driving_frames']
            source_frames2 = batch['source_frames_star']
            driving_frames2 = batch['driving_frames_star']

            num_frames = len(driving_frames)
            len_source_frames = len(source_frames)
            len_driving_frames = len(driving_frames)
            len_source_frames2 = len(source_frames2)
            len_driving_frames2 = len(driving_frames2)

            for idx in range(num_frames):
                # Get current frames
                source_frame = source_frames[idx % len_source_frames]
                driving_frame = driving_frames[idx % len_driving_frames]
                source_frame_star = source_frames2[idx % len_source_frames2]
                driving_frame_star = driving_frames2[idx % len_driving_frames2]

                # Generator forward pass
                with accelerator.accumulate(Gbase):
                    # Generate frames and pyramids
                    pred_frame, pred_pyramids = Gbase(source_frame, driving_frame)
                    cross_reenacted_image, _ = Gbase(source_frame_star, driving_frame)

                    # Calculate perceptual loss across pyramid levels
                    loss_G_per = 0
                    for scale, pred_scaled in pred_pyramids.items():
                        target_scaled = F.interpolate(driving_frame, size=pred_scaled.shape[2:], 
                                                    mode='bilinear', align_corners=False)
                        loss_G_per += perceptual_loss_fn(pred_scaled, target_scaled)

                    # Calculate adversarial ground truths
                    valid = Variable(torch.ones((driving_frame.size(0), *patch)), 
                                   requires_grad=False).to(accelerator.device)
                    fake = Variable(torch.ones((driving_frame.size(0), *patch)) * -1, 
                                  requires_grad=False).to(accelerator.device)

                    # Discriminator predictions
                    real_pred = Dbase(driving_frame, source_frame)
                    fake_pred = Dbase(pred_frame.detach(), source_frame)
                    
                    # Calculate GAN losses
                    loss_real = hinge_loss(real_pred, valid)
                    loss_fake = hinge_loss(fake_pred, fake)
                    loss_G_adv = 0.5 * (loss_real + loss_fake)

                    # Feature matching loss
                    loss_fm = feature_matching_loss(pred_frame, driving_frame)

                    # Calculate disentanglement losses
                    next_idx = (idx + 20) % len_source_frames
                    I1 = source_frame
                    I2 = source_frames[next_idx].to(accelerator.device)
                    I3 = source_frame_star
                    I4 = source_frames2[next_idx % len_source_frames2].to(accelerator.device)
                    
                    loss_pairwise = pairwise_transfer_loss(Gbase, I1, I2)
                    loss_identity = identity_similarity_loss(I3, I4)

                    # Get motion descriptors for cycle consistency
                    _, _, z_pred = Gbase.motionEncoder(pred_frame)
                    _, _, zd = Gbase.motionEncoder(driving_frame)
                    _, _, z_star_pred = Gbase.motionEncoder(cross_reenacted_image)
                    _, _, zd_star = Gbase.motionEncoder(driving_frame_star)

                    # Calculate cycle consistency loss
                    P = [(z_pred, zd), (z_star_pred, zd)]
                    N = [(z_pred, zd_star), (z_star_pred, zd_star)]
                    loss_G_cos = cosine_loss(P, N)

                    # Total generator loss
                    total_loss_G = (cfg.training.w_per * loss_G_per + 
                                  cfg.training.w_adv * loss_G_adv + 
                                  cfg.training.w_fm * loss_fm + 
                                  cfg.training.w_cos * loss_G_cos + 
                                  cfg.training.w_pairwise * loss_pairwise + 
                                  cfg.training.w_identity * loss_identity)

                    # Backward pass
                    accelerator.backward(total_loss_G)
                    optimizer_G.step()
                    optimizer_G.zero_grad()

                # Discriminator forward pass
                with accelerator.accumulate(Dbase):
                    real_pred = Dbase(driving_frame, source_frame)
                    fake_pred = Dbase(pred_frame.detach(), source_frame)
                    loss_D = discriminator_loss(real_pred, fake_pred, loss_type='lsgan')
                    
                    accelerator.backward(loss_D)
                    optimizer_D.step()
                    optimizer_D.zero_grad()

                # Update metrics
                metrics["train/loss_G"] += total_loss_G.item()
                metrics["train/loss_D"] += loss_D.item()
                metrics["train/loss_perceptual"] += loss_G_per.item()
                metrics["train/loss_adversarial"] += loss_G_adv.item()
                metrics["train/loss_feature_matching"] += loss_fm.item()
                metrics["train/loss_cosine"] += loss_G_cos.item()
                metrics["train/loss_pairwise"] += loss_pairwise.item()
                metrics["train/loss_identity"] += loss_identity.item()

                # Save sample images periodically
                if batch_idx % cfg.training.sample_interval == 0:
                    img_samples = {
                        "source": source_frame,
                        "driving": driving_frame,
                        "predicted": pred_frame,
                        "cross_reenacted": cross_reenacted_image
                    }
                    accelerator.log({"samples": img_samples}, step=epoch)


                # Sample and log images
                if global_step % cfg.training.sample_interval == 0:
                    sample_data = (pred_frame, source_frame, driving_frame)
                    sample_recon(Gbase, sample_data, accelerator,
                               f"samples/sample_{global_step}.png",
                               num_samples=min(4, pred_frame.size(0)))

                # Log gradients
                if global_step % cfg.training.log_grad_interval == 0:
                    log_grad_flow(Gbase.named_parameters(), global_step)
                    log_grad_flow(Dbase.named_parameters(), global_step)

                global_step += 1

            progress_bar.update(1)

        # Average and log metrics
        num_batches = len(dataloader)
        for key in metrics:
            metrics[key] /= num_batches
        
        metrics["learning_rate"] = scheduler_G.get_last_lr()[0]
        accelerator.log(metrics, step=epoch)

        # Step schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            accelerator.save_state(f"checkpoint_epoch{epoch+1}")

        # Print progress
        if accelerator.is_main_process:
            accelerator.print(
                f"Epoch [{epoch+1}/{cfg.training.base_epochs}] "
                f"Loss_G: {metrics['train/loss_G']:.4f} "
                f"Loss_D: {metrics['train/loss_D']:.4f}"
            )

    accelerator.end_training()



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


    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator().to(device)
    
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)

    # Load checkpoint if available
    checkpoint_path = cfg.training.checkpoint_path
    start_epoch = load_checkpoint(checkpoint_path, Gbase, Dbase, optimizer_G, optimizer_D)


    train_base(cfg, Gbase, Dbase, dataloader, start_epoch)
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(Dbase.state_dict(), 'Dbase.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)