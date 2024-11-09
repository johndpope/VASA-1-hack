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
import torchvision.utils as vutils
import wandb
from accelerate import Accelerator
from accelerate.utils import LoggerType
from tqdm import tqdm
from EmoDataset import EMODataset
from model import PerceptualLoss
import os
import random
import torchvision.transforms.functional as TF
from memory_profiler import profile

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# align to cyclegan
def discriminator_loss(real_preds, fake_preds, loss_type='lsgan'):
    """
    Compute discriminator loss for multi-scale predictions.
    
    Args:
        real_preds: List of discriminator predictions on real images at each scale
        fake_preds: List of discriminator predictions on fake images at each scale
        loss_type: Loss type ('lsgan' or 'vanilla')
    """
    # Initialize total loss
    loss = 0
    num_scales = len(real_preds)
    
    # Process each scale
    for real_pred, fake_pred in zip(real_preds, fake_preds):
        if loss_type == 'lsgan':
            # LSGAN loss for current scale
            real_loss = torch.mean((real_pred - 1)**2)
            fake_loss = torch.mean(fake_pred**2)
        elif loss_type == 'vanilla':
            # BCE loss for current scale
            real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        else:
            raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
        
        # Add loss for current scale
        loss += (real_loss + fake_loss) * 0.5
    
    # Average across scales and ensure gradients
    return (loss / num_scales).requires_grad_()


def contrastive_loss(z_source, z_driving, z_source_star, z_driving_star, accelerator, temperature=0.1):
    """
    Compute contrastive loss between motion descriptors to prevent identity leakage.
    
    Args:
        z_source: Motion descriptor from source image
        z_driving: Motion descriptor from driving image 
        z_source_star: Motion descriptor from different source image
        z_driving_star: Motion descriptor from different driving image
        temperature: Temperature parameter for scaling
    """
    # Form positive pairs
    positive_pairs = [
        (z_source, z_driving),      # Same video motion pair
        (z_source_star, z_driving)  # Cross-video motion pair with same driving
    ]
    
    # Form negative pairs 
    negative_pairs = [
        (z_source, z_driving_star),      # Different motion
        (z_source_star, z_driving_star)  # Different motion
    ]

    def compute_similarity(z1, z2):
        return torch.sum(z1 * z2, dim=-1) / (torch.norm(z1, dim=-1) * torch.norm(z2, dim=-1))

    # Calculate positive similarities
    pos_sims = torch.stack([compute_similarity(p[0], p[1]) for p in positive_pairs])
    
    # Calculate negative similarities
    neg_sims = torch.stack([compute_similarity(n[0], n[1]) for n in negative_pairs])
    
    # Scale similarities by temperature
    pos_sims = pos_sims / temperature
    neg_sims = neg_sims / temperature
    
    # Compute log sum exp
    neg_term = torch.logsumexp(neg_sims, dim=0)
    
    # Final contrastive loss
    loss = -torch.mean(pos_sims - neg_term)
    
    # Synchronize across processes if using distributed training
    if accelerator.num_processes > 1:
        loss = accelerator.gather(loss).mean()
        
    return loss.requires_grad_()


def feature_matching_loss(real_features, fake_features):
    """
    Calculate feature matching loss between real and fake feature maps
    from different layers and scales of the discriminator.
    
    Args:
        real_features: List of lists of features from discriminator for real images
                      [scale][layer] indexing
        fake_features: List of lists of features from discriminator for fake images
                      [scale][layer] indexing
    """
    fm_loss = 0.0
    num_d = len(real_features)  # Number of discriminators (scales)
    
    for i in range(num_d):  # For each discriminator
        num_layers = len(real_features[i])  # Number of layers in current discriminator
        for j in range(num_layers):  # For each layer
            fm_loss += F.l1_loss(fake_features[i][j], real_features[i][j].detach())
            
    return fm_loss / num_d  # Average over number of discriminators

def adversarial_loss(discriminator_preds, is_real=False):
    """
    Compute generator/discriminator adversarial loss for multi-scale predictions.
    
    Args:
        discriminator_preds: List of discriminator predictions at each scale
        is_real: If True, compute loss for real samples, else for fake samples
    
    Returns:
        Adversarial loss averaged over scales
    """
    loss = 0
    num_scales = len(discriminator_preds)
    
    for pred in discriminator_preds:
        if is_real:
            # For real samples: -D(real)
            loss += -torch.mean(pred)
        else:
            # For fake samples: D(fake)
            loss += torch.mean(pred)
            
    return loss / num_scales


def train_base(cfg, Gbase, Dbase, dataloader):
    # Initialize accelerator
    accelerator = Accelerator(
        log_with=["wandb"]
    )

    # Initialize wandb
    accelerator.init_trackers(
        project_name=cfg.wandb.project_name,
        config={
            "learning_rate": cfg.training.lr,
            "epochs": cfg.training.base_epochs,
            "batch_size": cfg.training.batch_size,
            "architecture": "Gbase-Dbase"
        }
    )

    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    perceptual_loss_fn = PerceptualLoss(accelerator.device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0, 'lpips': 10.0})
    
    # Prepare everything with accelerator
    Gbase, Dbase, optimizer_G, optimizer_D, dataloader = accelerator.prepare(
        Gbase, Dbase, optimizer_G, optimizer_D, dataloader
    )

    # Training loop
    for epoch in range(cfg.training.base_epochs):
        total_loss_G = 0
        total_loss_D = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.base_epochs}")
        
        for batch in progress_bar:
            source_frames = batch['source_frames']
            driving_frames = batch['driving_frames']
            source_frames2 = batch['source_frames_star']
            driving_frames2 = batch['driving_frames_star']

            num_frames = len(driving_frames)
            len_source_frames = len(source_frames)
            len_driving_frames = len(driving_frames)
            len_source_frames2 = len(source_frames2)
            len_driving_frames2 = len(driving_frames2)

            batch_loss_G = 0
            batch_loss_D = 0

            for idx in range(num_frames):
                source_frame = source_frames[idx % len_source_frames]
                driving_frame = driving_frames[idx % len_driving_frames]
                source_frame_star = source_frames2[idx % len_source_frames2]
                driving_frame_star = driving_frames2[idx % len_driving_frames2]

                # Generator forward pass
                with accelerator.accumulate(Gbase):
                    # Get outputs from Gbase including mixing predictions
                    outputs = Gbase(source_frame, driving_frame)
                    # print("outputs:", outputs)
                    pred_frame = outputs['pred_target_img']
                    mix_output = outputs['pred_mixing_img']
                    mix_mask = outputs['pred_mixing_mask']
                    mix_masked = outputs['pred_mixing_masked_img']


                    # Generate cross-reenacted image
                    cross_outputs = Gbase(source_frame_star, driving_frame)
                    cross_reenacted_image = cross_outputs['pred_target_img']

                    # Get motion encodings
                    Rs, ts, zs = outputs['source_rotation'], outputs['source_translation'], outputs['source_expression']
                    Rd, td, zd = outputs['target_rotation'], outputs['target_translation'], outputs['target_expression']
                    Rs_star, ts_star, zs_star = cross_outputs['source_rotation'], cross_outputs['source_translation'], cross_outputs['source_expression']


                  
                    # Get discriminator predictions for all outputs
                    fake_preds = Dbase(pred_frame)
                    real_preds = Dbase(driving_frame)
                    cross_preds = Dbase(cross_reenacted_image)
                    mix_preds = Dbase(mix_output) if mix_output is not None else None
                    
                    fake_features = Dbase.get_features(pred_frame)
                    real_features = Dbase.get_features(driving_frame)
                    cross_features = Dbase.get_features(cross_reenacted_image)


                    # Calculate standard generator losses
                    loss_G_adv = adversarial_loss(fake_preds)
                    loss_G_fm = feature_matching_loss(real_features, fake_features)
                    loss_G_per = perceptual_loss_fn(pred_frame, driving_frame)

                    # Cross-reenactment losses
                    loss_G_cross_adv = adversarial_loss(cross_preds)
                    loss_G_cross_fm = feature_matching_loss(real_features, cross_features)
                    loss_G_cross_per = perceptual_loss_fn(cross_reenacted_image, driving_frame)

                    # Expression consistency between predictions
                    loss_expr_consistency = F.mse_loss(zd, cross_outputs['target_expression'])


                    # Calculate mixing losses if enabled
                     # Calculate mixing losses if enabled
                    loss_G_mix = 0
                    if mix_output is not None:
                        # Standard mixing losses
                        loss_mix_per = perceptual_loss_fn(mix_masked, driving_frame * mix_mask)
                        loss_mix_adv = adversarial_loss(mix_preds)
                        loss_mix_id = torch.mean(torch.abs(mix_masked - source_frame * mix_mask))
                        
                        # Get mixing expressions
                        mix_out_dict = Gbase.get_mixing_expr_contrastive_data(outputs, source_frame, driving_frame)
                        
                        # Expression contrastive loss
                        loss_mix_expr = contrastive_loss(
                            mix_out_dict['mixing_expr'],
                            mix_out_dict['target_expr'],
                            zs_star,
                            zd_star,
                            accelerator
                        )
                        
                        # Cycle consistency if enabled
                        loss_cycle = 0
                        if cfg.training.use_cycle:
                            cycle_out_dict = Gbase.get_cycle_consistency_data(outputs, source_frame, driving_frame)
                            if 'cycle_mix_pred' in cycle_out_dict:
                                loss_cycle = F.l1_loss(cycle_out_dict['cycle_mix_pred'], pred_frame)
                        
                        loss_G_mix = (cfg.training.w_mix_per * loss_mix_per +
                                    cfg.training.w_mix_adv * loss_mix_adv + 
                                    cfg.training.w_mix_id * loss_mix_id +
                                    cfg.training.w_mix_expr * loss_mix_expr +
                                    cfg.training.w_cycle * loss_cycle)

                    # Calculate motion contrastive loss
                    loss_G_cos = contrastive_loss(
                        zs, zd, zs_star, zd_star,
                        accelerator
                    )

                    # Total generator loss
                    total_G_loss = (
                        cfg.training.w_per * loss_G_per +
                        cfg.training.w_adv * loss_G_adv +
                        cfg.training.w_fm * loss_G_fm +
                        cfg.training.w_cos * loss_G_cos +
                        cfg.training.w_cross_per * loss_G_cross_per +
                        cfg.training.w_cross_adv * loss_G_cross_adv +
                        cfg.training.w_cross_fm * loss_G_cross_fm +
                        cfg.training.w_expr_consistency * loss_expr_consistency +
                        loss_G_mix
                    )

                    accelerator.backward(total_G_loss)
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    batch_loss_G += total_G_loss.item()

                # Discriminator forward pass
                with accelerator.accumulate(Dbase):
                    fake_preds = Dbase(pred_frame.detach())
                    cross_preds = Dbase(cross_reenacted_image.detach())
                    real_preds = Dbase(driving_frame)
                    
                    loss_D = discriminator_loss(real_preds, fake_preds) + \
                            discriminator_loss(real_preds, cross_preds)
                    
                    if mix_output is not None:
                        mix_preds = Dbase(mix_output.detach())
                        loss_D += discriminator_loss(real_preds, mix_preds)
                    
                    accelerator.backward(loss_D)
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                    batch_loss_D += loss_D.item()

                # Log samples periodically
                if idx % cfg.training.sample_interval == 0:
                    def ensure_rgb(tensor):
                        """Convert tensor to RGB if needed"""
                        # Check if the tensor has 1 channel (grayscale)
                        if tensor.shape[0] == 1:
                            # Repeat the grayscale channel to create 3 channels
                            tensor = tensor.repeat(3, 1, 1)
                        return tensor

                    # Create image dictionary
                    images_dict = {
                        "source": ensure_rgb(source_frame[0]),
                        "source_star": ensure_rgb(source_frame_star[0]),
                        "driving": ensure_rgb(driving_frame[0]),
                        "predicted": ensure_rgb(pred_frame[0]),
                        "cross_reenacted": ensure_rgb(cross_reenacted_image[0])
                    }
                    
                    # Add mixing-related images if available
                    if mix_output is not None:
                        images_dict.update({
                            "mixing": ensure_rgb(mix_output[0]),
                            "mixing_masked": ensure_rgb(mix_masked[0]),
                            "mixing_mask": ensure_rgb(mix_mask[0])
                        })
                    
                    # Convert images to grid
                    grid = vutils.make_grid(
                        list(images_dict.values()),
                        nrow=2,
                        normalize=True
                    )
                    
                    # Log to wandb
                    accelerator.log({
                        f"samples/epoch_{epoch}_idx_{idx}": 
                        wandb.Image(grid)
                    })

            # Average batch losses
            batch_loss_G /= num_frames
            batch_loss_D /= num_frames
            total_loss_G += batch_loss_G
            total_loss_D += batch_loss_D
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f"{batch_loss_G:.4f}",
                'D_loss': f"{batch_loss_D:.4f}"
            })

        # Calculate epoch averages
        avg_loss_G = total_loss_G / num_batches
        avg_loss_D = total_loss_D / num_batches

        # Log metrics
        accelerator.log({
            "train/generator_loss": avg_loss_G,
            "train/discriminator_loss": avg_loss_D,
            "train/learning_rate": scheduler_G.get_last_lr()[0],
            "epoch": epoch
        })

        scheduler_G.step()
        scheduler_D.step()

        # Save checkpoints
        if (epoch + 1) % cfg.training.save_interval == 0:
            accelerator.save_state(f"checkpoint-epoch-{epoch+1}")

    # End training
    accelerator.end_training()

class RandomGaussianBlur(object):
    def __init__(self, kernel_range=(3, 7), sigma_range=(0.1, 2.0)):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range

    def __call__(self, img):
        kernel_size = random.randrange(self.kernel_range[0], self.kernel_range[1] + 1, 2)
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        return TF.gaussian_blur(img, kernel_size, [sigma, sigma])

class RandomSharpness(object):
    def __init__(self, range=(0.0, 4.0)):
        self.range = range
    
    def __call__(self, img):
        factor = random.uniform(self.range[0], self.range[1])
        return TF.adjust_sharpness(img, factor)

# Custom color jitter that applies transformations with random probability
class RandomColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        transforms_list = []
        
        if random.random() < 0.5 and self.brightness > 0:
            brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
            transforms_list.append(lambda x: TF.adjust_brightness(x, brightness_factor))
        
        if random.random() < 0.5 and self.contrast > 0:
            contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
            transforms_list.append(lambda x: TF.adjust_contrast(x, contrast_factor))
        
        if random.random() < 0.5 and self.saturation > 0:
            saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
            transforms_list.append(lambda x: TF.adjust_saturation(x, saturation_factor))
        
        if random.random() < 0.5 and self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            transforms_list.append(lambda x: TF.adjust_hue(x, hue_factor))
        
        random.shuffle(transforms_list)
        
        for t in transforms_list:
            img = t(img)
        
        return img




def main(cfg: OmegaConf) -> None:
 # Main transform pipeline
    # N.b. the npz saved file will freeze the random state
    # transform_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=(-5, 5)),
    #     transforms.RandomAffine(
    #         degrees=0,
    #         translate=(0.05, 0.05),
    #         scale=(0.95, 1.05),
    #         shear=(-5, 5, -5, 5)
    #     ),
    #     RandomColorJitter(
    #         brightness=0.3,
    #         contrast=0.3,
    #         saturation=0.3,
    #         hue=0.1
    #     ),
    #     # RandomGaussianBlur(kernel_range=(3, 5), sigma_range=(0.1, 1.0)),
    #     RandomSharpness(range=(0.0, 2.0)),
    #     transforms.RandomGrayscale(p=0.02),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EMODataset(
        use_gpu=torch.cuda.is_available(),
        remove_background=False,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform_train,
        max_frames=100,
        apply_warping=False 
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.training.num_workers
    )

    Gbase = model.Gbase()
    Dbase = model.MultiScalePatchDiscriminator()
    
    train_base(cfg, Gbase, Dbase, dataloader)

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)