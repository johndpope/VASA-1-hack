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


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

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


# cosine distance formula
# s · (⟨zi, zj⟩ − m)
def cosine_loss(pos_pairs, neg_pairs, accelerator, s=5.0, m=0.2):
    """
    Compute cosine distance loss with proper device handling for accelerate.
    
    Args:
        pos_pairs (list): List of positive pair tuples (z1, z2)
        neg_pairs (list): List of negative pair tuples (z1, z2)
        accelerator (Accelerator): Accelerator instance for device management
        s (float): Scaling factor for the cosine similarity
        m (float): Margin parameter
    """
    assert isinstance(pos_pairs, list) and isinstance(neg_pairs, list), "pos_pairs and neg_pairs should be lists"
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    assert len(neg_pairs) > 0, "neg_pairs should not be empty"
    assert s > 0, "s should be greater than 0"
    assert 0 <= m <= 1, "m should be between 0 and 1"
    
    # Initialize loss tensor on the correct device
    loss = torch.tensor(0.0, requires_grad=True, device=accelerator.device)

    for pos_pair in pos_pairs:
        assert isinstance(pos_pair, tuple) and len(pos_pair) == 2, "Each pos_pair should be a tuple of length 2"
        
        # Ensure tensors are on the correct device
        z1_pos = accelerator.prepare(pos_pair[0])
        z2_pos = accelerator.prepare(pos_pair[1])
        
        # Calculate positive similarity
        pos_sim = F.cosine_similarity(z1_pos, z2_pos, dim=0)
        pos_dist = s * (pos_sim - m)
        
        # Initialize negative term on the correct device
        neg_term = torch.tensor(0.0, requires_grad=True, device=accelerator.device)
        
        for neg_pair in neg_pairs:
            assert isinstance(neg_pair, tuple) and len(neg_pair) == 2, "Each neg_pair should be a tuple of length 2"
            
            # Ensure tensors are on the correct device
            z1_neg = accelerator.prepare(pos_pair[0])  # Using positive pair's first element
            z2_neg = accelerator.prepare(neg_pair[1])  # Using negative pair's second element
            
            # Calculate negative similarity
            neg_sim = F.cosine_similarity(z1_neg, z2_neg, dim=0)
            neg_term = neg_term + torch.exp(s * (neg_sim - m))
        
        # Verify shapes match across devices
        if accelerator.is_local_main_process:
            assert pos_dist.shape == neg_term.shape, f"Shape mismatch: pos_dist {pos_dist.shape}, neg_term {neg_term.shape}"
        
        # Calculate loss term
        loss = loss + torch.log(torch.exp(pos_dist) / (torch.exp(pos_dist) + neg_term))
    
    # Calculate mean loss and ensure gradient is maintained
    final_loss = torch.mean(-loss / len(pos_pairs))
    
    # Synchronize loss across processes if using distributed training
    if accelerator.num_processes > 1:
        final_loss = accelerator.gather(final_loss).mean()
    
    return final_loss.requires_grad_()


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

    # Move models, optimizers, and dataloader to appropriate device
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    
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
                    pred_frame = Gbase(source_frame, driving_frame)
                    
                    # Calculate losses
                    loss_G_per = perceptual_loss_fn(pred_frame, source_frame)
                    
                    # Adversarial ground truths
                    valid = Variable(torch.ones((driving_frame.size(0), *patch)), requires_grad=False).to(accelerator.device)
                    fake = Variable(torch.ones((driving_frame.size(0), *patch)) * -1, requires_grad=False).to(accelerator.device)

                    real_pred = Dbase(driving_frame, source_frame)
                    fake_pred = Dbase(pred_frame.detach(), source_frame)
                    
                    loss_real = hinge_loss(real_pred, valid)
                    loss_fake = hinge_loss(fake_pred, fake)
                    loss_G_adv = 0.5 * (loss_real + loss_fake)

                    # Feature matching loss
                    loss_fm = feature_matching_loss(pred_frame, driving_frame)

                    # Cross reenactment
                    cross_reenacted_image = Gbase(source_frame_star, driving_frame)
                    
                    # Motion descriptors
                    _, _, z_pred = Gbase.motionEncoder(pred_frame)
                    _, _, zd = Gbase.motionEncoder(driving_frame)
                    _, _, z_star_pred = Gbase.motionEncoder(cross_reenacted_image)
                    _, _, zd_star = Gbase.motionEncoder(driving_frame_star)

                    # Cosine loss
                    P = [(z_pred, zd), (z_star_pred, zd)]
                    N = [(z_pred, zd_star), (z_star_pred, zd_star)]
                    loss_G_cos = cosine_loss(P, N,accelerator)

                    # Total generator loss
                    total_G_loss = (
                        cfg.training.w_per * loss_G_per +
                        cfg.training.w_adv * loss_G_adv +
                        cfg.training.w_fm * loss_fm +
                        cfg.training.w_cos * loss_G_cos
                    )

                    accelerator.backward(total_G_loss)
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    batch_loss_G += total_G_loss.item()

                # Discriminator forward pass
                with accelerator.accumulate(Dbase):
                    real_pred = Dbase(driving_frame, source_frame)
                    fake_pred = Dbase(pred_frame.detach(), source_frame)
                    loss_D = discriminator_loss(real_pred, fake_pred, loss_type='lsgan')
                    
                    accelerator.backward(loss_D)
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                    batch_loss_D += loss_D.item()

                # Log samples periodically
                if idx % cfg.training.sample_interval == 0:
                    images = {
                        "source": source_frame[0],
                        "driving": driving_frame[0],
                        "predicted": pred_frame[0],
                        "cross_reenacted": cross_reenacted_image[0]
                    }
                    
                    # Log images to wandb
                    accelerator.log({
                        f"samples/epoch_{epoch}_idx_{idx}": 
                        wandb.Image(vutils.make_grid(
                            [images[k] for k in images], 
                            nrow=2, 
                            normalize=True
                        ))
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
        remove_background=True,
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
    Dbase = model.Discriminator()
    
    train_base(cfg, Gbase, Dbase, dataloader)

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)