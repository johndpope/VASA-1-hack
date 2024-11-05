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
from EmoDataset import EMODataset

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
from torch.cuda.amp import autocast, GradScaler


console = Console(width=3000)
# Install Rich traceback handling
# Install Rich traceback handler with custom settings
install()



from losses.FeatureMatchingLoss import FeatureMatchingLoss


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



# work from here https://github.com/johndpope/MegaPortrait-hack




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
def cosine_loss(pos_pairs, neg_pairs, s=5.0, m=0.2):
    assert isinstance(pos_pairs, list) and isinstance(neg_pairs, list), "pos_pairs and neg_pairs should be lists"
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    assert len(neg_pairs) > 0, "neg_pairs should not be empty"
    assert s > 0, "s should be greater than 0"
    assert 0 <= m <= 1, "m should be between 0 and 1"
    
    loss = torch.tensor(0.0, requires_grad=True).to(device)

    for pos_pair in pos_pairs:
        assert isinstance(pos_pair, tuple) and len(pos_pair) == 2, "Each pos_pair should be a tuple of length 2"
        pos_sim = F.cosine_similarity(pos_pair[0], pos_pair[1], dim=0)
        pos_dist = s * (pos_sim - m)
        
        neg_term = torch.tensor(0.0, requires_grad=True).to(device)
        for neg_pair in neg_pairs:
            assert isinstance(neg_pair, tuple) and len(neg_pair) == 2, "Each neg_pair should be a tuple of length 2"
            neg_sim = F.cosine_similarity(pos_pair[0], neg_pair[1], dim=0)
            neg_term = neg_term + torch.exp(s * (neg_sim - m))
        
        assert pos_dist.shape == neg_term.shape, f"Shape mismatch: pos_dist {pos_dist.shape}, neg_term {neg_term.shape}"
        loss = loss + torch.log(torch.exp(pos_dist) / (torch.exp(pos_dist) + neg_term))
        
    assert len(pos_pairs) > 0, "pos_pairs should not be empty"
    return torch.mean(-loss / len(pos_pairs)).requires_grad_()


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def train_base(cfg, Gbase, Dbase, dataloader):
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})

    scaler = GradScaler()

    # Initialize generator profiler
    for epoch in range(cfg.training.base_epochs):
        print("Epoch:", epoch)
        

        for batch in dataloader:




                source_frames = batch['source_frames']
                driving_frames = batch['driving_frames']
                video_id = batch['video_id'][0]

                # Access videos from dataloader2 for cycle consistency
                source_frames2 = batch['source_frames_star']
                driving_frames2 = batch['driving_frames_star']
                video_id2 = batch['video_id_star'][0]


                num_frames = len(driving_frames)
                len_source_frames = len(source_frames)
                len_driving_frames = len(driving_frames)
                len_source_frames2 = len(source_frames2)
                len_driving_frames2 = len(driving_frames2)

                for idx in range(num_frames):

                

                    # loop around if idx exceeds video length
                    source_frame = source_frames[idx % len_source_frames].to(device)
                    driving_frame = driving_frames[idx % len_driving_frames].to(device)

                    source_frame_star = source_frames2[idx % len_source_frames2].to(device)
                    driving_frame_star = driving_frames2[idx % len_driving_frames2].to(device)


                    with autocast():

                        # We use multiple loss functions for training, which can be split  into two groups.
                        # The first group consists of the standard training objectives for image synthesis. 
                        # These include perceptual [14] and GAN [ 33 ] losses that match 
                        # the predicted image ˆx𝑠→𝑑 to the  ground-truth x𝑑 . 
                        pred_frame = Gbase(source_frame, driving_frame)

                        # Obtain the foreground mask for the driving image
                        # foreground_mask = get_foreground_mask(source_frame)

                        # # Move the foreground mask to the same device as output_frame
                        # foreground_mask = foreground_mask.to(pred_frame.device)

                        # # Multiply the predicted and driving images with the foreground mask
                        # # masked_predicted_image = pred_frame * foreground_mask
                        # masked_target_image = driving_frame * foreground_mask

                        save_images = True
                        # Save the images
                        if save_images:
                            # vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
                            # vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
                            vutils.save_image(pred_frame, f"{output_dir}/pred_frame_{idx}.png")
                            # vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
                            # vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
                            # vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
                            # vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

                        # Calculate perceptual losses
                        loss_G_per = perceptual_loss_fn(pred_frame, source_frame)
                      
                        # Adversarial ground truths - from Kevin Fringe
                        valid = Variable(torch.Tensor(np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)
                        fake = Variable(torch.Tensor(-1 * np.ones((driving_frame.size(0), *patch))), requires_grad=False).to(device)

                        # real loss
                        real_pred = Dbase(driving_frame, source_frame)
                        loss_real = hinge_loss(real_pred, valid)

                        # fake loss
                        fake_pred = Dbase(pred_frame.detach(), source_frame)
                        loss_fake = hinge_loss(fake_pred, fake)

                        # Train discriminator
                        optimizer_D.zero_grad()
                        
                        # Calculate adversarial losses
                        real_pred = Dbase(driving_frame, source_frame)
                        fake_pred = Dbase(pred_frame.detach(), source_frame)
                        loss_D = discriminator_loss(real_pred, fake_pred, loss_type='lsgan')

                        # Backpropagate and update discriminator
                        scaler.scale(loss_D).backward()
                        scaler.step(optimizer_D)
                        scaler.update()

                        # Calculate adversarial losses
                        loss_G_adv = 0.5 * (loss_real + loss_fake)

                         # Feature matching loss
                        loss_fm = feature_matching_loss(pred_frame, driving_frame)
                    
                        # The other objective CycleGAN regularizes the training and introduces disentanglement between the motion and canonical space
                        # In order to calculate this loss, we use an additional source-driving  pair x𝑠∗ and x𝑑∗ , 
                        # which is sampled from a different video! and therefore has different appearance from the current x𝑠 , x𝑑 pair.

                        # produce the following cross-reenacted image: ˆx𝑠∗→𝑑 = Gbase (x𝑠∗ , x𝑑 )
                        cross_reenacted_image = Gbase(source_frame_star, driving_frame)
                        if save_images:
                            vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

                        # Store the motion descriptors z𝑠→𝑑(predicted) and z𝑠∗→𝑑 (star predicted) from the 
                        # respective forward passes of the base network.
                        _, _, z_pred = Gbase.motionEncoder(pred_frame) 
                        _, _, zd = Gbase.motionEncoder(driving_frame) 
                        
                        _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                        _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

              
                        # Calculate cycle consistency loss 
                        # We then arrange the motion descriptors into positive pairs P that
                        # should align with each other: P = (z𝑠→𝑑 , z𝑑 ), (z𝑠∗→𝑑 , z𝑑 ) , and
                        # the negative pairs: N = (z𝑠→𝑑 , z𝑑∗ ), (z𝑠∗→𝑑 , z𝑑∗ ) . These pairs are
                        # used to calculate the following cosine distance:

                        P = [(z_pred, zd)     ,(z_star__pred, zd)]
                        N = [(z_pred, zd_star),(z_star__pred, zd_star)]
                        loss_G_cos = cosine_loss(P, N)

                       
                        
                        # Backpropagate and update generator
                        optimizer_G.zero_grad()
                        total_loss = cfg.training.w_per * loss_G_per + \
                            cfg.training.w_adv * loss_G_adv + \
                            cfg.training.w_fm * loss_fm + \
                            cfg.training.w_cos * loss_G_cos
                        

                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()

                      

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G_cos.item():.4f}, Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def unnormalize(tensor):
    """
    Unnormalize a tensor using the specified mean and std.
    
    Args:
    tensor (torch.Tensor): The normalized tensor.
    mean (list): The mean used for normalization.
    std (list): The std used for normalization.
    
    Returns:
    torch.Tensor: The unnormalized tensor.
    """
    # Check if the tensor is on a GPU and if so, move it to the CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Ensure tensor is a float and detach it from the computation graph
    tensor = tensor.float().detach()
    
    # Unnormalize
    # Define mean and std used for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return tensor


def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        max_frames=100,  # Desired number of frames
        apply_warping=True 
   )


    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator().to(device)
    
    train_base(cfg, Gbase, Dbase, dataloader)    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(Dbase.state_dict(), 'Dbase.pth')


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)