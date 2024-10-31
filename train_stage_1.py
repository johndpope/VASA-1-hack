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



output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




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
    pairwise_transfer_loss = PairwiseTransferLoss()
  #  identity_similarity_loss = IdentitySimilarityLoss()
    identity_similarity_loss = PerceptualLoss(device, weights={'vgg19': 0.0, 'vggface': 1.0, 'gaze': 0.0,'lpips':0.0}) # focus on face

    scaler = GradScaler()


    for epoch in range(start_epoch, cfg.training.base_epochs):
        print("Epoch:", epoch)

        epoch_loss_G = 0
        epoch_loss_D = 0

        fid_score = 0
        csim_score = 0
        lpips_score = 0


        
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
                        pred_frame,pred_pyramids = Gbase(source_frame, driving_frame)

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

                        # Calculate perceptual losses - use pyramid 
                        # loss_G_per = perceptual_loss_fn(pred_frame, source_frame)
                      
                        loss_G_per = 0
                        for scale, pred_scaled in pred_pyramids.items():
                            target_scaled = F.interpolate(driving_frame, size=pred_scaled.shape[2:], mode='bilinear', align_corners=False)
                            loss_G_per += perceptual_loss_fn(pred_scaled, target_scaled)

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

                        scaler.scale(loss_D).backward()
                        scaler.step(optimizer_D)
                        scaler.update()

                        # Calculate adversarial losses
                        loss_G_adv = 0.5 * (loss_real + loss_fake)

                         # Feature matching loss
                        loss_fm = feature_matching_loss(pred_frame, driving_frame)

                        



                        # New disentangling losses - from VASA paper
                        # I1 and I2 are from the same video, I3 and I4 are from different videos
    
                        # Get the next frame index, wrapping around if necessary
                        next_idx = (idx + 20) % len_source_frames

                        I1 = source_frame
                        I2 = source_frames[next_idx].to(device)
                        I3 = source_frame_star
                        I4 = source_frames2[next_idx % len_source_frames2].to(device)
                        loss_pairwise = pairwise_transfer_loss(Gbase,I1, I2)
                        loss_identity = identity_similarity_loss(I3, I4)


                        

                        # The other objective CycleGAN regularizes the training and introduces disentanglement between the motion and canonical space
                        # In order to calculate this loss, we use an additional source-driving  pair x𝑠∗ and x𝑑∗ , 
                        # which is sampled from a different video! and therefore has different appearance from the current x𝑠 , x𝑑 pair.

                        # produce the following cross-reenacted image: ˆx𝑠∗→𝑑 = Gbase (x𝑠∗ , x𝑑 )
                        # 
                        cross_reenacted_image,_ = Gbase(source_frame_star, driving_frame)
                        if save_images:
                            vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")
                
                        # # Store the motion descriptors z𝑠→𝑑(predicted) and z𝑠∗→𝑑 (star predicted) from the 
                        # # respective forward passes of the base network.                        
                        _, _, z_pred = Gbase.motionEncoder(pred_frame) 
                        _, _, zd = Gbase.motionEncoder(driving_frame) 
                        
                        _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                        _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

                
                        # # Calculate cycle consistency loss 
                        # # We then arrange the motion descriptors into positive pairs P that
                        # # should align with each other: P = (z𝑠→𝑑 , z𝑑 ), (z𝑠∗→𝑑 , z𝑑 ) , and
                        # # the negative pairs: N = (z𝑠→𝑑 , z𝑑∗ ), (z𝑠∗→𝑑 , z𝑑∗ ) . These pairs are
                        # # used to calculate the following cosine distance:

                        P = [(z_pred, zd)     ,(z_star__pred, zd)]
                        N = [(z_pred, zd_star),(z_star__pred, zd_star)]
                        loss_G_cos = cosine_loss(P, N)
                        
                
                        
                        # Backpropagate and update generator
                        optimizer_G.zero_grad()
                          # Total generator loss
                        total_loss = cfg.training.w_per * loss_G_per + \
                            cfg.training.w_adv * loss_G_adv + \
                            cfg.training.w_fm * loss_fm + \
                            cfg.training.w_cos * loss_G_cos + \
                            cfg.training.w_pairwise * loss_pairwise + \
                            cfg.training.w_identity * loss_identity
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()


                        epoch_loss_G += total_loss.item()
                        epoch_loss_D += loss_D.item()





        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        
 


        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G_cos.item():.4f}, Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_G_state_dict': Gbase.state_dict(),
                'model_D_state_dict': Dbase.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, f"checkpoint_epoch{epoch+1}.pth")




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