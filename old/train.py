import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import get_cosine_schedule_with_warmup
from Net import FaceEncoder, FaceDecoder, DiffusionTransformer, IdentityLoss, DPELoss
import torchvision.transforms as transforms
from FaceHelper import FaceHelper
# from mae import VideoMAE
from model import PerceptualLoss
from EmoDataset import EMODataset
from omegaconf import OmegaConf
from score_sde_pytorch import VPSDE,get_score_fn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# VASA - disentangling - 
# Add pairwise head pose and facial dynamics transfer loss:
# The pairwise transfer loss encourages better separation between head pose and facial dynamics.
# The identity similarity loss reinforces disentanglement between identity and motions.
# You'll need to tune the lambda values (λ_pairwise and λ_identity) to balance these new losses with your existing losses. Also, ensure your dataloader can provide appropriate pairs of frames as required by these loss functions.

def pairwise_transfer_loss(self, I1, I2):
    # Extract latent variables
    V_app1, z_id1, z_pose1, z_dyn1 = self.encode(I1)
    V_app2, z_id2, z_pose2, z_dyn2 = self.encode(I2)
    
    # Transfer pose
    I_pose = self.decode(V_app1, z_id1, z_pose2, z_dyn1)
    
    # Transfer dynamics
    I_dyn = self.decode(V_app1, z_id1, z_pose1, z_dyn2)
    
    # Compute discrepancy loss
    loss = F.l1_loss(I_pose, I_dyn)
    
    return loss

#Add face identity similarity loss for cross-identity transfers:

def identity_similarity_loss(self, I1, I2):
    # Extract latent variables
    V_app1, z_id1, z_pose1, z_dyn1 = self.encode(I1)
    V_app2, z_id2, z_pose2, z_dyn2 = self.encode(I2)
    
    # Cross-identity transfer
    I_transfer = self.decode(V_app1, z_id1, z_pose2, z_dyn2)
    
    # Extract identity features
    id_features1 = self.face_recognition_model(I1)
    id_features_transfer = self.face_recognition_model(I_transfer)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(id_features1, id_features_transfer)
    
    # We want to maximize similarity, so we minimize negative similarity
    loss = -similarity.mean()
    
    return loss



def train_step(self, I1, I2, I3, I4):
    # I1 and I2 are from the same video, I3 and I4 are from different videos
    
    # Existing losses
    reconstruction_loss = self.reconstruction_loss(I1)
    
    # New disentangling losses
    pairwise_loss = self.pairwise_transfer_loss(I1, I2)
    identity_loss = self.identity_similarity_loss(I3, I4)
    
    # Total loss
    total_loss = reconstruction_loss + self.lambda_pairwise * pairwise_loss + self.lambda_identity * identity_loss
    
    # Optimization step
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()
    
    return total_loss
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


def compute_lip_sync_loss(original_landmarks, generated_landmarks):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(original_landmarks, generated_landmarks)


def train_stage1(cfg, encoder, decoder, diffusion_transformer, dataloader):
    # patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    # hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    optimizer_G = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion_transformer.parameters()), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = get_cosine_schedule_with_warmup(optimizer_G, num_warmup_steps=0, num_training_steps=len(dataloader) * cfg.training.base_epochs)

    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})


    identity_loss = IdentityLoss()
    dpe_loss = DPELoss()
    # SDE parameters
    sde = VPSDE(beta_min=0.1, beta_max=20, N=1000)

    for epoch in range(cfg.training.base_epochs):
        print("Epoch:", epoch)

        for batch in dataloader:
            source_frames = batch['source_frames']
            driving_frames = batch['driving_frames']
            video_id = batch['video_id'][0]

            num_frames = len(driving_frames)
            len_source_frames = len(source_frames)
            len_driving_frames = len(driving_frames)

            for idx in range(num_frames):
                source_frame = source_frames[idx % len_source_frames].to(device)
                driving_frame = driving_frames[idx % len_driving_frames].to(device)

                # Forward pass through encoder and decoder
                appearance_volume, identity_code, head_pose, facial_dynamics = encoder(source_frame)
                reconstructed_face = decoder(appearance_volume, identity_code, head_pose, facial_dynamics)

                # Diffusion process
                t = torch.rand(facial_dynamics.shape[0], device=facial_dynamics.device) * sde.T
                mean, std = sde.marginal_prob(facial_dynamics, t)
                noise = torch.randn_like(facial_dynamics)
                noisy_facial_dynamics = mean + std[:, None] * noise

                # Diffusion transformer for generating dynamics
                generated_dynamics = diffusion_transformer(noisy_facial_dynamics, audio_features[:, idx])

                # Compute losses
                loss_G_per = perceptual_loss_fn(reconstructed_face, source_frame)
                loss_G_cos = cosine_loss(generated_dynamics, facial_dynamics)
                loss_G_adv = discriminator_loss(reconstructed_face, source_frame)
                loss_fm = feature_matching_loss(reconstructed_face, source_frame)
                loss_identity = identity_loss(reconstructed_face, source_frame)
                loss_dpe = dpe_loss(reconstructed_face, source_frame)

                total_loss = cfg.training.w_per * loss_G_per + \
                             cfg.training.w_adv * loss_G_adv + \
                             cfg.training.w_fm * loss_fm + \
                             cfg.training.w_cos * loss_G_cos + \
                             cfg.training.w_identity * loss_identity + \
                             cfg.training.w_dpe * loss_dpe

                # Optimization step
                optimizer_G.zero_grad()
                total_loss.backward()
                optimizer_G.step()
                scheduler_G.step()

        print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], Total Loss: {total_loss.item()}")

        # Save models
        torch.save(encoder.state_dict(), f"encoder_epoch{epoch+1}.pth")
        torch.save(decoder.state_dict(), f"decoder_epoch{epoch+1}.pth")
        torch.save(diffusion_transformer.state_dict(), f"diffusion_transformer_epoch{epoch+1}.pth")



def train_stage2(cfg,  encoder, decoder, diffusion_transformer, dataloader):
    print("Stage 2: Holistic Facial Dynamics Generation")
    params_stage2 = list(diffusion_transformer.parameters())
    optimizer_stage2 = optim.Adam(params_stage2, lr=cfg.training.lr, weight_decay=1e-5)
    scheduler_stage2 = get_cosine_schedule_with_warmup(optimizer_stage2, num_warmup_steps=0, num_training_steps=len(dataloader) *  cfg.training.diffusion_epochs)

    guidance_scale = 1.5  # Set the guidance scale for CFG

    # SDE parameters
    sde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})

    identity_loss = IdentityLoss()

    fh = FaceHelper()
    for epoch in range( cfg.training.diffusion_epochs):
        for batch in dataloader:
            video_frames, _, audio_features, _ = batch
            video_frames = video_frames.cuda()
            audio_features = audio_features.cuda()

            for frame_idx in range(video_frames.shape[1]):
                frame = video_frames[:, frame_idx]

                # Forward pass through encoder
                appearance_volume, identity_code, head_pose, facial_dynamics = encoder(frame)

                # Generate dynamics using structured inputs
                gaze_direction = fh.estimate_gaze(frame)
                head_distance = fh.head_distance_estimator(frame)
                emotion_offset = fh.detect_emotions(frame)

                # Diffusion process
                t = torch.rand(facial_dynamics.shape[0], device=facial_dynamics.device) * sde.T
                mean, std = sde.marginal_prob(facial_dynamics, t)
                noise = torch.randn_like(facial_dynamics)
                noisy_facial_dynamics = mean + std[:, None] * noise

                # Diffusion transformer for generating dynamics with CFG
                generated_dynamics = diffusion_transformer(
                    noisy_facial_dynamics, audio_features[:, frame_idx], gaze_direction, head_distance, emotion_offset, guidance_scale=guidance_scale)

                # Compute diffusion loss
                score_fn = get_score_fn(sde, diffusion_transformer, train=True, continuous=True)
                score = score_fn(noisy_facial_dynamics, t)
                diffusion_loss = torch.mean(torch.sum((score * std[:, None] + noise) ** 2, dim=(1, 2)))

                # Face reconstruction using the modified FaceDecoder
                reconstructed_face = decoder(appearance_volume, identity_code, head_pose, generated_dynamics)

                # Get lip landmarks for original and reconstructed face
                original_lip_landmarks = fh.mediapipe_lip_landmark_detector(frame)
                generated_lip_landmarks = fh.mediapipe_lip_landmark_detector(reconstructed_face.detach())

                # Compute lip sync loss
                lip_sync_loss = compute_lip_sync_loss(original_lip_landmarks, generated_lip_landmarks)

                # VGG perceptual loss
                vgg_loss_frame = perceptual_loss_fn(reconstructed_face, frame)

                # Identity loss
                identity_loss_value = identity_loss(frame, reconstructed_face)

                # Total loss
                total_loss = diffusion_loss + lip_sync_loss + vgg_loss_frame + identity_loss_value

                # Optimization step
                optimizer_stage2.zero_grad()
                total_loss.backward()
                optimizer_stage2.step()
                scheduler_stage2.step()

            print(f"Stage 2 - Epoch [{epoch+1}/{cfg.training.diffusion_epochs}], Total Loss: {total_loss.item()}, Diffusion Loss: {diffusion_loss.item()}, Lip Sync Loss: {lip_sync_loss.item()}, Perceptual Loss: {vgg_loss_frame.item()}, Identity Loss: {identity_loss_value.item()}")

        # Save the diffusion transformer model
        torch.save(diffusion_transformer.state_dict(), f"diffusion_transformer_stage2_epoch{epoch+1}.pth")

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
    encoder = FaceEncoder().to(device)
    decoder = FaceDecoder().to(device)
    diffusion_transformer = DiffusionTransformer(num_layers=6, num_heads=8, hidden_size=512).to(device)

    # Stage 1 training
    train_stage1(cfg,  encoder, decoder, diffusion_transformer, dataloader)

    # Stage 2 training
    dataset.stage = 'stage2'
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    train_stage2(cfg,  encoder, decoder, diffusion_transformer, dataloader)


if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)