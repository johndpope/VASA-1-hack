import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import get_cosine_schedule_with_warmup
from Net import FaceEncoder, FaceDecoder, DiffusionTransformer, IdentityLoss, DPELoss
import torchvision.transforms as transforms
from FaceHelper import FaceHelper
from mae import VideoMAE
from vgg19 import VGGLoss
from EmoDataset import EMODataset
from omegaconf import OmegaConf
from score_sde_pytorch import VPSDE,get_score_fn


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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


def compute_lip_sync_loss(original_landmarks, generated_landmarks):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(original_landmarks, generated_landmarks)


def train_stage1(cfg, encoder, decoder, diffusion_transformer, dataloader):
    patch = (1, cfg.data.train_width // 2 ** 4, cfg.data.train_height // 2 ** 4)
    # hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
    feature_matching_loss = nn.MSELoss()
    optimizer_G = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion_transformer.parameters()), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = get_cosine_schedule_with_warmup(optimizer_G, num_warmup_steps=0, num_training_steps=len(dataloader) * cfg.training.epochs)

    perceptual_loss_fn = VGGLoss(device)
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

        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Total Loss: {total_loss.item()}")

        # Save models
        torch.save(encoder.state_dict(), f"encoder_epoch{epoch+1}.pth")
        torch.save(decoder.state_dict(), f"decoder_epoch{epoch+1}.pth")
        torch.save(diffusion_transformer.state_dict(), f"diffusion_transformer_epoch{epoch+1}.pth")



def train_stage2(cfg,  encoder, decoder, diffusion_transformer, dataloader):
    print("Stage 2: Holistic Facial Dynamics Generation")
    params_stage2 = list(diffusion_transformer.parameters())
    optimizer_stage2 = optim.Adam(params_stage2, lr=cfg.training.lr, weight_decay=1e-5)
    scheduler_stage2 = get_cosine_schedule_with_warmup(optimizer_stage2, num_warmup_steps=0, num_training_steps=len(dataloader) *  cfg.training.epochs)

    guidance_scale = 1.5  # Set the guidance scale for CFG

    # SDE parameters
    sde = VPSDE(beta_min=0.1, beta_max=20, N=1000)

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
                vgg_loss_frame = vgg_loss(reconstructed_face, frame)

                # Identity loss
                identity_loss_value = identity_loss(frame, reconstructed_face)

                # Total loss
                total_loss = diffusion_loss + lip_sync_loss + vgg_loss_frame + identity_loss_value

                # Optimization step
                optimizer_stage2.zero_grad()
                total_loss.backward()
                optimizer_stage2.step()
                scheduler_stage2.step()

            print(f"Stage 2 - Epoch [{epoch+1}/{num_epochs_stage2}], Total Loss: {total_loss.item()}, Diffusion Loss: {diffusion_loss.item()}, Lip Sync Loss: {lip_sync_loss.item()}, Perceptual Loss: {vgg_loss_frame.item()}, Identity Loss: {identity_loss_value.item()}")

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