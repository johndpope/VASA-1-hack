import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import get_cosine_schedule_with_warmup
from Net import FaceEncoder, FaceDecoder, DiffusionTransformer, IdentityLoss, DPELoss
import torchvision.transforms as transforms
from FaceHelper import FaceHelper
from mae import VideoMAE
from vgg19 import VGGLoss




# Denoising score matching objective [46 ] Score-based generative modeling through stochastic differential equations
# cherry pick from https://github.com/yang-song/score_sde_pytorch


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G

def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, VPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

# Training configuration
num_epochs_stage1 = 100
num_epochs_stage2 = 100
batch_size = 32
learning_rate = 0.001

# SDE parameters
sde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
sampling_eps = 1e-3

# Initialize models
encoder = FaceEncoder()
decoder = FaceDecoder()
fh = FaceHelper()
diffusion_transformer = DiffusionTransformer(num_layers=6, num_heads=8, hidden_size=512)
motion_field_estimator = MotionFieldEstimator(model_scale='small')
vgg_loss = VGGLoss()
identity_loss = IdentityLoss()
dpe_loss = DPELoss()

# Data loading and transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = VideoMAE(
    root='/path/to/voxceleb2/root',
    setting='/path/to/voxceleb2/train.txt',
    train=True,
    image_size=256,
    audio_conf={
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'noise': False,
        'mean': -4.6476,
        'std': 4.5699,
    },
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def compute_lip_sync_loss(original_landmarks, generated_landmarks):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(original_landmarks, generated_landmarks)

# Stage 1: Face Latent Space Construction
print("Stage 1: Face Latent Space Construction")
params_stage1 = list(encoder.parameters()) + list(decoder.parameters())
optimizer_stage1 = optim.Adam(params_stage1, lr=learning_rate, weight_decay=1e-5)
scheduler_stage1 = get_cosine_schedule_with_warmup(optimizer_stage1, num_warmup_steps=0, num_training_steps=len(dataloader) * num_epochs_stage1)
for epoch in range(num_epochs_stage1):
    for batch in dataloader:
        video_frames, _, _, _ = batch
        video_frames = video_frames.cuda()

        for frame_idx in range(0, video_frames.shape[1], 2):  # Process frames in pairs
            frame1 = video_frames[:, frame_idx]
            frame2 = video_frames[:, frame_idx + 1]

            # Forward pass through encoder and decoder for frame1
            appearance_volume1, identity_code1, head_pose1, facial_dynamics1 = encoder(frame1)
            reconstructed_face1 = decoder(appearance_volume1, identity_code1, head_pose1, facial_dynamics1)

            # Forward pass through encoder and decoder for frame2
            appearance_volume2, identity_code2, head_pose2, facial_dynamics2 = encoder(frame2)
            reconstructed_face2 = decoder(appearance_volume2, identity_code2, head_pose2, facial_dynamics2)

            # Pairwise head pose and facial dynamics transfer
            pose_transfer_face1 = decoder(appearance_volume1, identity_code1, head_pose2, facial_dynamics1)
            dyn_transfer_face2 = decoder(appearance_volume2, identity_code2, head_pose2, facial_dynamics1)

            # Cross-identity pose and facial motion transfer
            cross_id_transfer_face1 = decoder(appearance_volume1, identity_code2, head_pose1, facial_dynamics1)
            cross_id_transfer_face2 = decoder(appearance_volume2, identity_code1, head_pose2, facial_dynamics2)

            # Compute disentanglement losses
            dpe_loss_value = dpe_loss(frame1, frame2, pose_transfer_face1, dyn_transfer_face2, frame1, frame2, cross_id_transfer_face1, cross_id_transfer_face2)

            # Reconstruction loss
            reconstruction_loss = nn.L1Loss()(frame1, reconstructed_face1) + nn.L1Loss()(frame2, reconstructed_face2)

            # VGG perceptual loss
            vgg_loss_frame1 = vgg_loss(reconstructed_face1, frame1)
            vgg_loss_frame2 = vgg_loss(reconstructed_face2, frame2)
            perceptual_loss = vgg_loss_frame1 + vgg_loss_frame2

            # Identity loss
            identity_loss_value = identity_loss(frame1, reconstructed_face1) + identity_loss(frame2, reconstructed_face2)

            # Total loss
            total_loss = reconstruction_loss + dpe_loss_value + perceptual_loss + identity_loss_value

            # Optimization step
            optimizer_stage1.zero_grad()
            total_loss.backward()
            optimizer_stage1.step()
            scheduler_stage1.step()

    print(f"Stage 1 - Epoch [{epoch+1}/{num_epochs_stage1}], Total Loss: {total_loss.item()}, Reconstruction Loss: {reconstruction_loss.item()}, DPE Loss: {dpe_loss_value.item()}, Perceptual Loss: {perceptual_loss.item()}, Identity Loss: {identity_loss_value.item()}")

    # Save the encoder and decoder models
    torch.save(encoder.state_dict(), f"encoder_stage1_epoch{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"decoder_stage1_epoch{epoch+1}.pth")
# ...

# Stage 2: Holistic Facial Dynamics Generation
print("Stage 2: Holistic Facial Dynamics Generation")
params_stage2 = list(diffusion_transformer.parameters())
optimizer_stage2 = optim.Adam(params_stage2, lr=learning_rate, weight_decay=1e-5)
scheduler_stage2 = get_cosine_schedule_with_warmup(optimizer_stage2, num_warmup_steps=0, num_training_steps=len(dataloader) * num_epochs_stage2)

guidance_scale = 1.5  # Set the guidance scale for CFG

for epoch in range(num_epochs_stage2):
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