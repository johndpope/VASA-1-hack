import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import math

from dataset import VASADataset

from resnet import resnet18,resnet50
 
# Megaportraits - https://github.com/johndpope/MegaPortrait-hack/issues/36
from model import CustomResNet50,Eapp,FEATURE_SIZE,COMPRESS_DIM,WarpGeneratorS2C,WarpGeneratorC2D,G3d,G2d,apply_warping_field

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from insightface.app import FaceAnalysis


# TODO - wire this up as off the shelf pretrained model for Head Pose Encoder
# -  self.rotation_net =  SixDRepNet_Detector()
from mysixdrepnet import SixDRepNet_Detector

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AudioEncoder(nn.Module):
    """
    Audio feature encoder using Wav2Vec2 architecture
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=10, stride=5, padding=4)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(512, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.layer_norm(x)
        return x

class VASAFaceEncoder(nn.Module):
    """
    Enhanced face encoder with disentangled representations
    """
    def __init__(self, 
                 feature_dim: int = 512,
                 appearance_dim: int = 96,
                 identity_dim: int = 512,
                 dynamics_dim: int = 256):
        super().__init__()
        
        # 3D Appearance Volume Encoder (V_app)
        self.appearance_encoder = nn.Sequential(
            *list(resnet18(pretrained=True).children())[:-2],
            nn.Conv2d(512, appearance_dim, 1),
            nn.AdaptiveAvgPool3d((appearance_dim, 16, 64))
        )
        
        # Identity Encoder (z_id)
        self.identity_encoder = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-1],
            nn.Linear(2048, identity_dim)
        )
        
        # Head Pose Encoder (z_pos)
        self.pose_encoder = nn.Sequential(
            *list(resnet18(pretrained=True).children())[:-1],
            nn.Linear(512, 6)  # 3 rotation + 3 translation
        )
        
        # Facial Dynamics Encoder (z_exp)
        self.dynamics_encoder = nn.Sequential(
            *list(resnet18(pretrained=False).children())[:-2],
            nn.Conv2d(512, dynamics_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract all components
        app_vol = self.appearance_encoder(x)
        identity = self.identity_encoder(x)
        pose = self.pose_encoder(x)
        dynamics = self.dynamics_encoder(x)
        
        # Split pose into rotation and translation
        rotation, translation = pose.split([3, 3], dim=1)
        
        return {
            'appearance_volume': app_vol,
            'identity': identity,
            'rotation': rotation,
            'translation': translation,
            'dynamics': dynamics
        }

class VASADiffusionTransformer(nn.Module):
    """
    Diffusion Transformer with enhanced conditioning and CFG
    """
    def __init__(self,
                 seq_length: int = 25,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.1,
                 motion_dim: int = 256,
                 audio_dim: int = 128):
        super().__init__()
        
        # Embeddings
        self.motion_embed = nn.Linear(motion_dim, d_model)
        self.audio_embed = nn.Linear(audio_dim, d_model)
        self.gaze_embed = nn.Linear(2, d_model)
        self.distance_embed = nn.Linear(1, d_model)
        self.emotion_embed = nn.Linear(512, d_model)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, motion_dim)
        
        self.condition_dropout = dropout

    def _maybe_mask_condition(self, 
                            condition: torch.Tensor, 
                            force_mask: bool = False) -> torch.Tensor:
        """Randomly mask conditions for classifier-free guidance training"""
        if self.training or force_mask:
            mask = torch.rand(condition.shape[0], 1, 1, device=condition.device) < self.condition_dropout
            condition = torch.where(mask, torch.zeros_like(condition), condition)
        return condition

    def forward(self, 
                x: torch.Tensor,
                timestep: torch.Tensor,
                audio_features: torch.Tensor,
                conditions: Dict[str, torch.Tensor],
                cfg_scales: Optional[Dict[str, float]] = None) -> torch.Tensor:
        
        # Time embedding
        time_emb = self.time_embed(timestep.unsqueeze(-1))
        
        # Embed all inputs
        motion_emb = self.motion_embed(x)
        audio_emb = self.audio_embed(self._maybe_mask_condition(audio_features))
        gaze_emb = self.gaze_embed(self._maybe_mask_condition(conditions['gaze']))
        dist_emb = self.distance_embed(self._maybe_mask_condition(conditions['distance']))
        emotion_emb = self.emotion_embed(self._maybe_mask_condition(conditions['emotion']))
        
        # Combine embeddings
        combined = (motion_emb + time_emb.unsqueeze(1) + audio_emb + 
                   gaze_emb.unsqueeze(1) + dist_emb.unsqueeze(1) + 
                   emotion_emb.unsqueeze(1))
        
        # Add positional encoding
        combined = self.pos_encoding(combined)
        
        # Apply transformer
        output = self.transformer(combined)
        output = self.output_proj(output)
        
        # Apply classifier-free guidance if specified
        if cfg_scales is not None and self.training:
            outputs = {'full': output}
            
            # Generate unconditional output
            uncond_output = self.forward(
                x, timestep, audio_features,
                {k: torch.zeros_like(v) for k, v in conditions.items()},
                None
            )
            outputs['uncond'] = uncond_output
            
            # Generate outputs with individual conditions masked
            for cond_name, scale in cfg_scales.items():
                if scale > 0:
                    masked_conditions = conditions.copy()
                    masked_conditions[cond_name] = torch.zeros_like(conditions[cond_name])
                    cond_output = self.forward(
                        x, timestep, audio_features,
                        masked_conditions,
                        None
                    )
                    outputs[f'masked_{cond_name}'] = cond_output
            
            return outputs
        
        return output

class VASADiffusion:
    """
    Diffusion process handler for VASA
    """
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
            
        alpha_bar = self.alpha_bars[t]
        
        # Expand dimensions to match x0
        alpha_bar = alpha_bar.view(-1, *([1] * (len(x0.shape) - 1)))
        
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    def p_sample(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor, 
                 conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample from reverse diffusion process"""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        
        # Expand dimensions
        beta_t = beta_t.view(-1, *([1] * (len(xt.shape) - 1)))
        alpha_t = alpha_t.view(-1, *([1] * (len(xt.shape) - 1)))
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (len(xt.shape) - 1)))
        
        # Model prediction
        pred = model(xt, t, conditions)
        
        # Compute mean and variance
        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred)
        var = beta_t * (1 - alpha_bar_t / alpha_t)
        
        # Sample
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(var) * noise




class VASAFaceEncoder(nn.Module):
    """
    Enhanced face encoder aligned with VASA paper's disentangled representation.
    Building on MegaPortraits architecture with improved disentanglement.
    """
    def __init__(self, feature_size=64):
        super().__init__()
        # Base encoders from MegaPortraits
        self.appearanceEncoder = Eapp()
        self.identityEncoder = CustomResNet50()
        
        # Enhanced head pose estimator with better disentanglement
        self.headPoseEncoder = nn.Sequential(
            *list(resnet18(pretrained=True).children())[:-1],
            nn.Linear(512, 6)  # 3 for rotation, 3 for translation
        )
        
        # Enhanced facial dynamics encoder with holistic representation
        self.facialDynamicsEncoder = nn.Sequential(
            *list(resnet18(pretrained=False).children())[:-2],
            nn.AdaptiveAvgPool2d(feature_size),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * feature_size * feature_size, COMPRESS_DIM)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input face image [batch_size, 3, H, W]
        Returns:
            Tuple of (appearance_volume, identity_code, head_pose, facial_dynamics)
        """
        # Extract 3D appearance volume (V_app)
        appearance_volume = self.appearanceEncoder(x)[0]
        
        # Extract identity code (z_id)
        identity_code = self.identityEncoder(x)
        
        # Extract head pose (z_pos)
        head_pose = self.headPoseEncoder(x)
        rotation, translation = head_pose.split([3, 3], dim=1)
        
        # Extract holistic facial dynamics (z_exp)
        facial_dynamics = self.facialDynamicsEncoder(x)
        
        return appearance_volume, identity_code, (rotation, translation), facial_dynamics

class VASAFaceDecoder(nn.Module):
    """
    Enhanced face decoder aligned with VASA paper's generation process.
    """
    def __init__(self):
        super().__init__()
        # MegaPortraits warping components
        self.warp_s2c = WarpGeneratorS2C(num_channels=512)
        self.warp_c2d = WarpGeneratorC2D(num_channels=512)
        
        # Enhanced 3D and 2D generators
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

    def forward(self, appearance_volume: torch.Tensor, identity_code: torch.Tensor, 
                head_pose: Tuple[torch.Tensor, torch.Tensor], facial_dynamics: torch.Tensor) -> torch.Tensor:
        rotation, translation = head_pose
        
        # Source to canonical warping
        w_s2c = self.warp_s2c(rotation, translation, facial_dynamics, identity_code)
        canonical_volume = apply_warping_field(appearance_volume, w_s2c)
        
        # 3D to 2D processing
        vc2d = self.G3d(canonical_volume)
        w_c2d = self.warp_c2d(rotation, translation, facial_dynamics, identity_code)
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        
        # Final generation
        vc2d_projected = torch.sum(vc2d_warped, dim=2)
        output = self.G2d(vc2d_projected)
        
        return output

class VASADiffusionTransformer(nn.Module):
    """
    Diffusion Transformer aligned with VASA paper's specifications.
    8-layer transformer with improved conditioning and CFG support.
    """
    def __init__(
        self,
        seq_length: int = 25,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        motion_dim: int = None,
        audio_dim: int = None
    ):
        super().__init__()
        
        # Input embeddings
        self.motion_embed = nn.Linear(motion_dim, d_model)
        self.audio_embed = nn.Linear(audio_dim, d_model)
        self.gaze_embed = nn.Linear(2, d_model)  # (θ,φ)
        self.dist_embed = nn.Linear(1, d_model)
        self.emotion_embed = nn.Linear(512, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Main transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, motion_dim)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        cfg_scales: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input motion sequence [batch_size, seq_len, motion_dim]
            audio_features: Audio features [batch_size, seq_len, audio_dim]
            conditions: Dict containing 'gaze', 'distance', 'emotion'
            cfg_scales: Dict of classifier-free guidance scales
        """
        # Embed inputs
        motion_emb = self.motion_embed(x)
        audio_emb = self.audio_embed(audio_features)
        gaze_emb = self.gaze_embed(conditions['gaze'])
        dist_emb = self.dist_embed(conditions['distance'])
        emotion_emb = self.emotion_embed(conditions['emotion'])
        
        # Combine embeddings
        combined = motion_emb + audio_emb + gaze_emb + dist_emb + emotion_emb + self.pos_encoding
        
        # Apply transformer
        output = self.transformer(combined)
        output = self.output_proj(output)
        
        # Apply CFG if scales provided
        if cfg_scales is not None:
            null_conditions = {k: torch.zeros_like(v) for k, v in conditions.items()}
            null_output = self.forward(x, audio_features, null_conditions)
            
            for cond_type, scale in cfg_scales.items():
                if scale > 0:
                    curr_conditions = conditions.copy()
                    curr_conditions[cond_type] = torch.zeros_like(conditions[cond_type])
                    cond_output = self.forward(x, audio_features, curr_conditions)
                    output = output + scale * (output - cond_output)
        
        return output

class VASALoss(nn.Module):
    """
    Combined loss function aligned with VASA paper.
    Includes DPE losses and identity preservation.
    """
    def __init__(self):
        super().__init__()
        self.dpe_loss = DPELoss()
        self.identity_loss = IdentityLoss()
        self.recon_loss = nn.L1Loss()
        
    def forward(self, generated: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # DPE losses for disentanglement
        losses['dpe'] = self.dpe_loss(
            generated['source'], generated['target'],
            generated['source_pose_transfer'], generated['target_pose_transfer'],
            generated['source_identity'], generated['target_identity'],
            generated['cross_identity_transfer']
        )
        
        # Identity preservation
        losses['identity'] = self.identity_loss(
            generated['output'], target['source_image']
        )
        
        # Reconstruction
        losses['recon'] = self.recon_loss(
            generated['output'], target['target_image']
        )
        
        # Total loss
        losses['total'] = losses['dpe'] + losses['identity'] + losses['recon']
        
        return losses
    




    
