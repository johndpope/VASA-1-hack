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
from vasa_config import VASAConfig

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


import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from tqdm import tqdm

class AdaCacheDiffusionSampler:
    """
    Enhanced diffusion sampler with Adaptive Caching for VASA
    """
    def __init__(
        self,
        num_steps: int = 50,
        min_beta: float = 1e-4,
        max_beta: float = 0.02,
        cache_metrics: str = "l1",  # l1, l2, or cosine
        base_rates: Optional[Dict[str, int]] = None,
        use_motion_reg: bool = True
    ):
        self.num_steps = num_steps
        
        # Set up diffusion parameters
        self.beta = torch.linspace(min_beta, max_beta, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Pre-compute sampling parameters
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alpha)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        
        # AdaCache parameters
        self.cache_metrics = cache_metrics
        self.use_motion_reg = use_motion_reg
        
        # Default cache rates for different speed profiles
        self.base_rates = base_rates or {
            "fast": [12, 10, 8, 6, 4, 3],
            "medium": [8, 6, 4, 2, 1],
            "slow": [2, 1]
        }
        
        # Cache rate thresholds (can be tuned)
        self.thresholds = {
            0.08: 6,
            0.16: 5,
            0.24: 4,
            0.32: 3,
            0.40: 2,
            1.00: 1
        }

    def compute_distance_metric(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        steps_apart: int
    ) -> torch.Tensor:
        """Compute distance between current and previous representations"""
        if self.cache_metrics == "l1":
            return torch.abs(current - previous).mean() / steps_apart
        elif self.cache_metrics == "l2":
            return torch.sqrt(((current - previous) ** 2).mean()) / steps_apart
        else:  # cosine
            current_flat = current.view(current.size(0), -1)
            previous_flat = previous.view(previous.size(0), -1)
            return 1 - torch.nn.functional.cosine_similarity(
                current_flat, previous_flat, dim=1
            ).mean()

    def compute_motion_score(
        self,
        features: torch.Tensor,
        frame_step: int = 1
    ) -> torch.Tensor:
        """Compute motion score based on frame differences"""
        if not self.use_motion_reg:
            return torch.tensor(1.0, device=features.device)
            
        B, T = features.shape[:2]
        frame_diffs = []
        
        for i in range(0, T - frame_step, frame_step):
            diff = torch.abs(features[:, i+frame_step:] - features[:, :-frame_step])
            frame_diffs.append(diff.mean())
            
        motion_score = torch.stack(frame_diffs).mean()
        return motion_score

    def get_cache_rate(
        self,
        distance: torch.Tensor,
        motion_score: Optional[torch.Tensor] = None
    ) -> int:
        """Determine caching rate based on distance and motion"""
        if motion_score is not None:
            distance = distance * (1 + motion_score)
            
        for threshold, rate in self.thresholds.items():
            if distance < threshold:
                return rate
        return 1

    @torch.no_grad()
    def ddim_sample_with_adacache(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        conditions: Dict[str, torch.Tensor],
        cfg_scales: Dict[str, float],
        eta: float = 0.0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Enhanced DDIM sampling with AdaCache
        """
        device = device or next(model.parameters()).device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        # Cache storage
        cached_residuals = {}
        last_computed_step = -1
        
        pbar = tqdm(reversed(range(self.num_steps)), desc='DDIM Sampling with AdaCache')
        
        for t in pbar:
            # Get diffusion parameters
            at = self.alpha_bar[t]
            at_next = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
            t_embed = torch.ones(batch_size, device=device) * t
            
            # Check if we need to compute or can use cache
            compute_step = False
            if t == self.num_steps - 1:  # First step
                compute_step = True
            elif last_computed_step >= 0:
                # Get representations from model (without computing gradients)
                with torch.no_grad():
                    current_rep = model.get_intermediate_rep(x, t_embed, conditions)
                    
                # Compute distance metric
                distance = self.compute_distance_metric(
                    current_rep,
                    cached_residuals['rep'],
                    last_computed_step - t
                )
                
                # Get motion score if enabled
                motion_score = None
                if self.use_motion_reg:
                    motion_score = self.compute_motion_score(current_rep)
                
                # Determine cache rate
                cache_rate = self.get_cache_rate(distance, motion_score)
                compute_step = (last_computed_step - t) >= cache_rate
            
            if compute_step:
                # Compute model prediction with CFG
                uncond_conditions = {k: torch.zeros_like(v) for k, v in conditions.items()}
                eps_uncond = model(x, t_embed, uncond_conditions)
                eps_cond = model(x, t_embed, conditions)
                
                # Apply CFG scaling
                eps = eps_uncond
                for cond_type, scale in cfg_scales.items():
                    eps = eps + scale * (eps_cond - eps_uncond)
                
                # Cache the computed values
                cached_residuals = {
                    'eps': eps,
                    'rep': model.get_intermediate_rep(x, t_embed, conditions)
                }
                last_computed_step = t
            else:
                # Reuse cached values
                eps = cached_residuals['eps']
            
            # DDIM update step
            x0_pred = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)
            
            # Optional stochastic component
            sigma = eta * torch.sqrt((1 - at_next) / (1 - at)) * torch.sqrt(1 - at / at_next)
            noise = torch.randn_like(x) if eta > 0 else 0
            
            # Compute x_(t-1)
            x_prev = torch.sqrt(at_next) * x0_pred + \
                    torch.sqrt(1 - at_next - sigma**2) * eps + \
                    sigma * noise
            
            x = x_prev
            
        return x
    

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




    
import logging


from model import Gbase, Emtn, WarpGeneratorS2C, WarpGeneratorC2D

class VASAFaceEncoder(Gbase):
    """
    VASA Face Encoder that extends MegaPortraits' Gbase with specific VASA functionality.
    Implements the face encoding stage described in VASA paper section 3.1.
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        # Initialize components from Gbase (MegaPortraits)
        # Called via super().__init__() above

        # Add VASA-specific encoding stages
        self.feature_dim = feature_dim
        self.gaze_encoder = self._create_gaze_encoder()
        self.emotion_encoder = self._create_emotion_encoder()

    def _create_gaze_encoder(self):
        """Creates the gaze direction encoder network"""
        return nn.Sequential(
            nn.Linear(2, 128),  # Takes (θ,φ) angles
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

    def _create_emotion_encoder(self):
        """Creates the emotion encoding network"""
        return nn.Sequential(
            nn.Linear(8, 128),  # 8 emotion categories
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

    def encode_holistic(self, x, gaze=None, emotion=None):
        """
        Encode complete facial representation including appearance, motion, and control signals
        Args:
            x: Input image tensor
            gaze: Optional gaze direction tensor (θ,φ)
            emotion: Optional emotion tensor (8 categories)
        Returns:
            Dictionary containing:
            - appearance_volume: 3D appearance features
            - identity: Identity embedding
            - head_pose: Head pose parameters
            - facial_dynamics: Facial dynamics embedding
            - gaze_features: Optional gaze features
            - emotion_features: Optional emotion features
        """
        # Get base features from MegaPortraits Gbase
        vs, es = self.appearanceEncoder(x)
        Rs, ts, zs = self.motionEncoder(x)

        # Combine into facial dynamics representation
        facial_dynamics = self.combine_dynamics(zs, gaze, emotion)

        # Generate warping fields
        w_s2c = self.warp_generator_s2c(Rs, ts, zs, es)

        # Create canonical volume
        vc = self.apply_warping_field(vs, w_s2c)

        return {
            'appearance_volume': vc,
            'identity': es,
            'head_pose': (Rs, ts),
            'facial_dynamics': facial_dynamics,
            'gaze_features': self.gaze_encoder(gaze) if gaze is not None else None,
            'emotion_features': self.emotion_encoder(emotion) if emotion is not None else None
        }

    def combine_dynamics(self, base_dynamics, gaze=None, emotion=None):
        """Combines base dynamics with optional control signals"""
        features = [base_dynamics]
        
        if gaze is not None:
            gaze_features = self.gaze_encoder(gaze)
            features.append(gaze_features)
            
        if emotion is not None:
            emotion_features = self.emotion_encoder(emotion)
            features.append(emotion_features)
            
        # Combine all features
        combined = torch.cat(features, dim=-1)
        return combined

    def apply_warping_field(self, volume, warp_field):
        """Apply 3D warping field to volume features"""
        return super().apply_warping_field(volume, warp_field)


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





class IdentityLoss(nn.Module):
    """
    Identity preservation loss using pretrained face recognition model.
    Ensures generated faces maintain the identity of the source image.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Initialize with pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add identity projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.register_buffer('center', torch.zeros(256))
        self.register_buffer('std', torch.ones(256))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract identity features from face images"""
        features = self.backbone(x)
        features = self.projection(features)
        # Normalize features
        features = (features - self.center) / self.std
        return F.normalize(features, p=2, dim=1)

    def forward(self, generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss between generated and source images
        Args:
            generated: Generated face images [B, C, H, W]
            source: Source face images [B, C, H, W]
        Returns:
            Identity loss value
        """
        gen_features = self.extract_features(generated)
        src_features = self.extract_features(source)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(gen_features, src_features, dim=1)
        identity_loss = 1.0 - cos_sim.mean()
        
        return identity_loss

class PoseExtractionNet(nn.Module):
    """
    Network for extracting head pose parameters from face images.
    """
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Modify final layer for pose parameters (rotation + translation)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 6)  # 3 for rotation, 3 for translation
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose parameters
        Returns:
            rotation: Rotation parameters [B, 3]
            translation: Translation parameters [B, 3]
        """
        pose_params = self.backbone(x)
        rotation = pose_params[:, :3]
        translation = pose_params[:, 3:]
        return rotation, translation

class ExpressionExtractionNet(nn.Module):
    """
    Network for extracting facial expression parameters.
    """
    def __init__(self, expression_dim: int = 64):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, expression_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract expression parameters"""
        return self.backbone(x)

class DPELoss(nn.Module):
    """
    Disentanglement of Pose and Expression (DPE) loss.
    Ensures effective disentanglement between pose and facial expressions.
    """
    def __init__(self, 
                 expression_dim: int = 64,
                 lambda_pose: float = 1.0,
                 lambda_expr: float = 1.0):
        super().__init__()
        self.pose_net = PoseExtractionNet()
        self.expression_net = ExpressionExtractionNet(expression_dim)
        
        # Loss weights
        self.lambda_pose = lambda_pose
        self.lambda_expr = lambda_expr
        
        # Feature reconstruction loss
        self.recon_loss = nn.MSELoss()
        
        # Freeze networks
        for param in self.pose_net.parameters():
            param.requires_grad = False
        for param in self.expression_net.parameters():
            param.requires_grad = False

    def compute_pose_consistency(self, 
                               I_i: torch.Tensor,
                               I_j: torch.Tensor,
                               I_i_pose_j: torch.Tensor) -> torch.Tensor:
        """Compute pose consistency loss"""
        # Extract poses
        rot_i, trans_i = self.pose_net(I_i)
        rot_j, trans_j = self.pose_net(I_j)
        rot_transferred, trans_transferred = self.pose_net(I_i_pose_j)
        
        # Pose should match target
        pose_loss = (
            F.mse_loss(rot_transferred, rot_j) +
            F.mse_loss(trans_transferred, trans_j)
        )
        return pose_loss

    def compute_expression_consistency(self,
                                    I_i: torch.Tensor,
                                    I_j: torch.Tensor,
                                    I_i_pose_j: torch.Tensor) -> torch.Tensor:
        """Compute expression consistency loss"""
        # Extract expressions
        expr_i = self.expression_net(I_i)
        expr_j = self.expression_net(I_j)
        expr_transferred = self.expression_net(I_i_pose_j)
        
        # Expression should remain the same after pose transfer
        expr_loss = F.mse_loss(expr_transferred, expr_i)
        return expr_loss

    def forward(self, 
                I_i: torch.Tensor,
                I_j: torch.Tensor,
                I_i_pose_j: torch.Tensor,
                I_j_pose_i: torch.Tensor,
                I_s: torch.Tensor,
                I_d: torch.Tensor,
                I_s_pose_d_dyn_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DPE loss components
        Args:
            I_i, I_j: Source frames from same identity
            I_i_pose_j: I_i with I_j's pose
            I_j_pose_i: I_j with I_i's pose
            I_s: Source identity frame
            I_d: Different identity frame
            I_s_pose_d_dyn_d: Source frame with different identity's pose and dynamics
        """
        losses = {}
        
        # Pose consistency loss
        losses['pose_i'] = self.compute_pose_consistency(I_i, I_j, I_i_pose_j)
        losses['pose_j'] = self.compute_pose_consistency(I_j, I_i, I_j_pose_i)
        
        # Expression consistency loss
        losses['expr_i'] = self.compute_expression_consistency(I_i, I_j, I_i_pose_j)
        losses['expr_j'] = self.compute_expression_consistency(I_j, I_i, I_j_pose_i)
        
        # Cross-identity pose transfer loss
        losses['cross_pose'] = self.compute_pose_consistency(I_s, I_d, I_s_pose_d_dyn_d)
        
        # Cross-identity expression preservation
        losses['cross_expr'] = self.compute_expression_consistency(I_s, I_d, I_s_pose_d_dyn_d)
        
        # Total loss
        losses['total'] = (
            self.lambda_pose * (losses['pose_i'] + losses['pose_j'] + losses['cross_pose']) +
            self.lambda_expr * (losses['expr_i'] + losses['expr_j'] + losses['cross_expr'])
        )
        
        return losses

class CombinedVASALoss(nn.Module):
    """
    Combined loss function for VASA training
    """
    def __init__(self,
                 lambda_identity: float = 0.1,
                 lambda_dpe: float = 0.1):
        super().__init__()
        self.identity_loss = IdentityLoss()
        self.dpe_loss = DPELoss()
        
        self.lambda_identity = lambda_identity
        self.lambda_dpe = lambda_dpe

    def forward(self,
                generated: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        Args:
            generated: Dict containing generated images and intermediate results
            target: Dict containing ground truth images and attributes
        """
        losses = {}
        
        # Identity preservation loss
        losses['identity'] = self.identity_loss(
            generated['output'],
            target['source_image']
        )
        
        # DPE losses
        dpe_losses = self.dpe_loss(
            generated['source'], generated['target'],
            generated['source_pose_transfer'], generated['target_pose_transfer'],
            generated['source_identity'], generated['target_identity'],
            generated['cross_identity_transfer']
        )
        losses.update({f'dpe_{k}': v for k, v in dpe_losses.items()})
        
        # Total loss
        losses['total'] = (
            losses['identity'] * self.lambda_identity +
            dpe_losses['total'] * self.lambda_dpe
        )
        
        return losses

def test_losses():
    """Test loss computations"""
    batch_size = 4
    img_size = 256
    
    # Create dummy data
    dummy_data = {
        'source': torch.randn(batch_size, 3, img_size, img_size),
        'target': torch.randn(batch_size, 3, img_size, img_size),
        'source_pose_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'target_pose_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'source_identity': torch.randn(batch_size, 3, img_size, img_size),
        'target_identity': torch.randn(batch_size, 3, img_size, img_size),
        'cross_identity_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'output': torch.randn(batch_size, 3, img_size, img_size)
    }
    
    target_data = {
        'source_image': torch.randn(batch_size, 3, img_size, img_size)
    }
    
    # Test loss computation
    loss_fn = CombinedVASALoss()
    losses = loss_fn(dummy_data, target_data)
    
    print("Loss components:")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")



class VASALossModule:
    """Loss module for VASA training"""
    def __init__(self, config: VASAConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize loss components
        self.identity_loss = IdentityLoss().to(device)
        self.dpe_loss = DPELoss(
            expression_dim=config.motion_dim,
            lambda_pose=config.lambda_pose,
            lambda_expr=config.lambda_expr
        ).to(device)
        self.combined_loss = CombinedVASALoss(
            lambda_identity=config.lambda_identity,
            lambda_dpe=config.lambda_dpe
        ).to(device)

    def compute_losses(
        self,
        generated_frames: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        face_components: Dict[str, torch.Tensor],
        diffusion_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses
        Args:
            generated_frames: Generated video frames
            batch: Training batch data
            face_components: Face encoder outputs
            diffusion_output: Diffusion model outputs
        """
        # Prepare inputs for loss computation
        loss_inputs = {
            'source': batch['frames'][:, 0],  # First frame is source
            'target': batch['frames'][:, 1:],  # Remaining frames are targets
            'source_pose_transfer': generated_frames[:, 0],
            'target_pose_transfer': generated_frames[:, 1:],
            'source_identity': face_components['identity'],
            'target_identity': face_components['identity'],
            'cross_identity_transfer': generated_frames,
            'output': generated_frames
        }
        
        # Ground truth data
        target_data = {
            'source_image': batch['frames'][:, 0]
        }
        
        # Compute main losses
        main_losses = self.combined_loss(loss_inputs, target_data)
        
        # Add additional losses
        losses = {
            'reconstruction': F.l1_loss(generated_frames, batch['frames']),
            'identity': main_losses['identity'],
            'dpe_total': main_losses['dpe_total']
        }
        
        # Add individual DPE losses for monitoring
        losses.update({
            f'dpe_{k}': v for k, v in main_losses.items() 
            if k.startswith('dpe_') and k != 'dpe_total'
        })
        
        # Add CFG losses
        if 'uncond' in diffusion_output:
            losses.update(self._compute_cfg_losses(diffusion_output))
        
        # Compute weighted total loss
        losses['total'] = (
            self.config.lambda_recon * losses['reconstruction'] +
            self.config.lambda_identity * losses['identity'] +
            self.config.lambda_dpe * losses['dpe_total'] +
            sum(self.config.lambda_cfg * losses[f'cfg_{k}'] 
                for k in ['audio', 'gaze'] 
                if f'cfg_{k}' in losses)
        )
        
        return losses

    def _compute_cfg_losses(
        self,
        diffusion_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute classifier-free guidance losses"""
        cfg_losses = {}
        
        # Base unconditional output
        uncond_output = diffusion_output['uncond']
        
        # Audio CFG loss
        if 'masked_audio' in diffusion_output:
            cfg_losses['cfg_audio'] = F.mse_loss(
                diffusion_output['masked_audio'],
                uncond_output
            )
        
        # Gaze CFG loss
        if 'masked_gaze' in diffusion_output:
            cfg_losses['cfg_gaze'] = F.mse_loss(
                diffusion_output['masked_gaze'],
                uncond_output
            )
            
        return cfg_losses
        



    
class DiffusionSampler:
    """
    Implements sampling strategies for the diffusion model
    """
    def __init__(self, 
                 num_steps: int = 50,
                 min_beta: float = 1e-4,
                 max_beta: float = 0.02):
        self.num_steps = num_steps
        
        # Set up diffusion parameters
        self.beta = torch.linspace(min_beta, max_beta, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Pre-compute sampling parameters
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alpha)
        self.log_one_minus_alpha = torch.log(1.0 - self.alpha)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.posterior_variance = self.beta * (1.0 - self.alpha_bar.previous_frame) / (1.0 - self.alpha_bar)
        
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps uniformly"""
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    @torch.no_grad()
    def ddim_sample(self,
                    model: nn.Module,
                    shape: Tuple[int, ...],
                    conditions: Dict[str, torch.Tensor],
                    cfg_scales: Dict[str, float],
                    eta: float = 0.0,
                    device: torch.device = None) -> torch.Tensor:
        """
        Sample using DDIM for faster inference
        Args:
            model: Diffusion model
            shape: Output tensor shape
            conditions: Conditioning signals
            cfg_scales: Classifier-free guidance scales
            eta: DDIM stochastic sampling parameter (0 = deterministic)
        """
        device = device or next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Setup progress bar
        pbar = tqdm(reversed(range(self.num_steps)), desc='DDIM Sampling')
        
        for t in pbar:
            # Get diffusion parameters for current timestep
            at = self.alpha_bar[t]
            at_next = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
            
            # Time embedding
            t_embed = torch.ones(batch_size, device=device) * t
            
            # Model prediction with classifier-free guidance
            with torch.no_grad():
                # Get unconditional prediction
                uncond_conditions = {k: torch.zeros_like(v) for k, v in conditions.items()}
                eps_uncond = model(x, t_embed, uncond_conditions)
                
                # Get conditional prediction
                eps_cond = model(x, t_embed, conditions)
                
                # Apply CFG scaling
                eps = eps_uncond
                for cond_type, scale in cfg_scales.items():
                    eps = eps + scale * (eps_cond - eps_uncond)
            
            # DDIM update step
            x0_pred = (x - torch.sqrt(1 - at) * eps) / torch.sqrt(at)
            
            # Optional stochastic component
            sigma = eta * torch.sqrt((1 - at_next) / (1 - at)) * torch.sqrt(1 - at / at_next)
            noise = torch.randn_like(x) if eta > 0 else 0
            
            # Compute x_(t-1)
            x_prev = torch.sqrt(at_next) * x0_pred + \
                    torch.sqrt(1 - at_next - sigma**2) * eps + \
                    sigma * noise
            
            x = x_prev
            
        return x

class MotionGenerator:
    """
    Generates motion sequences using the diffusion model
    """
    def __init__(self,
                 model: nn.Module,
                 sampler: DiffusionSampler,
                 window_size: int = 25,
                 stride: int = 20):
        self.model = model
        self.sampler = sampler
        self.window_size = window_size
        self.stride = stride  # Stride between windows
        
    def generate_motion_sequence(self,
                               audio_features: torch.Tensor,
                               conditions: Dict[str, torch.Tensor],
                               cfg_scales: Dict[str, float],
                               device: torch.device) -> torch.Tensor:
        """
        Generate motion sequence using sliding windows
        Args:
            audio_features: Audio features [1, T, C]
            conditions: Dictionary of conditioning signals
            cfg_scales: Dictionary of CFG scales
        Returns:
            Generated motion sequence [1, T, motion_dim]
        """
        seq_length = audio_features.shape[1]
        motion_dim = self.model.motion_dim
        generated_motions = []
        
        # Initialize overlap buffer
        prev_window = None
        overlap_size = self.window_size - self.stride
        
        # Generate motions window by window
        for start_idx in range(0, seq_length, self.stride):
            end_idx = min(start_idx + self.window_size, seq_length)
            current_window_size = end_idx - start_idx
            
            # Get current window conditions
            window_conditions = {
                k: v[:, start_idx:end_idx] if len(v.shape) > 2 else v
                for k, v in conditions.items()
            }
            
            # Add previous window context if available
            if prev_window is not None:
                window_conditions['prev_motion'] = prev_window[:, -overlap_size:]
            
            # Generate motion for current window
            window_shape = (1, current_window_size, motion_dim)
            current_motion = self.sampler.ddim_sample(
                self.model,
                window_shape,
                window_conditions,
                cfg_scales,
                device=device
            )
            
            # Smooth transition in overlap region
            if prev_window is not None and overlap_size > 0:
                weights = torch.linspace(0, 1, overlap_size, device=device)
                weights = weights.view(1, -1, 1)
                
                overlap_region = weights * current_motion[:, :overlap_size] + \
                               (1 - weights) * prev_window[:, -overlap_size:]
                
                current_motion = torch.cat([
                    overlap_region,
                    current_motion[:, overlap_size:]
                ], dim=1)
            
            generated_motions.append(current_motion)
            prev_window = current_motion
        
        # Concatenate all windows
        full_sequence = torch.cat(generated_motions, dim=1)
        
        # Trim to exact sequence length if needed
        if full_sequence.shape[1] > seq_length:
            full_sequence = full_sequence[:, :seq_length]
            
        return full_sequence

class VideoGenerator:
    """
    Complete video generation pipeline
    """
    def __init__(self,
                 face_encoder: nn.Module,
                 motion_generator: MotionGenerator,
                 face_decoder: nn.Module,
                 device: torch.device):
        self.face_encoder = face_encoder
        self.motion_generator = motion_generator
        self.face_decoder = face_decoder
        self.device = device
        
    @torch.no_grad()
    def generate_video(self,
                      source_image: torch.Tensor,
                      audio_features: torch.Tensor,
                      conditions: Dict[str, torch.Tensor],
                      cfg_scales: Dict[str, float],
                      output_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Generate complete talking face video
        Args:
            source_image: Source face image [1, C, H, W]
            audio_features: Audio features [1, T, C]
            conditions: Dictionary of conditioning signals
            cfg_scales: Dictionary of CFG scales
            output_size: Size of output video frames
        Returns:
            Generated video frames [1, T, C, H, W]
        """
        # Extract source image components
        source_components = self.face_encoder(source_image)
        
        # Generate motion sequence
        motion_sequence = self.motion_generator.generate_motion_sequence(
            audio_features,
            conditions,
            cfg_scales,
            self.device
        )
        
        # Generate frames
        num_frames = motion_sequence.shape[1]
        frames = []
        
        for t in tqdm(range(num_frames), desc='Generating frames'):
            # Get current motion and conditions
            current_motion = motion_sequence[:, t]
            current_conditions = {
                k: v[:, t] if len(v.shape) > 2 else v
                for k, v in conditions.items()
            }
            
            # Generate frame
            frame = self.face_decoder(
                source_components['appearance_volume'],
                source_components['identity'],
                current_motion,
                current_conditions
            )
            
            # Resize if needed
            if frame.shape[-2:] != output_size:
                frame = F.interpolate(
                    frame,
                    size=output_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            frames.append(frame)
        
        # Stack frames into video
        video = torch.stack(frames, dim=1)  # [1, T, C, H, W]
        
        return video

class VideoPostProcessor:
    """
    Post-processing for generated videos
    """
    def __init__(self):
        pass
    
    @torch.no_grad()
    def apply_temporal_smoothing(self,
                               video: torch.Tensor,
                               window_size: int = 5) -> torch.Tensor:
        """Apply temporal smoothing to reduce jitter"""
        kernel = torch.ones(1, 1, window_size, 1, 1, device=video.device) / window_size
        smoothed = F.conv3d(
            video,
            kernel,
            padding=(window_size // 2, 0, 0)
        )
        return smoothed
    
    def enhance_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Enhance individual frames (if needed)"""
        return video  # Implement frame enhancement if needed
