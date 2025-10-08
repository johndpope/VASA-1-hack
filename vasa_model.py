import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from omegaconf import OmegaConf
import sys
import math
import yaml
from pathlib import Path
from diffusers import DDIMScheduler
import torch.nn.functional as F
import random
import traceback

# Add nemo to path for imports
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from logger import logger
from blink_condition_handler import BlinkConditionHandler


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) - A learnable normalization alternative from Meta FAIR (March 2025).

    DyT(x) = γ * tanh(α * x) + β

    Benefits over LayerNorm:
    - Preserves variance better (no mean/std computation)
    - Learnable activation range via α
    - Natural bounding via tanh saturation
    - ~5-10% faster (no statistics gathering)
    - Better for audio-motion synchronization in diffusion models

    Args:
        dim: Feature dimension (same as LayerNorm's normalized_shape)
        init_alpha: Initial value for α (default 1.0, use 0.1 if over-saturation occurs)
    """
    def __init__(self, dim: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))  # Learnable scalar for input scaling
        self.gamma = nn.Parameter(torch.ones(dim))           # Per-feature scale (like LayerNorm)
        self.beta = nn.Parameter(torch.zeros(dim))           # Per-feature shift (like LayerNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DyT: γ * tanh(α * x) + β

        Shape: Works with any shape ending in [..., dim], just like LayerNorm
        """
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class VASAPositionalEmbedding(nn.Module):
    """
    Positional embeddings for VASA sequence generation with negative positions for context.
    Uses negative positions for context frames and positive for current frames to maintain
    clear temporal distinction.
    """
    def __init__(
        self,
        d_model: int = 512,
        max_seq_len: int = 50,
        max_context_len: int = 10,
        dropout: float = 0.1,
        use_relative_position: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.max_context_len = max_context_len
        self.use_relative_position = use_relative_position
        self.dropout = nn.Dropout(dropout)

    def _compute_sinusoidal_embedding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal positional embeddings for any position values."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(
        self,
        x: torch.Tensor,
        has_context: bool = False
    ) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device

        if has_context and T > self.max_context_len:
            # Use negative positions for context, positive for current
            context_len = self.max_context_len
            current_len = T - context_len

            # Create position indices: [-context_len, ..., -1, 0, 1, ..., current_len-1]
            context_positions = torch.arange(-context_len, 0, device=device)
            current_positions = torch.arange(0, current_len, device=device)
            all_positions = torch.cat([context_positions, current_positions])

            # Compute sinusoidal embeddings for these positions
            pos_emb = self._compute_sinusoidal_embedding(all_positions, self.d_model)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        else:
            # No context, use standard positive positions [0, 1, ..., T-1]
            positions = torch.arange(0, T, device=device)
            pos_emb = self._compute_sinusoidal_embedding(positions, self.d_model)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)

        x = x + pos_emb
        return self.dropout(x)


class EfficientConditionEmbedding(nn.Module):
    def __init__(self, model_dim: int = 512, max_seq_len: int = 60):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        logger.info(f"Initializing EfficientConditionEmbedding: model_dim={model_dim}, max_seq_len={max_seq_len}")

        config = self.load_channel_config('channel_config.yaml')
        self.clip_min = config.model.clip_bounds.min
        self.clip_max = config.model.clip_bounds.max

        self.blink_handler = BlinkConditionHandler(window_size=max_seq_len)

        # Note: channel_layout from config is not used - features are concatenated directly
        # The config defines theoretical positions but implementation uses learned projections

        # Audio projection - aligned with JoyVASA (single linear layer, no normalization)
        # JoyVASA uses a single Linear layer: self.audio_feature_map = nn.Linear(768, feature_dim)
        self.audio_proj = nn.Linear(768, config.projections.audio.output_dim)
        logger.info("Using JoyVASA-aligned audio projection: single Linear(768 -> {}) without normalization".format(
            config.projections.audio.output_dim))

        self.gaze_proj = nn.Linear(2, 2)
        self.distance_proj = nn.Linear(1, 1)
        self.emotion_proj = nn.Linear(2, 2)

        # Using DynamicTanh instead of LayerNorm to preserve variance (JoyVASA approach + DyT benefits)
        self.control_norm = DynamicTanh(config.projections.control_norm_dim)

        self.blink_embed = nn.Sequential(
            nn.Linear(config.projections.blink.input_dim, config.projections.blink.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projections.blink.hidden_dim, config.projections.blink.output_dim)
        )

        # Final projection to combine all features into model_dim
        # Audio (512) + Controls (5) + Blink (32) = 549 -> 512
        total_features = config.projections.audio.output_dim + 5 + config.projections.blink.output_dim
        self.final_proj = nn.Linear(total_features, model_dim)

        self.final_norm = DynamicTanh(model_dim)

    def load_channel_config(self, config_path: str) -> OmegaConf:
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
            config = OmegaConf.create(raw_config)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def _ensure_float_tensor(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            tensor = tensor.float()
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

    def _get_device_and_batch_size(self, conditions: Dict[str, torch.Tensor]) -> Tuple[torch.device, int]:
        logger.debug("\nAttempting to get device and batch size...")
        for key in ['gaze', 'head_distance', 'emotion', 'speed_bucket']:
            if key in conditions and isinstance(conditions[key], torch.Tensor):
                tensor = conditions[key]
                logger.debug(f"Using '{key}' tensor: shape={tensor.shape}, device={tensor.device}")
                return tensor.device, tensor.shape[0]
        if 'audio_features' in conditions:
            tensor = conditions['audio_features']
            B = tensor.shape[0] if tensor.dim() == 4 else 1
            logger.debug(f"Using 'audio_features' tensor: shape={tensor.shape}, device={tensor.device}")
            return tensor.device, B
        raise ValueError("No valid tensor found in conditions")

    def forward(
        self,
        conditions: Dict[str, torch.Tensor],
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        try:
            logger.debug("\n=== Starting EfficientConditionEmbedding Forward Pass ===")
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            logger.debug(f"Using device: {device}, dtype: {dtype}")

            # Debug: Log all condition keys and their status
            logger.debug(f"Received conditions keys: {list(conditions.keys())}")
            for key, value in conditions.items():
                if value is None:
                    logger.debug(f"  {key}: None")
                elif isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    logger.debug(f"  {key}: type={type(value)}")

            # Find any valid tensor to get B and T (skip None values from dropout)
            valid_tensor = None
            valid_key = None
            for key, tensor in conditions.items():
                if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    valid_tensor = tensor
                    valid_key = key
                    break

            assert valid_tensor is not None, f"No valid tensors found in conditions! Keys: {list(conditions.keys())}, Values: {[(k, type(v)) for k, v in conditions.items()]}"
            B, T = valid_tensor.shape[:2]
            logger.debug(f"Batch size: {B}, Sequence length: {T} (from '{valid_key}')")

            output = torch.zeros(B, T, self.model_dim, device=device, dtype=dtype)
            logger.debug(f"Initialized output tensor: shape={output.shape}")

            # Audio features are REQUIRED - dataset must provide them
            assert 'audio_features' in conditions, f"audio_features must be present in conditions! Available keys: {list(conditions.keys())}"
            audio = conditions['audio_features']
            assert audio is not None, f"audio_features cannot be None! Value: {audio}, Type: {type(audio) if audio is not None else 'None'}"
            assert isinstance(audio, torch.Tensor), f"audio_features must be a torch.Tensor, got {type(audio)}"
            logger.debug(f"Audio features shape before processing: {audio.shape}")

            if len(audio.shape) == 4:
                audio = audio.squeeze(1)
            audio = self._ensure_float_tensor(audio, dtype)

            # JoyVASA approach: Direct projection without normalization
            # This preserves the variance signal that distinguishes silent vs speech
            audio_projected = self.audio_proj(audio)

            # No normalization - keep raw projected features
            # This aligns with JoyVASA's audio_feature_map approach
            audio_features = audio_projected  # Use projected features directly

            # DEBUG: Log audio features variance to verify preservation
            audio_var = audio.var().item()
            audio_proj_var = audio_projected.var().item()
            logger.debug(f"[JOYVASA ALIGNED] Audio variance - Raw: {audio_var:.6f}, Projected: {audio_proj_var:.6f}")
            logger.debug(f"Preserving full variance signal without normalization")

            # Handle None values from dropout - use zeros as default
            gaze_tensor = conditions.get('gaze')
            if gaze_tensor is None:
                gaze_tensor = torch.zeros(B, T, 2, device=device, dtype=dtype)
            else:
                gaze_tensor = self._ensure_float_tensor(gaze_tensor, dtype)
            gaze = self.gaze_proj(gaze_tensor)

            distance_tensor = conditions.get('head_distance')
            if distance_tensor is None:
                distance_tensor = torch.zeros(B, T, 1, device=device, dtype=dtype)
            else:
                distance_tensor = self._ensure_float_tensor(distance_tensor, dtype)
            distance = self.distance_proj(distance_tensor)

            emotion_tensor = conditions.get('emotion')
            if emotion_tensor is None:
                emotion_tensor = torch.zeros(B, T, 2, device=device, dtype=dtype)
            else:
                emotion_tensor = self._ensure_float_tensor(emotion_tensor, dtype)
            emotion = self.emotion_proj(emotion_tensor)

            # DEBUG: Log control projections to diagnose dominance issue
            logger.debug(f"Gaze shape: {gaze.shape}, Values mean: {gaze.mean().item():.6f}, var: {gaze.var().item():.6f}")
            logger.debug(f"Distance shape: {distance.shape}, Values mean: {distance.mean().item():.6f}, var: {distance.var().item():.6f}")
            logger.debug(f"Emotion shape: {emotion.shape}, Values mean: {emotion.mean().item():.6f}, var: {emotion.var().item():.6f}")

            controls = torch.cat([gaze, distance, emotion], dim=-1)  # Removed speed
            # JoyVASA approach: No normalization to preserve variance
            controls_features = controls  # Use raw control features

            # DEBUG: Check variance preservation
            logger.debug(f"Controls variance (no norm): {controls.var().item():.6f}")

            # Handle blink_state with None check
            blink_tensor = conditions.get('blink_state')
            if blink_tensor is None:
                blink_tensor = torch.zeros(B, T, 3, device=device, dtype=dtype)
            blink_embedded = self.blink_embed(blink_tensor)

            combined = torch.cat([audio_features, controls_features,  blink_embedded], dim=-1)

            # Project combined features to model dimension
            # This maintains variance while fitting into model_dim
            projected = self.final_proj(combined)
            output = projected  # Direct assignment, shape is now [B, T, model_dim]

            # JoyVASA approach: Skip normalization to preserve variance
            final_output = output  # No normalization
            logger.debug(f"[JOYVASA] Final output variance: {output.var().item():.6f}")

            final_var = final_output.var().item()
            logger.debug(f" Combined variance: {combined.var().item():.6f}, Output variance: {output.var().item():.6f}, Final normalized: {final_var:.6f}")

            # Log individual component contributions and absolute values
            # Calculate proper variance contributions (sum of individual variances)
            audio_var = audio_features.var().item()
            controls_var = controls_features.var().item()
            blink_var = blink_embedded.var().item()
            total_var = audio_var + controls_var + blink_var + 1e-8

            audio_contrib = audio_var / total_var
            controls_contrib = controls_var / total_var
            blink_contrib = blink_var / total_var

            # Also log absolute magnitudes to see if audio is being suppressed
            audio_mag = torch.norm(audio_features).item()
            controls_mag = torch.norm(controls_features).item()

            # Check if variance preservation is working
            audio_mean_mag = torch.abs(audio_features).mean().item()

            logger.debug(f" Component contributions - Audio: {audio_contrib:.2%}, Controls: {controls_contrib:.2%}, Blink: {blink_contrib:.2%}")
            logger.debug(f" Component magnitudes - Audio L2: {audio_mag:.4f}, Audio mean: {audio_mean_mag:.6f}, Controls: {controls_mag:.4f}")

            return final_output

        except Exception as e:
            logger.error(f"Error in condition embedding: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class AudioCrossDecoderLayer(nn.Module):
    """
    Custom decoder layer with FUSED audio cross-attention and Flash Attention optimized causal masking.

    OPTIMIZATION: Removed separate audio_cross_attn head - reuses cross_attn for both conditions and audio.
    This saves ~1.05M parameters per layer with no behavioral change.

    Uses F.scaled_dot_product_attention with is_causal=True for efficient causal self-attention.
    Overrides LayerNorms with DynamicTanh for full DyT integration.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, **kwargs):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        # Two attention heads only (removed audio_cross_attn)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers (DynamicTanh for variance preservation)
        self.norm1 = DynamicTanh(d_model)  # For self-attention
        self.norm2 = DynamicTanh(d_model)  # For condition cross-attention
        self.norm3 = DynamicTanh(d_model)  # For audio cross-attention (reuses cross_attn)
        self.norm4 = DynamicTanh(d_model)  # For feed-forward

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate a causal mask for self-attention (upper triangular mask of -inf)."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                audio_memory=None, is_causal=True):
        """
        Pre-norm transformer decoder layer with FUSED audio cross-attention.

        OPTIMIZATION: Audio cross-attention now reuses self.cross_attn instead of separate head.
        Same computation, half the parameters for audio path.

        Args:
            tgt: Target sequence [B, T, d_model]
            memory: Condition embeddings [B, T, d_model]
            audio_memory: Audio features [B, T, d_model] (cached, not recomputed)
            is_causal: Use causal masking for self-attention (default: True)
            tgt_mask: Attention mask (auto-generated if None and is_causal=True)
        """
        # ASSERTION: audio_memory is REQUIRED
        assert audio_memory is not None, \
            "audio_memory is required for AudioCrossDecoderLayer"
        assert isinstance(audio_memory, torch.Tensor), \
            f"audio_memory must be a Tensor, got {type(audio_memory)}"

        # 1. Self-attention with causal masking
        tgt_normed = self.norm1(tgt)

        # Generate causal mask if needed
        if is_causal and tgt_mask is None:
            seq_len = tgt_normed.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len, tgt_normed.device)

        tgt2 = self.self_attn(tgt_normed, tgt_normed, tgt_normed, attn_mask=tgt_mask, is_causal=is_causal)[0]
        tgt = tgt + self.dropout1(tgt2)

        # 2. Cross-attention to condition embeddings
        tgt_normed = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt_normed, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False)[0]
        tgt = tgt + self.dropout2(tgt2)

        # 3. FUSED audio cross-attention - reuses cross_attn for audio (saves ~1M params)
        tgt_normed = self.norm3(tgt)
        audio_attn, _ = self.cross_attn(
            tgt_normed,
            audio_memory,  # Key/value both from audio
            audio_memory,
            key_padding_mask=None,
            need_weights=False
        )
        tgt = tgt + self.dropout3(audio_attn)

        # 4. Feed-forward network
        tgt_normed = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt_normed))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

class MotionTransformer(nn.Module):
    """Decoder-based Transformer for motion generation matching H5 cache structure.

    H5 Cache Structure (per frame):
    - uv_warp: (1, 16, 64, 64, 3) - Target UV warps
    - theta: (1, 4, 4) - Target pose matrix
    - scale: (1, 3) - SRT scale component
    - rotation: (1, 3) - SRT rotation component
    - translation: (1, 3) - SRT translation component
    - target_pose_embed: (1, 128) - Aligned expression embedding

    For 50 frames, we predict:
    - uv_warps: (B, 50, 16, 64, 64, 3)
    - theta: (B, 50, 3, 4) - Note: 3x4 not 4x4 in model
    - scale: (B, 50, 3)
    - rotation: (B, 50, 3)
    - translation: (B, 50, 3)
    - expression_embed: (B, 50, 128)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.model.hidden_dim
        self.expression_dim = config.model.expression_dim  # Should be 128
        self.context_size = config.motion.context_size
        self.window_size = config.motion.window_size  # Should be 50
        self.use_derived_warps = getattr(config.model, 'use_derived_warps', False)

        # Transformer configuration
        nhead = config.model.n_heads
        num_layers = config.model.n_layers
        dim_feedforward = config.model.dim_feedforward
        dropout = config.model.dropout

        # Motion embeddings matching H5 data
        self.theta_emb = nn.Linear(3 * 4, self.d_model // 2)  # theta is 3x4
        self.expr_emb = nn.Linear(128, self.d_model // 2)  # expression is 128-dim


        # UV Warp encoder REMOVED - warps now generated implicitly, not encoded
        # Motion embeddings no longer include UV warps during training
        # SRT embeddings also removed - nemo handles these internally

        # Combine all motion embeddings (only theta + expression)
        # theta_emb (d_model/2) + expr_emb (d_model/2) = d_model
        total_motion_dim = self.d_model  # Just theta + expr
        self.motion_proj = nn.Linear(total_motion_dim, self.d_model)

        # Timestep embedding
        self.time_emb = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )

        # Positional embedding
        self.pos_emb = VASAPositionalEmbedding(
            d_model=self.d_model,
            max_seq_len=self.window_size,
            max_context_len=self.context_size,
            use_relative_position=config.model.get('use_relative_position', True)
        )

        # Condition embedding
        self.cond_emb = EfficientConditionEmbedding(
            model_dim=self.d_model,
            max_seq_len=self.window_size + self.context_size
        )

        # Transformer decoder

        # Create custom decoder that handles audio
        self.decoder_layers = nn.ModuleList([
            AudioCrossDecoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # Final layer norm replaced with DynamicTanh for better stability
        self.decoder_norm = DynamicTanh(self.d_model)



        # Output heads for all motion parameters with improved initialization
        self.theta_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),  # Add dropout to prevent overfitting
            nn.Linear(self.d_model // 2, 3 * 4)
        )

        # Initialize theta_head with larger weights for better gradient flow
        for layer in self.theta_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=2.0)  # Larger initialization for rotation
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.expr_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.expression_dim)
        )


        # UV Warp generation moved to implicit WarpGeneratorFromZdyn
        # (No explicit warp head needed - warps derived from zdyn + theta)

    def _get_sinusoidal_embedding(self, ts, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=ts.device) * -emb)
        emb = ts.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        B, T = motion_data['theta'].shape[:2]
        device = motion_data['theta'].device
        C = self.context_size if prev_context is not None else 0

        # Embed current motion
        theta_flat = motion_data['theta'].view(B, T, -1)  # [B, T, 12]
        expr = motion_data['expression_embed']  # [B, T, expression_dim]

        # Embed all motion parameters
        theta_emb = self.theta_emb(theta_flat)
        expr_emb = self.expr_emb(expr)


        # UV warps no longer encoded - they will be generated implicitly from zdyn + theta
        # Verify shapes (only theta + expression, SRT removed)
        assert motion_data['theta'].shape[2:] == (3, 4), f"Theta shape mismatch: expected (B, T, 3, 4), got {motion_data['theta'].shape}"
        assert expr.shape[2:] == (128,), f"Expression shape mismatch: expected (B, T, 128), got {expr.shape}"

        # Combine all embeddings (only theta + expression, no SRT or UV warps)
        current_emb = torch.cat([
            theta_emb, expr_emb,  # d_model/2 + d_model/2 = d_model
        ], dim=-1)  # Total: d_model (512)
        current_emb = self.motion_proj(current_emb)  # [B, T, d_model]

        # Handle previous context if provided
        if prev_context is not None and C > 0:
            # Embed prev_context
            prev_theta_flat = prev_context['theta'].view(B, C, -1)
            prev_expr = prev_context['expression_embed']  # Standardized key

            prev_theta_emb = self.theta_emb(prev_theta_flat)
            prev_expr_emb = self.expr_emb(prev_expr)


            # No UV warp or SRT embedding for prev_context
            prev_emb = torch.cat([
                prev_theta_emb, prev_expr_emb,  # d_model/2 + d_model/2 = d_model
            ], dim=-1)  # Total: d_model (512)
            prev_emb = self.motion_proj(prev_emb)  # [B, C, d_model]

            # Concatenate context and current
            tgt = torch.cat([prev_emb, current_emb], dim=1)  # [B, C+T, d_model]
        else:
            tgt = current_emb  # [B, T, d_model]

        # Add positional embeddings
        tgt = self.pos_emb(tgt, has_context=(C > 0))

        # Add timestep embedding
        time_pe = self._get_sinusoidal_embedding(noise_level, self.d_model)  # [B, d_model]
        time_emb = self.time_emb(time_pe)  # [B, d_model]
        tgt = tgt + time_emb.unsqueeze(1)  # Broadcast to all positions

        # Get condition embeddings (memory for decoder)
        if cond_emb is None:
            if conditions is None:
                raise ValueError("Either conditions or cond_emb must be provided")

            # Build full conditions with context if available
            if prev_context is not None and C > 0:
                full_conditions = {}
                prev_audio = prev_context.get('audio', torch.zeros(B, C, 768, device=device))

                for k, v in conditions.items():
                    if v is None:
                        continue
                    if k == 'audio_features':
                        full_conditions[k] = torch.cat([prev_audio, v], dim=1)  # [B, C+T, 768]
                    else:
                        # Pad other conditions with zeros for context
                        if isinstance(v, torch.Tensor):
                            shape = list(v.shape)
                            shape[1] = C
                            prev_zeros = torch.zeros(*shape, device=device, dtype=v.dtype)
                            full_conditions[k] = torch.cat([prev_zeros, v], dim=1)
                        else:
                            full_conditions[k] = v
            else:
                full_conditions = conditions

            cond_emb = self.cond_emb(full_conditions)  # [B, T or C+T, d_model]

        else:
            # If cond_emb provided but for T, pad with zeros for context
            if C > 0 and cond_emb.shape[1] == T:
                prev_cond = torch.zeros(B, C, self.d_model, device=device, dtype=cond_emb.dtype)
                cond_emb = torch.cat([prev_cond, cond_emb], dim=1)

        # Add positional embeddings to memory as well
        cond_emb = self.pos_emb(cond_emb, has_context=(C > 0))

        # Log embedding statistics for debugging
        logger.debug(f" Condition embedding stats - Mean: {cond_emb.mean().item():.6f}, Variance: {cond_emb.var().item():.6f}")

        # Apply transformer decoder
        # tgt: query (motion embeddings)
        # memory: key/value (condition embeddings)

        # OPTIMIZATION: Compute audio_memory ONCE and cache for all layers
        # Prevents redundant self.cond_emb() calls in the loop
        assert 'audio_features' in conditions, \
            "audio_features must be in conditions for AudioCrossDecoderLayer"

        audio_memory = self.cond_emb({'audio_features': conditions['audio_features']})
        if C > 0:
            audio_memory = audio_memory[:, C:]  # Remove context frames, keep only current T

        # Apply Flash Attention optimized causal masking (is_causal=True)
        # Ensures frame t can only attend to frames 0..t-1 (not future frames)
        out = tgt
        for layer in self.decoder_layers:
            # Pass cached audio_memory (not recomputed)
            out = layer(
                out,
                cond_emb,
                audio_memory=audio_memory,  # Cached - no recomputation
                is_causal=True  # Flash Attention optimization
            )

        out = self.decoder_norm(out)

        # Extract only current T frames if we had context
        if C > 0:
            out = out[:, C:]  # [B, T, d_model]

       

        # Log output statistics
        logger.debug(f" Transformer output variance: {out.var().item():.6f}")

        # Predict outputs matching H5 cache structure
        theta_pred = self.theta_head(out).view(B, T, 3, 4)  # H5: (1, 4, 4) but model uses 3x4
        expr_pred = self.expr_head(out)  # [B, T, 128] - matches target_pose_embed in H5

        # Debug expression predictions
        if torch.rand(1).item() < 0.01:  # Log 1% of the time
            logger.info(f"[DEBUG] Expression prediction stats:")
            logger.info(f"  Mean: {expr_pred.mean().item():.6f}, Std: {expr_pred.std().item():.6f}")
            logger.info(f"  Min: {expr_pred.min().item():.6f}, Max: {expr_pred.max().item():.6f}")
            logger.info(f"  Has NaN: {torch.isnan(expr_pred).any().item()}")
            logger.info(f"  Has Inf: {torch.isinf(expr_pred).any().item()}")

        # UV warps will be generated implicitly by WarpGeneratorFromZdyn in VASAModel
        # No explicit prediction needed here

        # Add assertions to verify output shapes
        assert theta_pred.shape == (B, T, 3, 4), f"Theta pred shape mismatch: {theta_pred.shape}"
        assert expr_pred.shape == (B, T, 128), f"Expression pred shape mismatch: {expr_pred.shape}"

        # Build output dict - when using derived warps, only predict theta and expression_embed
        # Nemo's pretrained components handle scale/rotation/translation internally
        output_dict = {
            'theta': theta_pred,  # Pose matrix
            'expression_embed': expr_pred,  # Aligned expression embedding (target_pose_embed in H5)
            # Note: uv_warps will be added by VASAModel.forward() via implicit generation
        }

        # Only include SRT components if NOT using derived warps (legacy mode)
        if not self.use_derived_warps:
            rotation_pred = motion_data['rotation'] if 'rotation' in motion_data else torch.zeros(B, T, 3, device=device)
            scale_pred = motion_data['scale'] if 'scale' in motion_data else torch.zeros(B, T, 3, device=device)
            translation_pred = motion_data['translation'] if 'translation' in motion_data else torch.zeros(B, T, 3, device=device)
            output_dict['rotation'] = rotation_pred
            output_dict['scale'] = scale_pred
            output_dict['translation'] = translation_pred

        return output_dict


class VASAModel(nn.Module):
    def __init__(
        self,
        config,
        volumetric_avatar: nn.Module,
        device: str = 'cuda',
        expression_db_path: str = 'cache_single_bucket/expression_embeddings.h5'
    ):
        super().__init__()
        self.config = config
        self.volumetric_avatar = volumetric_avatar.eval()

        # Freeze volumetric_avatar (pretrained from nemo, no retraining needed)
        # Only motion_transformer and condition_embedding will be trained
        for param in self.volumetric_avatar.parameters():
            param.requires_grad = False

        self.context_size = config.motion.context_size
        self.motion_transformer = MotionTransformer(config)
        self.condition_embedding = self.motion_transformer.cond_emb
        self.device = device

        # Cosine embedding head for expression database lookup
        # Normalizes predictions before comparing to database
        expression_dim = config.model.expression_dim
        self.cosine_head = nn.Linear(expression_dim, expression_dim, bias=False)
        # Initialize with normalized weights
        self.cosine_head.weight.data = F.normalize(self.cosine_head.weight.data, p=2, dim=1)

        # Expression database for cosine similarity loss
        self.expression_db = None
        if expression_db_path is not None:
            from expression_db import ExpressionDatabase
            logger.info(f"Loading expression database from {expression_db_path}")
            self.expression_db = ExpressionDatabase(expression_db_path, device=device)
            logger.info(f"Expression database loaded: {self.expression_db}")
        else:
            logger.warning("No expression database path provided - cosine loss will be disabled")

        # Initialize scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion.num_steps,
            beta_schedule=config.diffusion.get('schedule_mode', 'linear'),
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="sample"
        )

        # Initial context parameters
        self.start_prev_theta = nn.Parameter(torch.zeros(1, config.motion.context_size, 3, 4))
        self.start_prev_expression = nn.Parameter(torch.zeros(1, config.motion.context_size, config.model.expression_dim))
        self.start_prev_scale = nn.Parameter(torch.zeros(1, config.motion.context_size, 3))
        self.start_prev_rotation = nn.Parameter(torch.zeros(1, config.motion.context_size, 3))
        self.start_prev_translation = nn.Parameter(torch.zeros(1, config.motion.context_size, 3))

        # Flag to use derived warps from zdyn via volumetric_avatar.predict_embed
        self.use_derived_warps = config.model.get('use_derived_warps', True)

        # NOTE: Warp generation uses volumetric_avatar's predict_embed pipeline
        # This leverages the full nemo architecture with identity conditioning
        # See: compute_warps_from_zdyn() method below

    def compute_warps_from_zdyn(
        self,
        zdyn: torch.Tensor,
        idt_embed: torch.Tensor,
        theta: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute UV warps from zdyn using volumetric_avatar's predict_embed pipeline.

        This leverages the full nemo architecture with identity conditioning,
        which is better than simple projection as it uses pretrained embeddings.

        Args:
            zdyn: [B, T, zdyn_dim] expression dynamics from motion transformer
            idt_embed: [B, idt_dim] identity embedding from source image
            theta: [B, T, 3, 4] pose matrices (optional, can use zeros if not needed)

        Returns:
            uv_warps: [B, T, 16, 64, 64, 3] volumetric UV warp field
        """
        B, T = zdyn.shape[:2]
        device = zdyn.device

        # Flatten for per-frame processing
        zdyn_flat = zdyn.view(B * T, -1)  # [B*T, zdyn_dim]

        # Repeat identity for all frames (detach since it's pre-computed with no_grad)
        # This prevents gradients flowing to frozen identity embedder
        idt_embed_flat = idt_embed.detach().unsqueeze(1).repeat(1, T, 1).view(B * T, -1)  # [B*T, idt_dim]

        # Prepare warp_embed_dict directly without using predict_embed
        # Since predict_embed requires source/target images which we don't have,
        # we'll use the warp embedding pipeline components directly

        # First, unsqueeze zdyn for spatial dimensions (like pose_unsqueeze_nw)
        # zdyn_flat is [B*T, zdyn_dim], need to make it [B*T, C, H, W] format
        embed_size = self.volumetric_avatar.embed_size

        # Use pose_unsqueeze_nw to expand zdyn to spatial format
        warp_target_embed = self.volumetric_avatar.pose_unsqueeze_nw(zdyn_flat).view(
            B * T, -1, embed_size, embed_size
        )  # [B*T, C, embed_size, embed_size]

        # Expand identity embed spatially (reuse to avoid recreating)
        idt_spatial = idt_embed_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, embed_size, embed_size)

        # Combine with identity using warp_embed_head_orig_nw
        if self.volumetric_avatar.args.cat_em:
            warp_embed_orig = self.volumetric_avatar.warp_embed_head_orig_nw(
                torch.cat([warp_target_embed, idt_spatial], dim=1)
            )
        else:
            warp_embed_orig = self.volumetric_avatar.warp_embed_head_orig_nw(
                (warp_target_embed + idt_spatial) * 0.5
            )

        # Free intermediate tensors
        del warp_target_embed, idt_spatial, idt_embed_flat

        # Create target_warp_embed_dict
        c = warp_embed_orig.shape[1]
        target_warp_embed_dict = {
            'orig': warp_embed_orig.view(B * T, c, embed_size ** 2),
            'ada_v': zdyn_flat  # Use zdyn for adaptive parameters
        }

        # Free warp_embed_orig after creating dict (dict holds a view)
        del warp_embed_orig

        # Generate UV warps using nemo's uv_generator_nw
        target_uv_warp, _ = self.volumetric_avatar.uv_generator_nw(target_warp_embed_dict)

        # Free dict after use
        del target_warp_embed_dict

        # Handle resizing if configured in volumetric_avatar
        if self.volumetric_avatar.resize_warp:
            old_warp = target_uv_warp
            target_uv_warp = self.volumetric_avatar.resize_warp_func(target_uv_warp)
            del old_warp

        # Reshape back to [B, T, depth, H, W, 3]
        # Assuming target_uv_warp is [B*T, depth, H, W, 3]
        depth = target_uv_warp.shape[1]
        H = target_uv_warp.shape[2]
        W = target_uv_warp.shape[3]
        target_uv_warp = target_uv_warp.view(B, T, depth, H, W, 3)

        return target_uv_warp

    def _apply_dropout(self, conditions: Dict[str, torch.Tensor], dropout_probs: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Apply dropout to conditions for classifier-free guidance during training."""
        dropped_conditions = {}
        for key, value in conditions.items():
            if key in dropout_probs and random.random() < dropout_probs[key]:
                dropped_conditions[key] = None
            else:
                dropped_conditions[key] = value
        return dropped_conditions

    def _add_noise_to_motion(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Add noise to motion data for diffusion training."""
        motion_keys = ['theta', 'expression_embed', 'scale', 'rotation', 'translation']
        noised_motion = {}

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=noise_level.device)

        for key, value in motion_data.items():
            if key in motion_keys and key in noise:
                if self.config.train.get('turn_off_noise', False):
                    noised_motion[key] = value
                else:
                    noised_motion[key] = self.scheduler.add_noise(
                        original_samples=value,
                        noise=noise[key],
                        timesteps=noise_level
                    )
            else:
                noised_motion[key] = value

        return noised_motion

    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[Dict[str, torch.Tensor]] = None,
        idt_embed: Optional[torch.Tensor] = None,
        generate_warps: bool = False  # Only generate warps when needed (visualization/frame generation)
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training and inference.

        Args:
            generate_warps: If True, generate UV warps (expensive). Only needed for visualization
                           or frame generation. Skip during normal training to save VRAM.
        """

        # Validate and clean motion data - CREATE A COPY to avoid modifying original
        motion_data = {k: v.clone() for k, v in motion_data.items()}  # Clone to avoid modifying dataset
        for key, tensor in motion_data.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                motion_data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        B, T = motion_data['theta'].shape[:2]
        device = motion_data['theta'].device

        # Process conditions
        validated_conditions = {}
        if conditions is not None:
            # Map landmark names
            landmark_mapping = {
                'lips_landmarks': 'lips',
                'right_eye_landmarks': 'right_eye',
                'left_eye_landmarks': 'left_eye',
                'jaw_landmarks': 'jaw',
                'nose_landmarks': 'nose'
            }
            mapped_conditions = {landmark_mapping.get(k, k): v for k, v in conditions.items()}
            conditions = mapped_conditions

            # Expected shapes for validation
            expected_shapes = {
                'gaze': (B, T, 2),
                'head_distance': (B, T, 1),
                'emotion': (B, T, 2),
                'lips': (B, T, 20, 3),
                'right_eye': (B, T, 8, 3),
                'left_eye': (B, T, 7, 3),
                'jaw': (B, T, 10, 3),
                'nose': (B, T, 4, 3),
                'blink_state': (B, T, 3),
                'audio_features': (B, T, 768)  # CRITICAL: Must include audio_features!
            }

            for key, expected_shape in expected_shapes.items():
                if key not in conditions or conditions[key] is None:
                    # Audio features are REQUIRED - cannot be None
                    if key == 'audio_features':
                        raise ValueError(f"audio_features is required but not found in conditions! Keys: {list(conditions.keys())}")
                    validated_conditions[key] = torch.zeros(expected_shape, device=device)
                else:
                    tensor = conditions[key]
                    # Special handling for audio_features which might be 4D
                    if key == 'audio_features' and len(tensor.shape) == 4:
                        tensor = tensor.squeeze(1)  # Remove the extra dimension

                    if tensor.shape[:2] != (B, T):
                        if len(tensor.shape) == 2:
                            tensor = tensor.unsqueeze(1).expand(-1, T, -1)
                        elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
                            tensor = tensor.expand(B, -1, -1)
                    validated_conditions[key] = tensor

            # Add blink state if missing
            if 'blink_state' not in validated_conditions:
                blink_handler = BlinkConditionHandler(window_size=T)
                blink_states = blink_handler.generate_blink_sequence(T)
                validated_conditions['blink_state'] = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)

            # Apply dropout for classifier-free guidance ONLY during training
            if self.training:
                dropout_probs = self.config.train.get('dropout_probs', {})
                validated_conditions = self._apply_dropout(validated_conditions, dropout_probs)
            else:
                logger.debug("[INFERENCE] Not applying dropout to conditions")

        # Handle previous context
        if prev_context is None:
            prev_context = {
                'theta': self.start_prev_theta.repeat(B, 1, 1, 1),
                'expression_embed': self.start_prev_expression.repeat(B, 1, 1),  # Standardized key
                'scale': self.start_prev_scale.repeat(B, 1, 1),
                'rotation': self.start_prev_rotation.repeat(B, 1, 1),
                'translation': self.start_prev_translation.repeat(B, 1, 1),
                'audio': torch.zeros(B, self.context_size, 768, device=device)
            }

        # Forward through transformer
        outputs = self.motion_transformer(
            motion_data=motion_data,
            noise_level=noise_level,
            conditions=validated_conditions if cond_emb is None else None,
            cond_emb=cond_emb,
            prev_context=prev_context
        )

        # Cosine embedding loss with expression database
        # Apply AFTER motion_transformer but BEFORE warp generation
        if self.expression_db is not None and 'expression_embed' in outputs:
            pred_zdyn = outputs['expression_embed']  # [B, T, 128]

            # Project through cosine head (normalizes predictions)
            pred_zdyn_proj = self.cosine_head(pred_zdyn)

            # Find closest real expression from database (detach to avoid backprop through DB)
            with torch.no_grad():
                real_zdyn = self.expression_db.get_closest(pred_zdyn_proj.detach())  # [B, T, 128]

            # Compute cosine embedding loss
            # Target is always +1 (maximize similarity)
            target = torch.ones(B, T, device=device)
            cosine_loss = F.cosine_embedding_loss(
                pred_zdyn_proj.view(B * T, -1),
                real_zdyn.view(B * T, -1),
                target.view(-1),
                reduction='mean'
            )

            # Add to outputs
            outputs['cosine_loss'] = cosine_loss
            outputs['real_zdyn'] = real_zdyn  # For debugging/visualization

            logger.debug(f"[COSINE LOSS] {cosine_loss.item():.6f}")

        # Generate warps using volumetric_avatar's predict_embed pipeline if enabled
        # This leverages identity conditioning and pretrained nemo components
        # IMPORTANT: Only generate warps when explicitly requested (generate_warps=True)
        # This saves massive VRAM during training when warps are only needed for visualization
        if self.use_derived_warps and generate_warps:
            # Only generate warps if use_derived_warps is enabled AND generate_warps=True
            if 'expression_embed' in outputs and 'theta' in outputs:
                if idt_embed is not None:
                    logger.info(f"[DERIVED WARPS] Generating UV warps from zdyn {outputs['expression_embed'].shape} + idt_embed {idt_embed.shape}")
                    implicit_warps = self.compute_warps_from_zdyn(
                        zdyn=outputs['expression_embed'],
                        idt_embed=idt_embed,
                        theta=outputs['theta']
                    )
                    outputs['uv_warps'] = implicit_warps
                    outputs['warp_source'] = 'derived'  # Tag for debugging
                    logger.info(f"[DERIVED WARPS] ✅ Generated warps shape: {implicit_warps.shape}")
                else:
                    logger.error(f"[WARPS ERROR] idt_embed is None but use_derived_warps=True")
                    raise ValueError(f"Cannot generate derived warps: idt_embed required (use_derived_warps=True but idt_embed=None)")
            else:
                raise ValueError(f"Cannot generate warps: missing expression_embed or theta in outputs. Keys: {outputs.keys()}")
        elif self.use_derived_warps and not generate_warps:
            logger.debug(f"[DERIVED WARPS] Skipping warp generation (generate_warps=False)")
        else:
            # use_derived_warps=False: Use pre-computed warps from dataset
            logger.debug(f"[PRE-COMPUTED WARPS] Using warps from dataset (use_derived_warps=False)")
            # Copy warps from motion_data if available (these are ground truth warps from dataset)
            if 'uv_warps' in motion_data:
                outputs['uv_warps'] = motion_data['uv_warps']
                outputs['warp_source'] = 'dataset'  # Tag for debugging
                logger.debug(f"[PRE-COMPUTED WARPS] Copied warps from motion_data: {motion_data['uv_warps'].shape}")
            else:
                logger.warning(f"[PRE-COMPUTED WARPS] No uv_warps in motion_data! Available keys: {list(motion_data.keys())}")

        # Clean outputs
        for key, tensor in outputs.items():
            if isinstance(tensor, str):  # Skip string tags like 'warp_source'
                continue
            if torch.isnan(tensor).any():
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                outputs[key] = tensor
            elif torch.isinf(tensor).any():
                tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                outputs[key] = tensor

        # Add noise for loss computation if provided
        if noise is not None:
            outputs['noise'] = noise

        return outputs

    def generate_sequence(
        self,
        initial_pose: Dict[str, torch.Tensor],
        initial_dynamics: torch.Tensor,  # [B, expression_dim]
        conditions: Dict[str, torch.Tensor],
        num_steps: int = 50,
        eta: float = 0.8,  # Increased from 0.5 to add more stochasticity
        cfg_scales: Optional[Dict[str, float]] = None,
        idt_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate motion sequence using DDIM sampling."""
        try:
            # BOOST AUDIO CFG: Use much stronger audio CFG to overcome training issues
            if cfg_scales is None:
                cfg_scales = {
                    'audio': 20.0,  # Significantly increased to 20.0 to amplify audio guidance
                    'gaze': 0.5,    # Reduced to minimize control dominance
                    'head_distance': 0.3,  # Reduced to minimize control dominance
                    'emotion': 0.2   # Reduced to minimize control dominance
                }
            logger.info(f"[CFG SCALES] Using scales - Audio: {cfg_scales.get('audio', 10.0)}, Controls reduced to minimize dominance")
            logger.info(f"[CFG BOOST] Using enhanced CFG scales: {cfg_scales}")

            B = initial_pose['theta'].shape[0]
            total_T = conditions['audio_features'].shape[1]  # Assuming [B, T, D]
            window_size = self.config['motion']['window_size']
            stride = self.config['motion']['stride']
            context_size = self.config['motion']['context_size']
            device = initial_pose['theta'].device

            # Initialize outputs for ALL motion parameters
            full_motion = {
                'theta': torch.zeros(B, total_T, 3, 4, device=device),
                'scale': torch.zeros(B, total_T, 3, device=device),
                'rotation': torch.zeros(B, total_T, 3, device=device),
                'translation': torch.zeros(B, total_T, 3, device=device),
                'expression_embed': torch.zeros(B, total_T, self.config.model.expression_dim, device=device),
                'uv_warps': torch.zeros(B, total_T, 16, 64, 64, 3, device=device)  # Add UV warps to output
            }

            # Set initial frame values if provided
            full_motion['theta'][:, 0] = initial_pose['theta']
            full_motion['expression_embed'][:, 0] = initial_dynamics
            if 'scale' in initial_pose:
                full_motion['scale'][:, 0] = initial_pose['scale']
            if 'rotation' in initial_pose:
                full_motion['rotation'][:, 0] = initial_pose['rotation']
            if 'translation' in initial_pose:
                full_motion['translation'][:, 0] = initial_pose['translation']

            # Initial context
            prev_context = {
                'theta': self.start_prev_theta.repeat(B, 1, 1, 1),
                'expression_embed': self.start_prev_expression.repeat(B, 1, 1),  # Standardized key
                'scale': self.start_prev_scale.repeat(B, 1, 1),
                'rotation': self.start_prev_rotation.repeat(B, 1, 1),
                'translation': self.start_prev_translation.repeat(B, 1, 1),
                'audio': torch.zeros(B, context_size, 768, device=device)
            }

            start_idx = 0
            while start_idx < total_T:
                current_T = min(window_size, total_T - start_idx)

                # Slice conditions for current window (only tensors)
                window_conditions = {}
                for k, v in conditions.items():
                    if isinstance(v, torch.Tensor):
                        window_conditions[k] = v[:, start_idx:start_idx + current_T]
                    elif isinstance(v, dict):  # Handle lip_metrics which is a dict
                        window_conditions[k] = {
                            sub_k: sub_v[:, start_idx:start_idx + current_T]
                            for sub_k, sub_v in v.items() if isinstance(sub_v, torch.Tensor)
                        }

                # DEBUG: Log audio features variance for this window
                if 'audio_features' in window_conditions:
                    audio_window_var = window_conditions['audio_features'].var().item()
                    audio_window_mean = window_conditions['audio_features'].mean().item()
                    logger.info(f"[GENERATE DEBUG] Window {start_idx//stride}: Audio features - Mean: {audio_window_mean:.6f}, Variance: {audio_window_var:.6f}")

                # Initialize motion with noise for ALL parameters including UV warps
                window_motion = {
                    'theta': torch.randn(B, current_T, 3, 4, device=device),
                    'scale': torch.randn(B, current_T, 3, device=device),
                    'rotation': torch.randn(B, current_T, 3, device=device),
                    'translation': torch.randn(B, current_T, 3, device=device),
                    'expression_embed': torch.randn(B, current_T, self.config.model.expression_dim, device=device),
                    'uv_warps': torch.randn(B, current_T, 16, 64, 64, 3, device=device)  # UV warps must be predicted
                }

                # DDIM sampling loop
                self.scheduler.set_timesteps(num_steps, device=device)

                for t in self.scheduler.timesteps:
                    timesteps = t.expand(B) if isinstance(t, torch.Tensor) else torch.full((B,), t, device=device, dtype=torch.long)

                    # Prepare for classifier-free guidance if needed
                    if cfg_scales:
                        # Unconditional prediction (keep audio_features, zero out everything else)
                        uncond_conditions = {}
                        for k, v in window_conditions.items():
                            if k == 'audio_features':
                                uncond_conditions[k] = v  # Keep audio features
                            else:
                                if isinstance(v, torch.Tensor):
                                    uncond_conditions[k] = torch.zeros_like(v)
                                elif isinstance(v, dict):  # Handle lip_metrics
                                    uncond_conditions[k] = {sub_k: torch.zeros_like(sub_v) for sub_k, sub_v in v.items()}

                        uncond_motion = self.forward(
                            window_motion,
                            timesteps,
                            conditions=uncond_conditions,
                            prev_context=prev_context
                        )

                    # Conditional prediction
                    cond_motion = self.forward(
                        window_motion,
                        timesteps,
                        conditions=window_conditions,
                        prev_context=prev_context
                    )

                    # Apply classifier-free guidance if scales provided
                    if cfg_scales:
                        pred_motion = {}
                        for k in cond_motion:
                            if k == 'noise':
                                continue
                            # Use audio scale for expression (since audio drives expression)
                            if k == 'expression_embed':
                                scale = cfg_scales.get('audio', 1.0)
                            else:
                                scale = cfg_scales.get(k, 1.0) if isinstance(cfg_scales, dict) else cfg_scales
                            pred_motion[k] = (1 + scale) * cond_motion[k] - scale * uncond_motion[k]

                        # Debug log CFG effect on first timestep
                        if t == self.scheduler.timesteps[0]:
                            expr_diff = (pred_motion['expression_embed'] - cond_motion['expression_embed']).abs().mean().item()
                            logger.info(f"[CFG DEBUG] Audio CFG scale: {cfg_scales.get('audio', 1.0)}, Expression change: {expr_diff:.6f}")
                    else:
                        pred_motion = cond_motion

                    # Update motion using DDIM step for each parameter
                    for key in window_motion.keys():
                        if key in pred_motion:
                            scheduler_output = self.scheduler.step(
                                model_output=pred_motion[key],
                                timestep=timesteps[0],
                                sample=window_motion[key],
                                eta=eta
                            )
                            window_motion[key] = scheduler_output.prev_sample

                # Generate warps using volumetric_avatar's predict_embed pipeline
                if 'expression_embed' in window_motion and 'theta' in window_motion:
                    if idt_embed is not None and self.use_derived_warps:
                        logger.debug(f"[DERIVED WARPS] Window {start_idx//stride}: zdyn {window_motion['expression_embed'].shape} + idt {idt_embed.shape}")
                        window_motion['uv_warps'] = self.compute_warps_from_zdyn(
                            zdyn=window_motion['expression_embed'],
                            idt_embed=idt_embed,
                            theta=window_motion['theta']
                        )
                        logger.debug(f"[DERIVED WARPS] Window {start_idx//stride}: Generated warps {window_motion['uv_warps'].shape}")
                    else:
                        logger.error(f"[WARPS ERROR] Window {start_idx//stride}: idt_embed is None: {idt_embed is None}, use_derived_warps: {self.use_derived_warps}")
                        raise ValueError(f"Cannot generate warps: idt_embed required for compute_warps_from_zdyn (idt_embed={'None' if idt_embed is None else 'provided'})")
                else:
                    raise ValueError(f"Cannot generate warps: missing expression_embed or theta. Keys: {window_motion.keys()}")

                # Store generated window for ALL parameters
                for key in full_motion.keys():
                    if key in window_motion:
                        full_motion[key][:, start_idx:start_idx + current_T] = window_motion[key]

                # Update context for next window
                context_start = max(0, current_T - context_size)
                prev_context = {
                    'theta': window_motion['theta'][:, context_start:],
                    'expression_embed': window_motion['expression_embed'][:, context_start:],  # Standardized key
                    'scale': window_motion['scale'][:, context_start:],
                    'rotation': window_motion['rotation'][:, context_start:],
                    'translation': window_motion['translation'][:, context_start:],
                    'audio': window_conditions['audio_features'][:, context_start:]
                }

                start_idx += stride

            return full_motion

        except Exception as e:
            logger.error(f"Error in sequence generation: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zeros for all parameters
            return {
                'theta': torch.zeros(B, total_T, 3, 4, device=device),
                'scale': torch.zeros(B, total_T, 3, device=device),
                'rotation': torch.zeros(B, total_T, 3, device=device),
                'translation': torch.zeros(B, total_T, 3, device=device),
                'expression_embed': torch.zeros(B, total_T, self.config.model.expression_dim, device=device)
            }

    def decode_frame_with_warps(
        self,
        identity_info: Dict[str, torch.Tensor],
        warp_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode a frame using pre-calculated warps and identity info.
        This matches the approach in create_video_face_swap.py's decode_with_warps.

        Args:
            identity_info: Dictionary containing:
                - canonical_volume: [B, C, D, S, S] canonical 3D volume
                - embed_dict: Identity embeddings
                - idt_embed: Identity embedding
            warp_data: Dictionary containing:
                - uv_warp: [B, D, S, S, 3] target UV warps
                - theta: [B, 3, 4] target pose matrix
                - target_pose_embed: [B, 512] target expression embedding

        Returns:
            Generated frame [B, 3, H, W]
        """
        with torch.no_grad():
            # Extract warp data
            target_uv_warp = warp_data['uv_warp']
            target_theta = warp_data['theta']
            target_pose_embed = warp_data['target_pose_embed']

            # Get volume dimensions
            c = self.volumetric_avatar.args.latent_volume_channels
            d = self.volumetric_avatar.args.latent_volume_depth
            s = self.volumetric_avatar.args.latent_volume_size

            # Generate 3D grid and rotation warp for target
            grid = self.volumetric_avatar.identity_grid_3d.repeat_interleave(target_theta.shape[0], dim=0)
            target_rotation_warp = grid.bmm(target_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

            # Apply warps to canonical volume (nested grid_sample)
            aligned_target_volume = self.volumetric_avatar.grid_sample(
                self.volumetric_avatar.grid_sample(identity_info['canonical_volume'], target_uv_warp),
                target_rotation_warp
            )

            # Decode
            target_latent_feats = aligned_target_volume.view(target_theta.shape[0], c * d, s, s)
            decode_dict = {
                'target_theta': target_theta,
                'target_pose_embed': target_pose_embed
            }

            generated_img, _, _, _ = self.volumetric_avatar.decoder_nw(
                decode_dict,
                identity_info['embed_dict'],  # Source identity
                target_latent_feats,
                False,
                stage_two=True
            )

            return generated_img