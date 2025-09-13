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

# Add nemo to path for imports
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from logger import logger
from blink_condition_handler import BlinkConditionHandler

class VASAPositionalEmbedding(nn.Module):
    """
    Positional embeddings for VASA sequence generation with relative position encoding option.
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

        if use_relative_position:
            # Relative position encoding
            self.relative_pe = nn.Parameter(torch.randn(max_seq_len + max_context_len, d_model) * 0.02)
        else:
            # Standard sinusoidal embeddings
            pe = torch.zeros(max_seq_len + max_context_len, d_model)
            position = torch.arange(0, max_seq_len + max_context_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        self.context_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sequence_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        has_context: bool = False
    ) -> torch.Tensor:
        B, T, D = x.shape
        if self.use_relative_position:
            pos_encodings = self.relative_pe[:T].unsqueeze(0)
        else:
            pos_encodings = self.pe[:, :T]

        if has_context:
            type_embeddings = torch.cat([
                self.context_embedding.expand(B, self.max_context_len, D),
                self.sequence_embedding.expand(B, T - self.max_context_len, D)
            ], dim=1)
        else:
            type_embeddings = self.sequence_embedding.expand(B, T, D)

        x = x + pos_encodings + type_embeddings
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

        self.channel_layout = {}
        curr_idx = 0
        for key, channel_info in config.channel_layout.items():
            size = eval(str(channel_info.size)) if isinstance(channel_info.size, str) else channel_info.size
            self.channel_layout[key] = (curr_idx, curr_idx + size)
            curr_idx += size

        if curr_idx > self.model_dim:
            raise ValueError(f"Total feature dimension {curr_idx} exceeds model dimension {self.model_dim}")

        # self.landmark_dims = {
        #     name: info.points * info.coords
        #     for name, info in config.landmarks.items()
        # }

        self.audio_proj = nn.Sequential(
            nn.Linear(768, config.projections.audio.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projections.audio.hidden_dim, config.projections.audio.output_dim)
        )
        logger.info("Using full projection for Wav2Vec features (768 dimensions)")

        self.gaze_proj = nn.Linear(2, 2)
        self.distance_proj = nn.Linear(1, 1)
        self.emotion_proj = nn.Linear(2, 2)
        # self.speed_proj = nn.Linear(1, 1)  # Removed speed bucket

        # total_landmark_dims = sum(self.landmark_dims.values())
        # self.landmark_norm = nn.LayerNorm(total_landmark_dims)
        self.audio_norm = nn.LayerNorm(config.projections.audio.output_dim)
        self.control_norm = nn.LayerNorm(config.projections.control_norm_dim)

        self.blink_embed = nn.Sequential(
            nn.Linear(config.projections.blink.input_dim, config.projections.blink.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projections.blink.hidden_dim, config.projections.blink.output_dim)
        )

        self.final_norm = nn.LayerNorm(model_dim)

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

            audio_projected = self.audio_proj(audio)
            audio_normalized = self.audio_norm(audio_projected)

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

            # Speed bucket removed - was causing dtype issues
            # speed_tensor = conditions.get('speed_bucket')
            # if speed_tensor is None:
            #     speed_tensor = torch.zeros(B, T, 1, device=device, dtype=dtype)
            # else:
            #     speed_tensor = self._ensure_float_tensor(speed_tensor, dtype)
            # speed = self.speed_proj(speed_tensor)

            controls = torch.cat([gaze, distance, emotion], dim=-1)  # Removed speed
            controls_normalized = self.control_norm(controls)

            # landmarks = []
            # for key in self.landmark_dims.keys():
            #     lm = conditions.get(key, torch.zeros(B, T, self.landmark_dims[key], device=device))
            #     landmarks.append(lm)
            # landmarks_combined = torch.cat(landmarks, dim=-1)
            # landmarks_normalized = self.landmark_norm(landmarks_combined)

            # Handle blink_state with None check
            blink_tensor = conditions.get('blink_state')
            if blink_tensor is None:
                blink_tensor = torch.zeros(B, T, 3, device=device, dtype=dtype)
            blink_embedded = self.blink_embed(blink_tensor)

            combined = torch.cat([audio_normalized, controls_normalized,  blink_embedded], dim=-1) # landmarks_normalized
            output[:, :, :combined.shape[-1]] = combined

            return self.final_norm(output)

        except Exception as e:
            logger.error(f"Error in condition embedding: {str(e)}")
            raise


class MotionTransformer(nn.Module):
    """Diffusion Transformer for holistic facial dynamics generation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.model.hidden_dim
        self.num_layers = config.model.n_layers
        self.num_heads = config.model.n_heads

        self.pos_embed = VASAPositionalEmbedding(
            d_model=self.d_model,
            max_seq_len=config.motion.window_size,
            max_context_len=config.motion.context_size,
            dropout=config.model.dropout,
            use_relative_position=config.model.use_relative_position
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.cond_embed = EfficientConditionEmbedding(
            model_dim=config.model.condition_embedding_dim,
            max_seq_len=config.motion.window_size + config.motion.context_size
        )

        # Project motion (all parameters) to d_model dimensions
        # theta: 12, scale: 3, rotation: 3, translation: 3, expression: 128
        motion_input_dim = 12 + 3 + 3 + 3 + config.model.expression_dim  # Total: 12+3+3+3+128 = 149
        self.motion_proj = nn.Linear(motion_input_dim, self.d_model)

        # Log the expected dimensions for debugging
        logger.info(f"MotionTransformer initialized: motion_input_dim={motion_input_dim}, d_model={self.d_model}")

        # Output projections for all motion parameters
        self.pose_proj = nn.Linear(self.d_model, 12)  # theta
        self.scale_proj = nn.Linear(self.d_model, 3)  # scale
        self.rotation_proj = nn.Linear(self.d_model, 3)  # rotation
        self.translation_proj = nn.Linear(self.d_model, 3)  # translation
        self.dyn_proj = nn.Linear(self.d_model, config.model.expression_dim)  # expression

    def _check_and_fix_motion_proj(self):
        """Check if motion_proj has correct dimensions and reinitialize if needed."""
        # theta: 12, scale: 3, rotation: 3, translation: 3, expression: 128 = 149 total
        expected_input_dim = 12 + 3 + 3 + 3 + self.config.model.expression_dim  # 149
        if hasattr(self, 'motion_proj'):
            actual_input_dim = self.motion_proj.in_features
            if actual_input_dim != expected_input_dim:
                logger.warning(f"motion_proj has wrong input dimension: {actual_input_dim} vs expected {expected_input_dim}")
                logger.warning("Reinitializing motion_proj with correct dimensions...")
                self.motion_proj = nn.Linear(expected_input_dim, self.d_model).to(self.motion_proj.weight.device)
                nn.init.xavier_uniform_(self.motion_proj.weight)
                if self.motion_proj.bias is not None:
                    nn.init.zeros_(self.motion_proj.bias)

    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Check and fix motion_proj dimensions if needed (for checkpoint compatibility)
        self._check_and_fix_motion_proj()

        B, T = motion_data['theta'].shape[:2]
        device = motion_data['theta'].device

        # Ensure noise_level is float tensor
        noise_level = noise_level.float() if noise_level.dtype != torch.float32 else noise_level

        noise_emb = self.noise_embed(noise_level.unsqueeze(-1)).unsqueeze(1).expand(B, T, -1)

        if cond_emb is None:
            cond_emb = self.cond_embed(conditions, prev_context)

        # Concatenate all motion parameters, then project to d_model dimensions
        motion_components = [
            motion_data['theta'].view(B, T, -1),  # 12 dims
            motion_data.get('scale', torch.ones(B, T, 3, device=device)),  # 3 dims
            motion_data.get('rotation', torch.zeros(B, T, 3, device=device)),  # 3 dims
            motion_data.get('translation', torch.zeros(B, T, 3, device=device)),  # 3 dims
            motion_data['expression_embed']  # 128 dims
        ]
        motion_flat = torch.cat(motion_components, dim=-1)  # Total: 149 dims
        motion_emb = self.motion_proj(motion_flat)

        # Now all embeddings have the same dimension (d_model)
        input_emb = motion_emb + cond_emb + noise_emb

        has_context = prev_context is not None
        input_emb = self.pos_embed(input_emb, has_context)

        transformer_out = self.transformer(input_emb.transpose(0, 1)).transpose(0, 1)

        # Generate all motion parameters using dedicated projections
        outputs = {
            'theta': self.pose_proj(transformer_out).view(B, T, 3, 4),
            'scale': self.scale_proj(transformer_out),
            'rotation': self.rotation_proj(transformer_out),
            'translation': self.translation_proj(transformer_out),
            'expression_embed': self.dyn_proj(transformer_out)
        }

        return outputs


class VASAModel(nn.Module):
    def __init__(
        self,
        config,
        volumetric_avatar: nn.Module,
        device: str = 'cuda'
    ):
        super().__init__()
        self.config = config
        self.volumetric_avatar = volumetric_avatar.eval()
        for param in self.volumetric_avatar.parameters():
            param.requires_grad = False

        self.context_size = config.motion.context_size

        self.motion_transformer = MotionTransformer(config)

        expression_dim = config.model.expression_dim
        self.start_prev_theta = nn.Parameter(torch.randn(1, self.context_size, 3, 4) * 0.01)
        self.start_prev_expression = nn.Parameter(torch.randn(1, self.context_size, expression_dim) * 0.01)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.diffusion.num_steps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            clip_sample=True,
            prediction_type="sample",
            timestep_spacing="leading"
        )

        self.dropout_probs = config.train.dropout_probs

    def _apply_dropout(self, conditions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply classifier-free guidance dropout to conditions."""
        dropped_conditions = conditions.copy()
        for key, prob in self.dropout_probs.items():
            if random.random() < prob:
                if key in dropped_conditions:
                    dropped_conditions[key] = torch.zeros_like(dropped_conditions[key])
        return dropped_conditions

    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        for key, tensor in motion_data.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                motion_data[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

        B, T = motion_data['theta'].shape[:2]
        device = motion_data['theta'].device

        validated_conditions = {}
        if conditions is not None:
            landmark_mapping = {
                'lips_landmarks': 'lips',
                'right_eye_landmarks': 'right_eye',
                'left_eye_landmarks': 'left_eye',
                'jaw_landmarks': 'jaw',
                'nose_landmarks': 'nose'
            }
            mapped_conditions = {landmark_mapping.get(k, k): v for k, v in conditions.items()}
            conditions = mapped_conditions

            expected_shapes = {
                'gaze': (B, T, 2),
                'head_distance': (B, T, 1),
                'emotion': (B, T, 2),
                # 'speed_bucket': (B, T, 1),  # Removed speed bucket
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

            if 'blink_state' not in validated_conditions:
                blink_handler = BlinkConditionHandler(window_size=T)
                blink_states = blink_handler.generate_blink_sequence(T)
                validated_conditions['blink_state'] = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)

            # Apply dropout for classifier-free guidance
            validated_conditions = self._apply_dropout(validated_conditions)

        if prev_context is None:
            prev_context = {
                'theta': self.start_prev_theta.repeat(B, 1, 1, 1),
                'expression': self.start_prev_expression.repeat(B, 1, 1),
                'audio': torch.zeros(B, self.context_size, 768, device=device)
            }

        outputs = self.motion_transformer(
            motion_data=motion_data,
            noise_level=noise_level,
            conditions=validated_conditions if cond_emb is None else None,
            cond_emb=cond_emb,
            prev_context=prev_context
        )

        for key, tensor in outputs.items():
            if torch.isnan(tensor).any():
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                outputs[key] = tensor
            elif torch.isinf(tensor).any():
                tensor = torch.clamp(tensor, min=-10.0, max=10.0)
                outputs[key] = tensor

        if noise is not None:
            outputs['noise'] = noise

        return outputs

    def _add_noise_to_motion(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        motion_keys = ['theta', 'expression_embed']
        noised_motion = {}
        
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=noise_level.device)

        for key, value in motion_data.items():
            if key in motion_keys:
                if self.config.train.turn_off_noise:
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

    def generate_sequence(
        self,
        initial_pose: Dict[str, torch.Tensor],
        initial_dynamics: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        num_steps: int = 50,
        eta: float = 0.5,
        cfg_scales: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate sequence using DDIM sampling with sliding window and CFG."""
        self.eval()
        with torch.no_grad():
            audio_features = conditions['audio_features']
            B, total_T = audio_features.shape[:2]
            device = initial_pose['theta'].device

            window_size = self.config.motion.window_size
            stride = self.config.motion.stride
            context_size = self.config.motion.context_size

            full_motion = {
                'theta': torch.zeros(B, total_T, 3, 4, device=device),
                'expression_embed': torch.zeros(B, total_T, self.config.model.expression_dim, device=device)
            }

            full_motion['theta'][:, 0] = initial_pose['theta']
            full_motion['expression_embed'][:, 0] = initial_dynamics

            prev_context = None
            start_frame = 0

            while start_frame < total_T:
                end_frame = min(start_frame + window_size, total_T)
                current_T = end_frame - start_frame

                window_conditions = {k: v[:, start_frame:end_frame] for k, v in conditions.items() if isinstance(v, torch.Tensor)}

                window_motion = {
                    'theta': torch.randn(B, current_T, 3, 4, device=device),
                    'expression_embed': torch.randn(B, current_T, self.config.model.expression_dim, device=device)
                }

                self.scheduler.set_timesteps(num_steps, device=device)

                for t in self.scheduler.timesteps:
                    if cfg_scales:
                        uncond_outputs = self.forward(
                            motion_data=window_motion,
                            noise_level=t.expand(B),
                            conditions=None,
                            prev_context=prev_context
                        )

                    cond_outputs = self.forward(
                        motion_data=window_motion,
                        noise_level=t.expand(B),
                        conditions=window_conditions,
                        prev_context=prev_context
                    )

                    if cfg_scales:
                        model_output = {}
                        for key in cond_outputs:
                            scale = sum(cfg_scales.values()) if isinstance(cfg_scales, dict) else cfg_scales
                            model_output[key] = (1 + scale) * cond_outputs[key] - scale * uncond_outputs[key]
                    else:
                        model_output = cond_outputs

                    for key in window_motion.keys():
                        scheduler_output = self.scheduler.step(
                            model_output=model_output[key],
                            timestep=t,
                            sample=window_motion[key],
                            eta=eta
                        )
                        window_motion[key] = scheduler_output.prev_sample

                full_motion['theta'][:, start_frame:end_frame] = window_motion['theta']
                full_motion['expression_embed'][:, start_frame:end_frame] = window_motion['expression_embed']

                prev_start = max(0, current_T - context_size)
                prev_context = {
                    'theta': window_motion['theta'][:, prev_start:],
                    'expression': window_motion['expression_embed'][:, prev_start:],
                    'audio': window_conditions['audio_features'][:, prev_start:]
                }

                start_frame += stride

            return full_motion





# """
# VASA Model Implementation - Holistic Facial Dynamics Generation using Diffusion Transformers
# Aligns with VASA-1 documentation specifications.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Optional, Tuple
# from diffusers import DDIMScheduler
# from omegaconf import OmegaConf
# import yaml
# from pathlib import Path
# import math
# import random
# import logging
# import traceback
# import sys

# # Add nemo to path if needed
# if 'nemo' not in sys.path:
#     sys.path.insert(0, 'nemo')

# from logger import logger
# from blink_condition_handler import BlinkConditionHandler

# __all__ = ['VASAPositionalEmbedding', 'EfficientConditionEmbedding', 'MotionTransformer', 'VASAModel']

# class VASAPositionalEmbedding(nn.Module):
#     """
#     Positional embeddings for VASA sequence generation with relative position encoding option.
    
#     Args:
#         d_model: Embedding dimension (default: 512)
#         max_seq_len: Maximum sequence length (default: 50)
#         max_context_len: Maximum context length (default: 10)
#         dropout: Dropout probability (default: 0.1)
#         use_relative_position: Use relative positional encoding (default: True)
#     """
#     def __init__(
#         self,
#         d_model: int = 512,
#         max_seq_len: int = 50,
#         max_context_len: int = 10,
#         dropout: float = 0.1,
#         use_relative_position: bool = True
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.max_seq_len = max_seq_len
#         self.max_context_len = max_context_len
#         self.use_relative_position = use_relative_position
#         self.dropout = nn.Dropout(dropout)
        
#         # Relative positional encoding
#         if use_relative_position:
#             self.relative_pe = nn.Parameter(
#                 torch.zeros(max_seq_len + max_context_len, d_model)
#             )
        
#         # Standard sinusoidal positional encoding
#         pe = torch.zeros(1, max_seq_len + max_context_len, d_model)
#         position = torch.arange(0, max_seq_len + max_context_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
#         )
        
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
        
#         # Context and sequence markers
#         self.context_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
#         self.sequence_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        
#     def forward(
#         self,
#         x: torch.Tensor,  # [B, T, D]
#         has_context: bool = False
#     ) -> torch.Tensor:  # [B, T, D]
#         try:
#             B, T, D = x.shape
#             assert D == self.d_model, f"Expected dim {self.d_model}, got {D}"
            
#             # Get base positional encoding
#             positions = self.pe[:, :T, :]
            
#             # Add relative positions if enabled
#             if self.use_relative_position:
#                 relative_pos = self.relative_pe[:T].unsqueeze(0)
#                 positions = positions + relative_pos
            
#             # Add context/sequence markers
#             if has_context:
#                 # Assume first max_context_len frames are context
#                 context_len = min(self.max_context_len, T)
#                 positions[:, :context_len] += self.context_embedding
#                 positions[:, context_len:] += self.sequence_embedding
#             else:
#                 positions += self.sequence_embedding
            
#             # Add to input and apply dropout
#             x = x + positions
#             x = self.dropout(x)
            
#             assert x.shape == (B, T, self.d_model)
#             return x
            
#         except Exception as e:
#             logger.error(f"Error in positional embedding: {str(e)}")
#             logger.error(traceback.format_exc())
#             return torch.zeros_like(x)

# class EfficientConditionEmbedding(nn.Module):
#     """
#     Processes and embeds various conditioning signals for motion generation.
#     Handles None values robustly by replacing with zero tensors.
    
#     Args:
#         model_dim: Output embedding dimension (default: 512)
#         max_seq_len: Maximum sequence length (default: 60)
#     """
#     def __init__(
#         self,
#         model_dim: int = 512,
#         max_seq_len: int = 60
#     ):
#         super().__init__()
#         self.model_dim = model_dim
#         self.max_seq_len = max_seq_len
        
#         # Load channel config
#         channel_config_path = Path(__file__).parent / 'channel_config.yaml'
#         with open(channel_config_path, 'r') as f:
#             self.channel_config = yaml.safe_load(f)
        
#         # Audio projection (wav2vec features: 768 -> hidden -> output)
#         audio_cfg = self.channel_config['projections']['audio']
#         self.audio_proj = nn.Sequential(
#             nn.Linear(audio_cfg['input_dim'], audio_cfg['hidden_dim']),
#             nn.ReLU(),
#             nn.Linear(audio_cfg['hidden_dim'], audio_cfg['output_dim'])
#         )
        
#         # Control signal projections
#         self.gaze_proj = nn.Linear(2, 2)
#         self.distance_proj = nn.Linear(1, 1)
#         self.emotion_proj = nn.Linear(2, 2)
#         self.speed_proj = nn.Linear(1, 1)
        
#         # Blink state embedding
#         blink_cfg = self.channel_config['projections']['blink']
#         self.blink_embed = nn.Sequential(
#             nn.Linear(blink_cfg['input_dim'], blink_cfg['hidden_dim']),
#             nn.ReLU(),
#             nn.Linear(blink_cfg['hidden_dim'], blink_cfg['output_dim'])
#         )
        
#         # Final normalization
#         self.norm = nn.LayerNorm(model_dim)
        
#     def forward(
#         self,
#         conditions: Dict[str, torch.Tensor],
#         prev_context: Optional[Dict[str, torch.Tensor]] = None
#     ) -> torch.Tensor:  # [B, T, model_dim]
#         try:
#             # Validate required audio features
#             if 'audio_features' not in conditions or conditions['audio_features'] is None:
#                 raise ValueError("audio_features are required in conditions")
            
#             audio = conditions['audio_features']
#             if audio.dim() == 4:  # [B, 1, T, D] -> [B, T, D]
#                 audio = audio.squeeze(1)
            
#             B, T, _ = audio.shape
            
#             # Handle missing conditions with zeros
#             device = audio.device
#             dtype = audio.dtype
            
#             def get_or_zero(key: str, shape: Tuple[int, ...]) -> torch.Tensor:
#                 if key in conditions and conditions[key] is not None:
#                     tensor = conditions[key]
#                     assert tensor.shape == (B, T, *shape[2:]), f"Invalid shape for {key}: {tensor.shape}"
#                     return tensor
#                 return torch.zeros(B, T, *shape[2:], device=device, dtype=dtype)
            
#             # Gather all conditions
#             gaze = get_or_zero('gaze', (B, T, 2))
#             head_distance = get_or_zero('head_distance', (B, T, 1))
#             emotion = get_or_zero('emotion', (B, T, 2))
#             speed_bucket = get_or_zero('speed_bucket', (B, T, 1))
#             blink_state = get_or_zero('blink_state', (B, T, 3))
            
#             # Project conditions
#             audio_emb = self.audio_proj(audio)  # [B, T, audio_output_dim]
#             gaze_emb = self.gaze_proj(gaze)
#             dist_emb = self.distance_proj(head_distance)
#             emo_emb = self.emotion_proj(emotion)
#             speed_emb = self.speed_proj(speed_bucket)
#             blink_emb = self.blink_embed(blink_state)
            
#             # Concatenate all embeddings
#             emb = torch.cat([
#                 audio_emb,
#                 gaze_emb,
#                 dist_emb,
#                 emo_emb,
#                 speed_emb,
#                 blink_emb
#             ], dim=-1)  # [B, T, sum_dims]
            
#             # Handle padding to model_dim
#             current_dim = emb.shape[-1]
#             if current_dim < self.model_dim:
#                 padding = torch.zeros(B, T, self.model_dim - current_dim, device=device, dtype=dtype)
#                 emb = torch.cat([emb, padding], dim=-1)
#             elif current_dim > self.model_dim:
#                 emb = emb[..., :self.model_dim]
            
#             # Normalize
#             emb = self.norm(emb)
            
#             assert emb.shape == (B, T, self.model_dim)
#             return emb
            
#         except Exception as e:
#             logger.error(f"Error in condition embedding: {str(e)}")
#             logger.error(traceback.format_exc())
#             # Fallback to zero embedding
#             return torch.zeros(conditions['audio_features'].shape[:2] + (self.model_dim,), 
#                                device=conditions['audio_features'].device)

# class MotionTransformer(nn.Module):
#     """
#     Diffusion Transformer for holistic facial dynamics generation.
    
#     Args:
#         config: Configuration dictionary or OmegaConf
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config if isinstance(config, dict) else OmegaConf.to_container(config, resolve=True)
        
#         d_model = self.config['model']['hidden_dim']
#         self.d_model = d_model
#         self.motion_dim = self.config['model']['motion_dim']
        
#         # Noise level embedding
#         self.noise_embed = nn.Sequential(
#             nn.Linear(1, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model)
#         )
        
#         # Condition embedding module
#         self.cond_embed = EfficientConditionEmbedding(
#             model_dim=d_model,
#             max_seq_len=self.config['motion']['window_size']
#         )
        
#         # Positional encoding
#         self.pos_embed = VASAPositionalEmbedding(
#             d_model=d_model,
#             max_seq_len=self.config['motion']['window_size'],
#             max_context_len=self.config['motion']['context_size'],
#             dropout=self.config['model']['dropout'],
#             use_relative_position=self.config['model']['use_relative_position']
#         )
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=self.config['model']['n_heads'],
#             dim_feedforward=self.config['model']['dim_feedforward'],
#             dropout=self.config['model']['dropout'],
#             activation='relu'
#         )
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=self.config['model']['n_layers']
#         )
        
#         # Motion input projections
#         self.theta_proj_in = nn.Linear(3*4, d_model // 2)
#         self.expr_proj_in = nn.Linear(self.motion_dim, d_model // 2)
        
#         # Output projections
#         self.theta_proj_out = nn.Linear(d_model, 3*4)
#         self.expr_proj_out = nn.Linear(d_model, self.motion_dim)
        
#     def forward(
#         self,
#         motion_data: Dict[str, torch.Tensor],
#         noise_level: torch.Tensor,  # [B]
#         conditions: Optional[Dict[str, torch.Tensor]] = None,
#         cond_emb: Optional[torch.Tensor] = None,  # [B, T, D]
#         prev_context: Optional[Dict[str, torch.Tensor]] = None
#     ) -> Dict[str, torch.Tensor]:
#         try:
#             # Validate inputs
#             assert 'theta' in motion_data and 'expression_embed' in motion_data
#             B, T = motion_data['theta'].shape[:2]
#             assert noise_level.shape == (B,)
            
#             # Handle noise level as float tensor if needed
#             if not isinstance(noise_level, torch.Tensor):
#                 noise_level = torch.tensor([noise_level] * B, dtype=torch.float32, device=motion_data['theta'].device)
#             elif noise_level.dim() == 0:
#                 noise_level = noise_level.unsqueeze(0).repeat(B)
            
#             # Noise embedding [B, 1, D] -> [B, T, D]
#             noise_emb = self.noise_embed(noise_level.unsqueeze(-1))  # [B, D]
#             noise_emb = noise_emb.unsqueeze(1).repeat(1, T, 1)  # [B, T, D]
            
#             # Get condition embedding if not provided
#             if cond_emb is None and conditions is not None:
#                 cond_emb = self.cond_embed(conditions, prev_context)
            
#             # Flatten motion data
#             theta_flat = motion_data['theta'].view(B, T, -1)  # [B, T, 12]
#             expr_flat = motion_data['expression_embed']  # [B, T, motion_dim]
            
#             # Project motion inputs
#             theta_emb = self.theta_proj_in(theta_flat)
#             expr_emb = self.expr_proj_in(expr_flat)
#             motion_emb = torch.cat([theta_emb, expr_emb], dim=-1)  # [B, T, D]
            
#             # Combine with noise and conditions
#             x = motion_emb + noise_emb
#             if cond_emb is not None:
#                 x = x + cond_emb
            
#             # Add positional encoding
#             has_context = prev_context is not None
#             x = self.pos_embed(x, has_context=has_context)
            
#             # Transformer processing (src mask None for full attention)
#             x = x.permute(1, 0, 2)  # [T, B, D]
#             x = self.transformer(x)
#             x = x.permute(1, 0, 2)  # [B, T, D]
            
#             # Project to outputs
#             theta_out = self.theta_proj_out(x).view(B, T, 3, 4)
#             expr_out = self.expr_proj_out(x)
            
#             # Derive additional parameters
#             rotation = theta_out[:, :, :, :3].mean(dim=-1)  # Simplified; adjust as needed
#             translation = theta_out[:, :, :, 3]
#             scale = torch.ones(B, T, 3, device=x.device)
            
#             output = {
#                 'theta': theta_out,
#                 'expression_embed': expr_out,
#                 'rotation': rotation,
#                 'translation': translation,
#                 'scale': scale
#             }
            
#             return output
            
#         except Exception as e:
#             logger.error(f"Error in MotionTransformer: {str(e)}")
#             logger.error(traceback.format_exc())
#             # Return zero outputs on error
#             zero_theta = torch.zeros(B, T, 3, 4, device=motion_data['theta'].device)
#             zero_expr = torch.zeros(B, T, self.motion_dim, device=motion_data['theta'].device)
#             return {
#                 'theta': zero_theta,
#                 'expression_embed': zero_expr,
#                 'rotation': zero_theta[:, :, :, :3].mean(-1),
#                 'translation': zero_theta[:, :, :, 3],
#                 'scale': torch.zeros(B, T, 3, device=zero_theta.device)
#             }

# class VASAModel(nn.Module):
#     """
#     Main VASA model combining diffusion process with volumetric avatar rendering.
#     Aligns with VASA-1 specifications for motion diffusion.
    
#     Args:
#         config: Configuration (dict or OmegaConf)
#         volumetric_avatar: Pre-trained volumetric avatar module
#         device: Computation device (default: 'cuda')
#     """
#     def __init__(
#         self,
#         config,
#         volumetric_avatar: nn.Module,
#         device: str = 'cuda'
#     ):
#         super().__init__()
#         self.config = config if isinstance(config, dict) else OmegaConf.to_container(config, resolve=True)
#         self.device = device
#         self.volumetric_avatar = volumetric_avatar.eval()
#         for param in self.volumetric_avatar.parameters():
#             param.requires_grad = False
        
#         # Motion transformer
#         self.motion_transformer = MotionTransformer(self.config).to(device)
        
#         # Diffusion scheduler
#         self.scheduler = DDIMScheduler(
#             num_train_timesteps=self.config['diffusion']['num_steps'],
#             beta_start=self.config['diffusion']['beta_start'],
#             beta_end=self.config['diffusion']['beta_end'],
#             beta_schedule='linear',
#             clip_sample=False
#         )
        
#         # Initial context parameters
#         context_size = self.config['motion']['context_size']
#         self.start_prev_theta = nn.Parameter(torch.zeros(1, context_size, 3, 4))
#         self.start_prev_expression = nn.Parameter(torch.zeros(1, context_size, self.motion_transformer.motion_dim))
        
#         # Dropout probabilities for CFG
#         self.dropout_probs = self.config['train'].get('dropout_probs', {})
        
#         # Blink handler
#         self.blink_handler = BlinkConditionHandler()
        
#         # Clip bounds
#         self.clip_min = self.config['model']['clip_bounds']['min']
#         self.clip_max = self.config['model']['clip_bounds']['max']
        
#     def forward(
#         self,
#         motion_data: Dict[str, torch.Tensor],
#         noise_level: torch.Tensor,  # [B]
#         conditions: Optional[Dict[str, torch.Tensor]] = None,
#         cond_emb: Optional[torch.Tensor] = None,
#         prev_context: Optional[Dict[str, torch.Tensor]] = None,
#         noise: Optional[Dict[str, torch.Tensor]] = None
#     ) -> Dict[str, torch.Tensor]:
#         try:
#             # Sanitize inputs
#             for k in motion_data:
#                 motion_data[k] = torch.nan_to_num(
#                     motion_data[k],
#                     nan=0.0,
#                     posinf=self.clip_max,
#                     neginf=self.clip_min
#                 )
            
#             # Remap landmark keys if needed
#             remap_keys = {
#                 'lips_landmarks': 'lips',
#                 'right_eye_landmarks': 'right_eye',
#                 'left_eye_landmarks': 'left_eye',
#                 'jaw_landmarks': 'jaw',
#                 'nose_landmarks': 'nose'
#             }
#             for old_key, new_key in remap_keys.items():
#                 if old_key in conditions:
#                     conditions[new_key] = conditions.pop(old_key)
            
#             # Validate and prepare conditions
#             if conditions is None:
#                 conditions = {}
            
#             B, T = motion_data['theta'].shape[:2]
            
#             # Ensure consistent shapes
#             expected_shapes = {
#                 'gaze': (B, T, 2),
#                 'head_distance': (B, T, 1),
#                 'emotion': (B, T, 2),
#                 'speed_bucket': (B, T, 1),
#                 'lips': (B, T, 20, 3),
#                 'right_eye': (B, T, 8, 3),
#                 'left_eye': (B, T, 7, 3),
#                 'jaw': (B, T, 10, 3),
#                 'nose': (B, T, 4, 3),
#                 'blink_state': (B, T, 3)
#             }
            
#             device = motion_data['theta'].device
#             dtype = motion_data['theta'].dtype
            
#             for key, shape in expected_shapes.items():
#                 if key not in conditions or conditions[key] is None:
#                     conditions[key] = torch.zeros(shape, device=device, dtype=dtype)
#                 else:
#                     # Expand if needed
#                     if conditions[key].dim() < len(shape):
#                         conditions[key] = conditions[key].unsqueeze(0).expand(shape)
#                     assert conditions[key].shape == shape, f"Invalid shape for {key}: {conditions[key].shape} vs {shape}"
            
#             # Auto-generate blink state if not provided
#             if 'blink_state' not in conditions or torch.all(conditions['blink_state'] == 0):
#                 conditions['blink_state'] = self.blink_handler.generate_blink_conditions(B, T, device=device)
            
#             # Add noise if provided
#             if noise is not None:
#                 motion_data = self._add_noise_to_motion(motion_data, noise, noise_level)
            
#             # Forward through motion transformer
#             pred_motion = self.motion_transformer(
#                 motion_data=motion_data,
#                 noise_level=noise_level,
#                 conditions=conditions,
#                 cond_emb=cond_emb,
#                 prev_context=prev_context
#             )
            
#             # Sanitize outputs
#             for k in pred_motion:
#                 if isinstance(pred_motion[k], torch.Tensor):
#                     pred_motion[k] = torch.nan_to_num(
#                         pred_motion[k],
#                         nan=0.0,
#                         posinf=self.clip_max,
#                         neginf=self.clip_min
#                     )
#                     pred_motion[k] = torch.clamp(
#                         pred_motion[k],
#                         min=self.clip_min,
#                         max=self.clip_max
#                     )
            
#             return pred_motion
            
#         except Exception as e:
#             logger.error(f"Error in VASAModel forward: {str(e)}")
#             logger.error(traceback.format_exc())
#             # Return zero motion on error
#             B, T = motion_data['theta'].shape[:2]
#             zero_theta = torch.zeros(B, T, 3, 4, device=self.device)
#             zero_expr = torch.zeros(B, T, self.motion_transformer.motion_dim, device=self.device)
#             return {
#                 'theta': zero_theta,
#                 'expression_embed': zero_expr,
#                 'rotation': zero_theta[:, :, :, :3].mean(-1),
#                 'translation': zero_theta[:, :, :, 3],
#                 'scale': torch.zeros(B, T, 3, device=self.device)
#             }
    
#     def _apply_dropout(
#         self,
#         conditions: Dict[str, torch.Tensor]
#     ) -> Dict[str, torch.Tensor]:
#         """Apply classifier-free guidance dropout - replaces with zeros instead of None."""
#         dropped_conditions = conditions.copy()
#         for key, prob in self.dropout_probs.items():
#             if key != 'audio' and random.random() < prob:  # Never drop audio
#                 dropped_conditions[key] = torch.zeros_like(conditions[key])
#         return dropped_conditions
    
#     def _add_noise_to_motion(
#         self,
#         motion_data: Dict[str, torch.Tensor],
#         noise: Dict[str, torch.Tensor],
#         noise_level: torch.Tensor
#     ) -> Dict[str, torch.Tensor]:
#         if self.config['train'].get('turn_off_noise', False):
#             return motion_data
        
#         noisy_motion = {}
#         for key in ['theta', 'expression_embed']:
#             if key in motion_data:
#                 # Ensure scheduler on correct device
#                 self.scheduler = self.scheduler.to(motion_data[key].device)
                
#                 # Flatten for adding noise
#                 flat = motion_data[key].view(motion_data[key].shape[0], -1)
#                 noisy_flat = self.scheduler.add_noise(flat, noise[key].view_as(flat), noise_level)
#                 noisy_motion[key] = noisy_flat.view_as(motion_data[key])
        
#         return noisy_motion
    
#     def generate_sequence(
#         self,
#         initial_pose: Dict[str, torch.Tensor],
#         initial_dynamics: torch.Tensor,  # [B, expression_dim]
#         conditions: Dict[str, torch.Tensor],
#         num_steps: int = 50,
#         eta: float = 0.5,
#         cfg_scales: Optional[Dict[str, float]] = None
#     ) -> Dict[str, torch.Tensor]:
#         try:
#             B = initial_pose['theta'].shape[0]
#             total_T = conditions['audio_features'].shape[1]  # Assuming [B, T, D]
#             window_size = self.config['motion']['window_size']
#             stride = self.config['motion']['stride']
#             context_size = self.config['motion']['context_size']
            
#             # Initialize outputs
#             full_theta = torch.zeros(B, total_T, 3, 4, device=self.device)
#             full_expr = torch.zeros(B, total_T, self.motion_transformer.motion_dim, device=self.device)
            
#             # Initial context
#             prev_context = {
#                 'theta': self.start_prev_theta.repeat(B, 1, 1, 1),
#                 'expression': self.start_prev_expression.repeat(B, 1, 1),
#                 'audio': torch.zeros(B, context_size, 768, device=self.device)  # Assuming 768-dim audio
#             }
            
#             start_idx = 0
#             while start_idx < total_T:
#                 current_T = min(window_size, total_T - start_idx)
                
#                 # Slice conditions for current window
#                 window_conditions = {k: v[:, start_idx:start_idx + current_T] for k, v in conditions.items()}
                
#                 # Initialize noise
#                 noise = {
#                     'theta': torch.randn(B, current_T, 3, 4, device=self.device),
#                     'expression_embed': torch.randn(B, current_T, self.motion_transformer.motion_dim, device=self.device)
#                 }
                
#                 motion = {
#                     'theta': noise['theta'].clone(),
#                     'expression_embed': noise['expression_embed'].clone()
#                 }
                
#                 # DDIM sampling loop
#                 for t in reversed(range(num_steps)):
#                     timesteps = torch.full((B,), t, device=self.device, dtype=torch.long)
                    
#                     # Unconditional prediction
#                     uncond_motion = self.forward(
#                         motion,
#                         timesteps,
#                         conditions={k: torch.zeros_like(v) for k, v in window_conditions.items() if k != 'audio'},  # Keep audio
#                         prev_context=prev_context
#                     )
                    
#                     # Conditional prediction
#                     cond_motion = self.forward(
#                         motion,
#                         timesteps,
#                         conditions=window_conditions,
#                         prev_context=prev_context
#                     )
                    
#                     # Apply CFG
#                     if cfg_scales:
#                         pred_motion = {k: uncond_motion[k] + cfg_scales.get(k, 1.0) * (cond_motion[k] - uncond_motion[k])
#                                        for k in cond_motion}
#                     else:
#                         pred_motion = cond_motion
                    
#                     # Update motion
#                     for key in motion:
#                         model_output = pred_motion[key].view(B, -1)
#                         current_sample = motion[key].view(B, -1)
#                         motion[key] = self.scheduler.step(
#                             model_output=model_output,
#                             timestep=timesteps[0],
#                             sample=current_sample,
#                             eta=eta
#                         ).prev_sample.view_as(motion[key])
                
#                 # Store generated window
#                 full_theta[:, start_idx:start_idx + current_T] = pred_motion['theta']
#                 full_expr[:, start_idx:start_idx + current_T] = pred_motion['expression_embed']
                
#                 # Update context for next window
#                 context_start = max(0, current_T - context_size)
#                 prev_context['theta'] = pred_motion['theta'][:, context_start:]
#                 prev_context['expression'] = pred_motion['expression_embed'][:, context_start:]
#                 prev_context['audio'] = window_conditions['audio_features'][:, context_start:]
                
#                 start_idx += stride
                
#             return {
#                 'theta': full_theta,
#                 'expression_embed': full_expr
#             }
            
#         except Exception as e:
#             logger.error(f"Error in sequence generation: {str(e)}")
#             logger.error(traceback.format_exc())
#             return {
#                 'theta': torch.zeros(B, total_T, 3, 4, device=self.device),
#                 'expression_embed': torch.zeros(B, total_T, self.motion_transformer.motion_dim, device=self.device)
#             }
