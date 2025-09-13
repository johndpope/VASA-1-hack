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
        self.speed_proj = nn.Linear(1, 1)

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

            B, T = next(tensor.shape[:2] for tensor in conditions.values() if isinstance(tensor, torch.Tensor))
            logger.debug(f"Batch size: {B}, Sequence length: {T}")

            output = torch.zeros(B, T, self.model_dim, device=device, dtype=dtype)
            logger.debug(f"Initialized output tensor: shape={output.shape}")

            # Audio features are REQUIRED - dataset must provide them
            assert 'audio_features' in conditions, "audio_features must be present in conditions!"
            audio = conditions['audio_features']
            assert audio is not None, "audio_features cannot be None - check dataset!"

            if len(audio.shape) == 4:
                audio = audio.squeeze(1)
            audio = self._ensure_float_tensor(audio, dtype)

            audio_projected = self.audio_proj(audio)
            audio_normalized = self.audio_norm(audio_projected)

            gaze = self.gaze_proj(conditions.get('gaze', torch.zeros(B, T, 2, device=device)))
            distance = self.distance_proj(conditions.get('head_distance', torch.zeros(B, T, 1, device=device)))
            emotion = self.emotion_proj(conditions.get('emotion', torch.zeros(B, T, 2, device=device)))
            speed = self.speed_proj(conditions.get('speed_bucket', torch.zeros(B, T, 1, device=device)))

            controls = torch.cat([gaze, distance, emotion, speed], dim=-1)
            controls_normalized = self.control_norm(controls)

            # landmarks = []
            # for key in self.landmark_dims.keys():
            #     lm = conditions.get(key, torch.zeros(B, T, self.landmark_dims[key], device=device))
            #     landmarks.append(lm)
            # landmarks_combined = torch.cat(landmarks, dim=-1)
            # landmarks_normalized = self.landmark_norm(landmarks_combined)

            blink = conditions.get('blink_state', torch.zeros(B, T, 3, device=device))
            blink_embedded = self.blink_embed(blink)

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

        self.pose_proj = nn.Linear(self.d_model, 12)
        self.dyn_proj = nn.Linear(self.d_model, config.model.motion_dim)

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

        # Ensure noise_level is float tensor
        noise_level = noise_level.float() if noise_level.dtype != torch.float32 else noise_level

        noise_emb = self.noise_embed(noise_level.unsqueeze(-1)).unsqueeze(1).expand(B, T, -1)

        if cond_emb is None:
            cond_emb = self.cond_embed(conditions, prev_context)

        motion_flat = torch.cat([motion_data['theta'].view(B, T, -1), motion_data['expression_embed']], dim=-1)
        input_emb = motion_flat + cond_emb + noise_emb

        has_context = prev_context is not None
        input_emb = self.pos_embed(input_emb, has_context)

        transformer_out = self.transformer(input_emb.transpose(0, 1)).transpose(0, 1)

        outputs = {
            'theta': self.pose_proj(transformer_out).view(B, T, 3, 4),
            'expression_embed': self.dyn_proj(transformer_out)
        }

        outputs['rotation'] = outputs['theta'][:, :, :, :3].mean(dim=-2)
        outputs['translation'] = outputs['theta'][:, :, :, 3]
        outputs['scale'] = torch.ones(B, T, 3, device=device)

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

        expression_dim = config.model.motion_dim
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
                'speed_bucket': (B, T, 1),
                'lips': (B, T, 20, 3),
                'right_eye': (B, T, 8, 3),
                'left_eye': (B, T, 7, 3),
                'jaw': (B, T, 10, 3),
                'nose': (B, T, 4, 3),
                'blink_state': (B, T, 3)
            }

            for key, expected_shape in expected_shapes.items():
                if key not in conditions or conditions[key] is None:
                    validated_conditions[key] = torch.zeros(expected_shape, device=device)
                else:
                    tensor = conditions[key]
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
                'expression_embed': torch.zeros(B, total_T, self.config.model.motion_dim, device=device)
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
                    'expression_embed': torch.randn(B, current_T, self.config.model.motion_dim, device=device)
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




    # # overlapping windows for inference
    # def generate_sequence(
    #     self,
    #     initial_pose: Dict[str, torch.Tensor],
    #     initial_dynamics: torch.Tensor,
    #     conditions: Dict[str, torch.Tensor],
    #     num_steps: int = 50,
    #     eta: float = 0.0,  # DDIM stochasticity parameter
    #     cfg_scales: Optional[Dict[str, float]] = None
    # ) -> Dict[str, torch.Tensor]:
    #     """Generate sequence using DDIM sampling."""
    #     self.eval()
    #     with torch.no_grad():
    #         try:
    #             # Get batch size and sequence length from audio features
    #             audio_features = conditions['audio_features']
    #             B, T = audio_features.shape[:2]
    #             device = initial_pose['theta'].device

    #             logger.debug(f"\n=== Starting VASA Sequence Generation with DDIM ===")
    #             logger.debug(f"Batch size: {B}, Sequence length: {T}")

    #             # Set number of inference steps for scheduler
    #             self.scheduler.set_timesteps(num_steps, device=device)

    #             # Initialize motion sequence with random noise
    #             motion_sequence = {
    #                 'theta': torch.randn(B, T, 3, 4, device=device),
    #                 'scale': torch.randn(B, T, 3, device=device),
    #                 'rotation': torch.randn(B, T, 3, device=device),
    #                 'translation': torch.randn(B, T, 3, device=device),
    #                 'expression_embed': torch.randn(B, T, 128, device=device)
    #             }

    #             # Set initial frame values
    #             motion_sequence['theta'][:, 0] = initial_pose['theta']
    #             motion_sequence['rotation'][:, 0] = initial_pose['rotation']
    #             motion_sequence['scale'][:, 0] = initial_pose['scale']
    #             motion_sequence['translation'][:, 0] = initial_pose['translation']
    #             motion_sequence['expression_embed'][:, 0] = initial_dynamics

    #             # DDIM sampling loop
    #             for i, t in enumerate(self.scheduler.timesteps):
    #                 # Get model prediction
    #                 model_output = self.forward(
    #                     motion_data=motion_sequence,
    #                     noise_level=t.expand(B),
    #                     conditions=conditions
    #                 )

    #                 # DDIM step for each motion parameter
    #                 for key in motion_sequence.keys():
    #                     if key in model_output:
    #                         # Use scheduler step
    #                         scheduler_output = self.scheduler.step(
    #                             model_output=model_output[key],
    #                             timestep=t,
    #                             sample=motion_sequence[key],
    #                             eta=eta
    #                         )
    #                         motion_sequence[key] = scheduler_output.prev_sample

    #             return motion_sequence

    #         except Exception as e:
    #             logger.error(f"Error in sequence generation: {str(e)}")
    #             logger.error(traceback.format_exc())
    #             raise


