import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
from typing import *
from omegaconf import OmegaConf
from facenet_pytorch import InceptionResnetV1
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger, TorchDebugger
import traceback
import math
from memory_profiler import profile
from typing import Dict, Optional, List, Tuple, Generator
from mem import memory_stats, ModelCounter
import gc
from torchvision.utils import save_image
import os
import torchvision.transforms as transforms
from collections import defaultdict
from mem import TensorMemoryManager
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
# syncnet is in nemo directory
sys.path.insert(0, 'nemo') if 'nemo' not in sys.path else None
from syncnet import SyncNetInstance
import wandb
from video_tracker import VideoEventData, VideoEvent, ProblematicVideosTracker
import yaml
from pathlib import Path
from torch.utils.checkpoint import checkpoint 
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

class VASAPositionalEmbedding(nn.Module):
    """
    Positional embeddings for VASA sequence generation.
    Handles temporal positions and optional window context.
    """
    def __init__(
        self,
        d_model: int = 512,          # Transformer embedding dimension
        max_seq_len: int = 50,       # Maximum sequence length (T=50)
        max_context_len: int = 10,   # Maximum context length (K=10)
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Create standard sinusoidal embeddings
        pe = torch.zeros(max_seq_len + max_context_len, d_model)
        position = torch.arange(0, max_seq_len + max_context_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        # Sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Separate embeddings for context vs. main sequence
        self.context_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sequence_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        self.max_seq_len = max_seq_len
        self.max_context_len = max_context_len

    def forward(
        self, 
        x: torch.Tensor,
        has_context: bool = False
    ) -> torch.Tensor:
        """
        Add positional embeddings to input sequence.
        
        Args:
            x: Input tensor [B, T, D]
            has_context: Whether sequence includes context frames
            
        Returns:
            Tensor with positional information added
        """
        B, T, D = x.shape
        
        # Get positional encodings for this sequence length
        pos_encodings = self.pe[:, :T]
        
        # Add type embeddings based on whether position is context or main sequence
        if has_context:
            type_embeddings = torch.cat([
                self.context_embedding.expand(B, self.max_context_len, D),
                self.sequence_embedding.expand(B, T - self.max_context_len, D)
            ], dim=1)
        else:
            type_embeddings = self.sequence_embedding.expand(B, T, D)
            
        # Combine position encodings and type embeddings with input
        x = x + pos_encodings + type_embeddings
        
        return self.dropout(x)



class EfficientConditionEmbedding(nn.Module):
    def __init__(self, model_dim: int = 512, max_seq_len: int = 60):
        super().__init__()

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        logger.info(f"Initializing EfficientConditionEmbedding: model_dim={model_dim}, max_seq_len={max_seq_len}")

        # Load configuration
        config = self.load_channel_config('channel_config.yaml')

        # Set clipping bounds from config
        self.clip_min = config.model.clip_bounds.min
        self.clip_max = config.model.clip_bounds.max

        # Initialize blink handler (only once)
        self.blink_handler = BlinkConditionHandler(window_size=max_seq_len)

        # Build channel layout from config
        self.channel_layout = {}
        curr_idx = 0

        for key, channel_info in config.channel_layout.items():
            size = eval(str(channel_info.size)) if isinstance(channel_info.size, str) else channel_info.size
            self.channel_layout[key] = (curr_idx, curr_idx + size)
            curr_idx += size
            
        # Verify total dimension
        if curr_idx > self.model_dim:
            raise ValueError(f"Total feature dimension {curr_idx} exceeds model dimension {self.model_dim}")

        # Build landmark dimensions from config
        self.landmark_dims = {
            name: info.points * info.coords 
            for name, info in config.landmarks.items()
        }
        # Log which audio model is being used
        audio_input_dim = config.projections.audio.input_dim
        if audio_input_dim == 384:
            logger.info("Using Whisper audio features (384 dimensions)")
        elif audio_input_dim == 768:
            logger.info("Using Wav2Vec audio features (768 dimensions)")
        else:
            raise ValueError(f"Unexpected audio input dimension: {audio_input_dim}")

        # Initialize audio projection - separate for Whisper and Wav2Vec
        if audio_input_dim == 384:  # Whisper
            self.audio_proj = nn.Sequential(
                nn.Linear(384, config.projections.audio.output_dim)
            )
            logger.info("Using direct projection for Whisper features (384 dimensions)")
        else:  # Wav2Vec (768)
            self.audio_proj = nn.Sequential(
                nn.Linear(768, config.projections.audio.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.projections.audio.hidden_dim, config.projections.audio.output_dim)
            )
            logger.info("Using full projection for Wav2Vec features (768 dimensions)")
        
        # Control signal projections
        self.gaze_proj = nn.Linear(2, 2)
        self.distance_proj = nn.Linear(1, 1)
        self.emotion_proj = nn.Linear(2, 2)
        self.speed_proj = nn.Linear(1, 1)

        # Create landmark projections
        self.landmark_projections = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for key, dim in self.landmark_dims.items()
        })
            
        # Create landmark decoders
        self.landmark_decoders = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, dim)
            ) for key, dim in self.landmark_dims.items()
        })

        # Initialize all normalization layers
        total_landmark_dims = sum(self.landmark_dims.values())
        self.landmark_norm = nn.LayerNorm(total_landmark_dims)
        # self.motion_norm = nn.LayerNorm(config.projections.motion_norm_dim)
        self.audio_norm = nn.LayerNorm(config.projections.audio.output_dim)
        self.control_norm = nn.LayerNorm(config.projections.control_norm_dim)

        # Initialize blink embedding
        self.blink_embed = nn.Sequential(
            nn.Linear(config.projections.blink.input_dim, config.projections.blink.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.projections.blink.hidden_dim, config.projections.blink.output_dim)
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(model_dim)

    def load_channel_config(self,config_path: str) -> OmegaConf:
        """Load channel configuration from YAML file."""
        try:
            # Convert to Path object for better path handling
            config_path = Path(config_path)
            
            # Check if file exists
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
            # Load and parse YAML
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)
                
            # Convert to OmegaConf object
            config = OmegaConf.create(raw_config)
            
            return config
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise


    def get_predicted_landmarks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project transformer features to landmark predictions."""
        try:
            B, T = x.shape[:2]
            outputs = {}
            
            # Project each landmark group using decoders
            for name, decoder in self.landmark_decoders.items():
                # Project to flattened coordinates
                landmarks_flat = decoder(x)  # [B, T, N*3]
                
                # Reshape to [B, T, N, 3]
                num_points = self.landmark_dims[name] // 3
                landmarks = landmarks_flat.view(B, T, num_points, 3)
                
                # Store with correct name
                outputs[name] = landmarks
                
            return outputs
            
        except Exception as e:
            logger.error(f"Error projecting landmarks: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    def get_blink_states(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Get or generate blink states."""
        if not hasattr(self, 'blink_handler'):
            self.blink_handler = BlinkConditionHandler(window_size=T)
        
        blink_states = self.blink_handler.generate_blink_sequence(T)
        blink_states = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)
        return blink_states

    def _ensure_float_tensor(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert tensor to float with specified dtype."""
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            tensor = tensor.float()
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor
    
    def _process_landmarks(self, conditions: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Process facial landmark features with proper shape handling."""
        try:
            logger.debug("\nProcessing landmark features:")
            landmark_features = []
            
            # Get batch size and sequence length
            B = next(tensor.shape[0] for tensor in conditions.values() if isinstance(tensor, torch.Tensor))
            T = next(tensor.shape[1] for tensor in conditions.values() if isinstance(tensor, torch.Tensor))
            logger.debug(f"Batch size: {B}, Sequence length: {T}")

            # Process each landmark group
            for key, input_dim in self.landmark_dims.items():
                logger.debug(f"\nProcessing {key}:")
                
                if key in conditions and conditions[key] is not None:
                    landmarks = conditions[key]
                    logger.debug(f"  Found landmarks with shape: {landmarks.shape}")
                    
                    # Reshape [B, T, N, 3] -> [B, T, N*3]
                    landmarks_flat = landmarks.view(B, T, -1)
                    
                    # Verify dimension
                    if landmarks_flat.shape[-1] != input_dim:
                        logger.warning(
                            f"Dimension mismatch for {key}: got {landmarks_flat.shape[-1]}, "
                            f"expected {input_dim}"
                        )
                        landmarks_flat = torch.zeros(B, T, input_dim, device=device)
                else:
                    logger.debug(f"  No landmarks found, using zeros")
                    landmarks_flat = torch.zeros(B, T, input_dim, device=device)
                
                # Project landmarks - now dimensions should match
                projected = self.landmark_projections[key](landmarks_flat)
                logger.debug(f"  Projected shape: {projected.shape}")
                landmark_features.append(projected)

            # Combine all landmark features
            if landmark_features:
                combined = torch.cat(landmark_features, dim=-1)
                logger.debug(f"Combined landmark features shape: {combined.shape}")
                
                # Apply normalization
                normalized = self.landmark_norm(combined)
                logger.debug(f"Normalized landmark features shape: {normalized.shape}")
                
                return normalized
            else:
                # If no landmarks processed, return zero tensor
                total_dims = sum(self.landmark_dims.values())
                logger.warning("No valid landmarks found, returning zero tensor")
                return torch.zeros(B, T, total_dims, device=device)

        except Exception as e:
            logger.error(f"Error in landmark processing: {str(e)}")
            logger.error(traceback.format_exc())
            total_dims = sum(self.landmark_dims.values())
            return torch.zeros(B, T, total_dims, device=device)
        

    def _get_device_and_batch_size(self, conditions: Dict[str, torch.Tensor]) -> Tuple[torch.device, int]:
        """Safely get device and batch size from control signals first, then audio."""
        logger.debug("\nAttempting to get device and batch size...")
        
        # Try control signals first as they have consistent batch size
        for key in ['gaze', 'head_distance', 'emotion', 'speed_bucket']:
            if key in conditions and isinstance(conditions[key], torch.Tensor):
                tensor = conditions[key]
                logger.debug(f"Using '{key}' tensor: shape={tensor.shape}, device={tensor.device}")
                return tensor.device, tensor.shape[0]
        
        # Fallback to audio features
        if 'audio_features' in conditions:
            tensor = conditions['audio_features']
            if tensor.dim() == 4:  # [B, 1, T, D]
                B = tensor.shape[0]
            else:  # [1, T, D]
                B = 1
            logger.debug(f"Using 'audio_features' tensor: shape={tensor.shape}, device={tensor.device}")
            return tensor.device, B
        
        raise ValueError("No valid tensor found in conditions")

        

    def _place_control_embeddings(
        self,
        output: torch.Tensor,
        control_emb: torch.Tensor,
        B: int,
        T: int
    ) -> None:
        """Place control embeddings in output tensor with proper indexing."""
        try:
            logger.debug("\nPlacing control embeddings in output tensor:")
            # Gaze: first 2 channels
            start, end = self.channel_layout['gaze']
            gaze_range = control_emb[..., :2]
            logger.debug(f"  gaze: placing tensor {gaze_range.shape} at positions [{start}:{end}]")
            output[:, :, start:end] = gaze_range

            # Head distance: 1 channel
            start, end = self.channel_layout['head_distance']
            dist_range = control_emb[..., 2:3]
            logger.debug(f"  head_distance: placing tensor {dist_range.shape} at positions [{start}:{end}]")
            output[:, :, start:end] = dist_range

            # Emotion: 2 channels
            start, end = self.channel_layout['emotion']
            emotion_range = control_emb[..., 3:5]
            logger.debug(f"  emotion: placing tensor {emotion_range.shape} at positions [{start}:{end}]")
            output[:, :, start:end] = emotion_range

            # Speed bucket: 1 channel
            start, end = self.channel_layout['speed_bucket']
            speed_range = control_emb[..., 5:6]
            logger.debug(f"  speed_bucket: placing tensor {speed_range.shape} at positions [{start}:{end}]")
            output[:, :, start:end] = speed_range

        except Exception as e:
            logger.error(f"Error placing control embeddings: {str(e)}")
            logger.error("Debug info:")
            logger.error(f"  Output tensor shape: {output.shape}")
            logger.error(f"  Control embedding shape: {control_emb.shape}")
            logger.error(f"  B={B}, T={T}")
            raise

    def forward(
        self,
        conditions: Dict[str, torch.Tensor],
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Process conditions with proper dimension handling."""
        try:
            logger.debug("\n=== Starting EfficientConditionEmbedding Forward Pass ===")
            
            # Get device and dtype
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            logger.debug(f"Using device: {device}, dtype: {dtype}")
            
            # Get batch size and sequence length
            B, T = next(tensor.shape[:2] for tensor in conditions.values() if isinstance(tensor, torch.Tensor))
            logger.debug(f"Batch size: {B}, Sequence length: {T}")
            
            # Initialize output tensor
            output = torch.zeros(B, T, self.model_dim, device=device, dtype=dtype)
            logger.debug(f"Initialized output tensor: shape={output.shape}")

            # 1. Process audio features if present
            if 'audio_features' in conditions:
                audio = conditions['audio_features']

                # Handle audio features shape - expected [B, 1, T, D] or [B, T, D]
                if len(audio.shape) == 4:  # [B, 1, T, D]
                    B, _, T, D = audio.shape
                    # Remove channel dimension
                    audio = audio.squeeze(1)  # -> [B, T, D]
                else:  # [B, T, D]
                    B, T, D = audio.shape
                    
                logger.debug(f"Batch size: {B}, Sequence length: {T}")
                
                # if audio.dim() == 4:  # [B, 1, T, D]
                #     audio = audio.squeeze(1)
                audio = self._ensure_float_tensor(audio, dtype)
                audio_emb = self.audio_proj(audio)
                audio_emb = self.audio_norm(audio_emb)
                start, end = self.channel_layout['audio_features']
                output[..., start:end] = audio_emb
                logger.debug(f"Processed audio features shape: {audio_emb.shape}")

            # 2. Process control signals
            control_tensors = []
            control_configs = [
                ('gaze', self.gaze_proj, 2),
                ('head_distance', self.distance_proj, 1),
                ('emotion', self.emotion_proj, 2),
                ('speed_bucket', self.speed_proj, 1)
            ]
            
            for name, proj, feat_size in control_configs:
                start, end = self.channel_layout[name]
                if name in conditions and conditions[name] is not None:
                    value = self._ensure_float_tensor(conditions[name], dtype)
                    value = proj(value)
                    control_tensors.append(value)
                    logger.debug(f"Processed {name} shape: {value.shape}")
                else:
                    # Initialize with zeros
                    zero_tensor = torch.zeros(B, T, feat_size, device=device, dtype=dtype)
                    control_tensors.append(zero_tensor)
                    logger.debug(f"Using zero tensor for {name}: shape={zero_tensor.shape}")

            # Combine and normalize control signals
            control_combined = torch.cat(control_tensors, dim=-1)
            control_emb = self.control_norm(control_combined)
            logger.debug(f"Combined control embedding shape: {control_emb.shape}")
            
            # Place control embeddings
            control_ranges = [
                ('gaze', slice(0, 2)),
                ('head_distance', slice(2, 3)),
                ('emotion', slice(3, 5)),
                ('speed_bucket', slice(5, 6))
            ]
            
            for name, range_slice in control_ranges:
                start, end = self.channel_layout[name]
                output[..., start:end] = control_emb[..., range_slice]

            # 3. Process blink states if present
            if 'blink_state' in conditions:
                blink_emb = self.blink_embed(conditions['blink_state'])
            else:
                # Generate default blink sequence
                blink_handler = BlinkConditionHandler()
                blink_states = blink_handler.generate_blink_sequence(T)
                blink_states = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)
                blink_emb = self.blink_embed(blink_states)
            
            start, end = self.channel_layout['blink_state']
            output[..., start:end] = blink_emb

            # 4. Process landmarks
            landmark_emb = self._process_landmarks(conditions, device)
            if landmark_emb is not None:
                curr_idx = 0
                for key, dim in self.landmark_dims.items():
                    start, end = self.channel_layout[key]
                    feat_size = end - start
                    
                    logger.debug(f"\nPlacing {key} landmarks:")
                    logger.debug(f"  Channel range: {start}:{end} (size={feat_size})")
                    logger.debug(f"  Feature index: {curr_idx}:{curr_idx+feat_size}")
                    
                    curr_emb = landmark_emb[..., curr_idx:curr_idx+feat_size]
                    curr_idx += feat_size
                    
                    if curr_emb.shape[-1] != (end - start):
                        logger.error(
                            f"Shape mismatch for {key}: got {curr_emb.shape}, "
                            f"expected [..., {end-start}]"
                        )
                        continue
                        
                    output[..., start:end] = curr_emb

            # 5. Previous context if available
            if prev_context is not None:
                logger.debug("\nProcessing previous context...")
                prev_keys = [
                    ('theta', 12), ('rotation', 3),  ('scale', 3), 
                    ('translation', 3), ('expression', self.prev_expression_dim),
                    ('audio', self.prev_audio_dim)
                ]
                
                for name, dim in prev_keys:
                    context_key = f'prev_{name}'
                    if context_key not in self.channel_layout:
                        continue
                        
                    start, end = self.channel_layout[context_key]
                    if context_key in prev_context:
                        # Reshape if needed and place in output
                        context_value = prev_context[context_key]
                        if name == 'theta':
                            context_value = context_value.view(B, -1)  # Flatten 3x4
                        if context_value.shape[-1] != dim:
                            logger.warning(
                                f"Context dimension mismatch for {name}: "
                                f"got {context_value.shape[-1]}, expected {dim}"
                            )
                            context_value = torch.zeros(B, dim, device=device)
                        output[..., start:end] = context_value
                    else:
                        # Fill with zeros if missing
                        output[..., start:end] = torch.zeros(B, dim, device=device)

            # 6. Final processing
            # Clip to prevent extreme values
            output = torch.clamp(output, self.clip_min, self.clip_max)
            
            # Apply final layer norm
            output = self.final_norm(output)
            
            # Verify output is finite
            if not torch.isfinite(output).all():
                logger.error("Non-finite values detected in output!")
                logger.error(f"Output stats: min={output.min()}, max={output.max()}")
                # Replace non-finite values with zeros
                output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
            
            logger.debug(f"Final output shape: {output.shape}")
            return output

        except Exception as e:
            logger.error(f"Error in condition embedding: {str(e)}")
            logger.error(traceback.format_exc())
            logger.error("\nTensor shapes:")
            for k, v in conditions.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            raise

    def _ensure_float_tensor(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert tensor to float with specified dtype."""
        if tensor.dtype in [torch.int32, torch.int64, torch.long]:
            tensor = tensor.float()
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

class MotionResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x + residual

class MotionProjections(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            MotionResidualBlock(hidden_dim),
            MotionResidualBlock(hidden_dim)
        )
        
        # Initialize parameters with small values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.projection(x)



class HolisticMotionTransformer(nn.Module):
    def __init__(self, config, channels, depth, height, width):
        super().__init__()
        
        self.window_size = config.motion.window_size
        self.context_size = config.motion.context_size
        
        # Embedding dimensions
        self.transformer_dim = 512
        self.num_heads = 8
        self.num_layers = 8
        
        # Initialize condition embedding
        self.cond_embed = EfficientConditionEmbedding(
            model_dim=self.transformer_dim,
            max_seq_len=self.window_size
        )
        
        # Motion input projections with explicit batch and sequence handling
        self.motion_projections = nn.ModuleDict({
            'theta': MotionProjections(12, 12), # flattened 3x4 matrix
            'scale': MotionProjections(3, 3),
            'rotation': MotionProjections(3, 3),
            'translation': MotionProjections(3, 3),
            'expression': MotionProjections(128, 128)
        })
        # Combine motion projections
       # Combine motion projections to match transformer dim
        input_dim = 12 + 3 + 3 + 3 + 128  # Sum of individual feature dimensions
        self.motion_combine = nn.Sequential(
            nn.Linear(input_dim, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim)
        )
        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=self.transformer_dim,
                num_heads=self.num_heads,
                mlp_ratio=4,
                dropout=0.1
            ) for _ in range(self.num_layers)
        ])
        
        # Output projections
        self.output_projections = nn.ModuleDict({
            'theta': nn.Linear(self.transformer_dim, 12),
            'scale': nn.Linear(self.transformer_dim, 3),
            'rotation': nn.Linear(self.transformer_dim, 3),
            'translation': nn.Linear(self.transformer_dim, 3),
            'expression': nn.Linear(self.transformer_dim, 128)
        })
        
        self.gradient_checkpointing = True

    def _validate_context(self, prev_context: Optional[Dict[str, torch.Tensor]]) -> None:
        """Validate that prev_context contains exactly 10 frames."""
        if prev_context is not None:
            expected_keys = ['theta', 'rotation', 'translation', 'expression_embed']
            for key in expected_keys:
                assert key in prev_context, f"Missing {key} in prev_context"
                assert prev_context[key].dim() == 3, f"Expected 3D tensor for {key}, got {prev_context[key].dim()}D"
                assert prev_context[key].shape[1] == 10, (
                    f"Expected exactly 10 context frames for {key}, "
                    f"got {prev_context[key].shape[1]} frames. "
                    f"Full shape: {prev_context[key].shape}"
                )


    def _validate_conditions(
        self,
        conditions: Dict[str, Optional[torch.Tensor]],
        B: int,
        T: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Validate and process input conditions, handling None values."""
        validated = {}
        
        # Define expected shapes for each condition - now with correct sequence length T
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
        
        # Process each condition
        for name, expected_shape in expected_shapes.items():
            tensor = conditions.get(name)
            
            if tensor is None:
                # Create zero tensor with expected shape
                validated[name] = torch.zeros(expected_shape, device=device)
            else:
                # Validate shape and batch/sequence dimensions
                tensor = tensor.to(device)
                
                # Handle potential shape mismatches
                if tensor.shape != expected_shape:
                    # If tensor is missing sequence dimension, expand it
                    if len(tensor.shape) == len(expected_shape) - 1:
                        tensor = tensor.unsqueeze(1).expand(-1, T, *tensor.shape[1:])
                    # If tensor has batch size 1, expand if needed
                    elif tensor.shape[0] == 1 and B > 1:
                        tensor = tensor.expand(B, *tensor.shape[1:])
                        
                    # If still doesn't match, pad or truncate feature dimensions
                    if tensor.shape != expected_shape:
                        logger.warning(
                            f"Reshaping {name} from {tensor.shape} to {expected_shape}"
                        )
                        # Create zero tensor and copy what we can
                        reshaped = torch.zeros(expected_shape, device=device)
                        # Copy matching dimensions
                        min_dims = [min(s1, s2) for s1, s2 in zip(tensor.shape, expected_shape)]
                        slices = tuple(slice(0, d) for d in min_dims)
                        reshaped[slices] = tensor[slices]
                        tensor = reshaped
                        
                validated[name] = tensor

        # Handle audio features if present (special case due to varying input dims)
        if 'audio_features' in conditions and conditions['audio_features'] is not None:
            audio = conditions['audio_features']
            if len(audio.shape) == 4:  # [B, 1, T, D]
                validated['audio_features'] = audio.squeeze(1)
            else:  # [B, T, D]
                validated['audio_features'] = audio
        else:
            # Default audio features dimension (assuming Wav2Vec)
            validated['audio_features'] = torch.zeros(B, T, 768, device=device)
                
        return validated



    # def forward(self, motion_data, noise_level, cond_emb=None, conditions=None, prev_context=None):
    #     try:
    #         if self.training and self.gradient_checkpointing:
    #             # Check input tensors for inf/nan before checkpointing
    #             for key, tensor in motion_data.items():
    #                 if not torch.isfinite(tensor).all():
    #                     logger.error(f"Non-finite values in input {key}: min={tensor.min()}, max={tensor.max()}")
    #                     # Replace inf/nan with clamped values
    #                     tensor = torch.nan_to_num(tensor, 
    #                                             nan=0.0,
    #                                             posinf=100.0, 
    #                                             neginf=-100.0)
    #                     motion_data[key] = tensor

    #             # Check noise level
    #             if not torch.isfinite(noise_level).all():
    #                 logger.error(f"Non-finite values in noise_level: min={noise_level.min()}, max={noise_level.max()}")
    #                 noise_level = torch.nan_to_num(noise_level, nan=0.0, posinf=1.0, neginf=0.0)

    #             # Unpack dictionaries to tensors
    #             motion_tensors = [
    #                 motion_data['theta'],
    #                 motion_data['scale'], 
    #                 motion_data['rotation'],
    #                 motion_data['translation'],
    #                 motion_data['expression_embed']
    #             ]

    #             def run_function(*tensors):
    #                 # Check tensors after unpacking
    #                 for i, tensor in enumerate(tensors):
    #                     if not torch.isfinite(tensor).all():
    #                         logger.error(f"Non-finite values in tensor {i}: min={tensor.min()}, max={tensor.max()}")
    #                         # Replace inf/nan
    #                         tensor = torch.nan_to_num(tensor,
    #                                                 nan=0.0,
    #                                                 posinf=100.0,
    #                                                 neginf=-100.0)
    #                         tensors[i] = tensor

    #                 # Reconstruct dictionary 
    #                 theta, scale, rotation, translation, expression = tensors
                    
    #                 md = {
    #                     'theta': theta,
    #                     'scale': scale,
    #                     'rotation': rotation, 
    #                     'translation': translation,
    #                     'expression_embed': expression
    #                 }

    #                 output = self.custom_forward(md, noise_level, cond_emb, conditions, prev_context)

    #                 # Check outputs before returning
    #                 for key, tensor in output.items():
    #                     if not torch.isfinite(tensor).all():
    #                         logger.error(f"Non-finite values in output {key}: min={tensor.min()}, max={tensor.max()}")
    #                         output[key] = torch.nan_to_num(tensor,
    #                                                     nan=0.0,
    #                                                     posinf=100.0,
    #                                                     neginf=-100.0)

    #                 return output

    #             return checkpoint(run_function, *motion_tensors)
    #         else:
    #             return self.custom_forward(motion_data, noise_level, cond_emb, conditions, prev_context)

    #     except Exception as e:
    #         logger.error(f"Error in transformer forward: {str(e)}")
    #         logger.error(traceback.format_exc())
    #         raise
                        
    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor, 
        cond_emb: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with explicit dimension handling.
        
        Args:
            motion_data: Dictionary containing:
                - theta: [B, T, 3, 4]
                - rotation: [B, T, 3]
                - scale: [B, T, 3]
                - translation: [B, T, 3]
                - expression_embed: [B, T, 128]
            noise_level: Noise schedule timesteps [B]
            cond_emb: Optional pre-computed condition embeddings [B, T, D]
            conditions: Optional condition signals dictionary
            prev_context: Optional previous window context
            
        Returns:
            Dictionary of transformed motion parameters
        """
        
        try:
            # Get batch and sequence dimensions from theta
            B, T = motion_data['theta'].shape[:2]
            logger.debug(f"\nBatch size: {B}, Sequence length: {T}")

            # Validate input shapes with explicit assertions
            assert motion_data['theta'].shape == (B, T, 3, 4), f"Expected theta shape [B,T,3,4], got {motion_data['theta'].shape}"
            assert motion_data['rotation'].shape == (B, T, 3), f"Expected rotation shape [B,T,3], got {motion_data['rotation'].shape}"
            assert motion_data['scale'].shape == (B, T, 3), f"Expected scale shape [B,T,3], got {motion_data['scale'].shape}"
            assert motion_data['translation'].shape == (B, T, 3), f"Expected translation shape [B,T,3], got {motion_data['translation'].shape}" 
            assert motion_data['expression_embed'].shape == (B, T, 128), f"Expected expression shape [B,T,128], got {motion_data['expression_embed'].shape}"
            assert noise_level.shape == (B,), f"Expected noise_level shape [B], got {noise_level.shape}"

            self._validate_context(prev_context)

            self._validate_conditions(conditions,B,T,motion_data['theta'].device)

            # Handle condition embeddings with validation
            if cond_emb is None and conditions is not None:
                logger.debug("\nComputing condition embeddings...")
                
                 # Ensure blink state handling
                if 'blink_state' not in conditions or conditions['blink_state'] is None:
                    # Generate default blink sequence using handler
                    blink_states = self.cond_embed.blink_handler.generate_blink_sequence(T)
                    conditions['blink_state'] = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)
                    
                # Pre-validate condition shapes with explicit assertions
                if 'audio_features' in conditions:
                    audio = conditions['audio_features']
                    if len(audio.shape) == 4:  # [B, 1, T, D]
                        assert audio.shape[0] == B and audio.shape[2] == T, \
                            f"Wrong audio shape: expected [B={B}, 1, T={T}, D], got {audio.shape}"
                        assert audio.shape[1] == 1, f"Expected audio to have channel dim of 1, got {audio.shape[1]}"
                    else:
                        assert audio.shape[:2] == (B, T), \
                            f"Wrong audio shape: expected [B={B}, T={T}, D], got {audio.shape}"

                # Validate other condition shapes
                for key in ['gaze', 'head_distance', 'emotion', 'speed_bucket']:
                    if key in conditions:
                        tensor = conditions[key]
                        assert tensor.shape[:2] == (B, T), \
                            f"Wrong {key} shape: expected prefix [B={B}, T={T}], got {tensor.shape}"
                        
                        # Additional dimension checks
                        if key == 'gaze':
                            assert tensor.shape[2] == 2, f"Expected gaze to have 2 dimensions, got {tensor.shape[2]}"
                        elif key == 'head_distance':
                            assert tensor.shape[2] == 1, f"Expected head_distance to have 1 dimension, got {tensor.shape[2]}"
                        elif key == 'emotion':
                            assert tensor.shape[2] == 2, f"Expected emotion to have 2 dimensions, got {tensor.shape[2]}"
                        elif key == 'speed_bucket':
                            assert tensor.shape[2] == 1, f"Expected speed_bucket to have 1 dimension, got {tensor.shape[2]}"

                # Compute embeddings with validated inputs
                cond_emb = self.cond_embed(conditions, prev_context)
                
                # Validate condition embedding shape with explicit assertion
                assert cond_emb.shape == (B, T, self.transformer_dim), \
                    f"Wrong condition embedding shape: got {cond_emb.shape}, expected ({B}, {T}, {self.transformer_dim})"
                
                # Additional validations for the condition embedding tensor
                assert torch.isfinite(cond_emb).all(), "Non-finite values detected in condition embeddings"
                assert not torch.isnan(cond_emb).any(), "NaN values detected in condition embeddings"
                logger.debug(f"Computed condition embeddings shape: {cond_emb.shape}")

            # Project motion parameters with shape validation
            logger.debug("\nProjecting motion parameters...")
            motion_features = self._project_motion_parameters(motion_data, B, T)
            logger.debug(f"Motion features shape: {motion_features.shape}")

            # Add time embeddings
            time_emb = self._time_embedding(noise_level, self.transformer_dim)
            x = motion_features + time_emb.unsqueeze(1)

            # Add condition embeddings if present
            if cond_emb is not None:
                x = x + cond_emb
                logger.debug("Added condition embeddings")

            # Process through transformer blocks
            logger.debug("\nProcessing through transformer blocks...")
            for i, block in enumerate(self.transformer):
                x = block(x)
                if not torch.isfinite(x).all():
                    raise ValueError(f"Non-finite values detected after transformer block {i}")
                logger.debug(f"Block {i} output shape: {x.shape}")

            # Project outputs back to motion parameters
            logger.debug("\nProjecting outputs...")
            outputs = self._project_outputs(x, B, T)
            
            return outputs

        except Exception as e:
            logger.error(f"Error in transformer forward: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _project_motion_parameters(self, motion_data, B, T):
        """Enhanced motion parameter projection with residual connections."""
        try:
            device = motion_data['theta'].device
            dtype = motion_data['theta'].dtype
            
            # Process theta matrix
            theta = motion_data['theta'].view(B, T, -1)  # [B, T, 12]
            theta = torch.clamp(theta, -100.0, 100.0)
            theta = self.motion_projections['theta'](theta)  # Now includes residual blocks
            
            # Process rotation with residual path
            rotation = motion_data['rotation']  # [B, T, 3]
            rotation = torch.remainder(rotation + torch.pi, 2 * torch.pi) - torch.pi
            rotation = self.motion_projections['rotation'](rotation)
            
            # Process translation with residual path
            translation = motion_data['translation']  # [B, T, 3]
            translation = torch.clamp(translation, -10.0, 10.0)
            translation = self.motion_projections['translation'](translation)
            
            # Process expression with residual path
            expression = motion_data['expression_embed']  # [B, T, 128]
            expression = torch.clamp(expression, -10.0, 10.0)
            expression = self.motion_projections['expression'](expression)



            # scale
            scale = motion_data['scale']  # [B, T, 3]
            scale = torch.clamp(scale, -10.0, 10.0)
            scale = self.motion_projections['scale'](scale)
            # Combine features with residual connection
            features = torch.cat([theta,scale, rotation, translation, expression], dim=-1)
            motion_features = self.motion_combine(features)  # Includes residual block
            
            # Final validations
            motion_features = torch.nan_to_num(
                motion_features,
                nan=0.0,
                posinf=100.0,
                neginf=-100.0
            )
            
            assert motion_features.shape == (B, T, self.transformer_dim), \
                f"Wrong shape: {motion_features.shape}, expected ({B}, {T}, {self.transformer_dim})"
            assert torch.isfinite(motion_features).all(), "Non-finite values in features"
            
            return motion_features

        except Exception as e:
            logger.error(f"Error in motion parameter projection: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.zeros((B, T, self.transformer_dim), device=device, dtype=dtype)



    def _project_outputs(
        self,
        x: torch.Tensor,
        B: int,
        T: int
    ) -> Dict[str, torch.Tensor]:
        """Project transformer outputs back to motion parameters with validation."""
        try:
            outputs = {}

            # Project theta
            theta_flat = self.output_projections['theta'](x)  # [B, T, 12]
            outputs['theta'] = theta_flat.view(B, T, 3, 4)
            
            landmark_outputs = self.cond_embed.get_predicted_landmarks(x)
            outputs.update(landmark_outputs)

            # Project other parameters
            outputs['rotation'] = self.output_projections['rotation'](x)
            outputs['scale'] = self.output_projections['scale'](x)
            outputs['translation'] = self.output_projections['translation'](x)
            outputs['expression_embed'] = self.output_projections['expression'](x)
      
            # Validate shapes
            expected_shapes = {
                'theta': (B, T, 3, 4),
                'rotation': (B, T, 3),
                'scale': (B, T, 3),
                'translation': (B, T, 3),
                'expression_embed': (B, T, 128)
            }

            for key, expected_shape in expected_shapes.items():
                if outputs[key].shape != expected_shape:
                    raise ValueError(f"Wrong shape for {key}: expected {expected_shape}, got {outputs[key].shape}")

            return outputs

        except Exception as e:
            logger.error(f"Error projecting outputs: {str(e)}")
            raise

    def _time_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
 


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        
        # Pre-norm layers
        self.pre_norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # MoH Attention

        from mohattention import MoHAttention
        shared_heads = 2  # ~40% of 6 activated heads (75% of 8)
        routed_heads = 4  # ~60% of 6 activated heads
        total_activated = shared_heads + routed_heads  # 6 heads = 75% of 8 total

        self.attn = MoHAttention(
            dim=dim,
            num_heads=8,  # Total heads
            shared_head=shared_heads,  # 2 shared heads
            routed_head=routed_heads,  # 4 routed heads
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # GELU for better stability
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.batch_sequence_norm = nn.LayerNorm(dim, eps=1e-6)

        # Initialize parameters
        self._init_weights()
        
    def _batch_sequence_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization across both batch and sequence dimensions"""
        B, T, D = x.shape
        # Merge batch and sequence
        x = x.view(-1, D)
        x = self.batch_sequence_norm(x)
        # Restore shape
        return x.view(B, T, D)

    def _init_weights(self):
        # Initialize linear layers with small values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Change the current forward pass:
        
        # Pre-norm attention with residual and new norm
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x)
        x = residual + self._batch_sequence_norm(self.dropout(attn_out))  # Add norm here
        
        # Pre-norm MLP with residual and new norm
        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + self._batch_sequence_norm(self.dropout(mlp_out))  # Add norm here
        
        return x

class VASAModel(nn.Module):
    """VASA model for motion sequence generation."""
    def __init__(
        self,
        config: dict,
        volumetric_avatar: nn.Module,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Model dimensions
        self.num_steps = config.diffusion.num_steps
        self.window_size = config.motion.window_size
        self.context_size = config.motion.context_size
        self.stride = config.motion.stride
        self.overlap_size = self.window_size - self.stride

        # Stage 1 volumetric model (frozen)
        self.volumetric_avatar = volumetric_avatar.eval()
        for param in self.volumetric_avatar.parameters():
            param.requires_grad = False

        # Initialize motion transformer
        self.motion_transformer = HolisticMotionTransformer(
            config=config,
            channels=self.volumetric_avatar.args.latent_volume_channels, # DELETE THESE
            depth=self.volumetric_avatar.args.latent_volume_depth,
            height=self.volumetric_avatar.args.latent_volume_size,
            width=self.volumetric_avatar.args.latent_volume_size,
        )

        # Initialize diffusion parameters
        self._init_diffusion_params(
            num_steps=config.diffusion.num_steps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )


    def _init_diffusion_params(self, num_steps: int, beta_start: float, beta_end: float):
        """Initialize diffusion schedule parameters"""
        # Initialize DDIM scheduler
        from diffusers import DDIMScheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=True,
            #  clip_sample_range=1.0,  # Add this for motion parameters
            prediction_type="epsilon",  # Important to match paper's formulation 
            timestep_spacing="leading"  # Important for proper timestep spacing
        )
        # Set default inference steps
        self.scheduler.set_timesteps(num_steps)
        
    def forward(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None,
        noise: Optional[Dict[str, torch.Tensor]] = None # Noise tensor for diffusion

    ) -> Dict[str, torch.Tensor]:
        self.scheduler.config.prediction_type = "sample"  # Add this line
        try:
            # Get batch and sequence dimensions from theta
            B, T = motion_data['theta'].shape[:2]
            device = motion_data['theta'].device
            logger.debug(f"\n=== VASAModel Forward Pass Start ===")
            logger.debug(f"Batch size: {B}, Sequence length: {T}, Device: {device}")
            

            # Debug input conditions
            logger.debug("\nInput conditions:")
            if conditions is not None:
                for k, v in conditions.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")
                    else:
                        logger.debug(f"  {k}: None or not tensor")
            else:
                logger.debug("  No conditions provided")

            # Validate conditions 
            logger.debug("\nValidating conditions:")
            validated_conditions = {}
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

            if conditions is not None:
                # Log landmark mapping 
                logger.debug("\nMapping landmarks:")
                landmark_mapping = {
                    'lips_landmarks': 'lips',
                    'right_eye_landmarks': 'right_eye',
                    'left_eye_landmarks': 'left_eye', 
                    'jaw_landmarks': 'jaw',
                    'nose_landmarks': 'nose'
                }
                for old_key, new_key in landmark_mapping.items():
                    logger.debug(f"  {old_key} -> {new_key}")

                # Map conditions
                mapped_conditions = {}
                for k, v in conditions.items():
                    new_key = landmark_mapping.get(k, k)
                    mapped_conditions[new_key] = v
                    logger.debug(f"  Mapped {k} -> {new_key}")
                conditions = mapped_conditions

                # Process each expected condition
                for key, expected_shape in expected_shapes.items():
                    logger.debug(f"\nProcessing {key}:")
                    if key not in conditions or conditions[key] is None:
                        logger.debug(f"  Creating zero tensor with shape {expected_shape}")
                        validated_conditions[key] = torch.zeros(expected_shape, device=device)
                    else:
                        tensor = conditions[key]
                        logger.debug(f"  Input shape: {tensor.shape}")
                        
                        # Shape validation and expansion
                        if tensor.shape[:2] != (B, T):
                            logger.debug(f"  Reshaping to match batch/sequence dims")
                            if len(tensor.shape) == 2:  # [B, D]
                                tensor = tensor.unsqueeze(1).expand(-1, T, -1)
                            elif len(tensor.shape) == 3 and tensor.shape[0] == 1:  # [1, T, D]
                                tensor = tensor.expand(B, -1, -1)
                            logger.debug(f"  After reshape: {tensor.shape}")

                        validated_conditions[key] = tensor

                # Generate blink sequence if missing
                if 'blink_state' not in validated_conditions:
                    logger.debug("\nGenerating default blink sequence")
                    blink_handler = BlinkConditionHandler(window_size=T)
                    blink_states = blink_handler.generate_blink_sequence(T)
                    validated_conditions['blink_state'] = blink_states.unsqueeze(0).expand(B, -1, -1).to(device)
                    logger.debug(f"  Generated blink shape: {validated_conditions['blink_state'].shape}")

            logger.debug("\nFinal validated conditions:")
            for k, v in validated_conditions.items():
                logger.debug(f"  {k}: shape={v.shape}")

            # Process conditions through transformer
            outputs = self.motion_transformer(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=validated_conditions if cond_emb is None else None,
                cond_emb=cond_emb,
                prev_context=prev_context
            )

            if noise is not None:
                outputs['noise'] = noise 

            return outputs

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        

    
   


    def forward_with_cfg(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        cfg_scales: Optional[Dict[str, float]] = None,
        drop_conditions: Optional[List[str]] = None,
        num_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequence with classifier-free guidance, with conditional dropping.
        
        Args:
            motion_data: Input motion parameters
            noise_level: Current noise level
            conditions: Conditioning signals
            cfg_scales: Dictionary mapping condition names to their CFG scales
            drop_conditions: List of condition names to completely drop/ignore
            num_steps: Number of diffusion steps
        """
        try:
            # Get device and dimensions
            device = next(iter(motion_data.values())).device
            B, T = next(iter(motion_data.values())).shape[:2]
            
            logger.debug("\n=== Running CFG Generation ===")
            logger.debug(f"CFG scales: {cfg_scales}")
            logger.debug(f"Dropped conditions: {drop_conditions}")

            # If no CFG, just do regular forward pass
            if not cfg_scales:
                return self.forward(
                    motion_data=motion_data,
                    noise_level=noise_level,
                    conditions=conditions
                )

            # Filter out dropped conditions
            if drop_conditions:
                filtered_conditions = {
                    k: v for k, v in conditions.items() 
                    if k not in drop_conditions
                }
            else:
                filtered_conditions = conditions

            # Create unconditional input by removing specific conditions
            uncond_conditions = self._create_empty_conditions(
                filtered_conditions, device
            )

            logger.debug("\nGenerating conditional sequence...")
            logger.debug(f"Active conditions: {list(filtered_conditions.keys())}")
            
            cond_output = self.generate_sequence(
                initial_pose=motion_data,
                initial_dynamics=motion_data['expression_embed'][:, 0],
                conditions=filtered_conditions,
                num_steps=num_steps
            )
            
            logger.debug("\nGenerating unconditional sequence...")
            uncond_output = self.generate_sequence(
                initial_pose=motion_data,
                initial_dynamics=motion_data['expression_embed'][:, 0],
                conditions=uncond_conditions,
                num_steps=num_steps
            )

            # Apply CFG selectively
            output = {}
            logger.debug("\nApplying CFG scaling:")
            
            for key in cond_output.keys():
                try:
                    # Skip CFG for dropped conditions
                    if drop_conditions and key in drop_conditions:
                        output[key] = cond_output[key]
                        logger.debug(f"  {key}: skipped (dropped)")
                        continue

                    # Get scale for this parameter
                    scale = cfg_scales.get(key, 1.0)
                    logger.debug(f"  {key}: scale={scale}")

                    if key not in uncond_output:
                        output[key] = cond_output[key]
                        continue

                    # Apply CFG with validation
                    cond_tensor = cond_output[key]
                    uncond_tensor = uncond_output[key]

                    if cond_tensor.shape != uncond_tensor.shape:
                        output[key] = cond_tensor
                        continue

                    # Apply scaled difference
                    output[key] = uncond_tensor + scale * (cond_tensor - uncond_tensor)

                    # Ensure finite values
                    if not torch.isfinite(output[key]).all():
                        output[key] = torch.nan_to_num(
                            output[key],
                            nan=0.0,
                            posinf=1e6,
                            neginf=-1e6
                        )

                except Exception as e:
                    logger.error(f"Error processing {key}: {str(e)}")
                    output[key] = cond_output[key]

            return output

        except Exception as e:
            logger.error(f"Error in CFG generation: {str(e)}")
            logger.error(traceback.format_exc())
            return self.forward(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=conditions
            )
            
    def _create_empty_conditions(
        self,
        filled_conditions: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Create empty/zero conditions dictionary matching the shape of filled conditions.
        Used for classifier-free guidance to create unconditional samples.
        
        Args:
            filled_conditions: Dictionary of original condition tensors
            device: Target device for tensors
            
        Returns:
            Dictionary of zero-filled condition tensors
        """
        try:
            logger.debug("\n=== Creating Empty Conditions ===")
            empty_conditions = {}
            
            # Get shapes from filled conditions
            for key, tensor in filled_conditions.items():
                if tensor is None:
                    continue
                    
                # Get shape and create zero tensor
                shape = tensor.shape
                logger.debug(f"Creating empty tensor for {key}: shape={shape}")
                
                # Special handling for different condition types
                if key == 'audio_features':
                    # Zero audio features
                    empty_conditions[key] = torch.zeros(shape, device=device)
                    
                elif key in ['gaze', 'head_distance', 'emotion', 'speed_bucket']:
                    # Control signals - use neutral/mean values
                    if key == 'gaze':
                        # Neutral gaze (looking straight)
                        empty_conditions[key] = torch.zeros(shape, device=device)
                    elif key == 'head_distance':
                        # Middle distance
                        empty_conditions[key] = torch.ones(shape, device=device) * 0.5
                    elif key == 'emotion':
                        # Neutral emotion (centered in valence-arousal space)
                        empty_conditions[key] = torch.zeros(shape, device=device)
                    elif key == 'speed_bucket':
                        # Middle speed bucket
                        empty_conditions[key] = torch.ones(shape, device=device) * 4  # Assuming 9 buckets
                        
                elif key in ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']:
                    # Facial landmarks - use mean positions
                    if tensor is not None:
                        mean_positions = tensor.mean(dim=1, keepdim=True)  # Average over sequence
                        empty_conditions[key] = mean_positions.expand_as(tensor)
                        
                elif key == 'blink_state':
                    # Default to eyes open
                    if tensor is not None:
                        empty_conditions[key] = torch.zeros(shape, device=device)
                        empty_conditions[key][..., 1:] = 1.0  # Set openness to 1
                        
                else:
                    # Default to zeros for unknown conditions
                    empty_conditions[key] = torch.zeros_like(tensor, device=device)
                    
                logger.debug(f"Created empty tensor for {key}")
                
            return empty_conditions
            
        except Exception as e:
            logger.error(f"Error creating empty conditions: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty dict as fallback
            return {}
        
   
    def _add_noise_to_motion(
        self,
        motion_data: Dict[str, torch.Tensor],
        noise: Dict[str, torch.Tensor],
        noise_level: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        try:
            motion_keys = ['theta', 'rotation', 'scale', 'translation', 'expression_embed']
            noised_motion = {}
            
            # Get variance for current timestep
            timestep = noise_level[0].item()
            variance = self.scheduler._get_variance(timestep, max(timestep - 1, 0))
            
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=noise_level.device)

            for key, value in motion_data.items():
                if key in motion_keys:
                    noised_motion[key] = self.scheduler.add_noise(
                        original_samples=value,
                        noise=noise[key], 
                        timesteps=noise_level
                    )
                    # Apply variance scaling
                    if variance > 0:
                        noised_motion[key] = noised_motion[key] * (1 + variance).sqrt()
                else:
                    noised_motion[key] = value

            return noised_motion

        except Exception as e:
            logger.error(f"Error adding noise to motion: {str(e)}")
            logger.error(traceback.format_exc())
            return motion_data


    # helpers
    def load_and_process_image(
        image_path: str,
        size: Tuple[int, int] = (512, 512)
    ) -> torch.Tensor:
        """Load and preprocess image for VASA model."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocessing transform
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transform(image)


    # overlapping windows for training
    def generate_sequence(
        self,
        initial_pose: Dict[str, torch.Tensor],
        initial_dynamics: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        num_steps: int = 50,
        eta: float = 0.0,  # DDIM stochasticity parameter
        cfg_scales: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate sequence using DDIM sampling."""
        self.eval()
        with torch.no_grad():
            try:
                # Get batch size and sequence length from audio features
                audio_features = conditions['audio_features']
                B, T = audio_features.shape[:2]
                device = initial_pose['theta'].device

                logger.debug(f"\n=== Starting VASA Sequence Generation with DDIM ===")
                logger.debug(f"Batch size: {B}, Sequence length: {T}")

                # Set number of inference steps for scheduler
                self.scheduler.set_timesteps(num_steps, device=device)

                # Initialize motion sequence with random noise
                motion_sequence = {
                    'theta': torch.randn(B, T, 3, 4, device=device),
                    'scale': torch.randn(B, T, 3, device=device),
                    'rotation': torch.randn(B, T, 3, device=device),
                    'translation': torch.randn(B, T, 3, device=device),
                    'expression_embed': torch.randn(B, T, 128, device=device)
                }

                # Set initial frame values
                motion_sequence['theta'][:, 0] = initial_pose['theta']
                motion_sequence['rotation'][:, 0] = initial_pose['rotation']
                motion_sequence['scale'][:, 0] = initial_pose['scale']
                motion_sequence['translation'][:, 0] = initial_pose['translation']
                motion_sequence['expression_embed'][:, 0] = initial_dynamics

                # DDIM sampling loop
                for i, t in enumerate(self.scheduler.timesteps):
                    # Get model prediction
                    model_output = self.forward(
                        motion_data=motion_sequence,
                        noise_level=t.expand(B),
                        conditions=conditions
                    )

                    # DDIM step for each motion parameter
                    for key in motion_sequence.keys():
                        if key in model_output:
                            # Use scheduler step
                            scheduler_output = self.scheduler.step(
                                model_output=model_output[key],
                                timestep=t,
                                sample=motion_sequence[key],
                                eta=eta
                            )
                            motion_sequence[key] = scheduler_output.prev_sample

                return motion_sequence

            except Exception as e:
                logger.error(f"Error in sequence generation: {str(e)}")
                logger.error(traceback.format_exc())
                raise


    def generate_sequence_inference(
        self,
        initial_pose: Dict[str, torch.Tensor],
        initial_dynamics: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate sequence without overlapping windows for inference."""
        self.eval()
        with torch.no_grad():
            try:
                # Get batch size and sequence length from audio features
                audio_features = conditions['audio_features']
                B, T = audio_features.shape[:2]
                device = initial_pose['theta'].device

                logger.debug("\n=== Starting VASA Inference Generation ===")
                logger.debug(f"Batch size: {B}, Sequence length: {T}")

                # Initialize motion sequence with first frame
                motion_sequence = {
                    'theta': initial_pose['theta'].expand(-1, T, -1, -1),
                    'scale': initial_pose['scale'].expand(-1, T, -1),
                    'rotation': initial_pose['rotation'].expand(-1, T, -1),
                    'translation': initial_pose['translation'].expand(-1, T, -1),
                    'expression_embed': initial_dynamics.expand(-1, T, -1)
                }

                # Set up DDIM sampler
                self.scheduler.set_timesteps(50, device=device)

                # Generate frames sequentially
                for i, t in enumerate(self.scheduler.timesteps):
                    # Get model prediction
                    model_output = self.forward(
                        motion_data=motion_sequence,
                        noise_level=t.expand(B),
                        conditions=conditions,
                        prev_context=prev_context
                    )

                    # DDIM step for each motion parameter
                    for key in motion_sequence.keys():
                        if key in model_output:
                            # Use scheduler step
                            scheduler_output = self.scheduler.step(
                                model_output=model_output[key],
                                timestep=t,
                                sample=motion_sequence[key],
                                eta=0.0  # No randomness in inference
                            )
                            motion_sequence[key] = scheduler_output.prev_sample

                logger.debug("=== Generation Complete ===")
                for k, v in motion_sequence.items():
                    logger.debug(f"{k} shape: {v.shape}")

                return motion_sequence

            except Exception as e:
                logger.error(f"Error in sequence generation: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
    def _process_vasa_conditions(
        self,
        conditions: Dict[str, torch.Tensor],
        prev_context: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Process conditions following VASA's approach."""
        processed = {}
        
        # 1. Process audio features with temporal context
        if 'audio_features' in conditions:
            audio = conditions['audio_features']
            if prev_context and 'audio_features' in prev_context:
                # Add previous context for seamless transition
                prev_audio = prev_context['audio_features'][:, -5:]  # Last 5 frames
                audio = torch.cat([prev_audio, audio], dim=1)
            processed['audio_features'] = audio

        # 2. Process other conditions
        for key in ['gaze', 'head_distance', 'emotion', 'speed_bucket']:
            if key in conditions and conditions[key] is not None:
                processed[key] = conditions[key]

        return processed

    def _apply_temporal_coherence(
        self, 
        motion_data: Dict[str, torch.Tensor], 
        kernel_size: int = 3, 
        sigma: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Apply temporal coherence using grouped 1D convolution with dynamic smoothing.
        
        Args:
            motion_data: Dictionary of motion parameters:
                - theta: [B, T, 3, 4]
                - rotation: [B, T, 3]
                - scale: [B, T, 3]
                - translation: [B, T, 3]
                - expression_embed: [B, T, 128]
            kernel_size: Size of the smoothing kernel
            sigma: Standard deviation for Gaussian kernel
        
        Returns:
            Dictionary of smoothed motion parameters
        """
        # First, identify the core motion parameters we want to smooth
        core_motion_keys = ['theta', 'rotation', 'scale','translation', 'expression_embed']
        available_keys = [key for key in core_motion_keys if key in motion_data]
        
        logger.debug("\n=== Applying Temporal Coherence ===")
        logger.debug(f"Found motion keys: {available_keys}")
        
        # Create Gaussian kernel
        device = motion_data[available_keys[0]].device
        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize to preserve scale
        
        smoothed_data = {}
        
        # Only process core motion parameters
        for key in available_keys:
            tensor = motion_data[key]
            B = tensor.shape[0]
            T = tensor.shape[1]
            
            logger.debug(f"\nProcessing {key}:")
            logger.debug(f"  Input shape: {tensor.shape}")
            
            if key == 'theta':
                # Handle 4x4 matrices
                x = tensor.view(B, T, -1)  # [B, T, 12]
                C = x.shape[-1]  # 12
                
                # Manual padding
                pad_left = x[:, 0:kernel_size//2].flip(1)
                pad_right = x[:, -(kernel_size//2):].flip(1)
                x_padded = torch.cat([pad_left, x, pad_right], dim=1)  # [B, T+k-1, 12]
                
                # Reshape and smooth
                x_padded = x_padded.transpose(1, 2)  # [B, 12, T+k-1]
                kernel_expanded = kernel.view(1, 1, -1).repeat(C, 1, 1)  # [12, 1, k]
                x_smooth = F.conv1d(x_padded, kernel_expanded, groups=C)  # [B, 12, T]
                
                # Reshape back
                x_smooth = x_smooth.transpose(1, 2)  # [B, T, 12]
                smoothed_data[key] = x_smooth.view(B, T, 3, 4)
                
            else:
                # Handle other motion parameters (rotation[3], translation[3], expression_embed[128])
                C = tensor.shape[-1]
                
                # Manual padding
                pad_left = tensor[:, 0:kernel_size//2].flip(1)
                pad_right = tensor[:, -(kernel_size//2):].flip(1)
                x_padded = torch.cat([pad_left, tensor, pad_right], dim=1)  # [B, T+k-1, C]
                
                # Reshape and smooth
                x_padded = x_padded.transpose(1, 2)  # [B, C, T+k-1]
                kernel_expanded = kernel.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, k]
                x_smooth = F.conv1d(x_padded, kernel_expanded, groups=C)  # [B, C, T]
                
                # Reshape back
                x_smooth = x_smooth.transpose(1, 2)  # [B, T, C]
                smoothed_data[key] = x_smooth
                
            logger.debug(f"  Output shape: {smoothed_data[key].shape}")
        
        # Copy any non-motion parameters without smoothing
        for key in motion_data:
            if key not in available_keys:
                smoothed_data[key] = motion_data[key]
        
        return smoothed_data

    def _refine_expressions(
        self,
        sequence: Dict[str, torch.Tensor],
        audio_features: torch.Tensor,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply expression refinement in final denoising steps."""
        refined = sequence.copy()
        
        # Get expression embeddings
        expr = sequence['expression_embed']  # [B, T, 128]
        B, T = expr.shape[:2]
        
        # 1. Project audio using condition embedding's audio projection
        audio_proj = self.motion_transformer.cond_embed.audio_proj(audio_features)  # [B, T, 128]
        expr = expr + 0.1 * audio_proj  # Apply subtle audio influence
        
        # 2. Ensure temporal continuity with previous window
        if prev_context and 'expression_embed' in prev_context:
            prev_expr = prev_context['expression_embed'][:, -1:]  # Last frame
            # Smooth transition for first few frames
            alpha = torch.linspace(0, 1, 5, device=expr.device)
            expr[:, :5] = (
                (1 - alpha.view(1, -1, 1)) * prev_expr +
                alpha.view(1, -1, 1) * expr[:, :5]
            )
        
        # 3. Apply boundaries from training distribution
        expr = torch.clamp(expr, -2.0, 2.0)  # Typical bounds from training
        
        refined['expression_embed'] = expr
        return refined
      
class VASALossModule:
    """Loss module for VASA training, focusing on control signal adherence."""
    
    def __init__(
        self,
        volumetric_avatar: nn.Module,
        config: Dict,
        device: str = 'cuda'
    ):
        self.volumetric_avatar = volumetric_avatar
        self.config = config
        self.device = device
        

        self.speed_handler = SpeedLossHandler(num_buckets=9)

 # Initialize SyncNet evaluator
        self.syncnet = SyncNetInstance(device=device)
        self.syncnet.eval()
        
        self.batch_size =  config.train.batch_size
        # Extract loss weights from config
           # Extract loss weights from config
        self.lambda_pose = config.loss.lambda_pose
        self.lambda_dynamics = config.loss.lambda_dynamics
        self.lambda_gaze = config.loss.lambda_gaze_direction
        self.lambda_distance = config.loss.lambda_head_distance
        self.lambda_emotion = config.loss.lambda_emotion
        self.lambda_speed = config.loss.lambda_speed

        self.lambda_temporal = config.loss.lambda_temporal

        # Extract loss weights from config
        self.lambda_sync = config.loss.lambda_sync  # Weight for sync loss
        
        # New landmark loss weights
        self.lambda_lips = config.loss.lambda_lips  # Loss weight for lip motion
        self.lambda_nonlip = config.loss.lambda_nonlip  # Loss weight for other facial landmarks

        self.lambda_blink = config.loss.lambda_blink

      # Initialize transform for verification images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        import lpips
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.vis_freq = config.vis.vis_freq

    def _compute_blink_loss(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_blinks: torch.Tensor,
        lambda_blink: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss to ensure generated motion follows blink patterns.
        
        Args:
            pred_motion: Dictionary containing predicted motion parameters
            target_blinks: Target blink states [B, T, 3] tensor where:
                - Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
                - Channel 1: Left eye openness (0-1)
                - Channel 2: Right eye openness (0-1)
            lambda_blink: Weight for blink loss
            device: Computation device
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        try:
            metrics = {}
            
            # Extract predicted eye landmarks
            left_eye = pred_motion['left_eye']   # [B, T, 7, 3]  
            right_eye = pred_motion['right_eye'] # [B, T, 8, 3]
            
            B, T = left_eye.shape[:2]

            # Compute eye aspect ratios for predictions
            def compute_ear(eye_landmarks: torch.Tensor) -> torch.Tensor:
                """Compute Eye Aspect Ratio from landmarks."""
                # For left eye: vertical distances between landmarks
                A = torch.norm(eye_landmarks[..., 1, :] - eye_landmarks[..., 5, :], dim=-1)
                B = torch.norm(eye_landmarks[..., 2, :] - eye_landmarks[..., 4, :], dim=-1)
                
                # Horizontal distance
                C = torch.norm(eye_landmarks[..., 0, :] - eye_landmarks[..., 3, :], dim=-1)
                
                # Compute EAR with stability term
                return (A + B) / (2.0 * C.clamp(min=1e-6))

            # Get predicted eye openness
            left_ear = compute_ear(left_eye)    # [B, T]
            right_ear = compute_ear(right_eye)   # [B, T]
            
            # Normalize to [0,1] range
            left_openness = torch.clamp(left_ear / 0.3, 0, 1)   # [B, T]
            right_openness = torch.clamp(right_ear / 0.3, 0, 1) # [B, T]

            # Get target values
            target_phase = target_blinks[..., 0]      # [B, T]
            target_left = target_blinks[..., 1]       # [B, T]
            target_right = target_blinks[..., 2]      # [B, T]

            # Phase matching loss
            pred_phase = torch.zeros((B, T, 4), device=device)  # 4 phases
            
            # Fix: Pad the diff results to match sequence length
            is_closed = (left_openness < 0.2) | (right_openness < 0.2)
            
            # Compute diffs and pad
            left_diff = F.pad(left_openness.diff(dim=1), (0, 1))  # Pad right
            right_diff = F.pad(right_openness.diff(dim=1), (0, 1))  # Pad right
            
            is_opening = ~is_closed & (left_diff > 0.1)
            is_closing = ~is_closed & (left_diff < -0.1)

            pred_phase[..., 0] = ~(is_closed | is_opening | is_closing)  # Open
            pred_phase[..., 1] = is_closing
            pred_phase[..., 2] = is_closed
            pred_phase[..., 3] = is_opening
            
            phase_loss = F.cross_entropy(
                pred_phase.view(-1, 4),
                target_phase.long().view(-1)
            )
            
            # Openness matching loss
            left_loss = F.mse_loss(left_openness, target_left)
            right_loss = F.mse_loss(right_openness, target_right)
            
            # Symmetry loss to encourage eyes to blink together
            symmetry_loss = F.mse_loss(left_openness, right_openness)
            
            # Temporal smoothness loss
            if T > 1:
                temp_loss = F.mse_loss(
                    left_openness[:, 1:] - left_openness[:, :-1],
                    target_left[:, 1:] - target_left[:, :-1]
                ) + F.mse_loss(
                    right_openness[:, 1:] - right_openness[:, :-1],
                    target_right[:, 1:] - target_right[:, :-1]
                )
            else:
                temp_loss = torch.tensor(0.0, device=device)
                
            # Combine losses
            total_loss = (
                phase_loss + 
                left_loss + 
                right_loss + 
                0.5 * symmetry_loss +
                0.2 * temp_loss
            ) * lambda_blink
            
            # Record metrics
            metrics.update({
                'blink_phase_loss': phase_loss.item(),
                'blink_left_loss': left_loss.item(),
                'blink_right_loss': right_loss.item(),
                'blink_symmetry_loss': symmetry_loss.item(),
                'blink_temporal_loss': temp_loss.item(),
                'blink_total': total_loss.item()
            })
            
            return total_loss, metrics
            
        except Exception as e:
            logger.error(f"Error computing blink loss: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=device), {
                'blink_total': 0.0,
                'blink_error': str(e)
            }
    

    def plot_to_wandb_image(self,fig):
  
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from PIL import Image

        """Convert matplotlib figure to wandb image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return wandb.Image(Image.open(buf))

    
    
    
    


    def _compute_landmark_losses(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """Compute losses for lip and non-lip facial landmarks."""
            try:
                losses = {}
                
                # 1. Lip Motion Loss
                if 'lips' in targets:
                    pred_lips = outputs['lips']
                    target_lips = targets['lips']
                    
                    # Compute positional loss for lips
                    lips_pos_loss = F.mse_loss(pred_lips, target_lips)
                    
                    # Compute velocity loss for smoother lip motion
                    lips_vel_pred = pred_lips[:, 1:] - pred_lips[:, :-1]
                    lips_vel_target = target_lips[:, 1:] - target_lips[:, :-1]
                    lips_vel_loss = F.mse_loss(lips_vel_pred, lips_vel_target)
                    
                    # Combined lip loss
                    lips_loss = self.lambda_lips * (lips_pos_loss + 0.5 * lips_vel_loss)
                    losses['lips_pos_loss'] = lips_pos_loss
                    losses['lips_vel_loss'] = lips_vel_loss
                    losses['lips_total'] = lips_loss
                    
                # 2. Non-lip Facial Motion Losses
                nonlip_losses = []
                
                # Process each non-lip landmark group
                landmark_groups = {
                    'right_eye': 'right_eye',
                    'left_eye': 'left_eye',
                    'jaw': 'jaw',
                    'nose': 'nose'
                }
                
                for group_name, key in landmark_groups.items():
                    if key in targets:
                        pred = outputs[key]
                        target = targets[key]
                        
                        # Position loss
                        pos_loss = F.mse_loss(pred, target)
                        losses[f'{group_name}_pos_loss'] = pos_loss
                        
                        # Velocity loss for smooth motion
                        if pred.shape[1] > 1:  # Only if we have multiple frames
                            vel_pred = pred[:, 1:] - pred[:, :-1]
                            vel_target = target[:, 1:] - target[:, :-1]
                            vel_loss = F.mse_loss(vel_pred, vel_target)
                            losses[f'{group_name}_vel_loss'] = vel_loss
                            
                            # Combined loss for this landmark group
                            group_loss = pos_loss + 0.5 * vel_loss
                        else:
                            group_loss = pos_loss
                            
                        losses[f'{group_name}_total'] = group_loss
                        nonlip_losses.append(group_loss)
                
                # Combine non-lip losses
                if nonlip_losses:
                    nonlip_total = sum(nonlip_losses) * self.lambda_nonlip
                    losses['nonlip_total'] = nonlip_total
                    
                    # Add to total facial motion loss
                    if 'lips_total' in losses:
                        losses['facial_motion_total'] = losses['lips_total'] + nonlip_total
                    else:
                        losses['facial_motion_total'] = nonlip_total
                
                return losses
                
            except Exception as e:
                logger.error(f"Error computing landmark losses: {str(e)}")
                logger.error(traceback.format_exc())
                return {}




    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        return_metrics: bool = True,
        current_epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses including expression verification."""
        try:
            logger.info("\n=== Computing Losses ===")
            losses = {}
            metrics = {}
            device = outputs['theta'].device

            # 1. Reconstruction losses
            logger.debug("\nComputing reconstruction losses:")
            recon_losses = self._compute_reconstruction_losses(outputs, targets, noise, step)
            losses.update(recon_losses)
            logger.debug("Reconstruction losses:")
            for k, v in recon_losses.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: {v.item():.6f}")

            # 2. Expression Verification Loss
            logger.debug("\nChecking verification loss conditions:")
            should_compute_verify = (
                current_epoch is not None and 
                current_epoch >= self.config.train.control_start_epoch and  # Start verification with control
                hasattr(self.config.loss, 'use_verification') and 
                self.config.loss.use_verification
            )
            logger.debug(f"Should compute verification loss: {should_compute_verify}")

            if should_compute_verify:
    
                logger.debug("Computing verification loss...")
                try:
                    # Initialize LPIPS for perceptual loss
    

                    # Load reference image
                    data_dir = self.config.paths.data_dir if hasattr(self.config, 'paths') else "data"
                    test_img = Image.open(f"{data_dir}/A.png").convert('RGB')
                    test_tensor = self.transform(test_img).unsqueeze(0).to(device)
                    
                    # Get current expression embed
                    curr_expression = outputs['expression_embed']  # [B, T, 128]
                    B, T = curr_expression.shape[:2]
                    
                    verification_loss = 0.0
                    num_samples = min(T, 1)  # Check up to 4 frames to save compute
                    sample_indices = torch.linspace(0, T-1, num_samples).long()
                    
                    logger.debug(f"Verifying {num_samples} expression samples from sequence")
                    
                    for idx in sample_indices:
                        # Create data dict for current expression
                        source_tensor = test_tensor.repeat(B, 1, 1, 1)
                        source_mask = self.volumetric_avatar.face_idt.forward(source_tensor)[0]
                        source_mask = (source_mask > 0.6).float()
                        
                        data_dict = {
                            'source_img': source_tensor,
                            'source_mask': source_mask,
                            'target_img': source_tensor,
                            'target_mask': source_mask,
                            'source_theta': outputs['theta'][:, idx:idx+1],
                            'target_theta': outputs['theta'][:, idx:idx+1],
                            'idt_embed': self.volumetric_avatar.idt_embedder_nw.forward_image(
                                source_tensor * source_mask
                            )
                        }
                        
                        # Set current expression for verification
                        data_dict['target_pose_embed'] = curr_expression[:, idx:idx+1]
                        
                        # Generate reconstruction through full pipeline
                        frame = self._generate_verification_frame(data_dict)
                        save_image(frame, f"{data_dir}/verification_{idx}.png")

                        # Compute perceptual loss
                        verify_loss = self.loss_fn_alex(frame, source_tensor)
                        verification_loss += verify_loss.mean()
                        
                        logger.debug(f"  Sample {idx} verification loss: {verify_loss.mean().item():.6f}")
                    
                    # Average and scale verification loss
                    verification_loss = verification_loss / num_samples * self.config.loss.lambda_verification
                    losses['verification'] = verification_loss
                    logger.debug(f"Total verification loss: {verification_loss.item():.6f}")

                    # Log visualizations periodically
                    if step is not None and step % 100 == 0:
                        wandb.log({
                            'verification/source': wandb.Image(test_tensor[0].cpu()),
                            'verification/reconstruction': wandb.Image(frame[0].cpu()),
                            'verification/loss': verification_loss.item()
                        }, step=step)

                except Exception as e:
                    logger.error(f"Error in expression verification: {str(e)}")
                    logger.error(traceback.format_exc())
                    losses['verification'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("Skipping verification loss")
                losses['verification'] = torch.tensor(0.0, device=device)

            # 3. Control losses
            logger.debug("\nChecking control loss conditions:")
            logger.debug(f"Current epoch: {current_epoch}")
            logger.debug(f"Control start epoch: {self.config.train.control_start_epoch}")
            
            should_compute_control = (
                current_epoch is not None and 
                current_epoch >= self.config.train.control_start_epoch
            )
            logger.debug(f"Should compute control losses: {should_compute_control}")

            if should_compute_control:
                logger.debug("Computing control losses...")
                control_losses = self._compute_control_losses(outputs, conditions, current_epoch)
                losses.update(control_losses)
                logger.debug("Control losses:")
                for k, v in control_losses.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: {v.item():.6f}")
            else:
                logger.debug("Skipping control losses")
                losses.update(self._get_zero_losses())

            # 4. SyncNet losses
            logger.debug("\nChecking sync loss conditions:")
            logger.debug(f"Use sync loss: {self.config.loss.use_sync_loss}")
            
            if self.config.loss.use_sync_loss:
                logger.debug("Computing sync loss...")
                sync_loss = self._compute_sync_loss(outputs, targets)
                losses['sync_loss'] = sync_loss * self.lambda_sync
                logger.debug(f"Sync loss: {losses['sync_loss'].item():.6f}")
            else:
                logger.debug("Skipping sync loss")
                losses['sync_loss'] = torch.tensor(0.0, device=device)

            # Final loss aggregation
            # Log loss weights
            logger.debug("\nLoss weights:")
            logger.debug(f"  lambda_reconstruction: {self.config.loss.lambda_reconstruction}")
            logger.debug(f"  lambda_verification: {self.config.loss.lambda_verification}")
            logger.debug(f"  lambda_control: {self.config.loss.lambda_control}")
            logger.debug(f"  lambda_sync: {self.config.loss.lambda_sync}")

            # Compute total loss
            logger.debug("\nComputing total loss:")
            recon_term = self.config.loss.lambda_reconstruction * losses['reconstruction']
            verify_term = losses['verification']  # Already scaled in computation
            control_term = self.config.loss.lambda_control * losses.get('control_total', torch.tensor(0.0, device=device))
            sync_term = self.lambda_sync * losses.get('sync_loss', torch.tensor(0.0, device=device))

            logger.debug(f"  Reconstruction term: {recon_term.item():.6f}")
            logger.debug(f"  Verification term: {verify_term.item():.6f}")
            logger.debug(f"  Control term: {control_term.item():.6f}")
            logger.debug(f"  Sync term: {sync_term.item():.6f}")

            total_loss = recon_term + verify_term + control_term + sync_term
            losses['total'] = total_loss
            logger.debug(f"Total loss: {total_loss.item():.6f}")

            # Return results
            if return_metrics:
                metrics.update({k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
                return losses, metrics

            return losses

        except Exception as e:
            logger.error(f"ERROR in compute_losses: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Input shapes:")
            if outputs:
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"  outputs[{k}]: {v.shape}")
            if targets:
                for k, v in targets.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"  targets[{k}]: {v.shape}")
            logger.error(traceback.format_exc())
            # Return all required loss components with default value 1.0
            # This allows TDD tests to detect the failure properly
            default_losses = {
                'reconstruction': torch.tensor(1.0, device=device, requires_grad=True),
                'pose_loss': torch.tensor(1.0, device=device, requires_grad=True),
                'dynamics_loss': torch.tensor(1.0, device=device, requires_grad=True),
                'total': torch.tensor(1.0, device=device, requires_grad=True)
            }
            if return_metrics:
                metrics = {k: 1.0 for k in default_losses.keys()}
                return default_losses, metrics
            return default_losses

        

    def _generate_verification_frame(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper method to generate verification frame through EMO pipeline with memory optimizations."""
        try:
            logger.info("\n=== Starting Memory-Optimized Verification Frame Generation ===")
            
            # 1. Memory Optimization: Clear cache and freeze model
            torch.cuda.empty_cache()
            
            # Store original states to restore later if needed
            training_state = self.volumetric_avatar.training
            
            # Set model to eval mode and disable gradients
            self.volumetric_avatar.eval()
            
            with torch.no_grad():
                # 2. Memory Optimization: Use dimension reduction early
                data_dict['source_theta'] = data_dict['source_theta'].squeeze(1)
                data_dict['target_theta'] = data_dict['target_theta'].squeeze(1)
                data_dict['target_pose_embed'] = data_dict['target_pose_embed'].squeeze(1)
                
                # 3. Memory Optimization: Process in smaller batches if needed
                B = data_dict['source_img'].shape[0]
                if B > 4:
                    logger.info(f"Large batch size {B} detected, processing in chunks")
                    frames = []
                    for i in range(0, B, 4):
                        batch_dict = {k: v[i:i+4] if torch.is_tensor(v) else v 
                                    for k, v in data_dict.items()}
                        frame = self._process_single_batch(batch_dict)
                        frames.append(frame)
                        torch.cuda.empty_cache()
                    result = torch.cat(frames, dim=0)
                else:
                    result = self._process_single_batch(data_dict)
                

                    
                return result

        except Exception as e:

            logger.error(f"Error in frame generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _process_single_batch(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single batch with memory optimizations."""
        try:
            with torch.no_grad():  # Extra safety to ensure no gradients
                # 1. Generate source pose embedding
                source_data = {
                    'source_img': data_dict['source_img'],
                    'source_mask': data_dict['source_mask'],
                    'target_img': data_dict['source_img'],
                    'target_mask': data_dict['source_mask'],
                    'source_theta': data_dict['source_theta'],
                    'target_theta': data_dict['source_theta'],
                    'idt_embed': data_dict['idt_embed']
                }
                
                # 2. Memory Optimization: Process without storing gradients
                source_data = self.volumetric_avatar.expression_embedder_nw(source_data, True, False)
                data_dict['source_pose_embed'] = source_data['source_pose_embed'].detach()
                del source_data
                torch.cuda.empty_cache()
                
                # 3. Predict embeddings without gradient computation
                source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = \
                    self.volumetric_avatar.predict_embed(data_dict)
                
                # Detach embeddings to ensure no gradient history
                embed_dict = {k: v.detach() if torch.is_tensor(v) else v 
                            for k, v in embed_dict.items()}
                
                # 4. Generate UV warp
                target_uv_warp, _ = self.volumetric_avatar.uv_generator_nw(target_warp_embed_dict)
                target_uv_warp = target_uv_warp.detach()
                del target_warp_embed_dict
                torch.cuda.empty_cache()
                
                if self.volumetric_avatar.resize_warp:
                    target_uv_warp = F.avg_pool3d(
                        target_uv_warp.permute(0, 4, 1, 2, 3),
                        kernel_size=self.volumetric_avatar.warp_resize_stride,
                        stride=self.volumetric_avatar.warp_resize_stride
                    ).permute(0, 2, 3, 4, 1)
                
                # 5. Process volume with no gradients
                B = data_dict['source_img'].shape[0]
                source_latents = self.volumetric_avatar.local_encoder_nw(
                    data_dict['source_img'] * data_dict['source_mask']
                ).detach()
                
                source_volume = source_latents.view(B, -1,
                    self.volumetric_avatar.args.latent_volume_depth,
                    self.volumetric_avatar.args.latent_volume_size,
                    self.volumetric_avatar.args.latent_volume_size)
                del source_latents
                torch.cuda.empty_cache()
                
                if self.volumetric_avatar.args.source_volume_num_blocks > 0:
                    source_volume = self.volumetric_avatar.volume_source_nw(source_volume).detach()
                
                canonical_volume = self.volumetric_avatar.volume_process_nw(source_volume).detach()
                del source_volume
                torch.cuda.empty_cache()
                
                # 6. Process grid and rotation
                grid = self.volumetric_avatar.identity_grid_3d.repeat_interleave(B, dim=0)
                rotation_warp = grid.bmm(data_dict['target_theta'][:, :3].transpose(1, 2)).view(
                    B, self.volumetric_avatar.args.latent_volume_depth,
                    self.volumetric_avatar.args.latent_volume_size,
                    self.volumetric_avatar.args.latent_volume_size, 3)
                rotation_warp = rotation_warp.detach()
                del grid
                torch.cuda.empty_cache()
                
                # 7. Grid sampling
                aligned_volume = self.volumetric_avatar.grid_sample(
                    self.volumetric_avatar.grid_sample(canonical_volume, target_uv_warp),
                    rotation_warp
                ).detach()
                del canonical_volume, target_uv_warp, rotation_warp
                torch.cuda.empty_cache()
                
                # 8. Generate final frame
                frame, _, _, _ = self.volumetric_avatar.decoder_nw(
                    data_dict,
                    embed_dict,
                    aligned_volume.view(B, -1,
                        self.volumetric_avatar.args.latent_volume_size,
                        self.volumetric_avatar.args.latent_volume_size),
                    False,
                    stage_two=True
                )
                
                return frame.detach()  # Ensure final output is detached

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def evaluate_sync_quality(
        self,
        generated_frames: torch.Tensor,    # [B, T, C, H, W]
        audio_features: torch.Tensor,      # [B, T, D] or [B, 1, T, D]
        audio_mfcc: torch.Tensor,         # [B, T, 13] MFCC features for SyncNet
        window_size: int = 5              # Size of evaluation window
    ) -> Dict[str, float]:
        """
        Evaluate sync quality using SyncNet with MFCC features.
        
        Args:
            generated_frames: Generated video frames
            audio_features: Original wav2vec features (not used by SyncNet)
            audio_mfcc: MFCC features for SyncNet
            window_size: Size of sliding window
            
        Returns:
            Dictionary of sync quality metrics
        """
        try:
            logger.debug("\n=== Evaluating Sync Quality ===")
            logger.debug(f"Generated frames shape: {generated_frames.shape}")
            logger.debug(f"Audio MFCC shape: {audio_mfcc.shape}")
            
            # Get batch size and verify inputs
            B, T = generated_frames.shape[:2]
            device = generated_frames.device
            logger.debug(f"Processing batch size: {B}, sequence length: {T}")
            
            # Move tensors to appropriate device
            audio_mfcc = audio_mfcc.to(device)
            
            # Initialize metrics
            metrics = {
                'avg_sync_confidence': 0.0,
                'avg_sync_offset': 0.0,
                'min_confidence': float('inf'),
                'max_confidence': float('-inf')
            }
            
            # Process sequence in windows
            all_confidences = []
            all_offsets = []
            
            for start_idx in range(0, T - window_size + 1, window_size):
                try:
                    end_idx = min(start_idx + window_size, T)
                    
                    # Get window tensors
                    frame_window = generated_frames[:, start_idx:end_idx]
                    mfcc_window = audio_mfcc[:, start_idx:end_idx]
                    
                    # Evaluate sync for this window
                    offset, confidence = self.syncnet.evaluate(
                        frames=frame_window,
                        audio_features=mfcc_window,
                        batch_size=self.batch_size
                    )
                    
                    all_confidences.append(confidence)
                    all_offsets.append(offset)
                    
                except Exception as e:
                    logger.error(f"Error evaluating window {start_idx}:{end_idx}: {str(e)}")
                    continue
            
            # Aggregate metrics
            if all_confidences:
                confidences = torch.stack(all_confidences)
                metrics.update({
                    'avg_sync_confidence': confidences.mean().item(),
                    'avg_sync_offset': float(sum(all_offsets)) / len(all_offsets),
                    'min_confidence': confidences.min().item(),
                    'max_confidence': confidences.max().item()
                })
            
            logger.debug("Evaluation complete")
            logger.debug(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in sync quality evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'avg_sync_confidence': 0.0,
                'avg_sync_offset': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }

    def evaluate_window(
        self,
        frames: torch.Tensor,      # [B, T, C, H, W]
        audio: torch.Tensor,       # [B, T, D] or [B, 1, T, D]
        batch_size: int = 20
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Evaluate synchronization for a single window of frames and audio.
        
        Args:
            frames: Video frames for the window
            audio: Audio features for the window
            batch_size: Processing batch size
            
        Returns:
            Dictionary containing offset and confidence
        """
        try:
            logger.debug("\n=== Evaluating Window ===")
            logger.debug(f"Window frames shape: {frames.shape}")
            logger.debug(f"Window audio shape: {audio.shape}")
            
            # Ensure tensors are on same device
            device = frames.device
            audio = audio.to(device)
            
            # Prepare frames
            if len(frames.shape) == 5:  # [B, T, C, H, W]
                frames = frames.transpose(1, 2)  # -> [B, C, T, H, W]
            
            # Prepare audio
            if len(audio.shape) == 3:  # [B, T, D]
                audio = audio.unsqueeze(1)  # -> [B, 1, T, D]
            
            # Get predictions from SyncNet
            offset, confidence = self.syncnet.evaluate(
                frames=frames,
                audio_features=audio,
                batch_size=batch_size
            )
            
            logger.debug(f"Window evaluation results:")
            logger.debug(f"  Offset: {offset}")
            logger.debug(f"  Confidence: {confidence.mean().item():.4f}")
            
            return {
                'offset': offset,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error evaluating window: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'offset': 0.0,
                'confidence': torch.zeros(1, device=frames.device)
            }


    def _compute_reconstruction_losses(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        try:
            logger.debug("\n=== Computing Reconstruction Losses ===")
            
            # Get device and initialize losses
            device = pred['theta'].device
            losses = {}
            
            # Determine if we're in training mode
            comparison_target = noise if noise is not None else target
            is_training = noise is not None  # Use noise presence to determine training mode
            logger.debug(f"\nMode: {'training' if is_training else 'validation'}")


            # 1. Theta (pose matrix) loss
            if 'theta' in pred:
                if is_training:
                    # During training, compare predicted noise to target noise
                    pred_flat = pred['theta'].view(pred['theta'].shape[0], -1, 12)
                    target_flat = comparison_target['theta'].view(comparison_target['theta'].shape[0], -1, 12)
                    losses['theta_loss'] = F.mse_loss(pred_flat, target_flat)
                else:
                    # During validation, use pose matrix loss
                    losses['theta_loss'] = self._compute_pose_matrix_loss(
                        pred['theta'], 
                        target['theta']
                    )

            # 2. Scale loss
            if 'scale' in pred:
                losses['scale_loss'] = F.mse_loss(
                    pred['scale'],
                    comparison_target['scale']
                )

            # 3. Rotation loss
            if 'rotation' in pred:
                losses['rotation_loss'] = F.mse_loss(
                    pred['rotation'],
                    comparison_target['rotation']
                )

            # 4. Translation loss
            if 'translation' in pred:
                losses['translation_loss'] = F.mse_loss(
                    pred['translation'],
                    comparison_target['translation']
                )

            # 5. Expression loss 
            if 'expression_embed' in pred:
                # Apply dimension weights

                losses['expression_loss'] = F.mse_loss(
                            pred['expression_embed'],
                            comparison_target['expression_embed']
                        )
         
                should_visualize = step > 0 and step % self.vis_freq == 0
                if should_visualize:
                    self.visualize_sequence(pred['expression_embed'],target['expression_embed'],step,0)


            # Ensure losses have gradients when needed
            if is_training:
                for k, v in losses.items():
                    if not v.requires_grad:
                        losses[k] = v.clone().requires_grad_(True)

            # Combine into major loss components with proper scaling
            pose_loss = (
                losses.get('theta_loss', torch.tensor(0.0, device=device)) +
                losses.get('scale_loss', torch.tensor(0.0, device=device)) +
                losses.get('rotation_loss', torch.tensor(0.0, device=device)) +
                losses.get('translation_loss', torch.tensor(0.0, device=device))
            ) * self.lambda_pose

            dynamics_loss = losses['expression_loss'] * self.lambda_dynamics


            # Motion smoothness loss if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            if pred['theta'].shape[1] > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred) * self.lambda_temporal

            # Combined reconstruction loss
            reconstruction_loss = pose_loss + dynamics_loss + motion_loss

            # Return all losses
            return {
                'reconstruction': reconstruction_loss,
                'pose_loss': pose_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': losses.get('theta_loss', torch.tensor(0.0, device=device)),
                'scale_loss': losses.get('scale_loss', torch.tensor(0.0, device=device)),
                'rotation_loss': losses.get('rotation_loss', torch.tensor(0.0, device=device)),
                'translation_loss': losses.get('translation_loss', torch.tensor(0.0, device=device)),
                'expression_loss': losses.get('expression_loss', torch.tensor(0.0, device=device))
            }

        except Exception as e:
            logger.error(f"Error computing reconstruction losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device, requires_grad=True),
                'pose_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'dynamics_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'motion_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'theta_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'scale_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'rotation_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'translation_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'expression_loss': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
    def compute_motion_losses(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_motion: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all motion parameters with visualization and wandb logging.
        """
        try:
            B = pred_motion['theta'].shape[0]
            device = pred_motion['theta'].device
            losses = {}

            # Determine if we're in training (noise prediction) or validation mode
            is_training = noise is not None
            comparison_target = noise if is_training else target_motion
            mode_prefix = 'train' if is_training else 'val'
            
            logger.debug(f"\nMode: {'training' if is_training else 'validation'}")

            # Process predictions and targets
            param_dims = {
                'theta': 12,      # 3x4 matrix flattened
                'scale': 3,    
                'rotation': 3,    
                'translation': 3,
                'expression_embed': 128
            }

            # Initialize losses and logging metrics
            losses = {f'{param}_loss': torch.tensor(0.0, device=device) for param in param_dims.keys()}
            metrics = {}
            
            # Process each parameter and compute losses
            for param, dim in param_dims.items():
                try:
                    if param == 'theta':
                        pred_flat = pred_motion[param].view(B, -1, 12)
                        target_flat = comparison_target[param].view(B, -1, 12)
                        
                        if not is_training:
                            losses[f'{param}_loss'] = self._compute_pose_matrix_loss(
                                pred_motion[param], 
                                comparison_target[param]
                            )
                        else:
                            losses[f'{param}_loss'] = F.mse_loss(pred_flat, target_flat)
                            
                        # Log theta statistics
                        if wandb.run is not None:
                            metrics.update({
                                f'{mode_prefix}/theta/mean': pred_flat.mean().item(),
                                f'{mode_prefix}/theta/std': pred_flat.std().item(),
                                f'{mode_prefix}/theta/min': pred_flat.min().item(),
                                f'{mode_prefix}/theta/max': pred_flat.max().item(),
                                f'{mode_prefix}/theta/loss': losses[f'{param}_loss'].item()
                            })
                    
                    elif param == 'expression_embed':
                        losses['expression_loss'] = F.mse_loss(
                            pred_motion[param],
                            comparison_target[param]
                        )
                        
                        # Log expression statistics
                        if wandb.run is not None:
                            expr_pred = pred_motion[param]
                            metrics.update({
                                f'{mode_prefix}/expression/mean': expr_pred.mean().item(),
                                f'{mode_prefix}/expression/std': expr_pred.std().item(),
                                f'{mode_prefix}/expression/min': expr_pred.min().item(),
                                f'{mode_prefix}/expression/max': expr_pred.max().item(),
                                f'{mode_prefix}/expression/loss': losses['expression_loss'].item()
                            })
                            
                    else:
                        # Handle rotation and translation
                        losses[f'{param}_loss'] = F.mse_loss(
                            pred_motion[param],
                            comparison_target[param]
                        )
                        
                        # Log parameter statistics
                        if wandb.run is not None:
                            param_tensor = pred_motion[param]
                            metrics.update({
                                f'{mode_prefix}/{param}/mean': param_tensor.mean().item(),
                                f'{mode_prefix}/{param}/std': param_tensor.std().item(),
                                f'{mode_prefix}/{param}/min': param_tensor.min().item(),
                                f'{mode_prefix}/{param}/max': param_tensor.max().item(),
                                f'{mode_prefix}/{param}/loss': losses[f'{param}_loss'].item()
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing {param}: {str(e)}")
                    continue

            # Compute combined losses
            pose_loss = (
                losses['theta_loss'] +
                losses['rotation_loss'] + 
                losses['translation_loss'] +
                losses['scale_loss']
            ) * self.lambda_pose

            dynamics_loss = losses['expression_loss'] * self.lambda_dynamics

            # Compute motion smoothness if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            if pred_motion['theta'].shape[1] > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred_motion)

            # Combined reconstruction loss
            reconstruction_loss = pose_loss + dynamics_loss + motion_loss

            # Log combined losses and detailed parameter statistics
           # Log combined losses and detailed parameter statistics
            if wandb.run is not None:
                # Base losses
                metrics.update({
                    f'{mode_prefix}/loss/pose': pose_loss.item(),
                    f'{mode_prefix}/loss/dynamics': dynamics_loss.item(),
                    f'{mode_prefix}/loss/motion': motion_loss.item(),
                    f'{mode_prefix}/loss/total': reconstruction_loss.item()
                })
                
                # Add detailed statistics for core motion parameters
                core_params = ['theta', 'rotation', 'scale', 'translation', 'expression_embed']
                for param in core_params:
                    if param in pred_motion and param in comparison_target:
                        try:
                            # Safely process tensors
                            pred_tensor = pred_motion[param]
                            target_tensor = comparison_target[param]
                            
                            # Ensure tensors are detached and on CPU
                            pred_np = pred_tensor.detach().cpu().numpy()
                            target_np = target_tensor.detach().cpu().numpy()
                            
                            # Compute differences and statistics
                            diff = pred_np - target_np
                            
                            # Handle potential NaN or Inf values
                            max_diff = np.nan_to_num(np.abs(diff).max())
                            mean_diff = np.nan_to_num(np.abs(diff).mean())
                            
                            # Safely compute correlation
                            try:
                                correlation = np.nan_to_num(np.corrcoef(
                                    pred_np.flatten(), 
                                    target_np.flatten()
                                )[0,1])
                            except ValueError:
                                correlation = 0.0
                            
                            metrics.update({
                                f'{mode_prefix}/{param}/max_diff': float(max_diff),
                                f'{mode_prefix}/{param}/mean_diff': float(mean_diff),
                                f'{mode_prefix}/{param}/correlation': float(correlation),
                                f'{mode_prefix}/{param}/pred_range_min': float(np.nan_to_num(pred_np.min())),
                                f'{mode_prefix}/{param}/pred_range_max': float(np.nan_to_num(pred_np.max())),
                                f'{mode_prefix}/{param}/target_range_min': float(np.nan_to_num(target_np.min())),
                                f'{mode_prefix}/{param}/target_range_max': float(np.nan_to_num(target_np.max()))
                            })
                        except Exception as e:
                            logger.error(f"Error processing statistics for {param}: {str(e)}")
                            continue

                # Log all metrics to wandb
                wandb.log(metrics, step=step)

            # Return dictionary of all losses
            return {
                'reconstruction': reconstruction_loss,
                'pose_loss': pose_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': losses['theta_loss'],
                'rotation_loss': losses['rotation_loss'],
                'scale_loss': losses['scale_loss'],
                'translation_loss': losses['translation_loss'],
                'expression_loss': losses['expression_loss']
            }

        except Exception as e:
            logger.error(f"Error computing motion losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device),
                'pose_loss': torch.tensor(0.0, device=device),
                'dynamics_loss': torch.tensor(0.0, device=device),
                'motion_loss': torch.tensor(0.0, device=device),
                'theta_loss': torch.tensor(0.0, device=device),
                'rotation_loss': torch.tensor(0.0, device=device),
                'scale_loss': torch.tensor(0.0, device=device),
                'translation_loss': torch.tensor(0.0, device=device),
                'expression_loss': torch.tensor(0.0, device=device)
            }
        
    def _compute_motion_smoothness_loss(
            self, 
            pred: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            """
            Compute temporal smoothness loss for motion parameters.
            
            Args:
                pred: Dictionary of predicted motion parameters
                
            Returns:
                Motion smoothness loss
            """
            loss = 0.0
            
            # Compute velocity (first derivative)
            theta_vel = torch.diff(pred['theta'], dim=1)
            rotation_vel = torch.diff(pred['rotation'], dim=1)
            scale_vel = torch.diff(pred['scale'], dim=1)
            translation_vel = torch.diff(pred['translation'], dim=1)
            expression_vel = torch.diff(pred['expression_embed'], dim=1)
            
            # Compute acceleration (second derivative)
            if pred['theta'].shape[1] > 2:
                theta_acc = torch.diff(theta_vel, dim=1)
                rotation_acc = torch.diff(rotation_vel, dim=1)
                scale_acc = torch.diff(scale_vel, dim=1)
                translation_acc = torch.diff(translation_vel, dim=1)
                # expression_acc = torch.diff(expression_vel, dim=1)
            else:
                theta_acc = torch.zeros_like(theta_vel)
                rotation_acc = torch.zeros_like(rotation_vel)
                scale_acc = torch.zeros_like(scale_vel)
                translation_acc = torch.zeros_like(translation_vel)
                # expression_acc = torch.zeros_like(expression_vel)
            
            # Velocity smoothness
            loss += F.mse_loss(theta_vel, torch.zeros_like(theta_vel))
            loss += F.mse_loss(rotation_vel, torch.zeros_like(rotation_vel))
            loss += F.mse_loss(scale_vel, torch.zeros_like(scale_vel))
            loss += F.mse_loss(translation_vel, torch.zeros_like(translation_vel))
            # loss += F.mse_loss(expression_vel, torch.zeros_like(expression_vel))
            
            # Acceleration smoothness
            loss += 0.5 * F.mse_loss(theta_acc, torch.zeros_like(theta_acc))
            loss += 0.5 * F.mse_loss(rotation_acc, torch.zeros_like(rotation_acc))
            loss += 0.5 * F.mse_loss(scale_acc, torch.zeros_like(scale_acc))
            loss += 0.5 * F.mse_loss(translation_acc, torch.zeros_like(translation_acc))
            # loss += 0.5 * F.mse_loss(expression_acc, torch.zeros_like(expression_acc))
            
            return loss


    def create_expression_comparison_plot(self, pred: torch.Tensor, target: torch.Tensor, step: int, frame_losses=None):
        """Create visualization comparing predicted and target expressions with normalized difference plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from PIL import Image
        
        # Ensure we're working with numpy arrays on CPU
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Handle sequence vs single frame
        if len(pred_np.shape) > 1 and pred_np.shape[0] > 1:
            logger.info(f"Got sequence of shape {pred_np.shape}, extracting first frame")
            pred_np = pred_np[0]
            target_np = target_np[0]
        
        # Ensure we have shape [128]
        pred_np = pred_np.reshape(-1)[:128]
        target_np = target_np.reshape(-1)[:128]
        
        # Compute normalized difference relative to the value range
        value_range = max(pred_np.max(), target_np.max()) - min(pred_np.min(), target_np.min())
        diff = (pred_np - target_np) / (value_range + 1e-8)  # Normalize by value range
        
        # Find global min/max for consistent scale
        global_min = min(pred_np.min(), target_np.min())
        global_max = max(pred_np.max(), target_np.max())
        
        # Create figure with plots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5])
        
        # Plot predicted embedding
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(pred_np.reshape(1, -1), aspect='auto', 
                        cmap='viridis', vmin=global_min, vmax=global_max)
        ax1.set_title(f'Predicted Expression Embedding (Step {step})')
        plt.colorbar(im1, ax=ax1)
        ax1.set_yticks([])
        ax1.set_xlabel('Dimension (128)')
        
        # Plot target embedding
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(target_np.reshape(1, -1), aspect='auto',
                        cmap='plasma', vmin=global_min, vmax=global_max)
        ax2.set_title('Target Expression Embedding')
        plt.colorbar(im2, ax=ax2)
        ax2.set_yticks([])
        ax2.set_xlabel('Dimension (128)')
        
        # Plot normalized difference with white-centered colormap
        ax3 = fig.add_subplot(gs[2])
        # Use smaller scale for difference to make it more sensitive
        diff_scale = 0.2  # This means differences > 20% of value range will be full color
        im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto',
                        cmap='RdGy_r', vmin=-diff_scale, vmax=diff_scale)
        ax3.set_title('Normalized Difference (Pred - Target) / Range')
        plt.colorbar(im3, ax=ax3)
        ax3.set_yticks([])
        ax3.set_xlabel('Dimension (128)')
        
        # Plot value distributions
        ax4 = fig.add_subplot(gs[3])
        bins = np.linspace(global_min, global_max, 50)
        n_pred, _, _ = ax4.hist(pred_np.flatten(), bins=bins, alpha=0.5, 
                            label='Predicted', density=True, color='cornflowerblue')
        n_target, _, _ = ax4.hist(target_np.flatten(), bins=bins, alpha=0.5,
                                label='Target', density=True, color='orange')
        ax4.set_title('Value Distributions')
        ax4.legend()
        ax4.grid(True)
        
        # Compute correlation
        correlation = np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]
        
        # Add statistics with normalized differences
        stats_text = (
            f'Max Diff (% of range): {np.abs(diff).max()*100:.2f}%\n'
            f'Mean Diff (% of range): {np.abs(diff).mean()*100:.2f}%\n'
            f'Correlation: {correlation:.4f}\n'
            f'Value Range (Pred): [{pred_np.min():.2f}, {pred_np.max():.2f}]\n'
            f'Value Range (Target): [{target_np.min():.2f}, {target_np.max():.2f}]'
        )
        
        # Add frame losses if provided
        if frame_losses is not None:
            if isinstance(frame_losses, torch.Tensor):
                frame_losses_np = frame_losses.detach().cpu().numpy()
                stats_text += f'\nMean Frame Loss: {frame_losses_np.mean():.4f}'
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
        plt.tight_layout()

        # Convert to wandb image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return wandb.Image(Image.open(buf))

    def create_motion_comparison_plot(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        frame_idx: int,
        step: int,
        is_noise: bool = False
    ) -> wandb.Image:
        """
        Create visualization comparing predicted and target motion parameters for a single frame.
        
        Args:
            pred: Dictionary of predicted motion tensors
            target: Dictionary of target motion tensors
            frame_idx: Index of frame to visualize
            step: Current training step
            is_noise: Whether we're visualizing noise prediction
            
        Returns:
            wandb.Image with the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image

            # Extract frame data for each parameter
            param_data = {
                'theta': {
                    'pred': pred['theta'][:, frame_idx].reshape(-1),
                    'target': target['theta'][:, frame_idx].reshape(-1),
                    'title': 'Pose Matrix (θ)',
                    'dim': 12
                },
                'scale': {
                    'pred': pred['scale'][:, frame_idx],
                    'target': target['scale'][:, frame_idx],
                    'title': 'Scale',
                    'dim': 3
                },
                'rotation': {
                    'pred': pred['rotation'][:, frame_idx],
                    'target': target['rotation'][:, frame_idx],
                    'title': 'Rotation',
                    'dim': 3
                },
                'translation': {
                    'pred': pred['translation'][:, frame_idx],
                    'target': target['translation'][:, frame_idx],
                    'title': 'Translation',
                    'dim': 3
                },
                'expression': {
                    'pred': pred['expression_embed'][:, frame_idx],
                    'target': target['expression_embed'][:, frame_idx],
                    'title': 'Expression',
                    'dim': 128
                }
            }

            # Create figure with subplots for each parameter
            fig = plt.figure(figsize=(20, 25))
            gs = plt.GridSpec(len(param_data), 4, height_ratios=[1] * len(param_data))
            
            # Create title for the entire figure
            prefix = 'Predicted Noise' if is_noise else 'Predicted'
            fig.suptitle(f'{prefix} vs Target Motion Parameters (Frame {frame_idx}, Step {step})', 
                        fontsize=16, y=0.95)

            # Process each parameter
            for i, (param_name, data) in enumerate(param_data.items()):
                # Convert tensors to numpy
                pred_np = data['pred'].detach().cpu().numpy()
                target_np = data['target'].detach().cpu().numpy()
                
                # Compute difference
                diff = pred_np - target_np
                
                # Find global min/max for consistent scale
                global_min = min(pred_np.min(), target_np.min())
                global_max = max(pred_np.max(), target_np.max())
                
                # Create parameter title text
                param_title = f"{data['title']} (dim={data['dim']})"
                
                # 1. Plot predicted values
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.imshow(pred_np.reshape(1, -1), aspect='auto',
                            cmap='viridis', vmin=global_min, vmax=global_max)
                ax1.set_title(f'Predicted {param_title}')
                plt.colorbar(im1, ax=ax1)
                ax1.set_yticks([])
                
                # 2. Plot target values
                ax2 = fig.add_subplot(gs[i, 1])
                im2 = ax2.imshow(target_np.reshape(1, -1), aspect='auto',
                            cmap='plasma', vmin=global_min, vmax=global_max)
                ax2.set_title(f'Target {param_title}')
                plt.colorbar(im2, ax=ax2)
                ax2.set_yticks([])
                
                # 3. Plot difference
                max_diff = max(abs(diff.min()), abs(diff.max()))
                ax3 = fig.add_subplot(gs[i, 2])
                im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto',
                            cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
                ax3.set_title('Difference (Predicted - Target)')
                plt.colorbar(im3, ax=ax3)
                ax3.set_yticks([])
                
                # 4. Plot distributions
                ax4 = fig.add_subplot(gs[i, 3])
                bins = np.linspace(global_min, global_max, 50)
                ax4.hist(pred_np.flatten(), bins=bins, alpha=0.5,
                        label='Predicted', density=True, color='cornflowerblue')
                ax4.hist(target_np.flatten(), bins=bins, alpha=0.5,
                        label='Target', density=True, color='orange')
                ax4.set_title('Value Distributions')
                ax4.legend()
                ax4.grid(True)
                
                # Add statistics text for this parameter
                stats_text = (
                    f'Max Diff: {np.abs(diff).max():.4f}\n'
                    f'Mean Diff: {np.abs(diff).mean():.4f}\n'
                    f'Correlation: {np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]:.4f}\n'
                    f'Value Range (Pred): [{pred_np.min():.2f}, {pred_np.max():.2f}]\n'
                    f'Value Range (Target): [{target_np.min():.2f}, {target_np.max():.2f}]'
                )
                ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Convert to wandb image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plt.close(fig)
            return wandb.Image(Image.open(buf))

        except Exception as e:
            logger.error(f"Error creating motion comparison plot: {str(e)}")
            logger.error(traceback.format_exc())
            plt.close()  # Ensure figure is closed even on error
            return None

    def visualize_sequence(
        self, 
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        step: int,
        is_noise: bool = False,
        b:int = 0 
    ) -> None:
        """
        Create visualizations for a sequence of frames.
        
        Args:
            pred: Dictionary of predicted motion tensors
            target: Dictionary of target motion tensors
            step: Current training step
            is_noise: Whether we're visualizing noise prediction
            max_frames: Maximum number of frames to visualize
        """
        try:
          
          
            wandb_img = self.create_expression_comparison_plot(
                pred[b].detach(),
                target[b].detach(),
                step,
                b
            )
            wandb.log({
                f"expression/comparison_batch_{b}": wandb_img,
            }, step=step)
            
                
        except Exception as e:
            logger.error(f"Error in sequence visualization: {str(e)}")
            logger.error(traceback.format_exc())
            
   
    def _compute_control_losses(
        self,
        pred_motion: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Compute control signal losses with improved signal checking."""
        try:
            logger.debug("\n=== Computing Control Losses ===")
            logger.debug(f"Current epoch: {epoch}, Control start epoch: {self.config.train.control_start_epoch}")
            
            # Initialize loss dict and get device
            device = pred_motion['theta'].device
            losses = {}
            total_loss = torch.tensor(0.0, device=device)
            
            # Log available control signals
            logger.debug("\nChecking available control signals:")
            control_signals = ['gaze', 'head_distance', 'emotion', 'speed_bucket']
            logger.debug("\nChecking available control signals:")
            for signal in control_signals:
                if signal in conditions and conditions[signal] is not None:
                    logger.debug(f"  Found {signal}: shape={conditions[signal].shape}")
                else:
                    logger.debug(f"  Missing or None: {signal}")

            logger.debug(f"\nUsing device: {device}")

            # 1. Gaze Loss
            logger.debug("\nComputing Gaze Loss:")
            if 'gaze' in conditions and conditions['gaze'] is not None:
                pred_gaze = self._extract_gaze_from_motion(pred_motion)
                if pred_gaze is not None:
                    logger.debug(f"  Predicted gaze shape: {pred_gaze.shape}")
                    target_gaze = conditions['gaze'].to(device).float()
                    logger.debug(f"  Target gaze shape: {target_gaze.shape}")
                    logger.debug(f"  Target gaze range: [{target_gaze.min():.3f}, {target_gaze.max():.3f}]")
                    
                    # Ensure gaze has sequence dimension
                    if len(target_gaze.shape) == 2:  # [B, 2]
                        target_gaze = target_gaze.unsqueeze(1).expand(-1, pred_gaze.shape[1], -1)
                        logger.debug(f"  Expanded target gaze shape: {target_gaze.shape}")
                    
                    gaze_loss = (1 - torch.cos(pred_gaze - target_gaze)).mean()
                    losses['control_gaze'] = gaze_loss * self.lambda_gaze
                    total_loss = total_loss + losses['control_gaze']
                    logger.debug(f"  Gaze loss: {losses['control_gaze'].item():.6f}")
                else:
                    logger.debug("  Failed to extract gaze - using zero loss")
                    losses['control_gaze'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No gaze signal found")
                losses['control_gaze'] = torch.tensor(0.0, device=device)

            # 2. Head Distance Loss
            logger.debug("\nComputing Head Distance Loss:")
            if 'head_distance' in conditions and conditions['head_distance'] is not None:
                pred_distance = self._extract_distance_from_motion(pred_motion)
                logger.debug(f"  Predicted distance shape: {pred_distance.shape}")
                
                target_distance = conditions['head_distance'].to(device).float()
                logger.debug(f"  Target distance shape: {target_distance.shape}")
                logger.debug(f"  Target distance range: [{target_distance.min():.3f}, {target_distance.max():.3f}]")
                
                # Ensure distance has sequence dimension
                if len(target_distance.shape) == 2:  # [B, 1]
                    target_distance = target_distance.unsqueeze(1).expand(-1, pred_distance.shape[1], -1)
                    logger.debug(f"  Expanded target distance shape: {target_distance.shape}")
                
                pred_norm = pred_distance / (pred_distance.mean(dim=1, keepdim=True) + 1e-6)
                target_norm = target_distance / (target_distance.mean(dim=1, keepdim=True) + 1e-6)
                
                distance_loss = F.mse_loss(pred_norm, target_norm)
                losses['control_distance'] = distance_loss * self.lambda_distance
                total_loss = total_loss + losses['control_distance']
                logger.debug(f"  Distance loss: {losses['control_distance'].item():.6f}")
            else:
                logger.debug("  No head_distance signal found")
                losses['control_distance'] = torch.tensor(0.0, device=device)

            # 3. Emotion Loss
            logger.debug("\nComputing Emotion Loss:")
            if 'emotion' in conditions and conditions['emotion'] is not None:
                pred_emotion = self._extract_emotion_from_motion(pred_motion)
                logger.debug(f"  Predicted emotion shape: {pred_emotion.shape}")
                
                target_emotion = conditions['emotion'].to(device).float()
                logger.debug(f"  Target emotion shape: {target_emotion.shape}")
                logger.debug(f"  Target emotion range: [{target_emotion.min():.3f}, {target_emotion.max():.3f}]")
                
                # Ensure emotion has sequence dimension
                if len(target_emotion.shape) == 2:  # [B, 2]
                    target_emotion = target_emotion.unsqueeze(1).expand(-1, pred_emotion.shape[1], -1)
                    logger.debug(f"  Expanded target emotion shape: {target_emotion.shape}")
                
                emotion_loss = F.mse_loss(pred_emotion, target_emotion)
                losses['control_emotion'] = emotion_loss * self.lambda_emotion
                total_loss = total_loss + losses['control_emotion']
                logger.debug(f"  Emotion loss: {losses['control_emotion'].item():.6f}")
            else:
                logger.debug("  No emotion signal found")
                losses['control_emotion'] = torch.tensor(0.0, device=device)

             # 4. Speed Loss 
            logger.debug("\nComputing Speed Loss:")
            if 'speed_bucket' in conditions:
                speed_loss, speed_metrics = self.speed_handler.compute_speed_loss(
                    pred_motion=pred_motion,
                    target_buckets=conditions['speed_bucket'],
                    lambda_speed=self.lambda_speed,
                    device=self.device
                )
                losses['control_speed'] = speed_loss
                total_loss = total_loss + speed_loss

                # Convert metrics to tensors and add to losses
                for k, v in speed_metrics.items():
                    losses[f'speed_{k}'] = torch.tensor(v, device=self.device)
                
            else:
                logger.debug("  No speed_bucket signal found")
                losses['control_speed'] = torch.tensor(0.0, device=self.device)
                losses['speed_loss'] = torch.tensor(0.0, device=self.device)
                losses['speed_accuracy'] = torch.tensor(0.0, device=self.device)


            # 4. Blink Loss 
            logger.debug("\nComputing Blink Loss:")
            if 'blink_state' in conditions and conditions['blink_state'] is not None:
                try:
                    blink_loss, blink_metrics = self._compute_blink_loss(
                        pred_motion,
                        conditions['blink_state'].to(device),
                        lambda_blink=self.lambda_blink,
                        device=device
                    )
                    losses.update(blink_metrics)
                    total_loss = total_loss + blink_loss
                    losses['control_blink'] = blink_loss
                    logger.debug(f"  Blink loss: {blink_loss.item():.6f}")
                except Exception as e:
                    logger.error(f"Error computing blink loss: {str(e)}")
                    logger.error(traceback.format_exc())
                    losses['control_blink'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No blink_state signal found")
                losses['control_blink'] = torch.tensor(0.0, device=device)
     

            losses['control_total'] = total_loss


            # Add landmark losses if we have landmark conditions
            landmark_conditions = {k: v for k, v in conditions.items() if 'landmarks' in k}
            if landmark_conditions:
                landmark_losses = self._compute_landmark_losses(pred_motion, landmark_conditions)
                losses.update(landmark_losses)
                
                # Add landmark losses to total control loss
                if 'facial_motion_total' in landmark_losses:
                    losses['control_total'] = losses['control_total'] + landmark_losses['facial_motion_total']
            

            # Log all losses, ensuring they're tensors
            logger.debug("\nControl Loss Summary:")
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: {v.item():.6f}")
                else:
                    logger.debug(f"  {k}: {v:.6f}")
                    losses[k] = torch.tensor(v, device=self.device)

            return losses

        except Exception as e:
            logger.error(f"Error computing control losses: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_zero_losses(device=self.device)
            
    def _compute_motion_speed(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute motion speed logits from motion parameters."""
        B = motion['theta'].shape[0]
        T = motion['theta'].shape[1]
        device = motion['theta'].device
        
        # Get consecutive frame differences
        theta_diff = motion['theta'][:, 1:] - motion['theta'][:, :-1]
        speed = torch.norm(theta_diff.view(B, T-1, -1), dim=-1)  # [B, T-1]
        
        # Add zero for first frame
        speed = torch.cat([
            torch.zeros(B, 1, device=device),
            speed
        ], dim=1)  # [B, T]
        
        # Normalize to [-1, 1] range
        speed = torch.tanh(speed)
        
        # Convert to logits using bucket centers and radius
        centers = torch.tensor([-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0], device=device)
        radius = 0.1
        
        # Calculate distances to bucket centers
        speed = speed.unsqueeze(-1)  # [B, T, 1]
        centers = centers.view(1, 1, -1)  # [1, 1, num_buckets]
        
        # Convert to logits
        logits = -((speed - centers) / radius) ** 2
        
        return logits  # [B, T, num_buckets]

    def _get_zero_losses(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Return dictionary of zero losses."""
        try:
            # Get device from input or model parameters
            if device is None:
                device = self.device
                logger.debug(f"Using model device: {device}")

            logger.debug("Creating zero losses")
            return {
                'lips_total': torch.tensor(0.0, device=device),
                'nonlip_total': torch.tensor(0.0, device=device),
                'facial_motion_total': torch.tensor(0.0, device=device),
                'control_total': torch.tensor(0.0, device=device),
                'control_gaze': torch.tensor(0.0, device=device),
                'control_distance': torch.tensor(0.0, device=device),
                'control_emotion': torch.tensor(0.0, device=device),
                'control_speed': torch.tensor(0.0, device=device)
            }
        except Exception as e:
            logger.error(f"Error creating zero losses: {str(e)}")
            # Default to CPU if all else fails
            logger.warning("Defaulting to CPU device for zero losses")
            return {
                'control_total': torch.tensor(0.0),
                'control_gaze': torch.tensor(0.0),
                'control_distance': torch.tensor(0.0),
                'control_emotion': torch.tensor(0.0),
                'control_speed': torch.tensor(0.0)
            }

    
    def _extract_gaze_from_motion(
        self,
        motion: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Extract gaze angles from motion parameters."""
        try:
            theta = motion['theta']  # [B, T, 3, 4]
            R = theta[..., :3, :3]  # Extract rotation part [B, T, 3, 3]
            
            # Compute pitch (x-axis rotation)
            pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1, 1))
            
            # Compute yaw (y-axis rotation)
            cos_pitch = torch.cos(pitch)
            yaw = torch.atan2(R[..., 2, 1] / cos_pitch, R[..., 2, 2] / cos_pitch)
            
            # Stack gaze angles
            gaze = torch.stack([pitch, yaw], dim=-1)  # [B, T, 2]
            
            # Validate outputs
            if torch.isnan(gaze).any() or torch.isinf(gaze).any():
                logger.warning("Invalid values in gaze angles")
                return None
                
            return gaze
            
        except Exception as e:
            logger.error(f"Error extracting gaze: {str(e)}")
            return None

    def _extract_distance_from_motion(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract head distance from translation."""
        # Use z-translation as proxy for head distance
        translation = motion['translation']  # [B, T, 3]
        z_dist = translation[..., 2:3]  # Get z-component
        
        # Normalize to [0, 1]
        distance = torch.sigmoid(z_dist)
        
        return distance

    def _extract_emotion_from_motion(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract emotion values from expression embeddings."""
        # Project first two dimensions as valence-arousal
        expression = motion['expression_embed']  # [B, T, 128]
        emotion = expression[..., :2]  # [B, T, 2]
        
        # Normalize to [-1, 1]
        emotion = torch.tanh(emotion)
        
        return emotion

    def _compute_motion_speed(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute motion speed from consecutive frames."""
        # Compute pose difference between consecutive frames
        pose_diff = torch.norm(
            motion['theta'][:, 1:] - motion['theta'][:, :-1],
            dim=(-1, -2)
        )
        
        # Add dummy dimension for first frame
        speed = torch.cat([
            torch.zeros_like(pose_diff[:, :1]),
            pose_diff
        ], dim=1)
        
        return speed

    def _compute_angular_loss(
        self,
        pred_angles: torch.Tensor,
        target_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular difference loss considering periodicity.
        Args:
            pred_angles: Predicted angles in radians
            target_angles: Target angles in radians
        Returns:
            Angular difference loss
        """
        # Normalize angles to [-π, π]
        pred_norm = torch.atan2(torch.sin(pred_angles), torch.cos(pred_angles))
        target_norm = torch.atan2(torch.sin(target_angles), torch.cos(target_angles))
        
        # Compute shortest angular distance
        diff = pred_norm - target_norm
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return torch.mean(diff ** 2)

    def _compute_pose_matrix_loss(
        self,
        pred_theta: torch.Tensor,  # [B, T, 3, 4] or [B, 3, 4]
        target_theta: torch.Tensor,  # [B, T, 3, 4] or [B, 3, 4]
        separate_rotation: bool = True
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target pose matrices with proper handling
        of rotation and translation components.
        
        Args:
            pred_theta: Predicted transformation matrices [B, T, 3, 4] or [B, 3, 4]
            target_theta: Target transformation matrices [B, T, 3, 4] or [B, 3, 4]
            separate_rotation: Whether to compute rotation and translation losses separately
            
        Returns:
            Combined pose matrix loss (scalar tensor)
        """
        try:
            # Add time dimension if not present
            if pred_theta.dim() == 3:
                pred_theta = pred_theta.unsqueeze(1)
                target_theta = target_theta.unsqueeze(1)
                
            B, T = pred_theta.shape[:2]
            device = pred_theta.device
            
            if separate_rotation:
                # Extract rotation matrices (3x3)
                pred_R = pred_theta[..., :3, :3]
                target_R = target_theta[..., :3, :3]
                
                # Extract translation vectors
                pred_t = pred_theta[..., :3, 3]
                target_t = target_theta[..., :3, 3]
                
                # Compute rotation loss using geodesic distance
                R_diff = torch.matmul(pred_R, target_R.transpose(-2, -1))
                trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
                cos_theta = (trace - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1, 1)  # Numerical stability
                rotation_loss = torch.acos(cos_theta).mean()
                
                # Compute translation loss (L2)
                translation_loss = F.mse_loss(pred_t, target_t)
                
                # Combine losses with weighting
                total_loss = rotation_loss + 0.5 * translation_loss
                
            else:
                # Direct matrix comparison using Frobenius norm
                matrix_diff = pred_theta - target_theta
                total_loss = torch.norm(matrix_diff.view(B * T, -1), p='fro').mean()


            # Add velocity consistency
            # pred_vel = pred_theta[:, 1:] - pred_theta[:, :-1]
            # target_vel = target_theta[:, 1:] - target_theta[:, :-1]
            # velocity_loss = F.mse_loss(pred_vel, target_vel) * 0.1
            
            # total_loss = rotation_loss + 0.5 * translation_loss + velocity_loss
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing pose matrix loss: {str(e)}")
            logger.error("\nDebug info:")
            logger.error(f"pred_theta shape: {pred_theta.shape}")
            logger.error(f"target_theta shape: {target_theta.shape}")
            logger.error(f"pred_theta device: {pred_theta.device}")
            logger.error(f"target_theta device: {target_theta.device}")
            # Return default loss
            return torch.tensor(0.0, device=device if 'device' in locals() else 'cpu')

    def _compute_geodesic_loss(
        self,
        R1: torch.Tensor,  # [..., 3, 3]
        R2: torch.Tensor   # [..., 3, 3]
    ) -> torch.Tensor:
        """
        Helper function to compute geodesic distance between rotation matrices.
        
        Args:
            R1, R2: Rotation matrices [..., 3, 3]
            
        Returns:
            Geodesic distance loss
        """
        # Compute R1 @ R2.T
        R_diff = torch.matmul(R1, R2.transpose(-2, -1))
        
        # Get trace
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        
        # Compute angle (clamp for numerical stability)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)
        
        return theta.mean()

    def _validate_rotation_matrix(
        self,
        R: torch.Tensor,  # [..., 3, 3]
        eps: float = 1e-6
    ) -> bool:
        """
        Helper function to validate rotation matrix properties.
        
        Args:
            R: Rotation matrix to validate
            eps: Tolerance for numerical comparisons
            
        Returns:
            True if valid rotation matrix
        """
        # Check orthogonality
        I = torch.eye(3, device=R.device).expand_as(R)
        orth_error = torch.norm(
            torch.matmul(R, R.transpose(-2, -1)) - I
        )
        
        # Check determinant
        det = torch.linalg.det(R)
        det_error = torch.abs(det - 1)
        
        return orth_error < eps and det_error < eps



class MotionSequenceHandler:
    """Handles motion sequence processing for VASA."""
    def __init__(
        self,
        window_size: int = 50,       # Main sequence length (T)
        stride: int = 25,            # Window stride
        context_size: int = 10,      # Context length (K)
        min_window_size: int = 15,   # For curriculum learning
        max_window_size: int = 50    # Max window size
    ):
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.overlap_size = window_size - stride
        
        logger.info(f"Initialized MotionSequenceHandler:")
        logger.info(f"  window_size: {window_size}")
        logger.info(f"  stride: {stride}")
        logger.info(f"  context_size: {context_size}")
        logger.info(f"  min_window_size: {min_window_size}")
        logger.info(f"  max_window_size: {max_window_size}")
        logger.info(f"  overlap_size: {self.overlap_size}")


    def process_batch(self, batch: Dict[str, torch.Tensor], current_window_size: Optional[int] = None) -> List[Dict]:
        """Process batch into overlapping windows with adjusted validation."""
        try:
            # Use current window size or default
            window_size = current_window_size or self.window_size
            
            B = batch['frames'].shape[0]  # Get batch size
            T = batch['frames'].shape[1]  # Get sequence length
            logger.debug(f"Processing batch: B={B}, T={T}, window_size={window_size}")

            # Adjusted minimum frames - make context optional
            min_frames = window_size  # Just require window size
            
            if T < min_frames:
                logger.warning(
                    f"Sequence too short: {T} frames, need minimum {min_frames}\n"
                    f"window_size={window_size}"
                )
                return []

            windows = []
            stride = self.stride

            # Process each batch item
            for b in range(B):
                # Adjust window count calculation
                n_windows = 1  # Just one window per sequence if T == window_size
                if T > window_size:
                    n_windows = max(1, (T - window_size) // stride + 1)
                    
                logger.debug(f"Batch {b}: Creating {n_windows} windows")

                for window_idx in range(n_windows):
                    start_frame = window_idx * stride
                    end_frame = start_frame + window_size

                    if end_frame > T:
                        logger.debug(f"Window {window_idx} would exceed sequence length - breaking")
                        break

                    # Create window data preserving batch dimension
                    window_data = {}
                    for key, tensor in batch.items():
                        if isinstance(tensor, torch.Tensor):
                            # Handle different tensor shapes
                            if key == 'audio_features' and len(tensor.shape) == 4:  # [B, 1, T, D]
                                window_data[key] = tensor[b:b+1, :, start_frame:end_frame]
                            else:
                                # Remove extra dims if present
                                if len(tensor.shape) > 3 and tensor.shape[1] == 1:
                                    tensor = tensor.squeeze(1)
                                window_data[key] = tensor[b:b+1, start_frame:end_frame]

                    # Add metadata
                    window_data['metadata'] = {
                        'batch_idx': b,
                        'window_idx': window_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'has_context': window_idx > 0,
                        'total_windows': n_windows,
                        'window_size': window_size
                    }

                    windows.append(window_data)
                    logger.debug(f"  Window {window_idx}: {start_frame} -> {end_frame}")

                    # Log tensor shapes for debugging
                    logger.debug(f"\nWindow {window_idx} tensor shapes:")
                    for k, v in window_data.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"  {k}: {v.shape}")

            if windows:
                logger.info(
                    f"Created {len(windows)} windows\n"
                    f"  Window size: {window_size}\n"
                    f"  First window: 0 -> {window_size}"
                )

            return windows

        except Exception as e:
            logger.error(f"Error in process_batch: {str(e)}")
            logger.error(traceback.format_exc())
            return []
            
    def prepare_motion_data(self, window: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare motion data from window, ensuring batch dimension is preserved."""
        return {
            'theta': window['theta'],            # Should be [B, T, 3, 4]
            'scale': window['scale'],            # Should be [B, T, 3]
            'rotation': window['rotation'],      # Should be [B, T, 3]
            'translation': window['translation'], # Should be [B, T, 3]
            'expression_embed': window['expression_embed']  # Should be [B, T, 128]
        }  

    def merge_windows(self, windows, total_frames, device):
        """Merge overlapping motion sequence windows."""
        # Initialize output tensors
        merged_sequence = {
            'theta': torch.zeros((1, total_frames, 3, 4), device=device),
            'scale': torch.zeros((1, total_frames, 3), device=device),
            'rotation': torch.zeros((1, total_frames, 3), device=device),
            'translation': torch.zeros((1, total_frames, 3), device=device),
            'expression_embed': torch.zeros((1, total_frames, 128), device=device)  # Assuming embed dim is 128
        }
        
        # Initialize weight buffer for blending
        weights = torch.zeros(total_frames, device=device)
        
        # Process each window
        for window_data in windows:
            motion_data = window_data['frames']
            overlap_info = window_data['overlap_info']
            
            start_idx = overlap_info['overlap_start']
            end_idx = overlap_info['overlap_end']
            is_first = overlap_info['is_first']
            is_last = overlap_info['is_last']
            
            # Calculate blend weights
            window_length = end_idx - start_idx
            if is_first:
                window_weights = torch.ones(window_length, device=device)
            elif is_last:
                window_weights = torch.linspace(0, 1, window_length, device=device)
            else:
                window_weights = torch.linspace(0, 1, window_length, device=device)
            
            # Update weights buffer
            weights[start_idx:end_idx] += window_weights
            
            # Add motion sequences with weights
            window_slice = slice(start_idx, end_idx)
            merged_sequence['theta'][:, window_slice] += motion_data['theta'][:, :window_length] * window_weights.view(1, -1, 1, 1)
            merged_sequence['scale'][:, window_slice] += motion_data['scale'][:, :window_length] * window_weights.view(1, -1, 1)

            merged_sequence['rotation'][:, window_slice] += motion_data['rotation'][:, :window_length] * window_weights.view(1, -1, 1)
            merged_sequence['translation'][:, window_slice] += motion_data['translation'][:, :window_length] * window_weights.view(1, -1, 1)
            merged_sequence['expression_embed'][:, window_slice] += motion_data['expression_embed'][:, :window_length] * window_weights.view(1, -1, 1)
        
        # Normalize by weights
        weights = weights.clamp(min=1e-8)  # Avoid division by zero
        merged_sequence['theta'] /= weights.view(1, -1, 1, 1)
        merged_sequence['rotation'] /= weights.view(1, -1, 1)
        merged_sequence['scale'] /= weights.view(1, -1, 1)
        merged_sequence['translation'] /= weights.view(1, -1, 1)
        merged_sequence['expression_embed'] /= weights.view(1, -1, 1)
        
        return merged_sequence

        

    def _merge_theta_matrices(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        weights: torch.Tensor,
        weight_accumulator: torch.Tensor
    ):
        """Specially handle merging of theta (transformation) matrices."""
        # Split rotation and translation
        R1 = target[..., :3, :3]
        t1 = target[..., :3, 3]
        R2 = source[..., :3, :3]
        t2 = source[..., :3, 3]

        # Convert rotations to quaternions for proper interpolation
        q1 = self._matrix_to_quaternion(R1)
        q2 = self._matrix_to_quaternion(R2)

        # Perform SLERP
        dot_product = (q1 * q2).sum(-1)
        q2_adj = torch.where(dot_product < 0, -q2, q2)  # Ensure shortest path
        omega = torch.acos((q1 * q2_adj).sum(-1).clamp(-1, 1))
        sin_omega = torch.sin(omega)

        # Handle small angle case
        mask = sin_omega > 1e-6
        q_interp = torch.where(
            mask.unsqueeze(-1),
            (torch.sin((1 - weights) * omega) / sin_omega).unsqueeze(-1) * q1 +
            (torch.sin(weights * omega) / sin_omega).unsqueeze(-1) * q2_adj,
            q1 + weights.unsqueeze(-1) * (q2_adj - q1)
        )

        # Convert back to rotation matrices
        R_interp = self._quaternion_to_matrix(q_interp)

        # Linear interpolation for translation
        t_interp = t1 + weights.unsqueeze(-1) * (t2 - t1)

        # Update target
        target[..., :3, :3] = R_interp
        target[..., :3, 3] = t_interp
        weight_accumulator += weights

    def _normalize_theta_matrices(
        self,
        theta_matrices: torch.Tensor,
        weights: torch.Tensor
    ):
        """Normalize merged theta matrices ensuring valid rotations."""
        # Handle rotation part
        R = theta_matrices[..., :3, :3]
        U, _, V = torch.svd(R)
        R_normalized = torch.matmul(U, V.transpose(-2, -1))

        # Normalize translation part
        t = theta_matrices[..., :3, 3] / (weights.unsqueeze(-1) + 1e-8)

        # Reconstruct normalized matrices
        theta_matrices[..., :3, :3] = R_normalized
        theta_matrices[..., :3, 3] = t
        theta_matrices[..., 3, 3] = 1.0

    def _get_blend_weights(
        self, 
        window_idx: int, 
        total_windows: int,
        start_idx: int,
        window_size: int
    ) -> torch.Tensor:
        """Calculate blending weights for window transitions."""
        if window_idx == 0:  # First window
            weights = torch.cat([
                torch.ones(start_idx + self.stride),
                torch.linspace(1, 0, self.overlap_size)
            ])
        elif window_idx == total_windows - 1:  # Last window
            weights = torch.cat([
                torch.linspace(0, 1, self.overlap_size),
                torch.ones(window_size - self.overlap_size)
            ])
        else:  # Middle windows
            weights = torch.cat([
                torch.linspace(0, 1, self.overlap_size),
                torch.ones(self.stride),
                torch.linspace(1, 0, self.overlap_size)
            ])
            
        return weights

    def _interpolate_rotations(
        self,
        rot1: torch.Tensor,
        rot2: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate rotation matrices using SLERP."""
        # Convert to quaternions
        quat1 = self._matrix_to_quaternion(rot1)
        quat2 = self._matrix_to_quaternion(rot2)
        
        # Compute dot product
        dot = torch.sum(quat1 * quat2, dim=-1, keepdim=True)
        
        # If dot < 0, negate one of the inputs to take shorter interpolation path
        flip_mask = (dot < 0).float()
        quat2 = quat2 * (1 - 2 * flip_mask)
        
        # SLERP interpolation
        theta = torch.acos(torch.clamp(dot, -1, 1))
        sin_theta = torch.sin(theta)
        
        # Handle small angle case
        mask = (sin_theta > 1e-6).float()
        t = weights.unsqueeze(-1)
        
        interpolated = torch.zeros_like(quat1)
        interpolated = mask * (
            torch.sin((1-t) * theta) / sin_theta * quat1 +
            torch.sin(t * theta) / sin_theta * quat2
        ) + (1 - mask) * (quat1 + t * (quat2 - quat1))
        
        # Convert back to rotation matrix
        return self._quaternion_to_matrix(interpolated)

    def _matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert batch of 3x3 rotation matrices to quaternions."""
        m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
        m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
        m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
        
        trace = m00 + m11 + m22
        
        def when_trace_positive():
            r = torch.sqrt(1 + trace)
            s = 0.5 / r
            return torch.stack([
                0.5 * r,
                (m21 - m12) * s,
                (m02 - m20) * s,
                (m10 - m01) * s
            ], dim=-1)
            
        def when_m00_largest():
            r = torch.sqrt(1 + m00 - m11 - m22)
            s = 0.5 / r
            return torch.stack([
                (m21 - m12) * s,
                0.5 * r,
                (m01 + m10) * s,
                (m02 + m20) * s
            ], dim=-1)
            
        def when_m11_largest():
            r = torch.sqrt(1 - m00 + m11 - m22)
            s = 0.5 / r
            return torch.stack([
                (m02 - m20) * s,
                (m01 + m10) * s,
                0.5 * r,
                (m12 + m21) * s
            ], dim=-1)
            
        def when_m22_largest():
            r = torch.sqrt(1 - m00 - m11 + m22)
            s = 0.5 / r
            return torch.stack([
                (m10 - m01) * s,
                (m02 + m20) * s,
                (m12 + m21) * s,
                0.5 * r
            ], dim=-1)
        
        # Choose appropriate conversion based on largest diagonal element
        where_trace_positive = trace > 0
        where_m00_largest = (m00 > m11) & (m00 > m22) & ~where_trace_positive
        where_m11_largest = (m11 > m00) & (m11 > m22) & ~where_trace_positive
        where_m22_largest = (m22 >= m00) & (m22 >= m11) & ~where_trace_positive
        
        quaternion = torch.zeros(matrix.shape[:-2] + (4,), device=matrix.device)
        quaternion = torch.where(where_trace_positive.unsqueeze(-1), when_trace_positive(), quaternion)
        quaternion = torch.where(where_m00_largest.unsqueeze(-1), when_m00_largest(), quaternion)
        quaternion = torch.where(where_m11_largest.unsqueeze(-1), when_m11_largest(), quaternion)
        quaternion = torch.where(where_m22_largest.unsqueeze(-1), when_m22_largest(), quaternion)
        
        return quaternion

    def _quaternion_to_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert batch of quaternions to 3x3 rotation matrices."""
        qx, qy, qz, qw = torch.unbind(quaternion, dim=-1)
        
        # Compute matrix elements
        m00 = 1 - 2 * (qy**2 + qz**2)
        m01 = 2 * (qx*qy - qz*qw)
        m02 = 2 * (qx*qz + qy*qw)
        
        m10 = 2 * (qx*qy + qz*qw)
        m11 = 1 - 2 * (qx**2 + qz**2)
        m12 = 2 * (qy*qz - qx*qw)
        
        m20 = 2 * (qx*qz - qy*qw)
        m21 = 2 * (qy*qz + qx*qw)
        m22 = 1 - 2 * (qx**2 + qy**2)
        
        # Stack into rotation matrix
        matrix = torch.stack([
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22
        ], dim=-1).view(quaternion.shape[:-1] + (3, 3))
        
        return matrix

class SpeedLossHandler:
    """Handles speed loss computation and bucketing."""
    def __init__(self, num_buckets=9):
        # Centers for speed buckets from -1.0 to 1.0 
        self.centers = torch.tensor([
            -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0
        ])
        self.radius = 0.1  # Fixed radius for each bucket
        self.num_buckets = num_buckets

    def _get_bucket_index(self, speed_value: float) -> int:
        """Get bucket index for a single speed value."""
        # Find closest center
        distances = torch.abs(self.centers - speed_value)
        return torch.argmin(distances).item()

    def compute_motion_speed(self, motion_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute frame-to-frame rotation speeds and convert to bucket indices.
        Returns tensor of bucket indices [B, T].
        """
        try:
            logger.debug("\n=== Computing Motion Speed ===")
            
            B = motion_data['theta'].shape[0]
            T = motion_data['theta'].shape[1]
            device = motion_data['theta'].device

            # Get consecutive frame differences
            theta_diff = motion_data['theta'][:, 1:] - motion_data['theta'][:, :-1]
            raw_speed = torch.norm(theta_diff.view(B, T-1, -1), dim=-1)  # [B, T-1]
            logger.debug(f"Raw speed shape: {raw_speed.shape}")
            logger.debug(f"Raw speed range: [{raw_speed.min():.3f}, {raw_speed.max():.3f}]")

            # Add zero for first frame
            raw_speed = torch.cat([
                torch.zeros(B, 1, device=device),
                raw_speed
            ], dim=1)  # [B, T]

            # Normalize to [-1, 1] range
            speed = torch.tanh(raw_speed)
            logger.debug(f"Normalized speed range: [{speed.min():.3f}, {speed.max():.3f}]")

            # Move centers to device
            centers = self.centers.to(device)

            # Calculate distances to all centers
            speed_expanded = speed.unsqueeze(-1)  # [B, T, 1]
            centers_expanded = centers.view(1, 1, -1)  # [1, 1, num_buckets]
            distances = torch.abs(speed_expanded - centers_expanded)

            # Get bucket indices
            bucket_indices = torch.argmin(distances, dim=-1)  # [B, T]
            logger.debug(f"Bucket indices shape: {bucket_indices.shape}")
            logger.debug(f"Bucket range: [{bucket_indices.min()}, {bucket_indices.max()}]")

            # Sample logging
            if B > 0 and T > 0:
                example_speed = speed[0, 0].item()
                example_bucket = bucket_indices[0, 0].item()
                logger.debug(f"\nExample mapping:")
                logger.debug(f"  Speed value: {example_speed:.3f}")
                logger.debug(f"  Assigned bucket: {example_bucket}")
                logger.debug(f"  Center value: {centers[example_bucket]:.3f}")

            return bucket_indices

        except Exception as e:
            logger.error(f"Error computing motion speed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_speed_loss(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_buckets: Optional[torch.Tensor],
        lambda_speed: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute speed loss comparing predicted and target buckets."""
        try:
            logger.debug("\n=== Computing Speed Loss ===")
            
            # Early return if no target buckets provided
            if target_buckets is None:
                logger.debug("No target buckets provided, returning zero loss")
                return torch.tensor(0.0, device=device), {
                    'speed_loss': 0.0,
                    'speed_accuracy': 0.0
                }

            # Validate target buckets tensor
            if not isinstance(target_buckets, torch.Tensor):
                logger.warning(f"Invalid target_buckets type: {type(target_buckets)}")
                return torch.tensor(0.0, device=device), {
                    'speed_loss': 0.0,
                    'speed_accuracy': 0.0
                }

            # Move tensor to correct device and format
            target_buckets = target_buckets.to(device).long()
            if target_buckets.dim() == 3:  # If shape is [B,T,1]
                target_buckets = target_buckets.squeeze(-1)

            # Get batch and sequence dimensions
            B, T = target_buckets.shape
            logger.debug(f"Target buckets shape: {target_buckets.shape}")
            logger.debug(f"Target bucket range: [{target_buckets.min()}, {target_buckets.max()}]")

            # Verify pred_motion has required tensors
            if not all(k in pred_motion for k in ['theta', 'rotation', 'translation']):
                logger.warning("Missing required motion parameters")
                return torch.tensor(0.0, device=device), {
                    'speed_loss': 0.0,
                    'speed_accuracy': 0.0
                }

            # Get predicted buckets
            try:
                pred_buckets = self.compute_motion_speed(pred_motion)
            except Exception as e:
                logger.error(f"Error computing motion speed: {str(e)}")
                return torch.tensor(0.0, device=device), {
                    'speed_loss': 0.0,
                    'speed_accuracy': 0.0
                }

            logger.debug(f"Predicted buckets shape: {pred_buckets.shape}")
            logger.debug(f"Predicted bucket range: [{pred_buckets.min()}, {pred_buckets.max()}]")

            # Ensure tensors match in shape
            if pred_buckets.shape != target_buckets.shape:
                logger.error(f"Shape mismatch: pred={pred_buckets.shape}, target={target_buckets.shape}")
                return torch.tensor(0.0, device=device), {
                    'speed_loss': 0.0,
                    'speed_accuracy': 0.0
                }

            # Compute loss - simple MSE on bucket indices
            loss = F.mse_loss(
                pred_buckets.float(),
                target_buckets.float()
            )
            scaled_loss = loss * lambda_speed

            # Compute accuracy
            accuracy = (pred_buckets == target_buckets).float().mean()

            # Create metrics
            metrics = {
                'speed_loss': loss.item(),
                'speed_accuracy': accuracy.item()
            }

            logger.debug(f"\nSpeed metrics:")
            logger.debug(f"  Raw loss: {loss.item():.6f}")
            logger.debug(f"  Scaled loss: {scaled_loss.item():.6f}")
            logger.debug(f"  Accuracy: {accuracy.item():.3f}")

            return scaled_loss, metrics

        except Exception as e:
            logger.error(f"Error computing speed loss: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=device), {
                'speed_loss': 0.0,
                'speed_accuracy': 0.0
            }
        

class BlinkConditionHandler:
    """Handles blink state processing for facial landmarks."""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        # Typical blink parameters (in frames at 30fps)
        self.blink_freq = 0.08  # Probability of starting a blink per frame (~every 2-3 seconds)
        self.close_duration = 2  # Frames to close eye
        self.hold_duration = 1   # Frames to hold closed
        self.open_duration = 3   # Frames to open eye
        self.total_duration = self.close_duration + self.hold_duration + self.open_duration

    def generate_blink_sequence(self, sequence_length: int) -> torch.Tensor:
        """
        Generate a sequence of blink states.
        Returns: Tensor [sequence_length, 3] containing:
            - Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
            - Channel 1: Left eye openness (0-1)
            - Channel 2: Right eye openness (0-1)
        """
        states = torch.zeros(sequence_length, 3)
        states[:, 1:] = 1.0  # Start with eyes fully open
        
        current_frame = 0
        while current_frame < sequence_length:
            if torch.rand(1).item() < self.blink_freq and current_frame + self.total_duration < sequence_length:
                # Generate blink sequence
                # Closing phase
                for i in range(self.close_duration):
                    t = i / (self.close_duration - 1)
                    openness = 1 - t
                    frame = current_frame + i
                    states[frame, 0] = 1  # Closing phase
                    states[frame, 1:] = openness
                
                # Hold phase
                for i in range(self.hold_duration):
                    frame = current_frame + self.close_duration + i
                    states[frame, 0] = 2  # Closed phase
                    states[frame, 1:] = 0
                
                # Opening phase
                for i in range(self.open_duration):
                    t = i / (self.open_duration - 1)
                    openness = t
                    frame = current_frame + self.close_duration + self.hold_duration + i
                    states[frame, 0] = 3  # Opening phase
                    states[frame, 1:] = openness
                
                current_frame += self.total_duration
            else:
                states[current_frame, 0] = 0  # Open phase
                states[current_frame, 1:] = 1.0  # Fully open
                current_frame += 1
        
        return states

    def apply_blink_to_landmarks(
        self, 
        landmarks_dict: Dict[str, torch.Tensor],
        blink_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply blink states to eye landmarks.
        
        Args:
            landmarks_dict: Dictionary containing eye landmarks
            blink_states: [B, T, 3] tensor of blink states
            
        Returns:
            Updated landmarks with blink applied
        """
        out_dict = {}
        B, T = blink_states.shape[:2]
        
        # Process left eye
        if 'left_eye' in landmarks_dict:
            left_eye = landmarks_dict['left_eye']  # [B, T, N, 3]
            left_mean = left_eye.mean(dim=2, keepdim=True)  # Mean position (closed state)
            left_openness = blink_states[..., 1].unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            out_dict['left_eye'] = (
                left_eye * left_openness + 
                left_mean * (1 - left_openness)
            )
            
        # Process right eye
        if 'right_eye' in landmarks_dict:
            right_eye = landmarks_dict['right_eye']
            right_mean = right_eye.mean(dim=2, keepdim=True)
            right_openness = blink_states[..., 2].unsqueeze(-1).unsqueeze(-1)
            out_dict['right_eye'] = (
                right_eye * right_openness + 
                right_mean * (1 - right_openness)
            )
            
        # Copy other landmarks unchanged
        for k, v in landmarks_dict.items():
            if k not in out_dict:
                out_dict[k] = v
                
        return out_dict 
