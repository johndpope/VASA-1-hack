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

# Import loss classes from vasa_losses
from vasa_losses import VASALossModule, SpeedLossHandler

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
        
        # Store prev_context dimensions from config
        self.prev_expression_dim = config.dimensions.prev_expression
        self.prev_audio_dim = config.dimensions.prev_audio
        logger.info(f"Set prev_context dimensions: expression={self.prev_expression_dim}, audio={self.prev_audio_dim}")

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

            # 5. Previous context is now handled directly in the transformer
            # No need to compress it into the condition embedding

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
        self.num_layers = 2  # Reduced from 8 to 2 for faster convergence testing
        
        # Initialize condition embedding
        self.cond_embed = EfficientConditionEmbedding(
            model_dim=self.transformer_dim,
            max_seq_len=self.window_size
        )
        
        # Motion input projections with explicit batch and sequence handling
        # Simplified to single MotionResidualBlock for efficiency
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
            nn.ReLU(),  # Added ReLU for better feature combination
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
        
        self.gradient_checkpointing = False  # Disabled for speed testing

    def _validate_context(self, prev_context: Optional[Dict[str, torch.Tensor]]) -> None:
        """Validate that prev_context contains exactly context_size frames."""
        if prev_context is not None:
            expected_keys = ['theta', 'rotation', 'translation', 'expression_embed']
            context_size = getattr(self.config.training, 'context_size', 10)
            
            for key in expected_keys:
                assert key in prev_context, f"Missing {key} in prev_context"
                
                # Special handling for theta which is 4D [B, T, 3, 4]
                if key == 'theta':
                    assert prev_context[key].dim() == 4, f"Expected 4D tensor for {key}, got {prev_context[key].dim()}D"
                    assert prev_context[key].shape[1] == context_size, (
                        f"Expected exactly {context_size} context frames for {key}, "
                        f"got {prev_context[key].shape[1]} frames. "
                        f"Full shape: {prev_context[key].shape}"
                    )
                else:
                    assert prev_context[key].dim() == 3, f"Expected 3D tensor for {key}, got {prev_context[key].dim()}D"
                    assert prev_context[key].shape[1] == context_size, (
                        f"Expected exactly {context_size} context frames for {key}, "
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

            # Handle previous context if provided
            device = motion_data['theta'].device
            if prev_context is not None:
                # Process previous context frames through same motion projections
                K = prev_context['theta'].shape[1]  # Number of context frames
                prev_features = self._project_motion_parameters(prev_context, B, K)
                logger.debug(f"Previous context features shape: {prev_features.shape}")
            else:
                prev_features = None
                K = 0

            self._validate_conditions(conditions, B, T, device)

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
            
            # Concatenate previous context with current features if available
            if prev_features is not None:
                # Concatenate along sequence dimension: [B, K+T, D]
                x = torch.cat([prev_features, motion_features], dim=1)
                total_seq_len = K + T
                logger.debug(f"Combined features shape (with context): {x.shape}")
            else:
                x = motion_features
                total_seq_len = T

            # Add time embeddings (expand to match sequence length)
            time_emb = self._time_embedding(noise_level, self.transformer_dim)
            x = x + time_emb.unsqueeze(1).expand(-1, total_seq_len, -1)

            # Add condition embeddings if present (only to current frames, not context)
            if cond_emb is not None:
                if prev_features is not None:
                    # Only add conditions to current frames, not context
                    # Pad conditions with zeros for context frames
                    cond_emb_padded = torch.cat([
                        torch.zeros(B, K, self.transformer_dim, device=device),
                        cond_emb
                    ], dim=1)
                    x = x + cond_emb_padded
                else:
                    x = x + cond_emb
                logger.debug("Added condition embeddings")
            
            # Add positional embeddings to distinguish context from current
            pos_emb = self._get_positional_embedding(total_seq_len, self.transformer_dim, device)
            if prev_features is not None:
                # Use negative positions for context, positive for current
                context_positions = torch.arange(-K, 0, device=device)
                current_positions = torch.arange(0, T, device=device)
                all_positions = torch.cat([context_positions, current_positions])
                pos_emb = self._compute_sinusoidal_embedding(all_positions, self.transformer_dim)
                x = x + pos_emb.unsqueeze(0).expand(B, -1, -1)
            else:
                x = x + pos_emb.unsqueeze(0).expand(B, -1, -1)

            # Process through transformer blocks
            logger.debug("\nProcessing through transformer blocks...")
            for i, block in enumerate(self.transformer):
                x = block(x)
                if not torch.isfinite(x).all():
                    raise ValueError(f"Non-finite values detected after transformer block {i}")
                logger.debug(f"Block {i} output shape: {x.shape}")

            # Extract only the current frame outputs (not context)
            if prev_features is not None:
                # Take only the last T frames (current window)
                x_current = x[:, K:, :]  # [B, T, D]
            else:
                x_current = x

            # Project outputs back to motion parameters
            logger.debug("\nProjecting outputs...")
            outputs = self._project_outputs(x_current, B, T)
            
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
    
    def _get_positional_embedding(self, seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
        """Get standard positional embedding for sequence."""
        positions = torch.arange(seq_len, device=device)
        return self._compute_sinusoidal_embedding(positions, dim)
    
    def _compute_sinusoidal_embedding(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal positional embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=positions.device) * -emb)
        emb = positions[:, None] * emb[None, :]
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
        # Initialize DDIM scheduler with reduced steps for faster convergence
        from diffusers import DDIMScheduler

        
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_steps,  # Reduced for speed
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=True,
            #  clip_sample_range=1.0,  # Add this for motion parameters
            prediction_type="sample",  # VASA-1 predicts clean signal X0, not noise
            timestep_spacing="leading"  # Important for proper timestep spacing
        )
        # Set default inference steps
        self.scheduler.set_timesteps(num_steps)
        
        # Store prediction type for VASA-1 style score matching
        self.use_score_matching = True

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
        num_steps: int = 50,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
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
                    conditions=conditions,
                    prev_context=prev_context
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

            logger.debug("\nGenerating conditional output...")
            logger.debug(f"Active conditions: {list(filtered_conditions.keys())}")
            
            # Forward pass with conditions
            cond_output = self.forward(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=filtered_conditions,
                prev_context=prev_context
            )
            
            logger.debug("\nGenerating unconditional output...")
            # Forward pass without conditions
            uncond_output = self.forward(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=uncond_conditions,
                prev_context=prev_context
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
        cfg_scales: Optional[Dict[str, float]] = None,
        prev_context: Optional[Dict[str, torch.Tensor]] = None
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
                # Add stronger noise to encourage variation
                noise_scale = 2.0  # Increase initial noise to prevent collapse
                
                # Add temporal structure to noise to encourage motion
                base_noise = torch.randn(B, T, 1, device=device)
                temporal_modulation = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(-1)
                temporal_noise = base_noise * (1 + temporal_modulation * 0.5)  # Gradual increase
                
                motion_sequence = {
                    'theta': torch.randn(B, T, 3, 4, device=device) * noise_scale,
                    'scale': torch.randn(B, T, 3, device=device) * 0.1,  # Keep scale small
                    'rotation': torch.randn(B, T, 3, device=device) * noise_scale,
                    'translation': torch.randn(B, T, 3, device=device) * 0.1,  # Keep translation small
                    'expression_embed': torch.randn(B, T, 128, device=device) * noise_scale * (1 + temporal_modulation)
                }

                # Set initial frame values
                # Handle both single frame [B, ...] and sequence [B, T, ...] inputs
                if initial_pose['theta'].dim() == 3:  # [B, 3, 4]
                    motion_sequence['theta'][:, 0] = initial_pose['theta']
                    motion_sequence['rotation'][:, 0] = initial_pose['rotation']
                    motion_sequence['scale'][:, 0] = initial_pose['scale']
                    motion_sequence['translation'][:, 0] = initial_pose['translation']
                else:  # [B, T, 3, 4] - take first frame
                    motion_sequence['theta'][:, 0] = initial_pose['theta'][:, 0]
                    motion_sequence['rotation'][:, 0] = initial_pose['rotation'][:, 0]
                    motion_sequence['scale'][:, 0] = initial_pose['scale'][:, 0]
                    motion_sequence['translation'][:, 0] = initial_pose['translation'][:, 0]
                
                # Handle initial dynamics shape
                if initial_dynamics.dim() == 2:  # [B, D]
                    motion_sequence['expression_embed'][:, 0] = initial_dynamics
                else:  # [B, T, D] - take first frame
                    motion_sequence['expression_embed'][:, 0] = initial_dynamics[:, 0]

                # DDIM sampling loop
                for i, t in enumerate(self.scheduler.timesteps):
                    # Get model prediction with CFG
                    model_output = self.forward_with_cfg(
                        motion_data=motion_sequence,
                        noise_level=t.expand(B),
                        conditions=conditions,
                        cfg_scales=cfg_scales,
                        prev_context=prev_context  # Pass prev_context for temporal consistency
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
                    'expression_embed': initial_dynamics.unsqueeze(1).expand(-1, T, -1)
                }

                # Set up DDIM sampler
                self.scheduler.set_timesteps(50, device=device)

                # Get eta from config for stochasticity (VASA-1 uses mild randomness)
                eta = getattr(self.config.inference, 'eta', 0.1)  # Default 0.1 for mild stochasticity
                logger.info(f"Using eta={eta} for DDIM sampling (0=deterministic, 1=full stochastic)")
                
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
                            # Use scheduler step with stochasticity
                            scheduler_output = self.scheduler.step(
                                model_output=model_output[key],
                                timestep=t,
                                sample=motion_sequence[key],
                                eta=eta  # Add randomness per VASA-1
                            )
                            motion_sequence[key] = scheduler_output.prev_sample

                # Log variance to detect mode collapse
                logger.info("=== Motion Variance Analysis ===")
                for key in ['theta', 'rotation', 'translation', 'expression_embed']:
                    if key in motion_sequence:
                        motion = motion_sequence[key]
                        # Compute variance across time
                        temporal_var = motion.var(dim=1).mean().item()
                        # Compute frame-to-frame differences
                        if motion.shape[1] > 1:
                            frame_diffs = motion[:, 1:] - motion[:, :-1]
                            diff_norm = torch.norm(frame_diffs.reshape(B, -1), dim=-1).mean().item()
                        else:
                            diff_norm = 0.0
                        
                        logger.info(f"  {key}: temporal_var={temporal_var:.6f}, frame_diff_norm={diff_norm:.6f}")
                        
                        # WARNING if variance is too low (mode collapse)
                        if temporal_var < 1e-4:
                            logger.warning(f"  ⚠️ LOW VARIANCE in {key} - possible mode collapse!")
                        if diff_norm < 1e-3:
                            logger.warning(f"  ⚠️ STATIC MOTION in {key} - frames are too similar!")
                
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
        motion_data = {
            'theta': window['theta'],            # Should be [B, T, 3, 4]
            'scale': window['scale'],            # Should be [B, T, 3]
            'rotation': window['rotation'],      # Should be [B, T, 3]
            'translation': window['translation'], # Should be [B, T, 3]
            'expression_embed': window['expression_embed']  # Should be [B, T, 128]
        }
        
        # Include audio features for sync loss and other audio-related losses
        if 'audio_features' in window:
            motion_data['audio_features'] = window['audio_features']  # Should be [B, T, D]
        if 'mfcc' in window:
            motion_data['mfcc'] = window['mfcc']  # MFCC features for SyncNet
        
        return motion_data  

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
