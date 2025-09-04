# VASA Training Debug Context

Generated on: Thu Sep  4 02:03:20 PM AEST 2025

## Problem Description

### Main Issue
The VASA model training has a **loss stuck at 1.0** problem during training, even though:
1. The loss weights have been fixed (lambda_pose=1.0, lambda_dynamics=1.0 instead of 0)
2. Test scripts show loss computation works correctly (~3.06)
3. The model architecture is correct

### Symptoms
- Training loss returns exactly 1.0 every epoch
- This is the default error value returned when loss computation fails
- TDD tests correctly detect this as a failure and stop training
- The test_loss_computation.py script shows losses compute correctly (~3.06)

### What We're Trying to Do
- Overfit the model on a single video (junk/10.mp4 and variants)
- This is a standard debugging technique to verify the model can learn

### Files Already Fixed
1. **vasa_config_fixed.yaml**: Set lambda_pose and lambda_dynamics from 0 to 1.0
2. **vasa_trainer.py**: Added handling for non-windowed data in collate_vasa_batch
3. Created 50-frame and 100-frame videos to match model expectations

### Current Status
- Loss computation test works: ✅
- Training runs without crashes: ✅
- Loss decreases during training: ❌ (stuck at 1.0)
- Model learns/overfits: ❌


---

# Configuration Files

## File: vasa_config_fixed.yaml
Fixed configuration with correct loss weights
```yaml
# VASA Configuration - FIXED for TDD Training
# Based on test failure analysis from initial run

defaults:
  - _self_

# Model architecture (unchanged)
model:
  hidden_dim: 512
  n_heads: 8
  n_layers: 8
  dim_feedforward: 2048
  dropout: 0.1
  motion_dim: 256
  condition_embedding_dim: 512
  use_prev_motion: true
  max_motion_length: 1000
  use_relative_position: true

# Diffusion parameters - FIXED
diffusion:
  num_steps: 1000
  beta_start: 1e-4
  beta_end: 0.02
  cfg_start_epoch: 30  # Delayed from 5 to 30
  cfg_ramp_epochs: 10
  schedule_mode: 'cosine'
  schedule_s: 0.008

# Motion generation - CURRICULUM LEARNING
motion:
  num_speed_buckets: 9
  # Start with single frames, gradually increase
  window_size: 1  # Changed from 50 to 1
  stride: 1  # Changed from 25 to 1
  context_size: 0  # No context initially
  overlap_smooth: false  # Disable initially
  min_window_size: 1
  max_window_size: 50
  smooth_window: 5
  use_motion_prior: false  # Disable initially
  
  warmup_steps: 500  # Increased from 100
  scheduler:
    type: "cosine"
    min_lr: 1e-6
    warmup_ratio: 0.2  # Increased warmup
  
  # Progressive training schedule
  progressive:
    enabled: true
    start_sequence_length: 1  # Start with single frames
    max_sequence_length: 50
    length_increase_freq: 10  # Increase every 10 epochs
    
  amp: true
  scaler_growth: 2.0

# Training parameters - MAJOR FIXES
train:
  resume_from: ""
  
  # CRITICAL FIX: Delay control signals significantly
  control_start_epoch: 30  # Changed from 1750 to 30 (after basics work)
  
  # CRITICAL FIX: Higher learning rate
  lr: 5e-3  # Increased from 1e-3 to 5e-3
  motion_proj_lr: 2e-3  # Adjusted
  expression_lr: 2e-3  # Adjusted
  
  # CRITICAL FIX: No noise initially
  turn_off_noise: true  # Changed to true for initial training
  
  learning_rates:
    condition_embedding: 5e-3  # All increased 5x
    motion_projections: 5e-3
    transformer_early: 5e-3
    transformer_late: 2e-3
    output_projections: 5e-3

  batch_size: 1
  gradient_accumulation_steps: 1  # Reduced from 4 for faster feedback
  num_epochs: 100  # Reduced from 4000 for testing
  
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.001  # Reduced from 0.01
  max_grad_norm: 5.0  # Increased from 1.0
  save_freq: 5
  
  # Dropout - disabled initially
  dropout_probs:
    audio: 0.0  # All set to 0 initially
    gaze: 0.0
    head_distance: 0.0
    emotion: 0.0

  # CFG scales - disabled initially
  cfg_scales:
    audio: 0.0
    gaze: 0.0
    head_distance: 0.0
    emotion: 0.0

  gradient_monitoring:
    enabled: true
    log_freq: 10  # More frequent monitoring
    save_grad_norms: true
    save_grad_flows: true
    num_layers_to_monitor: 8

# Loss weights - STAGE-BASED ADJUSTMENTS
loss:
  expression_weighting:
    enabled: false  # Disable initially
    min_weight: 0.5
    
  use_sync_loss: false  # Disable initially
  use_verification: false  # Disable initially
  lambda_verification: 0
  
  # CRITICAL: Prioritize reconstruction
  lambda_reconstruction: 1.0  # Base reconstruction weight
  
  # Enable core losses for reconstruction
  lambda_lips: 0
  lambda_nonlip: 0
  lambda_speed: 0
  lambda_pose: 1.0  # FIXED: Must be non-zero for pose losses to count
  lambda_dynamics: 1.0  # FIXED: Must be non-zero for expression losses to count
  lambda_sync: 0
  lambda_control: 0
  lambda_lip: 0
  lambda_gaze_direction: 0
  lambda_head_distance: 0
  lambda_emotion: 0
  lambda_cfg: 0
  lambda_perceptual: 0
  lambda_temporal: 0
  lambda_blink: 0
  
  warmup_epochs: 0
  
  # Progressive loss schedule
  schedule:
    enabled: true
    stages:
      # Stage 1: Pure reconstruction (epochs 0-10)
      stage1:
        start_epoch: 0
        lambda_reconstruction: 10.0
        
      # Stage 2: Add dynamics (epochs 10-20)
      stage2:
        start_epoch: 10
        lambda_reconstruction: 5.0
        lambda_dynamics: 2.0
        lambda_temporal: 1.0
        
      # Stage 3: Add control (epochs 20-30)
      stage3:
        start_epoch: 20
        lambda_reconstruction: 2.0
        lambda_dynamics: 3.0
        lambda_temporal: 2.0
        lambda_pose: 1.0
        
      # Stage 4: Full training (epochs 30+)
      stage4:
        start_epoch: 30
        lambda_reconstruction: 1.0
        lambda_dynamics: 2.0
        lambda_temporal: 2.0
        lambda_control: 1.0
        lambda_gaze_direction: 0.5
        lambda_emotion: 0.5

# Inference parameters
inference:
  batch_size: 1
  cfg_scale: 0.0  # No CFG initially
  eta: 0.0
  output_size: [512, 512]
  num_inference_steps: 50
  temperature: 1.0
  top_k: 50
  top_p: 0.9
  use_window_cache: true

# Dataset parameters
dataset:
  sequence_length: 1  # Start with single frames
  frame_size: [512, 512]
  preextract_audio: true
  cache_audio: true
  hop_length: 1  # Minimal hop initially
  max_videos: 5  # Increased from 2 for more data
  min_sequence_length: 1
  max_sequence_samples: 100  # Small batches for fast iteration
  val_split: 0.2
  
  # Disable augmentation initially
  augmentation:
    enabled: false
    flip_prob: 0.0
    temporal_crop_prob: 0.0
    intensity_scale: 0.0
    temporal_mask_prob: 0.0

# Paths
paths:
  checkpoint_dir: "checkpoints"
  syncnet_path: "pretrained/syncnet.pth"
  emotion_model: "pretrained/emonet.pth"
  volumetric_model: "nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth"
  volumetric_config: "nemo/models/stage_1/volumetric_avatar/va.yaml"
  data_dir: "data"
  video_folder: "junk"
  cache_dir: "cache"

# Hardware/environment
device: "cuda"
seed: 42
num_workers: 0
debug: true  # Enable debug mode

# Wandb
wandb:
  project: "vasa"
  enabled: true
  name: "vasa_tdd_fixed"

# TDD Configuration - PROGRESSIVE TARGETS
tdd:
  enabled: true
  max_critical_failures: 5  # Reduced for faster iteration
  regression_tolerance: 3
  
  # Progressive targets based on epoch
  targets:
    # Initial targets (epochs 0-10) - Focus on reconstruction
    epoch_0:
      psnr: 20.0  # Very achievable
      ssim: 0.6
      lpips: 0.4
      static_loss: 0.5  # Achievable target
      
    # After reconstruction works (epochs 10-20)
    epoch_10:
      psnr: 23.0
      ssim: 0.7
      lpips: 0.3
      static_loss: 0.2
      dynamics_loss: 0.3
      
    # After basic motion (epochs 20-30)
    epoch_20:
      psnr: 25.0
      ssim: 0.75
      lpips: 0.25
      static_loss: 0.1
      dynamics_loss: 0.1
      flow_consistency: 0.3
      temporal_coherence: 0.5
      
    # Full pipeline (epochs 30+)
    epoch_30:
      psnr: 27.0
      ssim: 0.8
      lpips: 0.2
      static_loss: 0.08
      dynamics_loss: 0.05
      flow_consistency: 0.6
      temporal_coherence: 0.7
      control_response: 0.6
      
    # Target goals (epochs 50+)
    epoch_50:
      psnr: 28.0
      ssim: 0.85
      lpips: 0.15
      static_loss: 0.05
      dynamics_loss: 0.03
      flow_consistency: 0.9
      temporal_coherence: 0.8
      motion_smoothness: 0.85
      control_response: 0.7
      lip_sync_error: 100.0
      av_correlation: 0.6

# Additional components
face_analysis:
  face_detector: "retinaface"
  face_parser: "rtnet"
  emotion_recognizer: "hsemotion"
  
audio:
  sample_rate: 16000
  feature_type: "wav2vec2"
  feature_dim: 768
  normalize_audio: true

# Visualization
vis:
  save_videos: true
  save_frames: true
  vis_freq: 10  # More frequent visualization
  num_vis_samples: 2

# Logging
logging:
  level: "DEBUG"  # More verbose
  save_frequency: 10  # More frequent
  metrics:
    save_predictions: true
    compute_fid: false  # Disable expensive metrics initially
    compute_kid: false

validation:
  frequency: 5
  num_samples: 10
  metrics: ["reconstruction", "psnr", "ssim"]

# Distributed training
distributed:
  enabled: false  # Disable for debugging
  backend: "nccl"
  find_unused_parameters: false
  gradient_as_bucket_view: true

# Optimization
optimization:
  compile_model: false
  channels_last: true
  gradient_checkpointing: false
  sync_batchnorm: false```

## File: vasa_config_overfit_simple.yaml
Overfitting configuration for single video
```yaml
# Simple overfitting config - based on working vasa_config_fixed.yaml

# Model configuration
model:
  type: "VASAModel"
  model_dim: 512
  num_layers: 8
  num_heads: 8
  max_seq_len: 50  # Back to 50 frames
  dropout: 0.0  # No dropout for overfitting
  use_positional_encoding: true
  use_audio_features: true
  use_gaze_control: true
  use_distance_control: true
  use_emotion_control: true
  
  # Diffusion
  num_diffusion_steps: 1000
  ddim_steps: 50
  beta_schedule: "cosine"
  
  # Audio
  audio_encoder: "wav2vec"
  audio_dim: 768
  
  # Control
  gaze_dim: 2
  distance_dim: 1
  emotion_dim: 2
  speed_buckets: 4

diffusion:
  num_steps: 1000
  beta_schedule: "cosine"
  ddim_steps: 50

motion:
  window_size: 5
  context_size: 2
  stride: 3
  overlap: 2
  min_frames: 30
  progressive_training: false
  use_optical_flow: false
  use_temporal_coherence: false

# Training configuration
train:
  batch_size: 1  # Single batch
  num_epochs: 100
  learning_rate: 0.005  # High LR for fast overfitting
  lr_scheduler: "constant"
  gradient_clip: 1.0
  window_size: 5
  progressive_windows: false
  max_window_size: 5
  
  # No augmentation for overfitting
  augment_audio: false
  augment_video: false
  augment_expression: false
  
  # No dropout
  dropout_probs:
    audio: 0.0
    gaze: 0.0
    distance: 0.0
    emotion: 0.0
    speed: 0.0
    
  # Disable extra losses initially
  control_start_epoch: 1000
  sync_start_epoch: 1000
  
  save_freq: 5
  validate_freq: 5
  num_workers: 0
  pin_memory: false
  persistent_workers: false

# Loss configuration
loss:
  # High weights for overfitting
  lambda_reconstruction: 10.0
  lambda_pose: 5.0
  lambda_dynamics: 5.0
  lambda_verification: 0.0
  lambda_control: 0.0
  lambda_sync: 0.0
  
  use_pose_loss: true
  use_dynamics_loss: true
  use_motion_loss: false
  use_perceptual_loss: false
  use_sync_loss: false
  use_control_loss: false
  
  theta_weight: 1.0
  scale_weight: 1.0
  rotation_weight: 1.0
  translation_weight: 1.0
  expression_weight: 1.0

# Paths
paths:
  train_list: "single_video_list.txt"
  val_list: "single_video_list.txt"
  
  volumetric_model: "models/may_emo_base.pth"
  volumetric_config: "models/may_emo_base.yaml"
  
  audio_cache: "junk/audio_cache"
  face_cache: "junk/face_cache"
  
  output_dir: "checkpoints/overfit_single"

# Dataset
dataset:
  type: "VASAIntegratedDataset"
  max_frames: 50  # Back to 50 frames
  target_fps: 25
  require_face_detection: false
  min_face_confidence: 0.0
  audio_sample_rate: 16000
  filter_short_videos: false
  min_video_duration: 0.0

# Accelerator
accelerator:
  type: "gpu"
  devices: 1
  precision: 32
  gradient_accumulation_steps: 1

# Logging
wandb:
  enabled: false  # Disable for simple test
  
# TDD - disabled for simple overfitting
tdd:
  enabled: false```

# Core Implementation Files

## File: vasa_model.py
Main VASA model implementation with loss computation
```python
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

# ... [File truncated - total 4441 lines]
```

## File: vasa_trainer.py
Training loop implementation
```python
import h5py
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import logging
from rich.logging import RichHandler
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, List
from collections import defaultdict
from omegaconf import OmegaConf
from vasa_model import VASAModel,VASALossModule,MotionSequenceHandler
import importlib
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger,TorchDebugger
import traceback
from vasa_dataset import WorkerState, VASAIntegratedDataset
from torch.utils.data import random_split
import torch.multiprocessing as mp
import random
from torch.profiler import profile, record_function, ProfilerActivity
from mem import memory_stats,clean_memory,TensorMemoryManager
import gc
from torchvision.utils import save_image
import math
from typing import *
from torch.optim import AdamW


class CFGScheduleHandler:
    """Handles CFG scale scheduling during training."""
    
    def __init__(self, config):
        self.config = config
        self.start_epoch = config.diffusion.cfg_start_epoch
        self.ramp_epochs = 10
        self.max_scales = config.train.cfg_scales
        
    def get_scales(self, current_epoch: int) -> Optional[Dict[str, float]]:
        """Get current CFG scales based on training progress."""
        if current_epoch < self.start_epoch:
            return None
            
        # Calculate ramp progress
        progress = min(1.0, (current_epoch - self.start_epoch) / self.ramp_epochs)
        
        return {
            'audio': self.max_scales.audio * progress,
            'gaze': self.max_scales.gaze * progress,
            'head_distance': self.max_scales.head_distance * progress,
            'emotion': self.max_scales.emotion * progress
        }

class GradientMonitor:
    """Monitors and logs gradient statistics during training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.reset()
        
    def reset(self):
        """Reset gradient statistics."""
        self.grad_norms = defaultdict(list)
        self.grad_means = defaultdict(list)
        self.grad_vars = defaultdict(list)
        
    def update(self):
        """Update gradient statistics."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    self.grad_norms[name].append(grad.norm().item())
                    self.grad_means[name].append(grad.mean().item())
                    self.grad_vars[name].append(grad.var().item())
                    
    def get_stats(self) -> Dict[str, float]:
        """Get current gradient statistics."""
        stats = {}
        
        # Overall statistics
        all_norms = [n for norms in self.grad_norms.values() for n in norms]
        if all_norms:
            stats.update({
                'grad_norm_min': min(all_norms),
                'grad_norm_max': max(all_norms),
                'grad_norm_mean': np.mean(all_norms),
                'grad_norm_std': np.std(all_norms)
            })
            
        # Per-layer statistics
        for name, norms in self.grad_norms.items():
            if norms:
                stats.update({
                    f'grad_norm_{name}_mean': np.mean(norms),
                    f'grad_norm_{name}_std': np.std(norms)
                })
                
        # Add mean/variance statistics
        for name, means in self.grad_means.items():
            if means:
                stats[f'grad_mean_{name}'] = np.mean(means)
        for name, vars in self.grad_vars.items():
            if vars:
                stats[f'grad_var_{name}'] = np.mean(vars)
                
        return stats

class LearningRateMonitor:
    """Monitors learning rates for all parameter groups."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        
    def get_lrs(self) -> Dict[str, float]:
        """Get current learning rates for all parameter groups."""
        lrs = {}
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            lrs[f'lr_{group_name}'] = group['lr']
        return lrs
    
class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear learning rate scheduler with warmup.
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        self.expression_warmup_steps = 100  # Shorter warmup for expressions

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get updated learning rates with specialized handling for expression parameters.
        Implements stage-based warmup and decay with expression-specific scaling.
        
        Returns:
            List of learning rates for each parameter group
        """
        try:
            if self.last_epoch < 0:
                return [group['lr'] for group in self.optimizer.param_groups]
                
            # Get parameter group info
            lrs = []
            for group in self.optimizer.param_groups:
                group_name = group.get('name', '')
                initial_lr = group['initial_lr']
                
                # Expression parameters get special treatment
                is_expression = 'expression' in group_name
                
                # Custom warmup schedule for expressions
                if is_expression:
                    warmup_steps = self.num_warmup_steps // 2  # Faster warmup
                    peak_lr = initial_lr * 2.0  # Higher peak learning rate
                else:
                    warmup_steps = self.num_warmup_steps
                    peak_lr = initial_lr

                # Handle warmup phase
                if self.last_epoch < warmup_steps:
                    # Linear warmup with expression scaling
                    warmup_progress = float(self.last_epoch) / float(max(1, warmup_steps))
                    lr = peak_lr * warmup_progress
                    
                    # Add minimum lr during warmup
                    lr = max(self.min_lr, lr)
                    
                    lrs.append(lr)
                    continue

                # Post-warmup decay phase
                decay_steps = self.num_training_steps - warmup_steps
                current_decay_step = self.last_epoch - warmup_steps
                
                # Compute decay factor
                if decay_steps <= 0:
                    decay_factor = 1.0
                else:
                    decay_progress = float(current_decay_step) / float(max(1, decay_steps))
                    
                    # Expression parameters get slower decay
                    if is_expression:
                        decay_factor = max(0.2, 1.0 - (0.8 * decay_progress))  # Slower decay
                    else:
                        decay_factor = max(0.0, 1.0 - decay_progress)  # Linear decay

                # Apply decay to peak learning rate
                lr = max(self.min_lr, peak_lr * decay_factor)
                
                # Add noise to break plateaus for expression parameters
                if is_expression and self.last_epoch % 50 == 0:  # Every 50 steps
                    noise_scale = 0.1 * lr  # 10% noise
                    lr = lr * (1.0 + random.uniform(-noise_scale, noise_scale))
                
                lrs.append(lr)

            return lrs

        except Exception as e:
            logger.error(f"Error in learning rate calculation: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to minimum learning rate
            return [self.min_lr for _ in self.optimizer.param_groups]


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-7,
    last_epoch: int = -1
) -> LinearWarmupScheduler:
    """
    Creates a scheduler with a linear warmup and decay schedule.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        num_warmup_steps: Number of warmup steps at start
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    
    Returns:
        Configured learning rate scheduler
    """
    return LinearWarmupScheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
        last_epoch=last_epoch
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 1e-7,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Creates a scheduler with a linear warmup and cosine annealing schedule.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        num_warmup_steps: Number of warmup steps at start
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    
    Returns:
        Configured learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        # Cosine decay with minimum learning rate
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        decay = max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return decay * 0.8  

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def worker_init_fn(worker_id: int):
    """Initialize worker process with proper error handling"""
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(worker_id)
        np.random.seed(worker_id)
        random.seed(worker_id)
        
        # Initialize worker state
        WorkerState.initialize_worker(worker_id)
        
        logger.info(f"Successfully initialized worker {worker_id}")
        
    except Exception as e:
        logger.error(f"Failed to initialize worker {worker_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


class TrainingState:
    """Manages training state and scheduling"""
    def __init__(self, config: dict):
        self.config = config
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize schedules
        self.schedules = {
            'cfg': self._init_cfg_schedule(),
            'window': self._init_window_schedule(),
            'dropout': self._init_dropout_schedule()
        }
        
    def _init_cfg_schedule(self):
        """Initialize CFG scale scheduling"""
        return {
            'audio': lambda e: min(3.0, 0.5 + e * 0.1),
            'gaze': lambda e: 1.0,
            'head_distance': lambda e: 0.8,
            'emotion': lambda e: 0.5
        }
        
    def _init_window_schedule(self):
        """Initialize window size scheduling"""
        base_size = self.config.motion.window_size
        return lambda e: min(50, base_size + e * 2)
        
    def _init_dropout_schedule(self):
        """Initialize condition dropout scheduling"""
        return {
            'audio': lambda e: max(0.1, 0.3 - e * 0.02),
            'gaze': lambda e: 0.1,
            'head_distance': lambda e: 0.1,
            'emotion': lambda e: 0.1
        }
        
    def get_current_scales(self) -> Dict[str, float]:
        """Get current CFG scales based on epoch"""
        return {
            k: schedule(self.epoch) 
            for k, schedule in self.schedules['cfg'].items()
        }
        
    def get_current_dropouts(self) -> Dict[str, float]:
        """Get current dropout probabilities"""
        return {
            k: schedule(self.epoch)
            for k, schedule in self.schedules['dropout'].items()
        }
        
    def get_window_size(self) -> int:
        """Get current window size"""
        return int(self.schedules['window'](self.epoch))



class MetricsTracker:
    """Tracks and logs training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset metric accumulation"""
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for k, v in metrics.items():
            self.metrics[k] += v
            self.counts[k] += 1
            
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics"""
        return {
            k: v / self.counts[k]
            for k, v in self.metrics.items()
        }

def save_video_frames(
    frames: torch.Tensor,
    output_path: Path,
    fps: int = 25
):
    """Save tensor of frames as video"""
    # Convert to numpy and correct format
    frames = frames.cpu().numpy()
    if frames.shape[0] == 3:  # CHW -> HWC
        frames = frames.transpose(1, 2, 0)
    
    # Scale to uint8 range
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
        
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frames.shape[1], frames.shape[0])
    )
    
    # Write frames
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()



class VisualizationHandler:
    """Handles generation of visualizations"""
    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.vis_dir = output_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_batch(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        batch_idx: int
    ):
        """Save visualization of generated sequences"""
        epoch_dir = self.vis_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)
        
        # Save N samples
        for i in range(min(
            outputs['pred_target_img'].shape[0],
            self.config.vis.num_vis_samples
        )):
            # Save ground truth
            save_video_frames(
                batch['frames'][i],
                epoch_dir / f'batch_{batch_idx}_sample_{i}_true.mp4'
            )
            
            # Save prediction
            save_video_frames(
                outputs['pred_target_img'][i],
                epoch_dir / f'batch_{batch_idx}_sample_{i}_pred.mp4'
            )



def collate_vasa_batch(batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """Custom collate function that preserves required metadata for windowing."""
    try:
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # Get all windows from batch items
        all_windows = []
        for item in batch:
            if 'windows' in item:
                windows = item['windows']
                # Add video path to window metadata
                for window in windows:
                    if 'metadata' not in window:
                        window['metadata'] = {}
                    window['metadata']['video_path'] = item.get('video_path', '')
                all_windows.extend(windows)
            else:
                # If item doesn't have windows, treat it as a single window
                all_windows.append(item)

        if not all_windows:
            logger.error("No valid windows in batch")
            return None

        # Get tensor keys from first window
        first_window = all_windows[0]
        tensor_keys = [k for k, v in first_window.items() if isinstance(v, torch.Tensor)]


# ... [File truncated - total 1901 lines]
```

## File: vasa_dataset.py
Dataset implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import os
import json
from pathlib import Path
import subprocess
import random
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import logging
import torchvision.transforms as transforms
import traceback
import mediapipe as mp
import matplotlib.pyplot as plt
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger  
from l2cs import L2CS, select_device, Pipeline
import h5py
from tqdm import tqdm
from typing import *
from collections import defaultdict
from torchvision.utils import save_image
from datetime import datetime
import hashlib
from vasa_model import BlinkConditionHandler
from video_tracker import VideoEventData, VideoEvent, ProblematicVideosTracker

__all__ = ['VASAIntegratedDataset', 'WorkerState','VASADatasetMixin','SpeedEncoder']


class SpeedEncoder(nn.Module):
    """Speed bucketing using the EMO paper approach"""
    def __init__(self, num_buckets=9):
        super().__init__()
        self.num_buckets = num_buckets
        
        # Centers for speed buckets from -1.0 to 1.0 
        self.centers = torch.tensor([
            -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0
        ])
        
        # Fixed radius of 0.1 for each bucket
        self.radius = 0.1 
    def encode_speed(self, head_rotation_speed: float) -> int:
        """
        Convert speed to bucket index using tanh((w - c)/r * 3)
        Args:
            head_rotation_speed: Head rotation speed in [-1, 1] range
        Returns:
            int: Bucket index (0-8)
        """
        # Calculate distance to each bucket center 
        distances = torch.abs(
            torch.tanh((head_rotation_speed - self.centers) / self.radius * 3)
        )
        
        # Return index of closest bucket
        return torch.argmin(distances).item()

    def __call__(self, speed: float) -> int:
        """
        Convenience wrapper for encode_speed
        Args:
            speed: Head rotation speed in [-1, 1] range
        Returns:
            int: Bucket index (0-8)
        """
        return self.encode_speed(speed)

class WorkerState:
    """Manages per-worker state initialization with proper multiprocessing support"""
    _instance = None
    _initialized = False
  
    @property
    def whisper_model(self):
        """Lazy initialization of Whisper model"""
        if self._whisper_model is None:
            from transformers import WhisperModel
            logger.info("Loading Whisper model...")
            self._whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
            if torch.cuda.is_available():
                self._whisper_model = self._whisper_model.cuda()
            self._whisper_model.eval()
            logger.info(f"Whisper model loaded on device: {self._whisper_model.device}")
        return self._whisper_model

    @property  
    def whisper_processor(self):
        """Lazy initialization of Whisper processor"""
        if self._whisper_processor is None:
            from transformers import WhisperProcessor
            logger.info("Loading Whisper processor...")
            self._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            logger.info("Whisper processor loaded successfully")
        return self._whisper_processor

    def __init__(self):
        """Initialize worker state"""
        super().__init__()
        # Standard properties
        self._emotion_recognizer = None
        self._face_mesh = None
        self._modnet = None

        # L2CS properties  
        self._l2cs_model = None
        self._l2cs_device = None
        self._l2cs_pipeline = None

        # Audio properties - initialize as None but create the attributes
        self._audio_model = None  # wav2vec
        self._audio_processor = None  # wav2vec processor
        self._whisper_model = None  # whisper model
        self._whisper_processor = None  # whisper processor

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance for current process"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize_worker(cls, worker_id: int):
        """Initialize worker-specific resources"""
        try:
            instance = cls.get_instance()
            if not cls._initialized:
                logger.info(f"Worker {worker_id}: Initializing resources")
                cls._initialized = True
                
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error initializing resources - {str(e)}")
            raise
    

    @property
    def l2cs_pipeline(self):
        """Lazy initialization of L2CS pipeline"""
        if self._l2cs_pipeline is None:
            # Initialize device if not already set
            # if self._l2cs_device is None:
            #     self._l2cs_device = select_device('cpu', batch_size=1)
                
            # Create pipeline
            import pathlib
            # Use the existing L2CSNet_gaze360.pkl file which has 92MB
            weights_path = pathlib.Path('models/L2CSNet_gaze360.pkl')
            
            self._l2cs_pipeline =  Pipeline(
                weights=weights_path,
                arch='ResNet50',
                device='cuda'
            )
            
        return self._l2cs_pipeline



    @property
    def emotion_recognizer(self):
        """Lazy initialization of emotion recognizer"""
        if self._emotion_recognizer is None:
            self._emotion_recognizer = HSEmotionRecognizer(
                model_name='enet_b0_8_va_mtl'
            )
        return self._emotion_recognizer
    
    @property
    def audio_model(self):
        """Lazy initialization of audio model"""
        if self._audio_model is None:
            from transformers import Wav2Vec2Model
            self._audio_model = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-base'
            ).eval()
        return self._audio_model
    
    @property
    def audio_processor(self):
        """Lazy initialization of audio processor"""
        if self._audio_processor is None:
            from transformers import Wav2Vec2Processor
            self._audio_processor = Wav2Vec2Processor.from_pretrained(
                'facebook/wav2vec2-base'
            )
        return self._audio_processor
    
    @property
    def face_mesh(self):
        """Lazy initialization of face mesh"""
        if self._face_mesh is None:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        return self._face_mesh

    @property
    def modnet(self):
        """Lazy initialization of MODNet"""
        if self._modnet is None:
            from repos.MODNet.src.models.modnet import MODNet
            self._modnet = MODNet(backbone_pretrained=False)
            
            # Load pretrained weights
            state_dict = torch.load(
                '/media/oem/12TB/nemo/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt',
                map_location='cpu'
            )
            
            # Remove module prefix from state dict keys
            new_state_dict = {}
            for k in list(state_dict.keys()):
                new_k = k[7:]  # Remove 'module.' prefix
                new_state_dict[new_k] = state_dict[k]
                
            # Load state dict and move to GPU if available
            self._modnet.load_state_dict(new_state_dict)
            if torch.cuda.is_available():
                self._modnet = self._modnet.cuda()
            self._modnet.eval()
            
        return self._modnet

class VASADatasetMixin:
    """Mixin class to handle worker initialization for VASA dataset"""
    
    def __init__(self):
        self._worker_state = None
    
    @property
    def modnet(self):  # Added MODNet property
        return self.worker_state.modnet
    
    @property
    def worker_state(self):
        """Get worker state instance for current process"""
        if self._worker_state is None:
            self._worker_state = WorkerState.get_instance()
            # Set L2CS device based on dataset device
            # self._worker_state.set_l2cs_device(self.device)
        return self._worker_state
    
    @property
    def emotion_recognizer(self):
        return self.worker_state.emotion_recognizer
    
    @property
    def audio_model(self):
        return self.worker_state.audio_model
    
    @property
    def audio_processor(self):
        return self.worker_state.audio_processor

    @property
    def whisper_model(self):
        return self.worker_state.whisper_model
    
    @property
    def whisper_processor(self):
        return self.worker_state.whisper_processor
       
    @property
    def face_mesh(self):
        return self.worker_state.face_mesh
    
    def __getstate__(self):
        """Remove non-picklable attributes before serialization"""
        state = self.__dict__.copy()
        state['_worker_state'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state without initializing worker state"""
        self.__dict__.update(state)


class WindowCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, video_path: str) -> Path:
        """Generate unique cache file path for video."""
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        return self.cache_dir / f"{video_hash}.h5"
        
    def has_cache(self, video_path: str) -> bool:
        """Check if cache exists for video."""
        cache_path = self._get_cache_path(video_path)
        return cache_path.exists()


    def load_windows(self, video_path: str) -> List[Dict[str, torch.Tensor]]:
        """Load window data from H5 file with proper tensor reshaping."""
        cache_path = self._get_cache_path(video_path)
        windows_data = []

        try:
            if not cache_path.exists():
                logger.warning(f"Cache file not found: {cache_path}")
                return windows_data

            with h5py.File(cache_path, 'r') as f:
                # Validate metadata
                if 'video_path' not in f.attrs or f.attrs['video_path'] != video_path:
                    logger.warning("Cache file metadata mismatch")
                    return windows_data

                num_windows = f.attrs['num_windows']
                
                # Load each window
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        logger.warning(f"Missing {window_key} in cache file")
                        continue
                        
                    window_group = f[window_key]
                    window_data = {}
                    
                    # Load tensors
                    for key in window_group.keys():
                        if key == 'metadata':
                            continue
                            
                        try:
                            # Get dataset
                            dataset = window_group[key]
                            
                            # Load tensor data
                            data = dataset[()]
                            
                            # Convert to tensor
                            tensor = torch.from_numpy(data)
                            
                            # Fix tensor dtype if needed
                            if 'dtype' in dataset.attrs:
                                dtype_str = dataset.attrs['dtype']
                                if isinstance(dtype_str, tuple):
                                    dtype_str = dtype_str[0].decode('utf-8')
                                tensor = tensor.to(dtype=getattr(torch, dtype_str.split('.')[-1]))
                            
                            # Remove extra dimensions if needed
                            if key in ['theta', 'rotation', 'translation', 'expression_embed', 'scale']:
                                # These should be [B, T, ...] not [B, 1, T, ...]
                                if len(tensor.shape) > 3 and tensor.shape[1] == 1:
                                    tensor = tensor.squeeze(1)
                                
                            # Special handling for audio features
                            if key == 'audio_features' and len(tensor.shape) == 3:
                                # Add channel dim if missing: [B, T, D] -> [B, 1, T, D]
                                tensor = tensor.unsqueeze(1)
                            
                            window_data[key] = tensor
                            
                        except Exception as e:
                            logger.error(f"Error loading tensor {key}: {str(e)}")
                            continue

                    # Load metadata if present
                    if 'metadata' in window_group:
                        metadata = {}
                        for k, v in window_group['metadata'].attrs.items():
                            # Convert bytes to str if needed
                            if isinstance(v, bytes):
                                v = v.decode('utf-8')
                            metadata[k] = v
                        window_data['metadata'] = metadata

                    # Log shapes for debugging
                    logger.debug(f"\nWindow {i} tensor shapes:")
                    for k, v in window_data.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"  {k}: {v.shape}")
                    
                    windows_data.append(window_data)

                if not windows_data:
                    logger.warning("No valid windows loaded from cache")
                
                return windows_data

        except Exception as e:
            logger.error(f"Error loading cache file: {str(e)}")
            if cache_path.exists():
                logger.info(f"Removing corrupted cache file: {cache_path}")
                cache_path.unlink()
            return windows_data

    def save_windows(self, video_path: str, windows_data: List[Dict[str, torch.Tensor]]):
        """Save window data with proper metadata handling."""
        cache_path = self._get_cache_path(video_path)
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            with h5py.File(temp_path, 'w') as f:
                # Save metadata
                f.attrs['video_path'] = video_path
                f.attrs['num_windows'] = len(windows_data)
                
                # Save each window
                for i, window in enumerate(windows_data):
                    window_group = f.create_group(f'window_{i}')
                    
                    # Save tensors
                    for key, tensor in window.items():
                        if key == 'metadata':
                            continue
                            
                        if isinstance(tensor, torch.Tensor):
                            # Convert to numpy and save
                            data = tensor.cpu().numpy()
                            ds = window_group.create_dataset(
                                key,
                                data=data,
                                compression='gzip'
                            )
                            # Save tensor metadata as string
                            ds.attrs['dtype'] = str(tensor.dtype)
                            ds.attrs['shape'] = tensor.shape
                    
                    # Save metadata dict if present
                    if 'metadata' in window:
                        meta_group = window_group.create_group('metadata')
                        for k, v in window['metadata'].items():
                            # Convert any non-string values to strings
                            if not isinstance(v, (str, bytes)):
                                v = str(v)
                            meta_group.attrs[k] = v

            # Only after successful save, replace old cache
            if temp_path.exists():
                if cache_path.exists():
                    cache_path.unlink()
                temp_path.rename(cache_path)
                logger.info(f"Successfully saved cache to {cache_path}")

        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

class VASAIntegratedDataset(Dataset, VASADatasetMixin):

    def __init__(
        self, 
        video_folder: str,
        emo_model,
        window_size: int = 50,
        stride: int = 25,
        context_size: int = 10,
        frame_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 50,
        hop_length: int = 10,
        cache_audio: bool = True,
        preextract_audio: bool = True,
        max_videos: Optional[int] = 100,
        random_seed: int = 42,
        device: str = 'cuda',
        cache_dir: Optional[str] = 'cache',
    ):
        VASADatasetMixin.__init__(self)
        
        # Basic initialization
        self.video_folder = Path(video_folder)
        self.emo_model = emo_model
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size
        self.cache_audio = cache_audio
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        self.device = device
        self.model_device = next(emo_model.parameters()).device
        
        self.cache = WindowCache(Path(video_folder) / "window_cache")

        self.blink_handler = BlinkConditionHandler(window_size=sequence_length)

        self.tracker = ProblematicVideosTracker(Path("bad_videos"))


# ... [File truncated - total 3293 lines]
```

## File: run_tdd_training.py
TDD training runner script
```python
#!/usr/bin/env python3
"""
Run VASA Training with TDD Testing
===================================
This script runs the VASA model training with comprehensive TDD testing.
"""

import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
import logging
from omegaconf import OmegaConf
import importlib
import wandb
from torch.utils.data import DataLoader, random_split
import traceback
import gc

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from vasa_trainer_tdd import create_tdd_trainer
from vasa_trainer import collate_vasa_batch, worker_init_fn
from logger import logger

# Configure rich logging
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)


def setup_environment():
    """Set up environment for optimal CUDA operation."""
    import os
    
    # CUDA settings for better performance
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Environment variables for better CUDA operation
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    
    # Clean memory
    gc.collect()
    torch.cuda.empty_cache()


def load_volumetric_model(config):
    """Load the volumetric avatar model."""
    logger.info("Loading volumetric avatar model...")
    
    model_path = config.paths.volumetric_model
    emo_config = OmegaConf.load(config.paths.volumetric_config)
    
    # Import and create volumetric model
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)
    
    # Load weights
    model_dict = torch.load(model_path, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    
    logger.info("✅ Volumetric avatar model loaded successfully")
    return volumetric_avatar


def create_datasets(config, volumetric_avatar):
    """Create training and validation datasets."""
    logger.info("Creating datasets...")
    
    # Create full dataset
    full_dataset = VASAIntegratedDataset(
        video_folder=config.paths.video_folder,
        emo_model=volumetric_avatar,
        max_videos=config.dataset.max_videos,
        frame_size=(512, 512),
        sequence_length=config.dataset.sequence_length,
        cache_audio=True,
        preextract_audio=True,
        random_seed=42
    )
    
    # Print dataset stats
    logger.info(f"Dataset created:")
    logger.info(f"  Total videos: {len(full_dataset.video_paths)}")
    logger.info(f"  Total windows: {len(full_dataset.windows)}")
    
    # Split into train/val
    val_size = int(config.dataset.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"  Train size: {len(train_dataset)}")
    logger.info(f"  Val size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config):
    """Create data loaders for training and validation."""
    logger.info("Creating data loaders...")
    
    # Determine batch size
    batch_size = config.train.batch_size
    
    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_vasa_batch,
        persistent_workers=False,
        prefetch_factor=None,
        multiprocessing_context=None,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None
    )
    
    # Create validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=0,  # No multiprocessing for validation
        pin_memory=True,
        collate_fn=collate_vasa_batch,
        persistent_workers=False,
        worker_init_fn=None
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """Main training function with TDD."""
    try:
        # Set up multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Load configuration
        logger.info("Loading configuration...")
        config = OmegaConf.load('vasa_config.yaml')
        
        # Override with TDD-specific settings
        config.train.num_epochs = 50  # Enough epochs to see progression
        config.train.save_freq = 5    # Save every 5 epochs
        config.dataset.max_videos = 2    # Use 2 videos for testing
        config.dataset.sequence_length = 30  # Shorter sequences for faster testing
        config.train.batch_size = 1   # Small batch for testing
        
        # Add validation split to config
        config.dataset.val_split = 0.2   # 20% validation split
        config.num_workers = 0   # No workers initially for debugging
        
        # Enable TDD testing
        if 'tdd' not in config:
            config.tdd = OmegaConf.create({})
        config.tdd.enabled = True
        config.tdd.targets = {
            'psnr': 25.0,  # Start with achievable targets
            'ssim': 0.75,
            'lpips': 0.25
        }
        
        # Set up environment
        setup_environment()
        
        # Initialize wandb if enabled
        if config.wandb.enabled:
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.get('name', 'vasa')}_tdd",
                config=OmegaConf.to_container(config, resolve=True),
                tags=['tdd', 'testing']
            )
            logger.info("✅ Weights & Biases initialized")
        
        # Load volumetric model
        volumetric_avatar = load_volumetric_model(config)
        
        # Create VASA model
        logger.info("Creating VASA model...")
        model = VASAModel(
            config=config,
            volumetric_avatar=volumetric_avatar,
            device=config.device
        )
        model = model.cuda()
        logger.info("✅ VASA model created")
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(config, volumetric_avatar)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, config
        )
        
        # Create output directory
        output_dir = Path(config.paths.checkpoint_dir) / "tdd_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Create TDD-enhanced trainer
        logger.info("\n" + "="*60)
        logger.info("Creating TDD-Enhanced Trainer")
        logger.info("="*60)
        
        trainer = create_tdd_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir
        )
        
        logger.info("✅ TDD trainer created successfully")
        
        # Load checkpoint if resuming
        if hasattr(config.train, 'resume_from') and config.train.resume_from:
            checkpoint_path = config.train.resume_from
            if Path(checkpoint_path).exists():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                trainer.load_checkpoint(checkpoint_path)
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
        
        # Log training configuration
        logger.info("\n" + "="*60)
        logger.info("Training Configuration")
        logger.info("="*60)
        logger.info(f"Epochs: {config.train.num_epochs}")
        logger.info(f"Batch size: {config.train.batch_size}")
        logger.info(f"Learning rate: {config.train.lr}")
        logger.info(f"Turn off noise: {config.train.get('turn_off_noise', False)}")
        logger.info(f"Control start epoch: {config.train.control_start_epoch}")
        logger.info(f"TDD enabled: {config.tdd.enabled}")
        logger.info("="*60 + "\n")
        
        # Start training with TDD
        logger.info("🚀 Starting TDD-enhanced training...")
        logger.info("Tests will run after each epoch to ensure quality")
        logger.info("Watch for test results marked with ✅ (pass) or ❌ (fail)\n")
        
        # Run training
        trainer.train()
        
        # Training completed
        logger.info("\n" + "="*60)
        logger.info("Training Completed!")
        logger.info("="*60)
        
        # Generate final test report
        if trainer.test_runner:
            test_output_dir = output_dir / 'test_results'
            logger.info(f"\n📊 Test reports saved to: {test_output_dir}")
            logger.info("Check test_progress.png for visual progress")
            
            # Print final test summary
            if trainer.test_runner.results_history:
                last_results = trainer.test_runner.results_history[-1]
                passed = last_results['passed_count']
                total = last_results['total_count']
                pass_rate = 100 * passed / total if total > 0 else 0
                
                logger.info(f"\nFinal Test Results:")
                logger.info(f"  Tests Passed: {passed}/{total} ({pass_rate:.1f}%)")
        
        # Clean up
        if config.wandb.enabled:
            wandb.finish()
        
        logger.info("\n✅ Training with TDD completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'wandb' in locals() and wandb.run is not None:
            wandb.finish(exit_code=1)
        
        raise


if __name__ == "__main__":
    main()```

## File: test_loss_computation.py
Loss computation test (this works correctly)
```python
#!/usr/bin/env python3
"""
Test Loss Computation Directly
==============================
Find out why the loss computation is failing.
"""

import torch
import sys
import traceback
from omegaconf import OmegaConf
import importlib

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel, VASALossModule
from logger import logger
import logging

# Set debug logging
logging.basicConfig(level=logging.DEBUG)


def test_loss_computation():
    """Test the loss computation directly."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    try:
        # Load config
        config = OmegaConf.load('vasa_config_fixed.yaml')
        
        # Load volumetric model
        print("\n1. Loading volumetric avatar...")
        model_path = config.paths.volumetric_model
        emo_config = OmegaConf.load(config.paths.volumetric_config)
        volumetric_avatar = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)
        
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        
        # Create loss module
        print("\n2. Creating loss module...")
        loss_module = VASALossModule(
            volumetric_avatar=volumetric_avatar,
            config=config,
            device='cuda'
        )
        
        # Create dummy data
        print("\n3. Creating test data...")
        B, T = 1, 5
        
        outputs = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda(),
            'noise': {}
        }
        
        targets = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda() * 1.1,  # Slightly different
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        conditions = {
            'audio_features': torch.randn(B, T, 768).cuda(),
            'gaze': torch.randn(B, T, 2).cuda(),
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda()
        }
        
        # Add noise
        outputs['noise'] = {
            k: torch.randn_like(v) * 0.01 for k, v in targets.items()
        }
        
        # Test loss computation
        print("\n4. Computing losses...")
        losses, metrics = loss_module.compute_losses(
            outputs=outputs,
            targets=targets,
            conditions=conditions,
            noise=outputs['noise'],
            return_metrics=True,
            current_epoch=0,
            step=0
        )
        
        print("\n5. Loss results:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.item():.6f} (requires_grad: {v.requires_grad})")
        
        print("\n6. Checking if losses are reasonable...")
        if losses['total'].item() == 1.0:
            print("   ❌ Total loss is exactly 1.0 - likely using default error value!")
        elif losses['total'].item() > 0 and losses['total'].item() < 100:
            print(f"   ✅ Total loss is reasonable: {losses['total'].item():.6f}")
        else:
            print(f"   ⚠️ Total loss seems unusual: {losses['total'].item():.6f}")
        
        # Test backward pass
        print("\n7. Testing backward pass...")
        try:
            losses['total'].backward()
            print("   ✅ Backward pass successful!")
            
            # Check if any gradients were computed
            has_grads = False
            for name, param in loss_module.__dict__.items():
                if isinstance(param, torch.nn.Module):
                    for p in param.parameters():
                        if p.grad is not None:
                            has_grads = True
                            break
            
            if has_grads:
                print("   ✅ Gradients computed!")
            else:
                print("   ⚠️ No gradients found in loss module")
                
        except Exception as e:
            print(f"   ❌ Backward pass failed: {str(e)}")
        
        return losses['total'].item() != 1.0
        
    except Exception as e:
        print(f"\n❌ Error in loss computation: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        return False


def test_reconstruction_loss_directly():
    """Test just the reconstruction loss computation."""
    print("\n" + "="*60)
    print("Testing Reconstruction Loss Directly")
    print("="*60)
    
    try:
        # Simple MSE test
        print("\n1. Testing simple MSE loss...")
        pred = torch.randn(2, 5, 3, 4).cuda()
        target = torch.randn(2, 5, 3, 4).cuda()
        
        loss = torch.nn.functional.mse_loss(pred, target)
        print(f"   MSE loss: {loss.item():.6f}")
        
        if loss.item() > 0:
            print("   ✅ MSE loss computed successfully!")
            return True
        else:
            print("   ❌ MSE loss is zero or negative!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    """Run all loss tests."""
    results = {
        'Simple MSE': test_reconstruction_loss_directly(),
        'Full Loss Module': test_loss_computation()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✅ Loss computation is working!")
    else:
        print("\n❌ Loss computation has issues that need fixing.")


if __name__ == "__main__":
    main()```

# Test Results

## Loss Computation Test Output
When running test_loss_computation.py, the output shows:
```
5. Loss results:
   reconstruction: 3.059626 (requires_grad: True)
   pose_loss: 2.086096 (requires_grad: True)
   dynamics_loss: 0.973530 (requires_grad: True)
   ...
   total: 3.059626 (requires_grad: True)

6. Checking if losses are reasonable...
   ✅ Total loss is reasonable: 3.059626
```

## Training Output
When running training, the output shows:
```
Epoch 0:   0%|          | 0/2 [00:02<?, ?it/s]
WARNING     Reconstruction loss: 1.0000 - Above  vasa_tdd_tests.py:74
```

# Recent Error Logs

## Video Processing Errors
```
# Failed Videos Report


junk/10_50frames.mp4:
  2025-09-04 14:00:13 - video_too_short: {'total_frames': 50, 'min_frames': 85}

junk/10.mp4:
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:14 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:15 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:15 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:15 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:15 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
  2025-09-04 14:00:15 - processing_error: {'error': 'Face attribute error: Expected sequence length 50, got 30'}
```

# Key Code Snippets to Review

## Loss Module compute_losses method (from vasa_model.py)
Look for where it might return 1.0 as default:
```python
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
```

## Where loss gets computed in trainer (from vasa_trainer.py)
```python
                            
                            # Tell loss module which window we're on for visualization
                            self.loss_module._window_count_this_batch = window_idx
                            
                            # Compute losses using noise prediction
                            losses, metrics = self.loss_module.compute_losses(
                                outputs=outputs,
                                targets=motion_data,  # Original motion data
                                conditions=control_signals,
                                noise=outputs['noise'],    
                                return_metrics=True,
                                current_epoch=self.current_epoch,
                                step=self.global_step
                            )

                            # Update batch metrics - store per-window
                            for k, v in losses.items():
                                if isinstance(v, torch.Tensor):
                                    batch_metrics[k].append(v.detach())
                            if metrics:
                                for k, v in metrics.items():
```

# Summary of Investigation Needed

## Key Questions to Answer
1. Why does the loss computation return 1.0 during training but ~3.06 in the test script?
2. Is the config being loaded correctly during training?
3. Are the loss weights (lambda_pose, lambda_dynamics) actually being applied?
4. Is there an exception being caught that returns 1.0 as a fallback?

## Suspected Issues
1. Config not being passed correctly to loss module during training
2. Exception handling returning default 1.0 value
3. Loss weights not being applied despite config changes
4. Data format mismatch between training and test scenarios

## Next Steps
1. Add logging to see actual lambda values being used
2. Remove any try/except blocks that return 1.0
3. Verify config is loaded correctly in training
4. Check if loss module is initialized with correct config


---
