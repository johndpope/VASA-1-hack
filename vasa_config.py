from dataclasses import dataclass
import yaml
from typing import List, Optional, Tuple


@dataclass
class VASAConfig:
    """Configuration class for VASA training"""
    # Training settings
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    num_workers: int = 4
    
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    motion_dim: int = 256
    audio_dim: int = 128
    
    # Diffusion settings
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Generation settings
    sequence_length: int = 25
    overlap: int = 5
    smoothing_window: int = 5
    generation_interval: int = 100
    
    # CFG settings
    audio_scale: float = 0.5
    gaze_scale: float = 1.0
    
    # Loss weights
    lambda_recon: float = 1.0
    lambda_identity: float = 0.1
    lambda_dpe: float = 0.1
    lambda_cfg: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Data settings
    frame_size: Tuple[int, int] = (512, 512)
    apply_post_processing: bool = True
    


    # Loss weights
    lambda_recon: float = 1.0
    lambda_identity: float = 0.1
    lambda_dpe: float = 0.1
    lambda_pose: float = 1.0
    lambda_expr: float = 1.0
    lambda_cfg: float = 0.1
    
    @classmethod
    def from_yaml(cls, path: str) -> 'VASAConfig':
        """Load config from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)