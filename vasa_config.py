from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from omegaconf import OmegaConf, MISSING
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    num_workers: int = 4
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 8
    motion_dim: int = 256
    audio_dim: int = 128
    dropout: float = 0.1

@dataclass
class DiffusionConfig:
    """Diffusion process configuration"""
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02

@dataclass
class GenerationConfig:
    """Generation settings configuration"""
    sequence_length: int = 25
    overlap: int = 5
    smoothing_window: int = 5
    generation_interval: int = 100
    apply_post_processing: bool = True

@dataclass
class CFGConfig:
    """Classifier-free guidance configuration"""
    audio_scale: float = 0.5
    gaze_scale: float = 1.0
    condition_dropout: float = 0.1

@dataclass
class LossConfig:
    """Loss weights configuration"""
    lambda_recon: float = 1.0
    lambda_identity: float = 0.1
    lambda_dpe: float = 0.1
    lambda_pose: float = 1.0
    lambda_expr: float = 1.0
    lambda_cfg: float = 0.1

@dataclass
class DataConfig:
    """Data configuration"""
    frame_size: Tuple[int, int] = (512, 512)
    train_data: str = MISSING
    val_data: str = MISSING
    max_videos: Optional[int] = None
    cache_audio: bool = True
    preextract_audio: bool = True

@dataclass
class StageConfig:
    """Stage-specific configuration"""
    # Stage 1: Latent Space Learning
    latent_space_epochs: int = 100
    latent_space_lr: float = 1e-4
    # Stage 2: Dynamics Generation
    dynamics_epochs: int = 200
    dynamics_lr: float = 1e-4

@dataclass
class CFGConfig:
    """Classifier-free guidance configuration"""
    audio_scale: float = 0.5
    gaze_scale: float = 1.0
    condition_dropout: float = 0.1

@dataclass
class VASAConfig:
    experiment_name: str = MISSING
    output_dir: str = "outputs"

    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cfg: CFGConfig = field(default_factory=CFGConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage: StageConfig = field(default_factory=StageConfig)

    use_wandb: bool = True
    early_stopping: bool = True
    patience: int = 10

    @classmethod
    def load(cls, config_path: Path) -> 'VASAConfig':
        """Load configuration from YAML file"""
        # Load base config schema
        schema = OmegaConf.structured(cls)
        
        # Load user config
        user_config = OmegaConf.load(config_path)
        
        # Merge configs with validation
        try:
            config = OmegaConf.merge(schema, user_config)
            # Validate after merge
            OmegaConf.validate(config)
            return config
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

    def save(self, save_path: Path):
        """Save configuration to YAML file"""
        OmegaConf.save(self, save_path)

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    def validate(self):
        """Custom validation rules"""
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.num_epochs > 0, "Number of epochs must be positive"
        assert self.model.d_model % self.model.nhead == 0, "d_model must be divisible by nhead"
        assert self.generation.overlap < self.generation.sequence_length, "Overlap must be less than sequence length"
