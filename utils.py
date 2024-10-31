from datetime import datetime
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from typing import Dict, Any
import argparse

def get_vasa_exp_name(config: OmegaConf, args: Optional[argparse.Namespace] = None, mode: str = 'train') -> str:
    """
    Generate experiment name for VASA training/testing
    Args:
        config: VASA configuration
        args: Command line arguments
        mode: 'train' or 'test'
    """
    if mode == 'test':
        if args and args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            epoch = checkpoint_path.stem.split('_')[1]  # checkpoint_epoch_X.pt
            exp_name = checkpoint_path.parent.parent.name
            return f"Test_epoch{epoch}_{exp_name}"
        return f"Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start with timestamp
    name_parts = [datetime.now().strftime("%Y%m%d_%H%M")]
    
    # Add dataset info
    if config.data.train_data:
        name_parts.append(Path(config.data.train_data).stem)
    
    # Add model architecture details
    arch_parts = [
        f"d{config.model.d_model}",
        f"h{config.model.nhead}",
        f"l{config.model.num_layers}"
    ]
    name_parts.append('_'.join(arch_parts))
    
    # Add sequence and generation details
    gen_parts = [
        f"seq{config.generation.sequence_length}",
        f"ovl{config.generation.overlap}"
    ]
    name_parts.append('_'.join(gen_parts))
    
    # Add diffusion settings
    diff_parts = [
        f"steps{config.diffusion.num_steps}",
        f"b{config.diffusion.beta_start:.0e}"
    ]
    name_parts.append('_'.join(diff_parts))
    
    # Add training details
    train_parts = [
        f"bs{config.training.batch_size}",
        f"lr{config.training.learning_rate:.0e}",
        f"acc{config.training.gradient_accumulation_steps}"
    ]
    name_parts.append('_'.join(train_parts))
    
    # Add loss weights
    loss_parts = [
        f"rec{config.loss.lambda_recon:.1f}",
        f"id{config.loss.lambda_identity:.1f}",
        f"dpe{config.loss.lambda_dpe:.1f}"
    ]
    name_parts.append('_'.join(loss_parts))
    
    # Add CFG scales
    cfg_parts = [
        f"cfg_a{config.cfg.audio_scale:.1f}",
        f"cfg_g{config.cfg.gaze_scale:.1f}"
    ]
    name_parts.append('_'.join(cfg_parts))
    
    # Add stage-specific settings if in staged training
    stage_parts = [
        f"s1e{config.stage.latent_space_epochs}",
        f"s2e{config.stage.dynamics_epochs}"
    ]
    name_parts.append('_'.join(stage_parts))
    
    # Add command line overrides if any
    if args:
        override_parts = []
        if args.batch_size:
            override_parts.append(f"bs{args.batch_size}")
        if args.num_steps:
            override_parts.append(f"steps{args.num_steps}")
        if args.cfg_scale:
            override_parts.append(f"cfg{args.cfg_scale}")
        if override_parts:
            name_parts.append('_'.join(override_parts))
    
    # Combine all parts
    name = '__'.join(name_parts)
    
    # Handle max length limitations (if any)
    max_length = 255  # Common filesystem limit
    if len(name) > max_length:
        # Truncate while keeping essential info
        timestamp = name_parts[0]
        essential = '__'.join(name_parts[1:4])  # Keep dataset and key architecture info
        hash_suffix = hex(hash(name))[2:10]  # Use hash of full name as suffix
        name = f"{timestamp}__{essential}__{hash_suffix}"
    
    return name

def setup_experiment(args) -> Dict[str, Any]:
    """Setup experiment with proper naming"""
    # Load base config
    config = OmegaConf.load(args.config)
    
    # Generate experiment name
    exp_name = get_vasa_exp_name(config, args)
    
    # Update config with experiment name
    config.experiment_name = exp_name
    
    # Setup directories
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    samples_dir = exp_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    # Save full config
    config_path = exp_dir / 'config.yaml'
    OmegaConf.save(config, config_path)
    
    return {
        'exp_dir': exp_dir,
        'checkpoints_dir': checkpoints_dir,
        'samples_dir': samples_dir,
        'config': config
    }