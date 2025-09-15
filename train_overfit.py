#!/usr/bin/env python3
"""
Overfitting training script for VASA with improved VA bridge.
Uses a small dataset to quickly test if the model can learn.
"""

import torch
from pathlib import Path
from omegaconf import OmegaConf
import sys
import os
import wandb

# Add nemo to path before importing logger
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger

# The nemo logger is already configured with RichHandler and file logging

def main():
    """Main training function for overfitting."""
    
    # Load configuration
    config_path = 'overfit_config.yaml'
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return
        
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Initialize wandb for tracking
    wandb.init(
        project="vasa-overfitting",
        name="overfit-test",
        config=OmegaConf.to_container(config, resolve=True)
    )
    
    # Import after config to ensure paths are set
    sys.path.append('nemo')
    sys.path.append('.')
    
    from vasa_model import VASAModel
    from vasa_trainer import VASATrainer
    from vasa_dataset import VASAIntegratedDataset
    from vasa_sampler import WindowSequenceSampler, create_window_sequence_collate_fn
    from torch.utils.data import DataLoader
    import importlib
    
    # Initialize volumetric avatar
    logger.info("Loading volumetric avatar...")
    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)
    
    # Load EMO weights
    try:
        model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()  # Keep in eval mode
        logger.info("Volumetric avatar loaded successfully")
    except Exception as e:
        logger.error(f"Error loading EMO model: {str(e)}")
        raise
    
    # Initialize VASA model
    logger.info("Initializing VASA model...")
    model = VASAModel(config, volumetric_avatar)
    
    # Load checkpoint if resuming (check env var first, then config)
    import os
    env_resume = os.environ.get('VASA_RESUME_FROM')
    resume_path = env_resume if env_resume else config.train.get('resume_from')
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        checkpoint_path = Path(resume_path)
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Get epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Create data loader
    logger.info("Creating data loader...")
    use_single_bucket = config.dataset.get('use_single_bucket', True)  # Default to single-bucket
    train_dataset = VASAIntegratedDataset(
        video_folder=config.paths.video_folder,
        emo_model=volumetric_avatar,
        window_size=config.motion.window_size,
        stride=config.motion.stride,
        context_size=config.motion.context_size,
        frame_size=config.dataset.frame_size,
        sequence_length=config.dataset.sequence_length,
        hop_length=config.dataset.hop_length,
        cache_audio=config.dataset.cache_audio,
        preextract_audio=config.dataset.preextract_audio,
        max_videos=config.dataset.max_videos,
        cache_dir=config.paths.cache_dir,
        device=config.device,
        use_single_bucket=use_single_bucket
    )

    # Check if single-bucket cache exists, preprocess if needed
    if use_single_bucket and hasattr(train_dataset.cache, 'has_cache'):
        if not train_dataset.cache.has_cache():
            logger.info("Single-bucket cache not found. Consider running preprocess_single_bucket.py first.")
        else:
            cache_info = train_dataset.cache.get_cache_info()
            logger.info(f"Using single-bucket cache: {cache_info['num_windows']} windows, {cache_info['file_size_mb']:.1f} MB")
    
    # Create custom sampler for maintaining window sequences
    # Use windows_per_batch from config if available, otherwise default to 4
    windows_per_sequence = config.train.get('windows_per_batch', 4)
    train_sampler = WindowSequenceSampler(
        train_dataset,
        batch_size=config.train.batch_size,
        windows_per_sequence=windows_per_sequence,  # Number of consecutive windows from config
        shuffle=True
    )
    logger.info(f"Using {windows_per_sequence} consecutive windows per sequence")
    
    # Create custom collate function for adding prev_context
    collate_fn = create_window_sequence_collate_fn(
        context_size=config.motion.context_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues
        pin_memory=False  # Disabled because tensors are already on GPU
    )
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = VASATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=None,  # No validation for overfitting
        output_dir="checkpoints_overfit",
        config_path=config_path
    )
    
    # Set start epoch if resuming
    if start_epoch > 0:
        trainer.current_epoch = start_epoch
    
    # Start training
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"Configuration:")
    logger.info(f"  - Batch size: {config.train.batch_size}")
    logger.info(f"  - Learning rate: {config.train.lr}")
    logger.info(f"  - Num epochs: {config.train.num_epochs}")
    logger.info(f"  - Window size: {config.motion.window_size}")
    logger.info(f"  - Stride: {config.motion.stride}")
    logger.info(f"  - Use identity image: {config.dataset.use_identity_image}")
    
    if config.dataset.use_identity_image:
        logger.info(f"  - Identity image: {config.dataset.identity_image_path}")
    
    logger.info(f"  - Normalization enabled: {os.path.exists('va_motion_statistics.pkl')}")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()