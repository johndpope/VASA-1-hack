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
    main()