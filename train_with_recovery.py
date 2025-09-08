#!/usr/bin/env python3
"""Training script with automatic OOM recovery and checkpoint resumption."""

import os
import sys
import time
import torch
import gc
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    return str(checkpoints[-1])

def run_training(config_path, resume_from=None, max_retries=5):
    """Run training with automatic recovery from OOM errors."""
    
    from vasa_trainer import VASATrainer
    from vasa_model import VASAModel
    from vasa_dataset import VASAIntegratedDataset
    from torch.utils.data import DataLoader
    
    retry_count = 0
    last_checkpoint = resume_from
    
    while retry_count < max_retries:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training attempt {retry_count + 1}/{max_retries}")
            if last_checkpoint:
                logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            logger.info(f"{'='*60}\n")
            
            # Clean up memory before starting
            cleanup_memory()
            
            # Load config
            config = OmegaConf.load(config_path)
            
            # Update config to resume from checkpoint if available
            if last_checkpoint:
                config.train.resume_from = last_checkpoint
                logger.info(f"Updated config to resume from: {last_checkpoint}")
            
            # Reduce batch size on retry to avoid OOM
            if retry_count > 0:
                original_batch_size = config.train.batch_size
                config.train.batch_size = max(1, original_batch_size // (2 ** retry_count))
                logger.warning(f"Reduced batch size from {original_batch_size} to {config.train.batch_size}")
                
                # Also reduce gradient accumulation if needed
                if config.train.batch_size == 1 and config.train.gradient_accumulation_steps > 1:
                    config.train.gradient_accumulation_steps = max(1, config.train.gradient_accumulation_steps // 2)
                    logger.warning(f"Reduced gradient accumulation to {config.train.gradient_accumulation_steps}")
            
            # Initialize model
            logger.info("Initializing model...")
            model = VASAModel(config)
            
            # Create data loader
            logger.info("Creating data loader...")
            dataset = VASAIntegratedDataset(config)
            train_loader = DataLoader(
                dataset,
                batch_size=config.train.batch_size,
                shuffle=True,
                num_workers=config.get('num_workers', 4),
                pin_memory=True
            )
            
            # Initialize trainer
            logger.info("Initializing trainer...")
            trainer = VASATrainer(model, config, train_loader)
            
            # Run training
            logger.info("Starting training...")
            trainer.train()
            
            # If we get here, training completed successfully
            logger.info("Training completed successfully!")
            break
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"\n{'='*60}")
            logger.error(f"CUDA OUT OF MEMORY ERROR!")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"{'='*60}\n")
            
            # Clean up
            cleanup_memory()
            
            # Try to find the latest checkpoint
            checkpoint_dir = config.paths.checkpoint_dir
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")
                last_checkpoint = latest_checkpoint
            else:
                logger.warning("No checkpoint found to resume from")
            
            retry_count += 1
            
            if retry_count < max_retries:
                wait_time = min(60 * retry_count, 300)  # Wait up to 5 minutes
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Training failed.")
                raise
                
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user.")
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"\n{'='*60}")
            logger.error(f"Unexpected error: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"{'='*60}\n")
            
            # For other errors, try to resume but don't reduce batch size
            checkpoint_dir = config.paths.checkpoint_dir
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")
                last_checkpoint = latest_checkpoint
                retry_count += 1
                
                if retry_count < max_retries:
                    wait_time = 30
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Training failed.")
                    raise
            else:
                logger.error("No checkpoint found and error is not recoverable.")
                raise

def main():
    parser = argparse.ArgumentParser(description="VASA Training with OOM Recovery")
    parser.add_argument("--config", type=str, default="overfit_config.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum number of retry attempts")
    
    args = parser.parse_args()
    
    # Set environment variables for better CUDA memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better performance
    
    # Run training with recovery
    run_training(args.config, args.resume, args.max_retries)

if __name__ == "__main__":
    main()