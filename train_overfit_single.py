#!/usr/bin/env python3
"""
Overfit Training on Single Video
=================================
This script trains the VASA model to overfit on a single video (junk/10.mp4)
to verify the model can learn and memorize data properly.
"""

import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
import sys
from accelerate import Accelerator
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

# Import VASA modules
from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from vasa_trainer import VASATrainer, collate_vasa_batch
from logger import logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SingleVideoDataset(VASAIntegratedDataset):
    """Dataset that only loads a single video for overfitting."""
    
    def __init__(self, video_path, *args, **kwargs):
        # Override video list to only use one video
        kwargs['video_files'] = [video_path]
        super().__init__(*args, **kwargs)
        self.single_video_path = video_path
        logger.info(f"Created single video dataset with: {video_path}")
        
    def __len__(self):
        # Return a large number to repeat the single video
        return 1000  # Train on same video 1000 times per epoch
        
    def __getitem__(self, idx):
        # Always return the same video (index 0)
        try:
            actual_idx = 0  # Always use first (and only) video
            return super().__getitem__(actual_idx)
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            # Return None to skip
            return None

class OverfitTrainer(VASATrainer):
    """Specialized trainer for overfitting experiments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_loss = float('inf')
        self.overfit_metrics = []
        
    def train_epoch(self):
        """Training epoch with overfitting metrics."""
        self.model.train()
        epoch_losses = []
        
        # Use only first batch repeatedly for extreme overfitting
        first_batch = None
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", 
                 disable=not self.accelerator.is_local_main_process) as pbar:
            
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue
                    
                # Store first valid batch and reuse it
                if first_batch is None:
                    first_batch = batch
                    logger.info("Captured first batch for overfitting")
                else:
                    # Use the stored first batch
                    batch = first_batch
                
                try:
                    # Forward pass
                    loss, outputs = self.training_step(batch)
                    
                    if loss is not None:
                        # Backward pass
                        self.accelerator.backward(loss)
                        
                        # Gradient clipping
                        if self.config.train.gradient_clip > 0:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.gradient_clip
                            )
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Track loss
                        loss_value = loss.item()
                        epoch_losses.append(loss_value)
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f'{loss_value:.4f}',
                            'best': f'{self.best_train_loss:.4f}'
                        })
                        
                        # Log every step for overfitting monitoring
                        if batch_idx % 10 == 0:
                            logger.info(f"Step {batch_idx}: Loss = {loss_value:.4f}")
                            
                            # Check if we're overfitting successfully
                            if loss_value < self.best_train_loss:
                                self.best_train_loss = loss_value
                                logger.info(f"🎯 New best loss: {loss_value:.4f}")
                                
                                # Save checkpoint when improving
                                if loss_value < 0.1:  # Very low loss threshold
                                    self.save_checkpoint(is_best=True)
                                    logger.info("💾 Saved overfitted checkpoint!")
                        
                        # Early stopping if perfectly overfitted
                        if loss_value < 0.001:
                            logger.info("🎉 Perfect overfitting achieved! Loss < 0.001")
                            return {'loss': loss_value, 'status': 'perfect_overfit'}
                            
                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    continue
                
                # Only train on 100 iterations per epoch for faster overfitting
                if batch_idx >= 100:
                    break
        
        # Return epoch statistics
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        
        return {
            'loss': avg_loss,
            'best_loss': self.best_train_loss,
            'num_batches': len(epoch_losses)
        }
    
    def validate(self):
        """Validation on the same video for overfitting check."""
        # For overfitting, we validate on training data
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                    
                try:
                    loss, _ = self.training_step(batch)
                    if loss is not None:
                        val_losses.append(loss.item())
                except Exception as e:
                    logger.error(f"Validation error: {str(e)}")
                    continue
                
                # Only validate on first batch
                break
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        return {'val_loss': avg_val_loss}

def main():
    """Main overfitting training function."""
    
    # Load configuration
    config = OmegaConf.load('vasa_config_overfit.yaml')
    
    logger.info("="*60)
    logger.info("Starting Overfitting Training on junk/10.mp4")
    logger.info("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
        mixed_precision='no',  # No mixed precision for debugging
        log_with=["wandb"] if config.wandb.enabled else None,
    )
    
    # Create output directory
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing VASA model...")
    model = VASAModel(config.model)
    
    # Create single video dataset
    logger.info("Creating single video dataset...")
    train_dataset = SingleVideoDataset(
        video_path="junk/10.mp4",
        config=config,
        mode='train',
        volumetric_config_path=config.paths.volumetric_config,
        volumetric_model_path=config.paths.volumetric_model,
        cache_dir=Path(config.paths.face_cache),
        audio_cache_dir=Path(config.paths.audio_cache)
    )
    
    # Use same dataset for validation (overfitting check)
    val_dataset = train_dataset
    
    # Create data loaders with minimal batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Single sample
        shuffle=False,  # No shuffle for consistent overfitting
        num_workers=0,
        collate_fn=collate_vasa_batch,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_vasa_batch
    )
    
    # Create trainer
    logger.info("Creating overfitting trainer...")
    trainer = OverfitTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        accelerator=accelerator
    )
    
    # Training loop
    logger.info("Starting training loop...")
    logger.info("Goal: Reduce loss to < 0.001 on single video")
    
    for epoch in range(config.train.num_epochs):
        trainer.current_epoch = epoch
        
        # Train
        train_stats = trainer.train_epoch()
        logger.info(f"Epoch {epoch}: Train Loss = {train_stats['loss']:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_stats = trainer.validate()
            logger.info(f"Epoch {epoch}: Val Loss = {val_stats['val_loss']:.4f}")
        
        # Check for perfect overfitting
        if train_stats.get('status') == 'perfect_overfit':
            logger.info("🎊 Training complete - perfect overfitting achieved!")
            break
            
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            trainer.save_checkpoint(is_best=False)
            
        # Log to wandb
        if config.wandb.enabled and accelerator.is_local_main_process:
            accelerator.log({
                'train/loss': train_stats['loss'],
                'train/best_loss': train_stats['best_loss'],
                'epoch': epoch
            })
    
    logger.info("="*60)
    logger.info("Overfitting Training Complete!")
    logger.info(f"Final best loss: {trainer.best_train_loss:.6f}")
    logger.info("="*60)

if __name__ == "__main__":
    main()