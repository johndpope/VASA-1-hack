#!/usr/bin/env python3
"""
Simple test script for overfitted VASA model checkpoint.
"""

import torch
from pathlib import Path
from omegaconf import OmegaConf
import sys
import logging
import numpy as np
import cv2
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the best checkpoint."""
    
    # Load configuration
    config_path = 'overfit_config.yaml'
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Import modules
    sys.path.append('nemo')
    sys.path.append('.')
    
    from vasa_model import VASAModel
    from vasa_dataset import VASAIntegratedDataset
    from vasa_trainer import VASATrainer
    
    # Initialize model
    logger.info("Initializing VASA model...")
    model = VASAModel(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints_overfit/best_checkpoint.pt"
    if not Path(checkpoint_path).exists():
        logger.warning(f"Best checkpoint not found, trying latest...")
        checkpoints = sorted(Path("checkpoints_overfit").glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])
        else:
            logger.error("No checkpoints found!")
            return
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(config.device)
    model.eval()
    
    # Get training loss from checkpoint
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    logger.info(f"Checkpoint from epoch {epoch}, best loss: {best_loss:.4f}")
    
    # Create a simple test batch
    logger.info("Creating test batch...")
    batch_size = 1
    seq_len = 20
    device = config.device
    
    # Create dummy inputs
    test_batch = {
        'theta': torch.randn(batch_size, seq_len, 3, 4).to(device),
        'rotation': torch.randn(batch_size, seq_len, 3).to(device),
        'translation': torch.randn(batch_size, seq_len, 3).to(device),
        'expression_embed': torch.randn(batch_size, seq_len, 128).to(device),
        'audio_features': torch.randn(batch_size, seq_len, 768).to(device),
        'gaze': torch.randn(batch_size, seq_len, 2).to(device),
        'head_distance': torch.randn(batch_size, seq_len, 1).to(device),
        'emotion': torch.randn(batch_size, seq_len, 2).to(device),
    }
    
    # Add prev_context
    context_size = 10
    prev_context = {
        'theta': torch.randn(batch_size, context_size, 3, 4).to(device),
        'rotation': torch.randn(batch_size, context_size, 3).to(device),
        'translation': torch.randn(batch_size, context_size, 3).to(device),
        'expression_embed': torch.randn(batch_size, context_size, 128).to(device),
    }
    
    # Test forward pass
    logger.info("Testing forward pass...")
    with torch.no_grad():
        try:
            # Prepare motion data
            motion_data = {
                'theta': test_batch['theta'],
                'rotation': test_batch['rotation'],
                'translation': test_batch['translation'],
                'expression_embed': test_batch['expression_embed']
            }
            
            # Prepare conditions
            conditions = {
                'audio_features': test_batch['audio_features'],
                'gaze': test_batch['gaze'],
                'head_distance': test_batch['head_distance'],
                'emotion': test_batch['emotion']
            }
            
            # Add noise for diffusion
            noise_level = torch.tensor([0.1], device=device)
            
            # Forward pass
            output = model(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=conditions,
                prev_context=prev_context
            )
            
            logger.info(f"✓ Forward pass successful!")
            logger.info(f"  Output shape: {output['theta'].shape}")
            logger.info(f"  Output theta range: [{output['theta'].min():.3f}, {output['theta'].max():.3f}]")
            
        except Exception as e:
            logger.error(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Test complete!")
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()