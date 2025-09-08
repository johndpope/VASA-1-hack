#!/usr/bin/env python3
"""Test script for prev_context implementation in VASA model."""

import torch
import numpy as np
from pathlib import Path
import logging
from vasa_model import VASAModel
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prev_context():
    """Test the prev_context mechanism."""
    
    # Load config
    config_path = 'overfit_config.yaml'
    config = OmegaConf.load(config_path)
    
    # Initialize model
    logger.info("Initializing VASA model...")
    
    # Create a dummy volumetric avatar (not needed for testing prev_context)
    class DummyVolumetricAvatar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
        def eval(self):
            return self
        
        def parameters(self):
            return []
    
    volumetric_avatar = DummyVolumetricAvatar()
    model = VASAModel(config, volumetric_avatar)
    model.eval()
    
    # Move to CPU to avoid device issues
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy inputs
    batch_size = 1
    sequence_length = 20
    context_size = 10
    
    # Initialize motion data
    motion_data = {
        'theta': torch.randn(batch_size, sequence_length, 3, 4, device=device),
        'rotation': torch.randn(batch_size, sequence_length, 3, device=device),
        'translation': torch.randn(batch_size, sequence_length, 3, device=device),
        'scale': torch.ones(batch_size, sequence_length, 3, device=device),
        'expression_embed': torch.randn(batch_size, sequence_length, 128, device=device)
    }
    
    # Initialize conditions
    conditions = {
        'audio_features': torch.randn(batch_size, sequence_length, 768, device=device),
        'gaze': torch.randn(batch_size, sequence_length, 2, device=device),
        'head_distance': torch.randn(batch_size, sequence_length, 1, device=device),
        'emotion': torch.randn(batch_size, sequence_length, 2, device=device),
        'speed_bucket': torch.zeros(batch_size, sequence_length, 1, device=device)
    }
    
    # Test 1: Without prev_context
    logger.info("\n=== Test 1: Without prev_context ===")
    try:
        with torch.no_grad():
            output1 = model.generate_sequence(
                initial_pose=motion_data,
                initial_dynamics=motion_data['expression_embed'],
                conditions=conditions,
                num_steps=10,
                eta=0.5,
                prev_context=None
            )
        logger.info(f"✓ Without prev_context: output shape = {output1['expression_embed'].shape}")
    except Exception as e:
        logger.error(f"✗ Failed without prev_context: {e}")
        return False
    
    # Test 2: With zero prev_context (first window)
    logger.info("\n=== Test 2: With zero prev_context (first window) ===")
    prev_context = {
        'prev_theta': torch.zeros(batch_size, context_size, 3, 4, device=device),
        'prev_rotation': torch.zeros(batch_size, context_size, 3, device=device),
        'prev_translation': torch.zeros(batch_size, context_size, 3, device=device),
        'prev_expression': torch.zeros(batch_size, context_size, 64, device=device),  # Use config dimension
        'prev_audio': torch.zeros(batch_size, context_size, 64, device=device)  # Use config dimension
    }
    
    try:
        with torch.no_grad():
            output2 = model.generate_sequence(
                initial_pose=motion_data,
                initial_dynamics=motion_data['expression_embed'],
                conditions=conditions,
                num_steps=10,
                eta=0.5,
                prev_context=prev_context
            )
        logger.info(f"✓ With zero prev_context: output shape = {output2['expression_embed'].shape}")
    except Exception as e:
        logger.error(f"✗ Failed with zero prev_context: {e}")
        return False
    
    # Test 3: With non-zero prev_context (subsequent windows)
    logger.info("\n=== Test 3: With non-zero prev_context (subsequent windows) ===")
    prev_context = {
        'prev_theta': torch.randn(batch_size, context_size, 3, 4, device=device) * 0.1,
        'prev_rotation': torch.randn(batch_size, context_size, 3, device=device) * 0.1,
        'prev_translation': torch.randn(batch_size, context_size, 3, device=device) * 0.1,
        'prev_expression': torch.randn(batch_size, context_size, 64, device=device) * 0.1,
        'prev_audio': torch.randn(batch_size, context_size, 64, device=device) * 0.1
    }
    
    try:
        with torch.no_grad():
            output3 = model.generate_sequence(
                initial_pose=motion_data,
                initial_dynamics=motion_data['expression_embed'],
                conditions=conditions,
                num_steps=10,
                eta=0.5,
                prev_context=prev_context
            )
        logger.info(f"✓ With non-zero prev_context: output shape = {output3['expression_embed'].shape}")
    except Exception as e:
        logger.error(f"✗ Failed with non-zero prev_context: {e}")
        return False
    
    # Compare outputs to check if prev_context has effect
    logger.info("\n=== Comparing outputs ===")
    
    # Compare output1 (no context) vs output2 (zero context)
    diff_1_2 = (output1['expression_embed'] - output2['expression_embed']).abs().mean().item()
    logger.info(f"Difference between no context and zero context: {diff_1_2:.6f}")
    
    # Compare output2 (zero context) vs output3 (non-zero context)
    diff_2_3 = (output2['expression_embed'] - output3['expression_embed']).abs().mean().item()
    logger.info(f"Difference between zero and non-zero context: {diff_2_3:.6f}")
    
    # Check if outputs have variation
    std1 = output1['expression_embed'].std().item()
    std2 = output2['expression_embed'].std().item()
    std3 = output3['expression_embed'].std().item()
    
    logger.info(f"\nOutput standard deviations:")
    logger.info(f"  No context: {std1:.6f}")
    logger.info(f"  Zero context: {std2:.6f}")
    logger.info(f"  Non-zero context: {std3:.6f}")
    
    # Success criteria
    if diff_2_3 > 0.001:  # Non-zero context should have some effect
        logger.info("\n✓ SUCCESS: prev_context is having an effect on generation!")
        return True
    else:
        logger.warning("\n⚠ WARNING: prev_context seems to have minimal effect")
        return False

if __name__ == "__main__":
    success = test_prev_context()
    if success:
        logger.info("\n✅ All prev_context tests passed!")
    else:
        logger.error("\n❌ Some prev_context tests failed")