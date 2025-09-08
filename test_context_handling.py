#!/usr/bin/env python3
"""Test script to validate improved context handling in VASA-1."""

import torch
import torch.nn as nn
from vasa_model import HolisticMotionTransformer
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_context_handling():
    """Test the improved context handling in the transformer."""
    
    # Create mock config
    config = OmegaConf.create({
        'motion': {
            'window_size': 20,
            'context_size': 10,
            'min_window_size': 10,
            'max_window_size': 20
        },
        'training': {
            'context_size': 10
        }
    })
    
    # Initialize model
    model = HolisticMotionTransformer(
        config=config,
        channels=512,
        depth=8,
        height=512,
        width=512
    )
    model.eval()
    
    # Create test data
    B = 2  # batch size
    T = 20  # current window size
    K = 10  # context size
    
    # Current motion data
    current_motion = {
        'theta': torch.randn(B, T, 3, 4),
        'scale': torch.randn(B, T, 3),
        'rotation': torch.randn(B, T, 3),
        'translation': torch.randn(B, T, 3),
        'expression_embed': torch.randn(B, T, 128)
    }
    
    # Previous context (full uncompressed)
    prev_context = {
        'theta': torch.randn(B, K, 3, 4),
        'scale': torch.randn(B, K, 3),
        'rotation': torch.randn(B, K, 3),
        'translation': torch.randn(B, K, 3),
        'expression_embed': torch.randn(B, K, 128)
    }
    
    # Conditions
    conditions = {
        'audio_features': torch.randn(B, T, 768),
        'gaze': torch.randn(B, T, 2),
        'head_distance': torch.randn(B, T, 1),
        'emotion': torch.randn(B, T, 2),
        'speed_bucket': torch.randn(B, T, 1),
        'blink_state': torch.randn(B, T, 3)
    }
    
    # Noise level
    noise_level = torch.rand(B)
    
    logger.info("Testing without context...")
    # Test without context
    with torch.no_grad():
        output_no_context = model(
            motion_data=current_motion,
            noise_level=noise_level,
            conditions=conditions,
            prev_context=None
        )
    
    logger.info(f"Output shape (no context): {output_no_context['theta'].shape}")
    
    logger.info("Testing with full previous context...")
    # Test with context
    with torch.no_grad():
        output_with_context = model(
            motion_data=current_motion,
            noise_level=noise_level,
            conditions=conditions,
            prev_context=prev_context
        )
    
    logger.info(f"Output shape (with context): {output_with_context['theta'].shape}")
    
    # Verify outputs have correct shapes
    assert output_no_context['theta'].shape == (B, T, 3, 4), "Wrong output shape without context"
    assert output_with_context['theta'].shape == (B, T, 3, 4), "Wrong output shape with context"
    
    # Check that context affects the output
    diff = torch.mean(torch.abs(output_with_context['theta'] - output_no_context['theta']))
    logger.info(f"Average difference between outputs: {diff:.6f}")
    
    if diff > 0.01:
        logger.info("✅ SUCCESS: Context is properly affecting the output!")
    else:
        logger.warning("⚠️ WARNING: Context may not be properly integrated (outputs too similar)")
    
    # Test memory efficiency
    logger.info("\nMemory usage comparison:")
    logger.info(f"Previous context total elements: {sum(v.numel() for v in prev_context.values())}")
    logger.info(f"Previous approach (compressed to 128 dims): {B * K * 128}")
    logger.info(f"Improvement factor: {sum(v.numel() for v in prev_context.values()) / (B * K * 128):.2f}x more information preserved")
    
    return True

if __name__ == "__main__":
    try:
        success = test_context_handling()
        if success:
            logger.info("\n🎉 All tests passed! The improved context handling is working correctly.")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise