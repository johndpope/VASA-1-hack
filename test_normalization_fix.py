#!/usr/bin/env python3
"""Test script to verify normalization fixes preserve audio variance differences"""

import torch
import numpy as np
from vasa_model import VASAModel
from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger
import os

def test_normalization_fix():
    """Test that our normalization fixes preserve variance differences between silent and speech audio"""

    # Set log level to see debug output
    os.environ['VASA_LOG_LEVEL'] = 'DEBUG'

    logger.info("Testing normalization fixes...")

    # Initialize just the condition embedding module for testing
    logger.info("Loading condition embedding module...")
    from omegaconf import OmegaConf
    from vasa_model import EfficientConditionEmbedding
    import torch.nn as nn

    config = OmegaConf.load("overfit_config.yaml")

    # Create just the condition embedding module
    condition_embedding = EfficientConditionEmbedding(
        model_dim=config.model.hidden_dim,
        max_seq_len=config.motion.window_size
    )
    condition_embedding.eval()

    # Create test conditions with silent vs speech audio
    B, T = 1, 50  # Batch size 1, 50 frames
    device = next(condition_embedding.parameters()).device

    # Silent audio (all zeros)
    silent_audio = torch.zeros(B, T, 768, device=device)

    # Speech audio (random values simulating wav2vec2 features)
    speech_audio = torch.randn(B, T, 768, device=device) * 0.3  # Typical wav2vec2 magnitude

    # Create dummy conditions for both cases
    base_conditions = {
        'gaze': torch.randn(B, T, 2, device=device) * 0.1,
        'head_distance': torch.ones(B, T, 1, device=device) * 0.5,
        'emotion': torch.randn(B, T, 2, device=device) * 0.1,
        'blink_state': torch.zeros(B, T, 3, device=device)
    }

    # Test with silent audio
    logger.info("\n=== Testing SILENT audio ===")
    silent_conditions = {**base_conditions, 'audio_features': silent_audio}
    with torch.no_grad():
        silent_embedding = condition_embedding(silent_conditions)
    silent_var = silent_embedding.var().item()
    silent_mean = silent_embedding.mean().item()
    silent_l2 = torch.norm(silent_embedding).item()

    logger.info(f"Silent embedding - Variance: {silent_var:.6f}, Mean: {silent_mean:.6f}, L2 norm: {silent_l2:.4f}")

    # Test with speech audio
    logger.info("\n=== Testing SPEECH audio ===")
    speech_conditions = {**base_conditions, 'audio_features': speech_audio}
    with torch.no_grad():
        speech_embedding = condition_embedding(speech_conditions)
    speech_var = speech_embedding.var().item()
    speech_mean = speech_embedding.mean().item()
    speech_l2 = torch.norm(speech_embedding).item()

    logger.info(f"Speech embedding - Variance: {speech_var:.6f}, Mean: {speech_mean:.6f}, L2 norm: {speech_l2:.4f}")

    # Compute differences
    var_ratio = speech_var / (silent_var + 1e-8)
    l2_ratio = speech_l2 / (silent_l2 + 1e-8)

    logger.info("\n=== RESULTS ===")
    logger.info(f"Variance ratio (speech/silent): {var_ratio:.4f}")
    logger.info(f"L2 norm ratio (speech/silent): {l2_ratio:.4f}")

    # Check if the fix is working
    if var_ratio > 1.5:  # Speech should have noticeably higher variance
        logger.info("✓ SUCCESS: Speech has significantly higher variance than silence")
        logger.info("✓ The normalization fix is preserving audio variance differences!")
    else:
        logger.warning("✗ ISSUE: Speech and silence have similar variance")
        logger.warning("✗ The model may still need retraining with these fixes")

    if l2_ratio > 1.2:  # Speech should have higher magnitude
        logger.info("✓ SUCCESS: Speech has higher magnitude than silence")
    else:
        logger.warning("✗ ISSUE: Speech and silence have similar magnitude")

    return {
        'silent_var': silent_var,
        'speech_var': speech_var,
        'var_ratio': var_ratio,
        'l2_ratio': l2_ratio,
        'success': var_ratio > 1.5 and l2_ratio > 1.2
    }

if __name__ == "__main__":
    results = test_normalization_fix()

    if not results['success']:
        logger.warning("\n⚠️  The model needs to be retrained with these fixes in place")
        logger.warning("⚠️  The current checkpoint was trained with broken normalization")
    else:
        logger.info("\n✅ Normalization fixes are working correctly!")
        logger.info("✅ You can now proceed with training or inference")