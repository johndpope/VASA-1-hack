#!/usr/bin/env python3
"""Test prev_context in EfficientConditionEmbedding."""

import torch
import logging
from vasa_model import EfficientConditionEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prev_context_embedding():
    """Test the prev_context in condition embedding."""
    
    logger.info("Testing EfficientConditionEmbedding with prev_context...")
    
    # Initialize embedding layer
    embed = EfficientConditionEmbedding(model_dim=512, max_seq_len=20)
    embed.eval()
    
    # Move to CPU
    device = torch.device('cpu')
    embed = embed.to(device)
    
    batch_size = 1
    seq_len = 20
    context_size = 10
    
    # Create conditions
    conditions = {
        'audio_features': torch.randn(batch_size, seq_len, 768, device=device),
        'gaze': torch.randn(batch_size, seq_len, 2, device=device),
        'head_distance': torch.randn(batch_size, seq_len, 1, device=device),
        'emotion': torch.randn(batch_size, seq_len, 2, device=device),
        'speed_bucket': torch.zeros(batch_size, seq_len, 1, device=device)
    }
    
    # Test 1: Without prev_context
    logger.info("\n=== Test 1: Without prev_context ===")
    with torch.no_grad():
        output1 = embed(conditions, prev_context=None)
    logger.info(f"✓ Output shape without prev_context: {output1.shape}")
    logger.info(f"  Output std: {output1.std().item():.6f}")
    
    # Test 2: With prev_context
    logger.info("\n=== Test 2: With prev_context ===")
    prev_context = {
        'prev_theta': torch.randn(batch_size, seq_len, 3, 4, device=device) * 0.1,
        'prev_rotation': torch.randn(batch_size, seq_len, 3, device=device) * 0.1,
        'prev_translation': torch.randn(batch_size, seq_len, 3, device=device) * 0.1,
        'prev_expression': torch.randn(batch_size, seq_len, 64, device=device) * 0.1,
        'prev_audio': torch.randn(batch_size, seq_len, 64, device=device) * 0.1
    }
    
    with torch.no_grad():
        output2 = embed(conditions, prev_context=prev_context)
    logger.info(f"✓ Output shape with prev_context: {output2.shape}")
    logger.info(f"  Output std: {output2.std().item():.6f}")
    
    # Compare outputs
    diff = (output1 - output2).abs().mean().item()
    logger.info(f"\n=== Comparison ===")
    logger.info(f"Mean absolute difference: {diff:.6f}")
    
    # Check specific channel ranges for prev_context
    channel_layout = embed.channel_layout
    
    if 'prev_theta' in channel_layout:
        start, end = channel_layout['prev_theta']
        prev_theta_output = output2[:, :, start:end]
        logger.info(f"\nprev_theta channel [{start}:{end}]:")
        logger.info(f"  Mean: {prev_theta_output.mean().item():.6f}")
        logger.info(f"  Std: {prev_theta_output.std().item():.6f}")
    
    if 'prev_expression' in channel_layout:
        start, end = channel_layout['prev_expression']
        prev_expr_output = output2[:, :, start:end]
        logger.info(f"\nprev_expression channel [{start}:{end}]:")
        logger.info(f"  Mean: {prev_expr_output.mean().item():.6f}")
        logger.info(f"  Std: {prev_expr_output.std().item():.6f}")
    
    # Success check
    if diff > 0.001:
        logger.info("\n✅ SUCCESS: prev_context is properly integrated!")
        return True
    else:
        logger.warning("\n⚠ WARNING: prev_context may not be working correctly")
        return False

if __name__ == "__main__":
    success = test_prev_context_embedding()
    if not success:
        logger.error("Test failed!")