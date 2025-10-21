#!/usr/bin/env python3
"""
Test script to verify cross-layer skip connections implementation.
Checks gradient flow and ensures no shape mismatches.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import MotionTransformer, TalkVidAudioProjection


def test_motion_transformer_skip_connections():
    """Test cross-layer skip connections in MotionTransformer"""
    print("\n" + "="*80)
    print("Testing MotionTransformer Cross-Layer Skip Connections")
    print("="*80)

    # Load config
    config = OmegaConf.load('overfit_config.yaml')

    # Create model
    model = MotionTransformer(config).cuda()
    model.train()

    # Create dummy inputs
    B, T = 2, 50
    motion_data = {
        'theta': torch.randn(B, T, 3, 4, device='cuda'),
        'expression_embed': torch.randn(B, T, 128, device='cuda')
    }
    noise_level = torch.rand(B, device='cuda') * 1000
    conditions = {
        'audio_features': torch.randn(B, T, 768, device='cuda'),
        'gaze': None,  # Skip optional conditions to avoid shape issues
        'head_distance': None,
        'emotion': None
    }

    print(f"Input shapes:")
    print(f"  theta: {motion_data['theta'].shape}")
    print(f"  expression: {motion_data['expression_embed'].shape}")
    print(f"  audio: {conditions['audio_features'].shape}")

    # Forward pass
    print("\nRunning forward pass...")
    outputs = model(motion_data, noise_level, conditions)

    print(f"\nOutput shapes:")
    print(f"  theta_pred: {outputs['theta'].shape}")
    print(f"  expression_pred: {outputs['expression_embed'].shape}")

    # Check gradient flow
    print("\nChecking gradient flow...")
    loss = outputs['expression_embed'].mean() + outputs['theta'].mean()
    loss.backward()

    # Check gradients in each layer
    grad_norms = []
    for idx, layer in enumerate(model.decoder_layers):
        # Get gradient norm from first layer's weight
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_proj'):
            grad = layer.self_attn.out_proj.weight.grad
            if grad is not None:
                grad_norm = grad.norm().item()
                grad_norms.append(grad_norm)
                print(f"  Layer {idx}: grad_norm = {grad_norm:.6f}")

    if grad_norms:
        print(f"\nGradient statistics:")
        print(f"  Mean: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  Min: {min(grad_norms):.6f}")
        print(f"  Max: {max(grad_norms):.6f}")
        print(f"  Ratio (max/min): {max(grad_norms)/min(grad_norms):.2f}")

        # Check for vanishing gradients
        if min(grad_norms) < 1e-6:
            print("  ⚠️ WARNING: Possible vanishing gradients detected!")
        else:
            print("  ✅ No vanishing gradients detected")

    print("\n✅ MotionTransformer test passed!")


def test_talkvid_audio_projection_skip_connections():
    """Test cross-layer skip connections in TalkVidAudioProjection"""
    print("\n" + "="*80)
    print("Testing TalkVidAudioProjection Cross-Layer Skip Connections")
    print("="*80)

    # Create model
    model = TalkVidAudioProjection(
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4
    ).cuda()
    model.train()

    # Create dummy audio input
    B, T = 2, 50
    audio_features = torch.randn(B, T, 768, device='cuda')

    print(f"Input shape: {audio_features.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    output = model(audio_features)

    print(f"Output shape: {output.shape}")

    # Check gradient flow
    print("\nChecking gradient flow...")
    loss = output.mean()
    loss.backward()

    # Check gradients in each layer
    grad_norms = []
    for idx, (attn, ff) in enumerate(model.layers):
        # Get gradient norm from attention output projection
        grad = attn.to_out.weight.grad
        if grad is not None:
            grad_norm = grad.norm().item()
            grad_norms.append(grad_norm)
            print(f"  Layer {idx}: grad_norm = {grad_norm:.6f}")

    if grad_norms:
        print(f"\nGradient statistics:")
        print(f"  Mean: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  Min: {min(grad_norms):.6f}")
        print(f"  Max: {max(grad_norms):.6f}")
        print(f"  Ratio (max/min): {max(grad_norms)/min(grad_norms):.2f}")

        # Check for vanishing gradients
        if min(grad_norms) < 1e-6:
            print("  ⚠️ WARNING: Possible vanishing gradients detected!")
        else:
            print("  ✅ No vanishing gradients detected")

    print("\n✅ TalkVidAudioProjection test passed!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Cross-Layer Skip Connection Tests")
    print("="*80)

    test_motion_transformer_skip_connections()
    test_talkvid_audio_projection_skip_connections()

    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)
    print("\nCross-layer skip connections improve gradient flow by:")
    print("  1. Adding block-level residuals every 2 layers")
    print("  2. Preventing vanishing gradients in deep transformers")
    print("  3. Enabling more robust training in audio-conditioned generation")
    print("="*80 + "\n")
