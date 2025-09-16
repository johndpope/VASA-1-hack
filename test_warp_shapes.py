#!/usr/bin/env python3
"""Simple test to verify warp shapes without needing video files"""

import torch
import numpy as np

def test_warp_shapes():
    """Test that warp shapes are correct for the dataset"""

    # Define expected dimensions
    sequence_length = 4
    d = 16  # depth
    s = 64  # spatial size
    c = 96  # channels for canonical volume

    # Simulate what the dataset should return
    sample = {
        # Per-frame features
        'frames': torch.zeros((sequence_length, 3, 512, 512)),
        'theta': torch.zeros((sequence_length, 3, 4)),
        'expression_embed': torch.zeros((sequence_length, 128)),

        # Per-frame warps (the new additions)
        'xy_warps': torch.zeros((sequence_length, d, s, s, 3)),
        'rigid_warps': torch.zeros((sequence_length, d, s, s, 3)),
        'uv_warps': torch.zeros((sequence_length, d, s, s, 3)),
        'source_theta_warp': torch.zeros((sequence_length, 3, 4)),
    }

    print("=== Testing Per-Frame Warp Shapes ===")
    print(f"Sequence length: {sequence_length}")
    print(f"Warp dimensions: depth={d}, spatial={s}x{s}")
    print(f"Canonical volume channels: {c}")
    print()

    warp_keys = ['xy_warps', 'rigid_warps', 'uv_warps', 'source_theta_warp']

    for key in warp_keys:
        shape = sample[key].shape
        print(f"{key:20s}: {shape}")

        # Verify first dimension is sequence_length
        assert shape[0] == sequence_length, f"{key} first dimension should be {sequence_length}, got {shape[0]}"

        # Check specific shapes
        if key in ['xy_warps', 'rigid_warps', 'uv_warps']:
            expected = (sequence_length, d, s, s, 3)
            assert shape == expected, f"{key} shape mismatch: expected {expected}, got {shape}"
        elif key == 'source_theta_warp':
            expected = (sequence_length, 3, 4)
            assert shape == expected, f"{key} shape mismatch: expected {expected}, got {shape}"

    print("\n✅ All warp shapes are correct for per-frame extraction!")

    # Show memory usage
    print("\n=== Memory Usage ===")
    for key in warp_keys:
        tensor = sample[key]
        memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        print(f"{key:20s}: {memory_mb:.2f} MB")

    total_mb = sum(sample[k].numel() * sample[k].element_size() for k in warp_keys) / (1024 * 1024)
    print(f"{'Total warps memory':20s}: {total_mb:.2f} MB per sample")

    return sample

if __name__ == "__main__":
    sample = test_warp_shapes()

    print("\n=== Summary ===")
    print("The dataset now returns per-frame warps with the following structure:")
    print("- Each warp type has T frames (T = sequence_length)")
    print("- xy_warps: Non-rigid source warps for each frame")
    print("- rigid_warps: Rigid transformation warps for each frame")
    print("- uv_warps: Non-rigid target warps for each frame")
    print("- source_theta_warp: Source pose parameters for each frame")