#!/usr/bin/env python3
"""Simplified test script for refactored warp calculation and decoding functions."""

import torch
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import sys
import argparse

# Add the paths to sys.path
sys.path.insert(0, '/media/2TB/VASA-1-hack')

# Import from create_video_face_swap
from create_video_face_swap import (
    load_volumetric_model,
    load_image_tensor,
    extract_identity_features,
    apply_target_to_identity,
    calculate_target_warps,
    decode_with_warps,
    load_cached_warps
)

def test_warp_caching():
    """Test that cached warps produce identical results to fresh calculation."""

    print("=" * 60)
    print("Testing Refactored Warp Functions")
    print("=" * 60)

    # Test parameters
    source_path = "/media/2TB/VASA-1-hack/data/IMG_1.png"
    target_path = "/media/2TB/VASA-1-hack/data/IMG_1.png"  # Using same image for simplicity
    cache_base = "/media/2TB/VASA-1-hack/test_warps"

    print("\nLoading model...")
    model = load_volumetric_model()

    # Load images
    print("Loading images...")
    source_img = load_image_tensor(source_path)
    target_img = load_image_tensor(target_path)
    print(f"  Source shape: {source_img.shape}")
    print(f"  Target shape: {target_img.shape}")

    # Extract identity features (using None for missing detectors)
    print("\nExtracting identity features...")
    identity_info = extract_identity_features(model, source_img, None, None)
    print(f"  Identity info keys: {list(identity_info.keys())}")

    # Test 1: Original combined function (force fresh calculation)
    print("\n" + "=" * 60)
    print("TEST 1: Original apply_target_to_identity (fresh calc)")
    print("=" * 60)

    cache_path_original = f"{cache_base}_original.h5"
    with torch.no_grad():
        result_original, warp_data_original = apply_target_to_identity(
            model, identity_info, target_img,
            cache_h5_path=cache_path_original,
            frame_idx=0,
            use_cached=False  # Force fresh calculation
        )
    print(f"✅ Result shape: {result_original.shape}")
    print(f"   Warp data keys: {list(warp_data_original.keys())}")

    # Test 2: Separated calculation and decoding
    print("\n" + "=" * 60)
    print("TEST 2: Separated warp calc + decode")
    print("=" * 60)

    cache_path_separated = f"{cache_base}_separated.h5"
    with torch.no_grad():
        # Calculate warps and cache them
        print("  Calculating warps...")
        warp_data = calculate_target_warps(
            model, identity_info, target_img,
            cache_h5_path=cache_path_separated,
            frame_idx=0
        )
        print(f"  ✅ Warp data keys: {list(warp_data.keys())}")

        # Decode using the warps
        print("  Decoding with warps...")
        result_separated = decode_with_warps(
            model, identity_info, warp_data, target_img
        )
        print(f"  ✅ Result shape: {result_separated.shape}")

    # Test 3: Load cached warps and decode
    print("\n" + "=" * 60)
    print("TEST 3: Load cached warps + decode")
    print("=" * 60)

    with torch.no_grad():
        # Load warps from cache
        print(f"  Loading from: {cache_path_separated}")
        loaded_warps = load_cached_warps(cache_path_separated, frame_idx=0)

        if loaded_warps is None:
            print("  ❌ ERROR: Could not load cached warps!")
            return False

        print(f"  ✅ Loaded warp keys: {list(loaded_warps.keys())}")

        # Decode using loaded warps
        print("  Decoding with loaded warps...")
        result_cached = decode_with_warps(
            model, identity_info, loaded_warps, target_img
        )
        print(f"  ✅ Result shape: {result_cached.shape}")

    # Test 4: Use apply_target_to_identity with use_cached=True
    print("\n" + "=" * 60)
    print("TEST 4: apply_target_to_identity with use_cached=True")
    print("=" * 60)

    with torch.no_grad():
        result_cached_auto, warp_data_cached = apply_target_to_identity(
            model, identity_info, target_img,
            cache_h5_path=cache_path_separated,
            frame_idx=0,
            use_cached=True  # Use cached warps
        )
    print(f"✅ Result shape: {result_cached_auto.shape}")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)

    # Compare tensors
    diff_sep = torch.abs(result_original - result_separated).mean().item()
    diff_cache = torch.abs(result_original - result_cached).mean().item()
    diff_cache_auto = torch.abs(result_original - result_cached_auto).mean().item()
    diff_sep_cache = torch.abs(result_separated - result_cached).mean().item()

    print(f"  Original vs Separated:    {diff_sep:.8f}")
    print(f"  Original vs Cached:       {diff_cache:.8f}")
    print(f"  Original vs Cached Auto:  {diff_cache_auto:.8f}")
    print(f"  Separated vs Cached:      {diff_sep_cache:.8f}")

    # Save results as images for visual comparison
    print("\nSaving output images...")
    for name, tensor in [
        ("1_original", result_original),
        ("2_separated", result_separated),
        ("3_cached", result_cached),
        ("4_cached_auto", result_cached_auto)
    ]:
        # Convert to image
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip((img_np + 1) * 127.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        output_path = f"{cache_base}_{name}.png"
        img.save(output_path)
        print(f"  Saved: {output_path}")

    # Check if differences are negligible (allowing for floating point errors)
    tolerance = 1e-5
    success = all([
        diff_sep < tolerance,
        diff_cache < tolerance,
        diff_cache_auto < tolerance,
        diff_sep_cache < tolerance
    ])

    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: All methods produce identical results!")
    else:
        print("⚠️  TEST WARNING: Results differ beyond tolerance!")
        print(f"   Tolerance: {tolerance}")
        print("   This may be expected due to floating point precision.")
    print("=" * 60)

    # Check file sizes
    print("\nCache file sizes:")
    for path in [cache_path_original, cache_path_separated]:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"  {Path(path).name}: {size_mb:.2f} MB")

    return success

if __name__ == "__main__":
    success = test_warp_caching()
    sys.exit(0 if success else 1)