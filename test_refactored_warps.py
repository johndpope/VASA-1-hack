#!/usr/bin/env python3
"""Test script for refactored warp calculation and decoding functions."""

import torch
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import sys
import argparse
import os

# Add the paths to sys.path
sys.path.insert(0, '/media/2TB/VASA-1-hack')
sys.path.insert(0, '/media/2TB/VASA-1-hack/nemo')

# Import from create_video_face_swap (in root dir)
from create_video_face_swap import (
    load_volumetric_model,
    load_image_tensor,
    extract_identity_features,
    apply_target_to_identity,
    calculate_target_warps,
    decode_with_warps,
    load_cached_warps
)

# Import additional tools
from nemo.pipeline5 import Model as Pipeline5Model
from nemo.face_detect import FaceDetector
from nemo.pose_estimate import PoseEstimator

def load_model(checkpoint_path):
    """Load the model from checkpoint."""
    model = Pipeline5Model.from_checkpoint(checkpoint_path)
    model = model.cuda()
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image."""
    return load_image_tensor(image_path)

def test_warp_caching(model_path, source_path: str, target_path: str, cache_path: str):
    """Test that cached warps produce identical results to fresh calculation."""

    print("Loading model...")
    model = load_volumetric_model()

    # Initialize face detector and pose estimator
    face_detector = FaceDetector()
    pose_estimator = PoseEstimator()

    # Load and preprocess images
    print("Loading images...")
    source_img = preprocess_image(source_path)
    target_img = preprocess_image(target_path)

    print(f"Source shape: {source_img.shape}")
    print(f"Target shape: {target_img.shape}")

    # Extract identity features
    print("Extracting identity features...")
    identity_info = extract_identity_features(model, source_img, face_detector, pose_estimator)

    # Method 1: Original combined function
    print("\n1. Testing original apply_target_to_identity...")
    with torch.no_grad():
        result_original, warp_data_original = apply_target_to_identity(
            model, identity_info, target_img,
            cache_h5_path=cache_path + ".original.h5",
            use_cached=False  # Force fresh calculation
        )
    print(f"Original result shape: {result_original.shape}")

    # Method 2: Separated calculation and decoding
    print("\n2. Testing separated calculate_target_warps + decode_with_warps...")
    with torch.no_grad():
        # Calculate warps and cache them
        warp_data = calculate_target_warps(
            model, identity_info, target_img,
            cache_h5_path=cache_path + ".warps.h5",
            frame_idx=0
        )

        # Decode using the warps
        result_separated = decode_with_warps(
            model, identity_info, warp_data, target_img
        )
    print(f"Separated result shape: {result_separated.shape}")

    # Method 3: Load cached warps and decode
    print("\n3. Testing load_cached_warps + decode_with_warps...")
    with torch.no_grad():
        # Load warps from cache
        loaded_warps = load_cached_warps(cache_path + ".warps.h5", frame_idx=0)

        if loaded_warps is None:
            print("ERROR: Could not load cached warps!")
            return False

        # Decode using loaded warps
        result_cached = decode_with_warps(
            model, identity_info, loaded_warps, target_img
        )
    print(f"Cached result shape: {result_cached.shape}")

    # Compare results
    print("\n4. Comparing results...")

    # Compare original vs separated
    diff_sep = torch.abs(result_original - result_separated).mean().item()
    print(f"Diff (original vs separated): {diff_sep:.6f}")

    # Compare original vs cached
    diff_cache = torch.abs(result_original - result_cached).mean().item()
    print(f"Diff (original vs cached): {diff_cache:.6f}")

    # Compare separated vs cached
    diff_sep_cache = torch.abs(result_separated - result_cached).mean().item()
    print(f"Diff (separated vs cached): {diff_sep_cache:.6f}")

    # Save results as images for visual comparison
    print("\n5. Saving output images...")
    for name, tensor in [
        ("original", result_original),
        ("separated", result_separated),
        ("cached", result_cached)
    ]:
        # Convert to image
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip((img_np + 1) * 127.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        output_path = f"{cache_path}.{name}.png"
        img.save(output_path)
        print(f"Saved: {output_path}")

    # Check if differences are negligible (allowing for floating point errors)
    tolerance = 1e-5
    success = all([
        diff_sep < tolerance,
        diff_cache < tolerance,
        diff_sep_cache < tolerance
    ])

    if success:
        print("\n✅ TEST PASSED: All methods produce identical results!")
    else:
        print("\n❌ TEST FAILED: Results differ beyond tolerance!")

    # Test loading multiple frames from cache
    print("\n6. Testing multi-frame cache loading...")
    test_multi_frame_cache(model, identity_info, cache_path + ".multi.h5")

    return success

def test_multi_frame_cache(model, identity_info, cache_path):
    """Test caching and loading multiple frames."""

    print("Creating multi-frame cache...")

    # Generate dummy target images (just variations for testing)
    base_target = identity_info['source_img'].clone()

    with h5py.File(cache_path, 'w') as f:
        for frame_idx in range(3):
            # Add slight variation to simulate different frames
            target_variation = base_target + torch.randn_like(base_target) * 0.01

            # Calculate warps for this frame
            warp_data = calculate_target_warps(
                model, identity_info, target_variation,
                cache_h5_path=None,  # Don't save, we'll do it manually
                frame_idx=frame_idx
            )

            # Save warps to the multi-frame file
            frame_group = f.create_group(f'frame_{frame_idx:06d}')
            for key, value in warp_data.items():
                if isinstance(value, torch.Tensor):
                    frame_group.create_dataset(
                        key,
                        data=value.cpu().numpy(),
                        compression='gzip',
                        compression_opts=4
                    )

            print(f"  Cached frame {frame_idx}")

    print("Loading and testing multi-frame cache...")

    for frame_idx in range(3):
        loaded_warps = load_cached_warps(cache_path, frame_idx)
        if loaded_warps is None:
            print(f"  ❌ Failed to load frame {frame_idx}")
        else:
            print(f"  ✅ Successfully loaded frame {frame_idx}")
            # Verify we can decode with it
            with torch.no_grad():
                result = decode_with_warps(model, identity_info, loaded_warps)
                print(f"     Decoded shape: {result.shape}")

    print("Multi-frame test complete!")

def main():
    parser = argparse.ArgumentParser(description='Test refactored warp functions')
    parser.add_argument('--source', default='/media/2TB/VASA-1-hack/data/IMG_1.png',
                       help='Source image path')
    parser.add_argument('--target', default='/media/2TB/VASA-1-hack/data/IMG_1.png',
                       help='Target image path (can be same as source for testing)')
    parser.add_argument('--cache', default='/media/2TB/VASA-1-hack/test_warps',
                       help='Base path for cache files (extensions will be added)')

    args = parser.parse_args()

    # Check if files exist
    for path, name in [(args.source, "Source"), (args.target, "Target")]:
        if not Path(path).exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    # Run test (no model path needed - uses default)
    success = test_warp_caching(None, args.source, args.target, args.cache)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()