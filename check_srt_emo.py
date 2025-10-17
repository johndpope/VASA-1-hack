#!/usr/bin/env python3
"""
Diagnostic script to check SRT (Scale/Rotation/Translation) values and emo_frames.
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_h5_cache():
    """Check SRT values and emo_frames availability in H5 cache."""
    cache_path = Path("cache_single_bucket/all_windows_cache.h5")

    if not cache_path.exists():
        logger.error(f"Cache not found: {cache_path}")
        return

    with h5py.File(cache_path, 'r') as f:
        num_windows = f.attrs.get('num_windows', 0)
        logger.info(f"Total windows in cache: {num_windows}")

        # Check first 10 windows
        for i in range(min(10, num_windows)):
            window_key = f'window_{i}'
            if window_key not in f:
                continue

            window = f[window_key]

            # Check what's available
            has_emo_frames = 'emo_frames' in window
            has_theta = 'theta' in window
            has_scale = 'scale' in window
            has_rotation = 'rotation' in window
            has_translation = 'translation' in window

            logger.info(f"\nWindow {i}:")
            logger.info(f"  has_emo_frames: {has_emo_frames}")
            logger.info(f"  has_theta: {has_theta}")
            logger.info(f"  has_scale: {has_scale}")
            logger.info(f"  has_rotation: {has_rotation}")
            logger.info(f"  has_translation: {has_translation}")

            # Check SRT values
            if has_scale:
                scale = torch.from_numpy(window['scale'][()])
                logger.info(f"  scale shape: {scale.shape}, range: [{scale.min():.4f}, {scale.max():.4f}], mean: {scale.mean():.4f}")

            if has_rotation:
                rotation = torch.from_numpy(window['rotation'][()])
                logger.info(f"  rotation shape: {rotation.shape}, range: [{rotation.min():.4f}, {rotation.max():.4f}]")

            if has_translation:
                translation = torch.from_numpy(window['translation'][()])
                logger.info(f"  translation shape: {translation.shape}, range: [{translation.min():.4f}, {translation.max():.4f}]")

            if has_theta:
                theta = torch.from_numpy(window['theta'][()])
                logger.info(f"  theta shape: {theta.shape}, range: [{theta.min():.4f}, {theta.max():.4f}]")

            # Check emo_frames in H5 (should not be there, only on disk)
            if has_emo_frames:
                emo_shape = window['emo_frames'].shape
                logger.info(f"  emo_frames in H5: shape={emo_shape}")

def check_disk_emo_frames():
    """Check emo_frames on disk."""
    emo_dir = Path("cache_single_bucket/emo_frames")

    if not emo_dir.exists():
        logger.error(f"EMO frames directory not found: {emo_dir}")
        return

    video_dirs = list(emo_dir.iterdir())
    logger.info(f"\nEMO frames disk cache:")
    logger.info(f"  Total video directories: {len(video_dirs)}")

    # Check first video
    if video_dirs:
        first_video = video_dirs[0]
        window_dirs = list(first_video.iterdir())
        logger.info(f"  Example video {first_video.name}:")
        logger.info(f"    Window directories: {len(window_dirs)}")

        if window_dirs:
            first_window = window_dirs[0]
            frames = list(first_window.glob("*.png"))
            logger.info(f"    Example window {first_window.name}: {len(frames)} PNG frames")

def main():
    logger.info("=== Checking SRT values and emo_frames ===\n")
    check_h5_cache()
    check_disk_emo_frames()

    logger.info("\n=== Summary ===")
    logger.info("If SRT values look wrong (e.g., scale not around 1.0, rotation/translation extreme):")
    logger.info("  - This could indicate issues with 3DDFA processing")
    logger.info("  - Check the EMO volumetric avatar outputs")
    logger.info("\nIf emo_frames are missing from batches:")
    logger.info("  - Check that emo_frames exist on disk")
    logger.info("  - Verify FrameDiskCache is loading them correctly")
    logger.info("  - Some windows may legitimately not have emo_frames if generation failed")

if __name__ == "__main__":
    main()
