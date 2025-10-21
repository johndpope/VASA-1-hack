#!/usr/bin/env python3
"""
Inspect EMO frames from disk cache to diagnose black frame issue
"""

import torch
import numpy as np
from pathlib import Path
from frame_disk_cache import FrameDiskCache
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_emo_frames(cache_dir: str = "cache_single_bucket"):
    """Inspect EMO frames in cache to see their value ranges"""

    cache_dir = Path(cache_dir)
    emo_cache = FrameDiskCache(cache_dir, frame_type='emo_frames')

    logger.info("=" * 80)
    logger.info("Inspecting EMO Frame Cache")
    logger.info("=" * 80)

    # Get stats
    stats = emo_cache.get_cache_stats()
    logger.info(f"\n📊 Cache Statistics:")
    logger.info(f"  Total videos: {stats['total_videos']}")
    logger.info(f"  Total windows: {stats['total_windows']}")
    logger.info(f"  Total frames: {stats['total_frames']}")
    logger.info(f"  Total size: {stats['total_size_gb']:.4f} GB")

    if stats['total_frames'] == 0:
        logger.warning("⚠️ No EMO frames found in cache!")
        return

    # Find first video and window
    video_dirs = list(emo_cache.root.glob("*"))
    if not video_dirs:
        logger.warning("⚠️ No video directories found!")
        return

    video_hash = video_dirs[0].name
    window_dirs = list(video_dirs[0].glob("window_*"))
    if not window_dirs:
        logger.warning("⚠️ No window directories found!")
        return

    window_idx = int(window_dirs[0].name.split("_")[1])

    logger.info(f"\n🔍 Inspecting video hash: {video_hash}, window: {window_idx}")

    # Load first few frames
    frame_files = sorted(window_dirs[0].glob("frame_*.png"))[:3]

    for i, frame_file in enumerate(frame_files):
        logger.info(f"\n📷 Frame {i}: {frame_file.name}")

        # Load with PIL
        img = Image.open(frame_file)
        arr = np.array(img)

        logger.info(f"  Shape: {arr.shape}")
        logger.info(f"  Dtype: {arr.dtype}")
        logger.info(f"  Min: {arr.min()}")
        logger.info(f"  Max: {arr.max()}")
        logger.info(f"  Mean: {arr.mean():.2f}")
        logger.info(f"  Std: {arr.std():.2f}")

        # Check if all zeros
        if arr.max() == 0:
            logger.error(f"  ❌ Frame is all zeros (black)!")
        elif arr.max() < 10:
            logger.warning(f"  ⚠️ Frame is very dark (max={arr.max()})")
        else:
            logger.info(f"  ✅ Frame has valid values")

        # Show histogram of first channel
        hist, bins = np.histogram(arr[:, :, 0].flatten(), bins=10)
        logger.info(f"  Histogram (R channel): {hist}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect EMO frames in cache')
    parser.add_argument('--cache-dir', type=str, default='cache_single_bucket',
                       help='Cache directory')
    args = parser.parse_args()

    inspect_emo_frames(args.cache_dir)
