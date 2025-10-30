#!/usr/bin/env python3
"""
Upsert Action Unit Ground Truth to Existing Cache Files

This script updates existing H5 cache files in cache_per_video/ to include
AU ground truth data. It extracts AUs from the original video frames and
adds them to each window in the cache.

Usage:
    python upsert_au_gt.py [--cache-dir cache_per_video] [--force]
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List
import cv2

from au_extractor import ActionUnitExtractor
from logger import logger


def extract_frames_from_video(video_path: str, start_frame: int, num_frames: int) -> List[np.ndarray]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def update_cache_with_au(cache_path: Path, au_extractor: ActionUnitExtractor, force: bool = False):
    """Update a single cache file with AU ground truth."""
    logger.info(f"Processing: {cache_path.name}")

    try:
        # Open in read/write mode
        with h5py.File(cache_path, 'r+') as f:
            # Get metadata
            if 'video_path' not in f.attrs:
                logger.warning(f"  No video_path in cache, skipping")
                return False

            video_path = f.attrs['video_path']
            if not Path(video_path).exists():
                logger.warning(f"  Video not found: {video_path}")
                return False

            num_windows = f.attrs.get('num_windows', 0)
            if num_windows == 0:
                logger.warning(f"  No windows in cache, skipping")
                return False

            logger.info(f"  Video: {video_path}")
            logger.info(f"  Windows: {num_windows}")

            # Process each window
            updated_count = 0
            for i in range(num_windows):
                window_key = f'window_{i}'
                if window_key not in f:
                    continue

                window_group = f[window_key]

                # Check if au_gt already exists
                if 'au_gt' in window_group and not force:
                    logger.debug(f"    Window {i}: AU GT already exists, skipping")
                    continue

                # Get window metadata
                if 'metadata' not in window_group:
                    logger.warning(f"    Window {i}: No metadata, skipping")
                    continue

                metadata = window_group['metadata']
                start_frame = metadata.attrs.get('start_frame', 0)
                window_size = metadata.attrs.get('window_size', 50)

                # Extract frames from video
                try:
                    frames = extract_frames_from_video(video_path, start_frame, window_size)
                    if len(frames) == 0:
                        logger.warning(f"    Window {i}: No frames extracted")
                        continue

                    # Extract AU ground truth
                    au_gt = au_extractor.extract_aus_from_video(
                        frames=frames,
                        num_queries=8
                    )  # [8, 16]

                    # Save to cache
                    if 'au_gt' in window_group:
                        del window_group['au_gt']

                    window_group.create_dataset(
                        'au_gt',
                        data=au_gt.cpu().numpy(),
                        compression='gzip',
                        compression_opts=4
                    )

                    updated_count += 1
                    if updated_count % 10 == 0:
                        logger.info(f"    Updated {updated_count}/{num_windows} windows")

                except Exception as e:
                    logger.error(f"    Window {i}: Error extracting AUs: {e}")
                    continue

            logger.info(f"  ✅ Updated {updated_count}/{num_windows} windows")
            return updated_count > 0

    except Exception as e:
        logger.error(f"  ❌ Error processing cache: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description='Upsert AU ground truth to cache files')
    parser.add_argument('--cache-dir', type=str, default='cache_per_video',
                       help='Cache directory (default: cache_per_video)')
    parser.add_argument('--force', action='store_true',
                       help='Force update even if au_gt already exists')
    parser.add_argument('--video-filter', type=str, default=None,
                       help='Only process caches matching this video name pattern')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"❌ Cache directory not found: {cache_dir}")
        return

    # Find all cache files
    cache_files = list(cache_dir.glob('*/metadata.h5'))
    if not cache_files:
        logger.error(f"❌ No cache files found in {cache_dir}")
        return

    # Filter by video name if specified
    if args.video_filter:
        cache_files = [f for f in cache_files if args.video_filter in str(f)]
        logger.info(f"Filtered to {len(cache_files)} cache files matching '{args.video_filter}'")

    logger.info("="*80)
    logger.info("ACTION UNIT GROUND TRUTH UPSERT")
    logger.info("="*80)
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Cache files: {len(cache_files)}")
    logger.info(f"Force update: {args.force}")
    logger.info("="*80)

    # Initialize AU extractor
    logger.info("\n📦 Initializing AU extractor...")
    au_extractor = ActionUnitExtractor()
    logger.info("✅ AU extractor initialized")

    # Process each cache file
    logger.info(f"\n🔄 Processing {len(cache_files)} cache files...\n")

    success_count = 0
    for cache_file in tqdm(cache_files, desc="Updating caches"):
        try:
            if update_cache_with_au(cache_file, au_extractor, args.force):
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {cache_file.name}: {e}")
            continue

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total cache files: {len(cache_files)}")
    logger.info(f"Successfully updated: {success_count}")
    logger.info(f"Failed: {len(cache_files) - success_count}")
    logger.info("="*80)

    if success_count > 0:
        logger.info("\n✅ AU ground truth upsert complete!")
        logger.info("\nNext steps:")
        logger.info("  1. Restart training to use updated caches")
        logger.info("  2. Check WandB for AU visualizations")
        logger.info("  3. Run ./diagnose_au.sh to validate AU predictions")
    else:
        logger.warning("\n⚠️  No caches were updated. Check logs for errors.")


if __name__ == "__main__":
    main()
