#!/usr/bin/env python3
"""
Convert existing SingleBucketCache to PerVideoCache structure.

This migrates:
- cache_single_bucket/all_windows_cache.h5 (monolithic)
  → cache_per_video/<video_md5>/metadata.h5 (per-video)

- cache_single_bucket/frames/<video_md5>/... (already correct)
  → cache_per_video/<video_md5>/frames/... (moved)

- cache_single_bucket/emo_frames/<video_md5>/... (already correct)
  → cache_per_video/<video_md5>/emo_frames/... (moved)
"""

import h5py
import shutil
from pathlib import Path
import logging
import json
from collections import defaultdict
from single_bucket_cache import SingleBucketCache
from per_video_cache import PerVideoCache
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_single_to_per_video(
    single_cache_dir: Path,
    per_video_cache_dir: Path,
    copy_frames: bool = True
):
    """
    Migrate from SingleBucketCache to PerVideoCache.

    Args:
        single_cache_dir: Source cache directory (with all_windows_cache.h5)
        per_video_cache_dir: Destination cache directory (will create MD5 folders)
        copy_frames: If True, copy frame files; if False, create symlinks
    """
    logger.info("="*80)
    logger.info("MIGRATING SINGLE BUCKET CACHE → PER-VIDEO CACHE")
    logger.info("="*80)
    logger.info(f"Source: {single_cache_dir}")
    logger.info(f"Destination: {per_video_cache_dir}")
    logger.info(f"Copy frames: {copy_frames}")

    # Load source cache
    single_cache = SingleBucketCache(single_cache_dir, cache_name='all_windows_cache.h5')

    if not single_cache.has_cache():
        logger.error(f"❌ No source cache found at {single_cache_dir}/all_windows_cache.h5")
        return

    # Create destination cache
    per_video_cache = PerVideoCache(per_video_cache_dir)

    # Get cache info
    cache_info = single_cache.get_cache_info()
    total_windows = cache_info.get('num_windows', 0)
    logger.info(f"\n📊 Source cache has {total_windows} windows")

    # Group windows by video
    logger.info("\n📋 Grouping windows by video...")
    video_windows = defaultdict(list)

    for window_idx in range(total_windows):
        if window_idx % 50 == 0:
            logger.info(f"  Loading window {window_idx}/{total_windows}...")

        window_data = single_cache.load_window(window_idx)
        if window_data is None:
            logger.warning(f"  ⚠️  Window {window_idx} is None, skipping")
            continue

        # Get video path
        video_path = window_data.get('metadata', {}).get('video_path')
        if not video_path:
            logger.warning(f"  ⚠️  Window {window_idx} has no video_path, skipping")
            continue

        # Get video MD5
        video_md5 = per_video_cache.get_video_hash(video_path)

        # Store window
        video_windows[video_md5].append({
            'video_path': video_path,
            'window_idx_in_video': window_data.get('metadata', {}).get('window_idx', window_idx),
            'window_data': window_data
        })

    logger.info(f"✅ Grouped {total_windows} windows into {len(video_windows)} videos")

    # Migrate each video
    migrated_videos = 0
    migrated_windows = 0
    failed_videos = []

    for video_md5, windows_list in video_windows.items():
        video_path = windows_list[0]['video_path']
        video_name = Path(video_path).name

        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"📹 Migrating video {migrated_videos + 1}/{len(video_windows)}: {video_name}")
            logger.info(f"   MD5: {video_md5}")
            logger.info(f"   Windows: {len(windows_list)}")
            logger.info(f"{'='*80}")

            # Create video directory
            video_dir = per_video_cache_dir / video_md5
            video_dir.mkdir(parents=True, exist_ok=True)

            # Prepare window data for this video's metadata.h5
            video_window_data = []

            for window_info in windows_list:
                window_data = window_info['window_data']
                window_idx_in_video = window_info['window_idx_in_video']

                # Exclude frames from metadata (they'll be on disk)
                metadata_only = {}
                for key, value in window_data.items():
                    if key in ['frames', 'emo_frames']:
                        continue  # Skip - will be on disk
                    metadata_only[key] = value

                video_window_data.append(metadata_only)

            # Save metadata.h5 for this video
            logger.info(f"  💾 Saving metadata.h5 ({len(video_window_data)} windows)...")
            per_video_cache.save_video_windows(video_path, video_window_data)

            # Migrate frames and emo_frames
            source_frames_dir = single_cache_dir / 'frames' / video_md5
            source_emo_frames_dir = single_cache_dir / 'emo_frames' / video_md5
            dest_frames_dir = video_dir / 'frames'
            dest_emo_frames_dir = video_dir / 'emo_frames'

            # Migrate frames
            if source_frames_dir.exists():
                if copy_frames:
                    logger.info(f"  📂 Copying frames...")
                    if dest_frames_dir.exists():
                        shutil.rmtree(dest_frames_dir)
                    shutil.copytree(source_frames_dir, dest_frames_dir)
                else:
                    logger.info(f"  🔗 Symlinking frames...")
                    if dest_frames_dir.exists():
                        dest_frames_dir.unlink()
                    dest_frames_dir.symlink_to(source_frames_dir.absolute())
            else:
                logger.warning(f"  ⚠️  No frames found for {video_name}")

            # Migrate emo_frames
            if source_emo_frames_dir.exists():
                if copy_frames:
                    logger.info(f"  📂 Copying emo_frames...")
                    if dest_emo_frames_dir.exists():
                        shutil.rmtree(dest_emo_frames_dir)
                    shutil.copytree(source_emo_frames_dir, dest_emo_frames_dir)
                else:
                    logger.info(f"  🔗 Symlinking emo_frames...")
                    if dest_emo_frames_dir.exists():
                        dest_emo_frames_dir.unlink()
                    dest_emo_frames_dir.symlink_to(source_emo_frames_dir.absolute())
            else:
                logger.warning(f"  ⚠️  No emo_frames found for {video_name}")

            migrated_videos += 1
            migrated_windows += len(windows_list)

            logger.info(f"  ✅ Migrated {video_name}")
            logger.info(f"     Progress: {migrated_videos}/{len(video_windows)} videos")

        except Exception as e:
            logger.error(f"  ❌ Failed to migrate {video_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_videos.append((video_md5, video_name, str(e)))

    # Build index
    logger.info("\n" + "="*80)
    logger.info("📊 Building cache index...")
    logger.info("="*80)
    index = per_video_cache.rebuild_index()

    # Copy other files
    logger.info("\n📂 Copying auxiliary files...")
    for aux_file in ['video_hashes.txt', 'expression_embeddings.h5']:
        source_file = single_cache_dir / aux_file
        if source_file.exists():
            dest_file = per_video_cache_dir / aux_file
            shutil.copy2(source_file, dest_file)
            logger.info(f"  ✅ Copied {aux_file}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ MIGRATION COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Successfully migrated: {migrated_videos}/{len(video_windows)} videos")
    logger.info(f"✅ Total windows migrated: {migrated_windows}")

    if failed_videos:
        logger.warning(f"\n❌ Failed videos: {len(failed_videos)}")
        for video_md5, video_name, error in failed_videos[:10]:
            logger.warning(f"   {video_name}: {error}")

    # Show stats
    stats = per_video_cache.get_cache_stats()
    logger.info(f"\n📊 New Cache Statistics:")
    logger.info(f"   Videos: {stats['total_videos']}")
    logger.info(f"   Total windows: {stats['total_windows']}")
    logger.info(f"   Total metadata size: {stats['total_metadata_size_mb']:.2f} MB")
    logger.info(f"   Average per video: {stats['total_metadata_size_mb'] / max(stats['total_videos'], 1):.2f} MB")

    # Show sample structure
    if stats['total_videos'] > 0:
        sample_md5 = list(index.keys())[0]
        sample_path = per_video_cache_dir / sample_md5
        logger.info(f"\n📁 New cache structure:")
        logger.info(f"   {per_video_cache_dir}/")
        logger.info(f"   ├── {sample_md5}/")
        logger.info(f"   │   ├── metadata.h5")
        logger.info(f"   │   ├── frames/")
        logger.info(f"   │   │   └── window_X/frame_XXX.png")
        logger.info(f"   │   └── emo_frames/")
        logger.info(f"   │       └── window_X/frame_XXX.png")
        logger.info(f"   ├── cache_index.json")
        logger.info(f"   ├── video_hashes.txt")
        logger.info(f"   └── expression_embeddings.h5")


def main():
    parser = argparse.ArgumentParser(description='Migrate SingleBucketCache to PerVideoCache')
    parser.add_argument('--source', type=str, default='cache_single_bucket',
                       help='Source cache directory (default: cache_single_bucket)')
    parser.add_argument('--dest', type=str, default='cache_per_video',
                       help='Destination cache directory (default: cache_per_video)')
    parser.add_argument('--symlink', action='store_true',
                       help='Use symlinks instead of copying frames (faster, but fragile)')
    args = parser.parse_args()

    migrate_single_to_per_video(
        single_cache_dir=Path(args.source),
        per_video_cache_dir=Path(args.dest),
        copy_frames=not args.symlink
    )


if __name__ == "__main__":
    main()
