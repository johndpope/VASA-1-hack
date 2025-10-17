#!/usr/bin/env python3
"""
Clean up bad videos and their cached data.

This script:
1. Computes MD5 hashes of bad videos
2. Removes their cached data (H5, frames, emo_frames)
3. Deletes the video files
4. Updates the video tracking database
"""

import hashlib
from pathlib import Path
import logging
import h5py
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of bad videos to remove
BAD_VIDEOS = [
    "junk2/11.mp4",
    "junk2/videovideo-hOQAsNg8aw-scene3_scene3.mp4",
    "junk2/videovideoeI2V8Bd5X9s-scene6_scene1.mp4",
    "junk2/videovideozqKyByqbSs8-scene18_scene2.mp4",
    "junk2/videovideo_0MTaef81jQQ-scene23_scene4.mp4",
    "junk2/videovideokUUhJtbrIwU-scene241_scene6.mp4",
    "junk2/videovideoIfai_9UFDSU-scene1_scene7.mp4",
    "junk2/videovideo_dvrizT3014o-scene1_scene20.mp4",
    "junk2/videovideoAl5MFMZvpY4-scene2_scene1.mp4",
    "junk2/videovideoTwqES24jgRA-scene32_scene10.mp4",
]


def compute_md5(video_path: Path) -> str:
    """Compute MD5 hash of video file."""
    md5_hash = hashlib.md5()
    with open(video_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def remove_from_h5_cache(video_path: str, cache_dir: Path):
    """Remove windows for a specific video from H5 cache."""
    h5_cache = cache_dir / "all_windows_cache.h5"

    if not h5_cache.exists():
        logger.warning(f"H5 cache not found: {h5_cache}")
        return 0

    removed_count = 0
    windows_to_keep = []

    with h5py.File(h5_cache, 'r') as f:
        num_windows = f.attrs.get('num_windows', 0)
        logger.info(f"Checking {num_windows} windows in H5 cache...")

        for i in range(num_windows):
            window_key = f'window_{i}'
            if window_key not in f:
                continue

            window = f[window_key]

            # Check if this window belongs to the bad video
            if 'metadata' in window:
                meta = window['metadata']
                if 'video_path' in meta.attrs:
                    window_video_path = meta.attrs['video_path']
                    if isinstance(window_video_path, bytes):
                        window_video_path = window_video_path.decode('utf-8')

                    if window_video_path == video_path:
                        removed_count += 1
                        logger.debug(f"Marking window {i} for removal (video: {video_path})")
                    else:
                        windows_to_keep.append(i)
                else:
                    windows_to_keep.append(i)
            else:
                windows_to_keep.append(i)

    if removed_count > 0:
        logger.info(f"Would remove {removed_count} windows from {video_path}")
        logger.info(f"Would keep {len(windows_to_keep)} windows")

    return removed_count


def remove_disk_cache(video_path: str, md5_hash: str, cache_dir: Path):
    """Remove frames and emo_frames from disk cache."""
    removed_items = []

    # Remove frames cache
    frames_dir = cache_dir / "frames" / md5_hash
    if frames_dir.exists():
        logger.info(f"Removing frames cache: {frames_dir}")
        shutil.rmtree(frames_dir)
        removed_items.append(f"frames/{md5_hash}")

    # Remove emo_frames cache
    emo_frames_dir = cache_dir / "emo_frames" / md5_hash
    if emo_frames_dir.exists():
        logger.info(f"Removing emo_frames cache: {emo_frames_dir}")
        shutil.rmtree(emo_frames_dir)
        removed_items.append(f"emo_frames/{md5_hash}")

    return removed_items


def main():
    logger.info("="*80)
    logger.info("BAD VIDEO CLEANUP")
    logger.info("="*80)

    cache_dir = Path("cache_single_bucket")

    # Summary tracking
    total_videos = len(BAD_VIDEOS)
    existing_videos = 0
    missing_videos = 0
    total_h5_windows_removed = 0
    total_disk_cache_removed = 0

    video_md5_map = {}

    logger.info(f"\nProcessing {total_videos} bad videos...\n")

    for video_path_str in BAD_VIDEOS:
        video_path = Path(video_path_str)

        logger.info(f"Processing: {video_path}")

        if not video_path.exists():
            logger.warning(f"  ⚠️  Video not found, skipping: {video_path}")
            missing_videos += 1
            continue

        existing_videos += 1

        # Step 1: Compute MD5
        logger.info(f"  Computing MD5...")
        md5_hash = compute_md5(video_path)
        video_md5_map[str(video_path)] = md5_hash
        logger.info(f"  MD5: {md5_hash}")

        # Step 2: Remove from H5 cache
        logger.info(f"  Checking H5 cache...")
        h5_removed = remove_from_h5_cache(str(video_path), cache_dir)
        total_h5_windows_removed += h5_removed
        if h5_removed > 0:
            logger.info(f"  ✅ Would remove {h5_removed} windows from H5 cache")

        # Step 3: Remove disk cache
        logger.info(f"  Checking disk cache...")
        disk_removed = remove_disk_cache(str(video_path), md5_hash, cache_dir)
        total_disk_cache_removed += len(disk_removed)
        if disk_removed:
            logger.info(f"  ✅ Removed: {', '.join(disk_removed)}")
        else:
            logger.info(f"  ℹ️  No disk cache found")

        logger.info("")

    # Print summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"  Existing: {existing_videos}")
    logger.info(f"  Missing: {missing_videos}")
    logger.info(f"Total H5 windows to remove: {total_h5_windows_removed}")
    logger.info(f"Total disk cache directories removed: {total_disk_cache_removed}")

    # Print MD5 mapping
    logger.info("\nMD5 Hash Mapping:")
    logger.info("-" * 80)
    for video, md5 in video_md5_map.items():
        logger.info(f"{video}")
        logger.info(f"  -> {md5}")

    # Ask for confirmation before deleting videos
    logger.info("\n" + "="*80)
    logger.info("VIDEO DELETION")
    logger.info("="*80)
    logger.info("The following videos will be DELETED:")
    for video_path_str in BAD_VIDEOS:
        video_path = Path(video_path_str)
        if video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ❌ {video_path} ({size_mb:.2f} MB)")

    response = input("\n⚠️  Delete these videos? (yes/no): ")

    if response.lower() == 'yes':
        deleted_count = 0
        deleted_size = 0

        for video_path_str in BAD_VIDEOS:
            video_path = Path(video_path_str)
            if video_path.exists():
                size = video_path.stat().st_size
                video_path.unlink()
                deleted_count += 1
                deleted_size += size
                logger.info(f"  ✅ Deleted: {video_path}")

        logger.info(f"\n✅ Deleted {deleted_count} videos ({deleted_size / (1024*1024):.2f} MB)")
    else:
        logger.info("❌ Video deletion cancelled")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Rebuild H5 cache to remove bad windows:")
    logger.info("   rm cache_single_bucket/all_windows_cache.h5")
    logger.info("   python preprocess_single_bucket.py --cache-frames --cache-emo-frames")
    logger.info("")
    logger.info("2. Or use SingleBucketCache.invalidate_video() to remove specific videos")
    logger.info("")


if __name__ == "__main__":
    main()
