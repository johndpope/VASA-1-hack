#!/usr/bin/env python3
"""
Complete cache diagnostic tool to identify:
1. Which videos are missing from frames/emo_frames caches
2. Why H5 file is bloated (36GB instead of ~57MB)
3. Cache structure issues
"""

import h5py
import hashlib
from pathlib import Path
import argparse
import logging
from collections import defaultdict
from typing import Dict, Set, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_video_md5(video_path: Path) -> str:
    """Compute MD5 hash of video file (matching frame_disk_cache.py)."""
    hash_md5 = hashlib.md5()
    try:
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096 * 1024), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing {video_path}: {e}")
        return None


def scan_video_folder(video_folder: Path) -> Dict[str, Path]:
    """Scan video folder and return mapping of MD5 -> video_path."""
    logger.info(f"\n🎬 Scanning videos in: {video_folder}")

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpg', '.mpeg'}
    videos = {}

    for video_path in video_folder.iterdir():
        if video_path.suffix.lower() in video_extensions:
            md5 = get_video_md5(video_path)
            if md5:
                videos[md5] = video_path

    logger.info(f"   Found {len(videos)} videos")
    return videos


def scan_frame_cache(cache_dir: Path, frame_type: str) -> Set[str]:
    """Scan frames or emo_frames directory and return set of MD5 hashes."""
    cache_path = cache_dir / frame_type
    if not cache_path.exists():
        logger.warning(f"⚠️  {frame_type} directory not found: {cache_path}")
        return set()

    logger.info(f"\n📁 Scanning {frame_type} cache: {cache_path}")

    # Get all MD5 directories
    md5_dirs = [d.name for d in cache_path.iterdir() if d.is_dir()]

    logger.info(f"   Found {len(md5_dirs)} video hashes")

    # Count total windows per video
    total_windows = 0
    for md5_dir in md5_dirs:
        windows = [d for d in (cache_path / md5_dir).iterdir() if d.is_dir() and d.name.startswith('window_')]
        total_windows += len(windows)

    logger.info(f"   Total windows: {total_windows}")

    return set(md5_dirs)


def analyze_h5_bloat(h5_path: Path) -> Dict:
    """Analyze why H5 file is bloated."""
    logger.info(f"\n💾 Analyzing H5 file: {h5_path}")

    if not h5_path.exists():
        logger.error(f"❌ H5 file not found: {h5_path}")
        return {}

    file_size_gb = h5_path.stat().st_size / (1024**3)
    logger.info(f"   File size: {file_size_gb:.2f} GB")

    analysis = {
        'file_size_gb': file_size_gb,
        'num_windows': 0,
        'fields_per_window': {},
        'bloat_culprits': []
    }

    try:
        with h5py.File(h5_path, 'r') as f:
            # Count windows
            try:
                window_keys = [k for k in f.keys() if k.startswith('window_')]
            except (RuntimeError, OSError) as e:
                logger.error(f"   ❌ H5 file is corrupted: {e}")
                analysis['corrupted'] = True
                return analysis

            analysis['num_windows'] = len(window_keys)
            logger.info(f"   Number of windows: {analysis['num_windows']}")

            if analysis['num_windows'] == 0:
                logger.warning("⚠️  H5 file is empty!")
                return analysis

            # Sample first window to see what's stored
            first_window_key = window_keys[0]
            first_window = f[first_window_key]

            logger.info(f"\n📊 Fields in window_0:")

            field_sizes = {}
            total_window_size = 0

            for field_name in first_window.keys():
                if isinstance(first_window[field_name], h5py.Dataset):
                    dataset = first_window[field_name]
                    shape = dataset.shape
                    dtype = dataset.dtype

                    # Calculate size
                    if hasattr(dataset, 'size'):
                        size_bytes = dataset.size * dataset.dtype.itemsize
                        field_sizes[field_name] = size_bytes
                        total_window_size += size_bytes

                        # Format size
                        if size_bytes > 1024**3:
                            size_str = f"{size_bytes / (1024**3):.2f} GB"
                        elif size_bytes > 1024**2:
                            size_str = f"{size_bytes / (1024**2):.2f} MB"
                        elif size_bytes > 1024:
                            size_str = f"{size_bytes / 1024:.2f} KB"
                        else:
                            size_str = f"{size_bytes} bytes"

                        logger.info(f"   {field_name:30s} {str(shape):30s} {size_str}")

            analysis['fields_per_window'] = field_sizes

            # Calculate per-window size
            window_size_mb = total_window_size / (1024**2)
            logger.info(f"\n   Per-window size: {window_size_mb:.2f} MB")

            # Estimate total based on windows
            estimated_total_gb = (window_size_mb * analysis['num_windows']) / 1024
            logger.info(f"   Estimated total ({analysis['num_windows']} windows): {estimated_total_gb:.2f} GB")

            # Identify bloat culprits (fields that should be on disk)
            bloat_fields = ['frames', 'emo_frames']
            logger.info(f"\n🔍 Checking for bloat culprits:")

            for field in bloat_fields:
                if field in field_sizes:
                    size_mb = field_sizes[field] / (1024**2)
                    total_size_gb = (size_mb * analysis['num_windows']) / 1024
                    logger.warning(f"   ⚠️  '{field}' found in H5 ({size_mb:.2f} MB/window, {total_size_gb:.2f} GB total)")
                    logger.warning(f"       → Should be in disk cache: cache_single_bucket/{field}/")
                    analysis['bloat_culprits'].append({
                        'field': field,
                        'size_per_window_mb': size_mb,
                        'total_size_gb': total_size_gb
                    })
                else:
                    logger.info(f"   ✅ '{field}' NOT in H5 (correctly using disk cache)")

            # Check for other large fields
            logger.info(f"\n📊 Largest fields in H5:")
            sorted_fields = sorted(field_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
            for field_name, size_bytes in sorted_fields:
                size_mb = size_bytes / (1024**2)
                total_gb = (size_mb * analysis['num_windows']) / 1024
                logger.info(f"   {field_name:30s} {size_mb:8.2f} MB/window  ({total_gb:.2f} GB total)")

    except Exception as e:
        logger.error(f"❌ Error analyzing H5: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return analysis


def find_missing_videos(video_folder: Path, cache_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """Find which videos are missing from frame caches."""
    logger.info(f"\n🔍 Finding missing videos...")

    # Get all videos and their MD5s
    videos = scan_video_folder(video_folder)
    video_md5s = set(videos.keys())

    # Get cached MD5s
    frames_md5s = scan_frame_cache(cache_dir, 'frames')
    emo_frames_md5s = scan_frame_cache(cache_dir, 'emo_frames')

    # Find missing
    missing_frames = video_md5s - frames_md5s
    missing_emo_frames = video_md5s - emo_frames_md5s
    cached_both = frames_md5s & emo_frames_md5s

    logger.info(f"\n📈 Summary:")
    logger.info(f"   Total videos: {len(video_md5s)}")
    logger.info(f"   Cached in frames/: {len(frames_md5s)}")
    logger.info(f"   Cached in emo_frames/: {len(emo_frames_md5s)}")
    logger.info(f"   Cached in both: {len(cached_both)}")
    logger.info(f"   Missing from frames/: {len(missing_frames)}")
    logger.info(f"   Missing from emo_frames/: {len(missing_emo_frames)}")

    # Show missing videos
    if missing_frames:
        logger.warning(f"\n⚠️  Videos missing from frames/ cache:")
        for md5 in sorted(list(missing_frames)[:10]):  # Show first 10
            video_path = videos[md5]
            logger.warning(f"   {video_path.name} (MD5: {md5[:8]}...)")
        if len(missing_frames) > 10:
            logger.warning(f"   ... and {len(missing_frames) - 10} more")

    if missing_emo_frames:
        logger.warning(f"\n⚠️  Videos missing from emo_frames/ cache:")
        for md5 in sorted(list(missing_emo_frames)[:10]):  # Show first 10
            video_path = videos[md5]
            logger.warning(f"   {video_path.name} (MD5: {md5[:8]}...)")
        if len(missing_emo_frames) > 10:
            logger.warning(f"   ... and {len(missing_emo_frames) - 10} more")

    return missing_frames, missing_emo_frames, video_md5s


def generate_reprocess_script(missing_videos: Dict[str, Path], output_file: Path):
    """Generate a script to reprocess missing videos."""
    if not missing_videos:
        logger.info("\n✅ No videos to reprocess!")
        return

    logger.info(f"\n📝 Generating reprocess script: {output_file}")

    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to reprocess missing videos\n")
        f.write("# Run with: bash reprocess_missing.sh\n\n")
        f.write("set -e  # Exit on error\n\n")

        for md5, video_path in sorted(missing_videos.items(), key=lambda x: x[1].name):
            f.write(f"# {video_path.name}\n")
            f.write(f"python preprocess_single_bucket.py \\\n")
            f.write(f"  --video-folder {video_path.parent} \\\n")
            f.write(f"  --cache-dir cache_single_bucket \\\n")
            f.write(f"  --cache-frames \\\n")
            f.write(f"  --cache-emo-frames \\\n")
            f.write(f"  --frame-format png\n\n")

    output_file.chmod(0o755)  # Make executable
    logger.info(f"   Written {len(missing_videos)} video reprocessing commands")
    logger.info(f"   Run with: bash {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Complete cache diagnostic tool')
    parser.add_argument('--video-folder', type=str, default='s1',
                       help='Path to video folder (default: s1)')
    parser.add_argument('--cache-dir', type=str, default='cache_single_bucket',
                       help='Path to cache directory (default: cache_single_bucket)')
    parser.add_argument('--h5-file', type=str, default='cache_single_bucket/all_windows_cache.h5',
                       help='Path to H5 cache file')
    parser.add_argument('--generate-reprocess-script', action='store_true',
                       help='Generate bash script to reprocess missing videos')

    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    cache_dir = Path(args.cache_dir)
    h5_path = Path(args.h5_file)

    logger.info("="*80)
    logger.info("CACHE DIAGNOSTIC TOOL")
    logger.info("="*80)

    # 1. Analyze H5 bloat
    h5_analysis = analyze_h5_bloat(h5_path)

    # 2. Find missing videos
    missing_frames, missing_emo, all_videos = find_missing_videos(video_folder, cache_dir)

    # 3. Generate recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)

    if h5_analysis.get('corrupted'):
        logger.error("\n❌ H5 FILE CORRUPTED!")
        logger.error("   The H5 file is corrupted and cannot be read.")
        logger.error(f"   File size: {h5_analysis['file_size_gb']:.2f} GB")
        logger.error("")
        logger.error("   💡 Solution:")
        logger.error("   1. Delete the corrupted H5 file:")
        logger.error("      rm cache_single_bucket/all_windows_cache.h5")
        logger.error("   2. Rerun preprocessing with --cache-frames --cache-emo-frames")
        logger.error("   3. This will rebuild the H5 cache correctly")
        logger.error("   4. Expected H5 size: ~0.4 MB/window (~57MB total)")
    elif h5_analysis.get('bloat_culprits'):
        logger.warning("\n⚠️  H5 FILE BLOAT DETECTED!")
        logger.warning("   The H5 file contains frame data that should be on disk.")
        logger.warning("")
        logger.warning("   💡 Solution:")
        logger.warning("   1. Rerun preprocessing with --cache-frames --cache-emo-frames")
        logger.warning("   2. This will save frames to disk instead of H5")
        logger.warning("   3. Expected H5 size: ~0.4 MB/window (NOT 200+ MB/window)")
        logger.warning("")
        total_bloat = sum(c['total_size_gb'] for c in h5_analysis['bloat_culprits'])
        logger.warning(f"   Potential space saved: {total_bloat:.2f} GB")

    if missing_frames or missing_emo:
        logger.warning(f"\n⚠️  MISSING VIDEOS DETECTED!")
        logger.warning(f"   {len(missing_frames)} videos missing from frames/")
        logger.warning(f"   {len(missing_emo)} videos missing from emo_frames/")
        logger.warning("")
        logger.warning("   💡 Solution:")
        logger.warning("   Run preprocessing on missing videos:")
        logger.warning("   python preprocess_single_bucket.py --video-folder s1 --cache-frames --cache-emo-frames")

        if args.generate_reprocess_script:
            # Get video paths for missing videos
            videos = scan_video_folder(video_folder)
            missing_video_paths = {md5: videos[md5] for md5 in (missing_frames | missing_emo) if md5 in videos}
            generate_reprocess_script(missing_video_paths, Path('reprocess_missing.sh'))

    if not h5_analysis.get('bloat_culprits') and not missing_frames and not missing_emo:
        logger.info("\n✅ Cache is healthy!")
        logger.info("   - H5 file does not contain frame bloat")
        logger.info("   - All videos are cached on disk")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
