#!/usr/bin/env python3
"""
Validate and clean H5 cache by removing bad quality windows.

Usage:
    python validate_cache.py --cache-dir cache_single_bucket --validate-only
    python validate_cache.py --cache-dir cache_single_bucket --clean
    python validate_cache.py --cache-dir cache_single_bucket --invalidate-video "path/to/video.mp4"
"""

import argparse
from pathlib import Path
import logging
from single_bucket_cache import SingleBucketCache
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_cache(cache_dir: str):
    """Validate cache quality and generate report."""
    cache = SingleBucketCache(
        cache_dir=Path(cache_dir),
        cache_name="all_windows_cache.h5"
    )

    if not cache.has_cache():
        logger.error(f"No cache found at {cache_dir}/all_windows_cache.h5")
        return

    logger.info("Running quality validation on all windows...")
    report = cache.validate_all_quality()

    # Save report to JSON (convert numpy types to native Python types)
    report_path = Path(cache_dir) / "quality_report.json"

    # Convert numpy int64 to int for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

    serializable_report = convert_numpy_types(report)

    with open(report_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)

    logger.info(f"📊 Quality Report saved to {report_path}")
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Total windows: {report['total_windows']}")
    logger.info(f"  Bad windows: {report['bad_windows']}")
    logger.info(f"  Bad videos: {report['bad_video_count']}")
    logger.info(f"  Quality pass rate: {report['quality_pass_rate']:.1%}")
    logger.info(f"{'='*60}\n")

    if report['bad_videos']:
        logger.warning(f"\n⚠️ Found {len(report['bad_videos'])} videos with quality issues:")
        for video_path in report['bad_videos']:
            logger.warning(f"  - {video_path}")

    return report


def clean_cache(cache_dir: str, report: dict = None):
    """Remove all bad quality windows from cache."""
    cache = SingleBucketCache(
        cache_dir=Path(cache_dir),
        cache_name="all_windows_cache.h5"
    )

    if not cache.has_cache():
        logger.error(f"No cache found at {cache_dir}/all_windows_cache.h5")
        return

    # Validate first if no report provided
    if report is None:
        logger.info("No report provided, running validation first...")
        report = cache.validate_all_quality()

    if not report['bad_videos']:
        logger.info("✅ No bad videos found! Cache is clean.")
        return

    logger.warning(f"\n⚠️ Will remove windows from {len(report['bad_videos'])} bad videos:")
    for video_path in report['bad_videos']:
        logger.warning(f"  - {video_path}")

    response = input(f"\nRemove {report['bad_windows']} windows from {len(report['bad_videos'])} videos? [y/N]: ")
    if response.lower() != 'y':
        logger.info("Aborted.")
        return

    # Invalidate each bad video
    total_removed = 0
    for video_path in report['bad_videos']:
        removed = cache.invalidate_video(video_path)
        total_removed += removed

    logger.info(f"\n✅ Cleaned cache! Removed {total_removed} windows from {len(report['bad_videos'])} videos")

    # Validate again to confirm
    logger.info("\nValidating cleaned cache...")
    new_report = cache.validate_all_quality()
    logger.info(f"New quality pass rate: {new_report['quality_pass_rate']:.1%}")


def invalidate_video(cache_dir: str, video_path: str):
    """Invalidate a specific video from cache."""
    cache = SingleBucketCache(
        cache_dir=Path(cache_dir),
        cache_name="all_windows_cache.h5"
    )

    if not cache.has_cache():
        logger.error(f"No cache found at {cache_dir}/all_windows_cache.h5")
        return

    logger.info(f"Invalidating video: {video_path}")
    removed = cache.invalidate_video(video_path)

    if removed > 0:
        logger.info(f"✅ Removed {removed} windows from {video_path}")
    else:
        logger.warning(f"No windows found for {video_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate and clean H5 cache')
    parser.add_argument('--cache-dir', type=str, default='cache_single_bucket',
                        help='Cache directory')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate, do not clean')
    parser.add_argument('--clean', action='store_true',
                        help='Clean bad windows from cache')
    parser.add_argument('--invalidate-video', type=str,
                        help='Invalidate specific video path')

    args = parser.parse_args()

    if args.invalidate_video:
        invalidate_video(args.cache_dir, args.invalidate_video)
    elif args.clean:
        report = validate_cache(args.cache_dir)
        clean_cache(args.cache_dir, report)
    else:
        # Default: validate only
        validate_cache(args.cache_dir)


if __name__ == "__main__":
    main()
