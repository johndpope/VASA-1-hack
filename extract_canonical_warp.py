#!/usr/bin/env python3
"""
Extract a canonical UV warp from the cache to use as baseline initialization.

This script:
1. Loads windows from the single-bucket cache
2. Finds a high-quality UV warp (good magnitude, no collapse)
3. Saves it as canonical_warp.pt for model initialization

Usage:
    python extract_canonical_warp.py --cache_dir cache_single_bucket --output canonical_warp.pt
"""

import torch
import argparse
from pathlib import Path
from single_bucket_cache import SingleBucketCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_canonical_warp(cache_dir: Path, output_path: Path, target_magnitude: float = 0.65):
    """
    Extract a canonical UV warp from cache.

    Args:
        cache_dir: Directory containing single-bucket cache
        output_path: Where to save canonical_warp.pt
        target_magnitude: Target UV warp magnitude to look for
    """
    cache = SingleBucketCache(
        cache_dir=cache_dir,
        cache_name="all_windows_cache.h5",
        compression='gzip',
        compression_level=4
    )

    if not cache.has_cache():
        logger.error(f"❌ No cache found at {cache_dir}")
        return False

    info = cache.get_cache_info()
    num_windows = info.get('num_windows', 0)
    logger.info(f"📊 Found {num_windows} windows in cache")

    # Find best canonical warp
    best_warp = None
    best_magnitude = 0
    best_idx = -1
    best_distance = float('inf')

    for idx in range(num_windows):
        try:
            window = cache.load_window(idx)
            if window is None or 'uv_warps' not in window:
                continue

            uv_warps = window['uv_warps']

            # Get first frame's warp (canonical representation)
            if uv_warps.dim() == 5:  # (1, 16, 64, 64, 3)
                warp = uv_warps[0]
            elif uv_warps.dim() == 4:  # (16, 64, 64, 3)
                warp = uv_warps
            else:
                logger.warning(f"Unexpected warp shape at idx {idx}: {uv_warps.shape}")
                continue

            # Calculate quality metrics
            magnitude = warp.abs().mean().item()
            std = warp.std().item()

            # Quality checks
            if magnitude < 0.15:  # Too small (collapsed)
                continue
            if std < 0.01:  # No variance (bad)
                continue
            if magnitude > 2.0:  # Too large (outlier)
                continue

            # Find warp closest to target magnitude
            distance = abs(magnitude - target_magnitude)

            if distance < best_distance:
                best_distance = distance
                best_warp = warp.clone()
                best_magnitude = magnitude
                best_idx = idx
                logger.info(f"📍 New best candidate at idx {idx}: magnitude={magnitude:.4f}, std={std:.4f}")

        except Exception as e:
            logger.warning(f"Error loading window {idx}: {e}")
            continue

    if best_warp is None:
        logger.error(f"❌ No valid canonical warp found in cache!")
        return False

    # Save canonical warp
    torch.save(best_warp, output_path)

    logger.info(f"")
    logger.info(f"✅ Extracted canonical warp from window {best_idx}")
    logger.info(f"   Magnitude: {best_magnitude:.4f} (target: {target_magnitude:.4f})")
    logger.info(f"   Std: {best_warp.std().item():.4f}")
    logger.info(f"   Shape: {best_warp.shape}")
    logger.info(f"   Range: [{best_warp.min().item():.4f}, {best_warp.max().item():.4f}]")
    logger.info(f"   Saved to: {output_path}")
    logger.info(f"")
    logger.info(f"📝 To use this canonical warp, add to vasa_config.yaml:")
    logger.info(f"   model:")
    logger.info(f"     canonical_warp_path: \"{output_path}\"")

    return True


def main():
    parser = argparse.ArgumentParser(description='Extract canonical UV warp from cache')
    parser.add_argument('--cache_dir', type=str, default='cache_single_bucket',
                        help='Cache directory')
    parser.add_argument('--output', type=str, default='canonical_warp.pt',
                        help='Output file path')
    parser.add_argument('--target_magnitude', type=float, default=0.65,
                        help='Target UV warp magnitude to look for')
    args = parser.parse_args()

    success = extract_canonical_warp(
        cache_dir=Path(args.cache_dir),
        output_path=Path(args.output),
        target_magnitude=args.target_magnitude
    )

    if success:
        logger.info("✅ Done!")
    else:
        logger.error("❌ Failed to extract canonical warp")
        exit(1)


if __name__ == "__main__":
    main()
