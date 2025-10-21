#!/usr/bin/env python3
"""
Diagnostic tool for inspecting single-bucket cache contents
"""

import torch
import h5py
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def diagnose_cache(cache_path: Path, window_idx: int = 0, detailed: bool = False):
    """
    Diagnose what's in a single-bucket cache.

    Args:
        cache_path: Path to the all_windows_cache.h5 file
        window_idx: Which window to inspect in detail (default: 0)
        detailed: If True, show statistics for each field
    """

    if not cache_path.exists():
        logger.error(f"❌ Cache file not found: {cache_path}")
        return

    logger.info(f"🔍 Inspecting cache: {cache_path}")
    logger.info(f"   File size: {cache_path.stat().st_size / (1024**3):.2f} GB")
    logger.info("")

    try:
        with h5py.File(cache_path, 'r') as f:
            # Show metadata
            logger.info("📊 Cache Metadata:")
            if 'metadata' in f:
                for key in f['metadata'].attrs:
                    logger.info(f"   {key}: {f['metadata'].attrs[key]}")
            logger.info("")

            # Count windows
            num_windows = 0
            window_keys = [k for k in f.keys() if k.startswith('window_')]
            num_windows = len(window_keys)

            logger.info(f"📦 Total windows: {num_windows}")
            logger.info("")

            if num_windows == 0:
                logger.warning("⚠️ Cache is empty!")
                return

            # Inspect specific window
            window_key = f'window_{window_idx}'
            if window_key not in f:
                logger.error(f"❌ Window {window_idx} not found in cache")
                logger.info(f"   Available windows: 0 to {num_windows - 1}")
                return

            logger.info(f"🔬 Inspecting Window {window_idx}:")
            logger.info("=" * 80)

            window = f[window_key]

            # List all fields
            all_fields = []
            for key in window.keys():
                if isinstance(window[key], h5py.Dataset):
                    all_fields.append(key)

            # Categorize fields
            audio_fields = [k for k in all_fields if 'audio' in k.lower()]
            motion_fields = ['theta', 'scale', 'rotation', 'translation', 'expression_embed']
            landmark_fields = ['lips', 'right_eye', 'left_eye', 'jaw', 'nose', 'blink_state', 'lip_motion']
            control_fields = ['gaze', 'emotion', 'head_distance', 'speed_bucket']
            frame_fields = ['frames', 'emo_frames', 'emo_keyframe_indices']
            warp_fields = ['uv_warps', 'target_masks']
            other_fields = [k for k in all_fields if k not in audio_fields + motion_fields +
                           landmark_fields + control_fields + frame_fields + warp_fields]

            # Display by category
            def show_fields(category_name, field_list):
                if not field_list:
                    return
                logger.info(f"\n{category_name}:")
                for field in sorted(field_list):
                    if field in window:
                        data = window[field]
                        shape = data.shape
                        dtype = data.dtype

                        # Calculate size
                        if hasattr(data, 'size'):
                            size_bytes = data.size * data.dtype.itemsize
                            if size_bytes > 1024**3:
                                size_str = f"{size_bytes / (1024**3):.2f} GB"
                            elif size_bytes > 1024**2:
                                size_str = f"{size_bytes / (1024**2):.2f} MB"
                            elif size_bytes > 1024:
                                size_str = f"{size_bytes / 1024:.2f} KB"
                            else:
                                size_str = f"{size_bytes} bytes"
                        else:
                            size_str = "N/A"

                        logger.info(f"  ✓ {field:25s} {str(shape):30s} {str(dtype):15s} {size_str}")

                        # Show statistics if detailed
                        if detailed and dtype in [torch.float32, torch.float64, 'float32', 'float64']:
                            try:
                                arr = data[:]
                                logger.info(f"      Stats: min={arr.min():.4f}, max={arr.max():.4f}, "
                                          f"mean={arr.mean():.4f}, std={arr.std():.4f}")
                            except:
                                pass

            show_fields("🎵 Audio Fields", audio_fields)
            show_fields("🎭 Motion Parameters", motion_fields)
            show_fields("👁️ Facial Landmarks", landmark_fields)
            show_fields("🎮 Control Signals", control_fields)
            show_fields("🖼️ Frame Data", frame_fields)
            show_fields("🌊 Warp Fields", warp_fields)
            show_fields("📋 Other Fields", other_fields)

            logger.info("")
            logger.info("=" * 80)

            # Check for missing critical fields
            logger.info("\n🔍 Critical Field Check:")
            critical_fields = {
                'audio_waveform': 'Raw audio for Synchformer (needed for sync loss)',
                'audio_features': 'Wav2vec features (used in conditions)',
                'audio_mfcc': 'MFCC features (alternative audio representation)',
                'theta': 'Head pose matrix',
                'expression_embed': 'Expression embeddings',
                'uv_warps': 'UV warping fields (can be zeros if derived)',
            }

            for field, description in critical_fields.items():
                if field in all_fields:
                    logger.info(f"  ✅ {field:20s} - {description}")
                else:
                    logger.info(f"  ❌ {field:20s} - {description} [MISSING]")

            # Calculate total cache size breakdown
            logger.info("\n💾 Storage Breakdown:")
            total_size = 0
            category_sizes = {}

            for category_name, field_list in [
                ('Audio', audio_fields),
                ('Motion', motion_fields),
                ('Landmarks', landmark_fields),
                ('Control', control_fields),
                ('Frames', frame_fields),
                ('Warps', warp_fields),
                ('Other', other_fields),
            ]:
                cat_size = 0
                for field in field_list:
                    if field in window:
                        data = window[field]
                        if hasattr(data, 'size'):
                            field_size = data.size * data.dtype.itemsize
                            cat_size += field_size
                            total_size += field_size
                category_sizes[category_name] = cat_size

            for cat_name, cat_size in category_sizes.items():
                if total_size > 0:
                    pct = (cat_size / total_size) * 100
                    size_mb = cat_size / (1024**2)
                    logger.info(f"  {cat_name:12s}: {size_mb:8.2f} MB ({pct:5.1f}%)")

            total_mb = total_size / (1024**2)
            logger.info(f"  {'Total':12s}: {total_mb:8.2f} MB")
            logger.info(f"\nEstimated full cache size ({num_windows} windows): {total_mb * num_windows / 1024:.2f} GB")

    except Exception as e:
        logger.error(f"❌ Error reading cache: {e}")
        import traceback
        logger.error(traceback.format_exc())


def compare_windows(cache_path: Path, window1: int, window2: int):
    """Compare two windows to see differences"""
    logger.info(f"📊 Comparing Window {window1} vs Window {window2}")
    logger.info("=" * 80)

    with h5py.File(cache_path, 'r') as f:
        w1_key = f'window_{window1}'
        w2_key = f'window_{window2}'

        if w1_key not in f or w2_key not in f:
            logger.error("❌ One or both windows not found")
            return

        w1 = f[w1_key]
        w2 = f[w2_key]

        # Compare fields
        w1_fields = set(w1.keys())
        w2_fields = set(w2.keys())

        common = w1_fields & w2_fields
        only_w1 = w1_fields - w2_fields
        only_w2 = w2_fields - w1_fields

        if only_w1:
            logger.warning(f"⚠️ Only in Window {window1}: {only_w1}")
        if only_w2:
            logger.warning(f"⚠️ Only in Window {window2}: {only_w2}")

        logger.info(f"\n✅ Common fields: {len(common)}")
        logger.info("\nShape comparison:")
        for field in sorted(common):
            s1 = w1[field].shape
            s2 = w2[field].shape
            match = "✓" if s1 == s2 else "✗"
            logger.info(f"  {match} {field:25s} {str(s1):30s} vs {str(s2):30s}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose single-bucket cache contents')
    parser.add_argument('--cache-path', type=str,
                       default='cache_single_bucket/all_windows_cache.h5',
                       help='Path to cache file')
    parser.add_argument('--window', type=int, default=0,
                       help='Window index to inspect (default: 0)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed statistics for each field')
    parser.add_argument('--compare', type=int, nargs=2, metavar=('W1', 'W2'),
                       help='Compare two windows (e.g., --compare 0 1)')

    args = parser.parse_args()
    cache_path = Path(args.cache_path)

    if args.compare:
        compare_windows(cache_path, args.compare[0], args.compare[1])
    else:
        diagnose_cache(cache_path, args.window, args.detailed)


if __name__ == "__main__":
    main()
