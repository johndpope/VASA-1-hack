#!/usr/bin/env python3
"""
Preprocess all windows and save to single-bucket cache
"""

import torch
import numpy as np
from pathlib import Path
import logging
from vasa_dataset import VASAIntegratedDataset
from single_bucket_cache import SingleBucketCache
from frame_disk_cache import FrameDiskCache
import argparse
import gc
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_cache(dataset, cache_dir: Path, resume: bool = True,
                         cache_frames: bool = True, cache_emo_frames: bool = True,
                         frame_format: str = 'png'):
    """Preprocess all windows and save to single-bucket cache.

    Args:
        dataset: The VASA dataset
        cache_dir: Directory to save cache
        resume: If True, resume from existing partial cache
        cache_frames: If True, save original frames to disk
        cache_emo_frames: If True, save emo frames to disk
        frame_format: Image format for saved frames ('png' or 'jpg')
    """

    cache = SingleBucketCache(
        cache_dir=cache_dir,
        cache_name="all_windows_cache.h5",
        compression='gzip',
        compression_level=4
    )

    # Initialize frame disk caches if enabled
    frame_cache = None
    emo_frame_cache = None

    if cache_frames:
        frame_cache = FrameDiskCache(cache_dir, frame_type='frames')
        logger.info(f"✅ Frame disk cache enabled at {frame_cache.root}")

    if cache_emo_frames:
        emo_frame_cache = FrameDiskCache(cache_dir, frame_type='emo_frames')
        logger.info(f"✅ EMO frame disk cache enabled at {emo_frame_cache.root}")

    # Check if we can resume from existing cache
    existing_windows = []
    processed_indices = set()
    start_idx = 0

    if cache.has_cache() and resume:
        logger.info("🔄 Found existing cache, checking for resume capability...")
        info = cache.get_cache_info()
        logger.info(f"Cache info: {info}")

        # Load existing windows to check what's already processed
        try:
            # Try to load metadata to see how many windows are already cached
            existing_count = info.get('num_windows', 0)
            if existing_count > 0:
                logger.info(f"✅ Found {existing_count} already processed windows")
                # Load existing windows to get their indices
                for i in range(existing_count):
                    try:
                        window = cache.load_window(i)
                        if window and 'metadata' in window and 'window_index' in window['metadata']:
                            processed_indices.add(window['metadata']['window_index'])
                            existing_windows.append(window)
                    except:
                        break
                logger.info(f"📊 Successfully loaded {len(processed_indices)} existing windows")
                logger.info(f"🚀 Will skip already processed indices: {sorted(processed_indices)[:10]}...")
        except Exception as e:
            logger.warning(f"Could not load existing cache for resume: {e}")
            logger.info("Starting fresh...")
    elif cache.has_cache() and not resume:
        logger.info("Cache already exists. Delete it first if you want to rebuild from scratch.")
        info = cache.get_cache_info()
        logger.info(f"Cache info: {info}")
        return

    logger.info(f"Processing {len(dataset)} windows (skipping {len(processed_indices)} already done)...")

    all_windows = existing_windows.copy()  # Start with existing windows
    failed_indices = []
    skipped_count = 0
    quality_filtered_count = 0  # Track windows filtered for poor quality

    for idx in range(len(dataset)):
        # Skip if already processed
        if idx in processed_indices:
            skipped_count += 1
            if skipped_count % 10 == 0:
                logger.info(f"⏭️ Skipped {skipped_count} already processed windows...")
            continue

        if idx % 10 == 0:
            processed_new = len(all_windows) - len(existing_windows)
            logger.info(f"Processing window {idx}/{len(dataset)} (new: {processed_new}, skipped: {skipped_count})")

            # Clear CUDA cache and run garbage collection periodically
            if idx > 0 and idx % 50 == 0:
                logger.info(f"🧹 Cleaning memory at window {idx}...")
                torch.cuda.empty_cache()
                gc.collect()

                # Log memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        try:
            # Get window data from dataset
            window_data = dataset[idx]

            if window_data is not None:
                # QUALITY CHECK: Filter out bad windows before caching
                should_cache = True
                quality_reasons = []

                # Check 1: UV warp magnitude (prevent collapsed warps)
                if 'uv_warps' in window_data:
                    uv_magnitude = window_data['uv_warps'].abs().mean().item()
                    if uv_magnitude < 0.15:  # Threshold: warps too small
                        should_cache = False
                        quality_reasons.append(f"uv_magnitude={uv_magnitude:.4f}<0.15")

                # Check 2: Expression variance (prevent static predictions)
                if 'expression_embed' in window_data:
                    expr_std = window_data['expression_embed'].std().item()
                    if expr_std < 0.01:  # Near-zero variance
                        should_cache = False
                        quality_reasons.append(f"expr_std={expr_std:.6f}<0.01")

                # Check 3: Valid face landmarks (if available)
                if 'landmarks' in window_data:
                    # Count non-zero landmarks (zeros indicate failed detection)
                    valid_frames = (window_data['landmarks'].abs().sum(dim=-1).sum(dim=-1) > 0).float().mean().item()
                    if valid_frames < 0.5:  # Less than 50% frames have valid landmarks
                        should_cache = False
                        quality_reasons.append(f"valid_landmarks={valid_frames:.1%}<50%")

                if not should_cache:
                    quality_filtered_count += 1
                    if quality_filtered_count <= 10:  # Log first 10 filtered windows
                        logger.warning(f"❌ FILTERED window {idx}: {', '.join(quality_reasons)}")
                    elif quality_filtered_count % 50 == 0:  # Then every 50th
                        logger.info(f"⚠️ Filtered {quality_filtered_count} low-quality windows so far...")
                    continue  # Skip caching this window

                # Get video path and window index for frame caching
                video_path = None
                window_idx_in_video = None
                if 'metadata' in window_data:
                    video_path = window_data['metadata'].get('video_path')
                    window_idx_in_video = window_data['metadata'].get('window_idx', idx)

                # DEBUG: Log what keys are in window_data
                if idx == 0:
                    logger.info(f"🔑 Window {idx} keys: {list(window_data.keys())}")
                    logger.info(f"   'frames' in window_data: {'frames' in window_data}")
                    logger.info(f"   'emo_frames' in window_data: {'emo_frames' in window_data}")

                # Save frames to disk if enabled
                if frame_cache and video_path and 'frames' in window_data:
                    try:
                        frame_cache.save_frames(
                            video_path=video_path,
                            window_idx=window_idx_in_video,
                            frames=window_data['frames'],
                            format=frame_format
                        )
                        if idx % 50 == 0:
                            logger.debug(f"💾 Saved frames for window {idx} to disk")
                    except Exception as e:
                        logger.warning(f"Failed to save frames for window {idx}: {e}")

                # Save emo_frames to disk if enabled
                if emo_frame_cache and video_path and 'emo_frames' in window_data:
                    try:
                        # DEBUG: Check EMO frame values before saving
                        emo_frames = window_data['emo_frames']
                        if isinstance(emo_frames, torch.Tensor):
                            emo_min, emo_max = emo_frames.min().item(), emo_frames.max().item()
                            emo_mean = emo_frames.mean().item()

                            if idx % 10 == 0:  # Log every 10th window
                                logger.info(f"🎨 Window {idx} EMO frames: shape={emo_frames.shape}, range=[{emo_min:.3f}, {emo_max:.3f}], mean={emo_mean:.3f}")

                            if emo_max == 0:
                                logger.error(f"❌ Window {idx}: EMO frames are all zeros BEFORE saving!")
                            elif emo_max < 0.01:
                                logger.warning(f"⚠️ Window {idx}: EMO frames are very dark (max={emo_max:.6f})")

                        emo_frame_cache.save_frames(
                            video_path=video_path,
                            window_idx=window_idx_in_video,
                            frames=window_data['emo_frames'],
                            format=frame_format,
                            remove_green_background=True,  # Make green chroma key transparent
                            green_threshold=0.4  # Lower = more aggressive keying (remove more green)
                        )
                        if idx % 50 == 0:
                            logger.debug(f"💾 Saved emo_frames for window {idx} to disk")
                    except Exception as e:
                        logger.warning(f"Failed to save emo_frames for window {idx}: {e}")

                # Convert tensors to CPU and detach to avoid memory accumulation
                window_data_cpu = {}
                for key, value in window_data.items():
                    # Skip frames and emo_frames if they're being cached to disk
                    if cache_frames and key == 'frames':
                        continue
                    if cache_emo_frames and key == 'emo_frames':
                        continue

                    if isinstance(value, torch.Tensor):
                        # Move to CPU and detach from computation graph
                        window_data_cpu[key] = value.detach().cpu()
                    elif isinstance(value, dict):
                        # Handle nested dicts (like metadata)
                        window_data_cpu[key] = {}
                        for k, v in value.items():
                            if isinstance(v, torch.Tensor):
                                window_data_cpu[key][k] = v.detach().cpu()
                            else:
                                window_data_cpu[key][k] = v
                    else:
                        window_data_cpu[key] = value

                # Add index to metadata
                if 'metadata' not in window_data_cpu:
                    window_data_cpu['metadata'] = {}
                window_data_cpu['metadata']['window_index'] = idx

                all_windows.append(window_data_cpu)

                # Clear the original window_data to free memory
                del window_data

            else:
                failed_indices.append(idx)
                logger.warning(f"Window {idx} returned None")

        except Exception as e:
            logger.error(f"Error processing window {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_indices.append(idx)

        # Clear cache after each window to prevent accumulation
        if idx % 10 == 0:
            torch.cuda.empty_cache()

    # Save all windows to cache at once
    if all_windows:
        new_windows = len(all_windows) - len(existing_windows)

        if new_windows > 0:
            logger.info(f"💾 Saving {len(all_windows)} total windows to cache ({new_windows} new, {len(existing_windows)} existing)...")
        else:
            logger.info(f"✅ All windows already cached! Total: {len(all_windows)}")

        metadata = {
            'total_windows': len(dataset),
            'successful_windows': len(all_windows),
            'failed_windows': len(failed_indices),
            'skipped_windows': skipped_count,
            'quality_filtered_windows': quality_filtered_count,
            'dataset_config': {
                'window_size': dataset.window_size,
                'stride': dataset.stride,
                'sequence_length': dataset.sequence_length,
                'frame_size': dataset.frame_size,
            }
        }

        cache.save_all_windows(all_windows, metadata)

        logger.info(f"✅ Successfully cached {len(all_windows)} windows")
        if new_windows > 0:
            logger.info(f"   📈 Processed {new_windows} new windows")
        if skipped_count > 0:
            logger.info(f"   ⏭️ Skipped {skipped_count} already processed windows")
        if failed_indices:
            logger.warning(f"   ❌ Failed to process {len(failed_indices)} windows: {failed_indices[:10]}...")

        # Log frame cache statistics
        if frame_cache:
            frame_stats = frame_cache.get_cache_stats()
            logger.info(f"📊 Frame disk cache stats:")
            logger.info(f"   Videos: {frame_stats['total_videos']}")
            logger.info(f"   Windows: {frame_stats['total_windows']}")
            logger.info(f"   Frames: {frame_stats['total_frames']}")
            logger.info(f"   Size: {frame_stats['total_size_gb']:.2f} GB")

        if emo_frame_cache:
            emo_stats = emo_frame_cache.get_cache_stats()
            logger.info(f"📊 EMO frame disk cache stats:")
            logger.info(f"   Videos: {emo_stats['total_videos']}")
            logger.info(f"   Windows: {emo_stats['total_windows']}")
            logger.info(f"   Frames: {emo_stats['total_frames']}")
            logger.info(f"   Size: {emo_stats['total_size_gb']:.2f} GB")

        # Validate cache
        is_valid, issues = cache.validate_cache()
        if is_valid:
            logger.info("✅ Cache validation passed!")
        else:
            logger.error(f"❌ Cache validation failed: {issues}")

        # Auto-rebuild expression database if enabled
        # Read config from dataset (if available) or use default
        auto_rebuild = getattr(dataset, 'auto_rebuild_expression_db', False)
        frame_stride = getattr(dataset, 'expression_db_frame_stride', 5)

        if auto_rebuild and len(all_windows) > 0:
            logger.info("")
            logger.info("="*80)
            logger.info("🔨 Auto-rebuilding expression database...")
            logger.info("="*80)

            try:
                from build_expression_db import build_from_single_bucket_cache

                cache_file = cache_dir / "all_windows_cache.h5"
                db_output = cache_dir / "expression_embeddings.h5"

                build_from_single_bucket_cache(
                    cache_path=str(cache_file),
                    output_path=str(db_output),
                    frame_stride=frame_stride
                )

                logger.info("✅ Expression database rebuild complete!")
                logger.info(f"   Database location: {db_output}")

            except Exception as e:
                logger.error(f"❌ Failed to rebuild expression database: {e}")
                import traceback
                logger.error(traceback.format_exc())

    else:
        logger.error("❌ No windows were successfully processed!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess windows for single-bucket cache')
    parser.add_argument('--video_folder', type=str, default='junk',
                        help='Path to video folder')
    parser.add_argument('--cache_dir', type=str, default='cache_single_bucket',
                        help='Cache directory')
    parser.add_argument('--max_videos', type=int, default=100,
                        help='Maximum number of videos to process')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Window size')
    parser.add_argument('--stride', type=int, default=25,
                        help='Stride between windows')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh instead of resuming from existing cache')
    parser.add_argument('--cache-frames', action='store_true', default=True,
                        help='Cache original frames to disk (default: True)')
    parser.add_argument('--no-cache-frames', action='store_false', dest='cache_frames',
                        help='Disable caching original frames to disk')
    parser.add_argument('--cache-emo-frames', action='store_true', default=True,
                        help='Cache EMO frames to disk (default: True)')
    parser.add_argument('--no-cache-emo-frames', action='store_false', dest='cache_emo_frames',
                        help='Disable caching EMO frames to disk')
    parser.add_argument('--frame-format', type=str, default='png', choices=['png', 'jpg'],
                        help='Image format for cached frames (default: png)')
    args = parser.parse_args()

    # Load EMO model
    import importlib
    from omegaconf import OmegaConf

    logger.info("Loading volumetric avatar...")
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)

    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        logger.info("Volumetric avatar loaded successfully")
    else:
        logger.error(f"Model checkpoint not found: {model_path}")
        return

    emo_model = volumetric_avatar

    # Create dataset with single-bucket mode
    logger.info("Creating dataset...")

    # Import volumetric avatar bridge for EMO generation
    from vasa_va_bridge import VASAVolumetricAvatarBridge
    va_bridge = VASAVolumetricAvatarBridge(emo_model)

    dataset = VASAIntegratedDataset(
        video_folder=args.video_folder,
        emo_model=emo_model,
        window_size=args.window_size,
        stride=args.stride,
        max_videos=args.max_videos,
        cache_dir=args.cache_dir,
        use_single_bucket=True,  # Use SingleBucketCache for preprocessing
        generate_emo_frames=True,  # Enable EMO frame generation
        emo_identity_path="nemo/data/IMG_1.png",  # Identity image
        emo_keyframes_per_window=50,  # 50 keyframes per window
        va_bridge=va_bridge,  # Pass the bridge
        auto_rebuild_expression_db=True,  # Auto-rebuild expression DB
        expression_db_frame_stride=1  # Sample all frames (use 5 for every 5th)
    )

    # Preprocess and cache
    preprocess_and_cache(
        dataset,
        Path(args.cache_dir),
        resume=not args.no_resume,
        cache_frames=args.cache_frames,
        cache_emo_frames=args.cache_emo_frames,
        frame_format=args.frame_format
    )


if __name__ == "__main__":
    main()