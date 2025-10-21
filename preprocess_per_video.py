#!/usr/bin/env python3
"""
Preprocess videos using per-video cache structure (MD5 folders with separate H5 files)
"""

import torch
import numpy as np
from pathlib import Path
import logging
from vasa_dataset import VASAIntegratedDataset
from per_video_cache import PerVideoCache
import argparse
import gc
import traceback
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_per_video_cache(
    dataset,
    cache_dir: Path,
    frame_format: str = 'png',
    save_batch_size: int = 10
):
    """
    Preprocess videos using per-video cache structure.

    Each video gets its own MD5 folder with:
    - metadata.h5 (all windows' metadata for this video)
    - frames/window_X/*.png
    - emo_frames/window_X/*.png

    Args:
        dataset: The VASA dataset
        cache_dir: Directory to save cache (will create MD5 subfolders)
        frame_format: Image format for saved frames ('png' or 'jpg')
        save_batch_size: Save metadata every N windows per video
    """
    cache = PerVideoCache(
        cache_dir=cache_dir,
        compression='gzip',
        compression_level=4
    )

    logger.info("="*80)
    logger.info("PER-VIDEO CACHE PREPROCESSING")
    logger.info("="*80)
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Frame format: {frame_format}")
    logger.info(f"Total windows to process: {len(dataset)}")

    # Group windows by video
    video_windows = defaultdict(list)
    video_paths = {}

    logger.info("\n📋 Grouping windows by video...")
    for idx in range(len(dataset)):
        try:
            # Get window metadata without loading full data
            window_data = dataset[idx]

            if window_data is None:
                continue

            # Get video path from metadata
            if 'metadata' not in window_data:
                logger.warning(f"Window {idx} has no metadata, skipping")
                continue

            video_path = window_data['metadata'].get('video_path')
            if not video_path:
                logger.warning(f"Window {idx} has no video_path, skipping")
                continue

            # Get video MD5 hash
            video_md5 = cache.get_video_hash(video_path)

            # Store window index for this video
            video_windows[video_md5].append((idx, window_data))
            video_paths[video_md5] = video_path

            # Log progress
            if (idx + 1) % 50 == 0:
                logger.info(f"  Grouped {idx + 1}/{len(dataset)} windows...")

        except Exception as e:
            logger.error(f"Error grouping window {idx}: {e}")
            continue

    logger.info(f"\n✅ Grouped {len(dataset)} windows into {len(video_windows)} videos")

    # Process each video
    processed_videos = 0
    total_windows_saved = 0
    failed_videos = []

    for video_md5, windows_list in video_windows.items():
        video_path = video_paths[video_md5]
        video_name = Path(video_path).name

        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"📹 Processing video {processed_videos + 1}/{len(video_windows)}: {video_name}")
            logger.info(f"   MD5: {video_md5}")
            logger.info(f"   Windows: {len(windows_list)}")
            logger.info(f"{'='*80}")

            # Check if already cached
            if cache.has_video_cache(video_path):
                logger.info(f"⏭️  Video already cached, skipping...")
                processed_videos += 1
                total_windows_saved += len(windows_list)
                continue

            # Prepare window data for this video
            video_window_data = []

            for window_global_idx, window_data in windows_list:
                # Get window index within this video
                window_idx_in_video = window_data['metadata'].get('window_idx', window_global_idx)

                # Save frames to disk
                if 'frames' in window_data:
                    try:
                        cache.save_frames(
                            video_path=video_path,
                            window_idx=window_idx_in_video,
                            frames=window_data['frames'],
                            frame_type='frames',
                            format=frame_format
                        )
                        logger.debug(f"  💾 Saved frames for window {window_idx_in_video}")
                    except Exception as e:
                        logger.warning(f"Failed to save frames for window {window_idx_in_video}: {e}")

                # Save emo_frames to disk
                if 'emo_frames' in window_data:
                    try:
                        cache.save_frames(
                            video_path=video_path,
                            window_idx=window_idx_in_video,
                            frames=window_data['emo_frames'],
                            frame_type='emo_frames',
                            format=frame_format,
                            remove_green_background=True
                        )
                        logger.debug(f"  💾 Saved emo_frames for window {window_idx_in_video}")
                    except Exception as e:
                        logger.warning(f"Failed to save emo_frames for window {window_idx_in_video}: {e}")

                # Prepare metadata (exclude frames - they're on disk)
                window_metadata = {}
                for key, value in window_data.items():
                    if key in ['frames', 'emo_frames']:
                        continue  # Skip - already saved to disk

                    # Convert tensors to CPU
                    if isinstance(value, torch.Tensor):
                        window_metadata[key] = value.detach().cpu()
                    elif isinstance(value, dict):
                        # Handle nested dicts
                        window_metadata[key] = {}
                        for k, v in value.items():
                            if isinstance(v, torch.Tensor):
                                window_metadata[key][k] = v.detach().cpu()
                            else:
                                window_metadata[key][k] = v
                    else:
                        window_metadata[key] = value

                # Store window index for reconstruction
                if 'metadata' not in window_metadata:
                    window_metadata['metadata'] = {}
                window_metadata['metadata']['global_window_index'] = window_global_idx
                window_metadata['metadata']['window_idx'] = window_idx_in_video

                video_window_data.append(window_metadata)

                # Log progress
                if (len(video_window_data)) % 10 == 0:
                    logger.info(f"  Processed {len(video_window_data)}/{len(windows_list)} windows...")

            # Save all windows' metadata to this video's H5 file
            logger.info(f"💾 Saving {len(video_window_data)} windows to {video_name}/metadata.h5...")
            cache.save_video_windows(video_path, video_window_data)

            processed_videos += 1
            total_windows_saved += len(video_window_data)

            logger.info(f"✅ Completed video {video_name}")
            logger.info(f"   Progress: {processed_videos}/{len(video_windows)} videos")
            logger.info(f"   Total windows saved: {total_windows_saved}")

            # Clean up memory
            del video_window_data
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Error processing video {video_name}: {e}")
            logger.error(traceback.format_exc())
            failed_videos.append((video_md5, video_name, str(e)))

    # Build index
    logger.info("\n" + "="*80)
    logger.info("📊 Building cache index...")
    logger.info("="*80)
    index = cache.rebuild_index()

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Successfully processed: {processed_videos}/{len(video_windows)} videos")
    logger.info(f"✅ Total windows saved: {total_windows_saved}")

    if failed_videos:
        logger.warning(f"\n❌ Failed videos: {len(failed_videos)}")
        for video_md5, video_name, error in failed_videos[:10]:
            logger.warning(f"   {video_name}: {error}")
        if len(failed_videos) > 10:
            logger.warning(f"   ... and {len(failed_videos) - 10} more")

    # Show cache stats
    stats = cache.get_cache_stats()
    logger.info(f"\n📊 Cache Statistics:")
    logger.info(f"   Videos cached: {stats['total_videos']}")
    logger.info(f"   Total windows: {stats['total_windows']}")
    logger.info(f"   Total metadata size: {stats['total_metadata_size_mb']:.2f} MB")
    logger.info(f"   Average per video: {stats['total_metadata_size_mb'] / max(stats['total_videos'], 1):.2f} MB")

    # Show sample structure
    if stats['total_videos'] > 0:
        sample_md5 = list(index.keys())[0]
        sample_path = cache.cache_dir / sample_md5
        logger.info(f"\n📁 Sample cache structure:")
        logger.info(f"   {sample_path}/")
        logger.info(f"   ├── metadata.h5")
        logger.info(f"   ├── frames/")
        logger.info(f"   │   └── window_X/")
        logger.info(f"   │       └── frame_XXX.png")
        logger.info(f"   └── emo_frames/")
        logger.info(f"       └── window_X/")
        logger.info(f"           └── frame_XXX.png")


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos with per-video cache structure')
    parser.add_argument('--video_folder', type=str, default='s1',
                       help='Path to video folder')
    parser.add_argument('--cache_dir', type=str, default='cache_per_video',
                       help='Cache directory (will create MD5 subfolders)')
    parser.add_argument('--max_videos', type=int, default=100,
                       help='Maximum number of videos to process')
    parser.add_argument('--window_size', type=int, default=50,
                       help='Window size')
    parser.add_argument('--stride', type=int, default=25,
                       help='Stride between windows')
    parser.add_argument('--frame-format', type=str, default='png', choices=['png', 'jpg'],
                       help='Image format for cached frames (default: png)')
    parser.add_argument('--use-existing-cache', action='store_true',
                       help='Use existing SingleBucketCache data if available')
    args = parser.parse_args()

    # Check if we should use existing cache
    if args.use_existing_cache:
        existing_cache = Path('cache_single_bucket/all_windows_cache.h5')
        if existing_cache.exists():
            logger.info("🔄 Using existing SingleBucketCache data...")
            logger.info("   This will be faster as it reuses already-processed windows")
            # Import and use migration approach
            from convert_to_per_video_cache import migrate_single_to_per_video
            migrate_single_to_per_video(
                single_cache_dir=Path('cache_single_bucket'),
                per_video_cache_dir=Path(args.cache_dir)
            )
            return

    # Otherwise, process from scratch
    # Load EMO model
    import importlib
    from omegaconf import OmegaConf

    logger.info("Loading volumetric avatar...")
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'nemo.models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
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

    # Create dataset
    logger.info("Creating dataset...")
    from vasa_va_bridge import VASAVolumetricAvatarBridge
    va_bridge = VASAVolumetricAvatarBridge(emo_model)

    dataset = VASAIntegratedDataset(
        video_folder=args.video_folder,
        emo_model=emo_model,
        window_size=args.window_size,
        stride=args.stride,
        max_videos=args.max_videos,
        cache_dir='cache_single_bucket',  # Use existing cache for window creation
        use_single_bucket=True,
        generate_emo_frames=True,
        emo_identity_path="nemo/data/IMG_1.png",
        emo_keyframes_per_window=50,
        va_bridge=va_bridge,
        auto_rebuild_expression_db=False  # Don't rebuild - use existing
    )

    # Preprocess with per-video cache
    preprocess_per_video_cache(
        dataset,
        Path(args.cache_dir),
        frame_format=args.frame_format
    )


if __name__ == "__main__":
    main()
