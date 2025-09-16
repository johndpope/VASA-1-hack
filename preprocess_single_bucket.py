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
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_cache(dataset, cache_dir: Path):
    """Preprocess all windows and save to single-bucket cache."""

    cache = SingleBucketCache(
        cache_dir=cache_dir,
        cache_name="all_windows_cache.h5",
        compression='gzip',
        compression_level=4
    )

    if cache.has_cache():
        logger.info("Cache already exists. Delete it first if you want to rebuild.")
        info = cache.get_cache_info()
        logger.info(f"Cache info: {info}")
        return

    logger.info(f"Processing {len(dataset)} windows...")

    all_windows = []
    failed_indices = []

    for idx in range(len(dataset)):
        if idx % 10 == 0:
            logger.info(f"Processing window {idx}/{len(dataset)}")

        try:
            # Get window data from dataset
            window_data = dataset[idx]

            if window_data is not None:
                # Add index to metadata
                if 'metadata' not in window_data:
                    window_data['metadata'] = {}
                window_data['metadata']['window_index'] = idx

                all_windows.append(window_data)
            else:
                failed_indices.append(idx)
                logger.warning(f"Window {idx} returned None")

        except Exception as e:
            logger.error(f"Error processing window {idx}: {str(e)}")
            failed_indices.append(idx)

    # Save all windows to cache
    if all_windows:
        logger.info(f"Saving {len(all_windows)} windows to cache...")

        metadata = {
            'total_windows': len(dataset),
            'successful_windows': len(all_windows),
            'failed_windows': len(failed_indices),
            'dataset_config': {
                'window_size': dataset.window_size,
                'stride': dataset.stride,
                'sequence_length': dataset.sequence_length,
                'frame_size': dataset.frame_size,
            }
        }

        cache.save_all_windows(all_windows, metadata)

        logger.info(f"Successfully cached {len(all_windows)} windows")
        if failed_indices:
            logger.warning(f"Failed to process {len(failed_indices)} windows: {failed_indices[:10]}...")

        # Validate cache
        is_valid, issues = cache.validate_cache()
        if is_valid:
            logger.info("Cache validation passed!")
        else:
            logger.error(f"Cache validation failed: {issues}")
    else:
        logger.error("No windows were successfully processed!")


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
    dataset = VASAIntegratedDataset(
        video_folder=args.video_folder,
        emo_model=emo_model,
        window_size=args.window_size,
        stride=args.stride,
        max_videos=args.max_videos,
        cache_dir=args.cache_dir,
        use_single_bucket=False  # We'll handle caching manually
    )

    # Preprocess and cache
    preprocess_and_cache(dataset, Path(args.cache_dir))


if __name__ == "__main__":
    main()