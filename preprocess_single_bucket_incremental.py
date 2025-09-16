#!/usr/bin/env python3
"""
Preprocess and cache windows for single-bucket training with incremental saving.
Saves each window immediately after processing to avoid memory issues.
"""

import argparse
import torch
from pathlib import Path
from vasa_dataset import VASAIntegratedDataset
from single_bucket_cache import SingleBucketCache
import logging
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_and_cache_incremental(dataset, cache_dir: Path):
    """Process windows and save them incrementally to avoid memory issues."""

    cache_path = cache_dir / "all_windows_cache.h5"

    # Check if cache already exists
    if cache_path.exists():
        logger.info(f"Cache already exists at {cache_path}")
        logger.info("Deleting existing cache to rebuild...")
        cache_path.unlink()
        logger.info("Deleted existing cache.")

    logger.info(f"Creating new cache at {cache_path}")
    logger.info(f"Total windows to process: {len(dataset)}")

    # Create cache file with compression
    with h5py.File(cache_path, 'w') as f:
        # Set metadata
        f.attrs['cache_version'] = '1.0'
        f.attrs['total_windows'] = len(dataset)
        f.attrs['window_size'] = dataset.window_size
        f.attrs['stride'] = dataset.stride
        f.attrs['sequence_length'] = dataset.sequence_length
        f.attrs['frame_size'] = dataset.frame_size

        successful_windows = 0
        failed_windows = []

        # Process each window and save immediately
        for idx in range(len(dataset)):
            if idx % 10 == 0:
                logger.info(f"Processing window {idx}/{len(dataset)} ({idx/len(dataset)*100:.1f}%)")

            try:
                # Get window data from dataset
                window_data = dataset[idx]

                if window_data is not None:
                    # Create group for this window
                    window_group = f.create_group(f'window_{idx}')

                    # Save each tensor in the window (only save identity frame)
                    for key, value in window_data.items():
                        # Handle frames specially - only save first frame as identity_frame
                        if key == 'frames':
                            if isinstance(value, torch.Tensor) and len(value) > 0:
                                # Save only the first frame as identity_frame
                                window_group.create_dataset(
                                    'identity_frame',
                                    data=value[0].cpu().numpy(),  # Just the first frame
                                    compression='gzip',
                                    compression_opts=4
                                )
                                logger.debug(f"Saved identity frame for window {idx}")
                            continue

                        if key == 'metadata':
                            # Handle metadata group separately
                            metadata_group = window_group.create_group('metadata')
                            metadata_group.attrs['window_index'] = idx
                            metadata_group.attrs['video_path'] = str(dataset.windows[idx]['video_path'])
                            metadata_group.attrs['start_frame'] = dataset.windows[idx]['start_frame']
                            metadata_group.attrs['end_frame'] = dataset.windows[idx]['end_frame']
                        elif isinstance(value, dict):
                            # Handle nested dictionaries (like lip_metrics)
                            nested_group = window_group.create_group(key)
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, torch.Tensor):
                                    nested_group.create_dataset(
                                        sub_key,
                                        data=sub_value.cpu().numpy(),
                                        compression='gzip',
                                        compression_opts=4
                                    )
                        elif isinstance(value, torch.Tensor):
                            # Convert CUDA tensors to CPU before saving
                            window_group.create_dataset(
                                key,
                                data=value.cpu().numpy(),
                                compression='gzip',
                                compression_opts=4
                            )

                    successful_windows += 1

                    # Flush to disk every 50 windows
                    if successful_windows % 50 == 0:
                        f.flush()
                        logger.info(f"Flushed {successful_windows} windows to disk")

                else:
                    failed_windows.append(idx)
                    logger.warning(f"Window {idx} returned None")

            except Exception as e:
                logger.error(f"Error processing window {idx}: {str(e)}")
                failed_windows.append(idx)

        # Update final metadata
        f.attrs['successful_windows'] = successful_windows
        f.attrs['failed_windows'] = len(failed_windows)

        logger.info(f"Successfully cached {successful_windows} windows")
        if failed_windows:
            logger.warning(f"Failed to process {len(failed_windows)} windows: {failed_windows[:10]}...")

    # Verify the cache was created
    if cache_path.exists():
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Cache created successfully: {file_size_mb:.2f} MB")

        # Quick validation
        with h5py.File(cache_path, 'r') as f:
            num_windows = len([k for k in f.keys() if k.startswith('window_')])
            logger.info(f"Cache contains {num_windows} windows")
    else:
        logger.error("Failed to create cache file!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess windows for single-bucket cache with incremental saving')
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
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing cache')
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

    # Create dataset
    logger.info("Creating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder=args.video_folder,
        cache_dir=Path(args.cache_dir),
        emo_model=emo_model,
        window_size=args.window_size,
        stride=args.stride,
        use_single_bucket=False,  # Don't use SingleBucketCache class during preprocessing
        max_videos=args.max_videos
    )

    # Run preprocessing with incremental saving
    preprocess_and_cache_incremental(dataset, Path(args.cache_dir))


if __name__ == "__main__":
    main()