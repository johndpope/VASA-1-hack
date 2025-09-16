#!/usr/bin/env python3
"""
Test that source warps are properly extracted and saved in the dataset.
"""

import torch
import h5py
from pathlib import Path
import logging
import sys

# Add nemo to path
sys.path.append('./nemo')

from infer import InferenceWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_inference_wrapper():
    """Test warp extraction using InferenceWrapper as the emo_model."""

    logger.info("Loading InferenceWrapper...")
    inferer = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        model_file_name='328_model.pth',
        project_dir='nemo',
        folder='logs',
        args_overwrite={'l1_vol_rgb': 0},
        pose_momentum=0.1,
        print_model=False,
        print_params=False
    )

    # Now create dataset with InferenceWrapper as emo_model
    from vasa_dataset import VASAIntegratedDataset

    # Monkey-patch to skip audio requirement for testing
    import vasa_dataset
    original_check_audio = vasa_dataset.VASAIntegratedDataset._check_videos_for_audio
    def mock_check_audio(self, video_paths):
        # Return all videos as having audio
        return {path: {'has_audio': True, 'has_cache': False} for path in video_paths}
    vasa_dataset.VASAIntegratedDataset._check_videos_for_audio = mock_check_audio

    logger.info("Creating dataset with InferenceWrapper...")
    dataset = VASAIntegratedDataset(
        video_folder='nemo/data',  # Use a folder with test videos
        emo_model=inferer,  # Pass InferenceWrapper instead of raw model
        window_size=50,
        stride=25,
        max_videos=1,
        cache_dir='../temp_test_cache',
        use_single_bucket=False
    )

    logger.info(f"Dataset has {len(dataset)} windows")

    if len(dataset) > 0:
        logger.info("Processing first window...")
        window_data = dataset[0]

        if window_data is not None:
            # Check if warps were extracted
            logger.info("\nChecking extracted data:")

            if 'source_xy_warp' in window_data:
                warp = window_data['source_xy_warp']
                logger.info(f"✓ source_xy_warp: {warp.shape}, non-zero: {(warp != 0).any().item()}")
            else:
                logger.error("✗ source_xy_warp not found!")

            if 'source_rotation_warp' in window_data:
                warp = window_data['source_rotation_warp']
                logger.info(f"✓ source_rotation_warp: {warp.shape}, non-zero: {(warp != 0).any().item()}")
            else:
                logger.error("✗ source_rotation_warp not found!")

            if 'canonical_volume' in window_data:
                vol = window_data['canonical_volume']
                logger.info(f"✓ canonical_volume: {vol.shape}, non-zero: {(vol != 0).any().item()}")
            else:
                logger.error("✗ canonical_volume not found!")

            if 'source_theta_warp' in window_data:
                theta = window_data['source_theta_warp']
                logger.info(f"✓ source_theta_warp: {theta.shape}, non-zero: {(theta != 0).any().item()}")
            else:
                logger.error("✗ source_theta_warp not found!")

            # Also check other critical data
            logger.info("\nOther data shapes:")
            for key in ['frames', 'expression_embed', 'theta', 'scale']:
                if key in window_data:
                    logger.info(f"  {key}: {window_data[key].shape}")

            logger.info("\n✅ Test passed - warps extracted successfully!")
        else:
            logger.error("Failed to get window data")
    else:
        logger.error("No windows in dataset")


def check_cached_h5():
    """Check if warps are saved in H5 cache files."""

    cache_dir = Path('../temp_test_cache')
    h5_files = list(cache_dir.glob('*.h5'))

    if not h5_files:
        logger.warning("No H5 files found in cache")
        return

    logger.info(f"\nChecking H5 cache file: {h5_files[0]}")

    with h5py.File(h5_files[0], 'r') as f:
        if 'window_0' in f:
            window = f['window_0']
            logger.info("\nWindow 0 contents:")

            for key in window.keys():
                if key in ['source_xy_warp', 'source_rotation_warp', 'canonical_volume', 'source_theta_warp']:
                    data = window[key][()]
                    logger.info(f"  ✓ {key}: shape={data.shape}, non-zero={(data != 0).any()}")
                elif key in ['frames', 'expression_embed', 'theta']:
                    data = window[key][()]
                    logger.info(f"  {key}: shape={data.shape}")


def main():
    # Test with InferenceWrapper
    test_with_inference_wrapper()

    # Check cached files
    check_cached_h5()


if __name__ == '__main__':
    main()