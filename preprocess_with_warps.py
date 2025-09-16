#!/usr/bin/env python3
"""
Preprocess a single video and extract source warps for caching.
This extracts the source->canonical warps that neutralize the identity frame.
"""

import torch
import numpy as np
from pathlib import Path
import logging
import h5py
import cv2
from PIL import Image
from tqdm import tqdm

# Add nemo to path
import sys
sys.path.append('./nemo')

from infer import InferenceWrapper
from pipeline_face_attr_full import to_tensor, to_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_source_warps(inferer, identity_frame):
    """
    Extract source→canonical warps from the identity frame.
    These warps neutralize the source expression and pose to create a canonical volume.
    """
    logger.info("Extracting source warps from identity frame...")

    with torch.no_grad():
        results = inferer.forward(
            source_image=identity_frame,
            driver_image=identity_frame,  # Same as source for initialization
            crop=False,
            smooth_pose=False,
            target_theta=True,
            mix=True,
            mix_old=False,
            modnet_mask=False
        )

    # Extract the warps that were computed
    warps = {}

    if hasattr(inferer, 'source_xy_warp_resize'):
        warps['source_xy_warp'] = inferer.source_xy_warp_resize.clone().cpu().numpy()
        logger.info(f"Source XY warp shape: {warps['source_xy_warp'].shape}")

    if hasattr(inferer, 'source_rotation_warp'):
        warps['source_rotation_warp'] = inferer.source_rotation_warp.clone().cpu().numpy()
        logger.info(f"Source rotation warp shape: {warps['source_rotation_warp'].shape}")

    if hasattr(inferer, 'target_latent_volume'):
        warps['canonical_volume'] = inferer.target_latent_volume.clone().cpu().numpy()
        logger.info(f"Canonical volume shape: {warps['canonical_volume'].shape}")

    if hasattr(inferer, 'pred_source_theta'):
        warps['source_theta'] = inferer.pred_source_theta.clone().cpu().numpy()
        logger.info(f"Source theta shape: {warps['source_theta'].shape}")

    return warps


def preprocess_single_video_with_warps(video_path: str, cache_dir: Path, inferer):
    """Process a video and extract warps, saving to H5."""

    video_path = Path(video_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path
    import hashlib
    video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:16]
    h5_path = cache_dir / f"{video_hash}.h5"

    logger.info(f"Processing {video_path.name} -> {h5_path.name}")

    # Check if already exists with warps
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            if 'window_0' in f:
                window = f['window_0']
                if 'source_xy_warp' in window and 'source_rotation_warp' in window:
                    logger.info("Warps already cached!")
                    return h5_path

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    logger.info("Loading video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= 50:  # Just process first 50 frames for testing
            break

    cap.release()

    if len(frames) < 10:
        logger.warning(f"Video too short: {len(frames)} frames")
        return None

    # Use first frame as identity
    identity_frame = Image.fromarray(frames[0])

    # Extract source warps
    warps = extract_source_warps(inferer, identity_frame)

    # Save to H5
    with h5py.File(h5_path, 'w') as f:
        # Create a simple window for testing
        window = f.create_group('window_0')

        # Save frames (just for reference)
        frame_tensors = []
        for frame in frames[:50]:
            frame_pil = Image.fromarray(frame)
            frame_tensor = to_tensor(frame_pil)
            frame_tensors.append(frame_tensor)

        frames_data = torch.stack(frame_tensors).numpy()
        window.create_dataset('frames', data=frames_data, compression='gzip')

        # Save source warps (these are the same for all windows of this identity)
        for key, value in warps.items():
            if value is not None:
                window.create_dataset(key, data=value, compression='gzip')
                logger.info(f"Saved {key}: {value.shape}")

        # Add metadata
        window.attrs['video_path'] = str(video_path)
        window.attrs['num_frames'] = len(frames)

    logger.info(f"Saved to {h5_path}")
    return h5_path


def verify_warps_in_cache(cache_file: Path):
    """Verify that warps are properly saved in the cache."""
    logger.info(f"\nVerifying cache file: {cache_file}")

    with h5py.File(cache_file, 'r') as f:
        for window_key in f.keys():
            if window_key.startswith('window_'):
                window = f[window_key]
                logger.info(f"\n{window_key}:")

                # Check for source warps
                if 'source_xy_warp' in window:
                    shape = window['source_xy_warp'].shape
                    logger.info(f"  source_xy_warp: {shape}")
                else:
                    logger.info("  source_xy_warp: NOT FOUND")

                if 'source_rotation_warp' in window:
                    shape = window['source_rotation_warp'].shape
                    logger.info(f"  source_rotation_warp: {shape}")
                else:
                    logger.info("  source_rotation_warp: NOT FOUND")

                if 'canonical_volume' in window:
                    shape = window['canonical_volume'].shape
                    logger.info(f"  canonical_volume: {shape}")

                if 'source_theta' in window:
                    shape = window['source_theta'].shape
                    logger.info(f"  source_theta: {shape}")

                # Check other data
                if 'frames' in window:
                    logger.info(f"  frames: {window['frames'].shape}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='../junk/7.mp4',
                       help='Video file to process')
    parser.add_argument('--cache_dir', type=str, default='../temp_single_video/window_cache',
                       help='Cache directory')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    logger.info("Loading model...")
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

    # Process the video
    cache_file = preprocess_single_video_with_warps(args.video, cache_dir, inferer)

    if cache_file and cache_file.exists():
        # Verify the warps were saved
        verify_warps_in_cache(cache_file)
    else:
        logger.error("Failed to create cache file")


if __name__ == "__main__":
    main()