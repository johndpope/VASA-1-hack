#!/usr/bin/env python3
"""
Test pipeline with cached source warps to generate facial motion.
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import logging
import sys
import imageio

# Add nemo to path
sys.path.append('./nemo')

from infer import InferenceWrapper
from pipeline_face_attr_full import (
    to_tensor, to_image, inject_cached_source_warps, inject_cached_expressions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cached_data(h5_path):
    """Load cached face attributes and warps from H5 file."""

    logger.info(f"Loading cached data from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'window_0' not in f:
            raise ValueError("No window_0 in cache file")

        window = f['window_0']

        # Load source warps
        warps = {}
        if 'source_xy_warp' in window:
            warps['source_xy_warp'] = torch.from_numpy(window['source_xy_warp'][()])
            logger.info(f"Loaded source_xy_warp: {warps['source_xy_warp'].shape}")

        if 'source_rotation_warp' in window:
            warps['source_rotation_warp'] = torch.from_numpy(window['source_rotation_warp'][()])
            logger.info(f"Loaded source_rotation_warp: {warps['source_rotation_warp'].shape}")

        if 'canonical_volume' in window:
            warps['canonical_volume'] = torch.from_numpy(window['canonical_volume'][()])
            logger.info(f"Loaded canonical_volume: {warps['canonical_volume'].shape}")

        if 'source_theta' in window:
            warps['source_theta'] = torch.from_numpy(window['source_theta'][()])
            logger.info(f"Loaded source_theta: {warps['source_theta'].shape}")

        # Load frames for reference
        if 'frames' in window:
            frames = torch.from_numpy(window['frames'][()])
            logger.info(f"Loaded frames: {frames.shape}")
        else:
            frames = None

        # Load expressions if available (for future use)
        expressions = None
        if 'expression_embed' in window:
            expressions = torch.from_numpy(window['expression_embed'][()])
            logger.info(f"Loaded expressions: {expressions.shape}")

    return warps, frames, expressions


def generate_with_cached_warps(inferer, warps, frames, output_path):
    """Generate video using cached warps."""

    logger.info("Generating video with cached warps...")

    # Inject the cached source warps
    inject_cached_source_warps(
        inferer,
        source_xy_warp=warps.get('source_xy_warp'),
        source_rotation_warp=warps.get('source_rotation_warp'),
        canonical_volume=warps.get('canonical_volume'),
        source_theta=warps.get('source_theta')
    )

    # Use first frame as identity
    if frames is not None and len(frames) > 0:
        identity_frame = to_image(frames[0])
    else:
        # Create a dummy frame if needed
        logger.warning("No frames found, using dummy identity")
        identity_frame = None
        return

    generated_frames = []

    # Generate frames with different expressions/poses
    logger.info("Generating frames with motion...")
    for i in range(10):  # Generate 10 frames as test
        # For now, use identity as both source and driver
        # In a real scenario, you'd vary the driver to create motion
        result = inferer.forward(
            source_image=identity_frame,
            driver_image=identity_frame,  # In real use, this would change
            crop=False,
            smooth_pose=False,
            target_theta=True,
            mix=True,
            mix_old=False,
            modnet_mask=False,
            frame_idx=i
        )

        # Extract generated frame
        if isinstance(result, tuple):
            generated = result[0]
        else:
            generated = result

        # Convert to numpy for saving
        if isinstance(generated, torch.Tensor):
            frame = generated.squeeze(0).cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.array(generated)

        generated_frames.append(frame)
        logger.info(f"Generated frame {i+1}/10")

    # Save as video
    if generated_frames:
        logger.info(f"Saving video to {output_path}")
        imageio.mimsave(output_path, generated_frames, fps=10)
        logger.info("Video saved successfully!")

    return generated_frames


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str,
                       default='../temp_single_video/window_cache/6ae61842675c083d.h5',
                       help='Path to H5 cache file with warps')
    parser.add_argument('--output', type=str, default='test_warps_output.mp4',
                       help='Output video path')
    args = parser.parse_args()

    # Load model
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

    # Load cached data
    warps, frames, expressions = load_cached_data(args.h5_path)

    if not warps:
        logger.error("No warps found in cache!")
        return

    # Generate video with cached warps
    generate_with_cached_warps(inferer, warps, frames, args.output)

    logger.info("\n✅ Test completed successfully!")


if __name__ == '__main__':
    main()