#!/usr/bin/env python3
"""
Quick test to check if warps vary frame by frame.
"""

import torch
import numpy as np
import sys
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, 'nemo')

from vasa_dataset import VASAIntegratedDataset
import importlib
from omegaconf import OmegaConf

def load_volumetric_avatar():
    """Load the volumetric avatar model."""
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    return volumetric_avatar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Quick test of warp variations."""

    logger.info("Loading volumetric avatar model...")
    volumetric_avatar = load_volumetric_avatar()

    # Find test video
    test_video = Path("temp_single_video")
    if not test_video.exists():
        logger.error("Test video folder not found!")
        return

    logger.info("Creating dataset with small window...")
    dataset = VASAIntegratedDataset(
        video_folder=str(test_video),
        emo_model=volumetric_avatar,
        window_size=3,  # Just 3 frames for quick test
        stride=1,
        context_size=0,  # No context for simplicity
        max_videos=1,
        cache_dir='test_cache/',
        use_single_bucket=False
    )

    if len(dataset) == 0:
        logger.error("No windows found!")
        return

    logger.info(f"Dataset has {len(dataset)} windows")

    # Get first window
    logger.info("\nExtracting first window...")
    sample = dataset[0]

    # Check XY warps
    motion_data = sample.get('motion_data', {})
    if 'xy_warps' not in motion_data:
        logger.error("No XY warps found!")
        return

    xy_warps = motion_data['xy_warps']
    logger.info(f"XY warps shape: {xy_warps.shape}")

    # Quick check: are warps different?
    logger.info("\n=== Checking Warp Variations ===")

    if xy_warps.shape[0] < 2:
        logger.error("Need at least 2 frames!")
        return

    # Compare first two frames
    frame0 = xy_warps[0]
    frame1 = xy_warps[1]

    # Check if identical
    are_same = torch.allclose(frame0, frame1, atol=1e-6)

    # Compute difference
    diff = torch.abs(frame1 - frame0)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    logger.info(f"Frame 0 vs Frame 1:")
    logger.info(f"  Identical: {'YES ❌' if are_same else 'NO ✅'}")
    logger.info(f"  Mean difference: {mean_diff:.6f}")
    logger.info(f"  Max difference: {max_diff:.6f}")

    # If we have 3 frames, check the third too
    if xy_warps.shape[0] >= 3:
        frame2 = xy_warps[2]
        are_same_02 = torch.allclose(frame0, frame2, atol=1e-6)
        diff_02 = torch.abs(frame2 - frame0)

        logger.info(f"\nFrame 0 vs Frame 2:")
        logger.info(f"  Identical: {'YES ❌' if are_same_02 else 'NO ✅'}")
        logger.info(f"  Mean difference: {diff_02.mean().item():.6f}")
        logger.info(f"  Max difference: {diff_02.max().item():.6f}")

    # Final verdict
    logger.info("\n=== VERDICT ===")
    if are_same:
        logger.error("❌ WARPS ARE IDENTICAL - No frame variation detected!")
    else:
        logger.info("✅ WARPS DIFFER BETWEEN FRAMES - Expression changes are captured!")

if __name__ == "__main__":
    main()