#!/usr/bin/env python3
"""
Test that the dataset is now properly extracting XY warps for each frame.
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

def test_dataset_warp_extraction():
    """Test warp extraction from the dataset."""

    logger.info("Loading volumetric avatar model...")
    volumetric_avatar = load_volumetric_avatar()

    logger.info("Creating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder='vasa_train_data/',
        emo_model=volumetric_avatar,
        window_size=4,
        stride=4,
        context_size=2,
        max_videos=1,  # Just test with one video
        cache_dir='test_cache/',
        use_single_bucket=False
    )

    if len(dataset) == 0:
        logger.error("No windows found in dataset!")
        return

    logger.info(f"Dataset has {len(dataset)} windows")

    # Get first window
    logger.info("\nExtracting first window...")
    try:
        sample = dataset[0]
    except Exception as e:
        logger.error(f"Failed to extract window: {e}")
        return

    # Check what was extracted
    logger.info("\n=== Checking Extracted Data ===")

    # Check motion data
    motion_data = sample.get('motion_data', {})
    logger.info("Motion data keys: " + str(motion_data.keys()))

    # Check for warps
    warp_found = False
    for key in ['xy_warps', 'rigid_warps', 'uv_warps', 'source_theta']:
        if key in motion_data:
            data = motion_data[key]
            if isinstance(data, torch.Tensor):
                logger.info(f"✓ {key}: shape={data.shape}, dtype={data.dtype}")

                # Check if XY warps are non-trivial
                if key == 'xy_warps':
                    warp_found = True
                    # Check magnitude
                    magnitude = torch.sqrt(torch.sum(data**2, dim=-1))
                    max_disp = magnitude.max().item()
                    mean_disp = magnitude.mean().item()
                    logger.info(f"  XY warp stats: max_displacement={max_disp:.3f}, mean={mean_disp:.3f}")

                    if max_disp < 0.01:
                        logger.warning("  ⚠️ XY warps appear to be nearly zero (no expression change detected)")
                    else:
                        logger.info("  ✓ XY warps have meaningful values!")

                # Check rigid warps
                if key == 'rigid_warps':
                    # Check deviation from identity
                    data_np = data.cpu().numpy()
                    if data_np.shape[-1] == 3:
                        # Rough check for non-identity
                        deviation = np.std(data_np)
                        logger.info(f"  Rigid warp std deviation: {deviation:.3f}")
            else:
                logger.warning(f"✗ {key}: Not a tensor, type={type(data)}")
        else:
            logger.warning(f"✗ {key}: Not found in motion_data")

    # Final verdict
    logger.info("\n=== Summary ===")
    if warp_found:
        logger.info("✓ XY warps are being extracted from the dataset!")
        logger.info("  The warps capture expression differences between frames")
        logger.info("  They transform from current expression → canonical")
    else:
        logger.error("✗ XY warps were not found in the dataset")
        logger.error("  Check that frames have different expressions")

if __name__ == "__main__":
    test_dataset_warp_extraction()