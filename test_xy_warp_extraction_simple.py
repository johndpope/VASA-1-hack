#!/usr/bin/env python3
"""
Simplified test for direct XY warp extraction.
This version bypasses all the complex embed dict creation and directly tests xy_generator_nw.
"""

import torch
import numpy as np
import sys
import logging
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import cv2

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_volumetric_model():
    """Load the volumetric avatar model."""
    logger.info("Loading volumetric avatar model...")

    # Load config
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')

    # Initialize model
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Load weights
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        logger.info("Model weights loaded successfully")

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    # Set optimizer mode
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def extract_xy_warps_simple(model, identity_frame, target_frame):
    """
    Extract XY warps using the existing forward pass but only return XY warps.

    This is simpler than trying to manually construct the embed_dict.

    Args:
        model: Volumetric avatar model
        identity_frame: Identity/canonical reference frame [1, 3, H, W]
        target_frame: Target frame with expression [1, 3, H, W]

    Returns:
        xy_warps: XY warp field [1, D, H, W, 3]
    """

    with torch.no_grad():
        # Prepare data dict for the model's forward pass
        data_dict = {
            'source_img': identity_frame,  # Use identity as source
            'target_img': target_frame,     # Use target with expression
        }

        # Call the model's forward pass in test mode
        # This will generate all warps including XY warps
        _, _, _, output_dict = model.forward(
            data_dict,
            phase='test',
            optimizer_idx=0,
            visualize=False
        )

        # Extract XY warps from the output
        xy_warps = None

        # Check various possible keys where XY warps might be stored
        possible_keys = [
            'source_xy_warp',
            'xy_warp',
            'source_xy_warp_resize',
            'source_rotation_warp',  # Sometimes XY warps are called rotation warps
        ]

        for key in possible_keys:
            if key in output_dict:
                xy_warps = output_dict[key]
                logger.info(f"Found XY warps under key: {key}")
                logger.info(f"XY warps shape: {xy_warps.shape}")
                break

        if xy_warps is None:
            logger.warning("Could not find XY warps in output_dict")
            logger.warning(f"Available keys: {list(output_dict.keys())}")

        return xy_warps


def visualize_xy_warps(xy_warps, save_path="xy_warps_simple.png"):
    """Visualize the extracted XY warps."""

    if xy_warps is None:
        logger.error("No XY warps to visualize")
        return

    # Take middle depth slice
    mid_depth = xy_warps.shape[1] // 2
    xy_slice = xy_warps[0, mid_depth].cpu().numpy()  # [H, W, 3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X component
    im0 = axes[0].imshow(xy_slice[..., 0], cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0].set_title('XY Warp - X Component')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Y component
    im1 = axes[1].imshow(xy_slice[..., 1], cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title('XY Warp - Y Component')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Magnitude
    magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
    im2 = axes[2].imshow(magnitude, cmap='viridis', vmin=0, vmax=3)
    axes[2].set_title(f'XY Warp Magnitude (max: {magnitude.max():.3f})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle('XY Warp Extraction (Simplified)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()


def test_with_video():
    """Test XY warp extraction with real video frames."""

    # Load model
    model = load_volumetric_model()

    # Load test video
    video_path = Path("temp_single_video/15.mp4")
    if not video_path.exists():
        logger.error("Test video not found!")
        return

    logger.info(f"Loading frames from {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    # Get frames
    frames = []
    frame_indices = [0, 50, 100]  # Identity, expression 1, expression 2

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            frames.append(frame.cuda())

    cap.release()

    if len(frames) < 2:
        logger.error("Could not load enough frames")
        return

    # Use first frame as identity
    identity_frame = frames[0]

    logger.info("\n=== Testing Simplified XY Warp Extraction ===")

    # Test with different target frames
    all_xy_warps = []

    for i, target_frame in enumerate(frames[1:], 1):
        logger.info(f"\nExtracting XY warps for frame {i}...")

        # Extract XY warps using simplified method
        xy_warps = extract_xy_warps_simple(model, identity_frame, target_frame)

        if xy_warps is not None:
            # Check statistics
            xy_mean = xy_warps.mean().item()
            xy_std = xy_warps.std().item()
            xy_max = xy_warps.abs().max().item()

            logger.info(f"XY Warp statistics:")
            logger.info(f"  Mean: {xy_mean:.6f}")
            logger.info(f"  Std: {xy_std:.6f}")
            logger.info(f"  Max absolute: {xy_max:.6f}")

            all_xy_warps.append(xy_warps)

            # Visualize
            visualize_xy_warps(xy_warps, f"xy_warps_simple_frame{i}.png")

    # Compare warps between frames
    if len(all_xy_warps) > 1:
        logger.info("\n=== Comparing XY Warps Between Frames ===")

        for i in range(len(all_xy_warps) - 1):
            warp1 = all_xy_warps[i]
            warp2 = all_xy_warps[i + 1]

            diff = torch.abs(warp2 - warp1)
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()

            logger.info(f"Frame {i+1} vs Frame {i+2}:")
            logger.info(f"  Mean difference: {mean_diff:.6f}")
            logger.info(f"  Max difference: {max_diff:.6f}")

            if mean_diff < 1e-6:
                logger.warning("  ⚠️ Warps are nearly identical!")
            else:
                logger.info("  ✓ Warps show variation")

    logger.info("\n=== Simplified XY Warp Extraction Complete ===")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Simplified XY Warp Extraction Test")
    logger.info("Using model's forward pass to get XY warps")
    logger.info("=" * 60)

    test_with_video()

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete!")
    logger.info("This simplified method can be used in vasa_dataset.py")
    logger.info("It avoids the complexity of manually constructing embed_dict")
    logger.info("=" * 60)