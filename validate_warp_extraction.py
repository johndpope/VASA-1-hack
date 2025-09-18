#!/usr/bin/env python3
"""
Validate that warps are different frame by frame and capture expression variations.
"""

import torch
import numpy as np
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def validate_frame_by_frame_warps():
    """Validate that warps are different for each frame."""

    logger.info("Loading volumetric avatar model...")
    volumetric_avatar = load_volumetric_avatar()

    # Try to find a test video
    test_video = None
    video_paths = [
        Path("temp_single_video/15.mp4"),
        Path("nemo/data/VID_1.mp4"),
        Path("nemo/data/VID_2.mp4"),
    ]

    for video_path in video_paths:
        if video_path.exists():
            test_video = video_path.parent
            logger.info(f"Using test video from: {test_video}")
            break

    if test_video is None:
        logger.error("No test videos found!")
        return

    logger.info("Creating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder=str(test_video),
        emo_model=volumetric_avatar,
        window_size=8,  # Get more frames to compare
        stride=1,  # Consecutive frames
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
    logger.info("\nExtracting first window with multiple frames...")
    try:
        sample = dataset[0]
    except Exception as e:
        logger.error(f"Failed to extract window: {e}")
        return

    # Extract motion data
    motion_data = sample.get('motion_data', {})

    # Check XY warps
    if 'xy_warps' not in motion_data:
        logger.error("No XY warps found in motion data!")
        return

    xy_warps = motion_data['xy_warps']  # Shape: [window_size, D, H, W, 3]
    logger.info(f"XY warps shape: {xy_warps.shape}")

    if xy_warps.shape[0] < 2:
        logger.error("Need at least 2 frames to compare warps!")
        return

    # Analyze differences between consecutive frames
    logger.info("\n=== Frame-by-Frame Warp Analysis ===")

    differences = []
    for i in range(xy_warps.shape[0] - 1):
        warp_curr = xy_warps[i]
        warp_next = xy_warps[i+1]

        # Compute difference
        diff = torch.abs(warp_next - warp_curr)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        # Check if warps are identical (would indicate a problem)
        are_identical = torch.allclose(warp_curr, warp_next, atol=1e-6)

        differences.append(mean_diff)

        logger.info(f"Frame {i} → {i+1}:")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Identical: {'YES ⚠️' if are_identical else 'NO ✓'}")

        if are_identical:
            logger.warning(f"  ⚠️ Warps are identical between frames {i} and {i+1}!")

    # Compute overall statistics
    logger.info("\n=== Overall Statistics ===")

    # Check if all frames have the same warp (major problem)
    first_warp = xy_warps[0]
    all_same = True
    for i in range(1, xy_warps.shape[0]):
        if not torch.allclose(first_warp, xy_warps[i], atol=1e-6):
            all_same = False
            break

    if all_same:
        logger.error("✗ ALL WARPS ARE IDENTICAL! No frame-by-frame variation detected!")
        logger.error("  This means expression changes are not being captured")
    else:
        logger.info("✓ Warps differ between frames - expression changes are being captured!")
        logger.info(f"  Average difference between consecutive frames: {np.mean(differences):.6f}")
        logger.info(f"  Max difference between consecutive frames: {np.max(differences):.6f}")

    # Visualize warp variations
    logger.info("\n=== Creating Visualization ===")
    visualize_warp_variations(xy_warps)

    # Check other warp types if available
    if 'rigid_warps' in motion_data:
        rigid_warps = motion_data['rigid_warps']
        logger.info(f"\nRigid warps shape: {rigid_warps.shape}")
        check_warp_variation(rigid_warps, "Rigid")

    if 'uv_warps' in motion_data:
        uv_warps = motion_data['uv_warps']
        logger.info(f"\nUV warps shape: {uv_warps.shape}")
        check_warp_variation(uv_warps, "UV")

    # Final verdict
    logger.info("\n=== FINAL VERDICT ===")
    if all_same:
        logger.error("✗ FAILED: Warps are not capturing frame-by-frame expression changes")
        logger.error("Possible issues:")
        logger.error("  1. Identity frame might be the same as all other frames")
        logger.error("  2. Model might not be computing expression differences")
        logger.error("  3. Warp extraction might be using wrong frames")
    else:
        logger.info("✓ SUCCESS: Warps are different frame by frame!")
        logger.info("  The model is correctly capturing expression variations")
        logger.info("  Each frame's warp transforms its unique expression to canonical space")

def check_warp_variation(warps, name):
    """Check if warps vary between frames."""
    if warps.shape[0] < 2:
        return

    first_warp = warps[0]
    all_same = True
    for i in range(1, warps.shape[0]):
        if not torch.allclose(first_warp, warps[i], atol=1e-6):
            all_same = False
            break

    if all_same:
        logger.warning(f"  {name} warps: All frames identical ⚠️")
    else:
        logger.info(f"  {name} warps: Vary between frames ✓")

def visualize_warp_variations(xy_warps):
    """Create visualization showing how warps vary across frames."""

    num_frames = min(xy_warps.shape[0], 8)  # Visualize up to 8 frames

    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames*2, 4))
    fig.suptitle("XY Warp Variations Across Frames", fontsize=14)

    # Take middle depth slice
    mid_depth = xy_warps.shape[1] // 2

    for i in range(num_frames):
        warp = xy_warps[i, mid_depth].cpu().numpy()

        # X displacement
        im = axes[0, i].imshow(warp[..., 0], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, i].set_title(f"Frame {i}\nX disp", fontsize=10)
        axes[0, i].axis('off')

        # Y displacement
        im = axes[1, i].imshow(warp[..., 1], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, i].set_title(f"Y disp", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = "warp_frame_variations.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()

    # Create difference plot
    if num_frames > 1:
        fig, axes = plt.subplots(1, num_frames-1, figsize=(num_frames*2, 3))
        if num_frames == 2:
            axes = [axes]
        fig.suptitle("Warp Differences Between Consecutive Frames", fontsize=14)

        for i in range(num_frames - 1):
            warp_curr = xy_warps[i, mid_depth].cpu().numpy()
            warp_next = xy_warps[i+1, mid_depth].cpu().numpy()
            diff = np.sqrt((warp_next[..., 0] - warp_curr[..., 0])**2 +
                          (warp_next[..., 1] - warp_curr[..., 1])**2)

            im = axes[i].imshow(diff, cmap='viridis')
            axes[i].set_title(f"Frame {i}→{i+1}\nMax: {diff.max():.3f}", fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = "warp_frame_differences.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved difference visualization to {save_path}")
        plt.close()

if __name__ == "__main__":
    validate_frame_by_frame_warps()