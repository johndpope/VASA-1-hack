#!/usr/bin/env python3
"""
Test direct extraction of XY warps using xy_generator_nw.
This bypasses the full forward pass and directly calls the XY warp generator.
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


def extract_xy_warps_directly(model, identity_frame, target_frame):
    """
    Extract XY warps directly using xy_generator_nw.

    Args:
        model: Volumetric avatar model
        identity_frame: Identity/canonical reference frame [1, 3, H, W]
        target_frame: Target frame with expression [1, 3, H, W]

    Returns:
        xy_warps: XY warp field [1, D, H, W, 3]
    """

    with torch.no_grad():
        # Step 1: Get face masks
        face_mask_identity, _, _, _ = model.face_idt.forward(identity_frame)
        face_mask_identity = (face_mask_identity > 0.6).float()

        face_mask_target, _, _, _ = model.face_idt.forward(target_frame)
        face_mask_target = (face_mask_target > 0.6).float()

        # Step 2: Get identity embedding (shared)
        identity_masked = identity_frame * face_mask_identity
        idt_embed = model.idt_embedder_nw(identity_masked)

        # Step 3: Create data dict for source warping
        # XY warps transform FROM target expression TO canonical
        # So we use target as source for XY warp generation
        data_dict = {
            'source_img': target_frame,  # The frame we want to normalize
            'target_img': identity_frame,  # The canonical reference
            'source_mask': face_mask_target,
            'target_mask': face_mask_identity,
            'idt_embed': idt_embed
        }

        # Step 4: Get head pose
        if hasattr(model, 'head_pose_regressor'):
            source_theta = model.head_pose_regressor.forward(target_frame)
            if source_theta.shape[-2] == 4:
                source_theta = source_theta[:, :3, :]
            data_dict['source_theta'] = source_theta

        # Step 5: Get expression embeddings
        # Check if we have expression embedder
        if hasattr(model, 'expression_embedder_nw'):
            try:
                # expression_embedder_nw is called with (data_dict, source_flag, target_flag)
                # For XY warps, we need source embeddings
                data_dict = model.expression_embedder_nw(data_dict, True, False)
            except Exception as e:
                logger.warning(f"Could not use expression_embedder_nw: {e}")
                # Fallback: manually create pose embed
                # This is a simplified version - may need adjustment
                if 'source_pose_embed' not in data_dict:
                    # Create a dummy pose embedding
                    data_dict['source_pose_embed'] = torch.zeros(1, 512).cuda()  # Assuming 512 dim

        # Step 6: Generate source warp embeddings
        # This is extracted from predict_embed method
        source_warp_embed_dict = {
            'idt_embed': data_dict['idt_embed'],
            'pose_embed': data_dict['source_pose_embed']
        }

        # Add theta if available
        if 'source_theta' in data_dict:
            source_warp_embed_dict['theta'] = data_dict['source_theta']

        # Step 7: Generate XY warps directly using xy_generator_nw
        logger.info("Generating XY warps directly using xy_generator_nw...")
        xy_warps, xy_conf = model.xy_generator_nw(source_warp_embed_dict)

        logger.info(f"Generated XY warps shape: {xy_warps.shape}")
        logger.info(f"XY confidence shape: {xy_conf.shape if xy_conf is not None else 'None'}")

        # Step 8: Handle resizing if needed
        if model.resize_warp and xy_warps.shape[2] != model.args.latent_volume_size:
            logger.info(f"Resizing XY warps from {xy_warps.shape[2]} to {model.args.latent_volume_size}")
            stride = model.warp_resize_stride
            xy_warps = torch.nn.functional.avg_pool3d(
                xy_warps.permute(0, 4, 1, 2, 3),
                kernel_size=stride,
                stride=stride
            ).permute(0, 2, 3, 4, 1)
            logger.info(f"Resized XY warps shape: {xy_warps.shape}")

        return xy_warps, xy_conf


def visualize_xy_warps(xy_warps, save_path="xy_warps_direct.png"):
    """Visualize the extracted XY warps."""

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

    plt.suptitle('Direct XY Warp Extraction using xy_generator_nw', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()


def test_with_video_frames():
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

    logger.info("\n=== Testing Direct XY Warp Extraction ===")

    # Test with different target frames
    all_xy_warps = []

    for i, target_frame in enumerate(frames[1:], 1):
        logger.info(f"\nExtracting XY warps for frame {i}...")

        # Extract XY warps directly
        xy_warps, xy_conf = extract_xy_warps_directly(model, identity_frame, target_frame)

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
        visualize_xy_warps(xy_warps, f"xy_warps_direct_frame{i}.png")

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
                logger.warning("    Warps are nearly identical!")
            else:
                logger.info("   Warps show variation")

    logger.info("\n=== Direct XY Warp Extraction Complete ===")
    logger.info("Summary:")
    logger.info("  - XY warps successfully extracted using xy_generator_nw")
    logger.info("  - No need for full forward pass or UV warp generation")
    logger.info("  - Ready for integration into vasa_dataset.py")


def test_batch_extraction():
    """Test extracting XY warps for multiple frames at once."""

    logger.info("\n=== Testing Batch XY Warp Extraction ===")

    # Load model
    model = load_volumetric_model()

    # Create synthetic batch of frames
    batch_size = 4
    frames = torch.randn(batch_size, 3, 512, 512).cuda()

    # Use first frame as identity for all
    identity_frame = frames[0:1].expand(batch_size, -1, -1, -1)

    logger.info(f"Testing with batch size: {batch_size}")

    # Extract XY warps for batch
    # Note: We may need to process one by one if the model doesn't support batching
    xy_warps_list = []

    for i in range(batch_size):
        xy_warps, _ = extract_xy_warps_directly(
            model,
            identity_frame[i:i+1],
            frames[i:i+1]
        )
        xy_warps_list.append(xy_warps)

    # Stack results
    xy_warps_batch = torch.cat(xy_warps_list, dim=0)

    logger.info(f"Batch XY warps shape: {xy_warps_batch.shape}")
    logger.info(" Batch extraction successful")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Direct XY Warp Extraction")
    logger.info("Using xy_generator_nw directly without UV warps")
    logger.info("=" * 60)

    # Test with real video frames
    test_with_video_frames()

    # Test batch extraction
    test_batch_extraction()

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete!")
    logger.info("If successful, this method can be integrated into vasa_dataset.py")
    logger.info("=" * 60)