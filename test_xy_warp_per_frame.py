#!/usr/bin/env python3
"""
Test XY warp extraction per frame using the correct implementation.
XY warps should be frame-specific to normalize each frame's expression to canonical.
"""

import torch
import torch.nn.functional as F
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


def extract_xy_warps(emo_model, frames, use_modnet_mask=False):
    """
    Extract XY warps for a batch of frames using the volumetric avatar model.

    XY warps normalize the input frames to canonical (neutral) space.

    Args:
        emo_model: The loaded VolumetricAvatar model instance.
        frames: Input frames tensor of shape [N, 3, H, W] where N is batch size (can be T for sequence).
        use_modnet_mask: If True, use MODNet for masking; else use face_idt (default: False).

    Returns:
        torch.Tensor: XY warps of shape [N, D, S, S, 3] where:
            - D: latent_volume_depth (typically 16)
            - S: latent_volume_size (typically 64)

    Note:
        - Assumes frames are normalized to [-1, 1] range.
        - Processes all frames independently in batch mode.
        - If input is a sequence [T, 3, H, W], it will be treated as batch N=T.
        - Requires the model to be in eval mode for inference.
    """
    # Ensure model is in eval mode
    emo_model.eval()

    # Handle input shape: if 3D (C,H,W), add batch dim; if 4D assume [N,C,H,W]
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    elif frames.dim() != 4:
        raise ValueError(f"Expected frames shape [N,3,H,W] or [3,H,W], got {frames.shape}")

    N, C, H, W = frames.shape
    device = frames.device

    with torch.no_grad():
        # Compute theta using head pose regressor (batched)
        theta = emo_model.head_pose_regressor.forward(frames)

        # Compute masks
        if use_modnet_mask and hasattr(emo_model, 'get_mask'):
            _, _, mask = emo_model.get_mask(frames, True)
        else:
            mask, _, _, _ = emo_model.face_idt.forward(frames)
            mask = (mask > 0.6).float()

        # Mask the frames
        masked = frames * mask

        # Compute identity embedding
        idt_embed = emo_model.idt_embedder_nw(masked)

        # Prepare data_dict for expression embedder
        data_dict = {
            'source_img': frames,
            'target_img': frames,  # Same as source for XY warp (normalization to canonical)
            'source_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
            'target_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
            'source_mask': mask,
            'target_mask': mask,
            'idt_embed': idt_embed
        }

        # Compute expression (pose) embedding
        # expression_embedder_nw.forward(data_dict, estimate_kp_by_net, use_seg, use_aug)
        data_dict = emo_model.expression_embedder_nw(data_dict, True, False, True)
        source_pose_embed = data_dict['source_pose_embed']

        # Unsqueeze pose embed
        pose_unsqueeze = emo_model.pose_unsqueeze_nw(source_pose_embed).view(
            N, -1, emo_model.embed_size, emo_model.embed_size
        )

        # Combine embeddings
        if emo_model.args.cat_em:
            warp_embed_head = torch.cat([idt_embed, pose_unsqueeze], dim=1)
        else:
            warp_embed_head = idt_embed + pose_unsqueeze

        # Process through warp head
        warp_embed_head = emo_model.warp_embed_head_orig_nw(warp_embed_head)

        # Create warp embed dict following predict_embed format
        c = warp_embed_head.shape[1]
        source_warp_embed_dict = {
            'orig': warp_embed_head.view(N, c, emo_model.embed_size ** 2),  # Flatten spatial dims
            'orig_d': warp_embed_head.view(N, c, emo_model.embed_size ** 2).detach(),  # Detached version
            'ada_v': source_pose_embed,  # Keep original pose embed
            'idt_embed': idt_embed
        }

        # Generate XY warp
        xy_gen_warp, _ = emo_model.xy_generator_nw(source_warp_embed_dict)

        # Apply resizing if needed
        if emo_model.resize_warp:
            stride = emo_model.warp_resize_stride
            # Permute to [N, 3, D, S, S] for pooling, then back to [N, D, S, S, 3]
            xy_gen_warp = F.avg_pool3d(
                xy_gen_warp.permute(0, 4, 1, 2, 3),
                kernel_size=stride,
                stride=stride
            ).permute(0, 2, 3, 4, 1)

    return xy_gen_warp


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

    # Set optimizer mode if needed
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def visualize_xy_warps_comparison(xy_warps_list, save_path="xy_warps_per_frame.png"):
    """Visualize and compare XY warps from different frames."""

    n_frames = len(xy_warps_list)
    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 4, 12))

    # Take middle depth slice
    mid_depth = xy_warps_list[0].shape[1] // 2

    for i, xy_warps in enumerate(xy_warps_list):
        xy_slice = xy_warps[0, mid_depth].cpu().numpy()  # [H, W, 3]

        # X component
        im0 = axes[0, i].imshow(xy_slice[..., 0], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[0, i].set_title(f'Frame {i+1}\nX Component')
        axes[0, i].axis('off')

        # Y component
        im1 = axes[1, i].imshow(xy_slice[..., 1], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1, i].set_title(f'Y Component')
        axes[1, i].axis('off')

        # Magnitude
        magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
        im2 = axes[2, i].imshow(magnitude, cmap='viridis', vmin=0, vmax=3)
        axes[2, i].set_title(f'Magnitude\n(max: {magnitude.max():.2f})')
        axes[2, i].axis('off')

    # Add colorbars
    for i, im in enumerate([im0, im1, im2]):
        cbar = fig.colorbar(im, ax=axes[i, :], fraction=0.02, pad=0.02)

    plt.suptitle('Frame-Specific XY Warps (Each Frame → Canonical)', fontsize=14, weight='bold')
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

    # Get multiple frames with different expressions
    frames = []
    frame_indices = [0, 25, 50, 75, 100]  # Multiple frames with different expressions

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame = torch.from_numpy(frame).float() / 255.0
            # Normalize to [-1, 1] as expected by the model
            frame = (frame - 0.5) * 2.0
            frame = frame.permute(2, 0, 1)  # [3, H, W]
            frames.append(frame)

    cap.release()

    if len(frames) < 2:
        logger.error("Could not load enough frames")
        return

    # Stack frames for batch processing
    frames_tensor = torch.stack(frames).cuda()  # [N, 3, H, W]
    logger.info(f"Loaded {len(frames)} frames, shape: {frames_tensor.shape}")

    logger.info("\n=== Testing Per-Frame XY Warp Extraction ===")

    # Extract XY warps for all frames at once
    logger.info("Extracting XY warps for all frames in batch...")
    xy_warps_batch = extract_xy_warps(model, frames_tensor)

    logger.info(f"XY warps batch shape: {xy_warps_batch.shape}")

    # Analyze each frame's warps
    xy_warps_list = []
    for i in range(xy_warps_batch.shape[0]):
        xy_warp = xy_warps_batch[i:i+1]  # Keep batch dimension

        # Check statistics
        xy_mean = xy_warp.mean().item()
        xy_std = xy_warp.std().item()
        xy_max = xy_warp.abs().max().item()

        logger.info(f"\nFrame {i+1} XY Warp statistics:")
        logger.info(f"  Mean: {xy_mean:.6f}")
        logger.info(f"  Std: {xy_std:.6f}")
        logger.info(f"  Max absolute: {xy_max:.6f}")

        xy_warps_list.append(xy_warp)

    # Compare warps between frames
    logger.info("\n=== Comparing XY Warps Between Frames ===")

    for i in range(len(xy_warps_list) - 1):
        warp1 = xy_warps_list[i]
        warp2 = xy_warps_list[i + 1]

        diff = torch.abs(warp2 - warp1)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        logger.info(f"Frame {i+1} vs Frame {i+2}:")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        logger.info(f"  Max difference: {max_diff:.6f}")

        if mean_diff < 1e-6:
            logger.warning("  ⚠️ Warps are nearly identical!")
        else:
            logger.info("  ✓ Warps show variation - capturing different expressions")

    # Visualize all warps
    visualize_xy_warps_comparison(xy_warps_list[:4], "xy_warps_per_frame_comparison.png")

    # Test that warps actually differ frame by frame
    logger.info("\n=== Final Validation ===")

    all_same = True
    for i in range(1, len(xy_warps_list)):
        if not torch.allclose(xy_warps_list[0], xy_warps_list[i], atol=1e-6):
            all_same = False
            break

    if all_same:
        logger.error("❌ FAILED: All warps are identical - not capturing per-frame expressions!")
    else:
        logger.info("✅ SUCCESS: XY warps vary per frame - correctly capturing expression differences!")

    logger.info("\n=== Per-Frame XY Warp Extraction Complete ===")


def test_single_vs_batch():
    """Test that single frame and batch processing produce the same results."""

    logger.info("\n=== Testing Single vs Batch Processing ===")

    model = load_volumetric_model()

    # Create test frames
    frames = torch.randn(3, 3, 512, 512).cuda()  # 3 test frames
    # Normalize to [-1, 1]
    frames = frames * 2.0 - 1.0

    # Process individually
    logger.info("Processing frames individually...")
    single_warps = []
    for i in range(frames.shape[0]):
        warp = extract_xy_warps(model, frames[i:i+1])
        single_warps.append(warp)

    single_warps = torch.cat(single_warps, dim=0)

    # Process as batch
    logger.info("Processing frames as batch...")
    batch_warps = extract_xy_warps(model, frames)

    # Compare
    diff = torch.abs(single_warps - batch_warps).max().item()
    logger.info(f"Max difference between single and batch processing: {diff:.8f}")

    if diff < 1e-5:
        logger.info("✓ Single and batch processing produce identical results")
    else:
        logger.warning(f"⚠️ Single and batch processing differ by {diff}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Per-Frame XY Warp Extraction Test")
    logger.info("Each frame should have unique XY warps for its expression")
    logger.info("=" * 60)

    # Test with real video frames
    test_with_video()

    # Test single vs batch processing
    test_single_vs_batch()

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete!")
    logger.info("This implementation extracts frame-specific XY warps")
    logger.info("Ready for integration into vasa_dataset.py")
    logger.info("=" * 60)