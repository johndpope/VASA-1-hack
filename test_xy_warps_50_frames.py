#!/usr/bin/env python3
"""
Test XY warp extraction across 50 frames from a video.
Visualizes how warps change with different facial expressions throughout the video.
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
from tqdm import tqdm

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_xy_warps(emo_model, frames, use_modnet_mask=False):
    """
    Extract XY warps for a batch of frames using the volumetric avatar model.

    Args:
        emo_model: The loaded VolumetricAvatar model instance.
        frames: Input frames tensor of shape [N, 3, H, W].
        use_modnet_mask: If True, use MODNet for masking; else use face_idt.

    Returns:
        torch.Tensor: XY warps of shape [N, D, S, S, 3]
    """
    # Ensure model is in eval mode
    emo_model.eval()

    # Handle input shape
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    elif frames.dim() != 4:
        raise ValueError(f"Expected frames shape [N,3,H,W] or [3,H,W], got {frames.shape}")

    N, C, H, W = frames.shape
    device = frames.device

    with torch.no_grad():
        # Compute theta using head pose regressor
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
            'target_img': frames,
            'source_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
            'target_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
            'source_mask': mask,
            'target_mask': mask,
            'idt_embed': idt_embed
        }

        # Compute expression embedding
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

        # Create warp embed dict
        c = warp_embed_head.shape[1]
        source_warp_embed_dict = {
            'orig': warp_embed_head.view(N, c, emo_model.embed_size ** 2),
            'orig_d': warp_embed_head.view(N, c, emo_model.embed_size ** 2).detach(),
            'ada_v': source_pose_embed,
            'idt_embed': idt_embed
        }

        # Generate XY warp
        xy_gen_warp, _ = emo_model.xy_generator_nw(source_warp_embed_dict)

        # Apply resizing if needed
        if emo_model.resize_warp:
            stride = emo_model.warp_resize_stride
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


def visualize_xy_warps_grid(xy_warps_list, frame_indices, save_path="xy_warps_50_frames_grid.png"):
    """Create a grid visualization of XY warps across multiple frames."""

    n_frames = len(xy_warps_list)
    n_cols = 10  # 10 columns
    n_rows = (n_frames + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else axes

    # Take middle depth slice
    mid_depth = xy_warps_list[0].shape[1] // 2

    # Calculate global min/max for consistent colormap
    all_magnitudes = []
    for xy_warps in xy_warps_list:
        xy_slice = xy_warps[0, mid_depth].cpu().numpy()
        magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
        all_magnitudes.append(magnitude)

    vmin = min(m.min() for m in all_magnitudes)
    vmax = max(m.max() for m in all_magnitudes)

    for i in range(n_frames):
        xy_slice = xy_warps_list[i][0, mid_depth].cpu().numpy()
        magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)

        im = axes[i].imshow(magnitude, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'F{frame_indices[i]}', fontsize=8)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.08)

    plt.suptitle('XY Warp Magnitudes Across 50 Frames', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved grid visualization to {save_path}")
    plt.close()


def visualize_xy_warps_temporal(xy_warps_list, frame_indices, save_path="xy_warps_50_frames_temporal.png"):
    """Create temporal visualization showing warp evolution over time."""

    n_frames = len(xy_warps_list)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))

    # Take middle depth slice
    mid_depth = xy_warps_list[0].shape[1] // 2

    # Extract statistics over time
    means = []
    stds = []
    maxs = []
    x_means = []
    y_means = []

    for xy_warps in xy_warps_list:
        xy_slice = xy_warps[0, mid_depth].cpu().numpy()
        means.append(np.mean(np.abs(xy_slice)))
        stds.append(np.std(xy_slice))
        maxs.append(np.max(np.abs(xy_slice)))
        x_means.append(np.mean(xy_slice[..., 0]))
        y_means.append(np.mean(xy_slice[..., 1]))

    # Plot mean absolute warp
    axes[0].plot(frame_indices, means, 'b-', linewidth=2, label='Mean |Warp|')
    axes[0].fill_between(frame_indices,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.3)
    axes[0].set_ylabel('Mean Absolute Warp')
    axes[0].set_title('XY Warp Statistics Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot X and Y components separately
    axes[1].plot(frame_indices, x_means, 'r-', linewidth=2, label='X Component')
    axes[1].plot(frame_indices, y_means, 'g-', linewidth=2, label='Y Component')
    axes[1].set_ylabel('Mean Component Value')
    axes[1].set_title('X and Y Warp Components')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot max absolute warp
    axes[2].plot(frame_indices, maxs, 'purple', linewidth=2)
    axes[2].set_ylabel('Max Absolute Warp')
    axes[2].set_xlabel('Frame Index')
    axes[2].set_title('Maximum Warp Magnitude')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved temporal visualization to {save_path}")
    plt.close()


def visualize_xy_warps_comparison(xy_warps_list, frame_indices, save_path="xy_warps_50_frames_comparison.png"):
    """Create detailed comparison of selected frames."""

    # Select frames to compare (evenly spaced)
    selected_indices = [0, 12, 25, 37, 49]
    selected_indices = [i for i in selected_indices if i < len(xy_warps_list)]

    n_selected = len(selected_indices)
    fig, axes = plt.subplots(3, n_selected, figsize=(n_selected * 3, 9))

    # Take middle depth slice
    mid_depth = xy_warps_list[0].shape[1] // 2

    for col, idx in enumerate(selected_indices):
        xy_slice = xy_warps_list[idx][0, mid_depth].cpu().numpy()
        frame_num = frame_indices[idx]

        # X component
        im0 = axes[0, col].imshow(xy_slice[..., 0], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[0, col].set_title(f'Frame {frame_num}\nX Component', fontsize=10)
        axes[0, col].axis('off')

        # Y component
        im1 = axes[1, col].imshow(xy_slice[..., 1], cmap='RdBu_r', vmin=-2, vmax=2)
        axes[1, col].set_title(f'Y Component', fontsize=10)
        axes[1, col].axis('off')

        # Magnitude
        magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
        im2 = axes[2, col].imshow(magnitude, cmap='viridis', vmin=0, vmax=3)
        axes[2, col].set_title(f'Magnitude\n(max: {magnitude.max():.2f})', fontsize=10)
        axes[2, col].axis('off')

    # Add colorbars
    fig.colorbar(im0, ax=axes[0, :], fraction=0.02, pad=0.02)
    fig.colorbar(im1, ax=axes[1, :], fraction=0.02, pad=0.02)
    fig.colorbar(im2, ax=axes[2, :], fraction=0.02, pad=0.02)

    plt.suptitle('XY Warps Comparison - Selected Frames', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison visualization to {save_path}")
    plt.close()


def compute_warp_statistics(xy_warps_list):
    """Compute and log statistics about the warps."""

    logger.info("\n=== XY Warp Statistics Across 50 Frames ===")

    # Overall statistics
    all_warps = torch.cat(xy_warps_list, dim=0)
    overall_mean = all_warps.mean().item()
    overall_std = all_warps.std().item()
    overall_max = all_warps.abs().max().item()

    logger.info(f"Overall statistics:")
    logger.info(f"  Mean: {overall_mean:.6f}")
    logger.info(f"  Std: {overall_std:.6f}")
    logger.info(f"  Max absolute: {overall_max:.6f}")

    # Compute inter-frame differences
    differences = []
    for i in range(len(xy_warps_list) - 1):
        diff = torch.abs(xy_warps_list[i+1] - xy_warps_list[i])
        differences.append(diff.mean().item())

    logger.info(f"\nInter-frame differences:")
    logger.info(f"  Mean difference: {np.mean(differences):.6f}")
    logger.info(f"  Max difference: {np.max(differences):.6f}")
    logger.info(f"  Min difference: {np.min(differences):.6f}")

    # Check for variation
    variation_threshold = 1e-6
    num_identical = sum(1 for d in differences if d < variation_threshold)

    if num_identical > len(differences) * 0.9:
        logger.warning(f"⚠️  {num_identical}/{len(differences)} frame pairs have nearly identical warps!")
    else:
        logger.info(f"✓ Good variation: only {num_identical}/{len(differences)} frame pairs are nearly identical")


def test_with_50_frames():
    """Test XY warp extraction with 50 frames from a video."""

    # Load model
    model = load_volumetric_model()

    # Load test video
    video_path = Path("temp_single_video/15.mp4")
    if not video_path.exists():
        logger.error("Test video not found!")
        return

    logger.info(f"Loading 50 frames from {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video has {total_frames} total frames")

    # Select 50 frames evenly spaced throughout the video
    num_frames = min(50, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    actual_indices = []

    for idx in tqdm(frame_indices, desc="Loading frames"):
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
            actual_indices.append(idx)

    cap.release()

    if len(frames) < 2:
        logger.error("Could not load enough frames")
        return

    logger.info(f"Loaded {len(frames)} frames")

    logger.info("\n=== Extracting XY Warps for 50 Frames ===")

    # Process frames in batches to avoid memory issues
    batch_size = 5
    xy_warps_list = []

    for i in tqdm(range(0, len(frames), batch_size), desc="Extracting XY warps"):
        batch_frames = frames[i:i+batch_size]
        batch_tensor = torch.stack(batch_frames).cuda()

        # Extract XY warps for batch
        xy_warps_batch = extract_xy_warps(model, batch_tensor)

        # Store each frame's warps separately
        for j in range(xy_warps_batch.shape[0]):
            xy_warps_list.append(xy_warps_batch[j:j+1])

    logger.info(f"Extracted XY warps for {len(xy_warps_list)} frames")

    # Compute and log statistics
    compute_warp_statistics(xy_warps_list)

    # Create visualizations
    logger.info("\n=== Creating Visualizations ===")

    # Grid visualization
    visualize_xy_warps_grid(xy_warps_list, actual_indices)

    # Temporal evolution
    visualize_xy_warps_temporal(xy_warps_list, actual_indices)

    # Detailed comparison
    visualize_xy_warps_comparison(xy_warps_list, actual_indices)

    logger.info("\n=== XY Warp Extraction for 50 Frames Complete ===")

    return xy_warps_list, actual_indices


def create_animated_visualization(xy_warps_list, frame_indices, save_path="xy_warps_animated.gif"):
    """Create an animated GIF showing warp evolution."""

    import matplotlib.animation as animation

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Take middle depth slice
    mid_depth = xy_warps_list[0].shape[1] // 2

    # Initialize plots
    xy_slice = xy_warps_list[0][0, mid_depth].cpu().numpy()

    im0 = axes[0].imshow(xy_slice[..., 0], cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0].set_title('X Component')
    axes[0].axis('off')

    im1 = axes[1].imshow(xy_slice[..., 1], cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title('Y Component')
    axes[1].axis('off')

    magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
    im2 = axes[2].imshow(magnitude, cmap='viridis', vmin=0, vmax=3)
    axes[2].set_title('Magnitude')
    axes[2].axis('off')

    # Add colorbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    def update(frame_num):
        xy_slice = xy_warps_list[frame_num][0, mid_depth].cpu().numpy()

        im0.set_data(xy_slice[..., 0])
        im1.set_data(xy_slice[..., 1])

        magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
        im2.set_data(magnitude)

        fig.suptitle(f'XY Warps - Frame {frame_indices[frame_num]}', fontsize=14, weight='bold')

        return [im0, im1, im2]

    ani = animation.FuncAnimation(fig, update, frames=len(xy_warps_list),
                                 interval=100, blit=True)

    # Save as GIF
    writer = animation.PillowWriter(fps=10)
    ani.save(save_path, writer=writer)
    logger.info(f"Saved animated visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("XY Warp Extraction Test - 50 Frames")
    logger.info("Testing frame-specific warp extraction across entire video")
    logger.info("=" * 60)

    # Run the test
    xy_warps_list, frame_indices = test_with_50_frames()

    # Optionally create animated visualization
    if xy_warps_list:
        logger.info("\nCreating animated visualization...")
        create_animated_visualization(xy_warps_list, frame_indices)

    logger.info("\n" + "=" * 60)
    logger.info("Test Complete!")
    logger.info("Check the generated visualizations:")
    logger.info("  - xy_warps_50_frames_grid.png: Grid of all 50 frames")
    logger.info("  - xy_warps_50_frames_temporal.png: Temporal evolution")
    logger.info("  - xy_warps_50_frames_comparison.png: Detailed comparison")
    logger.info("  - xy_warps_animated.gif: Animated visualization")
    logger.info("=" * 60)