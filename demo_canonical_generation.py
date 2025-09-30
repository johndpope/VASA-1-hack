#!/usr/bin/env python3
"""
Visual demonstration of canonical view generation.
Shows how different expressions/poses map to the same canonical view.
"""

import torch
import sys
import logging
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Add nemo to path
sys.path.insert(0, 'nemo')

# Import bridge interface
from vasa_emo_bridge_interface import create_bridge, WarpExtractionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_bridge():
    """Load model and create bridge."""
    logger.info("Loading volumetric avatar model...")

    # Load config and model
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    model = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Load weights
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        model.load_state_dict(model_dict, strict=False)

    model = model.cuda()
    model.eval()

    # Create bridge
    bridge = create_bridge("emoportraits", model)

    return bridge


def load_test_frames(num_frames=8):
    """Load test frames with different expressions."""
    video_path = Path("temp_single_video/15.mp4")

    if not video_path.exists():
        logger.error("Test video not found!")
        return None

    logger.info(f"Loading {num_frames} frames from video...")
    cap = cv2.VideoCapture(str(video_path))

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
            frames.append(frame)

    cap.release()

    if frames:
        return torch.stack(frames).cuda()
    return None


def create_canonical_demo():
    """Create comprehensive demo of canonical generation."""

    # Load model and bridge
    bridge = load_model_and_bridge()

    # Load test frames
    frames = load_test_frames(num_frames=8)
    if frames is None:
        return

    logger.info(f"Loaded {frames.shape[0]} frames with different expressions")

    # Generate canonical views for all frames
    logger.info("\n=== Generating Canonical Views ===")
    canonical_views = []

    for i, frame in enumerate(frames):
        logger.info(f"Processing frame {i+1}/{len(frames)}...")
        canonical = bridge.generate_canonical_view(
            identity_frame=frame.unsqueeze(0),
            use_identity_warps=True
        )
        canonical_views.append(canonical.squeeze(0))

    canonical_views = torch.stack(canonical_views)

    # Create visualization
    create_visualization(frames, canonical_views)

    # Analyze consistency
    analyze_canonical_consistency(canonical_views)

    # Extract and visualize warps
    logger.info("\n=== Extracting Warps to Canonical ===")
    visualize_warps_to_canonical(bridge, frames, canonical_views)


def create_visualization(source_frames, canonical_frames):
    """Create main visualization showing source -> canonical mapping."""

    n = source_frames.shape[0]
    fig = plt.figure(figsize=(20, 6))

    # Create grid
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.1)

    # Title
    fig.suptitle("Canonical View Generation: Different Expressions → Same Canonical View",
                 fontsize=16, fontweight='bold')

    # Source frames (top row)
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        img = source_frames[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Source {i+1}", fontsize=10)
        ax.axis('off')

    # Canonical views (middle row)
    for i in range(n):
        ax = fig.add_subplot(gs[1, i])
        img = canonical_frames[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Canonical {i+1}", fontsize=10)
        ax.axis('off')

    # Difference from first canonical (bottom row)
    reference_canonical = canonical_frames[0]
    for i in range(n):
        ax = fig.add_subplot(gs[2, i])

        # Compute difference
        diff = torch.abs(canonical_frames[i] - reference_canonical)
        diff_map = diff.mean(dim=0).cpu().numpy()

        # Visualize difference
        im = ax.imshow(diff_map, cmap='hot', vmin=0, vmax=0.2)
        ax.set_title(f"Diff: {diff.mean().item():.4f}", fontsize=9)
        ax.axis('off')

    # Add colorbar for difference
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.2])
    plt.colorbar(im, cax=cbar_ax, label='Difference')

    # Add text annotations
    fig.text(0.02, 0.70, "Original\nFrames", fontsize=12, weight='bold',
             rotation=90, va='center')
    fig.text(0.02, 0.40, "Canonical\nViews", fontsize=12, weight='bold',
             rotation=90, va='center')
    fig.text(0.02, 0.10, "Difference\nMaps", fontsize=12, weight='bold',
             rotation=90, va='center')

    plt.savefig("canonical_demo_main.png", dpi=150, bbox_inches='tight')
    logger.info("Saved main visualization to canonical_demo_main.png")
    plt.close()


def analyze_canonical_consistency(canonical_views):
    """Analyze how consistent the canonical views are."""

    n = canonical_views.shape[0]

    # Compute pairwise differences
    diff_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            diff = torch.abs(canonical_views[i] - canonical_views[j]).mean().item()
            diff_matrix[i, j] = diff

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Difference matrix
    im = ax1.imshow(diff_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.15)
    ax1.set_title("Pairwise Canonical View Differences", fontsize=12, weight='bold')
    ax1.set_xlabel("Canonical View Index")
    ax1.set_ylabel("Canonical View Index")

    # Add values to heatmap
    for i in range(n):
        for j in range(n):
            text = ax1.text(j, i, f'{diff_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax1, label='Mean Absolute Difference')

    # Statistics
    ax2.axis('off')
    stats_text = f"""Canonical View Consistency Analysis

    Average difference: {diff_matrix[np.triu_indices(n, k=1)].mean():.4f}
    Maximum difference: {diff_matrix.max():.4f}
    Standard deviation: {diff_matrix[np.triu_indices(n, k=1)].std():.4f}

    Interpretation:
    • Values < 0.05: Excellent consistency ✓
    • Values 0.05-0.10: Good consistency
    • Values 0.10-0.15: Moderate consistency
    • Values > 0.15: Poor consistency

    Result: {'✓ Excellent' if diff_matrix[np.triu_indices(n, k=1)].mean() < 0.05
             else '✓ Good' if diff_matrix[np.triu_indices(n, k=1)].mean() < 0.10
             else '⚠ Moderate' if diff_matrix[np.triu_indices(n, k=1)].mean() < 0.15
             else '✗ Poor'} canonical consistency achieved

    The canonical views maintain the same:
    • Identity (face structure)
    • Pose (front-facing)
    • Expression (neutral)
    """

    ax2.text(0.1, 0.5, stats_text, fontsize=11, va='center', family='monospace')

    plt.suptitle("Canonical View Consistency Analysis", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("canonical_consistency_analysis.png", dpi=150, bbox_inches='tight')
    logger.info("Saved consistency analysis to canonical_consistency_analysis.png")
    plt.close()

    # Log summary
    avg_diff = diff_matrix[np.triu_indices(n, k=1)].mean()
    logger.info(f"\nCanonical Consistency: Average difference = {avg_diff:.4f}")


def visualize_warps_to_canonical(bridge, source_frames, canonical_views):
    """Visualize the warps needed to transform expressions to canonical."""

    # Configure warp extraction
    config = WarpExtractionConfig(
        compute_xy_warps=True,
        compute_rigid_warps=True,
        compute_uv_warps=False  # Skip UV for this demo
    )

    # Extract warps for a few frames
    num_samples = min(4, source_frames.shape[0])

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))
    fig.suptitle("Warping Fields: Expression → Canonical", fontsize=14, weight='bold')

    for i in range(num_samples):
        logger.info(f"Extracting warps for frame {i+1}...")

        # Extract warps using bridge
        frame_warp = bridge.extract_warps_for_frame(
            identity_frame=source_frames[0:1],  # Use first as identity
            target_frame=source_frames[i:i+1],
            config=config
        )

        # Source frame
        img = source_frames[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i+1}", fontsize=10)
        axes[0, i].axis('off')

        # XY warp visualization
        if frame_warp.xy_warp is not None:
            xy_warp = frame_warp.xy_warp[0]  # [D, H, W, 3]
            # Take middle depth slice
            mid_depth = xy_warp.shape[0] // 2
            xy_slice = xy_warp[mid_depth].cpu().numpy()

            # Compute magnitude
            magnitude = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)

            im = axes[1, i].imshow(magnitude, cmap='viridis', vmin=0, vmax=2)
            axes[1, i].set_title(f"XY Warp\nMax: {magnitude.max():.2f}", fontsize=10)
            axes[1, i].axis('off')

        # Canonical result
        img = canonical_views[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[2, i].imshow(img)
        axes[2, i].set_title(f"Canonical", fontsize=10)
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.3, 0.5, "Source", transform=axes[0, 0].transAxes,
                    fontsize=11, weight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.3, 0.5, "XY Warp\nMagnitude", transform=axes[1, 0].transAxes,
                    fontsize=11, weight='bold', rotation=90, va='center')
    axes[2, 0].text(-0.3, 0.5, "Canonical", transform=axes[2, 0].transAxes,
                    fontsize=11, weight='bold', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig("canonical_warp_visualization.png", dpi=150, bbox_inches='tight')
    logger.info("Saved warp visualization to canonical_warp_visualization.png")
    plt.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CANONICAL VIEW GENERATION DEMO")
    logger.info("=" * 60)

    create_canonical_demo()

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete! Check generated images:")
    logger.info("  - canonical_demo_main.png")
    logger.info("  - canonical_consistency_analysis.png")
    logger.info("  - canonical_warp_visualization.png")
    logger.info("=" * 60)