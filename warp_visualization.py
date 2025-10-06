"""
UV Warp Visualization for VASA-1

Creates "warp candles" similar to expression candles to debug 3D warping behavior.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
from logger import logger


def visualize_uv_warps(uv_warps: torch.Tensor, target_uv_warps: torch.Tensor = None,
                       step: int = 0, wandb_logger=None, save_path: str = None):
    """
    Create comprehensive UV warp visualization with candles similar to expression candles.

    Args:
        uv_warps: [B, T, 16, 64, 64, 3] Predicted UV warp fields
        target_uv_warps: [B, T, 16, 64, 64, 3] Target UV warp fields (optional)
        step: Training step
        wandb_logger: WandB logger instance
        save_path: Path to save figure (optional)

    Returns:
        matplotlib figure
    """
    B, T, D, H, W, C = uv_warps.shape

    # Move to CPU for visualization
    uv_warps_np = uv_warps.detach().cpu()
    target_np = target_uv_warps.detach().cpu() if target_uv_warps is not None else None

    # Compute statistics over time
    # 1. Magnitude: How strong are the warps?
    warp_magnitude = torch.norm(uv_warps_np, dim=-1).mean(dim=[-3, -2, -1])  # [B, T]

    # 2. Variance: How spatially diverse are the warps?
    warp_variance = uv_warps_np.reshape(B, T, -1).var(dim=-1)  # [B, T]

    # 3. Temporal change: How much do warps change frame-to-frame?
    warp_temporal_diff = torch.diff(uv_warps_np, dim=1).abs().mean(dim=[-4, -3, -2, -1])  # [B, T-1]

    # 4. X, Y, Z components separately
    warp_x = uv_warps_np[..., 0].mean(dim=[-3, -2, -1])  # [B, T]
    warp_y = uv_warps_np[..., 1].mean(dim=[-3, -2, -1])  # [B, T]
    warp_z = uv_warps_np[..., 2].mean(dim=[-3, -2, -1])  # [B, T]

    # If target provided, compute same stats
    if target_np is not None:
        target_magnitude = torch.norm(target_np, dim=-1).mean(dim=[-3, -2, -1])
        target_variance = target_np.reshape(B, T, -1).var(dim=-1)
        target_temporal_diff = torch.diff(target_np, dim=1).abs().mean(dim=[-4, -3, -2, -1])

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    if target_np is not None:
        # With target: 3x2 grid
        gs = plt.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    else:
        # Without target: 3x1 grid
        gs = plt.GridSpec(3, 1, hspace=0.3)

    # ========================================================================
    # Plot 1: Warp Magnitude Over Time (Main "candles")
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0] if target_np is not None else gs[0])

    frames = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, min(B, 4)))

    for b in range(min(B, 4)):  # Show first 4 batches
        ax1.plot(frames, warp_magnitude[b].numpy(),
                color=colors[b], label=f'Batch {b}', alpha=0.8, linewidth=2)

        # Add target if available
        if target_np is not None:
            ax1.plot(frames, target_magnitude[b].numpy(),
                    color=colors[b], linestyle='--', alpha=0.5, linewidth=1.5,
                    label=f'Target {b}')

    ax1.set_title('UV Warp Magnitude "Candles" (L2 Norm)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Warp Magnitude', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, T-1)

    # Add stats annotation
    mean_mag = warp_magnitude.mean().item()
    std_mag = warp_magnitude.std().item()
    ax1.text(0.02, 0.98, f'Mean: {mean_mag:.4f}\nStd: {std_mag:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================================================
    # Plot 2: Warp Variance Over Time (Spatial Diversity)
    # ========================================================================
    if target_np is not None:
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax2 = fig.add_subplot(gs[1])

    for b in range(min(B, 4)):
        ax2.plot(frames, warp_variance[b].numpy(),
                color=colors[b], label=f'Batch {b}', alpha=0.8, linewidth=2)

        if target_np is not None:
            ax2.plot(frames, target_variance[b].numpy(),
                    color=colors[b], linestyle='--', alpha=0.5, linewidth=1.5)

    ax2.set_title('UV Warp Spatial Variance (Diversity)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, T-1)

    # ========================================================================
    # Plot 3: Temporal Change (Frame-to-Frame Dynamics)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0] if target_np is not None else gs[2])

    frames_diff = np.arange(T-1)
    for b in range(min(B, 4)):
        ax3.plot(frames_diff, warp_temporal_diff[b].numpy(),
                color=colors[b], label=f'Batch {b}', alpha=0.8, linewidth=2)

        if target_np is not None:
            ax3.plot(frames_diff, target_temporal_diff[b].numpy(),
                    color=colors[b], linestyle='--', alpha=0.5, linewidth=1.5)

    ax3.set_title('Temporal Change (Frame-to-Frame Warp Diff)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('|Δ Warp|', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, T-2)

    # ========================================================================
    # Plot 4: X/Y/Z Components Breakdown
    # ========================================================================
    if target_np is not None:
        ax4 = fig.add_subplot(gs[1, 1])

        # Show first batch's XYZ components
        ax4.plot(frames, warp_x[0].numpy(), color='red', label='X', linewidth=2)
        ax4.plot(frames, warp_y[0].numpy(), color='green', label='Y', linewidth=2)
        ax4.plot(frames, warp_z[0].numpy(), color='blue', label='Z', linewidth=2)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

        ax4.set_title('Warp Components (Batch 0)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Frame', fontsize=12)
        ax4.set_ylabel('Mean Component Value', fontsize=12)
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(0, T-1)

    # ========================================================================
    # Plot 5: Heatmap of Warp Field (Middle Frame, Middle Depth)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 0] if target_np is not None else None)
    if ax5 is not None:
        mid_frame = T // 2
        mid_depth = D // 2

        # Compute warp norm at middle frame/depth
        warp_norm = torch.norm(uv_warps_np[0, mid_frame, mid_depth], dim=-1)  # [H, W]

        im = ax5.imshow(warp_norm.numpy(), cmap='viridis', aspect='auto')
        ax5.set_title(f'Warp Magnitude Heatmap (Frame={mid_frame}, Depth={mid_depth})',
                     fontsize=14, fontweight='bold')
        ax5.set_xlabel('Width', fontsize=12)
        ax5.set_ylabel('Height', fontsize=12)
        plt.colorbar(im, ax=ax5, label='Magnitude')

    # ========================================================================
    # Plot 6: Pred vs Target Comparison (if target available)
    # ========================================================================
    if target_np is not None:
        ax6 = fig.add_subplot(gs[2, 1])

        # Scatter: Predicted vs Target magnitude
        pred_flat = warp_magnitude.numpy().flatten()
        target_flat = target_magnitude.numpy().flatten()

        ax6.scatter(target_flat, pred_flat, alpha=0.5, s=20)

        # Add perfect prediction line
        min_val = min(pred_flat.min(), target_flat.min())
        max_val = max(pred_flat.max(), target_flat.max())
        ax6.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Match')

        ax6.set_title('Predicted vs Target Magnitude', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Target Magnitude', fontsize=12)
        ax6.set_ylabel('Predicted Magnitude', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--')

        # Add correlation coefficient
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {corr:.4f}',
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'UV Warp Visualization (Step {step})', fontsize=16, fontweight='bold')

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved warp visualization to {save_path}")

    # Log to WandB
    if wandb_logger is not None:
        wandb_logger.log({
            'warp_visualization/candles': wandb.Image(fig),
            'warp_stats/mean_magnitude': warp_magnitude.mean().item(),
            'warp_stats/std_magnitude': warp_magnitude.std().item(),
            'warp_stats/mean_variance': warp_variance.mean().item(),
            'warp_stats/mean_temporal_change': warp_temporal_diff.mean().item(),
        }, step=step)

        if target_np is not None:
            corr = np.corrcoef(pred_flat, target_flat)[0, 1]
            wandb_logger.log({
                'warp_stats/pred_target_correlation': corr,
                'warp_stats/magnitude_mse': ((warp_magnitude - target_magnitude) ** 2).mean().item()
            }, step=step)

    return fig


def log_warp_statistics(outputs: dict, targets: dict, step: int, wandb_logger=None):
    """
    Quick warp statistics logging without full visualization.

    Args:
        outputs: Model outputs dict containing 'uv_warps'
        targets: Targets dict containing 'uv_warps'
        step: Training step
        wandb_logger: WandB logger
    """
    if 'uv_warps' not in outputs or 'uv_warps' not in targets:
        return

    pred_warps = outputs['uv_warps']  # [B, T, 16, 64, 64, 3]
    target_warps = targets['uv_warps']

    # Compute key stats
    pred_magnitude = torch.norm(pred_warps, dim=-1).mean()
    target_magnitude = torch.norm(target_warps, dim=-1).mean()

    pred_variance = pred_warps.reshape(pred_warps.shape[0], pred_warps.shape[1], -1).var(dim=-1).mean()
    target_variance = target_warps.reshape(target_warps.shape[0], target_warps.shape[1], -1).var(dim=-1).mean()

    mse = ((pred_warps - target_warps) ** 2).mean()

    if wandb_logger:
        wandb_logger.log({
            'warp_quick_stats/pred_magnitude': pred_magnitude.item(),
            'warp_quick_stats/target_magnitude': target_magnitude.item(),
            'warp_quick_stats/pred_variance': pred_variance.item(),
            'warp_quick_stats/target_variance': target_variance.item(),
            'warp_quick_stats/mse': mse.item(),
        }, step=step)
