"""
Action Unit (AU) Visualization for VASA-1 Training

Creates 16-subplot visualization showing predicted vs ground truth AU intensities
over time (queries). Each AU is displayed in a separate subplot with:
- Green line: Ground truth intensities
- Blue line: Predicted intensities
- Error metric: Mean absolute error per AU
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from typing import Optional, List
import seaborn as sns

# Import AU names from extractor
from au_extractor import AU_NAMES


def create_au_visualization(
    au_gt: torch.Tensor,
    au_pred: torch.Tensor,
    au_names: List[str] = None,
    window_idx: int = 0,
    audio_filename: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive AU visualization with 16 subplots.

    Args:
        au_gt: Ground truth AU intensities [num_queries, 16]
        au_pred: Predicted AU intensities [num_queries, 16]
        au_names: List of 16 AU names (default: AU_NAMES from extractor)
        window_idx: Window index for title
        audio_filename: Optional audio filename for title

    Returns:
        Matplotlib figure with 16 subplots
    """
    if au_names is None:
        au_names = AU_NAMES

    # Convert to numpy for plotting
    if isinstance(au_gt, torch.Tensor):
        au_gt = au_gt.cpu().numpy()
    if isinstance(au_pred, torch.Tensor):
        au_pred = au_pred.cpu().numpy()

    # Validate shapes
    assert au_gt.shape == au_pred.shape, \
        f"Shape mismatch: gt={au_gt.shape}, pred={au_pred.shape}"
    assert au_gt.shape[-1] == 16, \
        f"Expected 16 AUs, got {au_gt.shape[-1]}"

    num_queries = au_gt.shape[0]
    queries = np.arange(num_queries)

    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Color scheme
    gt_color = '#2ecc71'  # Green
    pred_color = '#3498db'  # Blue
    error_color = '#e74c3c'  # Red

    # Compute overall statistics
    mae_per_au = np.abs(au_gt - au_pred).mean(axis=0)  # [16]
    overall_mae = mae_per_au.mean()
    overall_corr = np.corrcoef(au_gt.flatten(), au_pred.flatten())[0, 1]

    # Plot each AU in a subplot
    for i in range(16):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        # Plot GT and predictions
        ax.plot(queries, au_gt[:, i], color=gt_color, linewidth=2.5,
                marker='o', markersize=6, label='Ground Truth', alpha=0.8)
        ax.plot(queries, au_pred[:, i], color=pred_color, linewidth=2.5,
                marker='x', markersize=8, label='Predicted', alpha=0.8, linestyle='--')

        # Compute per-AU metrics
        mae = mae_per_au[i]
        corr = np.corrcoef(au_gt[:, i], au_pred[:, i])[0, 1] if au_gt[:, i].std() > 0 else 0.0

        # Styling
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Query Index', fontsize=10, fontweight='bold')
        ax.set_ylabel('Intensity [0,1]', fontsize=10, fontweight='bold')

        # AU name formatting: "AU12_Lip_Corner_Puller" -> "AU12: Lip Corner Puller"
        au_name_formatted = au_names[i].replace('_', ' ')
        if au_name_formatted.startswith('AU'):
            parts = au_name_formatted.split(' ', 1)
            if len(parts) == 2:
                au_name_formatted = f"{parts[0]}: {parts[1]}"

        ax.set_title(f'{au_name_formatted}\nMAE: {mae:.4f} | Corr: {corr:.3f}',
                     fontsize=11, fontweight='bold', pad=10)

        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Highlight high error AUs with red background
        if mae > 0.15:
            ax.set_facecolor('#ffebee')  # Light red background

    # Main title
    title_parts = []
    if audio_filename:
        title_parts.append(f'Audio: {audio_filename}')
    title_parts.append(f'Window {window_idx}')

    main_title = ' | '.join(title_parts)

    fig.suptitle(
        f'Action Unit Predictions - {main_title}\n'
        f'Overall MAE: {overall_mae:.4f} | Overall Correlation: {overall_corr:.3f}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Add color-coded summary box
    fig.text(0.5, 0.01,
             f'🟢 Green: Ground Truth | 🔵 Blue: Predicted | '
             f'🔴 Red Background: High Error (MAE > 0.15)',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    return fig


def create_au_summary_visualization(
    au_gt: torch.Tensor,
    au_pred: torch.Tensor,
    au_names: List[str] = None,
    window_idx: int = 0
) -> plt.Figure:
    """
    Create compact summary visualization with AU activation heatmaps.

    Args:
        au_gt: Ground truth AU intensities [num_queries, 16]
        au_pred: Predicted AU intensities [num_queries, 16]
        au_names: List of 16 AU names
        window_idx: Window index for title

    Returns:
        Matplotlib figure with heatmaps
    """
    if au_names is None:
        au_names = AU_NAMES

    # Convert to numpy
    if isinstance(au_gt, torch.Tensor):
        au_gt = au_gt.cpu().numpy()
    if isinstance(au_pred, torch.Tensor):
        au_pred = au_pred.cpu().numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Format AU names for display
    au_labels = [name.replace('_', '\n').replace('AU', 'AU\n') for name in au_names]

    # Plot 1: Ground Truth Heatmap
    sns.heatmap(au_gt.T, ax=axes[0], cmap='Greens', vmin=0, vmax=1,
                cbar_kws={'label': 'Intensity'}, yticklabels=au_labels,
                xticklabels=[f'Q{i}' for i in range(au_gt.shape[0])])
    axes[0].set_title('Ground Truth AU Intensities', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Query Index', fontsize=12)
    axes[0].set_ylabel('Action Units', fontsize=12)

    # Plot 2: Predicted Heatmap
    sns.heatmap(au_pred.T, ax=axes[1], cmap='Blues', vmin=0, vmax=1,
                cbar_kws={'label': 'Intensity'}, yticklabels=au_labels,
                xticklabels=[f'Q{i}' for i in range(au_pred.shape[0])])
    axes[1].set_title('Predicted AU Intensities', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Query Index', fontsize=12)
    axes[1].set_ylabel('Action Units', fontsize=12)

    # Plot 3: Error Heatmap
    error = np.abs(au_gt - au_pred)
    sns.heatmap(error.T, ax=axes[2], cmap='Reds', vmin=0, vmax=0.5,
                cbar_kws={'label': 'Absolute Error'}, yticklabels=au_labels,
                xticklabels=[f'Q{i}' for i in range(error.shape[0])])
    axes[2].set_title('Absolute Error (|GT - Pred|)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Query Index', fontsize=12)
    axes[2].set_ylabel('Action Units', fontsize=12)

    plt.suptitle(f'AU Heatmap Summary - Window {window_idx}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_au_temporal_plot(
    au_gt: torch.Tensor,
    au_pred: torch.Tensor,
    au_indices: List[int] = None,
    au_names: List[str] = None,
    window_idx: int = 0
) -> plt.Figure:
    """
    Create temporal plot for specific AUs of interest.

    Args:
        au_gt: Ground truth AU intensities [num_queries, 16]
        au_pred: Predicted AU intensities [num_queries, 16]
        au_indices: List of AU indices to plot (default: mouth AUs)
        au_names: List of 16 AU names
        window_idx: Window index for title

    Returns:
        Matplotlib figure
    """
    if au_names is None:
        au_names = AU_NAMES

    if au_indices is None:
        # Default: Plot mouth AUs (important for lip sync)
        # AU12, AU15, AU20, AU25, AU26, AU27
        au_indices = [8, 9, 11, 13, 14, 15]

    # Convert to numpy
    if isinstance(au_gt, torch.Tensor):
        au_gt = au_gt.cpu().numpy()
    if isinstance(au_pred, torch.Tensor):
        au_pred = au_pred.cpu().numpy()

    num_queries = au_gt.shape[0]
    queries = np.arange(num_queries)

    # Create figure
    fig, axes = plt.subplots(len(au_indices), 1, figsize=(12, 3 * len(au_indices)),
                            sharex=True)

    if len(au_indices) == 1:
        axes = [axes]

    for i, au_idx in enumerate(au_indices):
        ax = axes[i]

        # Plot GT and predictions
        ax.plot(queries, au_gt[:, au_idx], 'o-', color='#2ecc71',
                linewidth=2.5, markersize=8, label='Ground Truth', alpha=0.8)
        ax.plot(queries, au_pred[:, au_idx], 'x--', color='#3498db',
                linewidth=2.5, markersize=10, label='Predicted', alpha=0.8)

        # Fill area between curves
        ax.fill_between(queries, au_gt[:, au_idx], au_pred[:, au_idx],
                        alpha=0.2, color='gray')

        # Metrics
        mae = np.abs(au_gt[:, au_idx] - au_pred[:, au_idx]).mean()
        corr = np.corrcoef(au_gt[:, au_idx], au_pred[:, au_idx])[0, 1] \
               if au_gt[:, au_idx].std() > 0 else 0.0

        # Styling
        au_name = au_names[au_idx].replace('_', ' ')
        ax.set_ylabel('Intensity [0,1]', fontsize=11, fontweight='bold')
        ax.set_title(f'{au_name} | MAE: {mae:.4f} | Correlation: {corr:.3f}',
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel('Query Index', fontsize=12, fontweight='bold')

    plt.suptitle(f'Key AU Temporal Evolution - Window {window_idx}',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Test visualization with dummy data
    import torch

    # Create dummy data
    num_queries = 8
    au_gt = torch.rand(num_queries, 16) * 0.8  # Random GT [0, 0.8]
    au_pred = au_gt + torch.randn(num_queries, 16) * 0.1  # Add noise
    au_pred = torch.clamp(au_pred, 0, 1)

    # Test main visualization
    fig1 = create_au_visualization(
        au_gt=au_gt,
        au_pred=au_pred,
        window_idx=42,
        audio_filename="test_audio.wav"
    )
    fig1.savefig('/tmp/au_test_main.png', dpi=150, bbox_inches='tight')
    print("✅ Main visualization saved to /tmp/au_test_main.png")

    # Test summary visualization
    fig2 = create_au_summary_visualization(
        au_gt=au_gt,
        au_pred=au_pred,
        window_idx=42
    )
    fig2.savefig('/tmp/au_test_summary.png', dpi=150, bbox_inches='tight')
    print("✅ Summary visualization saved to /tmp/au_test_summary.png")

    # Test temporal plot
    fig3 = create_au_temporal_plot(
        au_gt=au_gt,
        au_pred=au_pred,
        window_idx=42
    )
    fig3.savefig('/tmp/au_test_temporal.png', dpi=150, bbox_inches='tight')
    print("✅ Temporal visualization saved to /tmp/au_test_temporal.png")

    plt.close('all')
