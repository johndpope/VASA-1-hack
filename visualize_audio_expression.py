#!/usr/bin/env python3
"""Visualization for audio features (wav2vec) and their predicted expressions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_audio_expression_visualization(
    audio_features: torch.Tensor,
    target_expression: torch.Tensor,
    predicted_expression: torch.Tensor,
    window_idx: int,
    save_path: Optional[Path] = None,
    audio_reduce_to: int = 32,
    expr_reduce_to: int = 32
) -> plt.Figure:
    """
    Create visualization showing audio features and resulting expressions.
    
    Args:
        audio_features: Wav2vec features [B, T, 768] or [T, 768]
        target_expression: Target expression tensor [B, T, 128] or [T, 128]
        predicted_expression: Predicted expression tensor [B, T, 128] or [T, 128]
        window_idx: Index of the current window
        save_path: Optional path to save the figure
        audio_reduce_to: Number of audio dimensions to reduce to (default 32)
        expr_reduce_to: Number of expression dimensions to reduce to (default 32)
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if audio_features.dim() == 3:
        audio_features = audio_features[0]  # [T, 768]
    if target_expression.dim() == 3:
        target_expression = target_expression[0]  # [T, 128]
    if predicted_expression.dim() == 3:
        predicted_expression = predicted_expression[0]  # [T, 128]
    
    # Convert to numpy
    audio = audio_features.detach().cpu().numpy()
    target = target_expression.detach().cpu().numpy()
    predicted = predicted_expression.detach().cpu().numpy()
    
    # Get dimensions
    T = audio.shape[0]
    audio_dim = audio.shape[1]
    expr_dim = target.shape[1]
    
    # Reduce audio dimension (768 -> 32)
    if audio_dim > audio_reduce_to:
        group_size = audio_dim // audio_reduce_to
        audio_reduced = audio.reshape(T, audio_reduce_to, group_size).mean(axis=2)
    else:
        audio_reduced = audio
        audio_reduce_to = audio_dim
    
    # Reduce expression dimension (128 -> 32)
    if expr_dim > expr_reduce_to:
        group_size = expr_dim // expr_reduce_to
        target_reduced = target.reshape(T, expr_reduce_to, group_size).mean(axis=2)
        predicted_reduced = predicted.reshape(T, expr_reduce_to, group_size).mean(axis=2)
    else:
        target_reduced = target
        predicted_reduced = predicted
        expr_reduce_to = expr_dim
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.5, 1, 1, 0.5], hspace=0.3, wspace=0.2)
    
    # === Top row: Audio Features (Wav2Vec) ===
    ax_audio = fig.add_subplot(gs[0, :])
    
    # Transpose for candle display
    audio_display = audio_reduced.T  # [32, 50]
    
    # Normalize for better visualization
    audio_norm = (audio_display - audio_display.mean()) / (audio_display.std() + 1e-8)
    
    im_audio = ax_audio.imshow(
        audio_norm,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    ax_audio.set_title(f'Window {window_idx} - Wav2Vec Audio Features (reduced to {audio_reduce_to} dims)', fontsize=14, fontweight='bold')
    ax_audio.set_xlabel('Frame Number', fontsize=12)
    ax_audio.set_ylabel('Audio Feature Dim', fontsize=12)
    
    # Set ticks
    ax_audio.set_xticks(np.arange(0, T, 5))
    ax_audio.set_xticklabels(np.arange(0, T, 5))
    ax_audio.set_yticks(np.arange(0, audio_reduce_to, 4))
    ax_audio.set_yticklabels(np.arange(0, audio_reduce_to, 4))
    
    # Add colorbar
    cbar_audio = plt.colorbar(im_audio, ax=ax_audio, fraction=0.046, pad=0.04)
    cbar_audio.set_label('Normalized Value', rotation=270, labelpad=15)
    
    # Add grid
    ax_audio.grid(True, alpha=0.2, linewidth=0.5, color='white')
    
    # === Second row: Target Expression ===
    ax_target = fig.add_subplot(gs[1, :])
    
    target_display = target_reduced.T  # [32, 50]
    
    # Use same scale for target and predicted
    vmin = min(target_display.min(), predicted_reduced.T.min())
    vmax = max(target_display.max(), predicted_reduced.T.max())
    
    im_target = ax_target.imshow(
        target_display,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    ax_target.set_title(f'Target Expression (reduced to {expr_reduce_to} dims)', fontsize=12)
    ax_target.set_xlabel('Frame Number', fontsize=11)
    ax_target.set_ylabel('Expression Dim', fontsize=11)
    
    ax_target.set_xticks(np.arange(0, T, 5))
    ax_target.set_xticklabels(np.arange(0, T, 5))
    ax_target.set_yticks(np.arange(0, expr_reduce_to, 4))
    ax_target.set_yticklabels(np.arange(0, expr_reduce_to, 4))
    
    cbar_target = plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
    cbar_target.set_label('Value', rotation=270, labelpad=15)
    
    ax_target.grid(True, alpha=0.2, linewidth=0.5)
    
    # === Third row: Predicted Expression ===
    ax_pred = fig.add_subplot(gs[2, :])
    
    predicted_display = predicted_reduced.T  # [32, 50]
    
    im_pred = ax_pred.imshow(
        predicted_display,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    ax_pred.set_title(f'Predicted Expression from Audio (reduced to {expr_reduce_to} dims)', fontsize=12)
    ax_pred.set_xlabel('Frame Number', fontsize=11)
    ax_pred.set_ylabel('Expression Dim', fontsize=11)
    
    ax_pred.set_xticks(np.arange(0, T, 5))
    ax_pred.set_xticklabels(np.arange(0, T, 5))
    ax_pred.set_yticks(np.arange(0, expr_reduce_to, 4))
    ax_pred.set_yticklabels(np.arange(0, expr_reduce_to, 4))
    
    cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    cbar_pred.set_label('Value', rotation=270, labelpad=15)
    
    ax_pred.grid(True, alpha=0.2, linewidth=0.5)
    
    # === Fourth row: Statistics ===
    ax_stats1 = fig.add_subplot(gs[3, 0])
    ax_stats2 = fig.add_subplot(gs[3, 1])
    
    # Audio statistics over time
    audio_energy = np.mean(np.abs(audio_reduced), axis=1)  # [T]
    ax_stats1.plot(audio_energy, color='green', linewidth=2, label='Audio Energy')
    ax_stats1.fill_between(range(T), audio_energy, alpha=0.3, color='green')
    ax_stats1.set_xlabel('Frame Number')
    ax_stats1.set_ylabel('Audio Energy')
    ax_stats1.set_title('Audio Energy Over Time')
    ax_stats1.grid(True, alpha=0.3)
    ax_stats1.legend()
    
    # Expression prediction error over time
    expr_error = np.mean(np.abs(target_reduced - predicted_reduced), axis=1)  # [T]
    ax_stats2.plot(expr_error, color='red', linewidth=2, label='Prediction Error')
    ax_stats2.fill_between(range(T), expr_error, alpha=0.3, color='red')
    ax_stats2.set_xlabel('Frame Number')
    ax_stats2.set_ylabel('Mean Abs Error')
    ax_stats2.set_title('Expression Prediction Error Over Time')
    ax_stats2.grid(True, alpha=0.3)
    ax_stats2.legend()
    
    # Add overall statistics
    mean_error = np.mean(np.abs(target_display - predicted_display))
    correlation = np.corrcoef(target_display.flatten(), predicted_display.flatten())[0, 1]
    audio_expr_corr = np.corrcoef(audio_norm.flatten()[:len(target_display.flatten())], 
                                   target_display.flatten())[0, 1]
    
    fig.suptitle(
        f'Audio → Expression Mapping - Window {window_idx}\n'
        f'Mean Error: {mean_error:.4f} | Target-Pred Correlation: {correlation:.3f} | Audio-Expression Correlation: {audio_expr_corr:.3f}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved audio-expression visualization to {save_path}")
    
    return fig


def create_audio_expression_correlation_map(
    audio_features: torch.Tensor,
    expression: torch.Tensor,
    window_idx: int,
    save_path: Optional[Path] = None,
    top_k: int = 10
) -> plt.Figure:
    """
    Create correlation map between audio features and expression dimensions.
    
    Args:
        audio_features: Wav2vec features [B, T, 768] or [T, 768]
        expression: Expression tensor [B, T, 128] or [T, 128]
        window_idx: Index of the current window
        save_path: Optional path to save the figure
        top_k: Number of top correlations to highlight
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if audio_features.dim() == 3:
        audio_features = audio_features[0]
    if expression.dim() == 3:
        expression = expression[0]
    
    # Convert to numpy
    audio = audio_features.detach().cpu().numpy()
    expr = expression.detach().cpu().numpy()
    
    # Reduce dimensions for correlation analysis
    audio_reduced = audio.reshape(audio.shape[0], 32, -1).mean(axis=2)  # [T, 32]
    expr_reduced = expr.reshape(expr.shape[0], 32, -1).mean(axis=2)  # [T, 32]
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            correlation_matrix[i, j] = np.corrcoef(audio_reduced[:, i], expr_reduced[:, j])[0, 1]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot correlation matrix
    im = ax1.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'Audio-Expression Correlation Matrix - Window {window_idx}')
    ax1.set_xlabel('Expression Dimension')
    ax1.set_ylabel('Audio Dimension')
    plt.colorbar(im, ax=ax1)
    
    # Find and highlight top correlations
    correlations_flat = correlation_matrix.flatten()
    top_indices = np.argpartition(np.abs(correlations_flat), -top_k)[-top_k:]
    top_correlations = correlations_flat[top_indices]
    
    # Plot top correlations
    ax2.barh(range(top_k), np.abs(top_correlations))
    ax2.set_xlabel('Absolute Correlation')
    ax2.set_title(f'Top {top_k} Audio-Expression Correlations')
    ax2.set_yticks(range(top_k))
    
    # Create labels for top correlations
    labels = []
    for idx in top_indices:
        audio_dim = idx // 32
        expr_dim = idx % 32
        labels.append(f'A{audio_dim}-E{expr_dim}')
    ax2.set_yticklabels(labels)
    
    # Color bars by positive/negative correlation
    colors = ['red' if c < 0 else 'blue' for c in top_correlations]
    bars = ax2.barh(range(top_k), np.abs(top_correlations), color=colors)
    
    # Add legend
    ax2.legend([bars[0]], ['Correlation'], loc='lower right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved correlation map to {save_path}")
    
    return fig


def create_temporal_audio_expression_flow(
    audio_features: torch.Tensor,
    expression: torch.Tensor,
    window_idx: int,
    save_path: Optional[Path] = None,
    num_timepoints: int = 5
) -> plt.Figure:
    """
    Create temporal flow visualization showing audio to expression mapping at key timepoints.
    
    Args:
        audio_features: Wav2vec features [B, T, 768] or [T, 768]
        expression: Expression tensor [B, T, 128] or [T, 128]
        window_idx: Index of the current window
        save_path: Optional path to save the figure
        num_timepoints: Number of timepoints to visualize
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if audio_features.dim() == 3:
        audio_features = audio_features[0]
    if expression.dim() == 3:
        expression = expression[0]
    
    # Convert to numpy
    audio = audio_features.detach().cpu().numpy()
    expr = expression.detach().cpu().numpy()
    
    T = audio.shape[0]
    
    # Select timepoints
    timepoints = np.linspace(0, T-1, num_timepoints, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(num_timepoints, 2, figsize=(12, 3*num_timepoints))
    
    for i, t in enumerate(timepoints):
        # Audio at timepoint t (reduced to 32 dims)
        audio_t = audio[t].reshape(32, -1).mean(axis=1)
        
        # Expression at timepoint t (reduced to 32 dims)
        expr_t = expr[t].reshape(32, -1).mean(axis=1)
        
        # Plot audio
        axes[i, 0].bar(range(32), audio_t, color='green', alpha=0.7)
        axes[i, 0].set_title(f'Audio Features - Frame {t}')
        axes[i, 0].set_xlabel('Dimension')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot expression
        axes[i, 1].bar(range(32), expr_t, color='blue', alpha=0.7)
        axes[i, 1].set_title(f'Expression - Frame {t}')
        axes[i, 1].set_xlabel('Dimension')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Temporal Audio → Expression Flow - Window {window_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved temporal flow to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization with dummy data
    T = 50  # 50 frames
    
    # Create dummy audio features (768 dims like wav2vec)
    audio_features = torch.randn(1, T, 768) * 0.5
    # Add some temporal patterns to audio
    for i in range(T):
        audio_features[0, i, :256] += torch.sin(torch.tensor(i * 0.1)) * 0.3
        audio_features[0, i, 256:512] += torch.cos(torch.tensor(i * 0.15)) * 0.2
    
    # Create target expression (128 dims)
    target_expression = torch.randn(1, T, 128) * 0.4
    # Make expression somewhat correlated with audio
    for i in range(T):
        target_expression[0, i, :64] += audio_features[0, i, :384].mean() * 0.5
    
    # Predicted expression with some error
    predicted_expression = target_expression + torch.randn_like(target_expression) * 0.1
    
    # Create visualizations
    output_dir = Path("audio_expr_viz_test")
    output_dir.mkdir(exist_ok=True)
    
    # Main audio-expression visualization
    fig1 = create_audio_expression_visualization(
        audio_features,
        target_expression,
        predicted_expression,
        window_idx=0,
        save_path=output_dir / "audio_expression_main.png"
    )
    
    # Correlation map
    fig2 = create_audio_expression_correlation_map(
        audio_features,
        target_expression,
        window_idx=0,
        save_path=output_dir / "audio_expression_correlation.png"
    )
    
    # Temporal flow
    fig3 = create_temporal_audio_expression_flow(
        audio_features,
        target_expression,
        window_idx=0,
        save_path=output_dir / "audio_expression_flow.png"
    )
    
    print(f"Visualizations saved to {output_dir}")