#!/usr/bin/env python3
"""Visualization for expression embeddings as candle-like patterns."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_expression_candles(
    target_expression: torch.Tensor,
    predicted_expression: torch.Tensor, 
    window_idx: int,
    save_path: Optional[Path] = None,
    reduce_to: int = 32
) -> plt.Figure:
    """
    Create candle-like visualization of expression embeddings.
    
    Args:
        target_expression: Target expression tensor [B, T, 128] or [T, 128]
        predicted_expression: Predicted expression tensor [B, T, 128] or [T, 128]
        window_idx: Index of the current window
        save_path: Optional path to save the figure
        reduce_to: Number of dimensions to reduce to (default 32)
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if target_expression.dim() == 3:
        target_expression = target_expression[0]  # [T, 128]
    if predicted_expression.dim() == 3:
        predicted_expression = predicted_expression[0]  # [T, 128]
    
    # Convert to numpy
    target = target_expression.detach().cpu().numpy()
    predicted = predicted_expression.detach().cpu().numpy()
    
    # Get dimensions
    T, embed_dim = target.shape
    
    # Reduce embedding dimension by averaging groups
    if embed_dim > reduce_to:
        group_size = embed_dim // reduce_to
        target_reduced = target.reshape(T, reduce_to, group_size).mean(axis=2)
        predicted_reduced = predicted.reshape(T, reduce_to, group_size).mean(axis=2)
    else:
        target_reduced = target
        predicted_reduced = predicted
        reduce_to = embed_dim
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Transpose for candle display (frames on x-axis, embedding dims as vertical candles)
    target_display = target_reduced.T  # [32, 50]
    predicted_display = predicted_reduced.T  # [32, 50]
    
    # Normalize for better visualization
    vmin = min(target_display.min(), predicted_display.min())
    vmax = max(target_display.max(), predicted_display.max())
    
    # Plot target expression
    im1 = ax1.imshow(
        target_display,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    ax1.set_title(f'Window {window_idx} - Target Expression (reduced to {reduce_to} dims)', fontsize=12)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Expression Dim')
    ax1.set_xticks(np.arange(0, T, 5))
    ax1.set_xticklabels(np.arange(0, T, 5))
    ax1.set_yticks(np.arange(0, reduce_to, 4))
    ax1.set_yticklabels(np.arange(0, reduce_to, 4))
    
    # Add colorbar for target
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Value', rotation=270, labelpad=15)
    
    # Plot predicted expression
    im2 = ax2.imshow(
        predicted_display,
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    ax2.set_title(f'Window {window_idx} - Predicted Expression (reduced to {reduce_to} dims)', fontsize=12)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Expression Dim')
    ax2.set_xticks(np.arange(0, T, 5))
    ax2.set_xticklabels(np.arange(0, T, 5))
    ax2.set_yticks(np.arange(0, reduce_to, 4))
    ax2.set_yticklabels(np.arange(0, reduce_to, 4))
    
    # Add colorbar for predicted
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Value', rotation=270, labelpad=15)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Calculate and display difference statistics
    diff = np.abs(target_display - predicted_display)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    fig.suptitle(
        f'Expression Embedding Visualization - Window {window_idx}\n'
        f'Mean Absolute Difference: {mean_diff:.4f}, Max Difference: {max_diff:.4f}',
        fontsize=14
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved expression visualization to {save_path}")
    
    return fig


def create_expression_difference_map(
    target_expression: torch.Tensor,
    predicted_expression: torch.Tensor,
    window_idx: int,
    save_path: Optional[Path] = None,
    reduce_to: int = 32
) -> plt.Figure:
    """
    Create a difference map between target and predicted expressions.
    
    Args:
        target_expression: Target expression tensor [B, T, 128] or [T, 128]
        predicted_expression: Predicted expression tensor [B, T, 128] or [T, 128]
        window_idx: Index of the current window
        save_path: Optional path to save the figure
        reduce_to: Number of dimensions to reduce to (default 32)
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if target_expression.dim() == 3:
        target_expression = target_expression[0]
    if predicted_expression.dim() == 3:
        predicted_expression = predicted_expression[0]
    
    # Convert to numpy
    target = target_expression.detach().cpu().numpy()
    predicted = predicted_expression.detach().cpu().numpy()
    
    # Get dimensions
    T, embed_dim = target.shape
    
    # Reduce embedding dimension
    if embed_dim > reduce_to:
        group_size = embed_dim // reduce_to
        target_reduced = target.reshape(T, reduce_to, group_size).mean(axis=2)
        predicted_reduced = predicted.reshape(T, reduce_to, group_size).mean(axis=2)
    else:
        target_reduced = target
        predicted_reduced = predicted
        reduce_to = embed_dim
    
    # Calculate difference
    difference = (predicted_reduced - target_reduced).T  # [32, 50]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    # Plot difference map
    im = ax.imshow(
        difference,
        aspect='auto',
        cmap='coolwarm',
        vmin=-np.abs(difference).max(),
        vmax=np.abs(difference).max(),
        interpolation='nearest'
    )
    
    ax.set_title(f'Window {window_idx} - Expression Difference (Predicted - Target)', fontsize=14)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Expression Dim')
    ax.set_xticks(np.arange(0, T, 5))
    ax.set_xticklabels(np.arange(0, T, 5))
    ax.set_yticks(np.arange(0, reduce_to, 4))
    ax.set_yticklabels(np.arange(0, reduce_to, 4))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference', rotation=270, labelpad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add statistics
    mean_abs_diff = np.abs(difference).mean()
    std_diff = difference.std()
    
    ax.text(
        0.02, 0.98,
        f'Mean Abs Diff: {mean_abs_diff:.4f}\nStd Dev: {std_diff:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved expression difference map to {save_path}")
    
    return fig


def visualize_expression_batch(
    expressions: torch.Tensor,
    batch_idx: int,
    save_dir: Path,
    prefix: str = "expression"
) -> None:
    """
    Visualize a batch of expressions.
    
    Args:
        expressions: Expression tensor [B, T, 128]
        batch_idx: Batch index for naming
        save_dir: Directory to save visualizations
        prefix: Prefix for filenames
    """
    B = expressions.shape[0]
    
    for b in range(B):
        expr = expressions[b]  # [T, 128]
        
        # Create a simple heatmap
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        
        # Reduce to 32 dims for visualization
        T, embed_dim = expr.shape
        reduce_to = 32
        if embed_dim > reduce_to:
            group_size = embed_dim // reduce_to
            expr_reduced = expr.reshape(T, reduce_to, group_size).mean(axis=2)
        else:
            expr_reduced = expr
        
        im = ax.imshow(
            expr_reduced.T.detach().cpu().numpy(),
            aspect='auto',
            cmap='RdBu_r',
            interpolation='nearest'
        )
        
        ax.set_title(f'{prefix} - Batch {batch_idx}, Sample {b}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Expression Dim')
        plt.colorbar(im, ax=ax)
        
        save_path = save_dir / f'{prefix}_batch{batch_idx}_sample{b}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Test visualization with dummy data
    T = 50  # 50 frames
    embed_dim = 128
    
    # Create dummy expressions with some patterns
    target = torch.randn(1, T, embed_dim) * 0.5
    # Add some temporal patterns
    for i in range(T):
        target[0, i, :32] += torch.sin(torch.tensor(i * 0.2)) * 0.3
        target[0, i, 32:64] += torch.cos(torch.tensor(i * 0.15)) * 0.2
    
    # Predicted with some noise
    predicted = target + torch.randn_like(target) * 0.1
    
    # Create visualizations
    output_dir = Path("expression_viz_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create candle visualization
    fig1 = create_expression_candles(
        target, predicted, 
        window_idx=0,
        save_path=output_dir / "expression_candles.png"
    )
    
    # Create difference map
    fig2 = create_expression_difference_map(
        target, predicted,
        window_idx=0,
        save_path=output_dir / "expression_difference.png"
    )
    
    # plt.show()  # Comment out to avoid blocking
    print(f"Visualizations saved to {output_dir}")