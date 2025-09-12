#!/usr/bin/env python3
"""Simplified audio-expression visualization for inference monitoring."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_inference_audio_expression_viz(
    audio_features: torch.Tensor,
    predicted_expression: torch.Tensor,
    motion_params: dict,
    window_idx: int,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a compact visualization for inference showing audio input and resulting expression/motion.
    
    Args:
        audio_features: Wav2vec features [B, T, 768] or [T, 768]
        predicted_expression: Predicted expression tensor [B, T, 128] or [T, 128]
        motion_params: Dict with 'rotation', 'translation', 'scale' tensors
        window_idx: Index of the current window
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure
    """
    # Remove batch dimension if present
    if audio_features.dim() == 3:
        audio_features = audio_features[0]
    if predicted_expression.dim() == 3:
        predicted_expression = predicted_expression[0]
    
    # Convert to numpy
    audio = audio_features.detach().cpu().numpy()
    expr = predicted_expression.detach().cpu().numpy()
    
    T = audio.shape[0]
    
    # Reduce dimensions
    audio_reduced = audio.reshape(T, 32, -1).mean(axis=2)  # [T, 32]
    expr_reduced = expr.reshape(T, 32, -1).mean(axis=2)  # [T, 32]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # 1. Audio features
    im1 = axes[0].imshow(
        audio_reduced.T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    axes[0].set_title(f'Window {window_idx} - Wav2Vec Audio Features (32 dims)', fontsize=12)
    axes[0].set_ylabel('Audio Dim')
    axes[0].set_xticks(np.arange(0, T, 5))
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Predicted expression
    im2 = axes[1].imshow(
        expr_reduced.T,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest'
    )
    axes[1].set_title('Predicted Expression (32 dims)', fontsize=12)
    axes[1].set_ylabel('Expression Dim')
    axes[1].set_xticks(np.arange(0, T, 5))
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Motion parameters over time
    if motion_params:
        # Extract motion data
        rotation = motion_params.get('rotation', None)
        translation = motion_params.get('translation', None)
        scale = motion_params.get('scale', None)
        
        if rotation is not None:
            if rotation.dim() == 3:
                rotation = rotation[0]  # Remove batch
            rotation_np = rotation.detach().cpu().numpy()
            
            # Plot rotation angles
            axes[2].plot(rotation_np[:, 0], label='Pitch', alpha=0.7)
            axes[2].plot(rotation_np[:, 1], label='Yaw', alpha=0.7)
            axes[2].plot(rotation_np[:, 2], label='Roll', alpha=0.7)
            
        if translation is not None:
            if translation.dim() == 3:
                translation = translation[0]
            trans_np = translation.detach().cpu().numpy()
            axes[2].plot(np.linalg.norm(trans_np, axis=1), label='Translation Magnitude', linestyle='--', alpha=0.7)
            
        axes[2].set_title('Motion Parameters', fontsize=12)
        axes[2].set_xlabel('Frame Number')
        axes[2].set_ylabel('Value')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add statistics
    audio_energy = np.mean(np.abs(audio_reduced))
    expr_variance = np.var(expr_reduced)
    
    fig.suptitle(
        f'Audio-Driven Generation - Audio Energy: {audio_energy:.3f}, Expression Variance: {expr_variance:.3f}',
        fontsize=13,
        y=1.02
    )
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved inference visualization to {save_path}")
    
    return fig


def create_audio_motion_summary(
    audio_windows: list,
    motion_sequences: list,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a summary visualization of all audio windows and their generated motion.
    
    Args:
        audio_windows: List of audio feature tensors
        motion_sequences: List of motion parameter dicts
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure
    """
    num_windows = len(audio_windows)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Collect audio energy per window
    audio_energies = []
    for window in audio_windows:
        if 'audio_features' in window:
            features = window['audio_features']
            if features.dim() == 3:
                features = features[0]
            energy = torch.abs(features).mean().item()
            audio_energies.append(energy)
    
    # Collect motion variance per window
    motion_variances = []
    for motion in motion_sequences:
        if 'expression_embed' in motion:
            expr = motion['expression_embed']
            if expr.dim() == 3:
                expr = expr[0]
            variance = expr.var().item()
            motion_variances.append(variance)
    
    # Plot audio energy
    axes[0].bar(range(len(audio_energies)), audio_energies, color='green', alpha=0.7)
    axes[0].set_title('Audio Energy per Window')
    axes[0].set_xlabel('Window Index')
    axes[0].set_ylabel('Mean Audio Energy')
    axes[0].grid(True, alpha=0.3)
    
    # Plot motion variance
    axes[1].bar(range(len(motion_variances)), motion_variances, color='blue', alpha=0.7)
    axes[1].set_title('Expression Variance per Window')
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel('Expression Variance')
    axes[1].grid(True, alpha=0.3)
    
    # Add correlation if both exist
    if audio_energies and motion_variances and len(audio_energies) == len(motion_variances):
        correlation = np.corrcoef(audio_energies, motion_variances)[0, 1]
        fig.suptitle(f'Audio-Motion Summary - Correlation: {correlation:.3f}', fontsize=14)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved summary to {save_path}")
    
    return fig