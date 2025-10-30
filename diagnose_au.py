#!/usr/bin/env python3
"""
Action Unit (AU) Diagnostics Tool

Analyzes AU prediction quality by comparing model predictions against
ground truth for a specific video and checkpoint. Provides detailed
metrics and visualizations for each of the 16 AUs.

Usage:
    python diagnose_au.py --video path/to/video.mp4 \
                          --identity path/to/identity.png \
                          --config overfit_config.yaml \
                          --checkpoint checkpoints/best.pt \
                          --output-dir au_diagnostics
"""

import torch
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from au_extractor import AU_NAMES
from visualize_au import create_au_visualization, create_au_summary_visualization
from logger import logger


def load_model_and_checkpoint(config_path: str, checkpoint_path: str) -> VASAModel:
    """Load VASA model with checkpoint."""
    config = OmegaConf.load(config_path)
    model = VASAModel(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    logger.info(f"✅ Loaded model from {checkpoint_path}")
    return model


def extract_au_predictions(
    model: VASAModel,
    video_path: str,
    identity_path: str,
    config: OmegaConf
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract AU predictions and ground truth from video.

    Returns:
        au_gt: [num_windows, num_queries, 16]
        au_pred: [num_windows, num_queries, 16]
        metadata: List of window metadata dicts
    """
    # Create dataset with single video
    dataset = VASAIntegratedDataset(
        video_paths=[video_path],
        config=config,
        window_size=50,
        sequence_length=50,
        use_cache=True
    )

    au_gt_list = []
    au_pred_list = []
    metadata_list = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing windows"):
            try:
                window = dataset[i]
                if window is None:
                    continue

                # Prepare batch
                batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in window.items()}

                # Move to GPU
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                # Get model predictions
                # Prepare conditions
                conditions = {
                    'audio_features': batch['audio_features'],
                    'phoneme_gt': batch.get('phoneme_gt'),
                    'au_gt': batch.get('au_gt')  # Pass GT for comparison
                }

                # Forward pass (simplified - just need AU predictions)
                outputs = model.motion_transformer.audio_proj(
                    batch['audio_features']
                )

                # Extract AU predictions
                if len(outputs) == 2:  # Returns (output, aux_predictions)
                    _, aux_predictions = outputs
                    if 'au_pred' in aux_predictions:
                        au_pred = aux_predictions['au_pred'][0].cpu().numpy()  # [8, 16]
                        au_gt = batch['au_gt'][0].cpu().numpy()  # [8, 16]

                        au_gt_list.append(au_gt)
                        au_pred_list.append(au_pred)
                        metadata_list.append({
                            'window_idx': i,
                            'video_path': video_path
                        })

            except Exception as e:
                logger.warning(f"Error processing window {i}: {e}")
                continue

    if not au_gt_list:
        raise ValueError("No AU predictions extracted!")

    au_gt_array = np.stack(au_gt_list, axis=0)  # [num_windows, 8, 16]
    au_pred_array = np.stack(au_pred_list, axis=0)  # [num_windows, 8, 16]

    logger.info(f"✅ Extracted {len(au_gt_list)} windows")
    logger.info(f"   AU GT shape: {au_gt_array.shape}")
    logger.info(f"   AU Pred shape: {au_pred_array.shape}")

    return au_gt_array, au_pred_array, metadata_list


def compute_au_metrics(
    au_gt: np.ndarray,
    au_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive metrics for AU predictions.

    Args:
        au_gt: [num_windows, num_queries, 16]
        au_pred: [num_windows, num_queries, 16]

    Returns:
        Dictionary with metrics per AU
    """
    num_aus = au_gt.shape[-1]

    # Reshape to [num_windows * num_queries, 16] for per-AU metrics
    au_gt_flat = au_gt.reshape(-1, num_aus)
    au_pred_flat = au_pred.reshape(-1, num_aus)

    metrics = {}

    # Per-AU metrics
    for i in range(num_aus):
        gt = au_gt_flat[:, i]
        pred = au_pred_flat[:, i]

        # Mean Absolute Error
        mae = np.abs(gt - pred).mean()

        # Root Mean Squared Error
        rmse = np.sqrt(((gt - pred) ** 2).mean())

        # Correlation
        if gt.std() > 0 and pred.std() > 0:
            corr = np.corrcoef(gt, pred)[0, 1]
        else:
            corr = 0.0

        # Mean values
        gt_mean = gt.mean()
        pred_mean = pred.mean()

        # Activation rate (> 0.5 threshold)
        gt_active_rate = (gt > 0.5).mean()
        pred_active_rate = (pred > 0.5).mean()

        metrics[AU_NAMES[i]] = {
            'mae': mae,
            'rmse': rmse,
            'correlation': corr,
            'gt_mean': gt_mean,
            'pred_mean': pred_mean,
            'gt_activation_rate': gt_active_rate,
            'pred_activation_rate': pred_active_rate,
            'bias': pred_mean - gt_mean
        }

    return metrics


def create_diagnostic_report(
    metrics: Dict[str, Dict],
    output_dir: Path
):
    """Create comprehensive diagnostic report."""
    # Convert to DataFrame
    df = pd.DataFrame(metrics).T
    df.index.name = 'AU'

    # Save CSV
    csv_path = output_dir / 'au_metrics.csv'
    df.to_csv(csv_path)
    logger.info(f"📊 Saved metrics to {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("ACTION UNIT PREDICTION METRICS SUMMARY")
    print("="*80)
    print(f"\n{df.to_string()}\n")

    # Identify problematic AUs
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    high_error_aus = df[df['mae'] > 0.15].index.tolist()
    if high_error_aus:
        print(f"\n⚠️  HIGH ERROR AUs (MAE > 0.15):")
        for au in high_error_aus:
            print(f"   - {au}: MAE={df.loc[au, 'mae']:.4f}")

    low_corr_aus = df[df['correlation'] < 0.5].index.tolist()
    if low_corr_aus:
        print(f"\n⚠️  LOW CORRELATION AUs (< 0.5):")
        for au in low_corr_aus:
            print(f"   - {au}: Corr={df.loc[au, 'correlation']:.3f}")

    # Best performing AUs
    best_aus = df.nsmallest(3, 'mae').index.tolist()
    print(f"\n✅ BEST PERFORMING AUs (Lowest MAE):")
    for au in best_aus:
        print(f"   - {au}: MAE={df.loc[au, 'mae']:.4f}, Corr={df.loc[au, 'correlation']:.3f}")

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Average MAE: {df['mae'].mean():.4f}")
    print(f"   Average RMSE: {df['rmse'].mean():.4f}")
    print(f"   Average Correlation: {df['correlation'].mean():.3f}")
    print(f"   AUs with MAE < 0.1: {(df['mae'] < 0.1).sum()}/{len(df)}")
    print(f"   AUs with Correlation > 0.7: {(df['correlation'] > 0.7).sum()}/{len(df)}")

    print("\n" + "="*80 + "\n")


def create_diagnostic_plots(
    au_gt: np.ndarray,
    au_pred: np.ndarray,
    metrics: Dict[str, Dict],
    output_dir: Path
):
    """Create diagnostic visualization plots."""

    # Plot 1: Per-AU MAE bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    au_names = list(metrics.keys())
    maes = [metrics[au]['mae'] for au in au_names]
    colors = ['#e74c3c' if mae > 0.15 else '#3498db' for mae in maes]

    bars = ax.bar(range(len(au_names)), maes, color=colors, alpha=0.7)
    ax.axhline(y=0.15, color='red', linestyle='--', linewidth=2, label='High Error Threshold')
    ax.set_xlabel('Action Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('AU Prediction Error by Action Unit', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(au_names)))
    ax.set_xticklabels([au.replace('_', '\n') for au in au_names], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'au_mae_by_unit.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"📊 Saved MAE bar chart")

    # Plot 2: Correlation scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations = [metrics[au]['correlation'] for au in au_names]
    ax.scatter(range(len(au_names)), correlations, s=100, alpha=0.6, c=correlations, cmap='RdYlGn')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Low Correlation Threshold')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Good Correlation Threshold')
    ax.set_xlabel('Action Units', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('AU Prediction Correlation by Action Unit', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(au_names)))
    ax.set_xticklabels([au.replace('_', '\n') for au in au_names], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'au_correlation_by_unit.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"📊 Saved correlation scatter")

    # Plot 3: Sample window visualizations
    num_samples = min(5, au_gt.shape[0])
    sample_indices = np.linspace(0, au_gt.shape[0]-1, num_samples, dtype=int)

    for idx in sample_indices:
        fig = create_au_visualization(
            au_gt=torch.from_numpy(au_gt[idx]),
            au_pred=torch.from_numpy(au_pred[idx]),
            au_names=AU_NAMES,
            window_idx=idx
        )
        fig.savefig(output_dir / f'au_window_{idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"📊 Saved {num_samples} sample window visualizations")


def main():
    parser = argparse.ArgumentParser(description='Diagnose AU predictions')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--identity', type=str, required=True, help='Path to identity image')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output-dir', type=str, default='au_diagnostics', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("ACTION UNIT DIAGNOSTICS")
    logger.info("="*80)
    logger.info(f"Video: {args.video}")
    logger.info(f"Identity: {args.identity}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)

    # Load config
    config = OmegaConf.load(args.config)

    # Load model
    logger.info("\n📦 Loading model...")
    model = load_model_and_checkpoint(args.config, args.checkpoint)

    # Extract AU predictions
    logger.info("\n🔍 Extracting AU predictions...")
    au_gt, au_pred, metadata = extract_au_predictions(
        model, args.video, args.identity, config
    )

    # Compute metrics
    logger.info("\n📊 Computing metrics...")
    metrics = compute_au_metrics(au_gt, au_pred)

    # Create report
    logger.info("\n📝 Creating diagnostic report...")
    create_diagnostic_report(metrics, output_dir)

    # Create plots
    logger.info("\n🎨 Creating diagnostic plots...")
    create_diagnostic_plots(au_gt, au_pred, metrics, output_dir)

    logger.info(f"\n✅ Diagnostics complete! Results saved to {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"   - au_metrics.csv: Detailed metrics per AU")
    logger.info(f"   - au_mae_by_unit.png: MAE bar chart")
    logger.info(f"   - au_correlation_by_unit.png: Correlation scatter")
    logger.info(f"   - au_window_XXXX.png: Sample window visualizations")


if __name__ == "__main__":
    main()
