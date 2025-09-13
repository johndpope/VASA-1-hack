#!/usr/bin/env python3
"""
Test VASA audio window processing
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import VASA modules
sys.path.insert(0, 'nemo')
from vi import VASAInference

def test_audio_windows():
    """Test audio window processing"""

    logger.info("=== Testing Audio Window Processing ===")

    # Initialize VASA
    checkpoint = "checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    vasa = VASAInference(checkpoint, config)

    # Create test audio (12 seconds of varying frequency sine wave)
    sample_rate = 16000
    duration = 12.0  # Longer duration to generate multiple windows
    t = torch.linspace(0, duration, int(sample_rate * duration))

    # More complex audio signal to simulate speech variation
    # Multiple frequency components with different modulation rates
    freq_base = 440 + 100 * torch.sin(2 * np.pi * 0.5 * t)  # Slow modulation
    freq_mod = freq_base + 50 * torch.sin(2 * np.pi * 3 * t)  # Faster modulation

    # Combine multiple harmonics for richer audio
    test_audio = (0.6 * torch.sin(2 * np.pi * freq_mod * t / sample_rate) +
                  0.3 * torch.sin(2 * np.pi * freq_mod * 2 * t / sample_rate) +  # 2nd harmonic
                  0.1 * torch.sin(2 * np.pi * freq_mod * 3 * t / sample_rate))   # 3rd harmonic

    # Add amplitude modulation to simulate speech dynamics
    amplitude_mod = 0.8 + 0.2 * torch.sin(2 * np.pi * 1.5 * t)
    test_audio = test_audio * amplitude_mod

    logger.info(f"Test audio shape: {test_audio.shape}")

    # Process audio into windows
    logger.info("Processing audio into windows...")
    audio_windows = vasa.process_audio(test_audio, sr=sample_rate, fps=25.0)

    if audio_windows:
        logger.info(f"Created {len(audio_windows)} audio windows")

        # Analyze first window
        first_window = audio_windows[0]
        logger.info("\nFirst window contents:")
        for key, value in first_window.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape {value.shape}")
                logger.info(f"    stats: min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")

        # Visualize audio features across windows
        if 'audio_features' in first_window:
            # Collect audio features from all windows
            all_features = []
            for window in audio_windows:
                if 'audio_features' in window:
                    features = window['audio_features']
                    if features.dim() == 3:
                        features = features.squeeze(0)  # Remove batch dim
                    all_features.append(features.cpu())

            if all_features:
                # Stack features
                features_tensor = torch.stack(all_features)  # [num_windows, T, 768]
                logger.info(f"All audio features shape: {features_tensor.shape}")

                # Plot feature analysis
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # 1. Mean activation per window
                mean_per_window = features_tensor.mean(dim=(1, 2))
                axes[0, 0].plot(mean_per_window)
                axes[0, 0].set_title('Mean Activation per Window')
                axes[0, 0].set_xlabel('Window Index')
                axes[0, 0].set_ylabel('Mean Activation')
                axes[0, 0].grid(True)

                # 2. Feature variance per window
                var_per_window = features_tensor.var(dim=(1, 2))
                axes[0, 1].plot(var_per_window)
                axes[0, 1].set_title('Feature Variance per Window')
                axes[0, 1].set_xlabel('Window Index')
                axes[0, 1].set_ylabel('Variance')
                axes[0, 1].grid(True)

                # 3. First window feature heatmap
                first_features = features_tensor[0, :, :100]  # First 100 dims
                axes[1, 0].imshow(first_features.T, aspect='auto', cmap='viridis')
                axes[1, 0].set_title('First Window Features (100 dims)')
                axes[1, 0].set_xlabel('Time Frame')
                axes[1, 0].set_ylabel('Feature Dimension')

                # 4. Feature correlation between adjacent windows
                if len(features_tensor) > 1:
                    correlations = []
                    for i in range(len(features_tensor) - 1):
                        flat1 = features_tensor[i].flatten()
                        flat2 = features_tensor[i + 1].flatten()
                        corr = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
                        correlations.append(corr.item())

                    axes[1, 1].plot(correlations)
                    axes[1, 1].set_title('Correlation Between Adjacent Windows')
                    axes[1, 1].set_xlabel('Window Pair Index')
                    axes[1, 1].set_ylabel('Correlation')
                    axes[1, 1].grid(True)

                plt.tight_layout()
                plt.savefig('audio_window_analysis.png')
                logger.info("Saved analysis to audio_window_analysis.png")
                plt.close()

                # Summary statistics
                logger.info("\n=== Audio Feature Statistics ===")
                logger.info(f"Total windows: {len(features_tensor)}")
                logger.info(f"Frames per window: {features_tensor.shape[1]}")
                logger.info(f"Feature dimensions: {features_tensor.shape[2]}")
                logger.info(f"Overall mean activation: {features_tensor.mean():.4f}")
                logger.info(f"Overall std activation: {features_tensor.std():.4f}")

                if len(correlations) > 0:
                    logger.info(f"Mean correlation between windows: {np.mean(correlations):.4f}")
                    logger.info(f"Min correlation: {np.min(correlations):.4f}")
                    logger.info(f"Max correlation: {np.max(correlations):.4f}")
                else:
                    logger.info("Not enough windows to compute correlations (need at least 2)")

    else:
        logger.warning("No audio windows created")

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    test_audio_windows()