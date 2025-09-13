#!/usr/bin/env python3
"""
Simple test for VASA audio feature extraction
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

def test_audio_extraction(video_path: str = "../junk/7.mp4"):
    """Test audio feature extraction from video"""

    logger.info("=== Testing Audio Feature Extraction ===")

    # Initialize VASA inference
    logger.info("Loading VASA model...")
    checkpoint = "checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    vasa = VASAInference(checkpoint, config)

    # Extract audio from video
    logger.info(f"Extracting audio from {video_path}")
    import subprocess
    audio_path = "test_audio.wav"
    cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y"
    subprocess.run(cmd.split(), check=True, capture_output=True)

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    logger.info(f"Audio loaded: shape {waveform.shape}, sr {sr}")

    # Extract 2 seconds of audio
    audio_segment = waveform[0, :sr*2].numpy()  # 2 seconds

    # Process through VASA's audio feature extraction
    logger.info("Extracting audio features...")
    audio_features = vasa.extract_audio_features_aligned(
        audio_segment,
        target_frames=50  # 2 seconds at 25fps
    )

    logger.info(f"Audio features shape: {audio_features.shape}")
    logger.info(f"  Min: {audio_features.min():.4f}, Max: {audio_features.max():.4f}")
    logger.info(f"  Mean: {audio_features.mean():.4f}, Std: {audio_features.std():.4f}")

    # Visualize features
    plt.figure(figsize=(12, 6))

    # Heatmap of features
    plt.subplot(2, 1, 1)
    plt.imshow(audio_features[:, :100].T.cpu(), aspect='auto', cmap='viridis')
    plt.title('Audio Features (First 100 dims)')
    plt.xlabel('Time Frame')
    plt.ylabel('Feature Dimension')
    plt.colorbar()

    # Mean activation over time
    plt.subplot(2, 1, 2)
    mean_activation = audio_features.mean(dim=-1).cpu()
    plt.plot(mean_activation)
    plt.title('Mean Feature Activation Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Mean Activation')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('audio_features_test.png')
    plt.close()
    logger.info("Saved visualization to audio_features_test.png")

    # Test with different window sizes
    logger.info("\nTesting different window sizes:")
    for num_frames in [10, 25, 50, 100]:
        features = vasa.extract_audio_features_aligned(
            audio_segment[:sr*(num_frames//25)],  # Adjust audio length
            target_frames=num_frames
        )
        logger.info(f"  {num_frames} frames: shape {features.shape}")

    return audio_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='../junk/15.mp4', help='Video path')
    args = parser.parse_args()

    test_audio_extraction(args.video)