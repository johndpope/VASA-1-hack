#!/usr/bin/env python3
"""
Simple test for VASA audio processing and generation
"""

import torch
import torchaudio
import numpy as np
from PIL import Image
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import VASA modules
sys.path.insert(0, 'nemo')
from vi import VASAInference

def test_vasa_generation():
    """Test VASA with simple audio and image"""

    logger.info("=== Testing VASA Audio-Driven Generation ===")

    # Initialize VASA inference
    logger.info("Loading VASA model...")
    checkpoint = "checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    vasa = VASAInference(checkpoint, config)

    # Test with simple source image
    source_image_path = "./data/A.png"
    video_path = "./nemo/data/VID_2.mp4"

    logger.info(f"Source image: {source_image_path}")
    logger.info(f"Test video: {video_path}")

    # Generate from video (includes audio processing)
    logger.info("Running VASA generation from video...")
    try:
        frames = vasa.generate_from_video(
            source_image_path=source_image_path,
            video_path=video_path,
            output_video_path="test_output.mp4"
        )
        logger.info(f"Generated {len(frames) if frames else 0} frames")

        if frames and len(frames) > 0:
            # Check first frame
            first_frame = frames[0]
            logger.info(f"First frame shape: {first_frame.shape}")
            logger.info(f"First frame min/max: {first_frame.min():.3f}/{first_frame.max():.3f}")

            # Save first, middle, last frames
            Image.fromarray(frames[0]).save("test_frame_first.png")
            Image.fromarray(frames[len(frames)//2]).save("test_frame_middle.png")
            Image.fromarray(frames[-1]).save("test_frame_last.png")
            logger.info("Saved sample frames")

            # Check motion variance
            frames_tensor = torch.tensor(np.stack(frames)).float() / 255.0
            motion_var = frames_tensor.var(dim=0).mean().item()
            logger.info(f"Motion variance: {motion_var:.6f}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test audio processing separately
    logger.info("\n=== Testing Audio Processing ===")
    try:
        # Load audio from video
        import subprocess
        audio_path = "test_audio.wav"
        cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y"
        subprocess.run(cmd.split(), check=True, capture_output=True)

        waveform, sr = torchaudio.load(audio_path)
        logger.info(f"Audio shape: {waveform.shape}, sr: {sr}")

        # Process audio
        audio_windows = vasa.process_audio(waveform[0], sr=sr, fps=25.0)
        logger.info(f"Audio windows: {len(audio_windows)}")

        if audio_windows:
            first_window = audio_windows[0]
            logger.info(f"First window audio features shape: {first_window['audio_features'].shape}")
            logger.info(f"Audio features min/max: {first_window['audio_features'].min():.3f}/{first_window['audio_features'].max():.3f}")

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    test_vasa_generation()