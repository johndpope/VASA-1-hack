#!/usr/bin/env python3
"""
Basic VASA functionality test
"""

import torch
import numpy as np
from PIL import Image
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import VASA modules
sys.path.insert(0, 'nemo')
from vi import VASAInference

def test_vasa_basic():
    """Test basic VASA functionality"""

    logger.info("=== Testing Basic VASA Functionality ===")

    # Initialize VASA inference
    logger.info("Loading VASA model...")
    checkpoint = "checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    vasa = VASAInference(checkpoint, config)

    # Test 1: Extract source parameters from image
    logger.info("\n=== Test 1: Source Parameter Extraction ===")
    source_image_path = "./data/A.png"

    if Path(source_image_path).exists():
        source_img = Image.open(source_image_path).convert('RGB')
        logger.info(f"Loaded source image: {source_img.size}")

        # Convert to tensor and extract parameters
        source_tensor = vasa.transform(source_img).unsqueeze(0).to(vasa.device)
        logger.info(f"Source tensor shape: {source_tensor.shape}")

        # Extract parameters using EMO method
        source_params = vasa.extract_emo_parameters(source_tensor)
        logger.info("Extracted source parameters:")
        for key, value in source_params.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                logger.info(f"    min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")
    else:
        logger.error(f"Source image not found: {source_image_path}")

    # Test 2: Generate from video
    logger.info("\n=== Test 2: Generate from Video ===")
    video_path = "./nemo/data/VID_2.mp4"

    if Path(video_path).exists():
        logger.info(f"Processing video: {video_path}")
        output_path = "test_output.mp4"

        try:
            frames = vasa.generate_from_video(
                input_video=video_path,
                output_path=output_path,
                fps=25.0
            )

            if frames and len(frames) > 0:
                logger.info(f"Generated {len(frames)} frames")

                # Analyze frames
                first_frame = frames[0]
                logger.info(f"Frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")

                # Save sample frames
                Image.fromarray(frames[0]).save("test_frame_0.png")
                Image.fromarray(frames[len(frames)//2]).save("test_frame_mid.png")
                Image.fromarray(frames[-1]).save("test_frame_last.png")
                logger.info("Saved sample frames: test_frame_0.png, test_frame_mid.png, test_frame_last.png")

                # Calculate motion statistics
                frames_array = np.stack(frames)
                frame_diffs = np.mean(np.abs(frames_array[1:] - frames_array[:-1]), axis=(1,2,3))
                logger.info(f"Frame-to-frame motion statistics:")
                logger.info(f"  Mean diff: {np.mean(frame_diffs):.3f}")
                logger.info(f"  Std diff: {np.std(frame_diffs):.3f}")
                logger.info(f"  Max diff: {np.max(frame_diffs):.3f}")

            else:
                logger.warning("No frames generated")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"Video not found: {video_path}")

    # Test 3: Check model components
    logger.info("\n=== Test 3: Model Components ===")
    logger.info(f"Model type: {type(vasa.model)}")
    logger.info(f"Volumetric avatar type: {type(vasa.volumetric_avatar)}")
    logger.info(f"Audio model available: {hasattr(vasa, 'audio_model')}")
    logger.info(f"Audio processor available: {hasattr(vasa, 'audio_processor')}")

    # Check motion transformer
    if hasattr(vasa.model, 'motion_transformer'):
        mt = vasa.model.motion_transformer
        logger.info(f"Motion transformer:")
        logger.info(f"  d_model: {mt.d_model}")
        logger.info(f"  num_layers: {mt.num_layers}")
        logger.info(f"  num_heads: {mt.num_heads}")

    # Test 4: Audio processing capability
    logger.info("\n=== Test 4: Audio Processing ===")
    if hasattr(vasa, 'audio_model') and hasattr(vasa, 'audio_processor'):
        # Create a simple test audio (1 second of sine wave)
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        test_audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        try:
            # Process through audio model
            inputs = vasa.audio_processor(
                test_audio.numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(vasa.device)

            with torch.no_grad():
                audio_features = vasa.audio_model(**inputs).last_hidden_state

            logger.info(f"Audio features shape: {audio_features.shape}")
            logger.info(f"Audio features stats: min={audio_features.min():.3f}, max={audio_features.max():.3f}")

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
    else:
        logger.warning("Audio processing components not available")

    logger.info("\n=== Tests Complete ===")

    # Summary
    logger.info("\nSummary:")
    logger.info("- Source parameter extraction: ✓" if 'source_params' in locals() else "- Source parameter extraction: ✗")
    logger.info("- Video generation: ✓" if 'frames' in locals() and frames else "- Video generation: ✗")
    logger.info("- Model components: ✓")
    logger.info("- Audio processing: ✓" if 'audio_features' in locals() else "- Audio processing: ✗")


if __name__ == "__main__":
    test_vasa_basic()