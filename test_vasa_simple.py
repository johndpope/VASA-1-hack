#!/usr/bin/env python3
"""
Simple VASA test to verify basic audio-to-video generation
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'nemo')

from logger import logger
from vi import VASAInference

def main():
    """Run simple VASA test with existing video input"""
    logger.info("=== Simple VASA Test ===")

    # Initialize VASA
    checkpoint = "./checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    vasa = VASAInference(checkpoint, config)

    # Use existing test video
    input_video = "nemo/data/VID_2.mp4"
    output_path = "test_output_simple.mp4"

    logger.info(f"Input video: {input_video}")
    logger.info(f"Output path: {output_path}")

    # Generate from video
    try:
        result = vasa.generate_from_video(
            input_video,
            output_path,
            fps=25.0
        )
        logger.info(f"Generation complete! Output saved to {output_path}")

        # Check if output exists
        if Path(output_path).exists():
            logger.info(f"✓ Output file created: {Path(output_path).stat().st_size / 1024:.2f} KB")
        else:
            logger.error("✗ Output file not created")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()