#!/usr/bin/env python3
"""Debug motion generation output"""

import torch
from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger
import torchaudio
from PIL import Image

def main():
    # Initialize tester
    tester = VASAAlignmentTester()

    # Load test image
    test_image = "data/VID_2_source.png"
    img = Image.open(test_image).convert('RGB')
    frame = tester.vasa_inference.transform(img).unsqueeze(0).to(tester.device)

    # Extract parameters
    params = tester.vasa_inference.extract_emo_parameters(frame)

    # Create simple conditions
    B, T = 1, 10
    conditions = {
        'audio_features': torch.randn(B, T, 768).to(tester.device),
        'gaze': torch.zeros(B, T, 2).to(tester.device),
        'head_distance': torch.ones(B, T, 1).to(tester.device),
        'emotion': torch.zeros(B, T, 2).to(tester.device)
    }

    # Generate motion
    logger.info("Generating motion...")
    with torch.no_grad():
        motion = tester.vasa_inference.model.generate_sequence(
            initial_pose={'theta': params['theta'][0]},
            initial_dynamics=params['expression_embed'][0],
            conditions=conditions,
            num_steps=10
        )

    logger.info(f"Motion type: {type(motion)}")
    logger.info(f"Motion keys: {motion.keys() if isinstance(motion, dict) else 'Not a dict!'}")

    if isinstance(motion, dict):
        for key, value in motion.items():
            logger.info(f"  {key}: shape {value.shape}, type {type(value)}")

        # Test the check that's failing
        logger.info(f"'scale' in motion: {'scale' in motion}")
        logger.info(f"'rotation' in motion: {'rotation' in motion}")
        logger.info(f"'translation' in motion: {'translation' in motion}")

if __name__ == "__main__":
    main()