#!/usr/bin/env python3
"""Run Test 0: Audio-Driven Motion Variance Test with debug output"""

from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger
import os

def main():
    # Set log level to INFO to see debug output
    os.environ['VASA_LOG_LEVEL'] = 'INFO'

    # Initialize tester
    logger.info("Initializing VASA Alignment Tester...")
    tester = VASAAlignmentTester()

    # Test inputs
    test_video = "nemo/data/VID_2.mp4"
    test_image = "data/VID_2_source.png"

    # Extract test audio from video
    logger.info("Extracting audio from video...")
    audio_path = tester.extract_audio_from_video(test_video)

    # Run Test 0: Audio-Driven Motion Variance Test
    logger.info("Running Test 0: Audio-Driven Motion Variance Test")
    try:
        result = tester.test_0_audio_motion_variance(test_image, audio_path)

        logger.info("\n=== Test 0 Results ===")
        if result.get('test_passed', False):
            logger.info("✓ PASS: Audio successfully drives motion generation")
        else:
            logger.warning("✗ FAIL: Audio does not significantly affect motion generation")

        # Print detailed results
        if 'expression' in result:
            expr = result['expression']
            logger.info(f"Expression variance ratio: {expr['var_ratio']:.2f}x (Silent: {expr['silent_var']:.6f}, Speech: {expr['speech_var']:.6f})")

        if 'temporal' in result:
            temp = result['temporal']
            logger.info(f"Temporal variation difference: {temp['temporal_diff']:.6f} (Silent: {temp['silent_temporal']:.6f}, Speech: {temp['speech_temporal']:.6f})")

    except Exception as e:
        logger.error(f"Test 0 failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()