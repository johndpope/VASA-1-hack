#!/usr/bin/env python3
"""Run Test 0: Audio Motion Variance Test"""

from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger

def main():
    # Initialize tester
    logger.info("Initializing VASA Alignment Tester...")
    tester = VASAAlignmentTester()

    # Test inputs
    test_video = "nemo/data/VID_2.mp4"
    test_image = "data/VID_2_source.png"

    # Extract test audio from video
    logger.info("Extracting audio from test video...")
    audio_path = tester.extract_audio_from_video(test_video)

    # Run Test 0: Audio Motion Variance Test
    try:
        logger.info("Running Test 0: Audio-Driven Motion Variance Test...")
        result = tester.test_0_audio_motion_variance(test_image, audio_path)

        if result.get('test_passed', False):
            logger.info("✓ Test 0 PASSED: Audio successfully drives motion generation")
        else:
            logger.warning("✗ Test 0 FAILED: Audio does not significantly affect motion generation")

        # Print detailed results
        logger.info("\n=== Detailed Results ===")
        if 'expression' in result:
            logger.info(f"Expression variance ratio: {result['expression']['var_ratio']:.2f}x")
            logger.info(f"Expression variance difference: {result['expression']['var_diff']:.6f}")

        if 'temporal' in result:
            logger.info(f"Temporal variation ratio: {result['temporal']['speech_temporal'] / (result['temporal']['silent_temporal'] + 1e-8):.2f}x")
            logger.info(f"Temporal variation difference: {result['temporal']['temporal_diff']:.6f}")

        if 'theta' in result:
            logger.info(f"Theta variance ratio: {result['theta']['var_ratio']:.2f}x")
            logger.info(f"Theta variance difference: {result['theta']['var_diff']:.6f}")

    except Exception as e:
        logger.error(f"Test 0 failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()