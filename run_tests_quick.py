#!/usr/bin/env python3
"""Run VASA alignment tests quickly (Tests 1-3 only)"""

from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger

def main():
    # Initialize tester
    tester = VASAAlignmentTester()

    # Test inputs
    test_video = "nemo/data/VID_2.mp4"
    test_image = "data/VID_2_source.png"

    # Extract test audio from video
    audio_path = tester.extract_audio_from_video(test_video)

    results = {}

    # Test 1: Audio Feature Extraction
    try:
        logger.info("Running Test 1...")
        results['test_1'] = tester.test_1_audio_feature_extraction(audio_path)
        logger.info(f"Test 1: PASSED")
    except Exception as e:
        logger.error(f"Test 1: FAILED - {e}")
        results['test_1'] = {'status': 'failed', 'error': str(e)}

    # Test 2: Motion Generation
    try:
        logger.info("Running Test 2...")
        results['test_2'] = tester.test_2_motion_generation(test_image, audio_path)
        logger.info(f"Test 2: PASSED")
    except Exception as e:
        logger.error(f"Test 2: FAILED - {e}")
        results['test_2'] = {'status': 'failed', 'error': str(e)}

    # Test 3: Isolated Rendering
    try:
        logger.info("Running Test 3...")
        results['test_3'] = tester.test_3_isolated_rendering(test_image, num_frames=10)
        logger.info(f"Test 3: PASSED")
    except Exception as e:
        logger.error(f"Test 3: FAILED - {e}")
        results['test_3'] = {'status': 'failed', 'error': str(e)}

    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, result in results.items():
        if isinstance(result, dict) and 'status' in result and result['status'] == 'failed':
            logger.error(f"{test_name}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"{test_name}: PASSED")

if __name__ == "__main__":
    main()