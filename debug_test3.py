#!/usr/bin/env python3
"""Debug Test 3 specifically"""

from test_vasa_alignment import VASAAlignmentTester
from nemo.logger import logger
import traceback

def main():
    # Initialize tester
    tester = VASAAlignmentTester()

    test_image = "data/VID_2_source.png"

    # Test 3: Isolated Rendering
    try:
        logger.info("Running Test 3 with detailed error tracking...")
        result = tester.test_3_isolated_rendering(test_image, num_frames=5)
        logger.info(f"Test 3: PASSED")
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Test 3: FAILED - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()