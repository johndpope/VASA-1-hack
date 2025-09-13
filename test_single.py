#!/usr/bin/env python3
"""Run single test from VASA alignment suite"""

from test_vasa_alignment import VASAAlignmentTester

def main():
    # Initialize tester
    tester = VASAAlignmentTester()

    # Extract test audio from video
    test_video = "nemo/data/VID_2.mp4"
    audio_path = tester.extract_audio_from_video(test_video)

    # Run Test 1: Audio Feature Extraction
    result = tester.test_1_audio_feature_extraction(audio_path)
    print(f"\nTest 1 Result: {result}")

if __name__ == "__main__":
    main()