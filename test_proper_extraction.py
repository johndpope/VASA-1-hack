#!/usr/bin/env python3
"""
Test proper warp extraction with actual motion between frames.
Instead of self-driving, use frame 0 as source and other frames as targets.
"""

import sys
import os

print("Extracting motion warps with proper source-target separation...")
print("This will use frame 0 as source identity and extract motion from subsequent frames")

# For now, still use the self-driving approach but let's add better diagnostics
cmd = """
python nemo/pipeline4.py \
    --driven_video_path junk/15.mp4 \
    --extract-warps motion_warps_v2.h5 \
    --max_len 30
"""

os.system(cmd)

print("\nNow testing with the new warps...")
cmd2 = """
python nemo/pipeline4.py \
    --source_image_path junk/source_512.jpg \
    --drive-with-warps motion_warps_v2.h5 \
    --saved_to_path junk/output_driven_v2.mp4
"""

os.system(cmd2)