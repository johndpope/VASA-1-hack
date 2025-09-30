#!/usr/bin/env python3
"""Test driving with better debugging"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os
sys.path.append('.')

# Load source image and ensure it's 512x512
source = Image.open('junk/source.jpg')
print(f"Original source size: {source.size}")

# Center crop to square
width, height = source.size
size = min(width, height)
left = (width - size) // 2
top = (height - size) // 2
source_cropped = source.crop((left, top, left + size, top + size))
source_512 = source_cropped.resize((512, 512), Image.LANCZOS)
source_512.save('junk/source_512.jpg')
print(f"Saved cropped/resized source: 512x512")

# Now test with properly sized image
print("\nTesting with corrected source image...")
cmd = "python nemo/pipeline4.py --source_image_path junk/source_512.jpg --drive-with-warps motion_warps.h5 --saved_to_path junk/output_driven_fixed.mp4"
os.system(cmd)