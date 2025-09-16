#!/bin/bash

# Simple script to run VASA-1 pipeline
cd nemo

echo "Running VASA-1 Video-driven Generation"
echo "======================================"
echo "Source image: data/IMG_1.png"
echo "Driving video: ../junk/15.mp4"

python pipeline2.py \
    --source_image_path data/IMG_1.png \
    --driven_video_path ../junk/15.mp4 \
    --saved_to_path data/result.mp4

echo ""
echo "✅ Complete! Output: nemo/data/result.mp4"

# Show file size
ls -lh data/result.mp4