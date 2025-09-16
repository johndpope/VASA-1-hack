#!/bin/bash

# Helper script to demonstrate pipeline functionality
# This script runs the working pipeline2.py to show video-driven avatar generation

cd nemo

# Run the standard video-driven generation using pipeline2.py
echo "========================================="
echo "Running VASA-1 Video-driven Generation"
echo "========================================="
echo ""
echo "Source image: data/IMG_1.png"
echo "Driving video: ../junk/15.mp4"
echo ""

python pipeline2.py \
    --source_image_path data/IMG_1.png \
    --driven_video_path ../junk/15.mp4 \
    --saved_to_path data/result.mp4

echo ""
echo "✅ Video generation complete!"
echo "Output saved to: nemo/data/result.mp4"

# Mode 2: Try with cached attributes if available
# Note: This requires pre-generated cache files from preprocessing
CACHE_DIR="../cache_single_bucket"
if [ -d "$CACHE_DIR" ] && [ "$(ls -A $CACHE_DIR/*.h5 2>/dev/null)" ]; then
    echo ""
    echo "========================================="
    echo "Mode 2: Cached attributes generation"
    echo "========================================="

    # Get first available H5 file
    FIRST_H5=$(ls $CACHE_DIR/*.h5 2>/dev/null | head -1)

    if [ -n "$FIRST_H5" ]; then
        echo "Using cached attributes from: $FIRST_H5"
        python pipeline_face_attr.py \
            --source_image_path data/IMG_1.png \
            --face_attrs_h5 "$FIRST_H5" \
            --window_idx 0 \
            --saved_to_path data/result_cached_attrs.mp4 \
            --max_len 50

        echo "Cached attributes result saved to: nemo/data/result_cached_attrs.mp4"
    fi
else
    echo ""
    echo "========================================="
    echo "Note: No cached attributes found"
    echo "========================================="
    echo "Cache directory not found or empty: $CACHE_DIR"
    echo "To generate cached attributes, you need to run the preprocessing pipeline"
    echo "which extracts face attributes from videos and saves them to H5 files."
    echo ""
    echo "For now, only video-driven mode is available."
fi

echo ""
echo "========================================="
echo "Comparing outputs (if both exist):"
echo "========================================="
if [ -f "data/result_video_driven.mp4" ]; then
    ls -lh data/result_video_driven.mp4
fi
if [ -f "data/result_cached_attrs.mp4" ]; then
    ls -lh data/result_cached_attrs.mp4
fi