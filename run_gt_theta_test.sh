#!/bin/bash
# Complete pipeline for GT theta sanity check

VIDEO="./junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4"
TARGET_IMAGE="./data/IMG_1.png"
CHECKPOINT="./checkpoints/best_checkpoint.pt"
CONFIG="vasa_config.yaml"
H5_CACHE="cache/videovideoeI2V8Bd5X9s-scene6_scene1_gt_theta.h5"
OUTPUT="test_gt_theta_from_cache.mp4"

echo "=========================================="
echo "GT THETA SANITY CHECK PIPELINE"
echo "=========================================="
echo ""

# Step 1: Extract and cache GT theta (only runs once)
echo "Step 1: Extracting GT theta to H5 cache..."
python extract_and_cache_gt_theta.py \
    --video "$VIDEO" \
    --output "$H5_CACHE" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "❌ Failed to extract GT theta"
    exit 1
fi

echo ""
echo "Step 2: Running inference with cached GT theta..."
python infer_with_cached_gt_theta.py \
    --input "$VIDEO" \
    --gt-theta-h5 "$H5_CACHE" \
    --target_image "$TARGET_IMAGE" \
    --output "$OUTPUT" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "❌ Failed to run inference"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ GT theta test complete!"
echo "Output: $OUTPUT"
echo "=========================================="
