#!/bin/bash
# Diagnose phoneme predictions for cached windows

set -e

CACHE_DIR="${1:-cache_per_video}"
VIDEO="${2:-}"
CHECKPOINT="${3:-checkpoints_overfit/best_checkpoint.pt}"

echo "=================================="
echo "PHONEME DIAGNOSIS"
echo "=================================="
echo "Cache directory: $CACHE_DIR"
echo ""

if [ -n "$VIDEO" ]; then
    echo "Analyzing specific video: $VIDEO"
    python diagnose_phoneme.py \
        --cache_dir "$CACHE_DIR" \
        --video "$VIDEO" \
        --checkpoint "$CHECKPOINT" \
        --max_windows 10
else
    echo "Scanning cache for videos..."
    python diagnose_phoneme.py \
        --cache_dir "$CACHE_DIR" \
        --checkpoint "$CHECKPOINT" \
        --max_videos 3 \
        --max_windows 5
fi

echo ""
echo "✅ Done!"
