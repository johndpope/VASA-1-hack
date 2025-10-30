#!/bin/bash

# Action Unit Ground Truth Upsert Wrapper
# Updates existing cache files with AU ground truth data

echo "================================"
echo "   AU Ground Truth Upsert"
echo "================================"
echo ""
echo "This script updates existing cache files with AU ground truth."
echo "It will extract AUs from video frames and add them to the cache."
echo ""

# Default values
DEFAULT_CACHE_DIR="cache_per_video"
FORCE_FLAG=""
VIDEO_FILTER=""

# Parse arguments or prompt
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [cache_dir] [--force] [--video-filter PATTERN]"
    echo ""
    echo "Options:"
    echo "  cache_dir         Cache directory (default: cache_per_video)"
    echo "  --force           Force update even if AU GT already exists"
    echo "  --video-filter    Only process caches matching video name pattern"
    echo ""
    echo "Examples:"
    echo "  $0                              # Update all caches"
    echo "  $0 --force                      # Force update all"
    echo "  $0 --video-filter scene6        # Only update scene6 videos"
    exit 0
fi

# Get cache directory
if [ -n "$1" ] && [ "$1" != "--force" ] && [ "$1" != "--video-filter" ]; then
    CACHE_DIR="$1"
    shift
else
    read -p "Cache directory [$DEFAULT_CACHE_DIR]: " CACHE_DIR
    CACHE_DIR=${CACHE_DIR:-$DEFAULT_CACHE_DIR}
fi

# Check for --force flag
if [ "$1" = "--force" ]; then
    FORCE_FLAG="--force"
    echo "⚠️  Force mode enabled - will overwrite existing AU data"
    shift
fi

# Check for --video-filter
if [ "$1" = "--video-filter" ]; then
    shift
    VIDEO_FILTER="--video-filter $1"
    echo "🔍 Filtering for videos matching: $1"
    shift
fi

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Error: Cache directory not found: $CACHE_DIR"
    exit 1
fi

# Count cache files
CACHE_COUNT=$(find "$CACHE_DIR" -name "metadata.h5" | wc -l)
if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "❌ Error: No cache files found in $CACHE_DIR"
    exit 1
fi

echo ""
echo "Found $CACHE_COUNT cache files in $CACHE_DIR"
echo ""

# Confirm before proceeding
read -p "Continue with AU upsert? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting AU upsert..."
echo "This may take a while depending on the number of videos."
echo ""

# Run the upsert script
python upsert_au_gt.py \
    --cache-dir "$CACHE_DIR" \
    $FORCE_FLAG \
    $VIDEO_FILTER

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ AU upsert complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Start training: ./safe-train.sh"
    echo "  2. Check WandB for AU visualizations"
    echo "  3. Run diagnostics: ./diagnose_au.sh"
else
    echo ""
    echo "❌ AU upsert failed with exit code $EXIT_CODE"
    echo "Check the log output above for errors."
    exit $EXIT_CODE
fi
