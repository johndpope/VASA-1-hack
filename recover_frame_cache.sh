#!/bin/bash
# Recover disk cache (PNG files) from H5 cache
# Use this when H5 cache exists but disk frames are missing

echo "🔧 Frame Cache Recovery Tool"
echo "================================"
echo ""
echo "This script will recreate missing PNG/JPG frame files from the H5 cache."
echo "It will skip frames that already exist on disk."
echo ""

# Default settings
CACHE_DIR="cache_single_bucket"
FRAME_FORMAT="png"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --frame-format)
            FRAME_FORMAT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cache-dir DIR      Cache directory (default: cache_single_bucket)"
            echo "  --frame-format FMT   Image format: png or jpg (default: png)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use defaults"
            echo "  $0 --cache-dir my_cache --frame-format jpg"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Cache directory: $CACHE_DIR"
echo "  Frame format: $FRAME_FORMAT"
echo ""

# Check if H5 cache exists
H5_FILE="$CACHE_DIR/all_windows_cache.h5"
if [ ! -f "$H5_FILE" ]; then
    echo "❌ Error: H5 cache not found at $H5_FILE"
    echo "   Please run preprocessing first to create the H5 cache."
    exit 1
fi

echo "✅ Found H5 cache: $H5_FILE"
echo ""
echo "Starting recovery..."
echo ""

# Run recovery
python preprocess_single_bucket.py \
    --recover \
    --frame-format "$FRAME_FORMAT" \
    --cache-frames \
    --cache-emo-frames

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Recovery completed successfully!"
    echo ""
    echo "You can now resume training with the recovered frames."
else
    echo ""
    echo "❌ Recovery failed! Check the logs above for errors."
    exit 1
fi
