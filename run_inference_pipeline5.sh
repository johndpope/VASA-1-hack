#!/bin/bash

# Pipeline5 Inference Script
# Enhanced version with motion warp extraction and driving capabilities

# Default values
SOURCE_IMAGE=""
DRIVEN_VIDEO=""
OUTPUT_PATH="output/result_pipeline5.mp4"
MAX_LEN=1000
FPS=25.0
MODE="drive"  # Options: drive, extract, apply
WARPS_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Pipeline5 inference script with motion warp capabilities"
    echo ""
    echo "OPTIONS:"
    echo "  -s, --source IMAGE       Source image path (required for all modes)"
    echo "  -d, --driven VIDEO       Driving video path (required for drive/extract modes)"
    echo "  -o, --output PATH        Output path (default: output/result_pipeline5.mp4)"
    echo "  -m, --max-len NUM        Maximum frames to process (default: 1000)"
    echo "  -f, --fps FPS           Output video FPS (default: 25.0)"
    echo "  --mode MODE             Operation mode: drive|extract|apply (default: drive)"
    echo "  --extract-warps FILE    Extract warps to H5 file (sets mode to extract)"
    echo "  --apply-warps FILE      Apply warps from H5 file (sets mode to apply)"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "MODES:"
    echo "  drive   - Standard video driving (requires -s and -d)"
    echo "  extract - Extract motion warps to H5 (requires -s, -d and --extract-warps)"
    echo "  apply   - Apply saved warps (requires -s and --apply-warps)"
    echo ""
    echo "EXAMPLES:"
    echo "  # Standard video driving"
    echo "  $0 -s data/source.jpg -d data/driver.mp4 -o output/result.mp4"
    echo ""
    echo "  # Extract motion warps"
    echo "  $0 --mode extract -s data/source.jpg -d data/driver.mp4 --extract-warps warps.h5"
    echo ""
    echo "  # Apply saved warps"
    echo "  $0 --mode apply -s data/source.jpg --apply-warps warps.h5 -o output/driven.mp4"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SOURCE_IMAGE="$2"
            shift 2
            ;;
        -d|--driven)
            DRIVEN_VIDEO="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -m|--max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --extract-warps)
            MODE="extract"
            WARPS_FILE="$2"
            shift 2
            ;;
        --apply-warps)
            MODE="apply"
            WARPS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to check file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File not found: $1${NC}"
        exit 1
    fi
}

# Function to create output directory if needed
ensure_output_dir() {
    OUTPUT_DIR=$(dirname "$1")
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"
    fi
}

# Validate inputs based on mode
case $MODE in
    drive|extract|apply)
        if [ -z "$SOURCE_IMAGE" ]; then
            echo -e "${RED}Error: All modes require source image (-s)${NC}"
            usage
        fi
        check_file "$SOURCE_IMAGE"
        ;;
esac

case $MODE in
    drive|extract)
        if [ -z "$DRIVEN_VIDEO" ]; then
            echo -e "${RED}Error: Drive/Extract modes require driving video (-d)${NC}"
            usage
        fi
        check_file "$DRIVEN_VIDEO"
        ;;
    apply)
        if [ -z "$WARPS_FILE" ]; then
            echo -e "${RED}Error: Apply mode requires warps file (--apply-warps)${NC}"
            usage
        fi
        check_file "$WARPS_FILE"
        ensure_output_dir "$OUTPUT_PATH"
        ;;
esac

if [ "$MODE" = "extract" ] && [ -z "$WARPS_FILE" ]; then
    echo -e "${RED}Error: Extract mode requires --extract-warps FILE${NC}"
    usage
fi

# Build command based on mode
echo -e "${GREEN}Running Pipeline5 in $MODE mode...${NC}"
echo "======================================"

case $MODE in
    drive)
        echo "Source Image: $SOURCE_IMAGE"
        echo "Driving Video: $DRIVEN_VIDEO"
        echo "Output: $OUTPUT_PATH"
        echo "Max Frames: $MAX_LEN"
        echo "FPS: $FPS"
        echo ""

        CMD="python nemo/pipeline5.py \
            --source_image_path \"$SOURCE_IMAGE\" \
            --driven_video_path \"$DRIVEN_VIDEO\" \
            --saved_to_path \"$OUTPUT_PATH\" \
            --max_len $MAX_LEN \
            --fps $FPS"
        ;;

    extract)
        echo "Source Image: $SOURCE_IMAGE"
        echo "Driving Video: $DRIVEN_VIDEO"
        echo "Output Warps: $WARPS_FILE"
        echo "Max Frames: $MAX_LEN"
        echo ""

        CMD="python nemo/pipeline5.py \
            --source_image_path \"$SOURCE_IMAGE\" \
            --driven_video_path \"$DRIVEN_VIDEO\" \
            --cache_h5_path \"$WARPS_FILE\" \
            --max_len $MAX_LEN"
        ;;

    apply)
        echo "Source Image: $SOURCE_IMAGE"
        echo "Warps File: $WARPS_FILE"
        echo "Output: $OUTPUT_PATH"
        echo "FPS: $FPS"
        echo "Num Frames: $MAX_LEN"
        echo ""

        CMD="python nemo/pipeline5.py \
            --source_image_path \"$SOURCE_IMAGE\" \
            --load_h5_path \"$WARPS_FILE\" \
            --saved_to_path \"$OUTPUT_PATH\" \
            --fps $FPS \
            --num_frames $MAX_LEN"
        ;;
esac

# Execute command
echo "Executing command:"
echo "$CMD"
echo "======================================"

# Set up Python environment if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run with timing
start_time=$(date +%s)
eval $CMD
exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

# Report results
echo "======================================"
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}Success! Processing completed in ${duration} seconds${NC}"

    case $MODE in
        drive|apply)
            if [ -f "$OUTPUT_PATH" ]; then
                echo "Output saved to: $OUTPUT_PATH"
                # Get video info if ffmpeg is available
                if command -v ffprobe &> /dev/null; then
                    echo ""
                    echo "Video Info:"
                    ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration -of default=noprint_wrappers=1 "$OUTPUT_PATH"
                fi
            fi
            ;;
        extract)
            if [ -f "$WARPS_FILE" ]; then
                echo "Warps saved to: $WARPS_FILE"
                # Show H5 file size
                echo "File size: $(du -h "$WARPS_FILE" | cut -f1)"
            fi
            ;;
    esac
else
    echo -e "${RED}Error: Processing failed with exit code $exit_code${NC}"
    exit $exit_code
fi