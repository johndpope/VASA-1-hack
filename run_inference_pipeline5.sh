#!/bin/bash

# Pipeline4 Inference Script
# Enhanced version with motion warp extraction and driving capabilities

# Default values
SOURCE_IMAGE=""
DRIVEN_VIDEO=""
OUTPUT_PATH="output/result_pipeline4.mp4"
MAX_LEN=1000
FPS=25.0
MODE="drive"  # Options: drive, extract, apply
WARPS_FILE=""
USE_UV_WARPS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Pipeline4 inference script with motion warp capabilities"
    echo ""
    echo "OPTIONS:"
    echo "  -s, --source IMAGE       Source image path (required for drive/apply modes)"
    echo "  -d, --driven VIDEO       Driving video path (required for drive/extract modes)"
    echo "  -o, --output PATH        Output path (default: output/result_pipeline4.mp4)"
    echo "  -m, --max-len NUM        Maximum frames to process (default: 1000)"
    echo "  -f, --fps FPS           Output video FPS (default: 25.0)"
    echo "  --mode MODE             Operation mode: drive|extract|apply (default: drive)"
    echo "  --extract-warps FILE    Extract warps to H5 file (sets mode to extract)"
    echo "  --apply-warps FILE      Apply warps from H5 file (sets mode to apply)"
    echo "  --no-uv-warps          Disable UV warps (rigid only)"
    echo "  -h, --help             Display this help message"
    echo ""
    echo "MODES:"
    echo "  drive   - Standard video driving (requires -s and -d)"
    echo "  extract - Extract motion warps to H5 (requires -d and --extract-warps)"
    echo "  apply   - Apply saved warps (requires -s and --apply-warps)"
    echo ""
    echo "EXAMPLES:"
    echo "  # Standard video driving"
    echo "  $0 -s data/source.jpg -d data/driver.mp4 -o output/result.mp4"
    echo ""
    echo "  # Extract motion warps"
    echo "  $0 --mode extract -d data/driver.mp4 --extract-warps warps.h5"
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
        --no-uv-warps)
            USE_UV_WARPS=false
            shift
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
    drive)
        if [ -z "$SOURCE_IMAGE" ] || [ -z "$DRIVEN_VIDEO" ]; then
            echo -e "${RED}Error: Drive mode requires both source image (-s) and driving video (-d)${NC}"
            usage
        fi
        check_file "$SOURCE_IMAGE"
        check_file "$DRIVEN_VIDEO"
        ensure_output_dir "$OUTPUT_PATH"
        ;;
    extract)
        if [ -z "$DRIVEN_VIDEO" ] || [ -z "$WARPS_FILE" ]; then
            echo -e "${RED}Error: Extract mode requires driving video (-d) and output warps file (--extract-warps)${NC}"
            usage
        fi
        check_file "$DRIVEN_VIDEO"
        ensure_output_dir "$WARPS_FILE"
        ;;
    apply)
        if [ -z "$SOURCE_IMAGE" ] || [ -z "$WARPS_FILE" ]; then
            echo -e "${RED}Error: Apply mode requires source image (-s) and warps file (--apply-warps)${NC}"
            usage
        fi
        check_file "$SOURCE_IMAGE"
        check_file "$WARPS_FILE"
        ensure_output_dir "$OUTPUT_PATH"
        ;;
    *)
        echo -e "${RED}Error: Invalid mode: $MODE${NC}"
        echo "Valid modes are: drive, extract, apply"
        exit 1
        ;;
esac

# Build command based on mode
echo -e "${GREEN}Running Pipeline4 in $MODE mode...${NC}"
echo "======================================"

case $MODE in
    drive)
        echo "Source Image: $SOURCE_IMAGE"
        echo "Driving Video: $DRIVEN_VIDEO"
        echo "Output: $OUTPUT_PATH"
        echo "Max Frames: $MAX_LEN"
        echo "FPS: $FPS"
        echo ""

        CMD="python nemo/pipeline4.py \
            --source_image_path \"$SOURCE_IMAGE\" \
            --driven_video_path \"$DRIVEN_VIDEO\" \
            --saved_to_path \"$OUTPUT_PATH\" \
            --max_len $MAX_LEN \
            --fps $FPS"
        ;;

    extract)
        echo "Driving Video: $DRIVEN_VIDEO"
        echo "Output Warps: $WARPS_FILE"
        echo "Max Frames: $MAX_LEN"
        echo ""

        CMD="python nemo/pipeline4.py \
            --driven_video_path \"$DRIVEN_VIDEO\" \
            --extract-warps \"$WARPS_FILE\" \
            --max_len $MAX_LEN"
        ;;

    apply)
        echo "Source Image: $SOURCE_IMAGE"
        echo "Warps File: $WARPS_FILE"
        echo "Output: $OUTPUT_PATH"
        echo "FPS: $FPS"
        echo -n "UV Warps: "
        if [ "$USE_UV_WARPS" = true ]; then
            echo "Enabled"
        else
            echo "Disabled (rigid only)"
        fi
        echo ""

        CMD="python nemo/pipeline4.py \
            --source_image_path \"$SOURCE_IMAGE\" \
            --drive-with-warps \"$WARPS_FILE\" \
            --saved_to_path \"$OUTPUT_PATH\" \
            --fps $FPS"

        if [ "$USE_UV_WARPS" = true ]; then
            CMD="$CMD --use-uv-warps"
        fi
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