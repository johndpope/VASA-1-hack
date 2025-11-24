#!/bin/bash

# Action Unit Diagnostics Runner
# Wrapper script for diagnose_au.py with common defaults

echo "================================"
echo "   AU Diagnostics Tool"
echo "================================"
echo ""

# Default paths
DEFAULT_VIDEO="junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4"
DEFAULT_IDENTITY="./data/IMG_1.png"
DEFAULT_CONFIG="overfit_config.yaml"
DEFAULT_CHECKPOINT="checkpoints_overfit/best_checkpoint.pt"
DEFAULT_OUTPUT="au_diagnostics"

# Prompt for inputs (with defaults)
read -p "Video path [$DEFAULT_VIDEO]: " VIDEO
VIDEO=${VIDEO:-$DEFAULT_VIDEO}

read -p "Identity image [$DEFAULT_IDENTITY]: " IDENTITY
IDENTITY=${IDENTITY:-$DEFAULT_IDENTITY}

read -p "Config file [$DEFAULT_CONFIG]: " CONFIG
CONFIG=${CONFIG:-$DEFAULT_CONFIG}

read -p "Checkpoint [$DEFAULT_CHECKPOINT]: " CHECKPOINT
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}

read -p "Output directory [$DEFAULT_OUTPUT]: " OUTPUT
OUTPUT=${OUTPUT:-$DEFAULT_OUTPUT}

echo ""
echo "Running AU diagnostics with:"
echo "  Video: $VIDEO"
echo "  Identity: $IDENTITY"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT"
echo ""

# Check if files exist
if [ ! -f "$VIDEO" ]; then
    echo "❌ Error: Video file not found: $VIDEO"
    exit 1
fi

if [ ! -f "$IDENTITY" ]; then
    echo "❌ Error: Identity file not found: $IDENTITY"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Run diagnostics
python diagnose_au.py \
    --video "$VIDEO" \
    --identity "$IDENTITY" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Diagnostics complete!"
    echo ""
    echo "View results:"
    echo "  - Metrics: $OUTPUT/au_metrics.csv"
    echo "  - MAE chart: $OUTPUT/au_mae_by_unit.png"
    echo "  - Correlation: $OUTPUT/au_correlation_by_unit.png"
    echo "  - Samples: $OUTPUT/au_window_*.png"
else
    echo ""
    echo "❌ Diagnostics failed. Check the error messages above."
    exit 1
fi
