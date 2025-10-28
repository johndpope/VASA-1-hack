#!/bin/bash

# VASA-1 Inference Runner
# Allows user to choose between different config files

echo "================================"
echo "   VASA-1 Inference Runner"
echo "================================"
echo ""
echo "Select configuration file:"
echo "1) overfit_config.yaml (for overfitting/testing)"
echo "2) vasa_config.yaml (for full training)"
echo "3) Custom config file"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        CONFIG_FILE="overfit_config.yaml"
        echo "Using overfit configuration..."
        ;;
    2)
        CONFIG_FILE="vasa_config.yaml"
        echo "Using full training configuration..."
        ;;
    3)
        read -p "Enter custom config file path: " CONFIG_FILE
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "Error: Config file '$CONFIG_FILE' not found!"
            exit 1
        fi
        echo "Using custom configuration: $CONFIG_FILE"
        ;;
    *)
        echo "Invalid choice! Defaulting to vasa_config.yaml"
        CONFIG_FILE="vasa_config.yaml"
        ;;
esac

echo ""
echo "Use ground truth theta for inference?"
echo "1) No - predict theta from model (default)"
echo "2) Yes - use GT theta from H5 cache (vi_v2.py)"
echo ""
read -p "Enter your choice (1-2): " theta_choice

# Set inference script based on theta choice
if [ "$theta_choice" = "2" ]; then
    INFERENCE_SCRIPT="vi_v2.py"
    USE_GT_THETA=true
    echo "Will use ground truth theta from H5 cache..."
    echo ""
    read -p "Enter path to GT theta H5 file: " GT_THETA_H5
    if [ ! -f "$GT_THETA_H5" ]; then
        echo "Error: GT theta H5 file '$GT_THETA_H5' not found!"
        exit 1
    fi
else
    INFERENCE_SCRIPT="vi.py"
    USE_GT_THETA=false
    echo "Will predict theta from model..."
fi

echo ""
echo "Select output mode:"
echo "1) Generate video"
echo "2) Generate visualizations"
echo ""
read -p "Enter your choice (1-2): " mode_choice

echo ""
echo "Starting inference with config: $CONFIG_FILE"
echo "Inference script: $INFERENCE_SCRIPT"
if [ "$USE_GT_THETA" = true ]; then
    echo "GT Theta H5: $GT_THETA_H5"
fi
echo "================================"
echo ""

# Run the inference script with the selected config and mode
if [ "$USE_GT_THETA" = true ]; then
    # Using vi_v2.py with GT theta
    if [ "$mode_choice" = "2" ]; then
        echo "Generating visualizations with GT theta..."
        python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE" --visualize --gt-theta-h5 "$GT_THETA_H5"
    else
        echo "Generating video with GT theta..."
        python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE" --gt-theta-h5 "$GT_THETA_H5"
    fi
else
    # Using vi.py with predicted theta
    if [ "$mode_choice" = "2" ]; then
        echo "Generating visualizations..."
        python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE" --visualize
    else
        echo "Generating video..."
        python "$INFERENCE_SCRIPT" --config "$CONFIG_FILE"
    fi
fi