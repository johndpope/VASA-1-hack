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
echo "Starting inference with config: $CONFIG_FILE"
echo "================================"
echo ""

# Run the inference script with the selected config
python vi.py --config "$CONFIG_FILE"