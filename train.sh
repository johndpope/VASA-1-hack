#!/bin/bash

# VASA-1 Training Runner
# Allows user to choose between different config files

echo "================================"
echo "   VASA-1 Training Runner"
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
echo "Select training mode:"
echo "1) Train from scratch"
echo "2) Resume from checkpoint"
echo "3) Fine-tune from checkpoint"
echo ""
read -p "Enter your choice (1-3): " mode_choice

case $mode_choice in
    1)
        RESUME_FLAG=""
        echo "Training from scratch..."
        ;;
    2)
        RESUME_FLAG="--resume"
        echo "Resuming from checkpoint..."
        ;;
    3)
        RESUME_FLAG="--resume --reset-optimizer"
        echo "Fine-tuning from checkpoint (reset optimizer)..."
        ;;
    *)
        RESUME_FLAG=""
        echo "Defaulting to train from scratch..."
        ;;
esac

echo ""
echo "Select log level:"
echo "1) ERROR (minimal output)"
echo "2) WARNING (warnings and errors)"
echo "3) INFO (normal logging)"
echo "4) DEBUG (verbose logging)"
echo ""
read -p "Enter your choice (1-4): " log_choice

case $log_choice in
    1)
        export VASA_LOG_LEVEL="ERROR"
        echo "Log level set to ERROR (minimal output)..."
        ;;
    2)
        export VASA_LOG_LEVEL="WARNING"
        echo "Log level set to WARNING..."
        ;;
    3)
        export VASA_LOG_LEVEL="INFO"
        echo "Log level set to INFO (normal)..."
        ;;
    4)
        export VASA_LOG_LEVEL="DEBUG"
        echo "Log level set to DEBUG (verbose)..."
        ;;
    *)
        export VASA_LOG_LEVEL="INFO"
        echo "Defaulting to INFO log level..."
        ;;
esac

echo ""
echo "Additional options:"
echo "1) Normal training"
echo "2) Fast mode (reduced validation)"
echo ""
read -p "Enter your choice (1-2): " debug_choice

case $debug_choice in
    2)
        FAST_FLAG="--fast"
        echo "Fast mode enabled..."
        ;;
    *)
        FAST_FLAG=""
        echo "Normal training mode..."
        ;;
esac

echo ""
echo "================================"
echo "Starting training with:"
echo "  Config: $CONFIG_FILE"
echo "  Mode: $mode_choice"
echo "  Log Level: $VASA_LOG_LEVEL"
echo "  Options: $debug_choice"
echo "================================"
echo ""

# Construct the command based on config file
if [ "$CONFIG_FILE" == "overfit_config.yaml" ]; then
    # Use dedicated overfit training script
    COMMAND="python train_overfit.py"
else
    # Use regular trainer for other configs
    COMMAND="python vasa_trainer.py --config $CONFIG_FILE"
fi

# Note: Resume is handled via config file's resume_from field
# Debug and fast modes would need to be implemented in the training scripts

# Show the command being run
echo "Running: $COMMAND"
echo ""

# Execute the training
$COMMAND

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "Training completed successfully!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "Training failed or was interrupted."
    echo "================================"
    exit 1
fi