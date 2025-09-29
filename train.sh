#!/bin/bash

# VASA-1 Training Runner
# Allows user to choose between different config files

echo "================================"
echo "   VASA-1 Training Runner"
echo "================================"
echo ""
echo "Select configuration file:"
echo "1) overfit_config.yaml (for overfitting/testing) [DEFAULT]"
echo "2) vasa_config.yaml (for full training)"
echo "3) Custom config file"
echo ""
read -p "Enter your choice (1-3, press Enter for default): " choice

# Default to overfitting when pressing enter or invalid input
case $choice in
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
    1|""|*)
        CONFIG_FILE="overfit_config.yaml"
        echo "Using overfit configuration (default)..."
        ;;
esac

# Determine checkpoint directory based on config
if [ "$CONFIG_FILE" == "overfit_config.yaml" ]; then
    CHECKPOINT_DIR="checkpoints_overfit"
else
    CHECKPOINT_DIR="checkpoints_overfit"
fi

# Auto-detect and use existing checkpoint
if [ -f "$CHECKPOINT_DIR/best_checkpoint.pt" ]; then
    RESUME_PATH="$CHECKPOINT_DIR/best_checkpoint.pt"
    echo ""
    echo "✅ Found existing checkpoint: $RESUME_PATH"
    echo "   Resuming training from this checkpoint..."
elif [ -f "$CHECKPOINT_DIR/latest_checkpoint.pt" ]; then
    RESUME_PATH="$CHECKPOINT_DIR/latest_checkpoint.pt"
    echo ""
    echo "✅ Found existing checkpoint: $RESUME_PATH"
    echo "   Resuming training from this checkpoint..."
else
    RESUME_PATH=""
    echo ""
    echo "📝 No checkpoint found in $CHECKPOINT_DIR/"
    echo "   Starting training from scratch..."
fi

# Optional: Allow override to start fresh
echo ""
read -p "Override and start from scratch? (y/N, press Enter for No): " override_choice
if [ "$override_choice" = "y" ] || [ "$override_choice" = "Y" ]; then
    RESUME_PATH=""
    echo "🔄 Override selected - will train from scratch"
else
    echo "📂 Keeping existing checkpoint configuration"
fi

echo ""
# Ask if user wants to clear old cache
read -p "Clear old cache directories? (y/N, press Enter for No): " clear_cache
if [ "$clear_cache" = "y" ] || [ "$clear_cache" = "Y" ]; then
    echo "🗑️  Cleaning old cache directories..."
    rm -rf cache_single_bucket/
    rm -rf cache/
    rm -rf window_cache*/
    echo "✅ Cache directories cleared."
else
    echo "📦 Keeping existing cache directories"
fi

echo ""
echo "Select log level:"
echo "1) INFO (normal logging)"
echo "2) ERROR (minimal output)"
echo "3) WARNING (warnings and errors)"
echo "4) DEBUG (verbose logging)"
echo ""
read -p "Enter your choice (1-4): " log_choice

case $log_choice in
    1)
        export VASA_LOG_LEVEL="INFO"
        echo "Log level set to INFO (normal)..."
        ;;
    2)
        export VASA_LOG_LEVEL="ERROR"
        echo "Log level set to ERROR (minimal output)..."
        ;;
    3)
        export VASA_LOG_LEVEL="WARNING"
        echo "Log level set to WARNING..."
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
echo "================================"
echo "Starting training with:"
echo "  Config: $CONFIG_FILE"
if [ ! -z "$RESUME_PATH" ]; then
    echo "  Mode: Resume from checkpoint"
    echo "  Checkpoint: $RESUME_PATH"
else
    echo "  Mode: Training from scratch"
fi
echo "  Log Level: $VASA_LOG_LEVEL"
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

# If resume path is set, we need to temporarily modify the config
# or pass it as an environment variable
if [ ! -z "$RESUME_PATH" ]; then
    export VASA_RESUME_FROM="$RESUME_PATH"
    echo "Resume checkpoint set to: $RESUME_PATH"
fi

# Show the command being run
echo "Running: $COMMAND"
if [ ! -z "$RESUME_PATH" ]; then
    echo "  with VASA_RESUME_FROM=$RESUME_PATH"
fi
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