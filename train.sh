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
echo "3) Fine-tune from checkpoint (not implemented yet)"
echo ""
read -p "Enter your choice (1-3): " mode_choice

# Determine checkpoint directory based on config
if [ "$CONFIG_FILE" == "overfit_config.yaml" ]; then
    CHECKPOINT_DIR="checkpoints_overfit"
else
    CHECKPOINT_DIR="checkpoints"
fi

case $mode_choice in
    1)
        RESUME_PATH=""
        echo "Training from scratch..."
        ;;
    2)
        # Look for the best checkpoint
        if [ -f "$CHECKPOINT_DIR/best_checkpoint.pt" ]; then
            RESUME_PATH="$CHECKPOINT_DIR/best_checkpoint.pt"
            echo "Found checkpoint: $RESUME_PATH"
        elif [ -f "$CHECKPOINT_DIR/latest_checkpoint.pt" ]; then
            RESUME_PATH="$CHECKPOINT_DIR/latest_checkpoint.pt"
            echo "Found checkpoint: $RESUME_PATH"
        else
            echo "No checkpoint found in $CHECKPOINT_DIR/"
            read -p "Enter checkpoint path manually (or press Enter to train from scratch): " RESUME_PATH
        fi
        ;;
    3)
        echo "Fine-tuning with optimizer reset not implemented yet."
        echo "Please manually edit the training script if needed."
        RESUME_PATH=""
        ;;
    *)
        RESUME_PATH=""
        echo "Defaulting to train from scratch..."
        ;;
esac

echo ""
echo "Select cache mode:"
echo "1) H5 cache (default - uses pre-computed warps from create_video_face_swap.py)"
echo "2) No cache (compute warps on the fly)"
echo ""
read -p "Enter your choice (1-2): " cache_choice

case $cache_choice in
    1)
        export USE_H5_CACHE="true"
        echo "Using H5 cache with pre-computed warps..."
        ;;
    2)
        export USE_H5_CACHE="false"
        echo "Computing warps on the fly (no H5 cache)..."
        ;;
    *)
        export USE_H5_CACHE="true"
        echo "Defaulting to H5 cache..."
        ;;
esac

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
echo "  Cache: $([ "$USE_H5_CACHE" == "true" ] && echo "H5 cache" || echo "No cache")"
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