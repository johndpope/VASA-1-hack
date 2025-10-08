#!/bin/bash

# VASA Expression Audit Tool
# Compares ground truth expressions vs model predictions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     VASA Expression Audit Tool                     ║${NC}"
echo -e "${BLUE}║     Compare GT vs Predicted Expressions           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to display menu
show_menu() {
    echo -e "${YELLOW}Select configuration:${NC}"
    echo "  1) Overfit config (overfit_config.yaml)"
    echo "  2) Full training config (vasa_config.yaml)"
    echo "  3) Custom config path"
    echo "  4) Exit"
    echo ""
}

# Function to get checkpoint path based on config
get_checkpoint() {
    local config=$1

    if [[ "$config" == *"overfit"* ]]; then
        if [ -f "checkpoints_overfit/best_checkpoint.pt" ]; then
            echo "checkpoints_overfit/best_checkpoint.pt"
        elif [ -f "checkpoints_overfit/checkpoint_epoch_latest.pt" ]; then
            echo "checkpoints_overfit/checkpoint_epoch_latest.pt"
        else
            # Find latest checkpoint
            latest=$(ls -t checkpoints_overfit/checkpoint_epoch_*.pt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                echo "$latest"
            else
                echo ""
            fi
        fi
    else
        if [ -f "checkpoints/best_checkpoint.pt" ]; then
            echo "checkpoints/best_checkpoint.pt"
        elif [ -f "checkpoints/checkpoint_epoch_latest.pt" ]; then
            echo "checkpoints/checkpoint_epoch_latest.pt"
        else
            # Find latest checkpoint
            latest=$(ls -t checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                echo "$latest"
            else
                echo ""
            fi
        fi
    fi
}

# Function to get default video path
get_video_path() {
    local config=$1

    # Default to overfitting video
    if [ -f "junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4" ]; then
        echo "junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4"
    else
        echo ""
    fi
}

# Function to run audit
run_audit() {
    local config=$1
    local checkpoint=$2
    local video=$3
    local identity=$4
    local output_dir=$5

    echo -e "${GREEN}Configuration:${NC}"
    echo "  Config:     $config"
    echo "  Checkpoint: $checkpoint"
    echo "  Video:      $video"
    echo "  Identity:   $identity"
    echo "  Output:     $output_dir"
    echo ""

    # Check if files exist
    if [ ! -f "$config" ]; then
        echo -e "${RED}Error: Config file not found: $config${NC}"
        return 1
    fi

    if [ ! -f "$checkpoint" ]; then
        echo -e "${RED}Error: Checkpoint not found: $checkpoint${NC}"
        return 1
    fi

    if [ ! -f "$video" ]; then
        echo -e "${RED}Error: Video not found: $video${NC}"
        return 1
    fi

    if [ ! -f "$identity" ]; then
        echo -e "${RED}Error: Identity image not found: $identity${NC}"
        return 1
    fi

    echo -e "${BLUE}Starting expression audit...${NC}"
    echo ""

    python audit_expressions.py \
        --video "$video" \
        --identity "$identity" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --output-dir "$output_dir"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║     Audit Complete!                                ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Results saved to: ${output_dir}${NC}"
        echo ""
        echo "Files generated:"
        echo "  - expression_comparison.png  (visualization)"
        echo "  - expression_metrics.csv     (detailed metrics)"
        echo ""

        # Show quick summary if metrics file exists
        if [ -f "$output_dir/expression_metrics.csv" ]; then
            echo -e "${BLUE}Quick Summary:${NC}"
            # Get average metrics from CSV
            python3 - <<EOF
import pandas as pd
df = pd.read_csv("$output_dir/expression_metrics.csv")
print(f"  Expression L2:     {df['expr_l2'].mean():.4f} ± {df['expr_l2'].std():.4f}")
print(f"  Theta L2:          {df['theta_l2'].mean():.4f} ± {df['theta_l2'].std():.4f}")
if not df['expr_cosine'].isna().all():
    print(f"  Expression Cosine: {df['expr_cosine'].mean():.4f} ± {df['expr_cosine'].std():.4f}")
print(f"  Total frames:      {len(df)}")
print(f"  Worst frame:       {df.loc[df['expr_l2'].idxmax(), 'frame']:.0f} (L2={df['expr_l2'].max():.4f})")
EOF
            echo ""
        fi

        # Ask if user wants to view the plot
        echo -e "${YELLOW}View comparison plot? (y/n)${NC}"
        read -r view_plot
        if [[ "$view_plot" == "y" || "$view_plot" == "Y" ]]; then
            if command -v xdg-open &> /dev/null; then
                xdg-open "$output_dir/expression_comparison.png" 2>/dev/null &
            elif command -v open &> /dev/null; then
                open "$output_dir/expression_comparison.png" 2>/dev/null &
            else
                echo "Please open: $output_dir/expression_comparison.png"
            fi
        fi
    else
        echo -e "${RED}Audit failed with exit code: $exit_code${NC}"
        return $exit_code
    fi
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter choice [1-4]: " choice

    case $choice in
        1)
            config="overfit_config.yaml"
            ;;
        2)
            config="vasa_config.yaml"
            ;;
        3)
            read -p "Enter config path: " config
            ;;
        4)
            echo -e "${BLUE}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            echo ""
            continue
            ;;
    esac

    # Get default values
    checkpoint=$(get_checkpoint "$config")
    video=$(get_video_path "$config")
    identity="./data/IMG_1.png"

    # Determine output directory name
    if [[ "$config" == *"overfit"* ]]; then
        output_dir="expression_audit_overfit"
    else
        output_dir="expression_audit_full"
    fi

    echo ""
    echo -e "${YELLOW}Configuration selected: ${config}${NC}"
    echo ""

    # Check if checkpoint found
    if [ -z "$checkpoint" ]; then
        echo -e "${RED}No checkpoint found automatically.${NC}"
        read -p "Enter checkpoint path: " checkpoint
    else
        echo -e "${GREEN}Auto-detected checkpoint: ${checkpoint}${NC}"
        read -p "Use this checkpoint? (Y/n): " use_checkpoint
        if [[ "$use_checkpoint" == "n" || "$use_checkpoint" == "N" ]]; then
            read -p "Enter checkpoint path: " checkpoint
        fi
    fi

    echo ""

    # Check if video found
    if [ -z "$video" ]; then
        echo -e "${RED}No video found automatically.${NC}"
        read -p "Enter video path: " video
    else
        echo -e "${GREEN}Auto-detected video: ${video}${NC}"
        read -p "Use this video? (Y/n): " use_video
        if [[ "$use_video" == "n" || "$use_video" == "N" ]]; then
            read -p "Enter video path: " video
        fi
    fi

    echo ""

    # Identity image
    echo -e "${GREEN}Identity image: ${identity}${NC}"
    read -p "Use this identity? (Y/n): " use_identity
    if [[ "$use_identity" == "n" || "$use_identity" == "N" ]]; then
        read -p "Enter identity image path: " identity
    fi

    echo ""

    # Output directory
    echo -e "${GREEN}Output directory: ${output_dir}${NC}"
    read -p "Use this output directory? (Y/n): " use_output
    if [[ "$use_output" == "n" || "$use_output" == "N" ]]; then
        read -p "Enter output directory: " output_dir
    fi

    echo ""

    # Confirm before running
    echo -e "${YELLOW}Ready to run audit with:${NC}"
    echo "  Config:     $config"
    echo "  Checkpoint: $checkpoint"
    echo "  Video:      $video"
    echo "  Identity:   $identity"
    echo "  Output:     $output_dir"
    echo ""
    read -p "Continue? (Y/n): " confirm

    if [[ "$confirm" == "n" || "$confirm" == "N" ]]; then
        echo -e "${YELLOW}Cancelled.${NC}"
        echo ""
        continue
    fi

    # Run the audit
    run_audit "$config" "$checkpoint" "$video" "$identity" "$output_dir"

    echo ""
    read -p "Run another audit? (y/N): " another
    if [[ "$another" != "y" && "$another" != "Y" ]]; then
        echo -e "${BLUE}Exiting...${NC}"
        exit 0
    fi
    echo ""
done
