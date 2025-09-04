#!/bin/bash

# VASA Training Context Dump Script
# This script collects all relevant files for debugging the VASA training issue

OUTPUT_FILE="vasa_training_context.md"

echo "# VASA Training Debug Context" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Generated on: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Problem Description" >> $OUTPUT_FILE
cat << 'EOF' >> $OUTPUT_FILE

### Main Issue
The VASA model training has a **loss stuck at 1.0** problem during training, even though:
1. The loss weights have been fixed (lambda_pose=1.0, lambda_dynamics=1.0 instead of 0)
2. Test scripts show loss computation works correctly (~3.06)
3. The model architecture is correct

### Symptoms
- Training loss returns exactly 1.0 every epoch
- This is the default error value returned when loss computation fails
- TDD tests correctly detect this as a failure and stop training
- The test_loss_computation.py script shows losses compute correctly (~3.06)

### What We're Trying to Do
- Overfit the model on a single video (junk/10.mp4 and variants)
- This is a standard debugging technique to verify the model can learn

### Files Already Fixed
1. **vasa_config_fixed.yaml**: Set lambda_pose and lambda_dynamics from 0 to 1.0
2. **vasa_trainer.py**: Added handling for non-windowed data in collate_vasa_batch
3. Created 50-frame and 100-frame videos to match model expectations

### Current Status
- Loss computation test works: ✅
- Training runs without crashes: ✅
- Loss decreases during training: ❌ (stuck at 1.0)
- Model learns/overfits: ❌

EOF

echo "" >> $OUTPUT_FILE
echo "---" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Function to add a file to the output
add_file() {
    local file_path=$1
    local description=$2
    
    if [ -f "$file_path" ]; then
        echo "## File: $file_path" >> $OUTPUT_FILE
        echo "$description" >> $OUTPUT_FILE
        echo '```python' >> $OUTPUT_FILE
        # Only include first 500 lines for very large files
        head -500 "$file_path" >> $OUTPUT_FILE
        local line_count=$(wc -l < "$file_path")
        if [ $line_count -gt 500 ]; then
            echo "" >> $OUTPUT_FILE
            echo "# ... [File truncated - total $line_count lines]" >> $OUTPUT_FILE
        fi
        echo '```' >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    else
        echo "## File: $file_path (NOT FOUND)" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    fi
}

add_yaml() {
    local file_path=$1
    local description=$2
    
    if [ -f "$file_path" ]; then
        echo "## File: $file_path" >> $OUTPUT_FILE
        echo "$description" >> $OUTPUT_FILE
        echo '```yaml' >> $OUTPUT_FILE
        cat "$file_path" >> $OUTPUT_FILE
        echo '```' >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    else
        echo "## File: $file_path (NOT FOUND)" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    fi
}

# Add configuration files
echo "# Configuration Files" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

add_yaml "vasa_config_fixed.yaml" "Fixed configuration with correct loss weights"
add_yaml "vasa_config_overfit_simple.yaml" "Overfitting configuration for single video"

# Add key Python files
echo "# Core Implementation Files" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

add_file "vasa_model.py" "Main VASA model implementation with loss computation"
add_file "vasa_trainer.py" "Training loop implementation"
add_file "vasa_dataset.py" "Dataset implementation"
add_file "run_tdd_training.py" "TDD training runner script"
add_file "test_loss_computation.py" "Loss computation test (this works correctly)"

# Add test results
echo "# Test Results" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Loss Computation Test Output" >> $OUTPUT_FILE
echo "When running test_loss_computation.py, the output shows:" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
cat << 'EOF' >> $OUTPUT_FILE
5. Loss results:
   reconstruction: 3.059626 (requires_grad: True)
   pose_loss: 2.086096 (requires_grad: True)
   dynamics_loss: 0.973530 (requires_grad: True)
   ...
   total: 3.059626 (requires_grad: True)

6. Checking if losses are reasonable...
   ✅ Total loss is reasonable: 3.059626
EOF
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Training Output" >> $OUTPUT_FILE
echo "When running training, the output shows:" >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
cat << 'EOF' >> $OUTPUT_FILE
Epoch 0:   0%|          | 0/2 [00:02<?, ?it/s]
WARNING     Reconstruction loss: 1.0000 - Above  vasa_tdd_tests.py:74
EOF
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Add error logs
echo "# Recent Error Logs" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -f "bad_videos/invalid_videos.txt" ]; then
    echo "## Video Processing Errors" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    head -20 "bad_videos/invalid_videos.txt" >> $OUTPUT_FILE
    echo '```' >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
fi

# Add key code snippets
echo "# Key Code Snippets to Review" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Loss Module compute_losses method (from vasa_model.py)" >> $OUTPUT_FILE
echo "Look for where it might return 1.0 as default:" >> $OUTPUT_FILE
echo '```python' >> $OUTPUT_FILE
grep -A 30 "def compute_losses" vasa_model.py | head -40 >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

echo "## Where loss gets computed in trainer (from vasa_trainer.py)" >> $OUTPUT_FILE
echo '```python' >> $OUTPUT_FILE
grep -B 5 -A 15 "compute_losses" vasa_trainer.py | head -30 >> $OUTPUT_FILE
echo '```' >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Summary
echo "# Summary of Investigation Needed" >> $OUTPUT_FILE
cat << 'EOF' >> $OUTPUT_FILE

## Key Questions to Answer
1. Why does the loss computation return 1.0 during training but ~3.06 in the test script?
2. Is the config being loaded correctly during training?
3. Are the loss weights (lambda_pose, lambda_dynamics) actually being applied?
4. Is there an exception being caught that returns 1.0 as a fallback?

## Suspected Issues
1. Config not being passed correctly to loss module during training
2. Exception handling returning default 1.0 value
3. Loss weights not being applied despite config changes
4. Data format mismatch between training and test scenarios

## Next Steps
1. Add logging to see actual lambda values being used
2. Remove any try/except blocks that return 1.0
3. Verify config is loaded correctly in training
4. Check if loss module is initialized with correct config

EOF

echo "" >> $OUTPUT_FILE
echo "---" >> $OUTPUT_FILE
echo "Context dump complete! Saved to $OUTPUT_FILE" 
echo "File size: $(du -h $OUTPUT_FILE | cut -f1)"
echo "Line count: $(wc -l < $OUTPUT_FILE) lines"