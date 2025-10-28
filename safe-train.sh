#!/bin/bash

# Safe Training Wrapper for VASA-1
# Automatically restarts training after OOM crashes with proper cleanup
#
# Features:
# - Detects OOM crashes and other GPU errors
# - Kills hanging Python processes using ~/killer.sh
# - Automatically resumes from last checkpoint
# - Logs crash history with timestamps
# - Configurable max restart attempts

set -e  # Exit on error (except for expected training crashes)

# ============================================================================
# Configuration
# ============================================================================

MAX_RESTARTS=999999  # Maximum number of automatic restarts (effectively infinite)
RESTART_DELAY=5      # Seconds to wait before restarting after crash
KILLER_SCRIPT="$HOME/killer.sh"
LOG_FILE="safe_train.log"

# Check if killer.sh exists
if [ ! -f "$KILLER_SCRIPT" ]; then
    echo "❌ Error: Killer script not found at $KILLER_SCRIPT"
    echo "   Please ensure ~/killer.sh exists"
    exit 1
fi

# Make killer.sh executable if needed
chmod +x "$KILLER_SCRIPT"

# ============================================================================
# Helper Functions
# ============================================================================

log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

cleanup_gpu_processes() {
    log_message "🧹 Cleaning up GPU processes..."

    # Kill any hanging Python processes
    if bash "$KILLER_SCRIPT" python 2>&1 | grep -q "smashed"; then
        log_message "   ✅ Killed hanging Python processes"
    else
        log_message "   ℹ️  No hanging Python processes found"
    fi

    # Wait for GPU memory to be released
    sleep 2

    # Clear CUDA cache (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        # Log current GPU memory usage
        local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        log_message "   📊 GPU Memory: ${gpu_mem}MB used"

        # Optional: Force GPU reset (commented out by default, can cause issues)
        # sudo nvidia-smi --gpu-reset
    fi
}

check_oom_crash() {
    local exit_code=$1
    local log_tail="$2"

    # Check for OOM patterns in recent output
    if echo "$log_tail" | grep -qi "out of memory\|OOM\|CUDA out of memory\|OutOfMemoryError"; then
        return 0  # OOM detected
    fi

    # Check for other GPU errors
    if echo "$log_tail" | grep -qi "CUDA error\|RuntimeError.*CUDA\|device-side assert"; then
        return 0  # GPU error detected
    fi

    # Non-zero exit code might indicate crash
    if [ $exit_code -ne 0 ]; then
        return 0  # Crash detected
    fi

    return 1  # No crash detected
}

display_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🛡️  SAFE TRAINING WRAPPER                          ║"
    echo "║                                                                        ║"
    echo "║  Automatic OOM Recovery & Process Cleanup                             ║"
    echo "║  Max Restarts: $MAX_RESTARTS                                                      ║"
    echo "║  Restart Delay: ${RESTART_DELAY}s                                                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================================
# Main Training Loop
# ============================================================================

display_banner

# Initialize counters
restart_count=0
total_crashes=0
start_time=$(date '+%s')

log_message "🚀 Starting safe training wrapper"
log_message "   Log file: $LOG_FILE"
log_message "   Max restarts: $MAX_RESTARTS"
log_message ""

while [ $restart_count -lt $MAX_RESTARTS ]; do
    # Calculate attempt number
    attempt_num=$((restart_count + 1))

    log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_message "📍 Training Attempt #$attempt_num"
    log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Cleanup before starting (except first run)
    if [ $restart_count -gt 0 ]; then
        cleanup_gpu_processes
        log_message "⏳ Waiting ${RESTART_DELAY}s before restart..."
        sleep $RESTART_DELAY
    fi

    # Create temporary files for log monitoring
    temp_log=$(mktemp)
    pid_file=$(mktemp)

    # Run train.sh and capture output
    log_message "▶️  Launching train.sh..."
    echo ""

    # Run with automatic "yes" responses for prompts (non-interactive mode)
    # This ensures train.sh runs without waiting for user input
    # Default selections: 1 (overfit_config), N (don't override), N (don't clear cache), 1 (INFO log level)
    printf "1\nN\nN\n1\n" | bash train.sh 2>&1 | tee >(tail -100 > "$temp_log") &

    # Store the PID of the training process
    training_pid=$!
    echo $training_pid > "$pid_file"
    log_message "   Training PID: $training_pid"

    # Monitor logs for OOM in background
    (
        while kill -0 $training_pid 2>/dev/null; do
            # Check last 20 lines for OOM
            if tail -20 "$temp_log" 2>/dev/null | grep -qi "out of memory\|OOM\|CUDA out of memory\|OutOfMemoryError"; then
                log_message "🚨 OOM DETECTED IN LOGS! Killing training process immediately..."

                # Kill the training process tree
                pkill -P $training_pid 2>/dev/null || true
                kill -9 $training_pid 2>/dev/null || true

                # Also run killer.sh for safety
                bash "$KILLER_SCRIPT" python 2>/dev/null || true

                log_message "   ✅ Training process killed due to OOM"
                exit 0
            fi
            sleep 2  # Check every 2 seconds
        done
    ) &
    monitor_pid=$!

    # Wait for training to complete
    wait $training_pid
    exit_code=$?

    # Kill the monitor process
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true

    echo ""
    log_message "📊 Training process exited with code: $exit_code"

    # Read last 100 lines for error detection
    log_tail=$(cat "$temp_log" 2>/dev/null || echo "")

    # Cleanup temp files
    rm -f "$temp_log" "$pid_file"

    # Check if training completed successfully
    if [ $exit_code -eq 0 ]; then
        log_message "✅ Training completed successfully!"
        log_message ""
        log_message "╔════════════════════════════════════════════════════════════════════╗"
        log_message "║                   🎉 TRAINING COMPLETE                             ║"
        log_message "║                                                                    ║"
        log_message "║  Total attempts: $attempt_num                                               ║"
        log_message "║  Total crashes: $total_crashes                                              ║"
        log_message "║  Runtime: $(($(date '+%s') - start_time))s                                     ║"
        log_message "╚════════════════════════════════════════════════════════════════════╝"
        exit 0
    fi

    # Check if crash was due to OOM or GPU error
    if check_oom_crash $exit_code "$log_tail"; then
        total_crashes=$((total_crashes + 1))
        log_message "⚠️  Detected crash (OOM/GPU error)"
        log_message "   Total crashes so far: $total_crashes"
        log_message "   Will attempt restart..."
    else
        log_message "❌ Training failed with unknown error (exit code: $exit_code)"
        log_message "   This might not be an OOM/GPU issue."
        log_message "   Check the logs above for details."
        log_message ""

        # Ask user if they want to restart anyway (with timeout)
        read -t 10 -p "Restart anyway? (Y/n, auto-Yes in 10s): " choice || choice="Y"

        if [ "$choice" = "n" ] || [ "$choice" = "N" ]; then
            log_message "❌ User chose not to restart. Exiting."
            exit $exit_code
        fi

        log_message "🔄 User chose to restart (or timeout)"
        total_crashes=$((total_crashes + 1))
    fi

    # Increment restart counter
    restart_count=$((restart_count + 1))

    # Check if we've hit max restarts
    if [ $restart_count -ge $MAX_RESTARTS ]; then
        log_message "🛑 Maximum restart attempts ($MAX_RESTARTS) reached."
        log_message "   Exiting safe training wrapper."
        exit 1
    fi
done

# This should never be reached due to the infinite loop check above
log_message "🛑 Unexpected exit from training loop"
exit 1
