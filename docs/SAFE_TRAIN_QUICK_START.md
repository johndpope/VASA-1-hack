# Safe Training - Quick Start Guide

## TL;DR

```bash
# Start safe training with auto OOM recovery
tmux new -s vasa
./safe-train.sh
# Detach: Ctrl+B, D

# Monitor progress
tmux attach -t vasa      # Reattach to session
tail -f safe_train.log   # Watch log file
watch -n 1 nvidia-smi    # Monitor GPU
```

## What It Does

✅ Automatically detects OOM crashes in logs (every 2 seconds)
✅ Kills hanging processes immediately (`~/killer.sh python`)
✅ Waits 5 seconds for GPU memory cleanup
✅ Resumes training from last checkpoint
✅ Repeats forever until training completes
✅ Logs all crashes with timestamps

## Quick Commands

```bash
# Start training
./safe-train.sh

# Stop gracefully (waits for checkpoint save)
# Press Ctrl+C in tmux session

# Force stop (emergency)
~/killer.sh python

# View crash history
grep "OOM DETECTED" safe_train.log

# Count restarts
grep "Training Attempt" safe_train.log | wc -l

# Check GPU memory
nvidia-smi
```

## Common Issues

### OOM every 5-10 minutes?
**Fix**: Edit `overfit_config.yaml`:
```yaml
gradient_accumulation_steps: 1  # Lower this (was 4)
batch_size: 4                   # Or lower this (was 8)
```

### Script exits immediately?
**Fix**: Check `~/killer.sh` exists:
```bash
ls -la ~/killer.sh
chmod +x ~/killer.sh
```

### Process won't die?
**Fix**: Nuclear option:
```bash
~/killer.sh python
sudo nvidia-smi --gpu-reset  # Last resort
```

## Files

- `safe-train.sh` - Main wrapper script
- `safe_train.log` - Crash history log
- `train.sh` - Original training script (called by safe-train)
- `SAFE_TRAIN_README.md` - Full documentation

## Expected Behavior

**Normal**:
- 1-2 OOM crashes at start (memory stabilization)
- Then smooth training with occasional checkpoints
- Auto-restart in ~5-10 seconds on any crash

**Warning**:
- OOM every 5-10 minutes = config too aggressive
- No OOM but process hangs = different issue (check logs)

## Configuration

Edit `safe-train.sh` (lines 15-18):
```bash
MAX_RESTARTS=999999  # Max auto-restarts
RESTART_DELAY=5      # Wait time between restarts
LOG_FILE="safe_train.log"  # Log location
```

## Pro Tips

1. **Always use tmux/screen** - Survives SSH disconnects
2. **Watch initial epochs** - Ensure OOM detection works
3. **Review logs daily** - Spot patterns in crashes
4. **Adjust config** - If >3 OOMs per hour, lower batch size
5. **Monitor WandB** - Gaps in loss curve = restarts

## Help

Full docs: `cat SAFE_TRAIN_README.md`
Training config: `overfit_config.yaml`
Model logs: Check WandB dashboard
