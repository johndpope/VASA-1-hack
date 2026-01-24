# UV Warp Oscillation Fix

## Problem Identified

The model was experiencing **catastrophic oscillation** in UV warp magnitude predictions:

### Symptoms (Epoch 13, Batch 7)
```
Window 0: UV magnitude = 0.684 ✅ (GOOD - learned correct scale)
Window 1: UV magnitude = 0.041 ❌ (COLLAPSED - 94% smaller!)
Window 2: UV magnitude = 0.014 ❌ (COLLAPSED - 98% smaller!)
Window 3: UV magnitude = 0.151 ⚠️  (RECOVERING but unstable)
```

### Root Cause
1. **Learning rate too high** (5e-4) - causing unstable updates
2. **Insufficient regularization** (weight_decay = 1e-5) - model overfitting to individual windows
3. **Weak magnitude loss** (lambda = 5.0) - not strong enough to prevent collapse
4. **Per-window overfitting** - memorizing instead of generalizing

## Solution Applied

### 1. Reduced Learning Rate
```yaml
# vasa_config.yaml line 64
lr: 1e-4  # Was 5e-4 (50% reduction)
```
**Why**: Slower updates allow model to learn stable patterns instead of oscillating

### 2. Increased Regularization
```yaml
# vasa_config.yaml line 88
weight_decay: 1e-4  # Was 1e-5 (10x increase)
```
**Why**: Stronger L2 penalty prevents overfitting to individual windows

### 3. Strengthened UV Warp Magnitude Loss
```yaml
# vasa_config.yaml line 235
lambda_warp_magnitude: 15.0  # Was 5.0 (3x increase)
```
**Why**: Much stronger penalty for UV warp collapse, forcing consistency

### 4. Deleted Bad Checkpoint
```bash
mv checkpoints/checkpoint_epoch_*.pt checkpoints_backup_epoch13/
```
**Why**: Start fresh training with new stable hyperparameters

## Expected Results

With these changes, you should see:

### Phase 1: Epochs 0-20 (Learning correct magnitude)
- UV magnitude gradually increases: 0.04 → 0.1 → 0.2 → 0.4 → 0.6
- All windows in a batch should be within ±0.1 of each other
- Fewer "SKIPPING WINDOW" messages

### Phase 2: Epochs 20-50 (Stabilization)
- UV magnitude stabilizes around 0.62-0.68 (target range)
- Consistent magnitudes across all windows in batch
- Face detection success rate increases to 80-100%

### Phase 3: Epochs 50-200 (Fine-tuning)
- UV warps refine for accurate facial motion
- Loss decreases steadily
- High-quality frame generation

## Monitoring Metrics

Watch these in wandb/logs:

1. **UV Warp Magnitude** - Should stabilize around 0.65 ± 0.1
2. **UV Warp Magnitude Loss** - Should decrease steadily
3. **Windows Processed vs Skipped** - More processed, fewer skipped
4. **Expression Std** - Should increase from 0.15 → 0.47
5. **Total Loss** - Should decrease from ~1300 → <100

## Warning Signs

If you still see oscillation:
- Further reduce LR to 5e-5
- Increase lambda_warp_magnitude to 20.0
- Add gradient value clipping (already at 1.0)

## Training Command

```bash
python vasa_trainer.py --config vasa_config.yaml
```

The model will now train from epoch 0 with stable hyperparameters.
