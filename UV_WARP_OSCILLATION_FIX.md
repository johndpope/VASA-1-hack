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

## Jumpstart Initialization (NEW)

Added smart initialization to UV warp head to avoid the 1-hour magnitude search.

### Mode 1: Canonical Warp Burn-In (RECOMMENDED) 🔥

Burn in an actual canonical warp from your dataset as the baseline:

```bash
# Extract canonical warp from cache
python extract_canonical_warp.py --cache_dir cache_single_bucket --output canonical_warp.pt

# Add to vasa_config.yaml
model:
  canonical_warp_path: "canonical_warp.pt"
```

**How it works**:
- Final layer bias = canonical warp values (196,608 values)
- Final layer weights = 0
- Model outputs canonical warp initially
- During training, model learns **residuals** from canonical warp

**Benefits**:
- Model starts with a REAL facial warp pattern (not random)
- Saves ~1-2 hours of training
- More stable training (learning deltas, not absolute values)
- Better generalization (canonical warp is a strong prior)

### Mode 2: Magnitude-Only Initialization (FALLBACK)

If no canonical warp provided, scales weights to output ~0.65 magnitude:

```python
# vasa_model.py lines 687-700
scale_factor = 15.0
final_layer.weight.mul_(scale_factor)
final_layer.bias.mul_(scale_factor)
```

**Why**: Saves ~1 hour by starting at correct magnitude instead of ~0.05.

## Expected Results

With these changes, you should see:

### Phase 1: Epochs 0-5 (Immediate correct magnitude) ✨ NEW
- UV magnitude starts near target: **0.5 → 0.65** (not 0.04 → 0.6!)
- Saves ~1 hour of training time
- All windows in a batch should be within ±0.1 of each other from epoch 0
- Fewer "SKIPPING WINDOW" messages

### Phase 2: Epochs 5-20 (Fine-tuning magnitude)
- UV magnitude stabilizes precisely around 0.62-0.68
- Model learns spatial patterns (not just magnitude)
- Consistent magnitudes across all windows in batch

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
