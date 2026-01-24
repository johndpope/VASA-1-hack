# Blink Loss Implementation

## Overview

Implemented the previously stubbed-out `_compute_blink_loss` function to enable blink supervision from extracted features in generated frames.

**Location**: `vasa_losses.py:242-311`

---

## Implementation Details

### Input Format

**Predicted blink states**: Extracted from generated frames via `extracted_features['blink_states']`
- Shape: `[B, T, 3]`
- Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
- Channel 1: Left eye openness (0-1, continuous)
- Channel 2: Right eye openness (0-1, continuous)

**Target blink states**: From dataset conditions `conditions['blink_state']`
- Same format as predicted

### Loss Components

#### 1. Eye Openness Loss (Channels 1-2)
```python
pred_openness = pred_blinks[:, :, 1:]   # [B, T, 2] (left, right)
target_openness = target_blinks[:, :, 1:]

openness_loss = F.mse_loss(pred_openness, target_openness)
```

**What it measures**: How well generated frames match target eye openness (0=closed, 1=open)

#### 2. Blink Phase Loss (Channel 0)
```python
pred_phase = pred_blinks[:, :, 0]   # [B, T]
target_phase = target_blinks[:, :, 0]

phase_loss = F.mse_loss(pred_phase, target_phase)
```

**What it measures**: How well generated frames match blink phase (open/closing/closed/opening)

**Note**: Using MSE instead of cross-entropy because phases are ordered (0→1→2→3→0), so treating as regression makes sense.

#### 3. Combined Loss
```python
total_loss = (openness_loss + phase_loss) * lambda_blink
```

**Weight**: `lambda_blink = 0.1` (from `overfit_config.yaml`)

### Metrics Computed

1. **blink_openness_loss**: MSE on eye openness (channels 1-2)
2. **blink_phase_loss**: MSE on blink phase (channel 0)
3. **blink_phase_accuracy**: Percentage of frames where predicted phase is within 0.5 of target
4. **blink_openness_mae**: Mean absolute error on eye openness

---

## Integration with Training

### Where It's Called

**Location**: `vasa_losses.py:2895-2921` (control losses section)

```python
if 'blink_states' in extracted_features and 'blink_state' in conditions:
    pred_blink_tensor = extracted_features['blink_states']
    sample_indices = extracted_features['sample_indices']

    # Sample target blink states
    target_blink = conditions['blink_state'][:, sample_indices]

    # Compute blink loss
    blink_loss, blink_metrics = self._compute_blink_loss(
        {'blink_state': pred_blink_tensor},  # Wrap in dict
        target_blink,
        lambda_blink=self.lambda_blink,
        device=device
    )
    losses.update(blink_metrics)
    losses['control_blink'] = blink_loss
```

### Extraction Pipeline

1. **Generate frames** from predicted motion (line 2640+)
2. **Extract blink states** from frames using MediaPipe (line 2484)
3. **Stack into tensor** (lines 2551-2562): `[1, T_sampled, 3]`
4. **Compute loss** using `_compute_blink_loss` (lines 2906-2911)

---

## Expected Behavior

### Healthy Training

- **blink_openness_loss**: 0.001 - 0.05 (similar to other MSE losses)
- **blink_phase_loss**: 0.001 - 0.05
- **blink_phase_accuracy**: 0.7 - 1.0 (70-100% correct phases)
- **blink_openness_mae**: 0.01 - 0.1 (1-10% error)

### Early Training (epoch < 5)

- Blink states NOT extracted (no frame generation)
- Loss returns 0.0 gracefully
- No errors

### When Face Detection Fails

- Extraction returns empty list
- Loss returns 0.0 with empty metrics dict
- No crashes

---

## Changes from Original

### Before (Stub Implementation)
```python
def _compute_blink_loss(...):
    # Blink loss disabled - model doesn't output individual eye landmarks
    return torch.tensor(0.0, device=device), {}
```

**Problem**: Extraction was working, but loss was always 0.0

### After (Full Implementation)
```python
def _compute_blink_loss(...):
    # Extract predicted blink states
    pred_blinks = pred_motion['blink_state']

    # Compute openness loss (channels 1-2)
    openness_loss = F.mse_loss(pred_openness, target_openness)

    # Compute phase loss (channel 0)
    phase_loss = F.mse_loss(pred_phase, target_phase)

    # Combine
    total_loss = (openness_loss + phase_loss) * lambda_blink

    return total_loss, metrics
```

**Benefit**: Now trains model to generate realistic blink patterns

---

## Testing Checklist

- [x] Implement blink loss computation ✅
- [ ] Run training and verify metrics appear in WandB
- [ ] Check loss values are in healthy range
- [ ] Verify no errors when face detection fails
- [ ] Confirm works during early training (epoch < 5)
- [ ] Compare blink_phase_accuracy over time (should improve)

---

## Configuration

**Config file**: `overfit_config.yaml`, `vasa_config.yaml`

```yaml
lambda_blink: 0.1  # Weight for blink loss
```

**Ramping**: Comment says "Will ramp to 2.0" - likely using lambda ramping in trainer

---

## Summary

**Before**: Blink states were extracted from generated frames but loss was stubbed (always 0.0)

**After**: Full blink loss implementation with:
- Eye openness MSE (channels 1-2)
- Blink phase MSE (channel 0)
- Accuracy metrics for monitoring
- Proper error handling

**Impact**: Model will now learn to generate realistic blinks that match target blink patterns from dataset.
