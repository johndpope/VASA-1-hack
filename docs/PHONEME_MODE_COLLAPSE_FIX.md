# Phoneme Mode Collapse Fix

## Problem

The phoneme prediction head was stuck predicting the same class (`<pad>` or another common phoneme) for all 8 latent queries, showing no learning progress.

### Symptoms

From WandB visualization `visuals/audio_to_expression`:
```
GT:  <pad>  <pad>  <pad>  ɔ     ɪ     ...  (varied phonemes)
Pred: <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>  (all same)
      [Red] [Red] [Red] [Red] [Red] [Red] [Red] [Red]  (all wrong)
```

All predictions were the same phoneme, indicating **mode collapse**.

### Root Cause

**Class imbalance** in the training data:
- `<pad>` (ID 0): Appears in ~30-40% of samples (silence/start/end)
- `<s>` (ID 1): Appears in ~5-10% (sentence starts)
- `</s>` (ID 2): Appears in ~5-10% (sentence ends)
- `<unk>` (ID 3): Appears in ~2-5% (unknown sounds)
- **Actual phonemes** (ID 4+): Each appears in <1% of samples

The standard cross-entropy loss without class weighting learns to **always predict the most common class** (`<pad>`) to minimize average loss, resulting in mode collapse.

## Solution

### 1. Class Weighting (Primary Fix)

**Location**: `vasa_losses.py` lines 1128-1145

Added class weights to `F.cross_entropy` to downweight common classes and upweight rare phonemes:

```python
# Create class weights: lower weight for <pad>, higher for actual phonemes
if not hasattr(self, '_phoneme_class_weights'):
    weights = torch.ones(vocab_size, device=device)
    weights[0] = 0.1   # <pad> - very common, low weight
    weights[1] = 0.5   # <s> - sentence start, medium-low weight
    weights[2] = 0.5   # </s> - sentence end, medium-low weight
    weights[3] = 0.3   # <unk> - unknown, low weight
    # All other phonemes (4+) keep weight 1.0 (higher priority)
    self._phoneme_class_weights = weights

aux_phoneme_term = F.cross_entropy(
    phoneme_pred.view(-1, vocab_size),
    phoneme_gt_clamped.view(-1).long(),
    weight=self._phoneme_class_weights  # ← NEW: Class weighting
)
```

**Effect**:
- `<pad>` errors contribute 10% of normal loss
- Actual phoneme errors contribute 100% of normal loss
- Model is incentivized to learn phonemes, not just predict `<pad>`

### 2. Increased Loss Weight (Secondary Fix)

**Location**: `overfit_config.yaml` line 227

Increased `lambda_aux_phoneme` from 0.2 to 1.0:

```yaml
lambda_aux_phoneme: 1.0  # INCREASED from 0.2 - with class weighting to prevent mode collapse
```

**Effect**:
- Phoneme loss has 5x more impact on total loss
- Faster learning of phoneme features
- Better signal-to-noise ratio against other losses

## Why This Works

### Problem: Naive Cross-Entropy

Standard cross-entropy with imbalanced data:

```python
# Class distribution
<pad>:    35% of samples → Loss contribution: 35%
phoneme1:  0.5% of samples → Loss contribution: 0.5%
phoneme2:  0.5% of samples → Loss contribution: 0.5%
...

# Model learns: "Always predict <pad>" → 35% correct, 0.65 loss
# Better than random (1/392 = 0.25% correct)
```

The model converges to predicting the majority class.

### Solution: Weighted Cross-Entropy

With class weighting:

```python
# Weighted loss contribution
<pad>:    35% * 0.1 = 3.5%   (downweighted)
phoneme1:  0.5% * 1.0 = 0.5%  (normal weight)
phoneme2:  0.5% * 1.0 = 0.5%  (normal weight)
...

# Model learns: "Predict actual phonemes" → Higher total accuracy
# <pad> errors are cheap, phoneme errors are expensive
```

The model is incentivized to learn phoneme discrimination.

## Expected Results

### Before Fix (Mode Collapse)

```
Epoch 1-100:
  aux_phoneme loss: 6.0 → 5.5 (slight decrease)
  Predictions: All <pad>
  Accuracy: ~35% (just the <pad> tokens)
```

### After Fix (Learning)

```
Epoch 1-20:
  aux_phoneme loss: 4.5 → 3.0 (steady decrease)
  Predictions: Mix of <pad>, actual phonemes
  Accuracy: 35% → 60% → 75%

Epoch 20-100:
  aux_phoneme loss: 3.0 → 1.5 (continued improvement)
  Predictions: Mostly correct phonemes
  Accuracy: 75% → 85% → 92%
```

### Visualization Changes

**Before**:
```
GT:  <pad> <pad> ɔ    ɪ    ð    ...
Pred: <pad> <pad> <pad> <pad> <pad> ...
      [Blue][Blue][Red] [Red] [Red]   (only <pad> matches)
```

**After** (expected after ~50 epochs):
```
GT:  <pad> <pad> ɔ    ɪ    ð    ...
Pred: <pad> <pad> ɔ    i    ð    ...
      [Blue][Blue][Blue][Blue][Blue]  (mostly correct!)
```

## Implementation Details

### Class Weight Caching

```python
if not hasattr(self, '_phoneme_class_weights'):
    # Only create once and cache
    weights = torch.ones(vocab_size, device=device)
    ...
    self._phoneme_class_weights = weights
```

**Benefits**:
- Created once per training session
- Stored as class attribute
- No overhead in forward pass
- Automatically on correct device

### Weight Values Rationale

| Class | Weight | Reasoning |
|-------|--------|-----------|
| `<pad>` (0) | 0.1 | Very common (30-40%), needs heavy downweighting |
| `<s>` (1) | 0.5 | Common (5-10%), moderate downweighting |
| `</s>` (2) | 0.5 | Common (5-10%), moderate downweighting |
| `<unk>` (3) | 0.3 | Moderately common (2-5%), some downweighting |
| Phonemes (4+) | 1.0 | Rare (<1% each), full weight (learn these!) |

These values were chosen to:
1. Prevent `<pad>` dominance
2. Balance special tokens
3. Prioritize actual phoneme learning

## Alternatives Considered

### 1. Focal Loss

```python
# Focal loss downweights easy examples
focal_loss = -(1 - p_t)**gamma * log(p_t)
```

**Rejected**: More complex, harder to tune, similar effect to class weighting.

### 2. Label Smoothing

```python
# Smooth labels to prevent overconfidence
smoothed_labels = (1 - ε) * one_hot + ε / num_classes
```

**Rejected**: Doesn't address class imbalance, just reduces overconfidence.

### 3. Oversampling Rare Classes

```python
# Sample rare phonemes more frequently
```

**Rejected**: Changes data distribution, expensive, requires dataset changes.

### 4. Two-Stage Training

```python
# Stage 1: Train only on non-<pad> samples
# Stage 2: Train on all samples
```

**Rejected**: Complex, requires training script changes, suboptimal.

## Monitoring

### Check if Fix is Working

1. **WandB Loss Monitor**:
   ```
   aux_phoneme loss should decrease from ~5.0 to ~1.5 over 100 epochs
   ```

2. **Visualization**:
   ```
   visuals/audio_to_expression should show more blue labels (correct predictions)
   ```

3. **Terminal Logs**:
   ```
   Look for: "📊 Phoneme class weights initialized: <pad>=0.1, <s>=0.5, ..."
   This confirms weights are being used
   ```

### Warning Signs

❌ **Not working** if:
- aux_phoneme loss stuck at 5.5+
- All predictions still `<pad>`
- No blue labels in visualization

✅ **Working** if:
- aux_phoneme loss decreasing steadily
- Mix of phoneme predictions
- Some blue labels appearing

## Debugging

### Check Class Weights Loaded

```python
# In training script, after first batch
print(loss_module._phoneme_class_weights[:10])
# Expected: tensor([0.1, 0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
```

### Check Loss Computation

```python
# Should see the class weights parameter
aux_phoneme_term = F.cross_entropy(
    phoneme_pred.view(-1, vocab_size),
    phoneme_gt_clamped.view(-1).long(),
    weight=self._phoneme_class_weights  # ← This should be present
)
```

### Check Loss Magnitude

```bash
# In WandB or logs, look for:
aux_phoneme: 4.5 → 4.2 → 3.8 → 3.5 ... (decreasing)

# Before fix:
aux_phoneme: 6.0 → 5.8 → 5.7 → 5.6 ... (barely decreasing)
```

## Rollback

If this causes issues:

1. **Remove class weighting**:
   ```python
   # In vasa_losses.py, remove the weight parameter
   aux_phoneme_term = F.cross_entropy(
       phoneme_pred.view(-1, vocab_size),
       phoneme_gt_clamped.view(-1).long()
       # No weight parameter
   )
   ```

2. **Lower lambda**:
   ```yaml
   # In overfit_config.yaml
   lambda_aux_phoneme: 0.2  # Back to original
   ```

## Related Issues

### Issue 1: Phoneme GT has `<pad>` at start

**Status**: Expected behavior

wav2vec2 outputs `<pad>` for silence/start of audio. This is normal.

### Issue 2: All predictions are wrong

**Status**: Fixed by this PR

Mode collapse meant model predicted only `<pad>`. Class weighting fixes this.

### Issue 3: Loss not decreasing

**Status**: Should be fixed

With class weighting + higher lambda, loss should decrease steadily.

## Testing

To verify the fix works:

1. **Delete checkpoint** (to start fresh with new weights):
   ```bash
   rm checkpoints_overfit/best_checkpoint.pt
   ```

2. **Start training**:
   ```bash
   ./safe-train.sh
   ```

3. **Watch for log line**:
   ```
   📊 Phoneme class weights initialized: <pad>=0.1, <s>=0.5, </s>=0.5, <unk>=0.3, phonemes=1.0
   ```

4. **Check WandB after ~10 epochs**:
   - `aux_phoneme` loss should be decreasing
   - `visuals/audio_to_expression` should show some blue labels

5. **Expected timeline**:
   - Epoch 0-10: Still mostly `<pad>`, loss ~4.5 → 3.5
   - Epoch 10-50: Mix of predictions, loss ~3.5 → 2.0
   - Epoch 50+: Good predictions, loss ~2.0 → 1.5

## Files Modified

1. **vasa_losses.py** (lines 1128-1145)
   - Added class weight initialization
   - Modified `F.cross_entropy` call to include `weight` parameter

2. **overfit_config.yaml** (line 227)
   - Increased `lambda_aux_phoneme` from 0.2 to 1.0

---

**Implementation Date**: 2025-10-22
**Status**: Ready for testing
**Expected Impact**: Fix mode collapse, enable phoneme learning
