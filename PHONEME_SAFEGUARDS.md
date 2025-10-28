# Phoneme Loss Safeguards - Hard Assertions

## Overview

The phoneme prediction auxiliary loss is **CRITICAL** for proper lip sync training. To prevent training from running without it, we've added multiple hard assertions that will **fail training immediately** if phoneme data is missing.

## Three-Layer Validation

### 1. Training Start Validation (vasa_trainer.py:999-1034)

**When**: At the very beginning of `train()` method, before any epochs start

**What it checks**:
- Samples first batch from training DataLoader
- Verifies `phoneme_gt` key exists in batch
- Validates shape is `[batch_size, 8]` (8 latent queries)

**Error messages**:
```
❌ FATAL: phoneme_gt not found in training data!
The phoneme prediction loss requires phoneme ground truth.
Solution: Run the phoneme upsert script:
  ./upsert_phoneme.sh
This will add phoneme_gt to all cached windows without full reprocessing.
```

```
❌ FATAL: phoneme_gt has wrong shape: [batch, X]
Expected: [batch_size, 8] for 8 latent queries.
Cache may be corrupted or using wrong num_queries setting.
```

**Why this matters**: Catches missing phoneme_gt before any training time is wasted.

---

### 2. Loss Computation Assertions (vasa_losses.py:1093-1103)

**When**: During every forward pass, when computing losses

**What it checks**:
1. `aux_predictions` exists in model outputs
2. `phoneme_pred` exists in aux_predictions (model configured correctly)
3. `phoneme_gt` exists in aux_predictions (cache has ground truth)

**Error messages**:
```
❌ FATAL: aux_predictions missing from model outputs!
Phoneme loss cannot be computed.
```

```
❌ FATAL: phoneme_pred missing from aux_predictions!
Model not configured for phoneme prediction.
```

```
❌ FATAL: phoneme_gt missing from aux_predictions!
Cache does not have phoneme ground truth.
Run ./upsert_phoneme.sh first!
```

**Why this matters**:
- Ensures model architecture is correct (phoneme head exists)
- Ensures data pipeline is working (phoneme_gt flows through)
- Fails immediately on first batch if anything is wrong

---

### 3. Loss Monitoring (loss_monitor.py:293-298)

**When**: After each training step

**What it checks**:
- `aux_phoneme` loss value is in healthy range: `(0.1, 2.0)`
- Warning threshold: `3.0`
- Critical threshold: `5.0`

**Why this matters**: Monitors if loss is being computed correctly and converging as expected.

---

## What Each Layer Catches

| Issue | Layer 1 (Train Start) | Layer 2 (Loss Compute) | Layer 3 (Monitor) |
|-------|----------------------|------------------------|-------------------|
| Missing phoneme_gt in cache | ✅ | ✅ | ❌ |
| Wrong phoneme_gt shape | ✅ | ⚠️ (shape mismatch error) | ❌ |
| Model missing phoneme head | ❌ | ✅ | ❌ |
| aux_predictions not propagated | ❌ | ✅ | ❌ |
| Loss value out of range | ❌ | ❌ | ✅ |
| Loss not converging | ❌ | ❌ | ✅ |

✅ = Catches and fails hard
⚠️ = Catches but different error
❌ = Doesn't catch this issue

## Error Resolution Guide

### Error: "phoneme_gt not found in training data"

**Root cause**: Cache files don't have phoneme_gt yet

**Solution**:
```bash
# Run the upsert script to add phoneme_gt to all cached windows
./upsert_phoneme.sh

# Expected output:
# Processing windows: 100%|██████████| 316/316 [00:45<00:00, 6.91windows/s]
# ✅ Successfully added phoneme_gt to 316 windows across 6 videos
```

**Verification**:
```bash
# Check if phoneme_gt was added
python diagnose_phoneme.py --no_model --max_videos 1 --max_windows 3

# Expected: Should show phoneme_gt for all windows
```

---

### Error: "phoneme_gt has wrong shape"

**Root cause**:
- Cache created with different `num_queries` setting
- Or cache is corrupted

**Solution**:
```bash
# Re-run upsert with correct num_queries
./upsert_phoneme.sh cache_per_video 8

# Or if cache is corrupted, clear and rebuild
rm -rf cache_per_video/*/metadata.h5
# Then re-run data processing
```

---

### Error: "phoneme_pred missing from aux_predictions"

**Root cause**: Model architecture doesn't have phoneme prediction head

**Solution**: Check vasa_model.py:154
```python
# TalkVidAudioProjection should have:
self.phoneme_head = nn.Linear(dim, 50)

# And forward should return:
return output, {'phoneme_pred': phoneme_pred}
```

---

### Error: "aux_predictions missing from model outputs"

**Root cause**: aux_predictions not propagated through model layers

**Solution**: Check these locations in vasa_model.py:
1. TalkVidAudioProjection.forward returns tuple: `(output, aux_predictions)`
2. EfficientConditionEmbedding.forward propagates: `audio_embed, audio_aux = self.audio_projection(...)`
3. VASAModel.forward includes in outputs: `outputs['aux_predictions'] = {'phoneme_pred': ..., 'phoneme_gt': ...}`

---

## Testing the Safeguards

### Test 1: Missing phoneme_gt in cache

```bash
# Temporarily rename a metadata.h5 file to simulate missing phoneme_gt
mv cache_per_video/<some_md5>/metadata.h5 cache_per_video/<some_md5>/metadata.h5.bak

# Try training - should fail immediately at train start
python train_overfit.py

# Expected: Training fails with error message and upsert instructions

# Restore file
mv cache_per_video/<some_md5>/metadata.h5.bak cache_per_video/<some_md5>/metadata.h5
```

### Test 2: Model missing phoneme head

```bash
# Temporarily comment out phoneme_head in vasa_model.py:154
# self.phoneme_head = nn.Linear(dim, 50)  # COMMENTED OUT

# Try training - should fail on first batch
python train_overfit.py

# Expected: AssertionError about missing phoneme_pred

# Restore code
```

### Test 3: Wrong shape

```bash
# Modify upsert script to use num_queries=4 instead of 8
# Then run upsert
./upsert_phoneme.sh cache_per_video 4

# Try training - should fail at train start with shape error
python train_overfit.py

# Expected: AssertionError about wrong shape [batch, 4] vs expected [batch, 8]

# Fix by re-running with correct setting
./upsert_phoneme.sh cache_per_video 8
```

## Summary

With these three layers of validation:

1. **Training will NEVER run without phoneme_gt** - fails at startup
2. **Training will NEVER compute without phoneme loss** - fails on first batch
3. **Training will WARN if loss is out of range** - helps debug convergence issues

All errors provide clear, actionable error messages with exact solutions.

**Before training, always verify**:
```bash
# 1. Check cache has phoneme_gt
python diagnose_phoneme.py --no_model --max_windows 3

# 2. Run training - should pass all validations
python train_overfit.py

# Expected first logs:
# 🔍 Validating phoneme_gt presence in dataset...
# ✅ Phoneme validation passed! phoneme_gt shape: torch.Size([4, 8])
#    Expected: [batch_size, num_queries=8]
# Starting epoch 0...
# ✅ Auxiliary phoneme loss: 3.234567
```

## Files Modified

1. **vasa_losses.py** (lines 1087-1115)
   - Added 3 hard assertions for aux_predictions, phoneme_pred, phoneme_gt
   - Changed debug logs to ✅ emoji for clarity

2. **vasa_trainer.py** (lines 999-1034)
   - Added phoneme_gt validation at training start
   - Checks existence and shape
   - Provides clear error messages with solutions

3. **loss_monitor.py** (lines 293-298)
   - Already configured with healthy ranges for aux_phoneme
   - No changes needed

## Next Steps

1. **Run upsert script** to add phoneme_gt to cache:
   ```bash
   ./upsert_phoneme.sh
   ```

2. **Verify with diagnostic tool**:
   ```bash
   python diagnose_phoneme.py --no_model --max_videos 1
   ```

3. **Start training** - safeguards will validate everything:
   ```bash
   python train_overfit.py
   ```

If all validations pass, you'll see:
```
🔍 Validating phoneme_gt presence in dataset...
✅ Phoneme validation passed! phoneme_gt shape: torch.Size([4, 8])
   Expected: [batch_size, num_queries=8]
Starting epoch 0...
```

**Training will NOT proceed without phoneme_gt present and correctly shaped.**
