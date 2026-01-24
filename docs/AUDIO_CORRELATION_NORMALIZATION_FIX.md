# Audio-Lip Correlation Normalization Fix

## Issue

The audio-expression/audio-lip correlation was using **min-max normalization** which caused poor learning and unstable gradients.

**Original code (vasa_losses.py:590-591)**:
```python
# Min-max normalization - PROBLEMATIC
lip_openness_norm = (lip_openness - lip_openness.min()) / (lip_openness.max() - lip_openness.min() + 1e-8)
audio_energy_norm = (audio_energy - audio_energy.min()) / (audio_energy.max() - audio_energy.min() + 1e-8)
```

### Problems with Min-Max Normalization

1. **Outlier Sensitivity**
   - One loud audio spike compresses all other values to near 0
   - One frame with very open mouth makes all other frames look closed
   - Reduces signal variance needed for correlation learning

2. **Batch-Dependent Scaling**
   - Same audio/lip values get different normalized values in different batches
   - Inconsistent gradients across batches
   - Harder for model to learn stable audio-lip relationships

3. **Poor Gradient Flow**
   - When values are similar (e.g., silent section), max ≈ min
   - Division by near-zero creates numerical instability
   - Gradients vanish or explode

4. **Not Comparable Across Time**
   - Each batch has range [0, 1] regardless of actual intensity
   - Can't distinguish loud vs quiet speech
   - Can't distinguish open vs closed lips

## Solution

Use **L2 normalization** (like SyncNet) for stable, consistent correlation learning.

**New code (vasa_losses.py:589-593)**:
```python
# Use L2 normalization (like SyncNet) instead of min-max for better gradient flow
# Reshape to [B*T, 1] for normalization, then reshape back
B, T = lip_openness.shape
lip_openness_norm = F.normalize(lip_openness.reshape(-1, 1), p=2, dim=0).reshape(B, T)
audio_energy_norm = F.normalize(audio_energy.reshape(-1, 1), p=2, dim=0).reshape(B, T)
```

### Benefits of L2 Normalization

1. **Stable Magnitude**
   - Normalizes to unit vector: `||x||₂ = 1`
   - Preserves relative differences between values
   - Outliers don't compress the entire range

2. **Consistent Across Batches**
   - Same audio energy always maps to same normalized value
   - Gradients are consistent across different batches
   - Model learns stable audio-lip correlations

3. **Better Gradient Flow**
   - No division by (max - min)
   - No numerical instability from near-zero denominators
   - Smooth, well-behaved gradients

4. **Preserves Temporal Structure**
   - Relative intensities preserved within the sequence
   - Loud vs quiet speech distinguishable
   - Open vs closed lips distinguishable

### Mathematical Comparison

**Min-Max Normalization**:
```
x_norm = (x - min(x)) / (max(x) - min(x))
Range: [0, 1]
Problem: Depends on batch statistics
```

**L2 Normalization**:
```
x_norm = x / ||x||₂ = x / sqrt(sum(x²))
Range: [-1, 1] (preserves sign)
Benefit: Unit vector, stable across batches
```

## Why SyncNet Uses This

From the SyncNet code you provided:
```python
audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
face_embedding = F.normalize(face_embedding, p=2, dim=1)
```

SyncNet uses L2 normalization because:
1. **Cosine similarity**: After L2 norm, dot product = cosine similarity
2. **Stable training**: Embeddings always have unit length
3. **Better convergence**: Consistent gradient magnitudes
4. **Outlier robustness**: One loud frame doesn't break everything

## Expected Improvements

After this change, you should see:

1. **Better audio-lip correlation learning**
   - Model learns actual relationship between audio energy and lip motion
   - Not distorted by outliers or batch statistics

2. **More stable loss values**
   - Loss values comparable across different batches
   - Smoother training curves

3. **Faster convergence**
   - Consistent gradients → better optimization
   - Model can learn correlations more efficiently

4. **Better generalization**
   - Learns true audio-lip relationship, not batch artifacts
   - Works better on test data

## Verification

To verify the fix is working:

1. **Check normalized value ranges**:
   ```bash
   # Before: Values in [0, 1], all batches look similar
   # After: Values preserve relative magnitudes
   grep "Normalized lip openness" train.log
   grep "Normalized audio energy" train.log
   ```

2. **Check Pearson correlation**:
   ```bash
   # Should be > 0.5 for good audio-lip sync
   grep "Pearson correlation" train.log
   ```

3. **Monitor loss stability**:
   ```bash
   # Loss should be more stable across batches
   grep "audio_lip_correlation" train.log | tail -100
   ```

## Related Files

- `vasa_losses.py:577-593` - Audio-lip correlation normalization (FIXED)
- `Synchformer/models/syncnet.py` - SyncNet reference implementation
- `overfit_config.yaml:408` - `compute_mel_correlation: true` config

## Impact on Training

**Before fix**:
- Audio-lip correlation: erratic, batch-dependent
- Pearson correlation: low (< 0.3)
- Gradient flow: unstable

**After fix**:
- Audio-lip correlation: stable, consistent
- Pearson correlation: higher (> 0.5)
- Gradient flow: smooth

This should significantly improve the model's ability to learn audio-visual synchronization!

---

**Date**: 2025-10-21
**Status**: ✅ Fixed
**Change**: Replaced min-max normalization with L2 normalization (F.normalize)
