# Audio-Lip Correlation Refactor Plan

## Current Problem

Audio-lip correlation (line 834) is computed **before** landmark extraction from generated frames, so it can only use dataset landmarks (targets), not the actual lips from generated frames.

## Current Flow

```
compute_losses()
├─ 1. Reconstruction losses (line 495)
├─ 2. Warp losses (line 527)
├─ 3. Audio-lip correlation (line 834) ❌ Uses target lip_metrics, not generated
├─ 4. Perceptual loss (line 868)
├─ 5. Control losses (line 2680+)
│   ├─ Extract features from generated frames (line 2706) ✅ Extracts landmarks
│   ├─ Gaze loss (line 2732) - Uses extracted gazes
│   ├─ Landmark losses (line 2850) - Uses extracted landmarks
│   ├─ Blink loss (line 2881) - Uses extracted blink_states
│   └─ Emotion loss (line 2909) - Uses extracted emotions
```

## Proposed Solution

Move audio-lip correlation into the **control losses section** AFTER landmark extraction:

```
compute_losses()
├─ 1. Reconstruction losses
├─ 2. Warp losses
├─ 3. Perceptual loss
├─ 4. Control losses
│   ├─ Extract features from generated frames
│   ├─ Gaze loss - Uses extracted gazes
│   ├─ Audio-lip correlation ✅ NEW - Uses extracted lip landmarks
│   ├─ Landmark losses - Uses extracted landmarks
│   ├─ Blink loss - Uses extracted blink_states
│   └─ Emotion loss - Uses extracted emotions
```

## Benefits

1. ✅ **Uses actual generated lips**: Correlates audio with lips from generated frames, not dataset
2. ✅ **Consistent with other losses**: Gaze, blink, emotion all use extracted features
3. ✅ **Better training signal**: Model sees if ITS lips match audio, not if dataset lips match
4. ✅ **No double extraction**: Reuses already-extracted landmarks

## Implementation Steps

### Step 1: Remove old audio-lip correlation from early section

**File**: `vasa_losses.py` lines 820-864

**Action**: Comment out or remove this entire block (will be replaced)

### Step 2: Add audio-lip correlation to control losses section

**File**: `vasa_losses.py` after line 2757 (after gaze loss)

**New code**:
```python
# 2. Audio-Lip Correlation Loss - from extracted features
logger.debug("\nComputing Audio-Lip Correlation Loss:")
audio_lip_term = torch.tensor(0.0, device=device)

audio_key = 'audio_features' if 'audio_features' in conditions else 'audio' if 'audio' in conditions else None

if audio_key and 'landmarks' in extracted_features and 'lips' in extracted_features['landmarks']:
    try:
        # Extract lip landmarks from generated frames
        pred_lips = extracted_features['landmarks']['lips']  # [1, T_sampled, N_points, 3]
        sample_indices = extracted_features['sample_indices']

        # Compute lip openness from extracted landmarks
        # lips shape: [1, T, 20, 3] where 20 points, upper half = upper lip, lower half = lower lip
        upper_lips = pred_lips[:, :, :pred_lips.shape[2]//2, 1]  # Upper lip y-coords
        lower_lips = pred_lips[:, :, pred_lips.shape[2]//2:, 1]  # Lower lip y-coords
        lip_openness = (lower_lips.mean(dim=-1) - upper_lips.mean(dim=-1)).abs()  # [1, T_sampled]

        # Get corresponding audio features
        audio_features = conditions[audio_key][:, sample_indices].to(device).float()  # [B, T_sampled, D]
        audio_energy = torch.norm(audio_features, dim=-1)  # [B, T_sampled]

        # Normalize both signals
        lip_openness_norm = (lip_openness - lip_openness.min()) / (lip_openness.max() - lip_openness.min() + 1e-8)
        audio_energy_norm = (audio_energy - audio_energy.min()) / (audio_energy.max() - audio_energy.min() + 1e-8)

        # MSE loss between normalized signals
        audio_lip_loss = F.mse_loss(lip_openness_norm, audio_energy_norm)
        audio_lip_term = audio_lip_loss * self.lambda_audio_lip

        losses['audio_lip_correlation'] = audio_lip_term
        logger.debug(f"  Audio-lip correlation loss: {audio_lip_term.item():.6f}")
        logger.debug(f"  Lip openness range: [{lip_openness.min():.4f}, {lip_openness.max():.4f}]")
        logger.debug(f"  Audio energy range: [{audio_energy.min():.4f}, {audio_energy.max():.4f}]")

    except Exception as e:
        logger.warning(f"Could not compute audio-lip correlation: {e}")
        losses['audio_lip_correlation'] = torch.tensor(0.0, device=device)
else:
    logger.debug("  Skipping audio-lip correlation (no audio or no extracted lips)")
    losses['audio_lip_correlation'] = torch.tensor(0.0, device=device)
```

### Step 3: Remove redundant audio_expr_coupling

**File**: `vasa_losses.py` lines 560-569

The `audio_expr_coupling` loss (lambda=2.0) tries to correlate expression magnitude with audio energy.

**Problem**: This conflicts with audio-lip correlation and is less direct.

**Action**:
- Keep it but reduce weight further OR
- Remove it entirely (audio-lip correlation is more direct)

### Step 4: Update compute_audio_lip_correlation signature

**Current** (line 376):
```python
def compute_audio_lip_correlation(self, pred_motion, audio_features, lip_metrics, generated_frames=None):
```

**New** (simplified, no longer needed):
Remove this function entirely since we're computing it inline in control losses section.

OR keep it but refactor to:
```python
def compute_audio_lip_correlation(self, lip_openness, audio_energy):
    """
    Compute correlation between lip openness and audio energy.

    Args:
        lip_openness: [B, T] - Lip openness values (from extracted landmarks)
        audio_energy: [B, T] - Audio energy values

    Returns:
        Correlation loss scaled by lambda_audio_lip
    """
    # Normalize
    lip_norm = (lip_openness - lip_openness.min()) / (lip_openness.max() - lip_openness.min() + 1e-8)
    audio_norm = (audio_energy - audio_energy.min()) / (audio_energy.max() - audio_energy.min() + 1e-8)

    # MSE loss
    loss = F.mse_loss(lip_norm, audio_norm) * self.lambda_audio_lip
    return loss
```

## Other Losses That Could Use Extracted Features

### 1. ✅ Already Using Extracted Features

- **Gaze loss** (line 2732) - Uses `extracted_features['gazes']`
- **Landmark losses** (line 2850) - Uses `extracted_features['landmarks']`
- **Blink loss** (line 2881) - Uses `extracted_features['blink_states']`
- **Emotion loss** (line 2909) - Uses `extracted_features['emotions']`

### 2. ❌ NOT Using Extracted Features (Should Be)

- **Audio-lip correlation** (line 834) - ❌ Uses `targets['lip_metrics']` (dataset)
  - **Fix**: Move to control losses, use `extracted_features['landmarks']['lips']`

- **Mouth openness direct** (line 855) - ❌ Uses `targets['lip_metrics']['openness']` (dataset)
  - **Fix**: Compute from `extracted_features['landmarks']['lips']`

### 3. ⚠️ Questionable (May Not Need Frames)

- **Audio-expression coupling** (line 566) - Uses predicted expression magnitude
  - **Current**: Correlates `||expression_embed||` with audio energy
  - **Question**: Should this use extracted emotion from frames instead?
  - **Answer**: Probably NO - expression_embed is latent, not visual emotion

## Expected Impact

### Before Refactor

Audio-lip correlation compares:
- **Audio energy** (from current input)
- **Lip openness** (from dataset target - may not match generated frames!)

**Problem**: Model can have perfect audio-lip correlation in dataset but terrible sync in generated frames.

### After Refactor

Audio-lip correlation compares:
- **Audio energy** (from current input)
- **Lip openness** (extracted from generated frames - actual model output!)

**Benefit**: Model must make ITS generated lips match audio, not just copy dataset lip timing.

## Testing Checklist

After refactor:

- [x] Audio-lip correlation uses extracted lips from generated frames ✅ COMPLETED
- [x] Mouth openness direct loss also uses extracted lips ✅ COMPLETED
- [ ] Loss value is similar magnitude to before (~0.001 - 0.1) - NEEDS TESTING
- [ ] WandB logs `audio_lip_correlation` metric - NEEDS TESTING
- [ ] No errors when faces aren't detected (returns 0.0 loss) - NEEDS TESTING
- [ ] Works correctly when `epoch < 5` (early training, no frame extraction) - NEEDS TESTING

## Implementation Status

### ✅ Completed (2025-10-03)

1. **Removed old audio-lip correlation** (vasa_losses.py:820-825)
   - Disabled old implementation that used `targets['lip_metrics']`
   - Added comments explaining why it was moved
   - Set placeholder losses to 0.0 (will be computed later)

2. **Added new audio-lip correlation** (vasa_losses.py:2731-2784)
   - Integrated into control losses section after gaze loss
   - Uses `extracted_features['landmarks']['lips']` from generated frames
   - Computes lip openness from upper/lower lip y-coordinates
   - Normalizes both lip openness and audio energy
   - Applies MSE loss with lambda_audio_lip scaling (3.0)
   - Includes proper exception handling and debug logging

3. **Updated mouth_openness_direct** (vasa_losses.py:2770-2775)
   - Now uses extracted lip landmarks instead of dataset targets
   - Computes same normalized lip openness as audio-lip correlation
   - Applies stronger weight (10.0) for direct supervision
   - Ensures consistency between both losses

### Key Changes

**Old implementation (lines 820-864, now disabled)**:
```python
# Used dataset lip_metrics
audio_lip_loss = self.compute_audio_lip_correlation(
    outputs,
    conditions[audio_key],
    targets['lip_metrics']  # ❌ Dataset targets, not generated
)
```

**New implementation (lines 2731-2784)**:
```python
# Uses extracted landmarks from generated frames
pred_lips = extracted_features['landmarks']['lips']  # ✅ From generated frames
upper_lips = pred_lips[:, :, :pred_lips.shape[2]//2, 1]
lower_lips = pred_lips[:, :, pred_lips.shape[2]//2:, 1]
lip_openness = (lower_lips.mean(dim=-1) - upper_lips.mean(dim=-1)).abs()

# Correlate with audio
audio_lip_loss = F.mse_loss(lip_openness_norm, audio_energy_norm)
```

### Next Steps

1. Run training to verify implementation works
2. Check WandB logs for `audio_lip_correlation` and `mouth_openness_direct`
3. Verify loss values are in healthy range
4. Confirm no errors when face detection fails
5. Test early training behavior (epoch < 5)

## Files to Modify

1. ✅ `vasa_losses.py`:
   - Remove old audio-lip correlation (lines 820-864)
   - Add new audio-lip correlation in control losses section (after line 2757)
   - Optional: Refactor `compute_audio_lip_correlation()` or remove it

2. ✅ `LOSS_AUDIT.md`:
   - Update audio-lip correlation description
   - Note it now uses extracted features, not dataset

3. ✅ Test training:
   - Run overfitting training
   - Check WandB for `audio_lip_correlation` metric
   - Verify no errors

## Summary

**Current Issue**: Audio-lip correlation uses dataset lips, not generated lips
**Root Cause**: Computed before landmark extraction from generated frames
**Solution**: Move to control losses section, use extracted lip landmarks
**Benefits**: Training signal based on actual generated output, not dataset
**Complexity**: Low - just move code and update references
