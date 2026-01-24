# Lip-Sync Validation - Quick Start Card

## 🚀 30-Second Overview

You now have a complete lip-sync validation system for `vasa_trainer.py` that measures:
- ✅ Mel spectrogram ↔ lip motion correlation
- ✅ Audio → visual projection quality
- ✅ Temporal sync lag detection

## 📁 Files to Use

| File | Purpose |
|------|---------|
| `lipsync_validation_methods.py` | **Copy methods from here** → paste into `vasa_trainer.py` |
| `LIPSYNC_VALIDATION_INTEGRATION_GUIDE.md` | **Step-by-step integration** with exact line numbers |
| `LIPSYNC_VALIDATION_PLAN.md` | **Design details** and mathematical formulas |
| `LIPSYNC_VALIDATION_SUMMARY.md` | **Complete overview** of implementation |
| `overfit_config.yaml` | **Already updated** with validation config |

## ⚡ 3-Step Integration

### Step 1: Copy Methods (5 min)
```bash
# Open vasa_trainer.py, go to line 2890 (after validate() method)
# Copy ALL methods from lipsync_validation_methods.py
# Paste them before _apply_condition_dropout()
```

**Methods to copy (6 total):**
- `_extract_lip_energy()`
- `_compute_lip_openness()`
- `_extract_lip_motion_per_frame()`
- `_compute_mel_sync_metrics()`
- `_compute_audio_projection_metrics()`
- `_compute_temporal_alignment_metrics()`

### Step 2: Update validate() (10 min)
```python
# In vasa_trainer.py, find line ~2849 (where frames are generated)
# Replace the existing sync evaluation block with the enhanced version
# See LIPSYNC_VALIDATION_INTEGRATION_GUIDE.md for exact code
```

**Key change:** Insert lip-sync metric computation before existing `evaluate_sync_quality()`

### Step 3: Enable & Test (5 min)
```yaml
# In overfit_config.yaml (already done!):
validation:
  enabled: true  # Change from false to true
  frequency: 5
```

```bash
# Run training
python train_overfit.py

# Check WandB for new metrics:
# - val/mel_lip_correlation
# - val/audio_pred_correlation
# - val/sync_lag_frames
```

## 📊 What You'll See in WandB

### New Metric Categories

**Mel Spectrogram Sync:**
```
val/mel_lip_correlation: 0.68  (target > 0.7)
val/mel_lip_mse: 0.045         (target < 0.1)
```

**Audio Projection:**
```
val/audio_pred_correlation: 0.71    (how well model predicts)
val/audio_target_correlation: 0.82  (ideal correlation)
val/audio_projection_error: 0.11    (gap to close)
```

**Temporal Alignment:**
```
val/sync_lag_frames: -1.0    (lips lag audio by 1 frame)
val/sync_lag_ms: -40ms       (temporal offset)
val/sync_peak_correlation: 0.75
val/sync_at_zero_lag: 0.72
```

## 🎯 Metric Targets

| Metric | Good | Excellent | Action if Poor |
|--------|------|-----------|----------------|
| `mel_lip_correlation` | > 0.7 | > 0.8 | ↑ `lambda_audio_lip` |
| `mel_lip_mse` | < 0.1 | < 0.05 | ↑ `lambda_lips` |
| `audio_projection_error` | < 0.15 | < 0.10 | ↑ `lambda_audio_expr_coupling` |
| `sync_lag_frames` | ±2 | 0 | Fix dataset timing |
| `sync_lag_ms` | ±80ms | 0ms | Adjust audio offset |

## 🔧 Quick Fixes

### Issue: Metrics are all zeros
```python
# Check logs for errors
tail -f training.log | grep "Error computing"

# Verify MediaPipe is installed
pip install mediapipe

# Verify torchaudio is installed
pip install torchaudio
```

### Issue: Validation too slow
```yaml
# In config:
validation:
  max_batches: 1  # Reduce from 2
  lipsync:
    compute_temporal_alignment: false  # Disable slowest metric
```

### Issue: Poor correlation scores
```yaml
# Increase audio-lip loss weights:
loss:
  lambda_audio_lip: 5.0        # from 3.0
  lambda_lips: 3.0             # from 2.0
  lambda_mouth_openness: 15.0  # from 10.0
```

## 📈 How to Use Metrics

### 1. Model Selection
```python
# Choose checkpoint with best mel_lip_correlation
best_checkpoint = max(checkpoints, key=lambda x: x['val/mel_lip_correlation'])
```

### 2. Debugging Sync Issues
```
If sync_lag_frames != 0:
  → Systematic timing problem
  → Fix dataset audio/video alignment

If audio_projection_error > 0.2:
  → Model not learning audio-visual mapping
  → Increase lambda_audio_expr_coupling

If mel_lip_correlation < 0.6:
  → Poor overall sync
  → Increase lambda_audio_lip
```

### 3. Tracking Improvement
```
Watch WandB for trends:
- mel_lip_correlation should increase over epochs
- audio_projection_error should decrease
- sync_lag_frames should stabilize near 0
```

## ⚠️ Common Pitfalls

❌ **Don't forget imports:**
```python
import torchaudio
import mediapipe as mp
import cv2
```

❌ **Don't enable validation before integration:**
```yaml
# First: Integrate methods
# Then: Enable validation
validation:
  enabled: true
```

❌ **Don't expect perfect scores immediately:**
- Early training: mel_lip_correlation ~0.3-0.5
- Mid training: mel_lip_correlation ~0.6-0.7
- Late training: mel_lip_correlation ~0.7-0.8

## ✅ Verification Checklist

- [ ] Copied 6 methods to `vasa_trainer.py`
- [ ] Updated `validate()` method
- [ ] Added imports (torchaudio, mediapipe, cv2)
- [ ] Config has `lipsync` section
- [ ] Training runs without errors
- [ ] WandB shows `val/mel_*` metrics
- [ ] Metrics have reasonable values (not 0 or NaN)

## 🎓 Learn More

- **Integration Details**: `LIPSYNC_VALIDATION_INTEGRATION_GUIDE.md`
- **Design Rationale**: `LIPSYNC_VALIDATION_PLAN.md`
- **Full Summary**: `LIPSYNC_VALIDATION_SUMMARY.md`
- **Method Code**: `lipsync_validation_methods.py`

## 📞 Need Help?

1. **Check integration guide** for detailed debugging tips
2. **Look at method docstrings** in `lipsync_validation_methods.py`
3. **Enable debug logging**:
   ```yaml
   logging:
     level: "DEBUG"
   ```
4. **Test methods individually**:
   ```python
   test_frames = torch.randn(1, 5, 3, 512, 512)
   lip_energy = trainer._extract_lip_energy(test_frames)
   print(f"Shape: {lip_energy.shape}")  # Should be [1, 5]
   ```

## 🏁 You're Ready!

Everything is implemented and documented. Just integrate and test!

**Estimated time:** 20 minutes
**Difficulty:** Low (mostly copy-paste)
**Value:** High (quantitative lip-sync validation)

---

**Quick command to get started:**
```bash
# 1. Open integration guide
cat LIPSYNC_VALIDATION_INTEGRATION_GUIDE.md

# 2. Open vasa_trainer.py at line 2890
# 3. Copy methods from lipsync_validation_methods.py
# 4. Update validate() method (see guide)
# 5. Enable validation in config
# 6. Run training!
python train_overfit.py
```

Good luck! 🚀
