# Lip-Sync Validation Integration Guide

## Overview
This guide provides step-by-step instructions to integrate comprehensive lip-sync validation metrics into `vasa_trainer.py`.

## Files Created

1. **LIPSYNC_VALIDATION_PLAN.md** - Detailed design document
2. **lipsync_validation_methods.py** - All helper methods and metric computations
3. **This file** - Integration instructions

## Integration Steps

### Step 1: Add Helper Methods to VASATrainer Class

Open `vasa_trainer.py` and add the following methods after the `validate()` method (around line 2890, before `_apply_condition_dropout`):

```python
# ============================================================================
# LIP-SYNC VALIDATION HELPER METHODS
# ============================================================================

def _extract_lip_energy(self, generated_frames: torch.Tensor) -> torch.Tensor:
    """Extract lip motion energy from generated video frames using MediaPipe."""
    # Copy implementation from lipsync_validation_methods.py
    ...

def _compute_lip_openness(self, lips: torch.Tensor) -> torch.Tensor:
    """Compute lip openness (vertical opening) from lip landmark coordinates."""
    # Copy implementation from lipsync_validation_methods.py
    ...

def _extract_lip_motion_per_frame(self, generated_frames: torch.Tensor) -> torch.Tensor:
    """Extract per-frame lip motion magnitude from generated frames."""
    # Copy implementation from lipsync_validation_methods.py
    ...

# ============================================================================
# LIP-SYNC VALIDATION METRIC METHODS
# ============================================================================

def _compute_mel_sync_metrics(
    self,
    generated_frames: torch.Tensor,
    audio_segment: torch.Tensor,
    window_metadata: Dict
) -> Dict[str, float]:
    """Compute mel spectrogram-based synchronization metrics."""
    # Copy implementation from lipsync_validation_methods.py
    ...

def _compute_audio_projection_metrics(
    self,
    generated_sequence: Dict[str, torch.Tensor],
    audio_features: torch.Tensor,
    window: Dict
) -> Dict[str, float]:
    """Measure audio-to-visual projection quality."""
    # Copy implementation from lipsync_validation_methods.py
    ...

def _compute_temporal_alignment_metrics(
    self,
    generated_frames: torch.Tensor,
    audio_features: torch.Tensor,
    fps: int = 25
) -> Dict[str, float]:
    """Compute frame-by-frame alignment metrics with lag detection."""
    # Copy implementation from lipsync_validation_methods.py
    ...
```

**Quick Copy Command:**
```bash
# Copy the method implementations directly
cat lipsync_validation_methods.py >> vasa_trainer_methods.txt
# Then manually insert at line 2890 in vasa_trainer.py
```

### Step 2: Modify the `validate()` Method

Find line ~2849 in `vasa_trainer.py` where frames are generated:

**Before:**
```python
                                    try:
                                        generated_frames = self._generate_synced_frames(
                                            window['frames'][:, 0],  # Use first frame as identity
                                            generated_sequence
                                        )
                                        # Evaluate sync quality
                                        sync_metrics = self.loss_module.evaluate_sync_quality(
                                            generated_frames=generated_frames,
                                            audio_features=window['audio_features'],
                                            audio_mfcc=window.get('audio_mfcc')
                                        )
                                        metrics.update(sync_metrics)
```

**After:**
```python
                                    try:
                                        generated_frames = self._generate_synced_frames(
                                            window['frames'][:, 0],  # Use first frame as identity
                                            generated_sequence
                                        )

                                        # === LIP-SYNC VALIDATION METRICS ===
                                        logger.debug("Computing lip-sync validation metrics...")

                                        # 1. Mel spectrogram sync
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_mel_correlation', True):
                                            try:
                                                mel_metrics = self._compute_mel_sync_metrics(
                                                    generated_frames,
                                                    window.get('audio_segment', window.get('audio_features')),
                                                    window.get('metadata', {})
                                                )
                                                metrics.update(mel_metrics)
                                                logger.debug(f"Mel sync metrics: {mel_metrics}")
                                            except Exception as mel_e:
                                                logger.warning(f"Error computing mel sync metrics: {mel_e}")

                                        # 2. Audio projection quality
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_audio_projection', True):
                                            try:
                                                proj_metrics = self._compute_audio_projection_metrics(
                                                    generated_sequence,
                                                    window['audio_features'],
                                                    window
                                                )
                                                metrics.update(proj_metrics)
                                                logger.debug(f"Audio projection metrics: {proj_metrics}")
                                            except Exception as proj_e:
                                                logger.warning(f"Error computing audio projection metrics: {proj_e}")

                                        # 3. Temporal alignment
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_temporal_alignment', True):
                                            try:
                                                align_metrics = self._compute_temporal_alignment_metrics(
                                                    generated_frames,
                                                    window['audio_features'],
                                                    fps=25
                                                )
                                                metrics.update(align_metrics)
                                                logger.debug(f"Temporal alignment metrics: {align_metrics}")
                                            except Exception as align_e:
                                                logger.warning(f"Error computing temporal alignment metrics: {align_e}")

                                        # === EXISTING SYNC EVALUATION ===
                                        # Evaluate sync quality (existing code)
                                        sync_metrics = self.loss_module.evaluate_sync_quality(
                                            generated_frames=generated_frames,
                                            audio_features=window['audio_features'],
                                            audio_mfcc=window.get('audio_mfcc')
                                        )
                                        metrics.update(sync_metrics)
```

### Step 3: Add Import Statements

At the top of `vasa_trainer.py`, ensure you have these imports:

```python
import torchaudio  # For mel spectrogram transform
import mediapipe as mp  # For lip landmark extraction
import cv2  # For frame processing
```

### Step 4: Update Config (Already Done!)

The `overfit_config.yaml` has been updated with:

```yaml
validation:
  enabled: false  # Set to true when ready to test
  metrics: ["reconstruction_loss", "lip_sync", "mel_spectrogram_sync", "audio_projection", "temporal_alignment"]

  lipsync:
    compute_mel_correlation: true
    compute_audio_projection: true
    compute_temporal_alignment: true
    mel_spec_n_mels: 128
    mel_spec_n_fft: 1024
    temporal_lag_frames: 5
```

### Step 5: Test the Integration

1. **Enable validation in config:**
   ```yaml
   validation:
     enabled: true
     frequency: 5  # Validate every 5 epochs
     max_batches: 2
   ```

2. **Run training:**
   ```bash
   python train_overfit.py
   ```

3. **Check WandB for new metrics:**
   - `val/mel_lip_correlation` - Should be > 0.6 for good sync
   - `val/mel_lip_mse` - Should decrease over training
   - `val/audio_pred_correlation` - Model's audio-to-lip prediction
   - `val/audio_target_correlation` - Ideal correlation
   - `val/audio_projection_error` - Gap between pred and ideal
   - `val/sync_lag_frames` - Temporal offset (should be ~0)
   - `val/sync_lag_ms` - Lag in milliseconds
   - `val/sync_peak_correlation` - Best correlation across lags

## Expected Output

After successful integration, validation logs should show:

```
=== Validation Epoch 5 ===
Processing validation batch 1/2...
  Computing lip-sync validation metrics...
  Mel sync metrics: {'val/mel_lip_correlation': 0.68, 'val/mel_lip_mse': 0.045}
  Audio projection metrics: {'val/audio_pred_correlation': 0.71, 'val/audio_target_correlation': 0.82, ...}
  Temporal alignment metrics: {'val/sync_lag_frames': -1.0, 'val/sync_lag_ms': -40.0, ...}

=== Validation Metrics Summary ===
Reconstruction:
  - reconstruction_loss: 0.042
  - lips_pos_loss: 0.008

Mel Spectrogram Sync:
  - mel_lip_correlation: 0.68 ✓ (target > 0.6)
  - mel_lip_mse: 0.045

Audio Projection:
  - audio_pred_correlation: 0.71
  - audio_target_correlation: 0.82
  - audio_projection_error: 0.11 (lower is better)

Temporal Alignment:
  - sync_lag_frames: -1.0 (lips lag audio by 1 frame)
  - sync_lag_ms: -40ms
  - sync_peak_correlation: 0.75
```

## Debugging Tips

### Issue 1: MediaPipe fails to detect faces
**Symptom:** `_extract_lip_energy` returns all zeros

**Fix:** Check frame format and size
```python
# In _extract_lip_energy, add debug:
logger.debug(f"Frame shape: {frame_np.shape}, dtype: {frame_np.dtype}, range: [{frame_np.min()}, {frame_np.max()}]")
```

### Issue 2: Audio segment not in window
**Symptom:** `mel_sync_metrics` gets wrong audio data

**Fix:** Ensure dataset includes audio_segment
```python
# In vasa_dataset.py, check if audio_segment is returned in window
# Should have: window['audio_segment'] = audio_tensor
```

### Issue 3: Lips data missing from generated_sequence
**Symptom:** `audio_projection_metrics` returns empty dict

**Fix:** Verify model outputs lips
```python
# In vasa_trainer.py validate(), add:
logger.debug(f"Generated sequence keys: {generated_sequence.keys()}")
logger.debug(f"Window keys: {window.keys()}")
```

### Issue 4: Validation too slow
**Symptom:** Validation takes > 5 minutes

**Fix:** Reduce max_batches or disable expensive metrics
```yaml
validation:
  max_batches: 1  # Only 1 batch for quick validation
  lipsync:
    compute_temporal_alignment: false  # This is expensive
```

## Metric Interpretation Guide

| Metric | Good Value | Bad Value | Meaning |
|--------|-----------|-----------|---------|
| `val/mel_lip_correlation` | > 0.7 | < 0.5 | Audio-visual sync quality |
| `val/mel_lip_mse` | < 0.05 | > 0.2 | Energy alignment error |
| `val/audio_pred_correlation` | > 0.7 | < 0.5 | Model's audio understanding |
| `val/audio_projection_error` | < 0.15 | > 0.3 | Prediction accuracy gap |
| `val/sync_lag_frames` | 0 ± 2 | > 5 or < -5 | Temporal offset (frames) |
| `val/sync_lag_ms` | 0 ± 80ms | > 200ms | Temporal offset (time) |
| `val/sync_at_zero_lag` | Close to peak | << peak | Alignment at t=0 |

### What to Optimize

1. **If `mel_lip_correlation` is low (<0.6)**:
   - Increase `lambda_audio_lip` loss weight
   - Increase `lambda_lips` loss weight
   - Check audio feature quality

2. **If `sync_lag_frames` is consistently non-zero**:
   - Model has systematic lag/lead issue
   - May need temporal offset correction in dataset
   - Consider adding temporal shift augmentation

3. **If `audio_projection_error` is high (>0.2)**:
   - Model not learning audio-to-visual mapping well
   - Increase `lambda_audio_expr_coupling`
   - Check if audio features are informative

4. **If `sync_peak_correlation` >> `sync_at_zero_lag`**:
   - Strong sync but at wrong temporal offset
   - Need to adjust dataset timing
   - Consider adding temporal alignment loss

## Next Steps After Integration

1. **Run validation on checkpoints:**
   ```bash
   # Test on existing checkpoint
   python vasa_trainer.py --config overfit_config.yaml --validate_only --checkpoint checkpoints_overfit/best_checkpoint.pt
   ```

2. **Create WandB dashboard:**
   - Add custom charts for lip-sync metrics
   - Compare across training runs
   - Track improvement over epochs

3. **Ablation studies:**
   - Train with different `lambda_audio_lip` weights
   - Compare models using validation metrics
   - Find optimal loss configuration

4. **Extend to full dataset:**
   - Apply to `vasa_config.yaml` for full training
   - Monitor generalization across speakers/videos

## References

- **Design Doc**: `LIPSYNC_VALIDATION_PLAN.md`
- **Method Implementations**: `lipsync_validation_methods.py`
- **Config Changes**: `overfit_config.yaml` lines 393-410
- **Mel Spectrogram**: Synchformer/dataset/transforms.py:815
- **Existing Validation**: vasa_trainer.py:2714-2890

## Troubleshooting Support

If you encounter issues:

1. Check logs for specific error messages
2. Verify all imports are available (mediapipe, torchaudio)
3. Test helper methods individually:
   ```python
   # Test lip extraction
   test_frames = torch.randn(1, 5, 3, 512, 512)
   lip_energy = trainer._extract_lip_energy(test_frames)
   print(f"Lip energy shape: {lip_energy.shape}")  # Should be [1, 5]
   ```
4. Use debug mode to see detailed logs:
   ```yaml
   debug: true
   logging:
     level: "DEBUG"
   ```

Good luck with the integration! 🎉
