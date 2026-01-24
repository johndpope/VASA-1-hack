# Lip-Sync Validation Enhancement Plan

## Overview
Add comprehensive lip-sync validation to `vasa_trainer.py` focusing on mel spectrogram analysis, audio projection, and mouth/lips losses to validate the generation of lip-synced videos.

## Current State (vasa_trainer.py:2714-2890)
The existing `validate()` function:
- ✅ Has basic reconstruction loss metrics
- ✅ Has control loss metrics (when not skipping expensive)
- ✅ Has sync evaluation using evaluate_sync_quality()
- ❌ **Missing**: Detailed mel spectrogram analysis
- ❌ **Missing**: Dedicated mouth/lips loss tracking
- ❌ **Missing**: Audio-visual correlation visualization
- ❌ **Missing**: Temporal audio-lip alignment metrics

## Proposed Enhancements

### 1. Mel Spectrogram Analysis Metrics
**Goal**: Validate audio-visual alignment using mel spectrogram correlation

**Implementation** (new method `_compute_mel_sync_metrics`):
```python
def _compute_mel_sync_metrics(self, generated_frames, audio_segment, window_metadata):
    """
    Compute mel spectrogram-based synchronization metrics.

    Args:
        generated_frames: [B, T, C, H, W] generated video frames
        audio_segment: [B, audio_samples] raw audio waveform
        window_metadata: Dict with timing info

    Returns:
        metrics: Dict with mel_correlation, mel_mse, mel_dtw_distance
    """
    # Extract mel spectrogram from audio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=audio_segment.shape[-1] // T,  # Match video frames
        n_mels=128,
        f_min=0.0,
        f_max=8000.0
    )
    mel_spec = mel_transform(audio_segment)  # [B, 128, T]
    mel_spec_log = torch.log(mel_spec + 1e-6)  # Log scale

    # Extract lip motion energy from generated frames
    # Using MediaPipe to extract lip landmarks -> compute openness
    lip_energy = self._extract_lip_energy(generated_frames)  # [B, T]

    # Compute audio energy from mel spectrogram (sum over frequency bins)
    audio_energy = mel_spec_log.mean(dim=1)  # [B, T]

    # Normalize both signals
    lip_energy_norm = (lip_energy - lip_energy.mean()) / (lip_energy.std() + 1e-6)
    audio_energy_norm = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)

    # Metrics:
    # 1. Cross-correlation (measure sync alignment)
    correlation = F.cosine_similarity(lip_energy_norm, audio_energy_norm, dim=1).mean()

    # 2. MSE between normalized energies
    mse = F.mse_loss(lip_energy_norm, audio_energy_norm)

    # 3. Optional: DTW distance for temporal alignment
    # dtw_distance = compute_dtw(lip_energy, audio_energy)

    return {
        'val/mel_lip_correlation': correlation.item(),
        'val/mel_lip_mse': mse.item(),
        # 'val/mel_dtw_distance': dtw_distance
    }
```

### 2. Enhanced Mouth/Lips Loss Tracking
**Goal**: Track mouth-specific losses separately in validation

**Current Issues**:
- `_compute_reconstruction_losses()` includes lips losses but they're mixed with other metrics
- Need to extract and emphasize lip-specific metrics

**Implementation**:
```python
# In validate() method, after reconstruction losses:
if 'lips' in window:
    # Extract lip-specific losses
    lip_specific_metrics = {
        'val/lips_pos_loss': metrics.get('lips_pos_loss', 0),
        'val/lips_vel_loss': metrics.get('lips_vel_loss', 0),
        'val/lips_total': metrics.get('lips_total', 0),
        'val/audio_lip_correlation': metrics.get('audio_lip_correlation', 0),
        'val/mouth_openness_direct': metrics.get('mouth_openness_direct', 0),
        'val/mouth_perceptual': metrics.get('mouth_perceptual', 0),
    }

    # Log separately for visibility
    self.val_metrics.update(lip_specific_metrics)
```

### 3. Audio Projection Quality Metrics
**Goal**: Measure how well audio features project to visual lip motion

**Implementation** (new method `_compute_audio_projection_metrics`):
```python
def _compute_audio_projection_metrics(self, generated_sequence, audio_features, window):
    """
    Measure audio-to-visual projection quality.

    Compares:
    - Audio features -> Expected lip motion (from model)
    - Generated lip motion -> Ground truth lips

    Returns metrics on projection accuracy.
    """
    # Extract predicted lips from generated_sequence
    pred_lips = generated_sequence['lips'] if 'lips' in generated_sequence else None
    target_lips = window.get('lips')

    if pred_lips is None or target_lips is None:
        return {}

    # 1. Audio-conditioned lip prediction accuracy
    # Compare how well audio features correlate with predicted lips
    B, T, D = audio_features.shape
    audio_energy = audio_features.norm(dim=-1)  # [B, T]

    # Compute lip openness from predicted lips (assuming lips: [B, T, 40])
    # Lip openness = vertical distance between upper/lower lip landmarks
    pred_lip_openness = self._compute_lip_openness(pred_lips)  # [B, T]
    target_lip_openness = self._compute_lip_openness(target_lips)  # [B, T]

    # Normalize
    audio_norm = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)
    pred_norm = (pred_lip_openness - pred_lip_openness.mean()) / (pred_lip_openness.std() + 1e-6)
    target_norm = (target_lip_openness - target_lip_openness.mean()) / (target_lip_openness.std() + 1e-6)

    # Metrics:
    # 1. Audio-to-prediction correlation
    audio_pred_corr = F.cosine_similarity(audio_norm, pred_norm, dim=1).mean()

    # 2. Audio-to-target correlation (ideal case)
    audio_target_corr = F.cosine_similarity(audio_norm, target_norm, dim=1).mean()

    # 3. Projection error (how far off is prediction from ideal)
    projection_error = audio_target_corr - audio_pred_corr

    return {
        'val/audio_pred_correlation': audio_pred_corr.item(),
        'val/audio_target_correlation': audio_target_corr.item(),
        'val/audio_projection_error': projection_error.item(),
    }
```

### 4. Temporal Alignment Metrics
**Goal**: Measure frame-by-frame audio-visual synchronization

**Implementation**:
```python
def _compute_temporal_alignment_metrics(self, generated_frames, audio_features, fps=25):
    """
    Compute frame-by-frame alignment metrics.

    Uses:
    - Cross-correlation with time lags to detect sync offset
    - Per-frame sync confidence scores
    """
    # Extract per-frame audio energy
    audio_energy = audio_features.norm(dim=-1)  # [B, T]

    # Extract per-frame lip motion
    lip_motion = self._extract_lip_motion_per_frame(generated_frames)  # [B, T]

    # Compute cross-correlation with time lags (-5 to +5 frames)
    max_lag = 5
    correlations = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_audio = audio_energy[:, :lag]
            shifted_lip = lip_motion[:, -lag:]
        elif lag > 0:
            shifted_audio = audio_energy[:, lag:]
            shifted_lip = lip_motion[:, :-lag]
        else:
            shifted_audio = audio_energy
            shifted_lip = lip_motion

        # Compute correlation at this lag
        corr = F.cosine_similarity(shifted_audio, shifted_lip, dim=1).mean()
        correlations.append(corr.item())

    # Find best lag (peak correlation)
    best_lag_idx = np.argmax(correlations)
    best_lag = best_lag_idx - max_lag  # Convert to actual lag
    best_corr = correlations[best_lag_idx]

    # Lag in milliseconds
    lag_ms = (best_lag / fps) * 1000

    return {
        'val/sync_lag_frames': best_lag,
        'val/sync_lag_ms': lag_ms,
        'val/sync_peak_correlation': best_corr,
        'val/sync_at_zero_lag': correlations[max_lag],  # Correlation at lag=0
    }
```

## Configuration Changes

Add to validation config in `overfit_config.yaml`:

```yaml
validation:
  enabled: true
  frequency: 5
  max_batches: 10
  skip_expensive_metrics: false
  metrics:
    - reconstruction_loss
    - control_loss
    - lip_sync
    - mel_spectrogram_sync  # NEW
    - audio_projection      # NEW
    - temporal_alignment    # NEW

  # Lip-sync specific settings
  lipsync:
    compute_mel_correlation: true
    compute_audio_projection: true
    compute_temporal_alignment: true
    mel_spec_n_mels: 128
    mel_spec_n_fft: 1024
    temporal_lag_frames: 5  # Check +/- 5 frames for sync
```

## Implementation Steps

1. **Add helper methods** to VASATrainer:
   - `_extract_lip_energy()` - Extract lip motion energy from frames
   - `_compute_lip_openness()` - Compute vertical lip opening
   - `_extract_lip_motion_per_frame()` - Extract per-frame lip motion
   - `_compute_mel_sync_metrics()` - Mel spectrogram correlation
   - `_compute_audio_projection_metrics()` - Audio projection quality
   - `_compute_temporal_alignment_metrics()` - Temporal sync metrics

2. **Modify `validate()` method**:
   - After generating frames (line 2853), add:
     ```python
     # Lip-sync specific validation
     if self.config.validation.lipsync.compute_mel_correlation:
         mel_metrics = self._compute_mel_sync_metrics(
             generated_frames, window['audio_segment'], window['metadata']
         )
         metrics.update(mel_metrics)

     if self.config.validation.lipsync.compute_audio_projection:
         proj_metrics = self._compute_audio_projection_metrics(
             generated_sequence, window['audio_features'], window
         )
         metrics.update(proj_metrics)

     if self.config.validation.lipsync.compute_temporal_alignment:
         align_metrics = self._compute_temporal_alignment_metrics(
             generated_frames, window['audio_features']
         )
         metrics.update(align_metrics)
     ```

3. **Update validation logging**:
   - Create separate WandB section for lip-sync metrics
   - Add visualization: mel spectrogram heatmap with lip motion overlay

## Expected Validation Output

After implementation, validation will log:

```
=== Validation Metrics ===
Reconstruction:
  - reconstruction_loss: 0.042
  - lips_pos_loss: 0.008
  - lips_vel_loss: 0.003
  - mouth_perceptual: 0.15

Mel Spectrogram Sync:
  - mel_lip_correlation: 0.78  # Higher is better (max 1.0)
  - mel_lip_mse: 0.023          # Lower is better

Audio Projection:
  - audio_pred_correlation: 0.72
  - audio_target_correlation: 0.85
  - audio_projection_error: 0.13  # Lower is better

Temporal Alignment:
  - sync_lag_frames: -1           # Negative = lips lag audio
  - sync_lag_ms: -40ms
  - sync_peak_correlation: 0.81
  - sync_at_zero_lag: 0.76       # Should be close to peak
```

## References

- **Mel Spectrogram**: Synchformer/dataset/transforms.py:815-823
- **MFCC**: vasa_dataset.py:1802-1859
- **Mouth Loss**: vasa_losses.py:253-348 (_extract_mouth_masks)
- **Audio-Lip Correlation**: vasa_losses.py:531-623 (compute_audio_lip_correlation)
- **Existing Validation**: vasa_trainer.py:2714-2890

## Benefits

1. **Quantitative lip-sync quality**: Numerical metrics to track sync improvement
2. **Debugging**: Identify temporal alignment issues (lag detection)
3. **Model selection**: Choose checkpoints with best audio-visual sync
4. **Ablation studies**: Measure impact of different loss weights on sync
5. **WandB visualization**: Track lip-sync metrics across training

## Next Steps

1. Implement helper methods for lip extraction
2. Add mel sync metrics computation
3. Add audio projection metrics
4. Add temporal alignment metrics
5. Update config schema
6. Test on validation set
7. Create WandB dashboard for lip-sync metrics
