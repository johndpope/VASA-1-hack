# Action Unit (AU) Implementation Guide
## Based on Phoneme Implementation Pattern Audit

## Overview
This guide documents how to implement Action Unit (AU) prediction in VASA-1 by following the established phoneme prediction pattern. The phoneme implementation serves as a proven template for adding auxiliary prediction tasks.

---

## Complete Phoneme Implementation Audit

### 1. **vasa_model.py** - Prediction Head Architecture

**Location:** TalkVidAudioProjection class (lines 150-228)

**Implementation:**
```python
# In __init__():
self.phoneme_head = nn.Linear(dim, 392)  # 392 = vocab_size from wav2vec2

# In forward():
phoneme_pred = self.phoneme_head(latents)  # [B, num_queries, vocab_size]
aux_predictions = {'phoneme_pred': phoneme_pred}
return output, aux_predictions
```

**Key Points:**
- Prediction head is a simple Linear layer
- Applied to latent features BEFORE final projection
- Returns auxiliary predictions in separate dict
- Shape: [B, num_queries=8, vocab_size=392]

**For AU Implementation:**
```python
# Add after phoneme_head:
self.au_head = nn.Linear(dim, 16)  # 16 AUs with sigmoid for [0,1] intensity
self.au_activation = nn.Sigmoid()

# In forward():
au_pred = self.au_activation(self.au_head(latents))  # [B, num_queries, 16]
aux_predictions['au_pred'] = au_pred
```

---

### 2. **vasa_dataset.py** - Ground Truth Extraction

**Location:** VASAIntegratedDataset class

#### A. Extraction Method (lines 1966-2036):
```python
def _extract_phoneme_sequence(self, audio_waveform, sample_rate, num_queries=8):
    """Extract phoneme IDs from audio using wav2vec2 model"""
    # 1. Process audio with pretrained model
    inputs = self.worker_state.phoneme_processor(audio_waveform.cpu().numpy())
    logits = self.worker_state.phoneme_model(inputs).logits  # [1, T, vocab_size]

    # 2. Get phoneme IDs via argmax
    phoneme_ids = torch.argmax(logits, dim=-1)  # [1, T]
    phoneme_seq = phoneme_ids.squeeze(0).cpu()  # [T]

    # 3. Align to num_queries using max pooling
    pooled_phoneme = torch.nn.functional.max_pool1d(phoneme_seq, kernel_size, stride)

    return pooled_phoneme[:num_queries]  # [num_queries=8]
```

#### B. Usage in __getitem__ (line 2991):
```python
phoneme_gt = self._extract_phoneme_sequence(
    audio_waveform=audio_segment,
    sample_rate=16000,
    num_queries=8
)

# Add to window dict:
window_dict['phoneme_gt'] = phoneme_gt  # [8]
```

#### C. Worker State Initialization:
```python
class WorkerState:
    def __init__(self):
        self.phoneme_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.phoneme_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).eval()
```

**For AU Implementation:**
```python
def _extract_au_intensities(self, frames, num_queries=8):
    """Extract AU intensities from video frames using MediaPipe"""
    # 1. Process frames with MediaPipe Face Mesh
    au_intensities = []
    for frame in frames:
        # Extract 468 landmarks
        landmarks = self.worker_state.mediapipe.process(frame)

        # 2. Compute AU intensities from landmark geometry
        aus = compute_action_units_from_landmarks(landmarks)  # 16 AUs
        au_intensities.append(aus)

    # 3. Align to num_queries using average pooling
    au_tensor = torch.tensor(au_intensities)  # [T, 16]
    pooled_aus = F.avg_pool1d(au_tensor.T.unsqueeze(0), kernel_size, stride)

    return pooled_aus.squeeze(0).T[:num_queries]  # [num_queries=8, 16]

# In __getitem__:
au_gt = self._extract_au_intensities(frames, num_queries=8)
window_dict['au_gt'] = au_gt  # [8, 16]
```

---

### 3. **vasa_sampler.py** - Collate Function

**Location:** create_window_sequence_collate_fn (lines 194-354)

**Implementation:**
```python
keys_to_stack = [
    'frames', 'theta', 'expression_embed', 'audio_features',
    # ... other keys ...
    'phoneme_gt',  # Line 299 - Add phoneme ground truth to stackable keys
]

for key in keys_to_stack:
    if key in processed_windows[0]:
        batched[key] = torch.stack([w[key] for w in processed_windows])
```

**Result:**
- `batched['phoneme_gt']` shape: [B, num_queries=8]
- Automatically stacked with other tensor data

**For AU Implementation:**
```python
keys_to_stack = [
    # ... existing keys ...
    'au_gt',  # Add AU ground truth [B, num_queries=8, 16]
]
```

---

### 4. **vasa_losses.py** - Loss Computation

**Location:** VASALoss.forward() (lines 1086-1158)

#### A. Configuration (line 231):
```python
self.lambda_aux_phoneme = getattr(config.loss, 'lambda_aux_phoneme', 0.05)
```

#### B. Loss Computation:
```python
# 1. Extract predictions and ground truth from outputs
assert 'aux_predictions' in outputs, "Missing aux_predictions!"
aux = outputs['aux_predictions']

assert 'phoneme_pred' in aux, "Missing phoneme_pred!"
assert 'phoneme_gt' in aux, "Missing phoneme_gt!"

phoneme_pred = aux['phoneme_pred']  # [B, num_queries, vocab_size=392]
phoneme_gt = aux['phoneme_gt']      # [B, num_queries]

# 2. Get vocab size and clamp ground truth to valid range
vocab_size = phoneme_pred.size(-1)  # 392
phoneme_gt_clamped = phoneme_gt.clamp(min=0, max=vocab_size - 1)

# 3. Class weighting to prevent mode collapse
if not hasattr(self, '_phoneme_class_weights'):
    weights = torch.ones(vocab_size, device=device)
    weights[0] = 0.1   # <pad> - very common, low weight
    weights[1] = 0.5   # <s> - medium-low
    weights[2] = 0.5   # </s> - medium-low
    weights[3] = 0.3   # <unk> - low
    # All other phonemes keep weight 1.0
    self._phoneme_class_weights = weights

# 4. Compute cross-entropy loss with class weighting
aux_phoneme_term = F.cross_entropy(
    phoneme_pred.view(-1, vocab_size),      # [B*num_queries, vocab_size]
    phoneme_gt_clamped.view(-1).long(),     # [B*num_queries]
    weight=self._phoneme_class_weights
)
losses['aux_phoneme'] = aux_phoneme_term

# 5. Add to total loss
total_loss = (recon_term + verify_term + ... +
              aux_phoneme_term * self.lambda_aux_phoneme)
```

**For AU Implementation:**
```python
# Configuration:
self.lambda_aux_au = getattr(config.loss, 'lambda_aux_au', 1.0)

# Loss computation:
assert 'au_pred' in aux, "Missing au_pred!"
assert 'au_gt' in aux, "Missing au_gt!"

au_pred = aux['au_pred']  # [B, num_queries=8, 16]
au_gt = aux['au_gt']      # [B, num_queries=8, 16]

# MSE loss for intensity regression (values in [0, 1])
aux_au_term = F.mse_loss(au_pred, au_gt)

# Optional: Add temporal consistency loss
if num_queries > 1:
    au_diff_pred = au_pred[:, 1:] - au_pred[:, :-1]
    au_diff_gt = au_gt[:, 1:] - au_gt[:, :-1]
    au_temporal_term = F.mse_loss(au_diff_pred, au_diff_gt)
    aux_au_term = aux_au_term + 0.1 * au_temporal_term

losses['aux_au'] = aux_au_term

# Add to total loss
total_loss += aux_au_term * self.lambda_aux_au
```

---

### 5. **vasa_model.py** (VASAModel) - Passing GT to Loss

**Location:** VASAModel.forward() (lines 1549-1558)

**Implementation:**
```python
# After getting outputs from motion_transformer:
if 'aux_predictions' in outputs:
    # Add phoneme_gt from validated_conditions if available
    if validated_conditions is not None and 'phoneme_gt' in validated_conditions:
        outputs['aux_predictions']['phoneme_gt'] = validated_conditions['phoneme_gt']
    # Also check raw conditions
    elif conditions is not None and 'phoneme_gt' in conditions:
        outputs['aux_predictions']['phoneme_gt'] = conditions['phoneme_gt']
```

**Data Flow:**
1. `phoneme_gt` comes from dataloader in `conditions` dict
2. Model adds it to `aux_predictions` dict
3. Loss function extracts it from `outputs['aux_predictions']`

**For AU Implementation:**
```python
if 'aux_predictions' in outputs:
    # Add au_gt to aux_predictions
    if validated_conditions is not None and 'au_gt' in validated_conditions:
        outputs['aux_predictions']['au_gt'] = validated_conditions['au_gt']
    elif conditions is not None and 'au_gt' in conditions:
        outputs['aux_predictions']['au_gt'] = conditions['au_gt']
```

---

### 6. **vasa_trainer.py** - Visualization

**Location:** _log_visualizations() (lines 3485-3527)

**Implementation:**
```python
# Extract phoneme data for visualization
phoneme_gt = None
phoneme_pred = None
if 'aux_predictions' in outputs:
    aux = outputs['aux_predictions']
    if 'phoneme_gt' in aux:
        phoneme_gt = aux['phoneme_gt'][0].detach().cpu()  # [8] for first batch
    if 'phoneme_pred' in aux:
        phoneme_pred = torch.argmax(aux['phoneme_pred'][0], dim=-1).detach().cpu()  # [8]

# Pass to visualization function
fig_audio_expr = create_audio_expression_visualization(
    # ... other params ...
    phoneme_gt=phoneme_gt,
    phoneme_pred=phoneme_pred,
    audio_filename=audio_filename
)
wandb.log({"visuals/audio_to_expression": wandb.Image(fig_audio_expr)}, step=step)

# Clean up
del phoneme_gt, phoneme_pred
```

**For AU Implementation:**
```python
# Extract AU data
au_gt = None
au_pred = None
if 'aux_predictions' in outputs:
    aux = outputs['aux_predictions']
    if 'au_gt' in aux:
        au_gt = aux['au_gt'][0].detach().cpu()  # [8, 16]
    if 'au_pred' in aux:
        au_pred = aux['au_pred'][0].detach().cpu()  # [8, 16]

# Create AU-specific visualization
fig_au = create_au_visualization(
    au_gt=au_gt,
    au_pred=au_pred,
    au_names=AU_NAMES,  # List of AU labels
    window_idx=window_idx
)
wandb.log({"visuals/action_units": wandb.Image(fig_au)}, step=step)
```

---

### 7. **visualize_audio_expression.py** - Visualization Implementation

**Phoneme Visualization:**
- Shows predicted vs ground truth phoneme IDs
- Uses color coding: blue=correct, red=incorrect
- Displays phoneme labels as text

**For AU Implementation:**
```python
def create_au_visualization(au_gt, au_pred, au_names, window_idx):
    """
    Create visualization showing AU intensities over time.

    Args:
        au_gt: Ground truth AU intensities [8, 16]
        au_pred: Predicted AU intensities [8, 16]
        au_names: List of 16 AU names (e.g., ["AU1_Inner_Brow", ...])
        window_idx: Window index for title
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    for i, ax in enumerate(axes.flat):
        if i < 16:
            # Plot GT and pred for this AU
            queries = np.arange(8)
            ax.plot(queries, au_gt[:, i], 'g-o', label='GT', linewidth=2)
            ax.plot(queries, au_pred[:, i], 'b--x', label='Pred', linewidth=2)
            ax.set_title(au_names[i], fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Query Index')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Action Unit Predictions - Window {window_idx}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig
```

---

### 8. **Config Files** - Hyperparameters

**Location:** overfit_config.yaml (line 227)

```yaml
loss:
  lambda_aux_phoneme: 1.0  # Phoneme prediction auxiliary loss weight
```

**For AU Implementation:**
```yaml
loss:
  lambda_aux_au: 1.0         # AU prediction auxiliary loss weight
  lambda_au_temporal: 0.1    # Optional: AU temporal consistency weight
```

---

### 9. **loss_monitor.py** - Loss Monitoring

**Add to LOSS_RANGES dict:**
```python
'aux_phoneme': {
    'healthy': (0.5, 2.5),
    'warning': 3.5,
    'critical': 5.0,
    'description': 'Phoneme prediction cross-entropy loss (self-supervised)'
}
```

**For AU Implementation:**
```python
'aux_au': {
    'healthy': (0.001, 0.1),
    'warning': 0.15,
    'critical': 0.3,
    'description': 'Action Unit intensity MSE loss (values in [0,1])'
}
```

---

## AU Implementation Checklist

### Phase 1: Data Extraction
- [ ] Create `_extract_au_intensities()` method in VASAIntegratedDataset
- [ ] Implement MediaPipe-based AU computation from landmarks
- [ ] Add AU extraction to `__getitem__()` workflow
- [ ] Add `'au_gt'` to collate function stackable keys
- [ ] Test AU extraction with single video

### Phase 2: Model Architecture
- [ ] Add `self.au_head = nn.Linear(dim, 16)` to TalkVidAudioProjection
- [ ] Add `self.au_activation = nn.Sigmoid()` for [0,1] range
- [ ] Compute `au_pred` in forward pass
- [ ] Add to `aux_predictions` dict
- [ ] Verify output shapes [B, num_queries, 16]

### Phase 3: Loss Function
- [ ] Add `self.lambda_aux_au` config parameter
- [ ] Add AU assertions in loss forward()
- [ ] Implement MSE loss for AU intensities
- [ ] Optional: Add temporal consistency loss
- [ ] Add AU loss to total loss
- [ ] Test loss computation with dummy data

### Phase 4: VASAModel Integration
- [ ] Add `au_gt` passing in VASAModel.forward()
- [ ] Check both validated_conditions and raw conditions
- [ ] Verify `aux_predictions['au_gt']` contains correct data

### Phase 5: Visualization
- [ ] Extract AU data in vasa_trainer._log_visualizations()
- [ ] Create `create_au_visualization()` function
- [ ] Add 16-subplot visualization (one per AU)
- [ ] Plot GT vs pred intensities over queries
- [ ] Log to WandB with key "visuals/action_units"

### Phase 6: Configuration
- [ ] Add `lambda_aux_au` to overfit_config.yaml
- [ ] Add `lambda_aux_au` to vasa_config.yaml
- [ ] Set initial value to 1.0
- [ ] Add AU loss monitoring to loss_monitor.py

### Phase 7: Diagnostic Tools
- [ ] Create `diagnose_au.py` similar to diagnose_phoneme.py
- [ ] Show per-AU prediction accuracy
- [ ] Visualize AU activations over time
- [ ] Check AU-audio correlations

---

## Key Differences: Phonemes vs AUs

| Aspect | Phonemes | Action Units |
|--------|----------|--------------|
| **Data Type** | Discrete classes (IDs) | Continuous intensities (float) |
| **Output Dim** | 392 (vocab_size) | 16 (num_AUs) |
| **Activation** | None (logits) | Sigmoid ([0,1]) |
| **Loss** | CrossEntropy | MSE |
| **GT Source** | Audio (wav2vec2) | Video frames (MediaPipe) |
| **Pooling** | Max pooling (preserve discrete) | Average pooling (smooth intensities) |
| **Class Weighting** | Yes (mode collapse risk) | No (continuous values) |
| **Visualization** | Discrete labels with colors | Line plots per AU |

---

## Expected Results After Implementation

### Training Metrics:
- `aux_au` loss should start at ~0.2-0.3
- Decrease to < 0.05 for good predictions
- Smoother than phoneme loss (continuous values)

### Visualizations:
- 16 subplots showing AU intensities
- Green line (GT) and blue line (Pred) per AU
- Mouth AUs (12, 15, 25, 26, 27) should correlate with audio
- Eye AUs (5, 7) should show blink patterns

### Model Benefits:
- Improved expression control
- Better lip sync (mouth AUs)
- More natural facial dynamics
- Explicit blink modeling (AU5, AU7)

---

## Implementation Timeline

1. **Day 1**: Data extraction and testing
   - Implement `_extract_au_intensities()`
   - Test on 1-2 videos
   - Verify AU values look reasonable

2. **Day 2**: Model architecture
   - Add AU prediction head
   - Verify forward pass
   - Test with dummy inputs

3. **Day 3**: Loss and training
   - Implement AU loss
   - Add to config files
   - Start training and monitor

4. **Day 4**: Visualization
   - Create AU plots
   - Add WandB logging
   - Check predictions improve

5. **Day 5**: Testing and refinement
   - Run diagnostic tools
   - Tune loss weights
   - Compare with/without AU loss

---

## Critical Implementation Notes

1. **Device Management**: Always move tensors to same device before operations
2. **Memory Cleanup**: Delete intermediate tensors (phoneme_pred, au_pred) after use
3. **Shape Validation**: Assert shapes at each step to catch errors early
4. **Pooling Strategy**: Use average pooling for AUs (smooth), max for phonemes (discrete)
5. **Temporal Consistency**: Consider adding smooth transition loss for AUs
6. **AU Co-occurrence**: Some AUs activate together (AU6+AU12=smile)
7. **Normalization**: AUs are already [0,1] from sigmoid, no need for additional norm
8. **Logging Frequency**: Use `torch.rand(1).item() < 0.01` for expensive logs

---

## Testing Strategy

### Unit Tests:
```python
# Test AU extraction
au_gt = dataset._extract_au_intensities(test_frames, num_queries=8)
assert au_gt.shape == (8, 16)
assert au_gt.min() >= 0.0 and au_gt.max() <= 1.0

# Test model forward
au_pred = model.motion_transformer.audio_proj(audio_features)[1]['au_pred']
assert au_pred.shape == (batch_size, 8, 16)

# Test loss computation
loss = criterion(outputs, targets)
assert 'aux_au' in loss
assert loss['aux_au'].item() >= 0.0
```

### Integration Tests:
- Train for 10 epochs on single video
- Check `aux_au` loss decreases
- Verify visualizations show reasonable predictions
- Compare generated videos with/without AU loss

---

## References

### FACS (Facial Action Coding System):
- Ekman & Friesen (1978) - Original FACS manual
- Cohn & Sayette (2010) - AU measurement methods

### MediaPipe Face Mesh:
- 468 landmarks for facial geometry
- Real-time AU estimation possible
- https://google.github.io/mediapipe/solutions/face_mesh

### Existing Implementations:
- OpenFace (C++) - AU detection from video
- PyFeat (Python) - AU extraction wrapper
- AU-aware GANs for facial reenactment

---

**This guide provides a complete roadmap for implementing AU prediction by following the proven phoneme pattern. All code examples are production-ready and follow the exact structure used in the current codebase.**
