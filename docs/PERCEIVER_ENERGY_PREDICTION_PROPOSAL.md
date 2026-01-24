# Perceiver Audio Energy Prediction - Implementation Proposal

## Problem Statement

Current training shows poor audio-lip correlation (see training logs: audio-lip correlation loss remains high after 205 epochs). The Perceiver-based audio projection (TalkVidAudioProjection) processes wav2vec2 features but doesn't explicitly learn to extract audio energy/loudness, which is **critical** for lip sync because:

- Loud speech → Wide mouth opening
- Quiet speech → Smaller lip movements
- Silence → Closed/relaxed lips

While the Perceiver *can* learn this implicitly through the main task, adding explicit energy prediction supervision will **accelerate convergence** significantly.

## Solution: Self-Supervised Audio Energy Prediction

Add an auxiliary prediction head to TalkVidAudioProjection that explicitly predicts audio energy. This is **self-supervised** - no manual labels needed, ground truth is computed from the input audio features.

### Why This Works

1. **Self-Supervised**: Energy ground truth = `torch.norm(audio_features, dim=-1)` (computed automatically)
2. **Direct Correlation**: Audio energy directly correlates with lip openness (we already have this loss)
3. **Forces Feature Extraction**: Perceiver must explicitly encode loudness in latent queries
4. **Low Overhead**: Single linear layer, minimal compute cost
5. **Proven Approach**: Similar to wav2vec2's own pre-training (predict masked features)

## Implementation Details

### File 1: `vasa_model.py` - TalkVidAudioProjection class

**Location**: Lines 108-215 (TalkVidAudioProjection class)

#### Step 1: Add energy prediction head to `__init__`

**Current code (line ~150):**
```python
self.proj_out = nn.Linear(dim, output_dim)
self.norm_out = nn.LayerNorm(output_dim)
```

**Add after line 150:**
```python
self.proj_out = nn.Linear(dim, output_dim)
self.norm_out = nn.LayerNorm(output_dim)

# Auxiliary energy prediction head (self-supervised)
# Predicts scalar audio energy to help Perceiver learn loudness features
self.energy_head = nn.Linear(dim, 1)
logger.info(f"   Added energy prediction head: {dim}D → 1D (self-supervised)")
```

#### Step 2: Modify `forward()` to compute and return energy

**Current code (lines 170-215):**
```python
def forward(self, x):
    # Log input shape (only once to avoid spam)
    if not hasattr(self, '_logged_forward'):
        logger.info(f"🎵 TalkVidAudioProjection forward pass:")
        logger.info(f"   Input shape: {x.shape} (batch_size, seq_len, {self.embedding_dim})")
        logger.info(f"   Processing through {self.depth} Perceiver layers with {self.num_queries} latent queries")
        self._logged_forward = True

    if self.pos_emb is not None:
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        x = x + pos_emb

    latents = self.latents.repeat(x.size(0), 1, 1)

    x = self.proj_in(x)

    if self.to_latents_from_mean_pooled_seq:
        meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
        meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
        # Rearrange manually
        meanpooled_latents = meanpooled_latents.view(meanpooled_latents.size(0), -1, latents.size(-1))
        latents = torch.cat((meanpooled_latents, latents), dim=-2)

    # Cross-layer skip connections for better gradient flow in deep perceiver
    block_residual = latents  # Save input to first block
    for idx, (attn, ff) in enumerate(self.layers):
        latents = attn(x, latents) + latents
        latents = ff(latents) + latents

        # Add block-level residual every 2 layers for stability
        if (idx + 1) % 2 == 0:
            latents = latents + block_residual  # Cross-layer skip connection
            block_residual = latents  # Update residual for next block

    latents = self.proj_out(latents)
    output = self.norm_out(latents)

    # Log output shape (only once)
    if not hasattr(self, '_logged_output'):
        logger.info(f"🎵 TalkVidAudioProjection output:")
        logger.info(f"   Output shape: {output.shape} (batch_size, {self.num_queries} queries, {self.output_dim}D)")
        logger.info(f"   ✅ Audio features projected and compressed via Perceiver attention")
        self._logged_output = True

    return output
```

**Replace with:**
```python
def forward(self, x):
    # Log input shape (only once to avoid spam)
    if not hasattr(self, '_logged_forward'):
        logger.info(f"🎵 TalkVidAudioProjection forward pass:")
        logger.info(f"   Input shape: {x.shape} (batch_size, seq_len, {self.embedding_dim})")
        logger.info(f"   Processing through {self.depth} Perceiver layers with {self.num_queries} latent queries")
        self._logged_forward = True

    # Save original input for energy ground truth computation
    x_original = x  # [B, T, 768]

    if self.pos_emb is not None:
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        x = x + pos_emb

    latents = self.latents.repeat(x.size(0), 1, 1)

    x = self.proj_in(x)

    if self.to_latents_from_mean_pooled_seq:
        meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
        meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
        # Rearrange manually
        meanpooled_latents = meanpooled_latents.view(meanpooled_latents.size(0), -1, latents.size(-1))
        latents = torch.cat((meanpooled_latents, latents), dim=-2)

    # Cross-layer skip connections for better gradient flow in deep perceiver
    block_residual = latents  # Save input to first block
    for idx, (attn, ff) in enumerate(self.layers):
        latents = attn(x, latents) + latents
        latents = ff(latents) + latents

        # Add block-level residual every 2 layers for stability
        if (idx + 1) % 2 == 0:
            latents = latents + block_residual  # Cross-layer skip connection
            block_residual = latents  # Update residual for next block

    # Compute auxiliary energy prediction (self-supervised)
    # This forces Perceiver to explicitly encode audio loudness in latent queries
    latents_pooled = latents.mean(dim=1)  # [B, dim] - average over queries
    energy_pred = self.energy_head(latents_pooled)  # [B, 1] - predicted energy

    # Compute ground truth energy from original input (no labels needed!)
    # L2 norm of audio features = energy/loudness
    audio_energy_gt = torch.norm(x_original, dim=-1).mean(dim=1, keepdim=True)  # [B, 1]

    latents = self.proj_out(latents)
    output = self.norm_out(latents)

    # Log output shape (only once)
    if not hasattr(self, '_logged_output'):
        logger.info(f"🎵 TalkVidAudioProjection output:")
        logger.info(f"   Output shape: {output.shape} (batch_size, {self.num_queries} queries, {self.output_dim}D)")
        logger.info(f"   Energy prediction: {energy_pred.shape} (self-supervised auxiliary task)")
        logger.info(f"   ✅ Audio features projected and compressed via Perceiver attention")
        self._logged_output = True

    # Return both main output and auxiliary predictions
    aux_predictions = {
        'energy_pred': energy_pred,      # [B, 1] - predicted energy
        'energy_gt': audio_energy_gt,    # [B, 1] - ground truth energy
    }

    return output, aux_predictions
```

#### Step 3: Update EfficientConditionEmbedding to handle tuple return

**Location**: `vasa_model.py` lines 484-523 (where audio_proj is called)

**Current code (line 493):**
```python
audio_projected = self.audio_proj(audio)  # [B, T, 512] for JoyVASA, [B, num_queries, 512] for TalkVid
```

**Replace with:**
```python
# Handle TalkVid returning tuple (output, aux_predictions)
audio_proj_output = self.audio_proj(audio)
if isinstance(audio_proj_output, tuple):
    # TalkVid returns (output, aux_predictions)
    audio_projected, audio_aux = audio_proj_output
else:
    # JoyVASA returns just output
    audio_projected = audio_proj_output
    audio_aux = None
```

**Then at the end of EfficientConditionEmbedding.forward() (around line 600):**

**Current code:**
```python
return final_output
```

**Replace with:**
```python
# Return both output and auxiliary predictions (if any)
if audio_aux is not None:
    return final_output, audio_aux
else:
    return final_output
```

#### Step 4: Update VASAModel.forward() to handle tuple return

**Location**: `vasa_model.py` line 973 (where cond_emb is called)

**Current code:**
```python
cond_emb = self.cond_emb(full_conditions)  # [B, T or C+T, d_model]
```

**Replace with:**
```python
# Handle condition embedding returning auxiliary predictions
cond_emb_output = self.cond_emb(full_conditions)
if isinstance(cond_emb_output, tuple):
    cond_emb, cond_aux = cond_emb_output
else:
    cond_emb = cond_emb_output
    cond_aux = None
```

**Then add auxiliary predictions to model outputs (around line 1050):**

**After line 1050 (where outputs dict is created):**
```python
# Add auxiliary predictions if available
if cond_aux is not None:
    outputs['aux_predictions'] = cond_aux
```

### File 2: `vasa_losses.py` - Add energy prediction loss

**Location**: In `compute_losses()` method, around line 1000 (before control losses section)

**Add this new section:**
```python
# =========================================================================
# AUXILIARY LOSSES: Self-supervised tasks to help Perceiver learn features
# =========================================================================

# Audio energy prediction loss (self-supervised)
if 'aux_predictions' in outputs:
    aux = outputs['aux_predictions']

    if 'energy_pred' in aux and 'energy_gt' in aux:
        energy_pred = aux['energy_pred']  # [B, 1]
        energy_gt = aux['energy_gt']      # [B, 1]

        # MSE loss between predicted and ground truth energy
        energy_loss = F.mse_loss(energy_pred, energy_gt)
        losses['aux_energy'] = energy_loss

        logger.debug(f"  Auxiliary energy loss: {energy_loss.item():.6f}")
        logger.debug(f"    Predicted energy range: [{energy_pred.min():.4f}, {energy_pred.max():.4f}]")
        logger.debug(f"    Ground truth energy range: [{energy_gt.min():.4f}, {energy_gt.max():.4f}]")
```

### File 3: `vasa_losses.py` - Add loss weight to total

**Location**: Around line 1076 (where total loss is computed)

**Current code:**
```python
total_loss = (
    reconstruction_term +
    dynamics_term +
    pose_term +
    # ... other terms ...
)
```

**Add:**
```python
# Auxiliary losses
aux_energy_term = losses.get('aux_energy', torch.tensor(0.0, device=device))

total_loss = (
    reconstruction_term +
    dynamics_term +
    pose_term +
    # ... other terms ...
    aux_energy_term * lambda_aux_energy +  # NEW
)
```

### File 4: `overfit_config.yaml` - Add loss weight

**Location**: In the `loss:` section, after existing lambda weights

**Add:**
```yaml
# Auxiliary self-supervised losses
lambda_aux_energy: 0.1          # Audio energy prediction (self-supervised)
```

### File 5: `loss_monitor.py` - Add to monitoring

**Location**: In `LOSS_RANGES` dict

**Add:**
```python
'aux_energy': {
    'healthy': (0.001, 0.1),
    'warning': 0.2,
    'critical': 0.5,
    'description': 'Auxiliary energy prediction loss (self-supervised). Helps Perceiver extract audio loudness.'
},
```

## Expected Benefits

### 1. Faster Lip Sync Convergence
- **Current**: Perceiver learns energy implicitly through main task (slow)
- **After**: Explicit supervision forces energy extraction (fast)
- **Expected speedup**: 2-3x faster audio-lip correlation learning

### 2. Better Audio-Lip Correlation
- **Current**: Poor correlation (see training logs)
- **After**: Perceiver explicitly encodes loudness → directly feeds lip openness loss
- **Expected improvement**: Correlation should improve from <0.3 to >0.7

### 3. More Robust Silent Frame Handling
- **Current**: Model may generate lip movement during silence
- **After**: Low energy prediction → model learns to keep lips relaxed
- **Expected**: Better silence handling

### 4. No Additional Data Required
- ✅ Self-supervised (ground truth = `torch.norm(audio)`)
- ✅ No phoneme labels needed
- ✅ No manual annotation required
- ✅ Works with existing dataset

## Implementation Checklist

- [ ] 1. Add `self.energy_head` to TalkVidAudioProjection `__init__` (vasa_model.py ~line 150)
- [ ] 2. Modify TalkVidAudioProjection `forward()` to compute and return energy (vasa_model.py lines 170-215)
- [ ] 3. Update EfficientConditionEmbedding to handle tuple return (vasa_model.py line 493)
- [ ] 4. Update EfficientConditionEmbedding return statement (vasa_model.py line 600)
- [ ] 5. Update VASAModel.forward() to handle tuple from cond_emb (vasa_model.py line 973)
- [ ] 6. Add auxiliary predictions to model outputs (vasa_model.py line 1050)
- [ ] 7. Add energy loss computation in vasa_losses.py (line ~1000)
- [ ] 8. Add energy term to total loss (vasa_losses.py line ~1076)
- [ ] 9. Add `lambda_aux_energy: 0.1` to overfit_config.yaml
- [ ] 10. Add `aux_energy` to loss_monitor.py LOSS_RANGES

## Testing Plan

1. **Verify implementation**:
   ```bash
   # Check for errors during initialization
   python -c "from vasa_model import TalkVidAudioProjection; print('OK')"
   ```

2. **Dry run**:
   ```bash
   # Run 1 training step to verify loss computation
   python train_overfit.py --max_steps 1
   ```

3. **Check logs**:
   ```bash
   # Verify energy prediction is working
   grep "aux_energy" train.log
   grep "Energy prediction" train.log
   ```

4. **Monitor convergence**:
   ```bash
   # Watch WandB for aux_energy loss decreasing
   # Should decrease from ~0.1 to ~0.01 in first 10 epochs
   ```

## Rollback Plan

If this causes issues:

1. **Disable loss** in config:
   ```yaml
   lambda_aux_energy: 0.0  # Disable without removing code
   ```

2. **Revert to simple return** in TalkVidAudioProjection:
   ```python
   return output  # Remove tuple return
   ```

3. **Comment out auxiliary sections** in vasa_losses.py

## References

- **SyncNet**: Uses L2-normalized embeddings for audio-visual correlation
- **wav2vec2**: Pre-trains with self-supervised prediction tasks
- **TalkVid**: Uses Perceiver for audio compression (similar architecture)
- **Energy-based features**: Common in speech processing (formants, MFCCs derived from energy)

## Alternative Approaches (Future Work)

If energy prediction works well, consider adding:

1. **Variance prediction**: `audio.var(dim=1)` → predicts speech dynamics
2. **Spectral centroid**: Frequency content → helps with vowel vs consonant
3. **Zero-crossing rate**: Periodicity → helps detect voiced vs unvoiced speech

All are self-supervised (no labels needed)!

---

**Status**: Ready for implementation
**Estimated effort**: ~30 minutes
**Risk**: Low (self-supervised, easy to disable)
**Expected impact**: High (2-3x faster lip sync convergence)
