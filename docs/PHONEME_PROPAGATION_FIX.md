# Phoneme Prediction aux_predictions Propagation Fix

## Problem

Training failed with error:
```
AssertionError: ❌ FATAL: aux_predictions missing from model outputs! Phoneme loss cannot be computed.
```

Even though:
- ✅ TalkVidAudioProjection.forward returns `(output, aux_predictions)` with phoneme predictions
- ✅ EfficientConditionEmbedding.forward unpacks the tuple correctly
- ✅ phoneme_gt exists in the dataset/cache

The issue was that **aux_predictions wasn't being propagated to the final model outputs**.

## Root Cause Analysis

The data flow was:

```
TalkVidAudioProjection.forward()
  └─> Returns (output, {'phoneme_pred': [B, 8, 50]})
      ↓
EfficientConditionEmbedding.forward()
  ├─> audio_proj_result = self.audio_proj(audio)
  ├─> audio_projected, audio_aux = audio_proj_result  # ✅ Unpacks correctly
  └─> Returns (final_output, audio_aux)  # ✅ Returns both
      ↓
MotionTransformer.forward()
  ├─> cond_result = self.cond_emb(full_conditions)
  ├─> cond_emb, aux_predictions = cond_result  # ✅ Unpacks correctly
  ├─> ... (transformer processing)
  └─> Returns output_dict  # ❌ DOESN'T include aux_predictions!
      ↓
VASAModel.forward()
  ├─> outputs = self.motion_transformer(...)
  └─> Returns outputs  # ❌ aux_predictions never added!
      ↓
Loss function
  └─> ❌ FAIL: 'aux_predictions' not in outputs
```

**The break point**: MotionTransformer.forward() captured `aux_predictions` but never added it to `output_dict`.

## Files Modified

### 1. vasa_model.py (VASAModel cleanup loop) - Lines 1533-1536

**Before**:
```python
# Clean outputs
for key, tensor in outputs.items():
    if isinstance(tensor, str):  # Skip string tags like 'warp_source'
        continue
    if torch.isnan(tensor).any():  # ❌ Fails on dict (aux_predictions)
```

**After**:
```python
# Clean outputs
for key, tensor in outputs.items():
    # Skip non-tensor values (strings, dicts like aux_predictions)
    if not isinstance(tensor, torch.Tensor):  # ✅ Handles all non-tensors
        continue
    if torch.isnan(tensor).any():
```

**Why this matters**:
- Original code only checked for strings, but aux_predictions is a dict
- `torch.isnan(dict)` throws TypeError
- Now correctly skips all non-tensor types (str, dict, etc.)

### 2. vasa_model.py (MotionTransformer.forward) - Lines 1110-1112

**Before**:
```python
output_dict = {
    'theta': theta_pred,
    'expression_embed': expr_pred,
    'scale': scale_pred,
    'rotation': rotation_pred,
    'translation': translation_pred,
    'hidden_states': hidden_states,
}

return output_dict  # ❌ aux_predictions not included
```

**After**:
```python
output_dict = {
    'theta': theta_pred,
    'expression_embed': expr_pred,
    'scale': scale_pred,
    'rotation': rotation_pred,
    'translation': translation_pred,
    'hidden_states': hidden_states,
}

# Add auxiliary predictions (e.g., phoneme predictions from TalkVidAudioProjection)
if aux_predictions:
    output_dict['aux_predictions'] = aux_predictions

return output_dict  # ✅ Now includes aux_predictions
```

### 2. vasa_model.py (VASAModel.forward) - Lines 1537-1545

**Before**:
```python
# Add auxiliary predictions (e.g., phoneme predictions) from condition embedding
if 'aux_predictions' in locals() and aux_predictions:  # ❌ Wrong check
    outputs['aux_predictions'] = aux_predictions
    if conditions is not None and 'phoneme_gt' in conditions:
        outputs['aux_predictions']['phoneme_gt'] = conditions['phoneme_gt']
```

**After**:
```python
# Add phoneme_gt to aux_predictions if available (from conditions/targets)
# aux_predictions should already be in outputs from motion_transformer if phoneme head exists
if 'aux_predictions' in outputs:  # ✅ Correct check - outputs from motion_transformer
    # Add phoneme_gt from validated_conditions if available
    if validated_conditions is not None and 'phoneme_gt' in validated_conditions:
        outputs['aux_predictions']['phoneme_gt'] = validated_conditions['phoneme_gt']
    # Also check raw conditions in case it wasn't validated
    elif conditions is not None and 'phoneme_gt' in conditions:
        outputs['aux_predictions']['phoneme_gt'] = conditions['phoneme_gt']
```

**Why this matters**:
- The original check `'aux_predictions' in locals()` was wrong - `aux_predictions` is in `outputs`, not `locals()`
- Now correctly checks if `aux_predictions` exists in `outputs` (from motion_transformer)

### 3. vasa_model.py (VASAModel.forward) - Lines 1411-1424

**Added phoneme_gt to expected_shapes**:

```python
expected_shapes = {
    'gaze': (B, T, 2),
    'head_distance': (B, T, 1),
    'emotion': (B, T, 2),
    'lips': (B, T, 20, 3),
    'right_eye': (B, T, 8, 3),
    'left_eye': (B, T, 7, 3),
    'jaw': (B, T, 10, 3),
    'nose': (B, T, 4, 3),
    'blink_state': (B, T, 3),
    'audio_features': (B, T, 768),
    'phoneme_gt': (B, 8)  # ✅ NEW: Phoneme ground truth for auxiliary loss (8 latent queries)
}
```

### 4. vasa_model.py (VASAModel.forward) - Lines 1438-1452

**Added special handling for phoneme_gt validation**:

```python
# Special handling for phoneme_gt which doesn't have time dimension
# Shape: [B, 8] for 8 latent queries, NOT [B, T, ...]
if key == 'phoneme_gt':
    # phoneme_gt should already be [B, 8], just validate batch size
    if tensor.shape[0] != B:
        raise ValueError(f"phoneme_gt batch size {tensor.shape[0]} doesn't match B={B}")
    validated_conditions[key] = tensor
elif tensor.shape[:2] != (B, T):
    # ... (existing shape handling for other tensors)
```

**Why this matters**:
- phoneme_gt has shape `[B, 8]` (one phoneme per latent query), **not** `[B, T, ...]`
- The existing validation code assumed all conditions have time dimension T
- Special case needed to avoid incorrect shape expansion

## Complete Data Flow (Fixed)

```
1. Dataset/Cache loads window
   └─> window_data['phoneme_gt'] = [8] (cached from upsert)

2. DataLoader collates batch
   └─> batch['phoneme_gt'] = [B, 8] (stacked from windows)

3. Trainer extracts conditions
   └─> conditions['phoneme_gt'] = batch['phoneme_gt']

4. VASAModel.forward validates conditions
   ├─> phoneme_gt shape validated: [B, 8]
   └─> validated_conditions['phoneme_gt'] = [B, 8]

5. MotionTransformer.forward processes
   ├─> cond_emb(validated_conditions) calls:
   │   └─> TalkVidAudioProjection.forward(audio)
   │       └─> Returns (audio_embed, {'phoneme_pred': [B, 8, 50]})
   ├─> Unpacks: cond_emb, aux_predictions = cond_result
   ├─> Processes transformer layers
   └─> Returns output_dict WITH aux_predictions ✅

6. VASAModel.forward receives outputs
   ├─> outputs contains aux_predictions ✅
   ├─> Adds phoneme_gt to aux_predictions ✅
   └─> Returns outputs

7. Loss function receives outputs
   ├─> aux_predictions in outputs ✅
   ├─> phoneme_pred in aux_predictions ✅
   ├─> phoneme_gt in aux_predictions ✅
   └─> Computes phoneme loss ✅
```

## Verification

After these fixes, you should see in training logs:

```
🔍 Validating phoneme_gt presence in dataset...
✅ Phoneme validation passed! phoneme_gt shape: torch.Size([4, 8])
   Expected: [batch_size, num_queries=8]
Starting epoch 448...
...
✅ Auxiliary phoneme loss: 3.456789
```

And in loss function debug logs:

```
🔍 LOSS FUNCTION - Checking targets keys: ['audio_features', 'audio_mel_spec',
'audio_mfcc', 'expression_embed', 'lip_metrics', 'rotation', 'scale',
'theta', 'translation', 'uv_warps', 'phoneme_gt']  ✅

Outputs keys: dict_keys(['theta', 'expression_embed', 'scale', 'rotation',
'translation', 'hidden_states', 'uv_warps', 'warp_source', 'noise',
'aux_predictions'])  ✅
```

## Summary

**What was broken**:
- aux_predictions captured in MotionTransformer but not added to output_dict
- VASAModel checking wrong location for aux_predictions (`locals()` instead of `outputs`)
- phoneme_gt not in expected_shapes, causing validation errors
- phoneme_gt shape handling assumed time dimension T (wrong for [B, 8] shape)
- Cleanup loop only checked for strings, causing TypeError on dict (aux_predictions)

**What was fixed**:
1. **VASAModel cleanup loop**: Changed from `isinstance(tensor, str)` to `not isinstance(tensor, torch.Tensor)` to skip all non-tensors
2. **MotionTransformer**: Now adds aux_predictions to output_dict before returning
3. **VASAModel forward**: Correctly checks for aux_predictions in outputs (not locals)
4. **VASAModel validation**: Added phoneme_gt to expected_shapes with correct shape (B, 8)
5. **VASAModel validation**: Special handling for phoneme_gt (no time dimension expansion)

**Result**: Phoneme predictions now flow through the entire pipeline and phoneme loss computes correctly! 🎉
