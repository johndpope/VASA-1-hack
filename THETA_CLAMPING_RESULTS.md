# Theta/SRT Clamping Results

## Problem Identified

During inference, the diffusion model was predicting **extreme geometric parameters** causing severe face distortions:

### Before Clamping (from diagnostic tool):
```
Window 0:
  Scale: [-3.02, 2.88]        ❌ Should be [0.8, 1.2]
  Rotation: [-2.77, 2.75] rad ❌ Should be [-0.785, 0.785] (±45°)
  Translation: [-3.42, 2.62]  ❌ Should be [-0.3, 0.3]

Window 1:
  Scale: [-2.31, 2.31]
  Rotation: [-2.58, 2.08] rad
  Translation: [-2.11, 2.74]

Window 2:
  Scale: [-2.47, 2.39]
  Rotation: [-2.37, 2.84] rad
  Translation: [-2.82, 2.94]
```

**Visual symptoms:**
- Negative scale values causing face mirroring
- Extreme rotations (±157°) causing upside-down faces
- Large translations moving face out of frame
- Severe warping and vertical compression

## Solution Implemented

Added SRT clamping in `vasa_model.py` (lines 1295-1324) after DDIM sampling:

```python
# Clamp to prevent extreme distortions
window_motion['scale'] = torch.clamp(window_motion['scale'], 0.7, 1.3)  # ±30% scale
window_motion['rotation'] = torch.clamp(window_motion['rotation'], -0.785, 0.785)  # ±45°
window_motion['translation'] = torch.clamp(window_motion['translation'], -0.3, 0.3)

# Recompose theta from clamped SRT for geometric consistency
theta_4x4 = get_transform_matrix(scale_flat, rotation_flat, translation_flat)
window_motion['theta'] = theta_4x4[:, :3, :].view(B_win, T_win, 3, 4)
```

### After Clamping:
```
Window 0:
  Scale: [0.70, 1.30]       ✅ Within safe range
  Rotation: [-0.79, 0.79]   ✅ Within ±45°
  Translation: [-0.30, 0.30] ✅ Within safe range

Window 1:
  Scale: [0.70, 1.30]
  Rotation: [-0.79, 0.79]
  Translation: [-0.30, 0.30]

Window 2:
  Scale: [0.70, 1.30]
  Rotation: [-0.79, 0.79]
  Translation: [-0.30, 0.30]
```

## Results

### Visual Quality Improvement:
✅ **Frame 0**: Still shows some warping (likely first-frame initialization issue)
✅ **Frames 1-125**: Significantly improved geometric stability
- No extreme rotations or mirroring
- Face maintains proper proportions
- Head pose stays within natural range
- No out-of-frame translations

### Diagnostic Comparison:

**Original Issue** (from user screenshot):
- Face severely compressed vertically
- Features distorted and warped
- Green background bleeding through

**After Clamping** (test_clamped_output.mp4):
- Face geometry mostly correct
- Proper head alignment
- Natural facial proportions maintained

## Limitations

This is a **band-aid fix** that addresses symptoms, not root causes:

1. **Model still predicts extreme values** - clamping just truncates them
2. **First frame issues** - frame 0 still shows some distortion
3. **Expression instability** - expressions still "all over the place" as user noted
4. **Generalization concerns** - model struggles with videos not in training set

## Root Cause Analysis

The model was **NOT trained with SRT prediction separately**:
- Currently only predicts theta directly
- Scale/rotation/translation in output are pass-through values
- No supervision on SRT components during training
- Diffusion learns to predict raw transformation matrices without geometric constraints

## Recommended Next Steps

### Short-term (Implemented ✅):
1. **Clamping** - Prevents catastrophic geometric failures
2. **Temporal smoothing** - Already applied in vi.py for pose stability

### Medium-term:
3. **Expression clamping/regularization** - Add constraints to keep expressions closer to database
4. **Better first-frame handling** - Use GT theta for frame 0 as anchor

### Long-term (Requires retraining):
5. **Train with `predict_srt_separately: true`**:
   - Add separate prediction heads for S, R, T
   - Add SRT reconstruction loss
   - Add geometric constraint losses (scale > 0, reasonable rotation ranges)
   - Compose theta from predicted SRT during training

6. **Add regularization losses**:
   - Scale regularization (push toward 1.0)
   - Rotation smoothness (penalize large angular changes)
   - Translation regularization (keep face centered)

## Testing Commands

```bash
# Run diagnostic tool
python diagnose_theta.py \
    --config overfit_config.yaml \
    --checkpoint checkpoints_overfit/best_checkpoint.pt \
    --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4

# Test inference with clamping
python vi.py \
    --input junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
    --target_image ./data/IMG_1.png \
    --output test_clamped_output.mp4
```

## Files Modified

1. **vasa_model.py:1295-1324** - Added SRT clamping and theta recomposition
2. **diagnose_theta.py** - Created diagnostic tool for theta/SRT analysis

## Conclusion

✅ **Immediate fix successful** - Clamping prevents extreme geometric distortions
⚠️ **Expression stability** - Still needs work (separate issue from geometry)
🔄 **Long-term solution** - Retrain with SRT prediction and geometric losses

The clamping provides a stable baseline for inference while we work on proper geometric learning through training.
