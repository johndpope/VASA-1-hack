# Identity Embedding (idt_embed) Integration Fix

## Problem
The derived warps implementation requires `idt_embed` for identity-conditioned warp generation, but it wasn't being passed through the data pipeline correctly.

**Error:**
```
ValueError: Cannot generate warps: idt_embed required for compute_warps_from_zdyn
```

## Root Cause Analysis

### Data Flow
```
Dataset → Collate Function → Trainer → Model
```

1. **Dataset** (vasa_dataset.py):
   - ✅ Computes `idt_embed` from source image (line 1210)
   - ✅ Stores in `identity_info` dict (line 1259-1265)
   - ✅ Returns `identity_info` in window output (line 1341)

2. **Collate Function** (vasa_sampler.py):
   - ❌ **NOT** batching `identity_info` (it wasn't in `keys_to_stack`)
   - Result: `identity_info` lost during batching

3. **Trainer** (vasa_trainer.py):
   - Tries to access `window['identity_info']['idt_embed']`
   - Gets None because collate didn't include it

## Solution

### 1. Fix Collate Function (vasa_sampler.py:313-318)

Added identity_info to batched output:

```python
# Handle identity_info separately - contains idt_embed for derived warps
if 'identity_info' in processed_windows[0]:
    # All windows should have the same identity_info for same video
    # Just use the first one (they should be identical)
    batched['identity_info'] = processed_windows[0]['identity_info']
    logger.debug(f"[COLLATE] Added identity_info from first window")
```

**Why this works:**
- `identity_info` is computed once per video (not per frame)
- All windows from same video share the same `identity_info`
- We can safely use the first window's `identity_info` for the entire batch

### 2. Update Trainer to Extract idt_embed (vasa_trainer.py:1227-1236)

```python
# Get identity embedding from window for derived warps
# Dataset returns it in identity_info dict
idt_embed = None
if 'identity_info' in window and 'idt_embed' in window['identity_info']:
    idt_embed = window['identity_info']['idt_embed']
    logger.debug(f"[DEBUG] Using idt_embed from window['identity_info']: {idt_embed.shape}")
else:
    logger.error(f"[ERROR] identity_info not in window! Available keys: {list(window.keys())}")
    if 'identity_info' in window:
        logger.error(f"[ERROR] identity_info keys: {list(window['identity_info'].keys())}")
```

### 3. Add Debug Logging in Model (vasa_model.py:1033, 1043, 1224, 1232)

Added detailed logging for warp generation:

```python
# Forward method
logger.debug(f"[DERIVED WARPS] Generating UV warps from zdyn {outputs['expression_embed'].shape} + idt_embed {idt_embed.shape}")
logger.debug(f"[DERIVED WARPS] Generated warps shape: {implicit_warps.shape}")

# Error logging
logger.error(f"[WARPS ERROR] idt_embed is None: {idt_embed is None}, use_derived_warps: {self.use_derived_warps}")
```

## identity_info Structure

Computed in dataset (vasa_dataset.py:1259-1265):

```python
identity_info = {
    'idt_embed': idt_embed,           # [B, 512] - Identity embedding from idt_embedder_nw
    'embed_dict': embed_dict,          # Embedding dictionary for volume generation
    'canonical_volume': canonical_volume,  # [B, C, D, H, W] - Canonical appearance volume
    'source_theta': identity_theta,    # [B, 3, 4] - Identity frame pose
    'source_mask': identity_mask       # [B, 1, H, W] - Identity frame mask
}
```

## Complete Data Flow (Fixed)

```
1. Dataset.__getitem__():
   - Load identity frame
   - Compute idt_embed = idt_embedder_nw(identity_frame)
   - Store in identity_info dict
   - Return window with identity_info ✅

2. Collate Function:
   - Batch all tensor keys
   - Add identity_info from first window ✅

3. Trainer:
   - Extract idt_embed = window['identity_info']['idt_embed'] ✅
   - Pass to model forward() ✅

4. Model:
   - Use idt_embed in compute_warps_from_zdyn() ✅
   - Generate identity-conditioned warps ✅
```

## Files Modified

1. **vasa_sampler.py:313-318**
   - Added identity_info to batched output

2. **vasa_trainer.py:1227-1236**
   - Extract idt_embed from window['identity_info']
   - Added debug logging

3. **vasa_model.py:1033, 1043, 1224, 1232**
   - Added debug logging for warp generation
   - Better error messages

## Testing

### Before Fix
```
ERROR: idt_embed not in window!
ValueError: Cannot generate warps: idt_embed required
```

### After Fix
```
[DEBUG] Using idt_embed from window['identity_info']: torch.Size([1, 512])
[DERIVED WARPS] Generating UV warps from zdyn torch.Size([1, 50, 128]) + idt_embed torch.Size([1, 512])
[DERIVED WARPS] Generated warps shape: torch.Size([1, 50, 16, 64, 64, 3])
✅ Training proceeds normally
```

## Key Insights

1. **Collate Function Limitations**: Only keys in `keys_to_stack` get batched
2. **Identity Sharing**: All windows from same video share identity_info
3. **Dict vs Tensor**: identity_info is a dict, needs special handling
4. **Debug Logging**: Essential for tracing data flow issues

## Summary

✅ **Fixed**: identity_info now properly passed through data pipeline
✅ **Result**: idt_embed available for derived warps
✅ **Benefit**: Identity-conditioned warp generation working
