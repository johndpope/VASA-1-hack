# Derived Warps Implementation - Final Summary

## Overview
Successfully implemented identity-conditioned derived warps in the trainer, computing `idt_embed` on-the-fly from the identity image.

## Key Implementation Changes

### 1. Trainer: idt_embed Computation (vasa_trainer.py:1233-1266)

```python
# Compute identity embedding from identity image for derived warps
idt_embed = None
if self.use_derived_warps:
    # Get source image for identity embedding
    if self.identity_image is not None:
        identity_frame = self.identity_image.to(self.accelerator.device)  # [1, C, H, W]
    else:
        # Use first frame from window
        target_frames_temp = window.get('frames', None)
        if target_frames_temp is not None:
            identity_frame = target_frames_temp[0:1, 0]  # [1, C, H, W]
        else:
            logger.error("[IDT_EMBED] No identity image or frames available!")
            identity_frame = None

    if identity_frame is not None:
        with torch.no_grad():
            # Get face mask
            identity_mask, _, _, _ = self.model.volumetric_avatar.face_idt.forward(identity_frame)
            identity_mask = (identity_mask > 0.6).float()
            identity_mask = torch.nn.functional.avg_pool2d(identity_mask, 3, stride=1, padding=1)

            # Mask source image
            masked_identity = identity_frame * identity_mask

            # Extract identity embedding (returns 4D spatial feature map)
            idt_embed_spatial = self.model.volumetric_avatar.idt_embedder_nw(masked_identity)  # [B, C, H, W]

            # CRITICAL FIX: Convert spatial feature map to feature vector
            # idt_embedder_nw returns [1, 512, 4, 4] but compute_warps_from_zdyn expects [1, 512]
            idt_embed = torch.nn.functional.adaptive_avg_pool2d(
                idt_embed_spatial, (1, 1)
            ).squeeze(-1).squeeze(-1)  # [1, 512]
            logger.debug(f"[IDT_EMBED] ✅ Computed idt_embed: {idt_embed.shape}")
    else:
        logger.error("[IDT_EMBED] ❌ Could not get identity frame!")

# Forward pass with idt_embed
outputs = self.model(
    motion_data=noised_motion,
    noise_level=t,
    conditions=control_signals,
    noise=noise,
    idt_embed=idt_embed  # Pass identity embedding
)
```

### 2. Configuration Fix (vasa_trainer.py:753)

**Before**: `self.use_derived_warps = getattr(config.motion, 'use_derived_warps', False)`
**After**: `self.use_derived_warps = getattr(config.model, 'use_derived_warps', False)`

The flag was under `model:` in config but trainer was reading from `config.motion`.

### 3. Files Reverted

Removed unnecessary dataset changes:
- `vasa_dataset.py` - Removed identity_info handling (not needed)
- `vasa_sampler.py` - Removed identity_info batching (not needed)

### 4. Enhanced Logging (vasa_model.py:1035-1043)

Changed from `logger.debug` to `logger.info` for visibility:
```python
logger.info(f"[DERIVED WARPS] Generating UV warps from zdyn {outputs['expression_embed'].shape} + idt_embed {idt_embed.shape}")
...
logger.info(f"[DERIVED WARPS] ✅ Generated warps shape: {implicit_warps.shape}")
```

## Critical Fix: Shape Conversion

### Problem
- `idt_embedder_nw` returns 4D spatial feature map: `[1, 512, 4, 4]`
- `compute_warps_from_zdyn` expects 2D feature vector: `[1, 512]`
- Error: `RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`

### Solution
Apply global average pooling to convert spatial features to vector:

```python
# [1, 512, 4, 4] → [1, 512, 1, 1] → [1, 512]
idt_embed = torch.nn.functional.adaptive_avg_pool2d(idt_embed_spatial, (1, 1)).squeeze(-1).squeeze(-1)
```

This preserves channel-wise features while converting to the expected shape.

## Configuration

### overfit_config.yaml
```yaml
model:
  use_derived_warps: true    # Enable identity-conditioned warp generation

dataset:
  use_identity_image: true
  identity_image_path: "nemo/data/IMG_1.png"
```

## How It Works

1. **Initialization**:
   - Trainer loads identity image from config
   - Reads `use_derived_warps` from `config.model`

2. **Per-Forward Pass**:
   - Extract identity frame (from identity_image or video)
   - Compute face mask using `volumetric_avatar.face_idt`
   - Mask the identity image
   - Extract spatial features: `idt_embedder_nw(masked_identity)` → `[1, 512, 4, 4]`
   - **Pool to vector**: `adaptive_avg_pool2d` → `[1, 512]`
   - Pass to `model.forward(idt_embed=idt_embed)`

3. **Warp Generation**:
   - Model uses `compute_warps_from_zdyn(zdyn, idt_embed, theta)`
   - Combines predicted motion + identity + pose
   - Uses pretrained nemo components:
     - `pose_unsqueeze_nw` - Expands zdyn to spatial
     - `warp_embed_head_orig_nw` - Combines with identity
     - `uv_generator_nw` - Generates UV warps
   - Returns: `[B, T, 16, 64, 64, 3]` volumetric UV warp field

## Benefits

✅ **No Dataset Dependency**: Identity computed in trainer, not cached
✅ **Uses Identity Image**: Leverages high-quality reference image
✅ **Person-Specific Warps**: Same identity for all predictions
✅ **Frozen Pretrained Components**: No retraining of warp generators
✅ **Clean Separation**: Dataset handles frames, trainer handles identity

## Training Configuration

- **Trainable**: motion_transformer (45.8M params)
- **Frozen**: volumetric_avatar (160.7M params) - including all warp pipeline

Training focuses on learning audio→motion mapping while leveraging pretrained person-specific warp generation.

## Notes

- CUDA OOM may occur with derived warps (more memory than pre-computed)
- Can be managed by reducing batch size or using gradient checkpointing
- The shape conversion fix is critical for compatibility with nemo's warp pipeline
