# Derived Warps - Trainer Implementation (2025-10-06)

## Summary
Successfully implemented identity-conditioned derived warps generation in the trainer, computing `idt_embed` on-the-fly from the identity image instead of relying on dataset caching.

## Implementation

### Key Changes

#### 1. Trainer Computes idt_embed (vasa_trainer.py:1233-1262)
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

            # Extract identity embedding
            idt_embed = self.model.volumetric_avatar.idt_embedder_nw(masked_identity)
            logger.debug(f"[IDT_EMBED] ✅ Computed idt_embed from identity image: {idt_embed.shape}")
    else:
        logger.error("[IDT_EMBED] ❌ Could not get identity frame!")

# Forward pass with CFG during inference
outputs = self.model(
    motion_data=noised_motion,
    noise_level=t,
    conditions=control_signals,
    noise=noise,
    idt_embed=idt_embed  # Pass identity for derived warps
)
```

#### 2. Trainer Initialization (vasa_trainer.py:752-757)
```python
# Get use_derived_warps flag from model config
self.use_derived_warps = getattr(config.model, 'use_derived_warps', False)
if self.use_derived_warps:
    logger.info("✅ Using derived warps: will compute identity-conditioned warps on-the-fly")
else:
    logger.info("Using pre-computed warps from dataset")
```

#### 3. Model Forward Method (vasa_model.py:1031-1048)
Already implemented to use idt_embed when available:
```python
if self.use_derived_warps:
    if 'expression_embed' in outputs and 'theta' in outputs:
        if idt_embed is not None:
            logger.info(f"[DERIVED WARPS] Generating UV warps from zdyn {outputs['expression_embed'].shape} + idt_embed {idt_embed.shape}")
            implicit_warps = self.compute_warps_from_zdyn(
                zdyn=outputs['expression_embed'],
                idt_embed=idt_embed,
                theta=outputs['theta']
            )
            outputs['uv_warps'] = implicit_warps
            outputs['warp_source'] = 'derived'
            logger.info(f"[DERIVED WARPS] ✅ Generated warps shape: {implicit_warps.shape}")
```

### Configuration

#### overfit_config.yaml
```yaml
model:
  use_derived_warps: true    # Use identity-conditioned warp generation via volumetric_avatar pipeline

dataset:
  use_identity_image: true
  identity_image_path: "nemo/data/IMG_1.png"
```

### Files Reverted
- `vasa_dataset.py` - Removed identity_info handling (not needed for trainer approach)
- `vasa_sampler.py` - Removed identity_info batching (not needed for trainer approach)

### Files Modified
- `vasa_trainer.py` - Added idt_embed computation from identity image
- `vasa_model.py` - Enhanced logging for derived warps generation

## How It Works

1. **Training Setup**:
   - Trainer loads high-quality identity image from config
   - Reads `use_derived_warps` flag from `config.model`

2. **Per-Batch Processing**:
   - Extract identity frame (from identity_image or first video frame)
   - Compute face mask using volumetric_avatar.face_idt
   - Mask the identity image
   - Extract idt_embed using volumetric_avatar.idt_embedder_nw
   - Pass idt_embed to model.forward()

3. **Model Forward**:
   - If use_derived_warps=True and idt_embed is provided:
     - Generate UV warps using compute_warps_from_zdyn()
     - Combines zdyn (predicted motion) + idt_embed (identity) + theta (pose)
     - Uses pretrained nemo components (pose_unsqueeze_nw, warp_embed_head_orig_nw, uv_generator_nw)

## Benefits

1. **No Dataset Changes**: Identity embedding computed on-the-fly in trainer
2. **Uses Identity Image**: Leverages high-quality identity image from config
3. **No Cache Issues**: Doesn't rely on H5 cache storing identity_info
4. **Clean Separation**: Dataset handles video frames, trainer handles identity
5. **Person-Specific Warps**: Each forward pass uses same identity for consistency

## Verification

Training logs show successful operation:
```
[12:00:08] INFO     ✅ Using derived warps: will compute identity-conditioned warps on-the-fly
           INFO     [IDT_EMBED] ✅ Computed idt_embed from identity image: torch.Size([1, 512, 4, 4])
           INFO     [DERIVED WARPS] Generating UV warps from zdyn torch.Size([1, 50, 128]) + idt_embed torch.Size([1, 512, 4, 4])
           INFO     [DERIVED WARPS] ✅ Generated warps shape: torch.Size([1, 50, 16, 64, 64, 3])
```

## Training Configuration

**Trainable**: motion_transformer (45.8M params)
**Frozen**: volumetric_avatar (160.7M params) - including all warp pipeline components

This approach leverages pretrained warp generators without retraining, focusing training on audio→motion learning.

## Critical Fix: idt_embed Shape Conversion

The `idt_embedder_nw` returns a 4D spatial feature map `[B, 512, 4, 4]`, but `compute_warps_from_zdyn` expects a 2D feature vector `[B, 512]`.

**Solution** (vasa_trainer.py:1258-1264):
```python
# Extract identity embedding (returns 4D spatial feature map)
idt_embed_spatial = self.model.volumetric_avatar.idt_embedder_nw(masked_identity)  # [B, C, H, W]

# Convert spatial feature map to feature vector via global average pooling
# This matches what nemo expects: [B, idt_dim]
idt_embed = torch.nn.functional.adaptive_avg_pool2d(idt_embed_spatial, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]
```

This converts `[1, 512, 4, 4]` → `[1, 512, 1, 1]` → `[1, 512]` using global average pooling to preserve channel-wise features.
