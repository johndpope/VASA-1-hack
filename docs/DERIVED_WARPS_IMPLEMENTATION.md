# Derived Warps Implementation

## Overview
Replaced the simple projection layer approach with a more sophisticated method that leverages volumetric_avatar's pretrained embedding pipeline components with identity conditioning.

## Architecture

### Previous Approach (Projection Layer)
```
zdyn + zpose → Linear(140 → 512) → nemo WarpGenerator → warps
```
**Problem:** Simple projection doesn't leverage identity conditioning or pretrained embedding transformations.

### New Approach (Derived Warps via predict_embed pipeline)
```
zdyn + idt_embed → pose_unsqueeze_nw → warp_embed_head_orig_nw → target_warp_embed_dict → uv_generator_nw → warps
```
**Benefits:** Uses pretrained nemo components with identity conditioning for better warp generation.

## Implementation

### 1. compute_warps_from_zdyn() Method (vasa_model.py:816-883)

```python
def compute_warps_from_zdyn(self, zdyn, idt_embed, theta=None):
    """
    Compute UV warps from zdyn using volumetric_avatar's embedding pipeline.

    Args:
        zdyn: [B, T, zdyn_dim] - expression dynamics from motion transformer
        idt_embed: [B, idt_dim] - identity embedding from source image
        theta: [B, T, 3, 4] - pose matrices (optional)

    Returns:
        uv_warps: [B, T, 16, 64, 64, 3] - volumetric UV warp field
    """
    # 1. Flatten zdyn: [B, T, 128] → [B*T, 128]
    # 2. Repeat identity: [B, idt_dim] → [B*T, idt_dim]
    # 3. Expand zdyn spatially via pose_unsqueeze_nw
    # 4. Combine zdyn + identity via warp_embed_head_orig_nw
    # 5. Create target_warp_embed_dict
    # 6. Generate warps via uv_generator_nw
    # 7. Reshape to [B, T, 16, 64, 64, 3]
```

### 2. Pipeline Components Used

#### pose_unsqueeze_nw
- Expands zdyn from [B*T, 128] to [B*T, C, embed_size, embed_size]
- Adds spatial dimensions needed for convolution layers

#### warp_embed_head_orig_nw
- Combines expression (zdyn) + identity (idt_embed)
- Two modes:
  - `cat_em=True`: Concatenates embeddings
  - `cat_em=False`: Averages embeddings (0.5 weight each)

#### uv_generator_nw
- Nemo's pretrained WarpGenerator
- Converts warp_embed_dict → volumetric UV warps
- Uses ProjectorNorm for adaptive normalization

### 3. Integration Points

#### Forward Method (vasa_model.py:913-1047)
```python
def forward(self, motion_data, noise_level, conditions=None,
            cond_emb=None, prev_context=None, noise=None, idt_embed=None):
    # ...motion transformer generates zdyn/theta...

    if idt_embed is not None and self.use_derived_warps:
        warps = self.compute_warps_from_zdyn(
            zdyn=outputs['expression_embed'],
            idt_embed=idt_embed,
            theta=outputs['theta']
        )
        outputs['uv_warps'] = warps
```

#### Generate Sequence Method (vasa_model.py:1049-1228)
```python
def generate_sequence(self, initial_pose, initial_dynamics, conditions,
                     num_steps=50, eta=0.8, cfg_scales=None, idt_embed=None):
    # ...DDIM sampling loop...

    if idt_embed is not None and self.use_derived_warps:
        window_motion['uv_warps'] = self.compute_warps_from_zdyn(
            zdyn=window_motion['expression_embed'],
            idt_embed=idt_embed,
            theta=window_motion['theta']
        )
```

## Configuration

### Enable/Disable (vasa_config.yaml)
```yaml
model:
  use_derived_warps: true  # Set to false to disable
```

### Default Behavior
- **use_derived_warps=true**: Use predict_embed pipeline with identity conditioning
- **use_derived_warps=false**: Raises error (old projection layer removed)

## Training Configuration

### Trainable Parameters (45.8M)
- **Motion Transformer**: 45,817,712 params
- **Start Prev Params**: 1,490 params

### Frozen Parameters (160.7M)
- **Entire Volumetric Avatar**: 160,704,361 params
  - pose_unsqueeze_nw ❄️
  - warp_embed_head_orig_nw ❄️
  - uv_generator_nw ❄️
  - All other nemo components ❄️

### Why This Works
1. Motion transformer learns audio → zdyn/zpose mapping
2. Pretrained nemo components convert zdyn+identity → warps
3. Identity conditioning ensures person-specific warp generation
4. No need to retrain expensive warp generators

## Advantages Over Projection Layer

### 1. Identity Conditioning
- **Projection**: No identity awareness
- **Derived**: Uses idt_embed for person-specific warps

### 2. Pretrained Knowledge
- **Projection**: Random initialization, needs learning
- **Derived**: Leverages pretrained warp_embed_head_orig_nw

### 3. Spatial Processing
- **Projection**: Simple linear mapping
- **Derived**: Uses pose_unsqueeze_nw for proper spatial expansion

### 4. Adaptive Normalization
- **Projection**: No adaptive parameters
- **Derived**: Uses ada_v for expression-dependent normalization

## Data Flow

```
Training:
  Audio → Motion Transformer → zdyn [B, T, 128]
  Source Image → idt_embedder_nw → idt_embed [B, 512]

  compute_warps_from_zdyn(zdyn, idt_embed):
    1. zdyn [B, T, 128] → flatten → [B*T, 128]
    2. idt_embed [B, 512] → repeat → [B*T, 512]
    3. pose_unsqueeze_nw(zdyn) → [B*T, C, H, W]
    4. warp_embed_head_orig_nw(zdyn + idt) → [B*T, C, H, W]
    5. reshape → target_warp_embed_dict['orig'] [B*T, C, H²]
    6. uv_generator_nw(dict) → warps [B*T, 16, 64, 64, 3]
    7. reshape → [B, T, 16, 64, 64, 3]
```

## Testing

### Test Files
- `test_derived_warps.py` - Test warp generation pipeline ✅
- `test_nemo_warp_integration.py` - Verify integration ✅

### Verification
```python
# Test shapes
zdyn: [1, 4, 128]
idt_embed: [1, 512]
→ warps: [1, 4, 16, 64, 64, 3] ✅

# Test identity conditioning
different idt_embed → different warps ✅
```

## Usage in Trainer

The trainer must pass `idt_embed` to both `forward()` and `generate_sequence()`:

```python
# In vasa_trainer.py
outputs = model(
    motion_data=motion_data,
    noise_level=timestep,
    conditions=conditions,
    idt_embed=batch['idt_embed']  # ADD THIS
)

# For inference
generated = model.generate_sequence(
    initial_pose=initial_pose,
    initial_dynamics=initial_dynamics,
    conditions=conditions,
    idt_embed=batch['idt_embed']  # ADD THIS
)
```

## Key Differences from Original Pseudocode

### Original Suggestion
Used `volumetric_avatar.predict_embed(data_dict)` which requires source_img and target_img.

### Our Implementation
Bypasses predict_embed and uses the pipeline components directly:
- pose_unsqueeze_nw
- warp_embed_head_orig_nw
- uv_generator_nw

This avoids needing source/target images while still leveraging pretrained components.

## Summary

✅ **Removed**: Simple 140→512 projection layer
✅ **Added**: Identity-conditioned warp generation using pretrained nemo pipeline
✅ **Benefits**: Better warps, person-specific, leverages pretrained knowledge
✅ **Trainable**: Only motion transformer (45.8M params)
✅ **Frozen**: Entire volumetric_avatar (160.7M params)
