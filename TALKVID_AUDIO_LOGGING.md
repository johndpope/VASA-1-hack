# TalkVid Audio Projection Logging

## Summary

Added comprehensive logging to the TalkVid audio projection system to provide clear visibility into:
1. Which audio projection architecture is being used (TalkVid vs JoyVASA)
2. Configuration source (main config vs channel_config.yaml)
3. Architecture parameters and initialization
4. Runtime behavior during forward passes

## Logging Additions

### 1. EfficientConditionEmbedding Configuration Logging

**Location:** `vasa_model.py:330-375`

**When:** During model initialization

**Output Example (TalkVid enabled):**
```
================================================================================
🎵 AUDIO PROJECTION CONFIGURATION
================================================================================
📁 Source: main config (overfit_config.yaml / vasa_config.yaml)
🔧 use_talkvid_audio_projection: True

📊 Selected Architecture: TalkVid (Perceiver-based)
────────────────────────────────────────────────────────────────────────────────
🎵 Initializing TalkVidAudioProjection (Perceiver architecture)
   Input: 768D → Output: 512D
   Architecture: 4 layers × 8 heads (dim_head=64)
   Latent queries: 8, Internal dim: 1024
   Feed-forward multiplier: 4x
   Max sequence length: 60
────────────────────────────────────────────────────────────────────────────────
✅ TalkVid audio projection initialized successfully
================================================================================
```

**Output Example (JoyVASA - default):**
```
================================================================================
🎵 AUDIO PROJECTION CONFIGURATION
================================================================================
📁 Source: main config (overfit_config.yaml / vasa_config.yaml)
🔧 use_talkvid_audio_projection: False

📊 Selected Architecture: JoyVASA (Linear projection)
────────────────────────────────────────────────────────────────────────────────
   Single Linear layer: 768D → 512D
   No normalization (preserves audio variance signal)
   Parameters: ~393.2K
────────────────────────────────────────────────────────────────────────────────
✅ JoyVASA audio projection initialized successfully
================================================================================
```

### 2. TalkVidAudioProjection Initialization Logging

**Location:** `vasa_model.py:137-142`

**When:** During TalkVidAudioProjection instantiation

**What it logs:**
- Input dimension (768D from wav2vec2)
- Output dimension (512D for model)
- Number of transformer layers (depth)
- Number of attention heads
- Dimension per head
- Number of latent queries
- Feed-forward multiplier
- Maximum sequence length

**Purpose:** Provides detailed architecture parameters for debugging and parameter counting

### 3. TalkVidAudioProjection Forward Pass Logging

**Location:** `vasa_model.py:172-176`

**When:** First forward pass only (logged once to avoid spam)

**Output Example:**
```
🎵 TalkVidAudioProjection forward pass:
   Input shape: torch.Size([4, 50, 768]) (batch_size, seq_len, 768)
   Processing through 4 Perceiver layers with 8 latent queries
```

**Purpose:** Confirms runtime behavior and tensor shapes

### 4. TalkVidAudioProjection Output Logging

**Location:** `vasa_model.py:209-213`

**When:** First forward pass only (logged once)

**Output Example:**
```
🎵 TalkVidAudioProjection output:
   Output shape: torch.Size([4, 8, 512]) (batch_size, 8 queries, 512D)
   ✅ Audio features projected and compressed via Perceiver attention
```

**Purpose:** Confirms output tensor shape and successful processing

## Configuration Flow

### Priority Order

1. **Main config** (`overfit_config.yaml` or `vasa_config.yaml`)
   - `model.use_talkvid_audio_projection: true/false`
   - **This takes priority**

2. **Channel config** (`channel_config.yaml`)
   - `projections.use_talkvid_audio_projection: true/false`
   - **Fallback only** if not set in main config

### How to Tell Which Source Is Used

Look for the log line:
```
📁 Source: main config (overfit_config.yaml / vasa_config.yaml)
```
or
```
📁 Source: channel_config.yaml (fallback)
```

## Architecture Comparison

### JoyVASA (Default)

**Logging signature:**
```
📊 Selected Architecture: JoyVASA (Linear projection)
```

**Key characteristics:**
- Simple linear layer: `nn.Linear(768, 512)`
- No normalization
- Preserves audio variance signal
- Parameters: ~393K
- Faster, lower memory

### TalkVid (Optional)

**Logging signature:**
```
📊 Selected Architecture: TalkVid (Perceiver-based)
🎵 Initializing TalkVidAudioProjection (Perceiver architecture)
```

**Key characteristics:**
- 4 transformer layers with 8 attention heads
- 8 learnable latent queries
- Internal dimension: 1024
- Parameters: ~5-10M
- Better audio-visual alignment, but slower

## Debugging Tips

### Verify Config Is Loaded

**Check for:**
```
🔧 use_talkvid_audio_projection: True
```

If you see `False` but expected `True`, check:
1. Config file syntax (YAML indentation)
2. Config file location
3. Whether the config is actually being loaded

### Verify Architecture Initialization

**TalkVid should show:**
```
🎵 Initializing TalkVidAudioProjection (Perceiver architecture)
```

**JoyVASA should show:**
```
Single Linear layer: 768D → 512D
```

### Verify Runtime Behavior

**First batch should trigger:**
```
🎵 TalkVidAudioProjection forward pass:
   Input shape: torch.Size([...])
```

If you don't see this, the architecture might not be receiving audio features correctly.

### Check Output Shapes

**TalkVid output:**
```
Output shape: torch.Size([batch, 8, 512])
```
- 8 = num_queries (latent compression)
- 512 = output_dim

**JoyVASA output:**
```
Output shape: torch.Size([batch, seq_len, 512])
```
- seq_len = same as input (no compression)
- 512 = output_dim

## Common Issues

### Issue: Config set to `true` but logs show `false`

**Symptoms:**
```
🔧 use_talkvid_audio_projection: False
📊 Selected Architecture: JoyVASA (Linear projection)
```

**Causes:**
1. YAML syntax error (indentation, missing colon)
2. Wrong config file being loaded
3. Config not propagating to model

**Fix:**
```yaml
# Correct YAML in overfit_config.yaml
model:
  use_talkvid_audio_projection: true  # Must be under 'model:' section
```

### Issue: No TalkVid forward logs appearing

**Symptoms:**
- Initialization logs show TalkVid
- But no forward pass logs during training

**Causes:**
1. Audio features not reaching the projection
2. Training crashed before first batch
3. Logs filtered out

**Fix:**
- Check earlier logs for errors
- Verify dataset is loading audio features
- Check log level (should be INFO)

### Issue: Shape mismatch errors

**Symptoms:**
```
RuntimeError: Expected tensor for argument #1 'other' to have the same shape as tensor for argument #2 'self'
```

**Causes:**
- TalkVid outputs `[batch, 8, 512]` (compressed queries)
- JoyVASA outputs `[batch, seq_len, 512]` (full sequence)
- Downstream code expecting different shape

**Fix:**
- Ensure model code handles both output shapes
- Or stick with one architecture

## Parameter Counting

### JoyVASA
```
Parameters = 768 × 512 = 393,216 (~393K)
Memory: Negligible
```

### TalkVid
```
Embedding: max_seq_len × 768
Latents: 1 × 8 × 1024
Perceiver layers: 4 × (MultiheadAttention + FFN)
  - Attention: 8 heads × (1024 → 64) per head
  - FFN: 1024 → 4096 → 1024

Total: ~5-10M parameters
Memory: +10-15% VRAM
```

## Usage Examples

### Enable TalkVid

```yaml
# In overfit_config.yaml or vasa_config.yaml
model:
  use_talkvid_audio_projection: true
```

**Expected logs:**
```
🎵 AUDIO PROJECTION CONFIGURATION
📁 Source: main config
🔧 use_talkvid_audio_projection: True
📊 Selected Architecture: TalkVid (Perceiver-based)
🎵 Initializing TalkVidAudioProjection
   Architecture: 4 layers × 8 heads
✅ TalkVid audio projection initialized successfully
```

### Disable TalkVid (Use JoyVASA)

```yaml
# In overfit_config.yaml or vasa_config.yaml
model:
  use_talkvid_audio_projection: false  # or omit this line
```

**Expected logs:**
```
🎵 AUDIO PROJECTION CONFIGURATION
📁 Source: main config
🔧 use_talkvid_audio_projection: False
📊 Selected Architecture: JoyVASA (Linear projection)
   Parameters: ~393.2K
✅ JoyVASA audio projection initialized successfully
```

## Files Modified

1. **vasa_model.py**
   - Line 137-142: TalkVidAudioProjection init logging
   - Line 172-176: Forward pass input logging
   - Line 209-213: Forward pass output logging
   - Line 330-375: EfficientConditionEmbedding audio projection selection logging

## Related Documentation

- `AUDIO_PROJECTION_CONFIG_WIRING.md` - Config system setup
- `channel_config.yaml` - Fallback configuration
- `overfit_config.yaml` - Main overfitting config
- `vasa_config.yaml` - Main training config

## Testing

To verify logging works:

```bash
# 1. Enable TalkVid
vim overfit_config.yaml
# Set: use_talkvid_audio_projection: true

# 2. Start training
./train.sh

# 3. Look for logs:
grep "🎵" train.log
grep "AUDIO PROJECTION" train.log
grep "TalkVid" train.log

# Should see initialization logs showing TalkVid architecture
```

---

**Date:** 2025-10-20
**Status:** ✅ Complete
**Tested:** Logging added, ready for runtime verification
