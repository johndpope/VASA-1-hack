# VASA Model Documentation (Updated - Decoder Architecture)

## Overview
VASA model implementation for holistic facial dynamics generation using diffusion transformer **decoder** architecture with cross-attention conditioning, classifier-free guidance, and temporal context handling with negative position encoding.

## Module Structure

### 1. VASAPositionalEmbedding
**Purpose**: Positional embeddings using **negative positions for context frames** to maintain clear temporal distinction between past context and current generation.

```python
__init__(
    d_model: int = 512,
    max_seq_len: int = 50,
    max_context_len: int = 10,
    dropout: float = 0.1,
    use_relative_position: bool = True  # Not used in current implementation
)
```

**Forward Method**:
```python
forward(
    x: torch.Tensor,  # Shape: [B, T, D]
    has_context: bool = False
) -> torch.Tensor  # Shape: [B, T, D]
```

**Position Encoding Strategy**:
- **With Context** (T=60 total):
  - Context frames: 10 frames at positions [-10, -9, ..., -1]
  - Current frames: 50 frames at positions [0, 1, ..., 49]
  - Clear semantic: negative = past, positive = current
- **Without Context** (T=50):
  - All frames: positions [0, 1, ..., 49]

**Implementation**:
```python
if has_context and T > self.max_context_len:
    # Create position indices: [-context_len, ..., -1, 0, 1, ..., current_len-1]
    context_positions = torch.arange(-context_len, 0, device=device)
    current_positions = torch.arange(0, current_len, device=device)
    all_positions = torch.cat([context_positions, current_positions])
    pos_emb = self._compute_sinusoidal_embedding(all_positions, self.d_model)
```

**Shape Assertions**:
```python
assert x.shape == (B, T, d_model)  # Input
assert output.shape == (B, T, d_model)  # Output with positions added
```

### 2. EfficientConditionEmbedding
**Purpose**: Processes and embeds various conditioning signals for motion generation with robust None handling for dropout.

```python
__init__(
    model_dim: int = 512,
    max_seq_len: int = 60
)
```

**Forward Method**:
```python
forward(
    conditions: Dict[str, torch.Tensor],
    prev_context: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor  # Shape: [B, T, model_dim]
```

**Expected Condition Tensor Shapes**:
```python
{
    'audio_features': [B, T, 768] or [B, 1, T, 768],  # REQUIRED - Wav2Vec2 features
    'gaze': [B, T, 2] or None,  # Can be None from dropout
    'head_distance': [B, T, 1] or None,
    'emotion': [B, T, 2] or None,
    'speed_bucket': [B, T, 1] or None,
    'lips': [B, T, 20, 3] or None,  # Currently commented out
    'right_eye': [B, T, 8, 3] or None,  # Currently commented out
    'left_eye': [B, T, 7, 3] or None,  # Currently commented out
    'jaw': [B, T, 10, 3] or None,  # Currently commented out
    'nose': [B, T, 4, 3] or None,  # Currently commented out
    'blink_state': [B, T, 3] or None
}
```

**Projections**:
- `audio_proj`: Linear(768) → ReLU → Linear(audio_output_dim)
- `gaze_proj`: Linear(2, 2)
- `distance_proj`: Linear(1, 1)
- `emotion_proj`: Linear(2, 2)
- `speed_proj`: Linear(1, 1)
- `blink_embed`: Linear(3) → ReLU → Linear(blink_output_dim)

**Critical Assertions**:
```python
# Audio features are mandatory
assert 'audio_features' in conditions
assert conditions['audio_features'] is not None
assert isinstance(conditions['audio_features'], torch.Tensor)

# At least one valid tensor must exist to determine batch size
assert any(tensor is not None and isinstance(tensor, torch.Tensor)
          for tensor in conditions.values())
```

**None Handling**:
- Automatically replaces None values with appropriate zero tensors
- Preserves device and dtype consistency
- Handles dropout-induced None values gracefully

### 3. MotionTransformer
**Purpose**: Diffusion Transformer using **decoder architecture with cross-attention** for superior conditioning.

```python
__init__(config)
```

**Configuration Requirements**:
```yaml
model:
  hidden_dim: 512
  n_layers: 8  # Decoder layers
  n_heads: 8
  dim_feedforward: 2048
  dropout: 0.1
  use_relative_position: true
  condition_embedding_dim: 512

motion:
  window_size: 50
  context_size: 10
```

**Architecture Components**:

1. **Motion Embeddings** (Hierarchical sizing):
   ```python
   self.theta_emb = nn.Linear(12, d_model // 2)  # 256D - primary motion
   self.expr_emb = nn.Linear(expression_dim, d_model // 2)  # 256D - primary
   self.scale_emb = nn.Linear(3, d_model // 4)  # 128D - auxiliary
   self.rotation_emb = nn.Linear(3, d_model // 4)  # 128D - auxiliary
   self.translation_emb = nn.Linear(3, d_model // 4)  # 128D - auxiliary
   self.motion_proj = nn.Linear(896, d_model)  # Combine all (256+256+128+128+128)
   ```

2. **Transformer Decoder**:
   ```python
   decoder_layer = nn.TransformerDecoderLayer(
       d_model=self.d_model,
       nhead=8,
       dim_feedforward=2048,
       dropout=0.1,
       activation=F.gelu,
       batch_first=True,
       norm_first=True  # Pre-LN for stability
   )
   self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
   ```

**Forward Method**:
```python
forward(
    motion_data: Dict[str, torch.Tensor],
    noise_level: torch.Tensor,  # Shape: [B]
    conditions: Optional[Dict[str, torch.Tensor]] = None,
    cond_emb: Optional[torch.Tensor] = None,  # Shape: [B, T, d_model]
    prev_context: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]
```

**Cross-Attention Mechanism**:
```python
# Decoder cross-attention: motion queries attend to condition memory
out = self.decoder(
    tgt=tgt,        # Query: motion embeddings [B, C+T, d_model]
    memory=cond_emb # Key/Value: condition embeddings [B, C+T, d_model]
)
```

**Key Advantages of Decoder Architecture**:
- **Per-frame conditioning**: Each frame's motion attends to audio/control conditions
- **Multi-layer attention**: Cross-attention happens at every decoder layer (8x)
- **Selective focus**: Different motion aspects can attend to relevant conditions
- **Better audio-motion coupling**: Direct attention from motion to audio features

**Internal Processing Flow**:
1. Embed motion parameters with hierarchical sizing
2. Add previous context if available
3. Apply negative position encoding for context frames
4. Add timestep embedding
5. Generate condition embeddings (memory)
6. **Process through decoder with cross-attention** (key difference)
7. Extract current frames if context was included
8. Project to output motion parameters

**Shape Assertions**:
```python
assert motion_data['theta'].shape == (B, T, 3, 4)
assert motion_data['expression_embed'].shape == (B, T, expression_dim)
assert noise_level.shape == (B,)
assert decoder_out.shape == (B, T, d_model)
```

### 4. VASAModel
**Purpose**: Main model combining diffusion process with volumetric avatar rendering.

```python
__init__(
    config,
    volumetric_avatar: nn.Module,
    device: str = 'cuda'
)
```

**Key Parameters**:
- `start_prev_theta`: Parameter[1, context_size, 3, 4] - Initial context pose
- `start_prev_expression`: Parameter[1, context_size, expression_dim] - Initial context expression
- `scheduler`: DDIMScheduler for diffusion process
- `dropout_probs`: Dictionary of dropout probabilities for CFG

**Main Forward Method**:
```python
forward(
    motion_data: Dict[str, torch.Tensor],
    noise_level: torch.Tensor,  # Shape: [B]
    conditions: Optional[Dict[str, torch.Tensor]] = None,
    cond_emb: Optional[torch.Tensor] = None,
    prev_context: Optional[Dict[str, torch.Tensor]] = None,
    noise: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]
```

**Condition Validation & Processing**:
1. **Landmark Remapping**: Automatically remaps keys:
   ```python
   'lips_landmarks' → 'lips'
   'right_eye_landmarks' → 'right_eye'
   'left_eye_landmarks' → 'left_eye'
   'jaw_landmarks' → 'jaw'
   'nose_landmarks' → 'nose'
   ```

2. **Shape Validation**: Ensures all conditions match expected shapes:
   ```python
   expected_shapes = {
       'gaze': (B, T, 2),
       'head_distance': (B, T, 1),
       'emotion': (B, T, 2),
       'speed_bucket': (B, T, 1),
       'lips': (B, T, 20, 3),
       'right_eye': (B, T, 8, 3),
       'left_eye': (B, T, 7, 3),
       'jaw': (B, T, 10, 3),
       'nose': (B, T, 4, 3),
       'blink_state': (B, T, 3)
   }
   ```

3. **Missing Condition Handling**: Creates zero tensors for missing conditions

4. **Blink State Generation**: Auto-generates if not provided using BlinkConditionHandler

5. **Classifier-Free Guidance Dropout**: Applies during training via `_apply_dropout`

**Previous Context Structure**:
```python
{
    'theta': Tensor[B, context_size, 3, 4],
    'expression': Tensor[B, context_size, expression_dim],
    'audio': Tensor[B, context_size, 768]
}
```

**NaN/Inf Handling**:
- Input sanitization: `torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)`
- Output clamping: `torch.clamp(tensor, min=-10.0, max=10.0)`

### 5. Key Methods

#### _apply_dropout (VASAModel)
```python
_apply_dropout(
    conditions: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]
```
- Applies probabilistic dropout for classifier-free guidance
- Replaces dropped conditions with zeros (not None)
- Uses config.train.dropout_probs

#### _add_noise_to_motion (VASAModel)
```python
_add_noise_to_motion(
    motion_data: Dict[str, torch.Tensor],
    noise: Dict[str, torch.Tensor],
    noise_level: torch.Tensor
) -> Dict[str, torch.Tensor]
```
- Uses DDIMScheduler.add_noise for 'theta' and 'expression_embed'
- Respects config.train.turn_off_noise flag
- Ensures scheduler device compatibility

#### generate_sequence (VASAModel)
```python
generate_sequence(
    initial_pose: Dict[str, torch.Tensor],
    initial_dynamics: torch.Tensor,  # Shape: [B, expression_dim]
    conditions: Dict[str, torch.Tensor],
    num_steps: int = 50,
    eta: float = 0.5,
    cfg_scales: Optional[Dict[str, float]] = None
) -> Dict[str, torch.Tensor]
```

**Sliding Window Processing**:
- Window size: config.motion.window_size
- Stride: config.motion.stride
- Context size: config.motion.context_size
- Maintains temporal consistency through prev_context

**DDIM Sampling Loop**:
1. Initialize with random noise
2. For each timestep:
   - Generate unconditional output (if CFG enabled)
   - Generate conditional output
   - Apply classifier-free guidance scaling
   - Update motion via scheduler.step
3. Slide window and update context

**Output Structure**:
```python
{
    'theta': Tensor[B, total_T, 3, 4],
    'expression_embed': Tensor[B, total_T, motion_dim]
}
```

## Configuration Dependencies

### Required YAML Configuration
```yaml
model:
  hidden_dim: 512
  n_layers: 6
  n_heads: 8
  dim_feedforward: 2048
  dropout: 0.1
  use_relative_position: true
  condition_embedding_dim: 512
 
  clip_bounds:
    min: -10.0
    max: 10.0

motion:
  window_size: 50
  stride: 25
  context_size: 10

diffusion:
  num_steps: 1000
  beta_start: 0.0001
  beta_end: 0.02

train:
  turn_off_noise: false  # Set to true for overfitting tests
  dropout_probs:
    audio: 0.0  # Never drop audio
    gaze: 0.1
    emotion: 0.1
    head_distance: 0.1
    speed_bucket: 0.1
    blink_state: 0.1
```

### Channel Configuration (channel_config.yaml)
```yaml
channel_layout:
  # Define channel allocations for embedding

landmarks:  # Currently commented out in code
  lips:
    points: 20
    coords: 3
  right_eye:
    points: 8
    coords: 3
  left_eye:
    points: 7
    coords: 3
  jaw:
    points: 10
    coords: 3
  nose:
    points: 4
    coords: 3

projections:
  audio:
    hidden_dim: 256
    output_dim: 128
  blink:
    input_dim: 3
    hidden_dim: 32
    output_dim: 16
  control_norm_dim: 6  # gaze(2) + distance(1) + emotion(2) + speed(1)

model:
  clip_bounds:
    min: -10.0
    max: 10.0
```

## Critical Implementation Notes

### 1. Audio Features Handling
- **MANDATORY**: audio_features must always be present in conditions
- **NEVER DROPPED**: Protected from dropout in trainer's _apply_condition_dropout
- **SHAPE FLEXIBILITY**: Accepts [B, T, 768] or [B, 1, T, 768]
- **WAV2VEC2**: Expects 768-dimensional features from Wav2Vec2 model

### 2. Dropout & None Handling
- **Trainer Side**: Sets conditions to None via _apply_condition_dropout
- **Model Side**: EfficientConditionEmbedding replaces None with zeros
- **Audio Exception**: Audio features preserved even during dropout
- **Device Consistency**: All zero tensors created with correct device/dtype

### 3. Debug Features
- Comprehensive logging of all condition keys and their status
- Detailed assertions with informative error messages
- Shape and type validation at multiple stages
- Automatic detection of valid tensors for batch size extraction

### 4. Temporal Context
- Previous context maintains continuity across windows
- Context size defined in config (default: 10 frames)
- Includes theta, expression, and audio from previous window
- Initial context uses learned parameters

### 5. Robustness Features
- Automatic NaN/Inf detection and replacement
- Tensor clamping to prevent extreme values
- Fallback to zero tensors for missing conditions
- Automatic blink state generation if not provided

## Data Flow

1. **Input Processing**:
   - Motion data sanitization
   - Condition validation and remapping
   - Shape expansion for batch consistency

2. **Condition Embedding**:
   - Audio features projection (required)
   - Control signal projections (with None handling)
   - Blink state embedding
   - Final normalization

3. **Motion Transformation**:
   - Noise embedding based on diffusion timestep
   - Motion flattening and combination with conditions
   - Positional encoding with context awareness
   - Transformer processing

4. **Output Generation**:
   - Pose parameter projection
   - Expression embedding projection
   - Derived outputs (rotation, translation, scale)
   - NaN/Inf sanitization

## Dependencies
- `torch`, `torch.nn`
- `diffusers.DDIMScheduler`
- `omegaconf.OmegaConf`
- `yaml`, `pathlib.Path`
- Custom modules: `logger`, `BlinkConditionHandler`
- Volumetric Avatar module (passed during initialization, frozen)