# Expression Database for Cosine Similarity Loss

## Overview

The expression database prevents "floating in space" artifacts by constraining predicted expression embeddings to be similar to real expressions from the training dataset.

This is implemented as a **cosine embedding loss** that pulls predicted expressions toward the nearest real expression in a pre-built database.

## Architecture

```
MotionTransformer
    ↓
expression_embed [B, T, 128]
    ↓
cosine_head (normalized linear layer)
    ↓
pred_zdyn_proj [B, T, 128] (normalized)
    ↓
ExpressionDatabase.get_closest() ← database lookup (GPU)
    ↓
real_zdyn [B, T, 128] (nearest neighbor)
    ↓
F.cosine_embedding_loss(pred, real, target=+1)
    ↓
cosine_loss (scalar)
```

## Building the Database

### Step 1: Extract Expression Embeddings

```bash
# Extract embeddings every 5th frame from all motion_attributes H5 files
python build_expression_db.py \
    --motion-dir /media/12TB/VASA/motion_attributes \
    --output expression_embeddings.h5 \
    --frame-stride 5
```

**Expected output:**
- File: `expression_embeddings.h5`
- Size: ~100-500 MB (depends on dataset size)
- Contains: All expression embeddings (128-dim) sampled every 5 frames

### Step 2: Verify Database

```bash
# Verify database structure and test GPU loading
python build_expression_db.py --verify --output expression_embeddings.h5
```

## Usage in Training

### Update vasa_config.yaml

```yaml
model:
  expression_dim: 128
  expression_db_path: "expression_embeddings.h5"  # Add this line
```

### Training Script Changes

No changes needed! The model automatically:
1. Loads the database into GPU VRAM during initialization
2. Computes cosine loss after motion transformer
3. Returns `cosine_loss` in outputs

### Loss Integration

The `cosine_loss` is added to the total loss in `vasa_losses.py`:

```python
# In your loss computation
outputs = model(...)
total_loss = theta_loss + dynamics_loss + motion_loss

# Add cosine loss if available
if 'cosine_loss' in outputs:
    total_loss = total_loss + 5.0 * outputs['cosine_loss']  # Weight = 5.0
```

## How It Works

### 1. Database Structure

```
expression_embeddings.h5
├── expression_embeddings: [N, 128] float32
└── attributes:
    ├── num_embeddings: int
    ├── embedding_dim: 128
    ├── frame_stride: 5
    └── num_videos: int
```

### 2. GPU-Resident Lookup

The database is loaded into GPU VRAM once during initialization:

```python
# In VASAModel.__init__
self.expression_db = ExpressionDatabase('expression_embeddings.h5', device='cuda')
# Database now in GPU: [N, 128] normalized embeddings
```

### 3. Fast Nearest Neighbor Search

Cosine similarity computed via matrix multiplication:

```python
# Normalize queries
pred_norm = F.normalize(pred_zdyn, p=2, dim=1)  # [B*T, 128]

# Cosine similarity with entire database
similarity = pred_norm @ db_embeddings.T  # [B*T, N]

# Find closest (highest similarity)
indices = similarity.argmax(dim=1)  # [B*T]
closest = db_embeddings[indices]  # [B*T, 128]
```

**Speed**: ~1-2ms for batch size 4×50 frames on GPU

### 4. Cosine Embedding Loss

```python
# Maximize similarity (target = +1)
loss = F.cosine_embedding_loss(
    pred_normalized,   # [B*T, 128]
    real_nearest,      # [B*T, 128]
    target=+1,         # Maximize similarity
    reduction='mean'
)
```

## Benefits

### 1. Prevents Mode Collapse
- Expressions stay grounded in real data distribution
- No "floating in space" or unrealistic expressions

### 2. Fast & Memory Efficient
- Entire database in GPU VRAM (100-500 MB)
- Lookup: ~1-2ms per batch
- No backprop through database

### 3. Simple Integration
- 3 lines added to VASAModel
- No architecture changes
- Works with existing losses

## Hyperparameters

### Frame Stride
- Default: `5` (every 5th frame)
- Increase for smaller database (faster lookup, less coverage)
- Decrease for larger database (slower lookup, better coverage)

### Cosine Loss Weight
```python
# Start conservative
cosine_weight = 5.0

# Increase if expressions still drift
cosine_weight = 10.0

# Decrease if expressions are too rigid
cosine_weight = 2.0
```

### Cosine Head Initialization
The linear layer is initialized with normalized weights:
```python
self.cosine_head.weight.data = F.normalize(self.cosine_head.weight.data, p=2, dim=1)
```

This ensures predictions start in a reasonable range for cosine similarity.

## Debugging

### Check Database Loading
```python
print(model.expression_db)
# Output: ExpressionDatabase(N embeddings, 128D, device=cuda:0)
```

### Monitor Cosine Loss
```python
# In training loop
if 'cosine_loss' in outputs:
    print(f"Cosine loss: {outputs['cosine_loss'].item():.6f}")
```

### Visualize Nearest Neighbors
```python
# Get predictions and closest real expressions
pred_zdyn = outputs['expression_embed']  # [B, T, 128]
real_zdyn = outputs['real_zdyn']         # [B, T, 128]

# Check similarity
similarity = F.cosine_similarity(pred_zdyn, real_zdyn, dim=-1)
print(f"Avg similarity: {similarity.mean().item():.4f}")
```

## File Sizes

| Dataset Size | Frame Stride | Database Size | VRAM Usage |
|--------------|--------------|---------------|------------|
| 6 videos     | 5            | ~10 MB        | ~10 MB     |
| 100 videos   | 5            | ~200 MB       | ~200 MB    |
| 1000 videos  | 5            | ~2 GB         | ~2 GB      |

For large datasets, consider:
- Increasing `frame_stride` (e.g., 10 or 20)
- Using batch lookup with `get_closest_batch()`
- Splitting database into chunks (not implemented)

## Testing

```bash
# Test database module
python expression_db.py
```

Expected output:
```
Loading test database...
Testing queries...
2D query: torch.Size([16, 128]) -> torch.Size([16, 128]) in 0.15ms
3D query: torch.Size([4, 50, 128]) -> torch.Size([4, 50, 128]) in 0.82ms
Average cosine similarity: 0.9234
✅ Test complete
```

## Troubleshooting

### Database Not Loading
```
ERROR: Expression database not found: expression_embeddings.h5
```
**Solution**: Run `build_expression_db.py` first

### OOM During Database Load
```
CUDA out of memory
```
**Solution**: Increase `frame_stride` or reduce dataset size

### Low Cosine Similarity
```
Avg similarity: 0.2341  # Too low!
```
**Causes**:
- Cosine head not properly normalized
- Predictions out of distribution
- Database too small

**Solution**: Check cosine_head initialization, increase database size

### High Cosine Loss
```
cosine_loss: 1.2450  # Should be < 0.5
```
**Causes**:
- Model predictions drifting
- Need more training
- Weight too high

**Solution**: Train longer, reduce cosine weight

## References

- Paper: Normalized embeddings for mode regularization
- Inspiration: Contrastive learning / metric learning
- Similar to: Memory banks in self-supervised learning
