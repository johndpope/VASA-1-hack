# Direct GT Expression Loss Implementation

## Summary

Added **direct ground truth expression matching loss** to force the model to learn GT expressions, not just their statistics.

## Problem

The model was using only **indirect** expression losses:
- `lambda_dynamics`: Regularizes expressions but doesn't match GT
- `lambda_expression_variance`: Matches variance statistics, not actual values
- `lambda_expression_temporal`: Encourages temporal changes, not GT matching
- `lambda_audio_expr_coupling`: Correlates with audio, not GT
- `lambda_sync`: Synchformer measures sync quality (designed for eval, not training)

**Result**: Expression L2 distance was ~5.5 (target: < 1.0 for good overfitting)

## Solution Implemented

### 1. Added Direct GT Expression Loss

**File**: `vasa_losses.py` lines 634-659

```python
# 1.7 Direct GT Expression Matching Loss - CRITICAL for overfitting
if 'expression_embed' in targets:
    gt_expr = targets['expression_embed']  # [B, T, D]
    pred_expr = outputs['expression_embed']  # [B, T, D]

    # Normalize both for cosine similarity
    pred_expr_norm = F.normalize(pred_expr, p=2, dim=-1)  # [B, T, D]
    gt_expr_norm = F.normalize(gt_expr, p=2, dim=-1)  # [B, T, D]

    # Cosine similarity loss: minimize 1 - cosine_similarity
    cosine_sim = (pred_expr_norm * gt_expr_norm).sum(dim=-1).mean()
    expression_cosine_loss = 1.0 - cosine_sim

    # Weight and add to losses
    lambda_expression_cosine = getattr(self.config.loss, 'lambda_expression_cosine', 0.0)
    if lambda_expression_cosine > 0:
        losses['expression_cosine'] = expression_cosine_loss * lambda_expression_cosine

        # Also compute L2 distance for monitoring
        l2_dist = (pred_expr - gt_expr).pow(2).sum(dim=-1).sqrt().mean()
        metrics['expression_gt/l2_distance'] = l2_dist.item()
        metrics['expression_gt/cosine_similarity'] = cosine_sim.item()
```

### 2. Updated Configuration

**File**: `overfit_config.yaml` line 171

```yaml
loss:
  lambda_expression_cosine: 10.0  # NEW: Direct GT expression matching via cosine similarity
  lambda_sync: 0.5                # REDUCED from 2.0 - Synchformer for evaluation, not training
```

## How It Works

1. **Normalize expressions**: Both predicted and GT expressions are L2-normalized
2. **Compute cosine similarity**: Dot product of normalized vectors
3. **Loss = 1 - similarity**: Minimize distance, maximize similarity
4. **Strong weight (10.0)**: Forces model to prioritize GT matching
5. **Monitor L2 distance**: Track actual Euclidean distance for audit tool

## Expected Outcomes

### Metrics to Watch

In WandB/training logs:
```
expression_gt/cosine_similarity: Should increase to > 0.95
expression_gt/l2_distance: Should decrease to < 1.0
loss/expression_cosine: Should decrease toward 0.0
```

### Overfitting Test

After retraining with this loss:
```bash
python audit_expressions.py \
    --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
    --identity ./data/IMG_1.png \
    --config overfit_config.yaml \
    --checkpoint checkpoints_overfit/best_checkpoint.pt
```

**Expected results**:
- Expression L2: < 1.0 (was 5.54)
- Cosine similarity: > 0.95 (closer to 1.0 = better)
- Visual comparison: Expressions should closely match GT video

### Inference Test

```bash
python vi_v2.py \
    --input './junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4' \
    --output 'output_with_gt_loss.mp4' \
    --target_image './data/IMG_1.png' \
    --gt-theta-h5 'cache/videovideoeI2V8Bd5X9s-scene6_scene1_gt_theta.h5' \
    --config overfit_config.yaml \
    --checkpoint './checkpoints_overfit/best_checkpoint.pt'
```

**Expected**: Generated expressions should now match GT video patterns closely

## Why Cosine Similarity?

1. **Scale invariant**: Focuses on direction, not magnitude
2. **Robust to normalization**: Works even if expression magnitudes vary
3. **Smooth gradients**: Better for optimization than L1/L2 alone
4. **Interpretable**: Similarity ∈ [0, 1], easy to monitor

## Synchformer Adjustment

Reduced `lambda_sync` from 2.0 to 0.5 because:
- Synchformer is designed for **evaluation** (see official README lines 150-275)
- It's a frozen pre-trained model for measuring sync quality
- Not optimized for providing training gradients
- Direct expression loss is more effective for learning GT

## Monitoring During Training

Watch these in WandB:
```
✅ expression_gt/cosine_similarity ↑  (target: > 0.95)
✅ expression_gt/l2_distance ↓        (target: < 1.0)
✅ loss/expression_cosine ↓           (target: < 0.05)
⚠️ loss/expression_variance           (should stay reasonable, not collapse)
```

## Files Changed

1. **vasa_losses.py**: Added GT expression loss computation (lines 634-659)
2. **overfit_config.yaml**: Added `lambda_expression_cosine: 10.0` (line 171), reduced `lambda_sync` (line 189)

## Next Steps

1. **Restart training** with updated config
2. **Monitor cosine similarity** - should improve within first few epochs
3. **Run audit tool** after ~10 epochs to verify improvement
4. **Adjust weight** if needed (increase if not learning fast enough, decrease if overfitting too aggressively)

## Related Documents

- Diagnosis: `EXPRESSION_MISMATCH_DIAGNOSIS.md`
- Expression DB: `expression_db.py`
- Audit tool: `AUDIT_TOOL_README.md`
- Synchformer: `SYNCHFORMER_INTEGRATION.md`
