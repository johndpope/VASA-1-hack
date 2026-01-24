# Expression Mismatch Diagnosis

## Problem
Generated expressions don't match ground truth expressions closely enough, even though:
- Silence test passes (low motion without audio)
- GT theta injection works (geometry is correct)
- Background compositing works
- Synchformer sync loss is enabled

## Root Cause Analysis

### 1. Missing Direct Expression Supervision

**Issue**: No direct L2/cosine loss comparing predicted expressions to GT expressions

**Current losses** (from overfit_config.yaml):
```yaml
lambda_dynamics: 10.0              # BUT: This may just regularize expression, not match GT
lambda_expression_variance: 10.0   # Matches variance, not actual values
lambda_expression_temporal: 5.0    # Matches temporal changes, not GT
lambda_audio_expr_coupling: 3.0    # Audio-expression correlation, not GT matching
```

**What's missing**:
```yaml
lambda_expression_cosine: X.X  # Direct cosine similarity to GT expressions
# OR
lambda_expression_l2: X.X      # Direct L2 distance to GT expressions
```

### 2. Synchformer Usage Issue

**Problem**: Synchformer is being used **during training** but the official repo says it's for **evaluation only**

From Synchformer README (lines 152-275):
- **Stage 1**: Pre-train feature extractors (segment-level audio-visual contrastive)
- **Stage 2**: Train synchronization module (frozen feature extractors)
- **Evaluation**: Test sync quality on held-out data

**Implications**:
1. Synchformer was **not designed** to provide training gradients for expression generation
2. It's a **pre-trained frozen evaluator** for measuring sync quality
3. Using it in training may **not** provide useful gradients to match GT expressions

### 3. What the Losses Actually Do

```yaml
# INDIRECT expression losses (don't directly match GT):
lambda_dynamics: 10.0             # Regularizes expression embeddings
lambda_expression_variance: 10.0  # Matches expression variance to dataset statistics
lambda_expression_temporal: 5.0   # Encourages temporal variation
lambda_audio_expr_coupling: 3.0   # Correlates expression changes with audio changes

# SYNC losses (measure audio-visual alignment, not GT matching):
lambda_sync: 2.0                  # Synchformer sync score
lambda_audio_lip: 3.0             # Audio-lip correlation via lip landmarks
lambda_lips: 2.0                  # Lip landmark matching
```

**None of these directly compare**: `predicted_expression_embed` vs `gt_expression_embed`

## Recommended Fixes

### Option 1: Add Direct Expression Loss (Recommended)

Add to `vasa_losses.py`:
```python
# Direct expression cosine similarity loss
if 'expression_embed' in predictions and 'expression_embed' in targets:
    pred_expr = F.normalize(predictions['expression_embed'], p=2, dim=-1)
    gt_expr = F.normalize(targets['expression_embed'], p=2, dim=-1)

    # Cosine similarity loss (higher is better, so we minimize 1 - cosine_sim)
    cosine_sim = (pred_expr * gt_expr).sum(dim=-1).mean()
    expression_cosine_loss = 1.0 - cosine_sim

    total_loss += lambda_expression_cosine * expression_cosine_loss
    loss_dict['expression_cosine'] = expression_cosine_loss.item()
```

Add to config:
```yaml
lambda_expression_cosine: 5.0  # Direct GT expression matching
```

### Option 2: Use Synchformer for Evaluation Only

Change training config:
```yaml
loss:
  use_sync_loss: false      # Disable during training
  use_synchformer: true     # Keep for evaluation/logging only
  lambda_sync: 0.0          # Zero weight during training
```

Use Synchformer only in evaluation:
- Monitor sync quality during validation
- Don't backprop through Synchformer during training
- Focus on direct expression losses instead

### Option 3: Expression Database Cosine Loss (Already Implemented)

The expression database approach (recently added) can help at **inference** time:
```yaml
# Clamp predictions to nearest valid expression
--expression-db ./expression_embeddings.h5
```

But for **training**, we still need direct GT matching loss.

## Testing Plan

1. **Add `lambda_expression_cosine`** loss to directly match GT
2. **Reduce `lambda_sync`** from 2.0 to 0.0 (use Synchformer for eval only)
3. **Increase `lambda_expression_cosine`** to 5.0-10.0 (strong GT matching)
4. **Test overfitting** - should now match GT expressions closely

## Expected Outcome

With direct expression loss:
- **Overfit test**: Expression L2 should drop from ~5.5 to < 1.0
- **Audio test**: Expressions should closely follow GT video patterns
- **Silence test**: Should still pass (no hallucinated motion)

## References

- Expression database: `expression_db.py` lines 51-92 (`get_closest()`)
- Synchformer README: `/media/2TB/VASA-1-hack/Synchformer/README.md` lines 150-275
- Loss implementation: `vasa_losses.py` lines 174-178 (Synchformer init)
- Audit tool findings: `AUDIT_TOOL_README.md` (Expression L2: 5.54, target: < 1.0)
