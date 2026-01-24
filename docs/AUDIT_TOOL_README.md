# VASA Expression Audit Tool

## Overview

The audit tool compares ground truth expressions from the training cache with model predictions to diagnose training issues.

## Quick Start

### Interactive Mode (Recommended)
```bash
./audit.sh
```

This will:
1. Show a menu to select config (overfit or full training)
2. Auto-detect checkpoint and video paths
3. Run the audit
4. Display results and offer to view the plot

### Command Line Mode
```bash
python audit_expressions.py \
    --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
    --identity ./data/IMG_1.png \
    --config overfit_config.yaml \
    --checkpoint checkpoints_overfit/best_checkpoint.pt \
    --output-dir expression_audit
```

## What It Does

1. **Loads Ground Truth** from training cache (`cache_single_bucket/all_windows_cache.h5`)
   - Theta (head pose): `[1, T, 3, 4]`
   - Expression embeddings: `[1, T, 128]`
   - Audio features: `[1, T, 768]`

2. **Generates Predictions** using the trained model
   - Uses **identical** audio features as ground truth
   - Runs DDIM sampling with CFG
   - Produces predicted theta and expressions

3. **Compares Frame-by-Frame**
   - Expression L2 distance
   - Theta L2 distance
   - Expression cosine similarity
   - Audio feature alignment check

4. **Outputs Results**
   - `expression_comparison.png` - Visualization of L2 distances over time
   - `expression_metrics.csv` - Detailed per-frame metrics
   - Console summary with statistics

## Understanding the Results

### Good Overfitting Signs
- **Expression L2 < 1.0** - Model reproduces training expressions well
- **Theta L2 < 0.3** - Head pose matches ground truth
- **Cosine Sim > 0.9** - Expression vectors highly aligned
- **Audio Diff = 0.0** - Confirmation that same audio is used

### Current Results (Epoch 226)
```
Expression L2:     5.54 ± 0.78   ❌ TOO HIGH (should be < 1.0)
Theta L2:          1.50 ± 0.06   ❌ TOO HIGH (should be < 0.3)
Expression Cosine: NaN            ⚠️  Issue with zero vectors
Audio Diff:        0.000000      ✅ GOOD
```

### Diagnosis
**Model is NOT overfitting** despite 226 epochs. Possible causes:
1. Learning rate too high (need to reduce)
2. Loss weights wrong (cosine loss too weak)
3. Model capacity too small (12.5M vs 29M target)
4. Diffusion sampling issues (CFG scales, num_steps)

## Output Files

### expression_comparison.png
Three subplots:
1. **Expression L2 Distance** - Higher = worse match
2. **Theta L2 Distance** - Higher = worse pose match
3. **Expression Cosine Similarity** - Higher = better match (1.0 = perfect)

Red dashed line = mean value

### expression_metrics.csv
Columns:
- `frame`: Frame number (0 to T-1)
- `expr_l2`: Expression L2 distance for this frame
- `theta_l2`: Theta L2 distance for this frame
- `expr_cosine`: Expression cosine similarity for this frame

Use this to identify worst frames and analyze temporal patterns.

## Troubleshooting

### "No cached windows found"
- Make sure you've run training and the cache exists at `cache_single_bucket/all_windows_cache.h5`
- For overfit mode, the cache should contain windows from your training video

### "Audio features DIFFER by X"
- Should show 0.000000 since we use GT audio
- If non-zero, this is a bug in the audit script

### "Cosine Sim: mean=nan"
- Happens when expression vectors are zero or near-zero
- Usually indicates model not learning at all

### Import errors
```bash
# Make sure you're in the right environment
conda activate actalker

# Check dependencies
pip install scipy pandas matplotlib
```

## Advanced Usage

### Comparing Multiple Checkpoints
```bash
# Epoch 100
python audit_expressions.py \
    --checkpoint checkpoints_overfit/checkpoint_epoch_100.pt \
    --output-dir audit_epoch100

# Epoch 200
python audit_expressions.py \
    --checkpoint checkpoints_overfit/checkpoint_epoch_200.pt \
    --output-dir audit_epoch200

# Compare results
python -c "
import pandas as pd
df1 = pd.read_csv('audit_epoch100/expression_metrics.csv')
df2 = pd.read_csv('audit_epoch200/expression_metrics.csv')
print(f'Epoch 100: expr_l2={df1.expr_l2.mean():.4f}')
print(f'Epoch 200: expr_l2={df2.expr_l2.mean():.4f}')
print(f'Improvement: {((df1.expr_l2.mean() - df2.expr_l2.mean()) / df1.expr_l2.mean() * 100):.1f}%')
"
```

### Analyzing Specific Frame Ranges
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('expression_audit/expression_metrics.csv')

# Find problematic regions
window_size = 50
for i in range(0, len(df) - window_size, window_size):
    window = df.iloc[i:i+window_size]
    if window['expr_l2'].mean() > 6.0:  # Threshold
        print(f"Problematic region: frames {i}-{i+window_size}, L2={window['expr_l2'].mean():.2f}")
```

### Custom Metrics
```python
import pandas as pd
import numpy as np

df = pd.read_csv('expression_audit/expression_metrics.csv')

# Temporal smoothness
df['expr_l2_diff'] = df['expr_l2'].diff().abs()
print(f"Expression jitter: {df['expr_l2_diff'].mean():.4f}")

# Percentile analysis
print(f"90th percentile L2: {np.percentile(df['expr_l2'], 90):.4f}")
print(f"95th percentile L2: {np.percentile(df['expr_l2'], 95):.4f}")
print(f"99th percentile L2: {np.percentile(df['expr_l2'], 99):.4f}")
```

## Integration with Training

### Check Progress During Training
```bash
# In terminal 1: Training
./train.sh  # Select overfit

# In terminal 2: Periodic audits
watch -n 300 './audit.sh <<< "1\ny\ny\ny\ny\nn"'  # Every 5 minutes
```

### Automated Checkpoint Comparison
```bash
#!/bin/bash
# audit_all_checkpoints.sh

for ckpt in checkpoints_overfit/checkpoint_epoch_*.pt; do
    epoch=$(basename $ckpt .pt | grep -oP '\d+$')
    echo "Auditing epoch $epoch..."

    python audit_expressions.py \
        --checkpoint "$ckpt" \
        --config overfit_config.yaml \
        --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
        --identity ./data/IMG_1.png \
        --output-dir "audit_epoch_$epoch"

    # Extract summary
    python -c "
import pandas as pd
df = pd.read_csv('audit_epoch_$epoch/expression_metrics.csv')
print(f'$epoch,{df.expr_l2.mean():.4f},{df.theta_l2.mean():.4f}')
" >> audit_summary.csv
done

echo "Summary saved to audit_summary.csv"
```

## Files

- `audit.sh` - Interactive menu script
- `audit_expressions.py` - Core audit tool
- `expression_audit/` - Default output directory
  - `expression_comparison.png`
  - `expression_metrics.csv`

## Related Documentation

- `CLAUDE.md` - Main project instructions
- `LOSS_CLEANUP_SUMMARY.md` - Loss function changes
- `TEMPORAL_STABILIZATION.md` - Motion smoothing details
