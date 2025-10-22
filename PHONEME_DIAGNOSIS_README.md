# Phoneme Diagnosis Tool

Inspect phoneme ground truth and model predictions for cached windows.

## Quick Start

### Option 1: Show Ground Truth Only (No Model)

```bash
# Scan cache and show phoneme_gt for first 3 videos
python diagnose_phoneme.py --no_model

# Show phoneme_gt for specific video
python diagnose_phoneme.py --video s1/video.mp4 --no_model
```

### Option 2: Show Predictions vs Ground Truth (With Model)

```bash
# Use trained model to compare predictions vs GT
python diagnose_phoneme.py --checkpoint checkpoints_overfit/best_checkpoint.pt

# Specific video
./diagnose_phoneme.sh cache_per_video s1/video.mp4

# Or directly
python diagnose_phoneme.py \
    --video s1/video.mp4 \
    --checkpoint checkpoints_overfit/best_checkpoint.pt
```

## What It Shows

### Ground Truth Only Mode

```
Window window_0:
  Ground Truth:
    IDs:     [12 34  5 23 12 45  8 15]
    Phonemes: iː ɛ d t iː ɹ f l
```

**Interpretation**:
- 8 phoneme IDs (one per latent query)
- Human-readable IPA phoneme symbols
- Each phoneme represents ~6-7 frames of the window

### With Model Predictions

```
Window window_0:
  Ground Truth:
    IDs:     [12 34  5 23 12 45  8 15]
    Phonemes: iː ɛ d t iː ɹ f l
  Prediction:
    IDs:     [12 34  5 23 16 45  8 15]
    Phonemes: iː ɛ d t m ɹ f l
    Confidence: [0.87 0.92 0.78 0.85 0.65 0.89 0.91 0.83]
    Accuracy: 87.50%
  Mismatches:
    Query 4: iː → m

SUMMARY STATISTICS
Average Accuracy: 82.30%
Average Confidence: 84.50%
```

**Interpretation**:
- **Accuracy**: Percentage of correctly predicted phonemes
- **Confidence**: Model's certainty (softmax probability)
- **Mismatches**: Where prediction differs from ground truth
- Query 4 mismatch: Model predicted /m/ instead of /iː/

## Command-Line Options

```bash
python diagnose_phoneme.py \
    --cache_dir cache_per_video \        # Cache directory
    --video s1/video.mp4 \               # Specific video (optional)
    --max_videos 5 \                     # Max videos to scan (if no --video)
    --max_windows 10 \                   # Max windows per video
    --config overfit_config.yaml \       # Model config
    --checkpoint best_checkpoint.pt \    # Model checkpoint
    --no_model                           # Skip model (GT only)
```

## Use Cases

### 1. Verify Upsert Worked

After running `./upsert_phoneme.sh`, verify phonemes were added:

```bash
python diagnose_phoneme.py --no_model --max_windows 3
```

**Expected**: Should show phoneme_gt for all windows

### 2. Check Model Training Progress

During training, check if model is learning phonemes:

```bash
# Early training (expect low accuracy)
python diagnose_phoneme.py --checkpoint checkpoints_overfit/checkpoint_epoch_10.pt

# Later training (expect high accuracy)
python diagnose_phoneme.py --checkpoint checkpoints_overfit/checkpoint_epoch_100.pt
```

**Expected accuracy progression**:
- Epoch 1-10: ~20% (random guessing)
- Epoch 50: ~60-70% (learning)
- Epoch 200: ~85-95% (well-trained)

### 3. Debug Specific Video

If a video shows poor lip sync, check its phoneme predictions:

```bash
./diagnose_phoneme.sh cache_per_video s1/problematic_video.mp4
```

### 4. Compare Different Checkpoints

```bash
# Baseline (untrained)
python diagnose_phoneme.py --checkpoint checkpoints_overfit/initial.pt

# After phoneme loss training
python diagnose_phoneme.py --checkpoint checkpoints_overfit/with_phoneme_loss.pt
```

## Interpreting Results

### Good Results

```
Average Accuracy: 92.30%
Average Confidence: 89.50%
```

- **High accuracy** (>85%): Model learned phoneme patterns well
- **High confidence** (>80%): Model is certain about predictions
- **Few mismatches**: Most predictions match ground truth

### Poor Results

```
Average Accuracy: 23.50%
Average Confidence: 45.20%
```

- **Low accuracy** (<40%): Model hasn't learned phonemes yet
- **Low confidence** (<60%): Model is uncertain
- **Many mismatches**: Predictions are mostly wrong

**Possible causes**:
- Model not trained yet (expected early in training)
- Phoneme loss weight too low (`lambda_aux_phoneme`)
- Phoneme ground truth is incorrect (check upsert)

### Mixed Results

```
Average Accuracy: 68.40%
Average Confidence: 72.30%
```

- **Medium accuracy** (60-80%): Model is learning but not converged
- **Medium confidence** (60-80%): Some certainty but improving
- **Some mismatches**: Expected during training

**Action**: Continue training, check again after more epochs

## Phoneme Reference

Common phonemes you'll see:

**Vowels**:
- `a` - as in "father"
- `i`, `iː` - as in "see"
- `u`, `uː` - as in "too"
- `ɛ` - as in "bed"
- `ə` - schwa (unstressed, as in "about")

**Consonants**:
- `m`, `n`, `ŋ` - nasals
- `p`, `b`, `t`, `d`, `k`, `g` - stops
- `f`, `v`, `s`, `z`, `ʃ`, `ʒ` - fricatives
- `l`, `r` (or `ɹ`) - liquids
- `w`, `j` - glides

**Special**:
- `<pad>` or `0` - padding/silence
- `ʔ` - glottal stop
- `ˈ` - stress marker

## Output Files

The script only prints to console. To save output:

```bash
python diagnose_phoneme.py --video s1/video.mp4 > phoneme_report.txt
```

## Performance

- **Speed**: ~5-10 windows/second (with model)
- **GPU**: ~2GB VRAM for model inference
- **Time**: Few seconds per video

## Troubleshooting

### "No phoneme_gt found"

**Problem**: Cache doesn't have phoneme_gt yet

**Solution**: Run upsert first:
```bash
./upsert_phoneme.sh
```

### "Checkpoint not found"

**Problem**: Checkpoint path is incorrect

**Solution**:
- Use `--no_model` to skip predictions
- Or specify correct checkpoint path

### "CUDA out of memory"

**Problem**: GPU memory full

**Solution**:
- Reduce `--max_windows`
- Use `--no_model` flag
- Free GPU memory first

## Advanced Usage

### Batch Analysis

Analyze multiple checkpoints:

```bash
for epoch in 10 50 100 200; do
    echo "=== Epoch $epoch ==="
    python diagnose_phoneme.py \
        --checkpoint checkpoints_overfit/checkpoint_epoch_${epoch}.pt \
        --max_videos 1 \
        --max_windows 5
done
```

### Export to CSV

```python
# Custom script to export results
import csv

results = []  # From diagnose_phoneme.py
with open('phoneme_analysis.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Window', 'Accuracy', 'Avg Confidence'])
    for r in results:
        writer.writerow([r['window'], r['accuracy'], r['confidence'].mean()])
```

## See Also

- `PHONEME_UPSERT_README.md` - How to add phoneme_gt to cache
- `PHONEME_LOSS_IMPLEMENTATION_SUMMARY.md` - Full implementation details
- `PERCEIVER_ENERGY_PREDICTION_PROPOSAL.md` - Original proposal
