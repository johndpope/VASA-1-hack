# Phoneme Visualization Update - WandB Changes

## Summary

Updated WandB visualizations to:
1. **Disable** cluttered expression visualizations (candles, diff_map)
2. **Add phoneme labels** to audio-to-expression plot showing GT and predictions

## Changes Made

### 1. Disabled Expression Visualizations (vasa_trainer.py:3397-3436)

**Removed visualizations**:
- `visuals/expression_candles` - Candle-style expression comparison
- `visuals/expression_diff_map` - Expression difference heatmap

**Why**: These were cluttering the WandB dashboard and provided redundant information already available in other plots.

**Code changes**:
```python
# BEFORE: Both visualizations enabled
fig_candles = create_expression_candles(...)
wandb.log({"visuals/expression_candles": wandb.Image(fig_candles)}, step=step)

fig_diff = create_expression_difference_map(...)
wandb.log({"visuals/expression_diff_map": wandb.Image(fig_diff)}, step=step)

# AFTER: Both disabled (commented out)
# DISABLED: Expression candles and diff map - too cluttered in wandb
# from visualize_expression import create_expression_candles, create_expression_difference_map
```

### 2. Added Phoneme Labels to Audio Visualization

**Updated visualization**: `visuals/audio_to_expression`

**New features**:
- **Green labels at top**: Ground truth phonemes from wav2vec2
- **Blue labels at bottom**: Correct predictions (match GT)
- **Red labels at bottom**: Incorrect predictions (don't match GT)

**Implementation**:

#### vasa_trainer.py (lines 3447-3467)
```python
# Get phoneme ground truth and predictions for visualization
phoneme_gt = None
phoneme_pred = None
if 'aux_predictions' in outputs:
    aux = outputs['aux_predictions']
    if 'phoneme_gt' in aux:
        phoneme_gt = aux['phoneme_gt'][0]  # [8] for first batch item
    if 'phoneme_pred' in aux:
        phoneme_pred = torch.argmax(aux['phoneme_pred'][0], dim=-1)  # [8] predicted IDs

fig_audio_expr = create_audio_expression_visualization(
    audio_features=targets['audio_features'][0],
    target_expression=targets['expression_embed'][0],
    predicted_expression=outputs['expression_embed'][0],
    window_idx=step // 100,
    audio_reduce_to=32,
    expr_reduce_to=32,
    audio_projected=audio_projected[0] if audio_projected is not None else None,
    use_perceiver=use_perceiver,
    phoneme_gt=phoneme_gt,  # NEW
    phoneme_pred=phoneme_pred  # NEW
)
```

#### visualize_audio_expression.py (lines 132-174)

**Added phoneme label rendering**:
```python
# Add phoneme labels if available (8 latent queries mapped to 50 frames)
if phoneme_gt is not None or phoneme_pred is not None:
    # Phoneme vocab mapping (simplified for common phonemes)
    PHONEME_LABELS = {
        0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>',
        4: 'n', 5: 's', 6: 't', 7: 'ə', 8: 'l', 9: 'a',
        10: 'i', 11: 'k', 12: 'd', 13: 'm', 14: 'ɛ', 15: 'ɾ',
        # ... (40 common phonemes)
    }

    # Each phoneme spans roughly T/8 frames (50 frames / 8 queries = 6.25 frames per query)
    frames_per_phoneme = T / 8.0

    if phoneme_gt is not None:
        # Add GT phoneme labels at top (green boxes)
        for i, phoneme_id in enumerate(phoneme_gt_np):
            frame_pos = int(i * frames_per_phoneme + frames_per_phoneme / 2)
            label = PHONEME_LABELS.get(int(phoneme_id), f'{int(phoneme_id)}')
            ax_audio.text(frame_pos, audio_reduce_to + 1, f'GT:{label}',
                        ha='center', va='bottom', fontsize=8, color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))

    if phoneme_pred is not None:
        # Add predicted phoneme labels at bottom (blue=correct, red=incorrect)
        for i, phoneme_id in enumerate(phoneme_pred_np):
            frame_pos = int(i * frames_per_phoneme + frames_per_phoneme / 2)
            label = PHONEME_LABELS.get(int(phoneme_id), f'{int(phoneme_id)}')
            match_gt = (phoneme_gt is not None and
                      i < len(phoneme_gt_np) and
                      int(phoneme_pred_np[i]) == int(phoneme_gt_np[i]))
            color = 'blue' if match_gt else 'red'
            ax_audio.text(frame_pos, -2, f'Pred:{label}',
                        ha='center', va='top', fontsize=8, color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

    # Adjust plot limits to show labels
    ax_audio.set_ylim(-3, audio_reduce_to + 2)
```

## Phoneme Label Interpretation

### Color Coding

- **Green labels (top)**: Ground truth phonemes from wav2vec2 model
  - Example: `GT:a`, `GT:n`, `GT:ə`
  - These are the "correct" phonemes the model should predict

- **Blue labels (bottom)**: Correct predictions
  - Example: If GT is `a` and prediction is `a` → Blue `Pred:a`
  - Model successfully learned this phoneme

- **Red labels (bottom)**: Incorrect predictions
  - Example: If GT is `a` but prediction is `ə` → Red `Pred:ə`
  - Model needs more training for this phoneme

### Phoneme Symbols (IPA)

Common symbols you'll see:
- **Vowels**: `a`, `i`, `e`, `o`, `u`, `ə` (schwa), `ɛ`, `ɪ`, `ʊ`, `ɑ`, `ɔ`, `ʌ`
- **Consonants**: `n`, `s`, `t`, `l`, `k`, `d`, `m`, `p`, `b`, `f`, `v`, `h`, `z`, `ɡ`
- **Special**: `<pad>`, `<s>` (start), `</s>` (end), `<unk>` (unknown)
- **If numeric**: ID not in common 40 phonemes (from full 392 vocab)

### Temporal Mapping

- **8 phoneme queries** map to **50 frames**
- Each phoneme spans ~6.25 frames (50 / 8)
- Labels positioned at center of each phoneme's time span
- Example for 50 frames:
  - Phoneme 0: frames 0-6 (label at frame 3)
  - Phoneme 1: frames 7-12 (label at frame 9)
  - ...
  - Phoneme 7: frames 44-49 (label at frame 46)

## Expected Training Progress

### Early Training (Epochs 1-50)
```
GT:  a   n   ə   t   l   d   m   s
Pred: 12  34  5   23  16  45  8   15  (mostly RED - random predictions)
```
- Most predictions red (incorrect)
- Low accuracy (~20-30%)

### Mid Training (Epochs 50-200)
```
GT:  a   n   ə   t   l   d   m   s
Pred: a   34  ə   t   45  d   8   s  (some BLUE, some RED)
```
- Mix of blue and red
- Medium accuracy (~60-70%)

### Well-Trained (Epochs 200+)
```
GT:  a   n   ə   t   l   d   m   s
Pred: a   n   ə   t   l   d   m   s  (mostly BLUE - correct)
```
- Most predictions blue (correct)
- High accuracy (~85-95%)

## Benefits

1. **Visual feedback on phoneme learning**: See which phonemes the model learns first
2. **Easy error identification**: Red labels instantly show mispredictions
3. **Temporal correlation**: See how phonemes align with audio features
4. **Training progress**: Watch blue labels increase over epochs
5. **Less clutter**: Removed redundant expression visualizations

## Files Modified

1. **vasa_trainer.py** (lines 3397-3470)
   - Commented out expression_candles and expression_diff_map
   - Added phoneme_gt and phoneme_pred extraction
   - Pass phonemes to create_audio_expression_visualization

2. **visualize_audio_expression.py** (lines 15-174)
   - Added phoneme_gt and phoneme_pred parameters
   - Implemented phoneme label rendering with color coding
   - Updated docstring

## Testing

To verify the changes:

1. Run training and check WandB after a few steps
2. Look for `visuals/audio_to_expression` in WandB
3. Verify:
   - Green labels at top (GT phonemes)
   - Blue/Red labels at bottom (predictions)
   - No more `expression_candles` or `expression_diff_map` clutter

Example WandB path:
```
https://wandb.ai/snoozie/vasa-overfitting/runs/<run_id>

→ Media → visuals/audio_to_expression
```

## Next Steps

1. **Monitor phoneme accuracy**: Track blue vs red labels over epochs
2. **Identify hard phonemes**: Which phonemes stay red longest?
3. **Correlate with lip sync**: Do better phoneme predictions → better lip sync?
4. **Adjust loss weight**: If accuracy too low, increase `lambda_aux_phoneme` in config

## Notes

- Phoneme labels only show if `aux_predictions` exists in outputs
- Fallback to numeric IDs for uncommon phonemes (not in 40 most common)
- Full wav2vec2 vocab is 392 phonemes, but only 40 most common shown as symbols
- Label positioning assumes 50 frames per window (may need adjustment for different window sizes)
