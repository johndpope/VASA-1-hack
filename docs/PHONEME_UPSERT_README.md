# Phoneme Ground Truth Upsert

This README explains how to add `phoneme_gt` to existing cache files without reprocessing everything.

## Overview

The phoneme prediction auxiliary loss requires `phoneme_gt` (ground truth phoneme sequences) to be available in the dataset. If you already have cached data, you can use the upsert script to add phoneme_gt without re-running the full preprocessing pipeline.

## Quick Start

### 1. Dry Run (Test Mode)

First, run in dry-run mode to see what would happen without making changes:

```bash
./upsert_phoneme.sh cache_per_video 8 --dry-run
```

This will:
- Load the phoneme model
- Scan all cached videos
- Extract phoneme sequences from cached audio
- Report what would be updated (but not save)

### 2. Actual Upsert

Once you're confident, run the actual upsert:

```bash
./upsert_phoneme.sh cache_per_video 8
```

Or with custom cache directory:

```bash
./upsert_phoneme.sh /path/to/cache 8
```

## Script Details

### `upsert_phoneme_gt.py`

Python script that:
1. Scans the per-video cache directory (MD5 folders)
2. Opens each `metadata.h5` file
3. For each window:
   - Checks if `phoneme_gt` already exists (skips if yes)
   - Loads `audio_waveform` from cache
   - Extracts phoneme sequence using wav2vec2-xlsr-53-espeak-cv-ft
   - Aligns phonemes to 8 latent queries using max pooling
   - Saves `phoneme_gt` to the H5 file

**Arguments:**

```bash
python upsert_phoneme_gt.py \
    --cache_dir cache_per_video \
    --num_queries 8 \
    --sample_rate 16000 \
    --dry_run  # Optional: test without saving
```

**Features:**
- ✅ Skips windows that already have `phoneme_gt`
- ✅ Handles missing `audio_waveform` gracefully
- ✅ Uses gzip compression (level 4) for storage efficiency
- ✅ Progress bars and detailed logging
- ✅ Dry-run mode for testing

### `upsert_phoneme.sh`

Convenience shell script wrapper.

**Usage:**

```bash
./upsert_phoneme.sh [CACHE_DIR] [NUM_QUERIES] [--dry-run]
```

**Examples:**

```bash
# Default: cache_per_video, 8 queries
./upsert_phoneme.sh

# Custom cache directory
./upsert_phoneme.sh /path/to/cache

# Custom queries
./upsert_phoneme.sh cache_per_video 16

# Dry run
./upsert_phoneme.sh cache_per_video 8 --dry-run
```

## What Gets Added

For each window in the cache, the script adds:

```python
phoneme_gt: torch.Tensor[num_queries]  # e.g., [8]
```

This is a 1D tensor of phoneme IDs (0-49) aligned to the audio projection's latent queries.

**Example:**
```python
phoneme_gt = [12, 34, 5, 23, 12, 45, 8, 15]  # 8 phoneme IDs
```

## Storage Impact

- **Per window**: ~64 bytes (8 queries × int64 with gzip compression)
- **For 1000 windows**: ~64 KB
- **Total impact**: Minimal (<1% increase in cache size)

## Phoneme Model

The script uses `facebook/wav2vec2-xlsr-53-espeak-cv-ft`:
- Multilingual phoneme recognition
- ~50 IPA phoneme classes
- Pre-trained on Common Voice dataset
- Outputs phoneme probabilities per audio frame

## Verification

After running the upsert, verify the changes:

```python
import h5py
from pathlib import Path

# Open any metadata.h5 file
h5_path = Path('cache_per_video/<video_md5>/metadata.h5')
with h5py.File(h5_path, 'r') as h5f:
    # Check first window
    window_0 = h5f['window_0']

    # Should see phoneme_gt
    print(f"Keys: {list(window_0.keys())}")
    print(f"phoneme_gt shape: {window_0['phoneme_gt'].shape}")  # Should be (8,)
    print(f"phoneme_gt values: {window_0['phoneme_gt'][:]}")    # e.g., [12, 34, 5, ...]
```

## Troubleshooting

### "Cache directory does not exist"
Ensure you're using the correct path to your per-video cache directory.

### "No audio_waveform found"
Some windows may not have audio (e.g., videos without audio track). These are skipped and logged.

### "CUDA out of memory"
The phoneme model runs on GPU. If you encounter OOM errors:
1. Ensure no training jobs are running
2. Free GPU memory: `nvidia-smi` to check usage
3. The script processes one window at a time, so this is rare

### "phoneme_gt already exists"
This is normal - the script skips windows that already have phoneme_gt to avoid redundant work.

## Integration with Training

After running the upsert, your training should automatically pick up `phoneme_gt`:

1. The dataset loads `phoneme_gt` from cache
2. VASAModel adds it to `aux_predictions['phoneme_gt']`
3. VASALosses computes `aux_phoneme` loss
4. Loss is weighted by `lambda_aux_phoneme: 0.05` (from config)

You can verify by checking the logs during training:

```bash
grep "aux_phoneme" train.log
```

You should see lines like:
```
Auxiliary phoneme loss: 3.2451
```

## Performance

On a typical setup:
- **Speed**: ~10-20 windows/second
- **Time for 1000 windows**: ~1-2 minutes
- **GPU usage**: ~1GB VRAM for phoneme model

## Notes

- The script is **idempotent**: Running it multiple times is safe (skips existing phoneme_gt)
- **Backup recommendation**: Though the script only adds data (doesn't modify existing), consider backing up your cache before running
- **Parallel processing**: Currently single-threaded; could be parallelized if needed

## See Also

- `PERCEIVER_ENERGY_PREDICTION_PROPOSAL.md` - Full proposal for phoneme prediction
- `vasa_dataset.py:1966-2036` - Phoneme extraction implementation
- `vasa_model.py:152-155` - Phoneme prediction head
- `vasa_losses.py:1086-1105` - Phoneme loss computation
