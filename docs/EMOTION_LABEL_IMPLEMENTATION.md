# Emotion Label Implementation

## Overview
This document describes the complete implementation of `emotion_label` field in the VASA dataset, ensuring it's present in every window to prevent KeyError crashes during training.

## Problem
Training was crashing with `KeyError: 'emotion_label'` because:
1. Old cached windows didn't have the `emotion_label` field
2. H5 cache didn't save/load `emotion_label` properly (it's a list of strings, not a tensor)
3. Collate function assumed all windows had `emotion_label`

## Solution
Multi-layered approach to ensure `emotion_label` is always present:

### 1. H5 Cache Save/Load (single_bucket_cache.py)

#### Saving emotion_label
**File**: `single_bucket_cache.py`

**In `save_all_windows()` (lines 118-123)**:
```python
elif key == 'emotion_label':
    # Save emotion_label as JSON string (list of strings)
    if isinstance(value, list):
        window_group.attrs['emotion_label'] = json.dumps(value)
    else:
        logger.warning(f"emotion_label is not a list, skipping: {type(value)}")
```

**In `append_windows()` (lines 381-386)**:
```python
if key == 'emotion_label':
    # Save emotion_label as JSON string (list of strings)
    if isinstance(value, list):
        window_group.attrs['emotion_label'] = json.dumps(value)
    else:
        logger.warning(f"emotion_label is not a list, skipping: {type(value)}")
```

#### Loading emotion_label
**In `load_window()` (lines 218-224)**:
```python
# Load emotion_label from window attributes if present
if 'emotion_label' in window_group.attrs:
    emotion_label_json = window_group.attrs['emotion_label']
    try:
        window_data['emotion_label'] = json.loads(emotion_label_json)
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode emotion_label for window {window_idx}")
```

### 2. Default Fallback During Loading (vasa_dataset.py)

When loading from cache, if `emotion_label` is missing (old cache), inject default values.

**For single_bucket cache (lines 2697-2707)**:
```python
# Ensure emotion_label exists (add default if missing from old cache)
if 'emotion_label' not in cached_data:
    # Get sequence length from any tensor in the data
    seq_len = 50  # default
    for key in ['theta', 'expression_embed', 'emotion']:
        if key in cached_data and isinstance(cached_data[key], torch.Tensor):
            seq_len = cached_data[key].shape[0]
            break
    # Create default neutral labels for all frames
    cached_data['emotion_label'] = ['neutral'] * seq_len
    logger.debug(f"Added default emotion_label for window {idx} (length: {seq_len})")
```

**For chunked cache (lines 2728-2738)** - Same logic

**For built-in cache (lines 2757-2767)** - Same logic

### 3. Robust Collate Function (vasa_sampler.py)

Handle cases where some windows might not have `emotion_label`.

**File**: `vasa_sampler.py` (lines 322-330)
```python
# Handle emotion_label separately (list of strings, don't stack)
# Check if emotion_label exists in all windows, not just the first one
if 'emotion_label' in processed_windows[0]:
    # Only include if ALL windows have it, otherwise use None
    if all('emotion_label' in w for w in processed_windows):
        batched['emotion_label'] = [w['emotion_label'] for w in processed_windows]
    else:
        # Some windows missing emotion_label, use None for all
        batched['emotion_label'] = [w.get('emotion_label', None) for w in processed_windows]
```

### 4. Window Creation (vasa_dataset.py)

New windows created during preprocessing already include `emotion_label`.

**File**: `vasa_dataset.py` (line 2908)
```python
'emotion_label': emotion_labels,  # List of strings like ["sad", "happy", "neutral", ...]
```

## Data Flow

### New Cache (Full Pipeline)
1. **Preprocessing** → Window created with `emotion_label` list (line 2908)
2. **Save to H5** → Stored as JSON in attributes (single_bucket_cache.py:118-123)
3. **Load from H5** → Parsed back to list (single_bucket_cache.py:218-224)
4. **Collate** → Batched as list of lists (vasa_sampler.py:322-330)

### Old Cache (Fallback)
1. **Load from H5** → `emotion_label` missing
2. **Fallback Injection** → Default `['neutral'] * seq_len` added (vasa_dataset.py:2697-2707)
3. **Collate** → Batched as list of lists (vasa_sampler.py:322-330)

## Testing

To verify the implementation:

1. **Delete old cache** to force recreation with emotion_label:
   ```bash
   rm cache_single_bucket/all_windows_cache.h5
   ```

2. **Run preprocessing** to create new cache:
   ```bash
   python preprocess_single_bucket.py --cache-frames --cache-emo-frames
   ```

3. **Verify emotion_label in cache**:
   ```python
   import h5py
   with h5py.File('cache_single_bucket/all_windows_cache.h5', 'r') as f:
       window = f['window_0']
       if 'emotion_label' in window.attrs:
           print("emotion_label found in cache:", json.loads(window.attrs['emotion_label']))
       else:
           print("emotion_label NOT in cache")
   ```

4. **Run training** to ensure no KeyError:
   ```bash
   python train_overfit.py
   ```

## Files Modified

1. **single_bucket_cache.py**
   - Added emotion_label saving in `save_all_windows()` (lines 118-123)
   - Added emotion_label loading in `load_window()` (lines 218-224)
   - Added emotion_label saving in `append_windows()` (lines 381-386)

2. **vasa_dataset.py**
   - Added default emotion_label fallback for single_bucket cache (lines 2697-2707)
   - Added default emotion_label fallback for chunked cache (lines 2728-2738)
   - Added default emotion_label fallback for built-in cache (lines 2757-2767)
   - emotion_label already created at line 2908 (no changes needed)

3. **vasa_sampler.py**
   - Made collate function robust to missing emotion_label (lines 322-330)

## Benefits

1. **Backward Compatible**: Old cache without emotion_label works via fallback
2. **Forward Compatible**: New cache saves emotion_label for future use
3. **Crash Resistant**: Multiple layers of defense against missing keys
4. **Type Safe**: JSON encoding/decoding ensures list integrity
5. **Memory Efficient**: emotion_label stored as attributes, not datasets

## Future Improvements

1. Consider storing emotion_label as HDF5 dataset instead of attribute if lists become very large
2. Add validation to ensure emotion_label length matches sequence length
3. Add option to recompute emotion_label for old cache windows
