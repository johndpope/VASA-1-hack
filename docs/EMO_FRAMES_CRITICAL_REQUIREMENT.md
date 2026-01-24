# EMO Frames: Critical Requirement for Training

## Problem Discovered

During training analysis, we discovered:

1. **312/462 windows (68%) had emo_frames**, 150 windows (32%) were missing
2. **Missing emo_frames caused critical losses to be skipped**:
   - `perceptual` (LPIPS) - lambda=1.0
   - `mouth_perceptual` - lambda=100.0 (most important!)
   - Any loss requiring target frames
3. **Training was silently degraded** - batches without complete emo_frames skipped perceptual losses
4. **Abnormal SRT values detected** - Some windows had scale 1.36-1.68 instead of ~1.0, causing geometric distortions

## Why EMO Frames Are Critical

**EMO frames** (from volumetric avatar with identity transfer) serve as **ground truth targets** for:

1. **Perceptual Loss (LPIPS)** - Measures perceptual similarity between generated and target frames
2. **Mouth Perceptual Loss** - 100x weighted loss on mouth region (TalkVid-style)
3. **Reconstruction Loss** - Pixel-level matching to target
4. **Visual Quality** - Without emo_frames, model has no visual supervision

**Without emo_frames, you're essentially training blind** - the model has motion/audio losses but no visual quality feedback.

## Solution Implemented

### 1. Preprocessing: Fail Fast (preprocess_single_bucket.py:173-179)

```python
# CRITICAL CHECK: Ensure emo_frames exist (required for training)
if 'emo_frames' not in window_data:
    quality_filtered_count += 1
    logger.error(f"❌ CRITICAL: Window {idx} has NO emo_frames! Cannot train without them. Skipping window.")
    if video_path:
        logger.error(f"   Video: {video_path}, window_idx: {window_idx_in_video}")
    continue  # Skip this window entirely
```

**What this does:**
- Checks EVERY window during preprocessing
- If emo_frames missing → **SKIP the window** (don't cache it)
- Logs which videos/windows failed
- Only caches windows that CAN be used for training

### 2. Training: Fail Hard (vasa_dataset.py:2713-2720)

```python
else:
    # FAIL HARD: emo_frames are REQUIRED for training
    logger.error(f"❌ CRITICAL: emo_frames missing from disk for window {idx}")
    logger.error(f"   Video: {video_path}, window_idx: {window['window_idx']}")
    logger.error(f"   Expected at: {self.emo_frame_cache.get_window_dir(video_path, window['window_idx'])}")
    logger.error(f"   This window CANNOT be used for training without emo_frames!")
    logger.error(f"   ACTION: Re-run preprocessing with --cache-emo-frames to generate missing emo_frames")
    # Return None to exclude this window from training
    return None
```

**What this does:**
- If training encounters a window without emo_frames on disk → **RETURN None**
- Logs detailed error with exact location
- Tells user to re-run preprocessing
- Window is excluded from the batch

### 3. Collate Function: Already Handles None

The collate function already filters out None windows (from vasa_sampler.py), so windows without emo_frames are automatically excluded from batches.

## How to Fix Your Cache

### Option 1: Re-run Preprocessing (Clean Start)

```bash
# Delete existing cache
rm -rf cache_single_bucket/all_windows_cache.h5
rm -rf cache_single_bucket/emo_frames/*

# Re-run preprocessing - will only cache windows WITH emo_frames
python preprocess_single_bucket.py --cache-frames --cache-emo-frames
```

### Option 2: Identify and Remove Bad Videos

```bash
# Check which windows are missing emo_frames
python check_srt_emo.py  # Shows which windows have issues

# Manually check which videos are failing
# Look for patterns in error logs from preprocessing

# Remove problematic videos from junk2/ folder
# Then re-run preprocessing
```

### Option 3: Continue with Current Cache (Degraded)

**NOT RECOMMENDED** - You'll train with reduced dataset and missing perceptual losses.

Current cache:
- ✅ 312 windows WITH emo_frames (will train normally)
- ❌ 150 windows WITHOUT emo_frames (will be skipped, perceptual losses missing)

## Expected Behavior After Fix

### During Preprocessing:
```
❌ CRITICAL: Window 208 has NO emo_frames! Cannot train without them. Skipping window.
   Video: junk2/videovideoheiLf1WySUw-scene13_scene2.mp4, window_idx: 1
...
✅ Successfully cached 312 windows
   ⚠️ Filtered 150 low-quality windows (no emo_frames)
```

### During Training:
```
✅ All windows have emo_frames
✅ Perceptual losses computed for ALL batches
✅ mouth_perceptual loss (lambda=100.0) is working
✅ No batches skipped due to missing emo_frames
```

## Diagnostic Commands

```bash
# Check current emo_frames coverage
python check_srt_emo.py

# Count windows in H5 cache
python -c "import h5py; f = h5py.File('cache_single_bucket/all_windows_cache.h5', 'r'); print(f'Total windows: {f.attrs[\"num_windows\"]}')"

# Count emo_frame directories on disk
find cache_single_bucket/emo_frames -name "window_*" -type d | wc -l

# Check for missing emo_frames during training
python train_overfit.py 2>&1 | grep "CRITICAL.*emo_frames"
```

## Root Cause Analysis

Why were emo_frames missing?

1. **EMO Volumetric Avatar Generation Failed** - Some videos/frames couldn't be processed by EMO
2. **3DDFA Issues** - Face detection or 3D reconstruction failed for some frames
3. **Disk Cache Corruption** - Files were deleted or corrupted
4. **Preprocessing Interruption** - Previous preprocessing run was interrupted

The solution: **Only cache windows that successfully generated emo_frames**.

## Benefits of This Approach

✅ **No Silent Degradation** - Training will use ONLY high-quality windows
✅ **Fail Fast** - Problems caught during preprocessing, not training
✅ **Clear Errors** - Exact video/window reported when issues occur
✅ **Data Quality** - Ensures all training data has required visual supervision
✅ **Reproducibility** - Cache contains only trainable windows

## Summary

**Before Fix:**
- 462 windows cached
- 312 (68%) usable for training
- 150 (32%) missing emo_frames → perceptual losses skipped
- Silent degradation of training quality

**After Fix:**
- Only cache windows WITH emo_frames
- 100% of cached windows usable for training
- Fail hard if emo_frames missing
- Clear error messages with actionable fixes

**Action Required:** Re-run preprocessing to rebuild cache with only valid windows.
