# Cache Validation and Cleaning Guide

## Problem
The H5 cache may contain windows with poor quality that cause training issues:
- UV warp magnitude collapse (< 0.15)
- Low expression variance (< 0.01)
- Invalid face landmarks (< 50% valid frames)

## Solution: Cache Validation Tools

### 1. Validate Cache Quality

Check what's in your cache without making changes:

```bash
python validate_cache.py --cache-dir cache_single_bucket --validate-only
```

This will:
- Scan all windows in the cache
- Check UV warp magnitude, expression variance, and landmark quality
- Generate a report showing bad windows and videos
- Save detailed report to `cache_single_bucket/quality_report.json`

### 2. Clean Bad Windows

Remove all bad quality windows from cache:

```bash
python validate_cache.py --cache-dir cache_single_bucket --clean
```

This will:
- Validate the cache (same as step 1)
- Show which videos will be removed
- Ask for confirmation
- Remove all windows from bad videos
- Rebuild the cache without bad data
- Re-validate to confirm success

### 3. Invalidate Specific Video

Remove a specific video from cache:

```bash
python validate_cache.py --cache-dir cache_single_bucket --invalidate-video "junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4"
```

This will:
- Find all windows from that video
- Remove them from cache
- Rebuild cache without those windows

## Quality Thresholds

Windows are considered bad if they fail ANY of these checks:

1. **UV Warp Magnitude** < 0.15
   - Indicates collapsed/identity warps
   - Model not learning proper deformations

2. **Expression Variance** < 0.01
   - Nearly identical expression parameters
   - Static predictions

3. **Valid Landmarks** < 50%
   - More than half the frames have no face landmarks
   - Poor face detection quality

## Integration with Training

### Automatic Bad Window Detection (vasa_dataset.py)

When EMO frames are generated during preprocessing, bad UV warps are detected immediately:

```python
# After EMO generation (line 2894-2919)
if uv_magnitude < 0.15 or uv_std < 0.01:
    # Log error
    # Dispatch BAD_UV_WARPS event to tracker
    # Return zero sample (skip window)
```

This prevents bad windows from entering the cache during preprocessing.

### Runtime Window Skipping (vasa_trainer.py)

During training, windows with poor frame generation quality are skipped:

```python
# Before loss computation (line 1311-1318)
if gen_stats['valid_masks_ratio'] < 0.5:
    # Skip window to avoid gradient issues
    continue
```

## Workflow

### Fresh Start (Recommended)

If your current cache has many bad windows:

```bash
# 1. Validate current cache
python validate_cache.py --validate-only

# 2. Clean bad windows
python validate_cache.py --clean

# 3. If still many bad windows, rebuild from scratch
rm -rf cache_single_bucket/all_windows_cache.h5
python preprocess_single_bucket.py

# 4. Resume training
python train_overfit.py
```

### Incremental Cleaning

If you want to keep most of the cache:

```bash
# 1. Validate and get report
python validate_cache.py --validate-only

# 2. Review quality_report.json to see bad videos

# 3. Clean automatically
python validate_cache.py --clean

# OR manually remove specific videos
python validate_cache.py --invalidate-video "path/to/bad/video.mp4"
```

## Expected Results

After cleaning, you should see:
- Quality pass rate: 100% (or close to it)
- No "UV warp magnitude collapse" warnings during training
- More windows actually processed (not skipped)
- Stable gradient flow
- Better convergence

## Files

- `validate_cache.py` - Main validation/cleaning script
- `single_bucket_cache.py` - Cache implementation with quality methods
- `preprocess_single_bucket.py` - Cache building with quality filtering
- `vasa_dataset.py` - Runtime bad window detection
- `video_tracker.py` - Tracks problematic videos

## API Usage

You can also use the cache validation programmatically:

```python
from single_bucket_cache import SingleBucketCache

cache = SingleBucketCache(cache_dir="cache_single_bucket")

# Validate entire cache
report = cache.validate_all_quality()
print(f"Pass rate: {report['quality_pass_rate']:.1%}")
print(f"Bad videos: {report['bad_videos']}")

# Check single window
is_valid, issues = cache.validate_window_quality(window_idx=0)
if not is_valid:
    print(f"Window 0 issues: {issues}")

# Remove bad video
removed = cache.invalidate_video("path/to/bad/video.mp4")
print(f"Removed {removed} windows")
```
