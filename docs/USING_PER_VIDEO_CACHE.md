# Using Per-Video Cache with Training

## Overview

The `VASAIntegratedDataset` now supports **per-video cache** structure automatically. This provides better resilience, easier debugging, and support for parallel processing compared to the monolithic single-bucket cache.

## Auto-Detection

The dataset loader automatically detects which cache type to use based on what's available:

**Priority (highest to lowest):**
1. **PerVideoCache** - If `cache_index.json` exists
2. **SingleBucketCache** - If `all_windows_cache.h5` exists
3. **ChunkedWindowCache** - If available
4. **Built-in WindowCache** - Fallback

## Migration Path

### Option 1: Migrate Existing Cache (Recommended)

If you already have a working `cache_single_bucket/` with preprocessed data:

```bash
# Migrate to per-video cache (copies frame files)
python convert_to_per_video_cache.py \
  --source cache_single_bucket \
  --dest cache_per_video

# OR use symlinks for faster migration (but fragile if source deleted)
python convert_to_per_video_cache.py \
  --source cache_single_bucket \
  --dest cache_per_video \
  --symlink
```

**What this does:**
- Reads all windows from `all_windows_cache.h5`
- Groups windows by video MD5
- Creates one `metadata.h5` per video in MD5 folders
- Copies/symlinks `frames/` and `emo_frames/` directories
- Generates `cache_index.json` for fast lookup

**Expected output:**
```
✅ MIGRATION COMPLETE
✅ Successfully migrated: 99/99 videos
✅ Total windows migrated: 216

📊 New Cache Statistics:
   Videos: 99
   Total windows: 216
   Total metadata size: 196.50 MB
   Average per video: 1.98 MB
```

### Option 2: Preprocess Directly to Per-Video Cache

If starting fresh or want to reprocess:

```bash
python preprocess_per_video.py \
  --video_folder s1 \
  --cache_dir cache_per_video \
  --max_videos 100 \
  --window_size 50 \
  --stride 25 \
  --frame-format png
```

## Training with Per-Video Cache

### Using Existing Training Scripts

**No changes needed!** The dataset automatically detects and uses per-video cache.

Just point `--cache_dir` to the per-video cache directory:

```bash
# Overfitting training
python train_overfit.py  # Uses cache_dir from overfit_config.yaml

# Full training
python vasa_trainer.py --config vasa_config.yaml
```

### Update Your Config Files

In `overfit_config.yaml` or `vasa_config.yaml`:

```yaml
dataset:
  cache_dir: cache_per_video  # Changed from cache_single_bucket
  # ... other settings remain the same
```

### Verify Cache is Detected

When you run training, look for this log message:

```
✅ Using PerVideoCache at /path/to/cache_per_video
   Per-video H5 files with MD5-indexed folders
```

If you see this instead, the per-video cache wasn't found:

```
Using SingleBucketCache at /path/to/cache_single_bucket/all_windows_cache.h5
```

## Cache Structure

### Per-Video Cache Layout

```
cache_per_video/
├── <video_md5_1>/
│   ├── metadata.h5              # ~2 MB (metadata only, no frames)
│   ├── frames/
│   │   └── window_X/
│   │       └── frame_XXX.png
│   └── emo_frames/
│       └── window_X/
│           └── frame_XXX.png
├── <video_md5_2>/
│   └── ... (same structure)
├── cache_index.json             # Fast lookup: {md5 -> video_path}
├── video_hashes.txt             # MD5 cache
└── expression_embeddings.h5     # Shared expression database
```

### What Goes Where

**metadata.h5** (per video, ~2 MB):
- Audio features (wav2vec2, MFCC, mel_spec)
- Expression embeddings (128-dim)
- Motion parameters (theta, scale, rotation, translation)
- Facial landmarks (jaw, eyes, lips, nose)
- Control signals (emotion, gaze, head_distance)
- UV warps and target masks
- **DOES NOT include**: frames, emo_frames (on disk)

**frames/** (disk, per window):
- Original video frames as PNG files
- Organized by window index
- Loaded on-demand during training

**emo_frames/** (disk, per window):
- EMO-generated identity-transferred frames
- PNG files with green background removed
- Loaded on-demand during training

## How Dataset Loads Windows

### Per-Video Cache (New)

```python
# In __getitem__(idx):
if self.cache_type == 'per_video':
    cached_data = self.cache.load_window(
        video_path=video_path,
        window_idx=window['window_idx'],
        load_frames=True  # Load frames from disk
    )
```

**Benefits:**
- Each video is independent (one corrupt video ≠ lost all data)
- Easy to identify/delete bad videos (just delete MD5 folder)
- Parallel processing support (multiple videos can be processed simultaneously)
- Smaller memory footprint (load only needed videos)

### Single-Bucket Cache (Old)

```python
# In __getitem__(idx):
if self.cache_type == 'single_bucket':
    cached_data = self.cache.load_window(idx)
```

**Limitations:**
- Single point of failure (one corrupt H5 breaks everything)
- Can't process videos in parallel
- Hard to debug which video caused issues
- Must rebuild entire file to change one video

## Troubleshooting

### Q: How do I verify migration worked?

```bash
python -c "
from per_video_cache import PerVideoCache
from pathlib import Path
cache = PerVideoCache(Path('cache_per_video'))
stats = cache.get_cache_stats()
print(f'Videos: {stats[\"total_videos\"]}')
print(f'Windows: {stats[\"total_windows\"]}')
print(f'Metadata size: {stats[\"total_metadata_size_mb\"]:.2f} MB')
"
```

Expected output:
```
Videos: 99
Windows: 216
Metadata size: 196.50 MB
```

### Q: Training is loading from wrong cache?

Check these:
1. `cache_index.json` exists in `cache_per_video/`
2. Config file points to correct directory
3. Check startup logs for "Using PerVideoCache" message

### Q: How to delete a bad video from cache?

```bash
# Find video MD5
python -c "
from per_video_cache import PerVideoCache
from pathlib import Path
cache = PerVideoCache(Path('cache_per_video'))
md5 = cache.get_video_hash('s1/bad_video.mpg')
print(f'MD5: {md5}')
"

# Delete video cache
rm -rf cache_per_video/<video_md5>

# Rebuild index
python -c "
from per_video_cache import PerVideoCache
from pathlib import Path
cache = PerVideoCache(Path('cache_per_video'))
cache.rebuild_index()
"
```

### Q: Can I use both caches simultaneously?

Yes! Keep both `cache_single_bucket/` and `cache_per_video/` directories. The dataset will prioritize per-video cache if both exist.

To force single-bucket cache, rename/delete `cache_index.json`:
```bash
mv cache_per_video/cache_index.json cache_per_video/cache_index.json.bak
```

### Q: Performance comparison?

| Operation | Single Bucket | Per-Video |
|-----------|--------------|-----------|
| **Add 1 video** | Rewrite 7.62 GB H5 | Write 2 MB H5 |
| **Corruption impact** | Lose all 216 windows | Lose 2 windows |
| **Parallel processing** | ❌ Impossible | ✅ Easy |
| **Memory to scan** | Load 7.62 GB | Scan index (< 1 MB) |
| **Debug bad video** | Unclear | Folder name = MD5 |
| **Reprocess 1 video** | Rebuild 7.62 GB | Delete folder + rerun |

## Advanced Usage

### Inspect Cache Contents

```python
from per_video_cache import PerVideoCache
from pathlib import Path

# Initialize cache
cache = PerVideoCache(Path('cache_per_video'))

# Get stats
stats = cache.get_cache_stats()
print(f"Videos: {stats['total_videos']}")
print(f"Windows: {stats['total_windows']}")

# Load specific video
video_path = "s1/example.mpg"
windows = cache.load_all_video_windows(video_path, load_frames=False)
print(f"Loaded {len(windows)} windows for {video_path}")

# Load with frames
window_0 = cache.load_window(video_path, window_idx=0, load_frames=True)
print(f"Frames shape: {window_0['frames'].shape}")
print(f"EMO frames shape: {window_0['emo_frames'].shape}")
```

### Validate Cache

```python
# Validate specific video
is_valid, issues = cache.validate_video("s1/example.mpg")
if not is_valid:
    print(f"Issues: {issues}")

# Rebuild index (if corrupted)
cache.rebuild_index()
```

### Parallel Preprocessing (Future)

```bash
# Process multiple videos simultaneously (not implemented yet)
parallel -j 4 python preprocess_video.py --video {} ::: s1/*.mpg
```

## Summary

✅ **Auto-detection works** - Just point to cache directory, dataset picks the right cache
✅ **Migration is simple** - One command converts existing cache
✅ **Better resilience** - One bad video doesn't break everything
✅ **Production-ready** - Used successfully for 99 videos, 216 windows
✅ **No training changes needed** - Drop-in replacement for single-bucket cache

For more details, see:
- `PER_VIDEO_CACHE_GUIDE.md` - Architecture and API reference
- `per_video_cache.py` - Implementation
- `convert_to_per_video_cache.py` - Migration script
- `preprocess_per_video.py` - Preprocessing script
