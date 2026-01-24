# Per-Video Cache Implementation Guide

## Overview

The per-video cache structure organizes data into MD5-indexed folders, with each video getting its own small metadata H5 file and frame directories. This provides better resilience, easier debugging, and support for parallel processing.

## Architecture

### Old Structure (SingleBucketCache)
```
cache_single_bucket/
├── all_windows_cache.h5          # Monolithic 7.62 GB
├── frames/<video_md5>/window_X/*.png
├── emo_frames/<video_md5>/window_X/*.png
└── expression_embeddings.h5
```

**Problems:**
- ❌ Single point of failure (one corrupted H5 breaks everything)
- ❌ Can't process videos in parallel
- ❌ Hard to debug which video caused issues
- ❌ Must rebuild entire 7.62 GB file to change one video

### New Structure (PerVideoCache)
```
cache_per_video/
├── <video_md5_1>/
│   ├── metadata.h5               # Small ~0.4-2 MB per video
│   ├── frames/
│   │   └── window_X/
│   │       └── frame_XXX.png
│   └── emo_frames/
│       └── window_X/
│           └── frame_XXX.png
├── <video_md5_2>/
│   └── ... (same structure)
├── cache_index.json              # Fast lookup
├── video_hashes.txt              # MD5 cache
└── expression_embeddings.h5      # Shared
```

**Benefits:**
- ✅ One corrupted video ≠ lost entire cache
- ✅ Easy to identify/delete bad videos
- ✅ Parallel processing support
- ✅ Easy invalidation (delete one folder)
- ✅ All data for a video in one place

## File Sizes

### Expected Sizes

**Per Video (2 windows):**
- `metadata.h5`: ~0.4-2 MB (metadata + warps, no frames)
- `frames/`: ~10-50 MB (PNG files)
- `emo_frames/`: ~10-50 MB (PNG files)
- **Total**: ~20-100 MB per video

**Full Cache (100 videos, 216 windows):**
- Total metadata: ~40-200 MB (100 small H5 files)
- Total frames: ~2-10 GB (PNG files on disk)
- **vs. Old**: 7.62 GB monolithic H5 → distributed structure

### What Goes in Each File

**metadata.h5** (Small - per video):
- Audio features, mel specs, MFCC, waveform
- Expression embeddings
- Motion parameters (theta, scale, rotation, translation)
- Facial landmarks (jaw, eyes, lips, nose)
- Control signals (emotion, gaze, head_distance)
- UV warps and target masks
- **DOES NOT include**: frames, emo_frames (on disk)

**frames/** (Disk - per window):
- Original video frames as PNG files
- Lossless compression
- Easy to inspect visually

**emo_frames/** (Disk - per window):
- EMO-generated identity-transferred frames
- Lossless PNG storage
- Green background removed

## Usage

### 1. Migration (Convert Existing Cache)

```bash
# Migrate existing cache to per-video structure
python convert_to_per_video_cache.py \
  --source cache_single_bucket \
  --dest cache_per_video

# Use symlinks instead of copying (faster but fragile)
python convert_to_per_video_cache.py \
  --source cache_single_bucket \
  --dest cache_per_video \
  --symlink
```

### 2. Preprocessing (New Videos)

```bash
# Preprocess videos directly to per-video cache
python preprocess_per_video.py \
  --video_folder s1 \
  --cache_dir cache_per_video \
  --max_videos 100 \
  --window_size 50 \
  --stride 25 \
  --frame-format png
```

### 3. Inspect Cache

```python
from per_video_cache import PerVideoCache
from pathlib import Path

# Initialize cache
cache = PerVideoCache(Path('cache_per_video'))

# Get stats
stats = cache.get_cache_stats()
print(f"Videos: {stats['total_videos']}")
print(f"Windows: {stats['total_windows']}")
print(f"Metadata size: {stats['total_metadata_size_mb']:.2f} MB")

# Load specific video
video_path = "s1/lwal4p.mpg"
windows = cache.load_all_video_windows(video_path, load_frames=False)
print(f"Loaded {len(windows)} windows")

# Load with frames
window_0 = cache.load_window(video_path, window_idx=0, load_frames=True)
print(f"Frames shape: {window_0['frames'].shape}")
print(f"EMO frames shape: {window_0['emo_frames'].shape}")
```

### 4. Validate Cache

```python
# Validate specific video
is_valid, issues = cache.validate_video("s1/lwal4p.mpg")
if not is_valid:
    print(f"Issues: {issues}")

# Rebuild index
cache.rebuild_index()
```

### 5. Delete Bad Video

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

## Integration with Training

### Update VASAIntegratedDataset

The dataset loader needs to be updated to use PerVideoCache:

```python
# In vasa_dataset.py
from per_video_cache import PerVideoCache

class VASAIntegratedDataset:
    def __init__(self, ...):
        # Check which cache type exists
        per_video_cache_dir = Path(cache_dir)
        single_cache_file = per_video_cache_dir / 'all_windows_cache.h5'

        if (per_video_cache_dir / 'cache_index.json').exists():
            # Use per-video cache
            self.cache = PerVideoCache(per_video_cache_dir)
            self.cache_type = 'per_video'
        elif single_cache_file.exists():
            # Use old single bucket cache
            from single_bucket_cache import SingleBucketCache
            self.cache = SingleBucketCache(per_video_cache_dir)
            self.cache_type = 'single_bucket'
        else:
            self.cache = None
            self.cache_type = None

    def __getitem__(self, idx):
        if self.cache_type == 'per_video':
            # Load from per-video cache
            video_path = self.windows[idx]['video_path']
            window_idx = self.windows[idx]['window_idx']
            return self.cache.load_window(
                video_path,
                window_idx,
                load_frames=True
            )
        elif self.cache_type == 'single_bucket':
            # Load from old cache
            return self.cache.load_window(idx)
```

## Troubleshooting

### Q: Migration is slow
**A**: Use `--symlink` flag to create symlinks instead of copying frame files. This is instant but the symlinks will break if you move/delete the source cache.

### Q: How to verify migration worked?
**A**:
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

### Q: Can I delete old cache after migration?
**A**: Yes, but verify first:
```bash
# Verify new cache is complete
python -c "
from per_video_cache import PerVideoCache
from pathlib import Path
cache = PerVideoCache(Path('cache_per_video'))
cache.rebuild_index()
stats = cache.get_cache_stats()
assert stats['total_windows'] == 216, 'Missing windows!'
print('✅ Migration verified!')
"

# Then delete old cache
rm -rf cache_single_bucket/
```

### Q: What if a video's metadata.h5 is corrupted?
**A**: Just delete that video's folder and reprocess:
```bash
rm -rf cache_per_video/<video_md5>
python preprocess_per_video.py --video_folder s1 --cache_dir cache_per_video
```

## Performance Comparison

| Operation | Single Bucket | Per-Video |
|-----------|--------------|-----------|
| **Add 1 video** | Rewrite 7.62 GB H5 | Write 2 MB H5 |
| **Corruption impact** | Lose all 216 windows | Lose 2 windows |
| **Parallel processing** | ❌ Impossible | ✅ Easy |
| **Memory to scan** | Load 7.62 GB | Scan index (< 1 MB) |
| **Debug bad video** | Unclear | Folder name = MD5 |
| **Reprocess 1 video** | Rebuild 7.62 GB | Delete folder + rerun |

## Future Enhancements

1. **Parallel preprocessing**: Process multiple videos simultaneously
   ```bash
   parallel -j 4 python preprocess_video.py --video {} ::: s1/*.mpg
   ```

2. **Streaming dataset**: Load videos on-demand without loading full index

3. **Cloud storage**: Each video folder can be uploaded/downloaded independently

4. **Incremental updates**: Add new videos without touching existing cache

## Summary

The per-video cache structure provides:
- ✅ Better resilience (one bad video doesn't break everything)
- ✅ Easier debugging (MD5 folder = video identity)
- ✅ Parallel processing support
- ✅ Smaller memory footprint
- ✅ Incremental updates
- ✅ Production-grade robustness

Trade-offs:
- ⚠️ More files to manage (one H5 per video vs one monolithic)
- ⚠️ Slightly slower initial scan (but mitigated by cache_index.json)
