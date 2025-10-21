# Cache Architecture Proposal: Per-Video H5 Files

## Current Issues

1. **Monolithic H5 file corrupted** (34GB instead of 57MB)
2. **Single point of failure** - one bad write corrupts entire cache
3. **Can't process videos in parallel** - all write to same file
4. **Hard to debug** - which video caused corruption?
5. **Can't invalidate single video** - must rebuild entire cache

## Proposed Architecture

### File Structure
```
cache_single_bucket/
├── frames/                           # UNCHANGED - keep disk cache
│   └── <video_md5>/
│       └── window_X/
│           └── frame_XXX.png
├── emo_frames/                       # UNCHANGED - keep disk cache
│   └── <video_md5>/
│       └── window_X/
│           └── frame_XXX.png
├── metadata/                         # NEW - per-video H5 files
│   ├── <video_md5_1>.h5             # Contains all windows for video 1
│   ├── <video_md5_2>.h5             # Contains all windows for video 2
│   └── ...
├── video_hashes.txt                  # UNCHANGED - MD5 lookup
└── cache_index.json                  # NEW - fast lookup of available videos
```

### Per-Video H5 Structure
Each `<video_md5>.h5` contains:
```
window_0/
  ├── audio_features          [T, 768]
  ├── expression_embed        [T, 128]
  ├── theta                   [T, 25]
  ├── scale                   [T, 1]
  ├── rotation                [T, 3]
  ├── translation             [T, 3]
  ├── gaze                    [T, 2]
  ├── emotion                 [T, 8]
  ├── head_distance           [T, 1]
  ├── landmarks               [T, 68, 2]
  ├── uv_warps                [T, H, W, 2]
  ├── lip_metrics/...
  └── metadata (attrs)
window_1/
  └── ...
```

## Benefits

### 1. **Resilience**
- One corrupted video ≠ lost entire cache
- Easy to identify and delete bad video H5 file
- Can continue training with remaining videos

### 2. **Incremental Processing**
```bash
# Process videos one at a time
python preprocess_single_bucket.py --video s1/video1.mpg
python preprocess_single_bucket.py --video s1/video2.mpg

# Or resume from failures automatically
python preprocess_single_bucket.py --video-folder s1  # Skips existing
```

### 3. **Parallel Processing**
```bash
# Process 1000 videos in parallel (10 jobs)
parallel -j 10 python preprocess_single_bucket.py --video {} ::: s1/*.mpg
```

### 4. **Easy Invalidation**
```bash
# Reprocess single bad video
rm cache_single_bucket/metadata/abc123def456.h5
python preprocess_single_bucket.py --video s1/bad_video.mpg
```

### 5. **Better Debugging**
```python
# Identify which video caused corruption
for h5_file in Path('cache_single_bucket/metadata').glob('*.h5'):
    try:
        with h5py.File(h5_file, 'r') as f:
            _ = list(f.keys())  # Try to read
    except Exception as e:
        print(f"Corrupted: {h5_file} - {e}")
```

### 6. **Smaller Memory Footprint**
- Old: Load all 1000 videos' metadata to find next window
- New: Scan directory, load only needed video H5 files

## Implementation Plan

### Phase 1: New Cache Class
Create `PerVideoCache` class:
```python
class PerVideoCache:
    def __init__(self, cache_dir: Path):
        self.metadata_dir = cache_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        self.index_path = cache_dir / 'cache_index.json'

    def get_video_h5_path(self, video_md5: str) -> Path:
        return self.metadata_dir / f"{video_md5}.h5"

    def save_video_windows(self, video_md5: str, windows: List[Dict]):
        """Save all windows for a video to its H5 file"""
        h5_path = self.get_video_h5_path(video_md5)
        with h5py.File(h5_path, 'w') as f:
            for i, window in enumerate(windows):
                # Save window data...

    def load_video_windows(self, video_md5: str) -> List[Dict]:
        """Load all windows from a video's H5 file"""
        h5_path = self.get_video_h5_path(video_md5)
        if not h5_path.exists():
            return []
        # Load windows...

    def rebuild_index(self):
        """Scan metadata dir and build index.json"""
        index = {}
        for h5_path in self.metadata_dir.glob('*.h5'):
            video_md5 = h5_path.stem
            with h5py.File(h5_path, 'r') as f:
                num_windows = len([k for k in f.keys() if k.startswith('window_')])
                index[video_md5] = {
                    'num_windows': num_windows,
                    'h5_path': str(h5_path),
                    'file_size_mb': h5_path.stat().st_size / (1024**2)
                }

        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2)
```

### Phase 2: Update Dataset
Modify `VASAIntegratedDataset` to:
1. Check if per-video cache exists
2. Fall back to old monolithic cache if needed
3. Build window list from all video H5 files

### Phase 3: Update Preprocessing
Modify `preprocess_single_bucket.py` to:
1. Write to per-video H5 files
2. Update cache_index.json after each video
3. Support `--video` flag for single video processing

### Phase 4: Migration Tool
```bash
# Convert existing monolithic cache to per-video
python migrate_cache_to_per_video.py \
  --old-cache cache_single_bucket/all_windows_cache.h5 \
  --cache-dir cache_single_bucket
```

## File Size Expectations

For 1000 videos with ~1-2 windows each:
- **Per-video H5**: ~50KB - 500KB each (depending on window count)
- **Total metadata**: 50MB - 500MB
- **Frames on disk**: ~10-50GB (already efficient as PNG)
- **Total cache**: 10-50GB

## Performance Comparison

| Operation | Monolithic H5 | Per-Video H5 |
|-----------|--------------|--------------|
| **Add 1 video** | Rewrite 36GB file | Write 500KB file |
| **Corruption impact** | Lose 1000 videos | Lose 1 video |
| **Parallel processing** | ❌ Impossible | ✅ Easy |
| **Memory to scan** | Load 36GB metadata | Scan directory |
| **Debug bad video** | Unclear | Clear (filename = MD5) |
| **Reprocess 1 video** | Rebuild 36GB | Delete 500KB, reprocess |

## Rollout Plan

1. ✅ Delete corrupted `all_windows_cache.h5`
2. ✅ Create `per_video_cache.py` with new class
3. ✅ Update `preprocess_single_bucket.py` to use per-video cache
4. ✅ Update `vasa_dataset.py` to load from per-video cache
5. ✅ Process all 1000 videos (can run in parallel!)
6. ✅ Keep old code path for backward compatibility

## Backward Compatibility

```python
# Auto-detect which cache type
if (cache_dir / 'metadata').exists() and (cache_dir / 'metadata').is_dir():
    cache = PerVideoCache(cache_dir)  # New per-video system
else:
    cache = SingleBucketCache(cache_dir)  # Old monolithic system
```

## Conclusion

**Recommendation**: Migrate to per-video H5 files

**Effort**: ~2-4 hours of implementation
**Benefit**:
- ✅ No more 36GB corruption disasters
- ✅ Parallel preprocessing (10x faster for 1000 videos)
- ✅ Easy debugging and invalidation
- ✅ Production-grade robustness
