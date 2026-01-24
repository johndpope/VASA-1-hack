# Bad Videos Cleanup Summary

## Videos Removed (10 total, ~4.5 MB)

### Reason: `audio_segment` is not defined error
1. `junk2/11.mp4` (MD5: `5f907923981049b2bb99d70fba1e2535`) - 11 windows
2. `junk2/videovideo-hOQAsNg8aw-scene3_scene3.mp4` (MD5: `e797e7e733254c21cc7352817986244b`) - 3 windows
3. `junk2/videovideoeI2V8Bd5X9s-scene6_scene1.mp4` (MD5: `ca8b16654d2a62bebc40d94348f311b8`) - 3 windows
4. `junk2/videovideozqKyByqbSs8-scene18_scene2.mp4` (MD5: `3f471aee7a73b5ea18e4b76d960db6b3`) - 3 windows
5. `junk2/videovideo_0MTaef81jQQ-scene23_scene4.mp4` (MD5: `aedf5435b58a3e95fc7eab47481bd85a`) - 3 windows
6. `junk2/videovideokUUhJtbrIwU-scene241_scene6.mp4` (MD5: `8fb51a416cb6b06215dcea432de15ef1`) - 3 windows
7. `junk2/videovideoIfai_9UFDSU-scene1_scene7.mp4` (MD5: `9e2ca41660b2674f33d0d83df5a8cf15`) - 3 windows
8. `junk2/videovideo_dvrizT3014o-scene1_scene20.mp4` (MD5: `b4ac40ce8c70ac0efcefb1627141413c`) - 3 windows
9. `junk2/videovideoAl5MFMZvpY4-scene2_scene1.mp4` (MD5: `33651b1cb78f7c3f37d57fac92085166`) - 3 windows

### Reason: Face detection failed
10. `junk2/videovideoTwqES24jgRA-scene32_scene10.mp4` (MD5: `c49b30d5fcc949ff1e10e97797da1369`) - 0 windows (not cached)

## Impact

### H5 Cache
- **35 windows** will be removed from H5 cache when rebuilt
- **427 windows** will remain (462 - 35 = 427)

### Disk Cache
- **18 cache directories** removed:
  - 9 from `frames/` directory
  - 9 from `emo_frames/` directory

### Dataset
- **10 videos** deleted from `junk2/`
- **~4.5 MB** disk space freed

## What Was Done

1. ✅ **Computed MD5 hashes** for all bad videos
2. ✅ **Identified affected windows** in H5 cache (35 total)
3. ✅ **Removed disk caches** (frames + emo_frames)
4. ✅ **Deleted video files** from junk2/
5. ⏳ **H5 cache rebuild needed** (next step)

## Next Steps

### 1. Rebuild H5 Cache (Required)

```bash
# Delete old H5 cache
rm cache_single_bucket/all_windows_cache.h5

# Rebuild with only valid videos
python preprocess_single_bucket.py --cache-frames --cache-emo-frames
```

**Expected result:**
- ~427 windows cached (down from 462)
- All windows will have emo_frames
- No bad videos in cache

### 2. Update Video Tracking Database (Optional)

If you have a `bad_videos/` tracking system, update it to mark these videos as deleted:

```bash
# Example: Update invalid_videos.txt
echo "junk2/11.mp4" >> bad_videos/invalid_videos.txt
echo "junk2/videovideo-hOQAsNg8aw-scene3_scene3.mp4" >> bad_videos/invalid_videos.txt
# ... etc
```

### 3. Verify Clean Cache

After rebuilding:

```bash
# Check H5 cache
python check_srt_emo.py

# Check disk caches
find cache_single_bucket/emo_frames -name "window_*" -type d | wc -l

# Should match H5 window count
```

## Error Analysis

### `audio_segment` is not defined
**Cause**: Variable scoping issue in face attribute extraction
**Effect**: Windows couldn't be processed, resulting in incomplete data
**Solution**: Videos removed, preprocessing will skip them automatically

### Face detection failed
**Cause**: No face landmarks detected in frames (corrupted video or non-face content)
**Effect**: No windows were cached for this video
**Solution**: Video removed, no cache cleanup needed

## Files Created

1. `clean_bad_videos.py` - Automated cleanup script
2. `BAD_VIDEOS_CLEANUP_SUMMARY.md` - This file
3. `/tmp/delete_bad_videos.sh` - Video deletion script (executed)

## MD5 Hash Reference

For future reference, these MD5 hashes correspond to deleted videos:

```
5f907923981049b2bb99d70fba1e2535  junk2/11.mp4
e797e7e733254c21cc7352817986244b  junk2/videovideo-hOQAsNg8aw-scene3_scene3.mp4
ca8b16654d2a62bebc40d94348f311b8  junk2/videovideoeI2V8Bd5X9s-scene6_scene1.mp4
3f471aee7a73b5ea18e4b76d960db6b3  junk2/videovideozqKyByqbSs8-scene18_scene2.mp4
aedf5435b58a3e95fc7eab47481bd85a  junk2/videovideo_0MTaef81jQQ-scene23_scene4.mp4
8fb51a416cb6b06215dcea432de15ef1  junk2/videovideokUUhJtbrIwU-scene241_scene6.mp4
9e2ca41660b2674f33d0d83df5a8cf15  junk2/videovideoIfai_9UFDSU-scene1_scene7.mp4
b4ac40ce8c70ac0efcefb1627141413c  junk2/videovideo_dvrizT3014o-scene1_scene20.mp4
33651b1cb78f7c3f37d57fac92085166  junk2/videovideoAl5MFMZvpY4-scene2_scene1.mp4
c49b30d5fcc949ff1e10e97797da1369  junk2/videovideoTwqES24jgRA-scene32_scene10.mp4
```

If you see these MD5s in any cache directories, they can be safely deleted.
