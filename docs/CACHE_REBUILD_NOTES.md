# Cache Rebuild Process

## Issue Detected
Cache corruption errors:
- `Link iteration failed (free block size is zero?)`
- `Link iteration failed (bad local heap signature)`
- `Unable to synchronously open object (bad object header version number)`

This typically happens from:
1. Interrupted writes (Ctrl+C during preprocessing)
2. Disk space issues
3. Process crashes during cache building
4. Concurrent writes to same file

## Solution Applied

### 1. Deleted Corrupted Cache
```bash
rm -f cache_single_bucket/all_windows_cache.h5
```

### 2. Rebuilding with Quality Filtering
```bash
python preprocess_single_bucket.py \
  --video_folder junk \
  --cache_dir cache_single_bucket \
  --max_videos 1 \
  --window_size 50 \
  --stride 25
```

**Quality filters applied** (from preprocess_single_bucket.py lines 112-140):
- UV warp magnitude < 0.15 → FILTERED
- Expression std < 0.01 → FILTERED
- Valid landmarks < 50% → FILTERED

### 3. Expected Output

The rebuild process will:
1. Load volumetric avatar model (~1 min)
2. Process video windows one by one
3. Filter out bad quality windows automatically
4. Save metadata showing:
   - `total_windows`: Total windows found
   - `successful_windows`: Windows that passed quality checks
   - `quality_filtered_windows`: Windows rejected for poor quality
   - `failed_windows`: Windows that errored during processing

### 4. Monitoring Progress

Check logs with:
```bash
tail -f cache_rebuild.log
```

Or monitor the running process:
```bash
ps aux | grep preprocess_single_bucket
```

### 5. When Complete

You should see:
```
✅ Successfully cached X windows
📈 Processed Y new windows
⚠️ Filtered Z low-quality windows so far...
✅ Cache validation passed!
```

Then validate:
```bash
python validate_cache.py --validate-only
```

Expected: 100% pass rate (all bad windows filtered during build)

### 6. Resume Training

After cache is rebuilt:
```bash
./train.sh  # Option 1 for overfitting
```

Training will now:
- Load clean windows from cache
- No corruption errors
- Model learns from good quality ground truth data
- UV warp magnitude loss will teach proper warp scales

## Prevention

To avoid future corruption:
1. Don't Ctrl+C during cache building
2. Let `preprocess_single_bucket.py` complete fully
3. Use `--no-resume` flag to rebuild from scratch if needed
4. Monitor disk space (cache is ~1.6GB for 1 video)

## Troubleshooting

If rebuild fails:
1. Check disk space: `df -h`
2. Check permissions: `ls -la cache_single_bucket/`
3. Try with fewer videos: `--max_videos 1`
4. Check model checkpoint exists: `ls -lh logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth`
