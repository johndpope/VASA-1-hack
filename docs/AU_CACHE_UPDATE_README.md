# Action Unit Cache Update Guide

## Overview
This guide explains how AU ground truth data is managed in the cache system and how to update existing caches.

---

## ✅ Lambda Parameters - CONFIRMED

Both config files already have AU lambda configured:

**overfit_config.yaml (line 228):**
```yaml
lambda_aux_au: 1.0  # Action Unit prediction (self-supervised)
```

**vasa_config.yaml (line 228):**
```yaml
lambda_aux_au: 1.0  # Action Unit prediction (self-supervised)
```

✅ **No changes needed to configs!**

---

## 🔄 Cache System Behavior

### For New Cache Files:
When you run training with updated code, **AU ground truth is automatically saved** to new cache files:

**Data Flow:**
```
Video Processing
    ↓
AU Extraction (vasa_dataset.py:3057-3060)
    ↓
window_data['au_gt'] = au_gt  # Added to window data
    ↓
_save_window_to_cache() (vasa_dataset.py:1672)
    ↓
Automatically saves ALL keys in window_data
    ↓
au_gt saved to H5 cache ✅
```

**Location:** `cache_per_video/<video_hash>/metadata.h5`

**Dataset Key:** `window_X/au_gt` with shape `[8, 16]`

---

## 📦 For Existing Cache Files:

Existing cache files (created before AU implementation) **do NOT have AU data**. You have two options:

### Option 1: Clear Cache (Simple)
Delete existing caches and let them rebuild with AU data:

```bash
rm -rf cache_per_video/*
python train_overfit.py  # Will rebuild with AU GT
```

**Pros:**
- Simple, guaranteed to work
- Fresh cache with all latest features

**Cons:**
- Takes time to rebuild (10-20 minutes)
- Loses any manual cache fixes

---

### Option 2: Update Existing Caches (Fast)
Use the AU upsert script to add AU data to existing caches:

```bash
./upsert_au.sh
# OR
python upsert_au_gt.py --cache-dir cache_per_video
```

**Pros:**
- Faster than rebuilding (~1-2 minutes)
- Preserves existing cache data

**Cons:**
- More complex
- Requires videos to still exist

---

## 🚀 Quick Start: Update Existing Caches

### Interactive Mode (Recommended):
```bash
./upsert_au.sh
```

The script will:
1. Count cache files
2. Ask for confirmation
3. Extract AUs from videos
4. Update cache files with AU GT
5. Show summary

### Command Line Mode:
```bash
# Update all caches
python upsert_au_gt.py --cache-dir cache_per_video

# Force update (overwrite existing AU data)
python upsert_au_gt.py --force

# Update only specific videos
python upsert_au_gt.py --video-filter scene6
```

---

## 📊 What the Upsert Script Does

1. **Finds cache files:** Scans `cache_per_video/*/metadata.h5`
2. **For each cache:**
   - Reads video path from cache metadata
   - Extracts frames from original video
   - Computes AU intensities using ActionUnitExtractor
   - Saves `au_gt` [8, 16] to cache
3. **Reports:** Shows success/failure count

**Output:**
```
Processing: metadata.h5
  Video: junk/video.mp4
  Windows: 316
  ✅ Updated 316/316 windows

SUMMARY
================================================================================
Total cache files: 6
Successfully updated: 6
Failed: 0
```

---

## 🔍 Verify Cache Has AU Data

Check if a cache file contains AU ground truth:

```python
import h5py

cache_path = "cache_per_video/<hash>/metadata.h5"
with h5py.File(cache_path, 'r') as f:
    # Check first window
    if 'au_gt' in f['window_0']:
        au_gt = f['window_0']['au_gt'][:]
        print(f"✅ AU GT found! Shape: {au_gt.shape}")  # Should be [8, 16]
    else:
        print("❌ No AU GT in cache")
```

Or use this one-liner:
```bash
python -c "import h5py; f=h5py.File('cache_per_video/*/metadata.h5','r'); print('au_gt' in f['window_0'])"
```

---

## ⚠️ Troubleshooting

### "Video not found" error:
**Problem:** Original video files moved or deleted

**Solution:**
- Move videos back to original location, OR
- Clear cache and rebuild: `rm -rf cache_per_video/*`

### "No frames extracted" error:
**Problem:** Video file corrupted or unreadable

**Solution:**
- Check video plays with: `ffplay video.mp4`
- Re-encode if needed: `ffmpeg -i old.mp4 -c copy new.mp4`
- Or skip that cache file

### Script runs but training still fails:
**Problem:** Cache not loading AU data correctly

**Solution:**
1. Check cache has au_gt: (see "Verify Cache" above)
2. Check dataset loads it:
```python
from vasa_dataset import VASAIntegratedDataset
dataset = VASAIntegratedDataset(...)
window = dataset[0]
print('au_gt' in window)  # Should be True
```
3. If still issues, clear cache and rebuild

---

## 🔄 Cache Update Workflow

### Starting Fresh (Recommended for first AU training):
```bash
# 1. Clear existing caches
rm -rf cache_per_video/*

# 2. Start training (will build new caches with AU GT)
./safe-train.sh

# 3. Monitor training
# Check WandB for aux_au loss and visualizations
```

### Updating Existing Caches:
```bash
# 1. Run upsert script
./upsert_au.sh

# 2. Verify update
python -c "import h5py; f=h5py.File('cache_per_video/*/metadata.h5','r'); print('✅ AU GT exists!' if 'au_gt' in f['window_0'] else '❌ No AU GT')"

# 3. Start training
./safe-train.sh
```

---

## 📝 Cache File Structure

**Before AU Implementation:**
```
cache_per_video/
└── <video_hash>/
    └── metadata.h5
        ├── @video_path
        ├── @num_windows
        └── window_0/
            ├── identity_frame [3, H, W]
            ├── theta [50, 3, 4]
            ├── expression_embed [50, 128]
            ├── audio_features [50, 768]
            ├── phoneme_gt [8]
            └── ... (other data)
```

**After AU Update:**
```
cache_per_video/
└── <video_hash>/
    └── metadata.h5
        ├── @video_path
        ├── @num_windows
        └── window_0/
            ├── identity_frame [3, H, W]
            ├── theta [50, 3, 4]
            ├── expression_embed [50, 128]
            ├── audio_features [50, 768]
            ├── phoneme_gt [8]
            ├── au_gt [8, 16]  # ← NEW!
            └── ... (other data)
```

---

## 🎯 Expected Results

### After Cache Update:
1. **Training logs:** Should see AU extraction logs
2. **Loss metrics:** `aux_au` loss appears in logs/WandB
3. **Visualizations:** `visuals/action_units` appears in WandB
4. **No errors:** Dataset loading works without warnings

### Training Behavior:
```
Epoch 1: aux_au = 0.25  # Initial loss
Epoch 10: aux_au = 0.18  # Decreasing
Epoch 50: aux_au = 0.08  # Getting better
Epoch 100: aux_au = 0.04  # Target achieved! ✅
```

---

## 📚 Related Documentation

- **AU_IMPLEMENTATION_FINAL.md** - Complete AU implementation summary
- **AU_IMPLEMENTATION_GUIDE.md** - Detailed implementation patterns
- **upsert_phoneme_gt.py** - Similar script for phonemes (reference)

---

## 🚦 Quick Decision Guide

**Choose Clear Cache if:**
- First time running with AU implementation
- Have time to wait for rebuild (10-20 min)
- Want guaranteed fresh cache
- Videos haven't changed location

**Choose Update Existing if:**
- Already have recent caches
- Want to save time (1-2 min vs 10-20 min)
- Videos still in original locations
- Made manual cache modifications

---

## ✅ Checklist

Before training with AU loss:
- [ ] Lambda parameters in configs (already done ✅)
- [ ] Either clear cache OR run upsert script
- [ ] Verify cache has `au_gt` data
- [ ] Start training with `./safe-train.sh`
- [ ] Check WandB for AU visualizations
- [ ] Monitor `aux_au` loss decreasing

---

**Status:** Ready to update caches and train! 🚀

**Recommendation:** For first AU training run, use **Option 1 (Clear Cache)** for simplicity. For subsequent runs with new videos, the upsert script will be faster.
