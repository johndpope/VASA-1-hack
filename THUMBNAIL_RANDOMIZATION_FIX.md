# Training Thumbnail Randomization Fix

## Problem

The `visuals/training_thumbnail` in WandB was always showing the same video/identity combination:
- Always used batch index `[0]` (first video in the batch)
- Always showed the same static identity image (from config)
- User has many videos in training but thumbnails never changed

## Root Cause

The thumbnail generation logic had hardcoded batch index selection:
1. `window_idx == 0` - Always first window (correct for simplicity)
2. `original_frames[:, frame_idx]` - Used all batch items but later selected `[0]`
3. `self.identity_image` - Static identity image from config

## Solution

### 1. Randomized Batch Item Selection

**Location**: vasa_trainer.py:1754-1757

Added random batch item selection at the start of thumbnail generation:
```python
# Randomize batch item selection to show variety across different videos
B = motion_data['theta'].shape[0] if 'theta' in motion_data else 1
batch_item_idx = random.randint(0, B - 1) if B > 1 else 0
logger.info(f"🎲 Selected batch item {batch_item_idx}/{B-1} for thumbnail (randomized to show different videos)")
```

### 2. Updated Target Frame Extraction

**Location**: vasa_trainer.py:1770

Changed from using all batch items to selecting the random one:
```python
# Before:
single_frame_target = original_frames[:, frame_idx:frame_idx+1]

# After:
single_frame_target = original_frames[batch_item_idx:batch_item_idx+1, frame_idx:frame_idx+1]
```

### 3. Updated Source Image Extraction

**Location**: vasa_trainer.py:1776-1781

Updated identity/source image selection:
```python
if self.identity_image is not None:
    # For static identity image, still show it (left panel), but use random video for target/predicted
    source_img = self.identity_image.to(self.accelerator.device)  # [1, C, H, W]
else:
    # Use first frame of the randomly selected batch item
    source_img = target_frames[batch_item_idx:batch_item_idx+1, 0]  # [1, C, H, W]
```

### 4. Updated Motion Parameter Extraction

**Location**: vasa_trainer.py:1789-1790

Changed motion parameter extraction to use random batch item:
```python
# Before:
single_motion[key] = stored_outputs[key][:, frame_idx:frame_idx+1].to(device)

# After:
single_motion[key] = stored_outputs[key][batch_item_idx:batch_item_idx+1, frame_idx:frame_idx+1].to(device)
```

### 5. Updated EMO Frame Extraction

**Location**: vasa_trainer.py:1835-1838

Changed EMO frame batch indexing:
```python
# Before:
if emo_frames.dim() == 5:  # [B, num_keyframes, C, H, W]
    emo_frames = emo_frames[window_idx]  # Wrong: window_idx is for sequence

# After:
if emo_frames.dim() == 5:  # [B, num_keyframes, C, H, W]
    emo_frames = emo_frames[batch_item_idx]  # Correct: use random batch item
```

### 6. Updated On-the-Fly EMO Generation

**Location**: vasa_trainer.py:1863-1870

Fixed motion extraction for EMO generation:
```python
# Before:
frame_motion['theta'] = theta_tensor[:, frame_idx:frame_idx+1].to(self.device)

# After:
frame_motion['theta'] = theta_tensor[batch_item_idx:batch_item_idx+1, frame_idx:frame_idx+1].to(self.device)
```

### 7. Updated Emotion Label Extraction

**Location**: vasa_trainer.py:1906-1918

Added support for batched emotion labels:
```python
# emotion_label could be batched [B, T] or just [T]
emotion_labels = window['emotion_label']
if isinstance(emotion_labels, list):
    # Check if it's batched (list of lists)
    if len(emotion_labels) > 0 and isinstance(emotion_labels[0], list):
        # Batched: [B, T], extract for random batch item
        if batch_item_idx < len(emotion_labels) and frame_idx < len(emotion_labels[batch_item_idx]):
            emotion_label_target = emotion_labels[batch_item_idx][frame_idx]
```

### 8. Enhanced Caption with Video Info

**Location**: vasa_trainer.py:1937-1956

Added video name and batch item info to caption:
```python
# Get video name from metadata if available
video_name = "unknown"
if 'metadata' in window and 'video_path' in window['metadata']:
    video_path = window['metadata']['video_path']
    video_name = video_path.split('/')[-1] if '/' in video_path else video_path
    # Truncate if too long
    if len(video_name) > 40:
        video_name = video_name[:37] + "..."

if single_frame_generated is not None:
    frame_info = f"Identity | Target | Predicted (batch_item={batch_item_idx}/{B-1}, frame={frame_idx}/T={stored_outputs['theta'].shape[1] if 'theta' in stored_outputs else '?'})"
else:
    frame_info = f"Identity | Target | (generation failed)"

wandb.log({
    "visuals/training_thumbnail": wandb.Image(
        thumbnail,
        caption=f"Epoch {self.current_epoch}, Batch {batch_idx}, Video: {video_name}\n{frame_info}"
    )
}, step=self.global_step)
```

## Expected Behavior

After this fix:
1. **Varied Videos**: Each thumbnail shows a random video from the batch
2. **Varied Frames**: Frame index is already randomized (T//3 to T-1)
3. **Video Identification**: Caption shows which video file is being displayed
4. **Batch Info**: Caption shows which batch item was selected (e.g., "batch_item=2/3")
5. **Static Identity Option**: If using `use_identity_image=True` in config, left panel still shows static identity image, but middle/right panels show random videos

## Example Caption

**Before**:
```
Epoch 5, Batch 12, Identity | Target | Predicted (frame 37 of T=50)
```

**After**:
```
Epoch 5, Batch 12, Video: videovideoeI2V8Bd5X9s-scene6_scene1.mp4
Identity | Target | Predicted (batch_item=2/3, frame=37/T=50)
```

## Benefits

1. **Debugging**: Can see how model performs on different videos
2. **Overfitting Detection**: If always showing same video, would miss per-video issues
3. **Training Progress**: See model quality across diverse videos
4. **Data Verification**: Confirm that all videos in dataset are being processed

## Notes

- Random selection uses `random.randint(0, B-1)` where B is batch size
- Frame index is already randomized (from T//3 to T-1) to avoid showing early frames
- Video name is truncated to 40 chars to avoid cluttering caption
- Batch item index is logged with emoji 🎲 for easy search in logs
- Works with both static identity images and per-video identity extraction

## Files Modified

- **vasa_trainer.py** (lines 1754-1956)
  - Added `batch_item_idx` randomization
  - Updated all tensor indexing from `[0]` or `[:]` to `[batch_item_idx:batch_item_idx+1]`
  - Enhanced caption with video name and batch info
  - Fixed EMO frame extraction for batched data
  - Added batched emotion label support
