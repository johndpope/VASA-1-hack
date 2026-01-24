# Expression Database Build Script

## Quick Start

```bash
# Build from single H5 file (default)
./build_expr_db.sh

# Build from custom H5 file
./build_expr_db.sh /path/to/motion.h5

# Build from directory of H5 files
./build_expr_db.sh /path/to/motion_attributes/

# Custom output and stride
./build_expr_db.sh /path/to/motion.h5 my_db.h5 10
```

## Parameters

```bash
./build_expr_db.sh [MOTION_PATH] [OUTPUT] [STRIDE]
```

1. **MOTION_PATH** (default: `/media/2TB/VASA-1-hack/motion_attributes.h5`)
   - Path to single H5 file OR directory containing H5 files
   - Script auto-detects whether it's a file or directory

2. **OUTPUT** (default: `expression_embeddings.h5`)
   - Output database filename

3. **STRIDE** (default: `5`)
   - Sample every Nth frame
   - Smaller = more embeddings, larger file
   - Larger = fewer embeddings, smaller file

## Examples

### Example 1: Use default settings
```bash
./build_expr_db.sh
# Uses: motion_attributes.h5, output: expression_embeddings.h5, stride: 5
```

### Example 2: Custom file and output
```bash
./build_expr_db.sh /media/12TB/VASA/my_motion.h5 my_expr_db.h5
# Uses: my_motion.h5, output: my_expr_db.h5, stride: 5 (default)
```

### Example 3: Process directory with custom stride
```bash
./build_expr_db.sh /media/12TB/VASA/motion_attributes/ expr_db.h5 10
# Processes all H5 files in directory, every 10th frame
```

### Example 4: Dense sampling
```bash
./build_expr_db.sh motion_attributes.h5 dense_db.h5 1
# Sample every frame (no stride)
```

## Output

The script will:
1. ✅ Extract expression embeddings from frames
2. ✅ Save to compressed H5 file
3. ✅ Auto-verify the database
4. ✅ Show GPU memory usage

**Expected output:**
```
🚀 Building expression database
Motion path: /media/2TB/VASA-1-hack/motion_attributes.h5
Output: expression_embeddings.h5
Frame stride: 5

📄 Processing single H5 file...
Loading motion file: /media/2TB/VASA-1-hack/motion_attributes.h5
Found 50 frames
Sampling 10 frames (stride=5)
Extracting embeddings: 100%|██████████| 10/10 [00:00<00:00, 934.93it/s]
Collected 10 embeddings
Saving to expression_embeddings.h5...
✅ Database saved: expression_embeddings.h5
   Shape: (10, 128)
   Size: 0.00 MB

✅ Done! Verifying database...
INFO     Database shape: (10, 128)
INFO     Mean: 0.003315, Std: 0.509409
INFO     GPU tensor shape: torch.Size([10, 128]), device: cuda:0
INFO     ✅ Database verification complete
```

## Troubleshooting

### No embeddings collected
```
Collected 0 embeddings
ValueError: need at least one array to stack
```
**Cause**: H5 file doesn't have expression embeddings
**Solution**: Check frame structure with:
```bash
python3 -c "
import h5py
with h5py.File('motion_attributes.h5', 'r') as f:
    frame = f['frame_0000']
    print('Keys:', list(frame.keys()))
    for k in frame.keys():
        if hasattr(frame[k], 'shape'):
            print(f'{k}: {frame[k].shape}')
"
```

### File not found
```
❌ Error: Path not found: /path/to/file
```
**Solution**: Check the path exists and is accessible

### Wrong shape
```
Warning: Invalid shape for frame_0000: (1, 256)
```
**Cause**: Expression embeddings are not 128-dimensional
**Solution**: Check your motion extraction process

## File Format

The script handles both:

### Single H5 File Format
```
motion_attributes.h5
├── frame_0000/
│   ├── expression_embed: [1, 128] or [128]
│   ├── theta: [1, 4, 4]
│   └── ...
├── frame_0001/
│   └── ...
```

### Directory Format
```
motion_attributes/
├── video1.h5
│   ├── 0/
│   │   ├── target_pose_embed: [128]
│   │   └── ...
├── video2.h5
│   └── ...
```

The script automatically detects which format you have.

## Using the Database

After building, configure in `vasa_config.yaml`:

```yaml
model:
  expression_dim: 128
  expression_db_path: "expression_embeddings.h5"
```

The cosine loss will automatically be computed during training.
