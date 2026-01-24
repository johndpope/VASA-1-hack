# Synchformer Integration for VASA

This document describes the integration of Synchformer as a replacement for SyncNet in the VASA implementation.

## Installation

Synchformer has been added as a git submodule. To initialize it:

```bash
git submodule update --init --recursive
```

## Path Configuration

The wrapper automatically searches for Synchformer in multiple locations:

1. **Submodule** (default): `./Synchformer`
2. **Parent directory**: `../Synchformer`
3. **Environment variable**: Set `SYNCHFORMER_PATH=/path/to/Synchformer`
4. **Home directory**: `~/Synchformer`

### Environment Variables

- `SYNCHFORMER_PATH`: Override Synchformer installation path
- `SYNCHFORMER_CHECKPOINT`: Specify checkpoint file location
- `SYNCHFORMER_CONFIG`: Specify config file location
- `SYNCHFORMER_DEVICE`: Set device (default: cuda)

### Configuration File

Create `synchformer_config.json` (see `synchformer_config.json.example`):

```json
{
    "path": "./Synchformer",
    "checkpoint": "./Synchformer/pretrained/synchformer.pt",
    "config": "./Synchformer/configs/sync.yaml",
    "device": "cuda"
}
```

## Configuration

The integration can be enabled in your config files (`vasa_config.yaml` or `overfit_config.yaml`):

```yaml
loss:
  use_sync_loss: true      # Enable sync loss during training
  use_synchformer: true    # Use Synchformer instead of SyncNet
  lambda_sync: 0.5         # Weight for sync loss
```

## Features

### 1. Paper-Accurate Sync Loss

The sync loss now implements the formulation from the paper:

**L_sync = (Δt_p - Δt_gt)² + (t_p - t_gt)²**

Where:
- `Δt_p, Δt_gt`: Predicted and ground truth temporal offset magnitudes
- `t_p, t_gt`: Timestamps where misalignment occurs

### 2. Synchformer Advantages

- **Transformer Architecture**: Uses separate audio (AST) and visual (MotionFormer) encoders
- **Better Temporal Modeling**: 21-class offset prediction from -2 to +2 seconds
- **Robust Features**: Pre-trained on large-scale audio-visual data

### 3. Automatic Preprocessing

The wrapper handles all necessary transformations:
- Video segmentation into 14 segments of 16 frames
- Audio conversion to mel-spectrograms
- Normalization for AST and MotionFormer

## Usage

### Training with Synchformer

1. Download a pre-trained Synchformer checkpoint and place it in one of:
   - `Synchformer/checkpoints/sync_model.pt`
   - `Synchformer/pretrained/synchformer.pt`
   - `Synchformer/logs/sync_models/best.pt`

2. Enable in config:
   ```yaml
   use_sync_loss: true
   use_synchformer: true
   ```

3. Train as usual:
   ```bash
   python vasa_trainer.py --config vasa_config.yaml
   ```

### Backward Compatibility

To use the original SyncNet, simply set:
```yaml
use_synchformer: false
```

## Architecture Details

### SynchformerWrapper

Located in `synchformer_wrapper.py`, provides:

- `compute_sync_score()`: Returns synchronization probability
- `compute_sync_loss()`: Returns loss for training
- `_compute_temporal_offset()`: Detects temporal misalignment

### Integration Points

1. **vasa_losses.py**:
   - Conditionally uses Synchformer or SyncNet based on config
   - Implements paper's sync loss equation

2. **synchformer_wrapper.py**:
   - Wraps Synchformer model
   - Handles preprocessing
   - Provides SyncNet-compatible interface

## Benefits

1. **Better Sync Quality**: Synchformer provides more accurate lip-sync detection
2. **Temporal Offset Detection**: Can identify specific misalignment magnitudes
3. **Research-Backed**: Based on state-of-the-art audio-visual synchronization research
4. **Easy Switch**: Can toggle between SyncNet and Synchformer via config

## Troubleshooting

### Missing Checkpoint
If no checkpoint is found, the model will use random initialization. Download pre-trained weights from the Synchformer repository.

### CUDA Out of Memory
Synchformer requires more memory than SyncNet. Reduce batch size if needed.

### Import Errors
Ensure the submodule is initialized:
```bash
git submodule update --init --recursive
```

## References

- [Synchformer Paper](https://arxiv.org/abs/2203.16437)
- [Synchformer GitHub](https://github.com/v-iashin/Synchformer)