# VASA-1 Project Structure

This directory contains the untangled VASA implementation with updated nemo components.

## Directory Structure

```
/media/2TB/VASA-1-hack/
├── nemo/                    # Updated EMOPortraits/nemo code (non-VASA components)
│   ├── models/             # Model implementations
│   ├── networks/           # Network architectures
│   ├── losses/             # Loss functions
│   ├── datasets/           # Dataset loaders
│   ├── utils/              # Utility functions
│   ├── repos/              # External repositories
│   └── logs/               # Pre-trained model checkpoints
│
├── vasa_*.py               # VASA-specific implementations
│   ├── vasa_trainer.py     # Main training script
│   ├── vasa_model.py       # VASA model architecture
│   ├── vasa_dataset.py     # VASA dataset handler
│   ├── vasa_scheduler.py   # VASA scheduler
│   └── vasa_lip_normalizer.py  # Lip normalization
│
├── vasa_config.yaml        # VASA configuration file
├── expression_normalizer.py # Expression normalization utilities
├── video_tracker.py        # Video tracking utilities
├── data/                   # Data directory for test images
├── cache/                  # Cache directory for processed data
├── checkpoints/            # Directory for saving model checkpoints
└── junk/                   # Directory for test videos
```

## Configuration

All paths are now configured in `vasa_config.yaml`:
- `volumetric_model`: Path to pre-trained volumetric model
- `volumetric_config`: Path to volumetric model config
- `data_dir`: Directory for data files
- `video_folder`: Directory for video files
- `cache_dir`: Cache directory

## Running VASA Trainer

To run the VASA trainer:

```bash
python vasa_trainer.py
```

## Testing Setup

To verify the setup is working correctly:

```bash
python test_vasa_setup.py
```

## Key Changes Made

1. **Separated VASA components** from EMOPortraits codebase
2. **Updated nemo directory** with latest EMOPortraits code
3. **Fixed hardcoded paths** to use configuration file
4. **Added path configuration** in vasa_config.yaml
5. **Created proper directory structure** for data, cache, and checkpoints
6. **Added sys.path management** to import nemo modules correctly

## Notes

- The volumetric model checkpoint is located in `nemo/logs/`
- All VASA-specific code is in the root directory
- The nemo directory contains the base EMOPortraits functionality
- FFmpeg warnings during import can be ignored if not using audio processing