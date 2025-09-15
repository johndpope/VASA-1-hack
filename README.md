# VASA-1-hack

This repository contains the VASA implementation separated from EMOPortraits, with all components properly configured for standalone training.






### Setup Instructions

1. **Clone the repository with submodules:**
```bash
# Clone with submodules included
git clone --recurse-submodules https://github.com/johndpope/VASA-1-hack.git
cd VASA-1-hack

# Or if you already cloned without submodules:
git submodule update --init --recursive
```




###  Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x ~/miniconda.sh
~/miniconda.sh
# carefully accept - type yes - 

# Create conda environment
conda create -n vasa python=3.12
conda activate vasa

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install required packages
pip install omegaconf wandb opencv-python-headless pillow scipy matplotlib tqdm
pip install transformers diffusers accelerate einops
pip install facenet-pytorch insightface hsemotion-onnx
pip install mediapipe OmegaConf wandb
pip install git+https://github.com/Ahmednull/L2CS-Net.git memory-profiler rich
pip install diffusers h5py scikit-learn seaborn python_speech_features
pip install onnxruntime-gpu lpips pytorch_msssim

# EMOPortaits
cd nemo
chmod +x ./bootstrap.sh
./bootstrap.sh

```



3. **Create necessary symlinks:**
```bash
# Create symlink for repos (required for relative paths)
ln -s nemo/repos repos

# Create symlink for data directory (required for aligned keypoints)
ln -s nemo/data data

# Create symlink for losses directory (required for loss model weights)
ln -s nemo/losses losses
```

4. **Download pre-trained volumetric avatar model:**

The pre-trained model should be placed in:
```
nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth
```

5. **Prepare your training data:**
```bash
# Create directories
mkdir -p junk cache checkpoints

# Place your training videos in the junk directory
# Videos should be .mp4 format
cp your_training_videos/*.mp4 junk/
```

## 📁 Project Structure

```
VASA-1-hack/
├── nemo/                        # Git submodule: nemo repository (base EMOPortraits code)
│   ├── models/                  # Model implementations
│   ├── networks/                # Network architectures
│   ├── losses/                  # Loss functions
│   ├── datasets/                # Dataset loaders
│   ├── repos/                   # External repositories (face_par_off, etc.)
│   └── logs/                    # Pre-trained model checkpoints
│
├── vasa_*.py                    # VASA-specific implementations
│   ├── vasa_trainer.py          # Main training script
│   ├── vasa_model.py            # VASA model architecture
│   ├── vasa_dataset.py          # VASA dataset handler
│   ├── vasa_scheduler.py        # Diffusion scheduler
│   └── vasa_lip_normalizer.py   # Lip normalization utilities
│
├── vasa_config.yaml             # Main configuration file
├── video_tracker.py             # Video tracking utilities
├── syncnet.py                   # Sync network implementation
│
├── data/                        # Data files
│   └── aligned_keypoints_3d.npy
├── losses/                      # Loss model weights
│   └── loss_model_weights/
├── junk/                        # Training videos directory
├── cache/                       # Cache for processed data
├── checkpoints/                 # Model checkpoints
└── repos/                       # Symlink to nemo/repos
```

## ⚙️ Configuration

Edit `vasa_config.yaml` to configure paths and training parameters:

```yaml
paths:
  volumetric_model: "nemo/logs/[...]/328_model.pth"  # Pre-trained model
  volumetric_config: "nemo/models/stage_1/volumetric_avatar/va.yaml"
  data_dir: "data"
  video_folder: "junk"  # Your training videos directory
  cache_dir: "cache"
  checkpoint_dir: "checkpoints"

train:
  batch_size: 1
  num_epochs: 4000
  lr: 1e-3
  # ... other training parameters
```

## 🏃 Running Training

### Test the Setup
```bash
python test_vasa_setup.py
```

Expected output:
```
✓ Config loaded successfully
✓ All paths exist
✓ All modules import correctly
✓ Setup looks good! You can now run vasa_trainer.py
```

### Training Modes

#### 1. **Vanilla Training** (Full Dataset)
Use the standard configuration for training on your complete dataset:

```bash
# Uses vasa_config.yaml by default
python vasa_trainer.py

# Or explicitly specify the config
python vasa_trainer.py --config vasa_config.yaml
```

**Key parameters in `vasa_config.yaml`:**
- `window_size: 50` - Full 50-frame windows
- `n_layers: 8` - Full 8 transformer layers
- `num_steps: 1000` - Full 1000 diffusion steps
- `batch_size: 1` - Adjust based on GPU memory
- `num_epochs: 4000` - Full training schedule

#### 2. **Overfitting Training** (Fast Convergence Testing)
Use the overfitting configuration for rapid testing and debugging:

```bash
# Use the overfitting configuration
python vasa_trainer.py --config overfit_config.yaml
```

**Key differences in `overfit_config.yaml`:**
- `window_size: 20` - Smaller windows for faster processing
- `n_layers: 2` - Reduced transformer depth (2x-4x faster)
- `num_steps: 100` - Reduced diffusion steps (10x faster)
- `batch_size: 4` - Larger batch for better GPU utilization
- `num_epochs: 100` - Shorter training for quick iteration
- `max_videos: 100` - Limited dataset size
- `num_workers: 8` - Multi-threaded data loading
- No augmentation - Pure overfitting test

**When to use overfitting mode:**
- Testing new model architectures
- Debugging training pipeline
- Verifying data loading and caching
- Quick convergence tests
- Checking if model can overfit to small dataset (sanity check)

### Monitoring Training

Both training modes support WandB logging:

```bash
# View training progress
# Visit the URL printed at training start, e.g.:
# wandb: 🚀 View run at https://wandb.ai/your-username/vasa/runs/run-id
```

For overfitting mode, runs are grouped as "overfit-experiments" in WandB for easy comparison.

### Custom Dataset Path

To use a different dataset (e.g., CelebV-HQ):

```bash
# Edit the config file or create a custom one
# Update video_folder path in the config:
# video_folder: "/path/to/your/dataset"

# For example, using CelebV-HQ:
# video_folder: "/media/12TB/Downloads/CelebV-HQ/celebvhq/35666"
```

The trainer will:
- Load the pre-trained volumetric avatar model
- Process videos from the configured directory
- Cache processed windows for faster subsequent epochs
- Save checkpoints periodically based on `save_freq`
- Save checkpoints to `checkpoints/` (or `checkpoints_overfit/` for overfitting mode)
- Log to Weights & Biases (if enabled)

### Performance Comparison

| Parameter | Vanilla Training | Overfitting Mode | Speedup |
|-----------|-----------------|------------------|---------|
| Window Size | 50 frames | 20 frames | 2.5x |
| Transformer Layers | 8 | 2 | 4x |
| Diffusion Steps | 1000 | 100 | 10x |
| Batch Size | 1 | 4 | 4x |
| Workers | 0 | 8 | Parallel loading |
| Epoch Time (RTX 5090) | ~5 min | ~1.5 min | 3.3x |
| Convergence | 1000+ epochs | 10-20 epochs | 50x+ |

## 📝 Logging Configuration

### Logging Levels (nemo/logger.py)

The project uses Python's logging module with three configurable levels defined in `nemo/logger.py:28-30`:

```python
# log_level = logging.WARNING    # Minimal output - only warnings and errors
log_level = logging.INFO         # Standard output - informational messages (default)
# log_level = logging.DEBUG       # Verbose output - detailed debugging information
```

**Logging Levels Explained:**

1. **WARNING** (`logging.WARNING`)
   - Shows only warnings, errors, and critical messages
   - Use when you want minimal console output during training
   - Best for production runs where you only need to know about issues

2. **INFO** (`logging.INFO`) - **Currently Active**
   - Shows informational messages, warnings, and errors
   - Provides training progress, epoch updates, and key metrics
   - Default and recommended level for normal training runs
   - Balances visibility with readability

3. **DEBUG** (`logging.DEBUG`)
   - Shows all messages including detailed debugging information
   - Includes tensor shapes, gradient information, and internal state
   - Use when troubleshooting model issues or understanding data flow
   - Can be verbose - recommended only for debugging sessions

**To change the logging level:**
1. Edit `nemo/logger.py` line 29
2. Uncomment the desired level and comment out the others
3. The change takes effect on next run

**Additional Features:**
- Logs are saved to `project.log` file for later review
- Rich formatting with color-coded output and timestamps
- Third-party library logging is suppressed to reduce noise
- TorchDebugger class available for advanced PyTorch debugging

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError: No module named 'logger'**
   ```bash
   # The logger module is in nemo, paths are already configured
   # If still having issues, check that nemo is cloned properly
   ```

2. **FileNotFoundError: './repos/face_par_off/res/cp/79999_iter.pth'**
   ```bash
   # Ensure the symlink exists:
   ln -s nemo/repos repos
   ```

3. **ValueError: num_samples should be a positive integer value, but got num_samples=0**
   ```bash
   # No videos found. Add videos to junk/ directory:
   cp your_video.mp4 junk/
   ```

4. **FileNotFoundError: Config file not found at channel_config.yaml**
   ```bash
   # Copy from EMOPortraits or create a basic one
   ```

5. **CUDA out of memory**
   - Reduce `batch_size` in vasa_config.yaml
   - Enable gradient checkpointing
   - Reduce `sequence_length` in dataset config

6. **FFmpeg warnings**
   - These can be safely ignored if not processing audio
   - To fix: `pip install ffmpeg-python`

### Required Files from EMOPortraits

If you're missing files, you'll need these from EMOPortraits:
- `channel_config.yaml` - Channel configuration
- `syncnet.py` - Sync network implementation  
- `data/aligned_keypoints_3d.npy` - 3D keypoint alignments
- `losses/loss_model_weights/*.pth` - Pre-trained loss models
- Pre-trained volumetric avatar checkpoint

## 📊 Monitoring Training

Training progress is logged to:
- **Console**: Real-time training metrics
- **Weights & Biases**: Detailed metrics and visualizations (if enabled)
- **Checkpoints**: Saved every N epochs to `checkpoints/`

Monitor training:
```bash
# Watch training logs
tail -f project.log

# Check W&B dashboard
# https://wandb.ai/YOUR_USERNAME/vasa/
```

## 🛠️ Development

### Project Organization

- **VASA-specific code**: Root directory (`vasa_*.py`)
- **Base EMOPortraits code**: `nemo/` directory
- **Configuration**: `vasa_config.yaml`
- **Training data**: `junk/` directory
- **Model outputs**: `checkpoints/` directory

### Key Improvements Made

1. **Separated VASA components** from EMOPortraits codebase
2. **Fixed all hardcoded paths** to be relative or configurable
3. **Proper module imports** with sys.path management
4. **Configurable paths** via vasa_config.yaml
5. **Auto-detection** of project directories in nemo code
6. **Clean separation** between VASA-specific and base code

### Working with the Submodule

**Update nemo to latest version:**
```bash
cd nemo
git pull origin main
cd ..
git add nemo
git commit -m "Update nemo submodule to latest"
```

**Lock to specific nemo version:**
```bash
cd nemo
git checkout <commit-hash>
cd ..
git add nemo
git commit -m "Lock nemo to specific version"
```


## 📝 Notes

- The volumetric model must be pre-trained (from EMOPortraits)
- Training requires at least one video in the `junk/` directory
- All paths in configs are relative to the project root
- The `repos` symlink is required for backward compatibility

## 🚨 Known Issues

- Training requires significant GPU memory (recommended: 24GB+)
- Some imports show FFmpeg warnings (can be ignored)
- Initial dataset processing can be slow (cached afterward)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: The nemo submodule and other dependencies may have their own licenses.

## 🙏 Acknowledgments

- EMOPortraits team for the base implementation
- VASA paper authors for the architecture design
- Contributors to the nemo repository 
