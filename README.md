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
# Create conda environment
conda create -n vasa python=3.10
conda activate vasa

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install omegaconf wandb opencv-python pillow scipy matplotlib tqdm
pip install transformers diffusers accelerate
pip install facenet-pytorch insightface hsemotion-onnx
pip install mediapipe 
pip install l2cs memory-profiler rich

# Audio-visual synchronization requirements
pip install phonemizer  # For phoneme extraction
pip install librosa soundfile  # For audio processing
pip install praat-parselmouth  # For detailed phonetic analysis (optional)


# EMOPortaits
cd nemo
bootstrap.sh

```



3. **Create necessary symlinks:**
```bash
# Create symlink for repos (required for relative paths)
ln -s nemo/repos repos
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

1. **Test the setup:**
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

2. **Start training with TDD (Test-Driven Development) losses:**
```bash
python train_tdd_wandb.py
```

3. **Or run original training:**
```bash
python vasa_trainer.py
```

The trainer will:
- Load the pre-trained volumetric avatar model
- Process videos from the configured directory
- Save checkpoints to `checkpoints/`
- Log to Weights & Biases (if enabled)
- Apply TDD losses for:
  - Expression preservation
  - Motion quality
  - Lip synchronization
  - Phoneme-to-visual mapping

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

## 🎯 TDD (Test-Driven Development) Features

### Expression Preservation
The model includes an expression reconstruction loss that enforces preservation of facial expressions:

```python
# In tdd_loss_balanced.py
expression_reconstruction_loss = F.mse_loss(
    outputs['expression_embed'],
    targets['expression_embed']
)
```

This ensures:
- Expressions remain consistent with the input video
- No expression drift during generation
- Acts like an expression "codebook"

### Phoneme-to-Visual Mapping
Accurate lip synchronization through:
- Phoneme extraction from audio
- Visual feature mapping for each phoneme
- Curriculum learning stages:
  1. **Stage 1**: Basic mouth open/close
  2. **Stage 2**: Phoneme mapping
  3. **Stage 3**: Fine synchronization

### Audio-Visual Synchronization Tests
Automated tests ensure quality:
- Mouth openness correlation with audio amplitude
- Silence detection (lips closed when quiet)
- Phoneme-visual consistency
- Temporal alignment

### Running with TDD Losses
```bash
# Train with strong expression preservation (RECOMMENDED)
python train_expression_preserving.py

# Train with balanced TDD losses
python train_tdd_wandb.py --expression_weight 2.0

# Test inference with multi-step denoising
python vi_complete.py --steps 20 --input video.mp4

# Validate audio-visual sync
python test_complete_system.py

# Test expression preservation
python tdd_expression_preserving.py
```

### Expression Preservation Training
The `train_expression_preserving.py` script uses:
- **3x stronger** expression reconstruction loss
- Expression codebook to prevent drift
- Temporal consistency enforcement
- Curriculum learning with expression focus
- Separate learning rates for expression parameters

## 📊 Monitoring Training

Training progress is logged to:
- **Console**: Real-time training metrics
- **Weights & Biases**: Detailed metrics and visualizations (if enabled)
- **Checkpoints**: Saved every N epochs to `checkpoints/`
- **TDD Test Results**: Pass/fail rates for quality criteria

Monitor training:
```bash
# Watch training logs
tail -f project.log

# Check W&B dashboard
# https://wandb.ai/YOUR_USERNAME/vasa/

# Monitor TDD test results
grep "TDD:" project.log | tail -20
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
7. **Test-Driven Development (TDD) losses** for measurable quality
8. **Expression preservation** through reconstruction loss
9. **Phoneme-to-visual mapping** for accurate lip sync
10. **Multi-step DDIM inference** for quality improvement
11. **Audio-visual synchronization** with curriculum learning

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
