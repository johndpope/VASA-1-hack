# Claude Code Instructions

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

## VASA-1 Project Status and Findings

### Current Training Status
- **Overfitting Training**: Successfully running with wandb logging
  - Loss decreased from 40+ to ~1.2-1.3 showing good convergence
  - Using WindowSequenceSampler for temporal context preservation
  - Caching optimization working (source embeddings computed once per batch)
  - Wandb project: https://wandb.ai/snoozie/vasa-overfitting

### Key Fixes Applied

#### 1. DataLoader Issues (FIXED)
- **Problem**: VASAIntegratedDataset.__len__() returned video count (6) instead of window count (316)
- **Solution**: Fixed to return len(self.windows) and implemented WindowSequenceSampler
- **Files**: vasa_dataset.py, vasa_sampler.py, train_overfit.py, vasa_trainer.py

#### 2. Temporal Context Handling (FIXED)
- **Problem**: Breaking prev_context mechanism when changing __getitem__
- **Solution**: Custom WindowSequenceSampler maintains sequences of 4 consecutive windows
- **Implementation**: create_window_sequence_collate_fn adds prev_context to batches

#### 3. CUDA Multiprocessing (FIXED)
- **Problem**: "Cannot re-initialize CUDA in forked subprocess"
- **Solution**: Set num_workers=0 in DataLoader

#### 4. Redundant Computations (FIXED)
- **Problem**: Source embeddings computed repeatedly for same identity image
- **Solution**: Improved caching with stable tensor-based keys instead of id()

### Audio Context Implementation

#### JoyVASA Approach (from paper):
- Uses frozen wav2vec2 encoder for audio features
- Includes both past audio features A_{-w_prev, w_prev} and current motion
- Concatenates past speech with current noisy motion in diffusion

#### Our Implementation:
- ✅ Using wav2vec2 for audio features (768 dimensions)
- ✅ Including prev_context with previous motion parameters
- ✅ Including previous audio features in context
- ⚠️ May need to adjust concatenation strategy to match JoyVASA

### Known Issues

#### 1. Wandb Visualization
- `disentangle/frame_j` visualization broken due to shape mismatches
- Occurs when generated_frames has different shape than expected
- Non-critical - doesn't affect training

#### 2. Batch Size Optimization
- Currently using 4 windows per batch (7GB/32GB VRAM)
- Could increase to 16 windows but shape mismatch in collate function needs debugging

### Training Scripts

#### For Overfitting Test:
```bash
./train.sh  # Select option 1 for overfitting
# OR directly:
python train_overfit.py
```

#### For Full Training:
```bash
./train.sh  # Select option 2 for full training
# OR directly:
python vasa_trainer.py --config vasa_config.yaml
```

### Important Configuration

#### overfit_config.yaml:
- batch_size: 28 (config value, actual is 4 windows due to sampler)
- learning_rate: 5e-3
- gradient_accumulation_steps: 2
- num_epochs: 1000
- resume_from: "checkpoints_overfit/best_checkpoint.pt"

#### vasa_config.yaml:
- resume_from: "" (set to checkpoint path to resume)
- Similar settings but for full dataset

### JoyVASA Reference
- Location: /media/12TB/JoyVASA
- Uses wav2vec2 model with linear interpolation for audio features
- Different architecture than VASA-1 (uses LivePortrait wrapper)
