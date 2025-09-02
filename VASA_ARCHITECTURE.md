# VASA Architecture Overview

## System Components

### 1. VASADataset (`vasa_dataset.py`)

#### Key Features
- **Window-based Processing**: Sequences of 50 frames per window
- **Audio Processing**: 
  - Supports both Wav2Vec (768-dim) and Whisper (384-dim) audio features
  - Currently configured to use Wav2Vec (768 dimensions)
  - MFCC features (13-dim) for SyncNet compatibility
- **Face Attribute Extraction**:
  - MediaPipe for facial landmarks (68 points)
  - L2CS for gaze estimation (pitch/yaw)
  - Emotion recognition
  - Face distance/size estimation
- **Caching System**: Audio preprocessing cache to speed up training

#### Data Pipeline
```
Video Input → Frame Extraction → Face Detection → Landmark Extraction
                                                 ↓
                                          Audio Extraction
                                                 ↓
                                          Window Creation (50 frames)
                                                 ↓
                                          Feature Bundling
```

### 2. VASAModel (`vasa_model.py`)

#### Architecture Components

##### Input Processing
- **EfficientConditionEmbedding**: Combines all conditioning signals into 512-dim embeddings
  - Control signals (6-dim): gaze (2), head_distance (1), emotion (2), speed_bucket (1)
  - Landmark features (147-dim total):
    - Lips: 20 points × 3 coords = 60-dim
    - Right eye: 8 points × 3 coords = 24-dim  
    - Left eye: 7 points × 3 coords = 21-dim
    - Jaw: 10 points × 3 coords = 30-dim
    - Nose: 4 points × 3 coords = 12-dim
  - Blink state: 3-dim
  - Audio features: 768-dim (Wav2Vec) or 384-dim (Whisper)

##### Core Architecture
- **Transformer Blocks**: 8 transformer blocks processing 512-dim features
- **Motion Projection Heads**:
  - `motion_proj_theta`: Projects to expression parameters
  - `motion_proj_rotation`: Projects to head rotation
  - `motion_proj_translation`: Projects to head translation
  - `motion_proj_expression`: Projects to facial expressions

##### Output Structure
```
Input Conditions → Condition Embedding (512-dim) → Transformer Blocks × 8
                                                   ↓
                                            Motion Projections
                                                   ↓
                                    [Theta, Rotation, Translation, Expression]
```

### 3. VASATrainer (`vasa_trainer.py`)

#### Training Configuration
- **Batch Processing**: Processes windows of 50 frames
- **Loss Components**:
  1. **Reconstruction Losses** (always active):
     - `reconstruction`: Overall reconstruction quality (3.495)
     - `pose_loss`: Head pose accuracy (2.767)
     - `dynamics_loss`: Motion smoothness (0.251)
     - `motion_loss`: General motion quality (0.478)
     - `theta_loss`: Expression parameter accuracy (0.687)
     - `scale_loss`: Face scale consistency (0.778)
     - `rotation_loss`: Head rotation accuracy (0.631)
     - `translation_loss`: Head translation accuracy (0.669)
     - `expression_loss`: Facial expression accuracy (0.025)

  2. **Verification Loss** (conditional):
     - Activated based on specific conditions
     - Weight: λ=10

  3. **Control Loss** (delayed activation):
     - Starts at epoch 1750
     - Weight: λ=0.3
     - Ensures conditioning signals properly control output

  4. **Sync Loss** (optional):
     - Audio-visual synchronization
     - Weight: λ=1.0
     - Currently disabled in config

#### Training Schedule
- **Epoch-based Activation**:
  - Epochs 0-1749: Reconstruction losses only
  - Epochs 1750+: Control losses activated
  - Current epoch: 38 (control losses not yet active)

## Data Flow Analysis

### Forward Pass Flow

1. **Input Processing** (Batch=1, Seq=50):
   ```
   Raw Conditions → Validation → Mapping → Embedding
   ```

2. **Condition Embedding**:
   ```
   Control (6-dim) + Landmarks (147-dim) + Blink (3-dim) → 512-dim
   Audio (768-dim) → Projection → Integrated into 512-dim
   ```

3. **Transformer Processing**:
   ```
   512-dim → Block 0 → Block 1 → ... → Block 7 → 512-dim
   ```

4. **Motion Projection**:
   ```
   512-dim → Individual Projection Heads → Motion Parameters
   ```

5. **Loss Computation**:
   ```
   Predictions vs Ground Truth → Multiple Loss Terms → Weighted Sum
   ```

## Key Observations

### Current State
- **Audio Dimensions**: System shows 768-dim (Wav2Vec) despite output showing 384
- **Training Phase**: Early stage (epoch 38 of minimum 1750)
- **Active Losses**: Only reconstruction losses active
- **Device**: CUDA GPU (RTX 5090)

### Architecture Design Decisions
1. **Windowed Processing**: 50-frame windows with 25-frame stride for temporal consistency
2. **Delayed Control Loss**: Allows model to learn basic reconstruction before enforcing control
3. **Modular Conditioning**: Separate processing for different input modalities
4. **Transformer Backbone**: 8-layer transformer for temporal modeling

### Potential Issues/Notes
1. **Audio Feature Dimension Mismatch**: Debug output shows 384-dim but code uses 768-dim
2. **Control Loss Delay**: Won't activate until epoch 1750 (very late)
3. **Sync Loss Disabled**: Audio-visual sync not being enforced
4. **Zero Tensors**: Missing conditions (emotion, head_distance) filled with zeros

## Configuration Parameters

### Model Configuration
- Model dimension: 512
- Max sequence length: 50
- Transformer blocks: 8
- Learning rate: 0.001

### Loss Weights
- Reconstruction: 1.0
- Verification: 10.0 (when active)
- Control: 0.3 (when active)
- Sync: 1.0 (when active)

### Training Schedule
- Control loss start: epoch 1750
- Current epoch: 38
- Batch size: 1
- Window size: 50 frames
- Window stride: 25 frames
- Context size: 10 frames

## Dependencies
- PyTorch for deep learning
- MediaPipe for face detection
- L2CS for gaze estimation
- Wav2Vec/Whisper for audio encoding
- SyncNet for audio-visual synchronization
- LPIPS for perceptual loss
- Weights & Biases for experiment tracking