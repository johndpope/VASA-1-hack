# Claude Code Instructions

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

## Documentation

All project documentation is stored in the `/docs` folder. Key documents include:
- `docs/AUDIT_TOOL_README.md` - Expression audit tool documentation
- `docs/LOSS_AUDIT.md` - Loss function audit and configuration
- `docs/LOSS_CLEANUP_SUMMARY.md` - Loss cleanup history
- `docs/vasa_model_documentation.md` - VASA model architecture docs
- `docs/vasa-prd.md` - Product requirements document

When creating new documentation, place `.md` files in `/docs` folder (except README.md and CLAUDE.md which stay in root).

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

### Expression Audit Tool

#### Purpose
Diagnose training issues by comparing ground truth expressions vs model predictions frame-by-frame.

#### Quick Start
```bash
./audit.sh  # Interactive menu
```

Or direct:
```bash
python audit_expressions.py \
    --video junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4 \
    --identity ./data/IMG_1.png \
    --config overfit_config.yaml \
    --checkpoint checkpoints_overfit/best_checkpoint.pt \
    --output-dir expression_audit
```

#### What It Analyzes
1. **Expression L2 Distance** - Per-frame difference in expression embeddings
   - Target: < 1.0 for good overfitting
   - Current: ~5.5 (model NOT overfitting yet)

2. **Theta L2 Distance** - Per-frame head pose difference
   - Target: < 0.3 for good overfitting
   - Current: ~1.5 (poor pose matching)

3. **Audio Alignment** - Verifies identical audio features used
   - Should be: 0.000000 ✅
   - Confirms audio preprocessing is correct

#### Current Findings (Epoch 226)
- ❌ **Model is NOT overfitting** despite 226 epochs
- ❌ Expression L2: 5.54 (should be < 1.0)
- ❌ Theta L2: 1.50 (should be < 0.3)
- ✅ Audio features identical (preprocessing correct)

**Root causes**:
1. Model capacity too small (12.5M vs 29M target)
2. Loss weights may need tuning
3. Learning rate may be too high
4. Need more training epochs

#### Output Files
- `expression_comparison.png` - Visualization with 3 subplots showing L2 distances over time
- `expression_metrics.csv` - Per-frame metrics for detailed analysis

See `docs/AUDIT_TOOL_README.md` for full documentation.

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

### Loss Function Management

#### Loss Monitoring System
The project uses `loss_monitor.py` to track loss values and warn when they're outside healthy ranges.

**IMPORTANT**: When adding or removing losses in `vasa_losses.py`, you MUST update `loss_monitor.py`:

1. **Adding a new loss**:
   - Add entry to `LossRangeMonitor.LOSS_RANGES` dict in `loss_monitor.py`
   - Define healthy range (min, max), warning threshold, and critical threshold
   - Include description of what the loss measures
   - Example:
   ```python
   'my_new_loss': {
       'healthy': (0.001, 0.1),  # Expected range during good training
       'warning': 0.2,            # Warn if loss exceeds this
       'critical': 0.5,           # Critical if loss exceeds this
       'description': 'What this loss measures and why it matters'
   }
   ```

2. **Removing a loss**:
   - Remove or comment out the entry in `LOSS_RANGES`
   - Update any documentation in `docs/LOSS_AUDIT.md` if it exists

3. **Finding healthy ranges**:
   - Run training and observe typical values in WandB
   - Set healthy range to cover 90% of observed values
   - Set warning threshold at ~2x healthy max
   - Set critical threshold at ~5x healthy max
   - For accuracy metrics (should be high), invert: healthy = (0.7, 1.0), warning = 0.5, critical = 0.3

#### Recent Loss Changes

**Audio-Lip Refactor (2025-10-03)**:
- Moved audio-lip correlation from early section to control losses (after landmark extraction)
- Now uses extracted lips from generated frames instead of dataset targets
- Also updated mouth_openness_direct to use extracted lips
- See `docs/AUDIO_LIP_REFACTOR_PLAN.md` for details

**Blink Loss Implementation (2025-10-03)**:
- Implemented previously stubbed `_compute_blink_loss()` function
- Computes eye openness loss (channels 1-2) and blink phase loss (channel 0)
- Extracts blink states from generated frames using MediaPipe
- See `docs/BLINK_LOSS_IMPLEMENTATION.md` for details

**Loss Cleanup (Previous)**:
- Disabled conflicting losses (L1 vs L2, redundant smoothness losses)
- Reduced conflicting weights (audio_expr_coupling, audio_lip)
- Added real-time monitoring with warnings/critical alerts
- See `docs/LOSS_CLEANUP_SUMMARY.md` and `docs/LOSS_AUDIT.md` for details

**Temporal Stabilization (2025-10-03)**:
- Added Gaussian smoothing to motion parameters before frame generation
- Fixes severe geometric distortions (warping, shearing, tilting) in generated frames
- Applied automatically for sequences with 4+ frames (sigma=1.0 default)
- See `docs/TEMPORAL_STABILIZATION.md` for details



In the context of the VASA-1 paper and its reference to the MegaPortraits codebase (which builds on 3D-aided facial reenactment frameworks like those in [19], likely referring to Drobyshev et al.'s MegaPortraits work), rigid and non-rigid 3D warping are key steps in decomposing a facial image into a canonical (neutral, standardized) 3D appearance volume \( V^{app} \). This process enables disentangled representations for high-fidelity face reenactment, where appearance, identity, pose, and dynamics are separated for tasks like generating nuanced facial animations from a single source image.

### Overview of the Process
To extract \( V^{app} \) from an input face image:
1. First, a **posed 3D volume** is estimated directly from the image. This represents the face in its observed orientation and configuration, capturing the raw 3D geometry (e.g., using a 3D morphable model or volumetric reconstruction network).
2. This posed volume is then transformed back to a **canonical 3D volume** (a front-facing, neutral-pose representation) via a combination of **rigid** and **non-rigid 3D warping**. Warping here refers to deforming or mapping the 3D voxels/features from the posed space to the canonical space, ensuring the intrinsic appearance (e.g., texture, shape details like eye shape or skin tone) is isolated from extrinsic factors like head orientation or expression.

This inverse warping (from posed to canonical) is the reverse of the forward process used during reconstruction, where the canonical volume is warped to match a target pose/expression. The MegaPortraits codebase implements this in its encoding pipeline (e.g., via networks for pose estimation and deformation fields), often leveraging voxel-based or feature volume representations for megapixel-resolution outputs.

### What Rigid 3D Warping Means
- **Definition**: Rigid warping applies a global, isometric transformation to the entire 3D volume without altering its internal structure, distances, or proportions. It preserves the shape and size of the face while only changing its position, orientation, or scale in 3D space.
- **How it works in this context**: 
  - Uses the estimated **head pose** parameters (e.g., \( z^{pose} \), typically 6 degrees of freedom: 3 for rotation via Euler angles or quaternion, and 3 for translation).
  - This is akin to applying a rigid-body transformation matrix to align the posed volume's overall head orientation to a canonical (e.g., zero-rotation, front-facing) pose.
  - Mathematically, for a 3D point \( \mathbf{p} \) in the posed volume, the rigid warp is \( \mathbf{p}' = R \mathbf{p} + \mathbf{t} \), where \( R \) is the rotation matrix and \( \mathbf{t} \) is the translation vector derived from inverting the head pose.
- **Purpose**: Handles large-scale head movements (e.g., turning the head left/right or tilting), ensuring the canonical volume isn't distorted by global rotations. Without this, reenactment would fail for off frontal views, as the decoder couldn't align features properly.
- **Why rigid?** It models the skull/jaw as a semi-rigid structure, avoiding deformation of core facial proportions during pose correction.

### What Non-Rigid 3D Warping Means
- **Definition**: Non-rigid warping applies local, deformable transformations to specific parts of the 3D volume, allowing stretching, bending, or shearing while rigid warping handles the global alignment. It does not preserve all distances or angles, enabling fine-grained adjustments.
- **How it works in this context**:
  - Builds on the rigid-warped volume and uses **facial dynamics** parameters (e.g., \( z^{dyn} \), which encode expression via blendshapes, landmarks, or deformation fields from models like FLAME or EMOCA).
  - Involves estimating a dense deformation field (e.g., per-voxel displacements or a warp grid) to "undo" expression-specific deformations, mapping the posed/expressive face back to a neutral canonical state.
  - In MegaPortraits-style implementations, this often uses a neural network (e.g., a U-Net or flow predictor) to regress non-rigid fields based on facial landmarks or emotional cues, ensuring subtle details like cheek puffs or lip curls are normalized.
  - Mathematically, it extends the rigid transform with a displacement field \( \Delta \mathbf{p} \), so \( \mathbf{p}'' = R (\mathbf{p} + \Delta \mathbf{p}) + \mathbf{t} \), where \( \Delta \mathbf{p} \) varies spatially (e.g., higher around the mouth/eyes).
- **Purpose**: Captures deformable facial movements (e.g., smiling, frowning, or eye squinting) that rigid transforms can't handle. This disentangles dynamics from appearance, allowing the latent generator to focus on nuanced expressions during reenactment without contaminating identity.
- **Why non-rigid?** Faces aren't perfectly rigid; soft tissues deform independently of the head's global motion, so this step ensures the canonical volume truly represents a "neutral" identity without expression artifacts.

### Why Both Are Needed Together
- **Sequential application**: Rigid warping first corrects the global pose (coarse alignment), followed by non-rigid warping for local expression normalization (fine-tuning). This two-stage approach, as in MegaPortraits, prevents error propagation and enables robust disentanglement.
- **Benefits for reenactment**: The resulting canonical \( V^{app} \) can then be re-warped forward (rigid + non-rigid in the opposite direction) using target pose/dynamics to generate new videos with preserved high-quality details (e.g., megapixel textures). Without this, 2D-only methods suffer from artifacts in 3D views, while pure rigid methods ignore expressions.
- **Implementation note**: In the MegaPortraits codebase (and hacks like those on GitHub), this is encoded in modules like pose estimators (e.g., based on ResNet for landmarks) and warp generators (e.g., for deformation fields). Training uses reconstruction losses to supervise the warps, ensuring photometric fidelity.

This framework draws from prior works like EMOCA (for emotional 3D reconstruction) and general 3D face models (e.g., FLAME), emphasizing explicit 3D control over implicit 2D warping for better generalization in one-shot scenarios.