# Action Unit (AU) Prediction Implementation PRD

## Overview
Upgrade VASA-1 system to predict facial Action Units (AUs) alongside or instead of phoneme prediction. Action Units are anatomically based facial movements that provide fine-grained control over facial expressions.

## Objectives
1. Add AU prediction capability to the model architecture
2. Extract AU ground truth from cached video data
3. Implement AU-based auxiliary losses
4. Update configuration files with AU-specific parameters
5. Create visualization and diagnostic tools for AU predictions
6. Ensure AU predictions improve lip sync and expression quality

## Target Action Units
The following AUs should be predicted:
- **AU1**: Inner Brow Raiser (frontalis, pars medialis)
- **AU2**: Outer Brow Raiser (frontalis, pars lateralis)
- **AU4**: Brow Lowerer (corrugator supercilii, depressor supercilii)
- **AU5**: Upper Lid Raiser (levator palpebrae superioris)
- **AU6**: Cheek Raiser (orbicularis oculi, pars orbitalis)
- **AU7**: Lid Tightener (orbicularis oculi, pars palpebralis)
- **AU9**: Nose Wrinkler (levator labii superioris alaeque nasi)
- **AU10**: Upper Lip Raiser (levator labii superioris, caput infraorbitalis)
- **AU12**: Lip Corner Puller (zygomaticus major)
- **AU15**: Lip Corner Depressor (depressor anguli oris)
- **AU17**: Chin Raiser (mentalis)
- **AU20**: Lip Stretcher (risorius)
- **AU23**: Lip Tightener (orbicularis oris)
- **AU25**: Lips Part (depressor labii inferioris, relaxation of mentalis/orbicularis oris)
- **AU26**: Jaw Drop (masseter, temporal and internal pterygoid)
- **AU27**: Mouth Stretch (pterygoids, digastric)

Total: 16 AUs with intensity values [0.0, 1.0]

## Technical Requirements

### 1. Dataset Enhancement
- Extract AU intensities from MediaPipe or OpenFace landmarks
- Cache AU ground truth in H5 files per video window
- Add AU data loading to VASAIntegratedDataset
- Validate AU extraction with visualization tools

### 2. Model Architecture Changes
- Add AU prediction head to MotionTransformer (similar to phoneme_head)
- Output: 16-dimensional vector with sigmoid activation
- Input: Perceiver audio features or motion embeddings
- Position: After audio projection layer

### 3. Loss Function Implementation
- **Primary loss**: MSE or BCE for AU intensity regression
- **Optional losses**:
  - AU temporal consistency loss (smooth transitions)
  - AU co-occurrence loss (certain AUs activate together)
  - Audio-AU coupling loss (speech drives mouth AUs)
- Add loss monitoring with healthy ranges

### 4. Configuration Updates
- Add `lambda_aux_au` parameter (initial: 1.0)
- Add AU-specific hyperparameters
- Update both overfit_config.yaml and vasa_config.yaml

### 5. Visualization & Diagnostics
- Create AU prediction visualization (similar to audio_to_expression)
- Add AU intensity plots over time
- Show predicted vs ground truth AU activations
- Add diagnostic script to check AU extraction quality

### 6. Training Integration
- Cache AU ground truth during preprocessing
- Load AUs in dataloader collate function
- Pass AUs to loss computation
- Log AU metrics to WandB

## Implementation Pattern (Based on Phoneme)

### Files to Modify (following phoneme pattern):
1. **vasa_dataset.py**: Add AU ground truth extraction and caching
2. **vasa_model.py**: Add AU prediction head to MotionTransformer
3. **vasa_losses.py**: Add AU loss computation
4. **vasa_trainer.py**: Add AU visualization logging
5. **overfit_config.yaml**: Add lambda_aux_au parameter
6. **vasa_config.yaml**: Add lambda_aux_au parameter
7. **loss_monitor.py**: Add AU loss monitoring

### New Files to Create:
1. **extract_au_gt.py**: Extract AU ground truth from videos (similar to upsert_phoneme_gt.py)
2. **diagnose_au.py**: Diagnostic tool for AU predictions (similar to diagnose_phoneme.py)
3. **check_au_extraction.py**: Validate AU extraction quality

## Success Criteria
1. AU prediction head trains without errors
2. AU loss decreases during training (target: < 0.1 MSE)
3. AU predictions correlate with audio features (especially mouth AUs)
4. Visualizations show reasonable AU activations
5. Generated videos show improved expression control
6. No regression in existing phoneme/expression losses

## Technical Considerations
1. **AU Extraction Method**: Use MediaPipe or OpenFace? MediaPipe is already integrated.
2. **Loss Type**: MSE (regression) vs BCE (binary classification with intensity)
3. **Class Imbalance**: Some AUs (AU26 jaw drop) more common than others (AU9 nose wrinkle)
4. **Temporal Consistency**: AUs should transition smoothly across frames
5. **Co-occurrence Patterns**: Model AU relationships (AU12 + AU6 = smile)

## Dependencies
- MediaPipe Face Mesh (already integrated)
- OpenFace (optional, for comparison)
- PyTorch for model changes
- WandB for logging

## Timeline Estimate
- Phase 1: Dataset enhancement (AU extraction) - 2-3 days
- Phase 2: Model architecture changes - 1 day
- Phase 3: Loss implementation - 1 day
- Phase 4: Configuration & integration - 1 day
- Phase 5: Testing & visualization - 1-2 days
- Total: ~1 week of development + training validation

## References
- FACS (Facial Action Coding System) by Ekman & Friesen
- MediaPipe Face Mesh documentation
- OpenFace AU detection
- Existing phoneme implementation in VASA-1
