# VASA Loss Function Audit

## Summary Statistics

**Total Loss Functions**: 40+
**Active Losses (lambda > 0)**: ~25
**Disabled Losses (lambda = 0)**: ~15
**Potential Conflicts Identified**: 7 CRITICAL

---

## Loss Functions Table

| # | Loss Name | Lambda Value | Pred/Target | Expected Range | Purpose | Status | Conflicts |
|---|-----------|--------------|-------------|----------------|---------|--------|-----------|
| **RECONSTRUCTION LOSSES (Core)** |
| 1 | `theta_loss` | `lambda_pose: 1.0` | MSE(pred_theta, target_theta) | 0.0 - 1.0 | Match 3DMM theta parameters | ✅ Active | None |
| 2 | `scale_loss` | `lambda_pose: 1.0` | MSE(pred_scale, target_scale) | 0.0 - 0.1 | Match 3D scale | ✅ Active | None |
| 3 | `rotation_loss` | `lambda_pose: 1.0` | MSE(pred_rotation, target_rotation) | 0.0 - 1.0 | Match head rotation (Euler angles) | ✅ Active | ⚠️ #1 |
| 4 | `translation_loss` | `lambda_pose: 1.0` | MSE(pred_translation, target_translation) | 0.0 - 1.0 | Match head translation | ✅ Active | None |
| 5 | `expression_loss` | `lambda_expression_l1: 0.1` | L1(pred_expr, target_expr) | 0.0 - 5.0 | Match expression embeddings (L1 for sparsity) | ✅ Active | ⚠️ #2 |
| 6 | `expression_mse` | `1.0 - lambda_expression_l1 = 0.9` | MSE(pred_expr, target_expr) | 0.0 - 5.0 | Match expression embeddings (L2 for smoothness) | ✅ Active | ⚠️ #2 |
| 7 | `expression_variance_loss` | `lambda_expression_variance: 5.0` | MSE(std(pred_expr), std(target_expr)) | 0.0 - 1.0 | Force prediction variance to match target | ✅ Active | ⚠️ #3 |
| 8 | `expression_temporal_loss` | `lambda_expression_temporal: 2.0` | MSE(diff(pred_expr), diff(target_expr)) | 0.0 - 1.0 | Force temporal variation in expressions | ✅ Active | ⚠️ #3 |
| **WARP LOSSES** |
| 9 | `xy_warp_loss` | `lambda_warp: 2.0` | MSE(pred_xy_warp, target_xy_warp) | 0.0 - 10.0 | Match 2D XY warps | ✅ Active | None |
| 10 | `xy_warp_smooth` | `lambda_warp_smooth: 0.01` | TV-L1 spatial smoothness | 0.0 - 1.0 | Spatial smoothness for XY warps | ✅ Active | None |
| 11 | `rigid_warp_loss` | `lambda_warp: 2.0` | MSE(pred_rigid_warp, target_rigid_warp) | 0.0 - 10.0 | Match rigid warps | ✅ Active | None |
| 12 | `rigid_warp_smooth` | `lambda_warp_smooth: 0.01 * 2` | TV-L1 spatial smoothness (2x) | 0.0 - 1.0 | Extra smoothness for rigid warps | ✅ Active | None |
| 13 | `uv_warp_loss` | `lambda_warp: 2.0` | MSE(pred_uv_warp, target_uv_warp) | 0.0 - 10.0 | Match 3D UV warps | ✅ Active | None |
| 14 | `uv_warp_l1` | `lambda_warp_l1: 0.1` | L1(pred_uv_warp, target_uv_warp) | 0.0 - 5.0 | Sparsity for UV warps | ✅ Active | ⚠️ #4 |
| 15 | `uv_warp_magnitude` | `lambda_warp_magnitude: 5.0` | MSE(\|\|pred_warp\|\|, \|\|target_warp\|\|) | 0.0 - 1.0 | Prevent warp collapse to zero | ✅ Active | ⚠️ #4 |
| 16 | `uv_warp_velocity_l1` | `lambda_warp_l1 * 0.5 = 0.05` | L1(diff(pred_warp), diff(target_warp)) | 0.0 - 1.0 | Temporal velocity sparsity | ✅ Active | None |
| 17 | `uv_warp_smooth` | `lambda_warp_smooth: 0.01` | TV-L1 spatial smoothness | 0.0 - 1.0 | Spatial smoothness for UV warps | ✅ Active | ⚠️ #4 |
| 18 | `uv_warp_tv` | `lambda_warp_tv: 0.1` | Total variation (spatial gradients) | 0.0 - 1.0 | Total variation regularization | ✅ Active | ⚠️ #4 |
| 19 | `source_theta_warp_loss` | `lambda_source_theta: 0.5` | MSE(pred_source_theta_warp, target) | 0.0 - 1.0 | Match source theta warp | ✅ Active | None |
| 20 | `warp_temporal_consistency` | `lambda_warp_temporal: 0.1` | Mean(\|diff(warps)\|) | 0.0 - 1.0 | Frame-to-frame warp consistency | ✅ Active | ⚠️ #5 |
| **CONTROL LOSSES** |
| 21 | `control_gaze` | `lambda_gaze_direction: 1.0` | MSE(pred_gaze, target_gaze) | 0.0 - 1.0 | Control gaze direction | ✅ Active | None |
| 22 | `control_distance` | `lambda_head_distance: 0.1` | MSE(pred_distance, target_distance) | 0.0 - 1.0 | Control head distance from camera | ✅ Active | None |
| 23 | `control_emotion` | `lambda_emotion: 0.0` | MSE(pred_emotion, target_emotion) | 0.0 - 1.0 | Control emotion (valence/arousal) | ❌ Disabled | Stage 5 unlock |
| 24 | `speed_loss` | `lambda_speed: 0.1` | CrossEntropy(pred_speed_bucket, target) | 0.0 - 2.0 | Motion speed classification | ✅ Active | None |
| **LIP SYNC LOSSES** |
| 25 | `lips_loss` | `lambda_lips: 2.0` | Position + Velocity loss for lips | 0.0 - 5.0 | Lip landmark matching | ✅ Active | ⚠️ #6 |
| 26 | `nonlip_loss` | `lambda_nonlip: 0.1` | Position loss for non-lip landmarks | 0.0 - 5.0 | Other facial landmark matching | ✅ Active | None |
| 27 | `sync_loss` | `lambda_sync: 0.0` | SyncNet confidence loss | 0.0 - 1.0 | Audio-visual synchronization | ❌ Disabled | Evaluation only |
| 28 | `audio_lip_correlation` | `lambda_audio_lip: 5.0` | MSE(audio_energy, lip_motion) | 0.0 - 1.0 | Force audio energy to correlate with lip motion | ✅ Active | ⚠️ #6 |
| 29 | `audio_expr_coupling` | `lambda_audio_expr_coupling: 5.0` | MSE(expr_magnitude, audio_energy) | 0.0 - 1.0 | Force expression magnitude to correlate with audio energy | ✅ Active | ⚠️ #7 |
| **TEMPORAL SMOOTHNESS LOSSES** |
| 30 | `motion_smoothness` | `lambda_temporal: 0.0` | Temporal consistency for all motion | 0.0 - 1.0 | Overall motion smoothness | ❌ Disabled | Prevents expression variance |
| 31 | `velocity_loss` | `lambda_velocity: 0.0` | MSE(pred_vel, target_vel) | 0.0 - 1.0 | Match velocities (rotation, translation, expression) | ❌ Disabled | Prevents expression variance |
| 32 | `smoothness_loss` | `lambda_smoothness: 0.0` | MSE(pred_acc, target_acc) | 0.0 - 1.0 | Match accelerations (smoothness) | ❌ Disabled | Prevents expression variance |
| **DISENTANGLEMENT LOSSES** |
| 33 | `l_consist` | `lambda_consist: 0.5` | Pairwise transfer consistency | 0.0 - 5.0 | Disentanglement: transfer dynamics to different frames | ✅ Active | None |
| 34 | `l_cross_id` | `lambda_cross_id: 0.0` | Cross-identity similarity loss | 0.0 - 1.0 | Identity preservation across frames | ❌ Disabled | Re-enabled later |
| **PERCEPTUAL/QUALITY LOSSES** |
| 35 | `perceptual_loss` | `lambda_perceptual: 0.5` | LPIPS perceptual distance | 0.0 - 1.0 | Perceptual quality matching | ✅ Active | None |
| 36 | `verification_loss` | `lambda_verification` | Face verification loss | 0.0 - 1.0 | Identity preservation | ✅ Active | None |
| 37 | `blink_loss` | `lambda_blink: 0.0` | Blink detection/matching | 0.0 - 1.0 | Natural blink behavior | ❌ Disabled | Not needed for overfitting |
| **EMO MATCHING (DISABLED)** |
| 38 | `emo_match_loss` | `lambda_emo_match: 0.0` | L1/L2/LPIPS vs EMO output | 0.0 - 5.0 | Match high-quality EMO keyframes | ❌ Disabled | EMO keyframes too static |

---

## CRITICAL CONFLICTS IDENTIFIED

### ⚠️ Conflict #1: Rotation Loss Computation
**Location**: vasa_losses.py line 1706-1709

**Problem**:
```python
losses['rotation_loss'] = F.mse_loss(pred['rotation'], comparison_target['rotation'])
```
- **Prediction**: Euler angles (3D rotation as pitch/yaw/roll)
- **Target**: Also Euler angles
- **Issue**: MSE on Euler angles is NOT rotationally invariant
  - 359° and 1° are close but MSE treats them as distant
  - Gimbal lock causes discontinuities

**Expected Fix**:
- Convert to rotation matrices and use geodesic loss
- Or use quaternion representation with quaternion distance

**Impact**: **MEDIUM** - Can cause training instability near rotation boundaries

---

### ⚠️ Conflict #2: Expression L1 vs L2
**Location**: vasa_losses.py lines 1721-1730

**Problem**:
```python
# L1 loss
losses['expression_loss'] = F.l1_loss(pred, target) * 0.1

# L2 loss
losses['expression_mse'] = F.mse_loss(pred, target) * 0.9
```

**Conflict**:
- **L1 (0.1 weight)**: Encourages **sparsity** (many zeros)
- **L2 (0.9 weight)**: Encourages **smoothness** (penalizes outliers)
- **Combined effect**: L2 dominates (9x stronger), but L1 still pulls toward sparsity

**Expected Behavior**:
- L1 should encourage expressive features to activate strongly (non-zero)
- L2 should smooth out noise

**Actual Behavior**:
- L1 weak (0.1) → minimal sparsity effect
- L2 strong (0.9) → dominates, penalizes large activations
- **Net result**: Expression embeddings stay small → less expressive

**Recommendation**:
- Either use **pure L2** (remove L1 entirely) for expressions
- OR increase L1 to 0.5 if sparsity is desired

**Impact**: **LOW** - Currently L2 dominates (0.9 vs 0.1)

---

### ⚠️ Conflict #3: Expression Variance vs Temporal Variation
**Location**: vasa_losses.py lines 1733-1745

**Problem**:
```python
# Variance loss: forces std(pred_expr) ≈ std(target_expr)
losses['expression_variance_loss'] = MSE(std(pred), std(target)) * 5.0

# Temporal loss: forces diff(pred_expr) ≈ diff(target_expr)
losses['expression_temporal_loss'] = MSE(diff(pred), diff(target)) * 2.0
```

**Conflict**:
- **Variance loss**: Encourages **overall variance** across all frames/features
- **Temporal loss**: Encourages **frame-to-frame changes** to match target
- **Both active**: Can pull in opposite directions

**Scenario**:
1. Variance loss wants: `std(pred_expr) = 0.5` (match target variance)
2. Temporal loss wants: `diff(pred_expr) = 0.01` (match target temporal change)
3. **If target has high variance but low temporal change** → conflict

**Expected Behavior**:
- Variance: Match overall "spread" of expression values
- Temporal: Match frame-to-frame dynamics

**Actual Behavior**:
- Both losses active simultaneously
- Variance loss (5.0) stronger than temporal (2.0)
- **Net result**: Variance dominates, but temporal still adds conflicting gradient

**Recommendation**:
- **Keep variance loss** for preventing collapse
- **Re-evaluate temporal loss** - may be redundant with expression_loss already matching frame-to-frame

**Impact**: **MEDIUM** - May cause jitter if both pull differently

---

### ⚠️ Conflict #4: UV Warp Sparsity vs Magnitude
**Location**: vasa_losses.py lines 1546, 1554, 1571, 1576

**Problem**:
```python
# L1 loss: encourages sparsity (warps → 0)
losses['uv_warp_l1'] = F.l1_loss(pred_warp, target_warp) * 0.1

# Magnitude loss: prevents collapse to zero
losses['uv_warp_magnitude'] = MSE(||pred_warp||, ||target_warp||) * 5.0

# Smoothness: encourages small spatial gradients
losses['uv_warp_smooth'] = tv_l1_loss(pred_warp) * 0.01

# Total variation: penalizes spatial gradients
losses['uv_warp_tv'] = tv_loss(pred_warp) * 0.1
```

**Conflict**:
- **L1 (0.1)**: Pulls warps toward **zero** (sparsity)
- **Magnitude (5.0)**: Pulls warps toward **non-zero** (prevent collapse)
- **Smooth (0.01) + TV (0.1)**: Pulls warps toward **spatially uniform** (small gradients)

**These are contradictory**:
1. L1 wants sparse (many zeros)
2. Magnitude wants non-zero
3. Smooth+TV want spatially uniform

**Expected Behavior**:
- L1: Encourage only necessary warps to be non-zero (sparse facial deformation)
- Magnitude: Ensure warps have sufficient magnitude to deform face
- Smooth: Prevent noisy/jagged warp fields

**Actual Behavior**:
- **Magnitude dominates** (5.0 vs 0.1 L1) → warps won't collapse
- L1 too weak to encourage sparsity effectively
- Smooth+TV compete with magnitude (want uniform, magnitude wants varied)

**Net Result**:
- Warps stay non-zero (magnitude wins)
- But spatially over-smoothed (smooth+TV win)
- **Loss**: Sharp facial details (e.g., lip corners) may blur

**Recommendation**:
- **Remove L1 loss** on warps (conflicts with magnitude, too weak anyway)
- **Keep magnitude loss** (5.0) to prevent collapse
- **Keep smooth OR TV, not both** (redundant, TV is stronger)
  - Suggested: `lambda_warp_smooth: 0.0`, `lambda_warp_tv: 0.05`

**Impact**: **HIGH** - Warps may be over-smoothed, losing fine facial details

---

### ⚠️ Conflict #5: Warp Temporal Consistency vs Warp Reconstruction
**Location**: vasa_losses.py lines 1543, 1593

**Problem**:
```python
# Reconstruction: match target warps exactly
losses['uv_warp_loss'] = MSE(pred_warp, target_warp) * 2.0

# Temporal consistency: minimize frame-to-frame warp changes
losses['warp_temporal_consistency'] = mean(|diff(warps)|) * 0.1
```

**Conflict**:
- **Reconstruction**: Wants warps to **match target exactly** (including temporal variations)
- **Temporal consistency**: Wants warps to **change slowly** over time

**Scenario**:
1. Target has rapid warp change (e.g., sudden mouth opening)
2. Reconstruction loss wants: `pred_warp = target_warp` (match rapid change)
3. Temporal loss wants: `diff(pred_warp) → 0` (smooth change)
4. **Contradiction**: Can't match rapid target AND be smooth

**Expected Behavior**:
- Reconstruction: Match target dynamics
- Temporal: Regularize to prevent jitter

**Actual Behavior**:
- **Reconstruction dominates** (2.0 vs 0.1) → matches target
- Temporal consistency too weak to smooth effectively
- **Net result**: Temporal loss adds noise to gradients without helping

**Recommendation**:
- **If target is already smooth**: Temporal consistency is redundant, remove it
- **If target is janky**: Increase temporal weight to 0.5-1.0 OR remove reconstruction loss

**Current Setup (Overfitting)**:
- Target is single video (should be smooth)
- **Recommendation**: `lambda_warp_temporal: 0.0` (disable, reconstruction handles it)

**Impact**: **LOW** - Temporal too weak (0.1), reconstruction dominates (2.0)

---

### ⚠️ Conflict #6: Lip Losses vs Audio-Lip Correlation
**Location**: vasa_losses.py lines 311, 447

**Problem**:
```python
# Lip landmark loss: match target lip positions exactly
losses['lips_loss'] = lambda_lips * (lips_pos_loss + 0.5 * lips_vel_loss)  # lambda_lips=2.0

# Audio-lip correlation: force lip motion to correlate with audio energy
losses['audio_lip_correlation'] = MSE(audio_energy, lip_motion) * 5.0
```

**Conflict**:
- **Lip loss (2.0)**: Wants lips to **match target positions exactly**
- **Audio-lip correlation (5.0)**: Wants lips to **correlate with audio energy**

**Scenario**:
1. Target has lips moving (lip_motion = 0.8) during silence (audio_energy = 0.0)
2. Lip loss wants: `pred_lips = target_lips` (moving)
3. Audio-lip wants: `lip_motion ≈ audio_energy` (static, since audio is silent)
4. **Contradiction**: Can't match target lips AND correlate with audio if they're misaligned

**Expected Behavior**:
- Lip loss: Supervise with ground truth landmarks
- Audio-lip: Encourage audio-driven motion

**Actual Behavior**:
- **If target is well-synced**: Both losses align → no conflict
- **If target has desync**: Losses fight → training instability

**Recommendation**:
- **For overfitting (single video)**: Target should be synced, both losses work
- **For general training**: May need to choose:
  - Trust target landmarks → keep lip loss, reduce audio-lip
  - Trust audio correlation → reduce lip loss, keep audio-lip

**Current Setup**:
- `lambda_lips: 2.0` (medium)
- `lambda_audio_lip: 5.0` (high - audio correlation dominates)

**Impact**: **MEDIUM** - If target video has slight desync, losses will conflict

---

### ⚠️ Conflict #7: Audio-Expression Coupling vs Expression Reconstruction
**Location**: vasa_losses.py lines 566-567, 1721-1730

**Problem**:
```python
# Expression reconstruction: match target expressions exactly
losses['expression_loss'] = L1(pred_expr, target_expr) * 0.1
losses['expression_mse'] = MSE(pred_expr, target_expr) * 0.9

# Audio-expression coupling: force expression magnitude to match audio energy
losses['audio_expr_coupling'] = MSE(||pred_expr||, audio_energy) * 5.0
```

**Conflict**:
- **Expression reconstruction (1.0 total)**: Wants expressions to **match target exactly**
- **Audio-expression coupling (5.0)**: Wants expression **magnitude** to **correlate with audio energy**

**Scenario**:
1. Target has high expression (||target_expr|| = 0.8) during silence (audio_energy = 0.0)
2. Reconstruction wants: `pred_expr = target_expr` (high expression)
3. Audio coupling wants: `||pred_expr|| ≈ audio_energy` (low expression, since audio is silent)
4. **Contradiction**: Can't match target AND correlate with audio if they're misaligned

**Expected Behavior**:
- Reconstruction: Match ground truth expressions
- Audio coupling: Encourage audio-driven expressions

**Actual Behavior**:
- **Audio coupling dominates** (5.0 vs 1.0 total reconstruction)
- **Net result**: Model learns to correlate with audio MORE than matching target
- **Risk**: If target has expressions during silence → model will suppress them

**Recommendation**:
- **For overfitting (single video)**:
  - If target is well-synced (expressions match audio), both work
  - If target has silent expressions → reduce audio coupling to 1.0
- **For general training**:
  - Reduce audio coupling to 1.0-2.0 (let reconstruction dominate)
  - Or normalize audio energy to prevent collapse

**Current Setup**:
- `lambda_audio_expr_coupling: 5.0` (very high - dominates reconstruction)

**Impact**: **HIGH** - Audio coupling may override target expressions if misaligned

---

## Losses That May Not Make Sense

### 1. ❓ `expression_temporal_loss` (lambda=2.0) - Redundant?
**Location**: vasa_losses.py line 1744

**Issue**:
```python
# Already have expression reconstruction loss
losses['expression_loss'] = L1(pred_expr, target_expr)  # Matches frame-by-frame

# And now temporal loss
losses['expression_temporal_loss'] = MSE(diff(pred_expr), diff(target_expr))
```

**Question**: If `expression_loss` already matches each frame to target, why separately match frame-to-frame differences?

**Analysis**:
- `expression_loss` at frame t: `pred[t] → target[t]`
- `expression_loss` at frame t+1: `pred[t+1] → target[t+1]`
- **Implied**: `diff(pred) = pred[t+1] - pred[t] → target[t+1] - target[t] = diff(target)`

**Conclusion**: Temporal loss is **redundant** if reconstruction loss is working.

**When Useful**: Only if model has temporal instability (jitter) despite matching individual frames.

**Recommendation**: **Monitor** expression_temporal_loss value:
- If always near zero → redundant, disable
- If high → model has temporal jitter, keep enabled

---

### 2. ❓ `warp_temporal_consistency` (lambda=0.1) - Too Weak?
**Location**: vasa_losses.py line 1593

**Issue**:
```python
losses['warp_temporal_consistency'] = temporal_diff.abs().mean() * 0.1
```

**Analysis**:
- Weight is 0.1 (10x weaker than warp reconstruction at 2.0)
- **Effect**: Barely influences training
- **Question**: Is this loss doing anything useful, or just adding gradient noise?

**Recommendation**: Either:
- Increase to 0.5-1.0 (if temporal smoothness is desired)
- Disable (set to 0.0) if reconstruction handles it

---

### 3. ❓ `lambda_warp_l1` (0.1) + `lambda_warp_tv` (0.1) - Redundant?
**Location**: vasa_losses.py lines 1546, 1576

**Issue**:
```python
# L1 encourages sparsity
losses['uv_warp_l1'] = L1(pred_warp, target_warp) * 0.1

# TV encourages spatial smoothness (also uses L1 internally)
losses['uv_warp_tv'] = tv_loss(pred_warp) * 0.1
```

**Analysis**:
- Both use L1 norm
- Both regularize warps
- L1 on values, TV on gradients
- **But**: Both pull toward zero → redundant effect

**Recommendation**:
- Keep **TV only** (0.1) for spatial smoothness
- Disable **L1** (set to 0.0) to avoid redundancy

---

### 4. ❓ `lambda_consist` (0.5) - Computationally Expensive for Overfitting
**Location**: vasa_losses.py line 3204

**Issue**:
- Disentanglement loss computes pairwise transfers (sample 3 pairs × 2 transfers = 6 forward passes)
- **For overfitting**: Single video → identity already disentangled
- **Question**: Is this loss necessary for single-identity overfitting?

**Recommendation**:
- **For overfitting**: Disable (set to 0.0) to save compute
- **For multi-identity training**: Keep enabled (0.5)

---

### 5. ❓ `lambda_audio_lip` (5.0) - May Conflict with Target Lips
**Location**: vasa_losses.py line 447

**Issue**:
- Forces lip motion to correlate with audio energy
- **But**: If target video has slight audio desync → loss fights reconstruction

**Analysis**:
- Weight is 5.0 (2.5x stronger than lip landmark loss at 2.0)
- **Risk**: May override target lip positions if audio misaligned

**Recommendation**:
- **For overfitting**: If target is well-synced, keep at 5.0
- **If seeing lip jitter**: Reduce to 2.0-3.0 to let landmark loss dominate

---

### 6. ❓ `lambda_audio_expr_coupling` (5.0) - Too Strong?
**Location**: vasa_losses.py line 566

**Issue**:
- Forces expression magnitude to match audio energy
- Weight is 5.0 (5x stronger than expression reconstruction at 1.0)
- **Risk**: Model learns audio correlation MORE than matching target expressions

**Analysis**:
- If target has expressions during silence → model will suppress them
- If target has silence during expressions → model will hallucinate expressions

**Recommendation**:
- Reduce to 1.0-2.0 (let expression reconstruction dominate)
- Or add audio energy floor (min 0.1) to prevent collapse during silence

---

## Recommended Loss Tuning

### High Priority Fixes

| Loss | Current | Recommended | Reason |
|------|---------|-------------|--------|
| `lambda_audio_expr_coupling` | 5.0 | **2.0** | Too strong, overrides target expressions |
| `lambda_warp_l1` | 0.1 | **0.0** | Conflicts with magnitude loss, redundant with TV |
| `lambda_warp_temporal` | 0.1 | **0.0** | Too weak, redundant for overfitting (target is smooth) |
| `lambda_consist` | 0.5 | **0.0** | Expensive, not needed for single-identity overfitting |

### Medium Priority Adjustments

| Loss | Current | Recommended | Reason |
|------|---------|-------------|--------|
| `lambda_audio_lip` | 5.0 | **3.0** | Reduce to balance with lip landmark loss (2.0) |
| `lambda_warp_tv` | 0.1 | **0.05** | Reduce after removing warp_l1 |
| `lambda_expression_temporal` | 2.0 | **Monitor** | May be redundant, watch loss value |

### Low Priority (Optional)

| Loss | Current | Recommended | Reason |
|------|---------|-------------|--------|
| `lambda_expression_l1` | 0.1 | **0.0** | Too weak, L2 (0.9) dominates anyway |
| Rotation loss | MSE on Euler | **Geodesic** | Use rotation matrix geodesic distance (future improvement) |

---

## Expected Loss Ranges (Well-Trained Model)

| Loss | Healthy Range | Warning if > | Critical if > |
|------|---------------|--------------|---------------|
| `theta_loss` | 0.001 - 0.05 | 0.1 | 0.5 |
| `rotation_loss` | 0.001 - 0.05 | 0.1 | 0.5 |
| `translation_loss` | 0.001 - 0.05 | 0.1 | 0.5 |
| `expression_loss` | 0.01 - 0.2 | 0.5 | 1.0 |
| `expression_variance_loss` | 0.001 - 0.05 | 0.1 | 0.5 |
| `uv_warp_loss` | 0.01 - 0.5 | 1.0 | 5.0 |
| `uv_warp_magnitude` | 0.001 - 0.1 | 0.2 | 1.0 |
| `lips_loss` | 0.01 - 0.5 | 1.0 | 2.0 |
| `audio_lip_correlation` | 0.001 - 0.1 | 0.2 | 0.5 |
| `audio_expr_coupling` | 0.001 - 0.1 | 0.2 | 0.5 |

---

## Summary of Issues

### Critical Conflicts (Fix Now)
1. ✅ **UV Warp L1 vs Magnitude** - Remove L1 (conflicts, too weak)
2. ✅ **Audio-Expression Coupling Too Strong** - Reduce from 5.0 to 2.0
3. ⚠️ **Warp Temporal vs Reconstruction** - Disable temporal (redundant for overfitting)

### Medium Conflicts (Monitor)
4. ⚠️ **Expression Variance vs Temporal** - May conflict, monitor loss values
5. ⚠️ **Lip Loss vs Audio-Lip** - May conflict if target has desync
6. ⚠️ **Warp Smoothness Overload** - Using both smooth + TV (redundant)

### Low Priority (Future Improvements)
7. ⚠️ **Rotation Loss Non-Invariant** - Use geodesic distance (future)
8. ⚠️ **Expression L1+L2 Mix** - Remove L1 (too weak, redundant)
9. ⚠️ **Disentanglement for Overfitting** - Not needed for single identity

---

## Verification Checklist

Use this to verify losses are behaving correctly:

- [ ] `expression_variance_loss` should **decrease** over training (variance matching target)
- [ ] `expression_std` (logged) should **match** `target_std` (~0.5-0.7 for good expressions)
- [ ] `uv_warp_magnitude` should **stay low** (warps not collapsing to zero)
- [ ] `audio_expr_coupling` should **not dominate** expression_loss (check ratio in WandB)
- [ ] `lips_loss` and `audio_lip_correlation` should **both decrease** (not fight each other)
- [ ] `warp_temporal_consistency` should be **near zero** (target is smooth video)
- [ ] Total loss should **decrease steadily** without oscillations (sign of conflicting losses)

---

**Status**: ✅ Audit complete. 7 conflicts identified, 4 high-priority recommendations.
