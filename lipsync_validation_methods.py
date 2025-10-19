"""
Lip-Sync Validation Helper Methods for VASATrainer

These methods should be added to the VASATrainer class in vasa_trainer.py
Insert after the validate() method (around line 2890) and before _apply_condition_dropout()
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER METHOD 1: Extract Lip Energy from Generated Frames
# ============================================================================

def _extract_lip_energy(self, generated_frames: torch.Tensor) -> torch.Tensor:
    """
    Extract lip motion energy from generated video frames using MediaPipe.

    Args:
        generated_frames: [B, T, C, H, W] generated video frames (RGB, range 0-1)

    Returns:
        lip_energy: [B, T] lip motion energy per frame (normalized 0-1)
    """
    try:
        B, T, C, H, W = generated_frames.shape
        device = generated_frames.device

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        lip_energies = []

        for b in range(B):
            batch_energies = []

            for t in range(T):
                # Get frame [C, H, W] and convert to numpy [H, W, C]
                frame = generated_frames[b, t]  # [C, H, W]
                frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # Convert RGB to BGR for MediaPipe
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                # Detect landmarks
                results = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]

                    # Extract lip landmarks
                    # Upper lip outer: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
                    # Lower lip outer: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
                    # Inner mouth: 78, 191, 80, 81, 82, 13, 312, 311, 310, 415

                    upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
                    lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

                    # Get coordinates
                    upper_lip_y = np.mean([landmarks.landmark[i].y for i in upper_lip_indices])
                    lower_lip_y = np.mean([landmarks.landmark[i].y for i in lower_lip_indices])

                    # Compute vertical opening (normalized to image height)
                    lip_opening = abs(lower_lip_y - upper_lip_y)
                    batch_energies.append(lip_opening)
                else:
                    # No face detected, use 0
                    batch_energies.append(0.0)

            lip_energies.append(batch_energies)

        face_mesh.close()

        # Convert to tensor [B, T]
        lip_energy_tensor = torch.tensor(lip_energies, dtype=torch.float32, device=device)

        return lip_energy_tensor

    except Exception as e:
        logger.warning(f"Error extracting lip energy, returning zeros: {e}")
        return torch.zeros(B, T, device=device)


# ============================================================================
# HELPER METHOD 2: Compute Lip Openness from Lip Landmarks
# ============================================================================

def _compute_lip_openness(self, lips: torch.Tensor) -> torch.Tensor:
    """
    Compute lip openness (vertical opening) from lip landmark coordinates.

    Args:
        lips: [B, T, 40] lip landmarks (20 points x 2 coords = 40 dims)
              Format: [x0, y0, x1, y1, ..., x19, y19]

    Returns:
        openness: [B, T] vertical lip opening per frame
    """
    try:
        B, T, D = lips.shape

        if D != 40:
            logger.warning(f"Expected lips shape [B, T, 40], got {lips.shape}")
            return torch.zeros(B, T, device=lips.device)

        # Reshape to [B, T, 20, 2] (20 points with x, y)
        lips_reshaped = lips.reshape(B, T, 20, 2)

        # Extract y-coordinates
        y_coords = lips_reshaped[:, :, :, 1]  # [B, T, 20]

        # Compute vertical span (max_y - min_y for upper and lower lips)
        # Assuming first 10 points = upper lip, last 10 = lower lip
        upper_lip_y = y_coords[:, :, :10]  # [B, T, 10]
        lower_lip_y = y_coords[:, :, 10:]  # [B, T, 10]

        # Compute opening as distance between mean upper and mean lower
        upper_mean = upper_lip_y.mean(dim=2)  # [B, T]
        lower_mean = lower_lip_y.mean(dim=2)  # [B, T]

        openness = torch.abs(lower_mean - upper_mean)  # [B, T]

        return openness

    except Exception as e:
        logger.warning(f"Error computing lip openness: {e}")
        return torch.zeros(B, T, device=lips.device)


# ============================================================================
# HELPER METHOD 3: Extract Per-Frame Lip Motion
# ============================================================================

def _extract_lip_motion_per_frame(self, generated_frames: torch.Tensor) -> torch.Tensor:
    """
    Extract per-frame lip motion magnitude from generated frames.

    Args:
        generated_frames: [B, T, C, H, W] generated video frames

    Returns:
        lip_motion: [B, T] lip motion magnitude per frame
    """
    try:
        # Extract lip energy (openness) for each frame
        lip_energy = self._extract_lip_energy(generated_frames)  # [B, T]

        # Compute temporal derivative (motion = change in openness)
        # Use finite difference: motion[t] = |energy[t] - energy[t-1]|
        B, T = lip_energy.shape

        lip_motion = torch.zeros_like(lip_energy)

        if T > 1:
            # Compute differences
            diffs = torch.abs(lip_energy[:, 1:] - lip_energy[:, :-1])  # [B, T-1]

            # Set first frame motion to 0, rest to diffs
            lip_motion[:, 1:] = diffs

        return lip_motion

    except Exception as e:
        logger.warning(f"Error extracting per-frame lip motion: {e}")
        B, T = generated_frames.shape[:2]
        return torch.zeros(B, T, device=generated_frames.device)


# ============================================================================
# METRIC METHOD 1: Mel Spectrogram Sync Metrics
# ============================================================================

def _compute_mel_sync_metrics(
    self,
    generated_frames: torch.Tensor,
    audio_segment: torch.Tensor,
    window_metadata: Dict
) -> Dict[str, float]:
    """
    Compute mel spectrogram-based synchronization metrics.

    Args:
        generated_frames: [B, T, C, H, W] generated video frames
        audio_segment: [B, audio_samples] raw audio waveform
        window_metadata: Dict with timing info

    Returns:
        metrics: Dict with mel_correlation, mel_mse metrics
    """
    try:
        import torchaudio

        B, T, C, H, W = generated_frames.shape
        device = generated_frames.device

        # Get sample rate from config or default to 16000
        sample_rate = getattr(self.config.audio, 'sample_rate', 16000)

        # Create mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=audio_segment.shape[-1] // T,  # Match video frames
            n_mels=128,
            f_min=0.0,
            f_max=8000.0
        ).to(device)

        # Compute mel spectrogram [B, 128, T]
        mel_spec = mel_transform(audio_segment)
        mel_spec_log = torch.log(mel_spec + 1e-6)  # Log scale

        # Extract lip energy from frames
        lip_energy = self._extract_lip_energy(generated_frames)  # [B, T]

        # Compute audio energy from mel spectrogram (mean over frequency bins)
        audio_energy = mel_spec_log.mean(dim=1)  # [B, T]

        # Normalize both signals to zero mean, unit variance
        lip_energy_norm = (lip_energy - lip_energy.mean(dim=1, keepdim=True)) / \
                          (lip_energy.std(dim=1, keepdim=True) + 1e-6)
        audio_energy_norm = (audio_energy - audio_energy.mean(dim=1, keepdim=True)) / \
                            (audio_energy.std(dim=1, keepdim=True) + 1e-6)

        # Metric 1: Cross-correlation (cosine similarity)
        correlation = F.cosine_similarity(
            lip_energy_norm,
            audio_energy_norm,
            dim=1
        ).mean()

        # Metric 2: MSE between normalized energies
        mse = F.mse_loss(lip_energy_norm, audio_energy_norm)

        return {
            'val/mel_lip_correlation': correlation.item(),
            'val/mel_lip_mse': mse.item(),
        }

    except Exception as e:
        logger.error(f"Error computing mel sync metrics: {e}")
        return {
            'val/mel_lip_correlation': 0.0,
            'val/mel_lip_mse': 0.0,
        }


# ============================================================================
# METRIC METHOD 2: Audio Projection Quality Metrics
# ============================================================================

def _compute_audio_projection_metrics(
    self,
    generated_sequence: Dict[str, torch.Tensor],
    audio_features: torch.Tensor,
    window: Dict
) -> Dict[str, float]:
    """
    Measure audio-to-visual projection quality.

    Compares:
    - Audio features -> Expected lip motion (from model)
    - Generated lip motion -> Ground truth lips

    Args:
        generated_sequence: Dict with 'lips' key [B, T, 40]
        audio_features: [B, T, D] audio features (wav2vec2 or MFCC)
        window: Dict with 'lips' ground truth

    Returns:
        metrics: Dict with audio projection quality metrics
    """
    try:
        # Extract predicted and target lips
        pred_lips = generated_sequence.get('lips')
        target_lips = window.get('lips')

        if pred_lips is None or target_lips is None:
            logger.debug("Skipping audio projection metrics (no lips data)")
            return {}

        B, T, D_audio = audio_features.shape
        device = audio_features.device

        # Compute audio energy (L2 norm over feature dimension)
        audio_energy = audio_features.norm(dim=-1)  # [B, T]

        # Compute lip openness for predicted and target
        pred_lip_openness = self._compute_lip_openness(pred_lips)  # [B, T]
        target_lip_openness = self._compute_lip_openness(target_lips)  # [B, T]

        # Normalize all signals
        audio_norm = (audio_energy - audio_energy.mean(dim=1, keepdim=True)) / \
                     (audio_energy.std(dim=1, keepdim=True) + 1e-6)
        pred_norm = (pred_lip_openness - pred_lip_openness.mean(dim=1, keepdim=True)) / \
                    (pred_lip_openness.std(dim=1, keepdim=True) + 1e-6)
        target_norm = (target_lip_openness - target_lip_openness.mean(dim=1, keepdim=True)) / \
                      (target_lip_openness.std(dim=1, keepdim=True) + 1e-6)

        # Metric 1: Audio-to-prediction correlation
        audio_pred_corr = F.cosine_similarity(audio_norm, pred_norm, dim=1).mean()

        # Metric 2: Audio-to-target correlation (ideal case)
        audio_target_corr = F.cosine_similarity(audio_norm, target_norm, dim=1).mean()

        # Metric 3: Projection error (how far off is prediction from ideal)
        projection_error = audio_target_corr - audio_pred_corr

        # Metric 4: Prediction accuracy (how close is pred to target)
        pred_target_corr = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()

        return {
            'val/audio_pred_correlation': audio_pred_corr.item(),
            'val/audio_target_correlation': audio_target_corr.item(),
            'val/audio_projection_error': projection_error.item(),
            'val/pred_target_correlation': pred_target_corr.item(),
        }

    except Exception as e:
        logger.error(f"Error computing audio projection metrics: {e}")
        return {}


# ============================================================================
# METRIC METHOD 3: Temporal Alignment Metrics
# ============================================================================

def _compute_temporal_alignment_metrics(
    self,
    generated_frames: torch.Tensor,
    audio_features: torch.Tensor,
    fps: int = 25
) -> Dict[str, float]:
    """
    Compute frame-by-frame alignment metrics with lag detection.

    Uses cross-correlation with time lags to detect sync offset.

    Args:
        generated_frames: [B, T, C, H, W] generated video frames
        audio_features: [B, T, D] audio features
        fps: Frames per second (default 25)

    Returns:
        metrics: Dict with sync lag and correlation metrics
    """
    try:
        B, T = audio_features.shape[:2]
        device = audio_features.device

        # Extract per-frame audio energy
        audio_energy = audio_features.norm(dim=-1)  # [B, T]

        # Extract per-frame lip motion
        lip_motion = self._extract_lip_motion_per_frame(generated_frames)  # [B, T]

        # Compute cross-correlation with time lags (-5 to +5 frames)
        max_lag = min(5, T // 4)  # Don't exceed T/4
        correlations = []

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Audio ahead of lips
                shifted_audio = audio_energy[:, :lag]
                shifted_lip = lip_motion[:, -lag:]
            elif lag > 0:
                # Lips ahead of audio
                shifted_audio = audio_energy[:, lag:]
                shifted_lip = lip_motion[:, :-lag]
            else:
                # No lag
                shifted_audio = audio_energy
                shifted_lip = lip_motion

            # Normalize
            audio_norm = (shifted_audio - shifted_audio.mean(dim=1, keepdim=True)) / \
                         (shifted_audio.std(dim=1, keepdim=True) + 1e-6)
            lip_norm = (shifted_lip - shifted_lip.mean(dim=1, keepdim=True)) / \
                       (shifted_lip.std(dim=1, keepdim=True) + 1e-6)

            # Compute correlation at this lag
            corr = F.cosine_similarity(audio_norm, lip_norm, dim=1).mean()
            correlations.append(corr.item())

        # Find best lag (peak correlation)
        best_lag_idx = int(np.argmax(correlations))
        best_lag = best_lag_idx - max_lag  # Convert to actual lag
        best_corr = correlations[best_lag_idx]

        # Lag in milliseconds
        lag_ms = (best_lag / fps) * 1000

        # Correlation at zero lag
        zero_lag_corr = correlations[max_lag]

        return {
            'val/sync_lag_frames': float(best_lag),
            'val/sync_lag_ms': lag_ms,
            'val/sync_peak_correlation': best_corr,
            'val/sync_at_zero_lag': zero_lag_corr,
        }

    except Exception as e:
        logger.error(f"Error computing temporal alignment metrics: {e}")
        return {}


# ============================================================================
# USAGE EXAMPLE: Insert into VASATrainer.validate() method
# ============================================================================

"""
# After line 2856 in vasa_trainer.py (after generated_frames = ...)

                                        generated_frames = self._generate_synced_frames(
                                            window['frames'][:, 0],
                                            generated_sequence
                                        )

                                        # === LIP-SYNC VALIDATION METRICS ===
                                        # 1. Mel spectrogram sync
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_mel_correlation', True):
                                            mel_metrics = self._compute_mel_sync_metrics(
                                                generated_frames,
                                                window.get('audio_segment', window.get('audio_features')),
                                                window.get('metadata', {})
                                            )
                                            metrics.update(mel_metrics)

                                        # 2. Audio projection quality
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_audio_projection', True):
                                            proj_metrics = self._compute_audio_projection_metrics(
                                                generated_sequence,
                                                window['audio_features'],
                                                window
                                            )
                                            metrics.update(proj_metrics)

                                        # 3. Temporal alignment
                                        if hasattr(self.config.validation, 'lipsync') and \
                                           getattr(self.config.validation.lipsync, 'compute_temporal_alignment', True):
                                            align_metrics = self._compute_temporal_alignment_metrics(
                                                generated_frames,
                                                window['audio_features'],
                                                fps=25
                                            )
                                            metrics.update(align_metrics)

                                        # Evaluate sync quality (existing code)
                                        sync_metrics = self.loss_module.evaluate_sync_quality(
                                            generated_frames=generated_frames,
                                            audio_features=window['audio_features'],
                                            audio_mfcc=window.get('audio_mfcc')
                                        )
                                        metrics.update(sync_metrics)
"""
