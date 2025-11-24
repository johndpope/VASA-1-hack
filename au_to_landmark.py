"""
AU-to-Landmark Prediction Module

Based on: "Talking Head Generation via AU-Guided Landmark Prediction"
Chang et al., 2025

This module implements a Variational Motion Generator (VMG) that maps
audio features and Action Unit intensities to 2D facial landmark sequences.

Key insight: Explicit AU→Landmark→Video pipeline provides better geometric
grounding than implicit AU→Video mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class DilatedConvBlock(nn.Module):
    """
    Dilated 1D convolution block for temporal modeling.

    Uses exponentially increasing dilation rates to capture
    long-range temporal dependencies efficiently.
    """
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - batch, time, feature_dim
        Returns:
            [B, T, D] - processed features
        """
        # Conv1d expects [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.norm(x)
        x = self.relu(x)
        return x


class AUToLandmarkVAE(nn.Module):
    """
    Variational Motion Generator (VMG) from:
    'Talking Head Generation via AU-Guided Landmark Prediction'

    Architecture:
        1. Input: Audio features + AU intensities
        2. Encoder: Stack of dilated conv blocks
        3. VAE bottleneck: mu, logvar for latent distribution
        4. Decoder: Stack of dilated conv blocks
        5. Output: 2D facial landmarks (x, y) for each keypoint

    This provides explicit geometric scaffolding for AU-driven animation,
    improving temporal coherence and expression accuracy.
    """

    def __init__(
        self,
        audio_dim: int = 768,  # wav2vec2/HuBERT features
        au_dim: int = 16,      # 16 Action Units
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_landmarks: int = 68,  # 68-point facial landmarks (dlib format)
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.latent_dim = latent_dim

        logger.info(f"Initializing AUToLandmarkVAE:")
        logger.info(f"  Audio dim: {audio_dim}, AU dim: {au_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
        logger.info(f"  Num landmarks: {num_landmarks}, Layers: {num_layers}")

        # Input projection
        input_dim = audio_dim + au_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Encoder: Stack of dilated convolutions with exponentially increasing dilation
        # Dilation rates: [1, 2, 4, 8, 16, 32] for receptive field growth
        self.encoder_blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_layers)
        ])

        # VAE bottleneck: Predict mu and logvar for latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Mirror encoder with decreasing dilation rates
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ])

        # Output projection: 2D landmarks (x, y) for each keypoint
        # Normalized coordinates in [0, 1] range
        self.output_proj = nn.Linear(hidden_dim, num_landmarks * 2)

        # Initialize output layer to small values for stable training
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: [B, T, input_dim] - concatenated audio + AU features
        Returns:
            mu: [B, T, latent_dim] - mean of latent distribution
            logvar: [B, T, latent_dim] - log variance of latent distribution
        """
        x = self.input_proj(x)  # [B, T, hidden_dim]
        x = self.dropout(x)

        # Apply dilated convolution blocks
        for block in self.encoder_blocks:
            x = block(x)

        mu = self.fc_mu(x)      # [B, T, latent_dim]
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sigma

        Args:
            mu: [B, T, latent_dim] - mean
            logvar: [B, T, latent_dim] - log variance
        Returns:
            z: [B, T, latent_dim] - sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to landmark sequences.

        Args:
            z: [B, T, latent_dim] - latent code
        Returns:
            landmarks: [B, T, num_landmarks, 2] - predicted 2D landmarks
        """
        x = self.latent_proj(z)  # [B, T, hidden_dim]

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Project to landmarks
        landmarks_flat = self.output_proj(x)  # [B, T, num_landmarks*2]

        # Reshape to [B, T, num_landmarks, 2]
        B, T = landmarks_flat.shape[:2]
        landmarks = landmarks_flat.view(B, T, self.num_landmarks, 2)

        # Apply sigmoid to ensure normalized coordinates [0, 1]
        landmarks = torch.sigmoid(landmarks)

        return landmarks

    def forward(
        self,
        audio: torch.Tensor,
        aus: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: (audio, AUs) → landmarks

        Args:
            audio: [B, T, 768] - audio features (wav2vec2/HuBERT)
            aus: [B, T, 16] - AU intensity values [0-1]
            deterministic: If True, use mean (mu) instead of sampling (for inference)
        Returns:
            landmarks: [B, T, num_landmarks, 2] - predicted 2D landmarks
            mu: [B, T, latent_dim] - latent mean
            logvar: [B, T, latent_dim] - latent log variance
        """
        # Concatenate audio and AU features
        x = torch.cat([audio, aus], dim=-1)  # [B, T, 768+16]

        # Encode to latent distribution
        mu, logvar = self.encode(x)

        # Sample latent code (or use mean for deterministic inference)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        # Decode to landmarks
        landmarks = self.decode(z)

        return landmarks, mu, logvar


def vae_loss(
    recon_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss: Reconstruction + KL divergence

    Args:
        recon_landmarks: [B, T, num_landmarks, 2] - predicted landmarks
        target_landmarks: [B, T, num_landmarks, 2] - ground truth landmarks
        mu: [B, T, latent_dim] - latent mean
        logvar: [B, T, latent_dim] - latent log variance
        kl_weight: Weight for KL divergence term
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction (MSE) loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss: MSE on landmark positions
    recon_loss = F.mse_loss(recon_landmarks, target_landmarks, reduction='mean')

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize by batch size and time steps
    kl_loss = kl_loss / (mu.shape[0] * mu.shape[1])

    # Combined loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def extract_facial_landmarks_68pt(
    frames: np.ndarray,
    use_mediapipe: bool = True
) -> torch.Tensor:
    """
    Extract 68-point facial landmarks from frames.

    Uses MediaPipe Face Mesh with a mapping to the standard 68-point
    landmark convention (similar to dlib).

    Args:
        frames: [T, H, W, 3] - RGB frames in uint8 format
        use_mediapipe: Use MediaPipe (True) or fallback to zeros
    Returns:
        landmarks: [T, 68, 2] - normalized landmark coordinates [0, 1]
    """
    if not use_mediapipe:
        # Return zeros if MediaPipe not available
        T = len(frames)
        return torch.zeros(T, 68, 2, dtype=torch.float32)

    try:
        import mediapipe as mp

        # MediaPipe logs suppressed by logger.py (imported at module level)
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe to 68-point mapping (approximate)
        # This maps MediaPipe's 468 landmarks to standard 68-point format
        LANDMARK_INDICES = [
            # Jaw line: 0-16 (17 points)
            127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
            # Right eyebrow: 17-21 (5 points)
            70, 63, 105, 66, 107,
            # Left eyebrow: 22-26 (5 points)
            336, 296, 334, 293, 300,
            # Nose bridge: 27-30 (4 points)
            168, 6, 197, 195,
            # Nose base: 31-35 (5 points)
            5, 4, 19, 94, 2,
            # Right eye: 36-41 (6 points)
            33, 160, 159, 158, 133, 153,
            # Left eye: 42-47 (6 points)
            362, 385, 386, 387, 263, 373,
            # Outer mouth: 48-59 (12 points)
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
            # Inner mouth: 60-67 (8 points)
            78, 95, 88, 178, 87, 14, 317, 402
        ]

        landmarks_list = []

        for frame in frames:
            # Process frame
            results = mp_face_mesh.process(frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Extract 68 keypoints
                landmarks_2d = []
                for idx in LANDMARK_INDICES:
                    lm = face_landmarks.landmark[idx]
                    landmarks_2d.append([lm.x, lm.y])

                landmarks_list.append(np.array(landmarks_2d))
            else:
                # No face detected - use previous frame or zeros
                if len(landmarks_list) > 0:
                    landmarks_list.append(landmarks_list[-1])  # Repeat last frame
                else:
                    landmarks_list.append(np.zeros((68, 2)))

        mp_face_mesh.close()

        # Convert to tensor
        landmarks_tensor = torch.tensor(
            np.stack(landmarks_list),
            dtype=torch.float32
        )  # [T, 68, 2]

        return landmarks_tensor

    except Exception as e:
        logger.warning(f"Failed to extract landmarks with MediaPipe: {e}")
        T = len(frames)
        return torch.zeros(T, 68, 2, dtype=torch.float32)


def visualize_landmarks_on_frame(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2
) -> np.ndarray:
    """
    Visualize 68-point landmarks on a frame for debugging.

    Args:
        frame: [H, W, 3] - RGB image
        landmarks: [68, 2] - normalized landmark coordinates [0, 1]
        color: RGB color for landmarks
        radius: Radius of landmark circles
    Returns:
        frame_vis: Frame with landmarks drawn
    """
    import cv2

    frame_vis = frame.copy()
    H, W = frame.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= W
    landmarks_px[:, 1] *= H
    landmarks_px = landmarks_px.astype(np.int32)

    # Draw landmarks
    for (x, y) in landmarks_px:
        cv2.circle(frame_vis, (x, y), radius, color, -1)

    return frame_vis


if __name__ == "__main__":
    # Test the AU-to-Landmark VAE
    logger.info("Testing AUToLandmarkVAE...")

    # Create model
    model = AUToLandmarkVAE(
        audio_dim=768,
        au_dim=16,
        hidden_dim=256,
        latent_dim=128,
        num_landmarks=68,
        num_layers=6
    )

    # Test inputs
    B, T = 2, 50
    audio = torch.randn(B, T, 768)
    aus = torch.rand(B, T, 16)  # AU intensities [0-1]

    # Forward pass
    landmarks_pred, mu, logvar = model(audio, aus)

    logger.info(f"Input shapes:")
    logger.info(f"  Audio: {audio.shape}")
    logger.info(f"  AUs: {aus.shape}")
    logger.info(f"Output shapes:")
    logger.info(f"  Landmarks: {landmarks_pred.shape}")
    logger.info(f"  Mu: {mu.shape}")
    logger.info(f"  Logvar: {logvar.shape}")

    # Test loss
    landmarks_gt = torch.rand(B, T, 68, 2)
    loss, recon_loss, kl_loss = vae_loss(landmarks_pred, landmarks_gt, mu, logvar)

    logger.info(f"Losses:")
    logger.info(f"  Total: {loss.item():.6f}")
    logger.info(f"  Reconstruction: {recon_loss.item():.6f}")
    logger.info(f"  KL: {kl_loss.item():.6f}")

    logger.info("✅ AUToLandmarkVAE test passed!")
