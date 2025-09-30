"""
Synchformer Wrapper for VASA
Integrates Synchformer model as a replacement for SyncNet for audio-visual synchronization.

Configuration:
    Set SYNCHFORMER_PATH environment variable to override default Synchformer location.
    Set SYNCHFORMER_CHECKPOINT to specify checkpoint file path.
    Set SYNCHFORMER_CONFIG to specify config file path.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import einops
import json

# Dynamically find and add Synchformer to path
import os

def get_synchformer_path():
    """Find Synchformer installation path."""
    # Try multiple possible locations
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Check if it's a submodule in the current project
    submodule_path = os.path.join(current_dir, 'Synchformer')
    if os.path.exists(submodule_path):
        return submodule_path

    # 2. Check parent directory
    parent_path = os.path.join(os.path.dirname(current_dir), 'Synchformer')
    if os.path.exists(parent_path):
        return parent_path

    # 3. Check environment variable
    env_path = os.environ.get('SYNCHFORMER_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    # 4. Default fallback paths
    fallback_paths = [
        '/media/12TB/Synchformer',
        os.path.expanduser('~/Synchformer'),
    ]

    for path in fallback_paths:
        if os.path.exists(path):
            return path

    raise RuntimeError(
        "Synchformer not found! Please either:\n"
        "1. Run 'git submodule update --init --recursive' to initialize the submodule\n"
        "2. Set SYNCHFORMER_PATH environment variable\n"
        "3. Clone Synchformer to one of the expected locations"
    )

synchformer_path = get_synchformer_path()

# Store original path to restore later
_original_syspath = sys.path.copy()

def _import_synchformer_modules():
    """Import Synchformer modules with proper path management."""
    import warnings

    # Save current sys.path
    original_path = sys.path.copy()

    try:
        # Add Synchformer to the front of path
        if synchformer_path not in sys.path:
            sys.path.insert(0, synchformer_path)

        # Try to import the Synchformer modules
        from model.sync_model import Synchformer
        from scripts.train_utils import get_model
        from utils.utils import instantiate_from_config

        # Try to import dataset transforms, but create fallbacks if not available
        try:
            from dataset.transforms import AudioMelSpectrogram, AudioLog, AudioNormalizeAST, PadOrTruncate
        except ImportError:
            # Create fallback implementations using torchaudio
            warnings.warn("Using fallback audio transforms as Synchformer transforms not available")
            AudioMelSpectrogram = None
            AudioLog = None
            AudioNormalizeAST = None
            PadOrTruncate = None

        return Synchformer, get_model, instantiate_from_config, AudioMelSpectrogram, AudioLog, AudioNormalizeAST, PadOrTruncate

    except ImportError as e:
        # Restore original path before raising
        sys.path = original_path
        raise ImportError(
            f"Failed to import Synchformer modules: {e}\n"
            f"Please ensure Synchformer is properly installed with:\n"
            f"  git submodule update --init --recursive"
        )


def load_synchformer_config():
    """
    Load Synchformer configuration from environment or config file.

    Returns dict with configuration options.
    """
    config = {
        'path': synchformer_path,
        'checkpoint': os.environ.get('SYNCHFORMER_CHECKPOINT'),
        'config': os.environ.get('SYNCHFORMER_CONFIG'),
        'device': os.environ.get('SYNCHFORMER_DEVICE', 'cuda'),
    }

    # Try to load from a JSON config file if it exists
    config_file = os.path.join(os.path.dirname(__file__), 'synchformer_config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded Synchformer config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")

    return config

# Delay imports until needed to avoid circular dependencies
logger = logging.getLogger(__name__)


class SynchformerWrapper(nn.Module):
    """
    Wrapper for Synchformer model to replace SyncNet in VASA.
    Handles preprocessing of video frames and audio to match Synchformer's expected input format.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.synchformer_base = synchformer_path

        # Import Synchformer modules with proper path management
        (Synchformer, get_model, instantiate_from_config,
         AudioMelSpectrogram, AudioLog, AudioNormalizeAST, PadOrTruncate) = _import_synchformer_modules()

        # Store imported modules as instance attributes for later use
        self._instantiate_from_config = instantiate_from_config
        self._AudioMelSpectrogram = AudioMelSpectrogram

        # Load Synchformer config
        if config_path is None:
            config_path = os.path.join(synchformer_path, 'configs', 'sync.yaml')
        self.config = OmegaConf.load(config_path)

        # Initialize model
        logger.info("Initializing Synchformer model...")
        self.model = self._instantiate_from_config(self.config.model)

        # Load checkpoint
        checkpoint_loaded = False

        # If checkpoint_path not provided, try to find one automatically
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint()

        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading Synchformer checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.info("Checkpoint loaded successfully")
                checkpoint_loaded = True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")

        if not checkpoint_loaded:
            logger.warning("No checkpoint loaded - using random initialization")

        self.model = self.model.to(device)
        self.model.eval()

        # Audio preprocessing parameters from Synchformer config
        self.audio_sample_rate = 16000  # Synchformer expects 16kHz audio
        self.n_mels = 128
        self.n_fft = 1024
        self.win_length = 400  # 25ms at 16kHz
        self.hop_length = 160  # 10ms at 16kHz
        self.max_spec_t = 66  # Time dimension for spectrogram

        # Audio transforms
        if self._AudioMelSpectrogram is not None:
            self.mel_spec_transform = self._AudioMelSpectrogram(
                sample_rate=self.audio_sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels
            )
        else:
            # Use torchaudio fallback
            self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels
            )

        # AST normalization parameters
        self.audio_mean = -4.2677393
        self.audio_std = 4.5689974

        # Visual normalization (MotionFormer uses [-1, 1])
        self.visual_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
        self.visual_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)

        # Expected input shapes
        self.segment_size_vframes = 16  # 16 frames per segment
        self.n_segments = 14  # 14 segments
        self.input_size = 224  # Spatial resolution

    def _find_checkpoint(self) -> Optional[str]:
        """
        Try to find a Synchformer checkpoint in common locations.

        Returns:
            Path to checkpoint file, or None if not found
        """
        possible_paths = [
            # Relative to Synchformer base directory
            os.path.join(self.synchformer_base, 'checkpoints', 'sync_model.pt'),
            os.path.join(self.synchformer_base, 'pretrained', 'synchformer.pt'),
            os.path.join(self.synchformer_base, 'pretrained', 'sync_model.pt'),
            os.path.join(self.synchformer_base, 'logs', 'sync_models', 'best.pt'),
            os.path.join(self.synchformer_base, 'weights', 'synchformer.pt'),

            # Check environment variable
            os.environ.get('SYNCHFORMER_CHECKPOINT', ''),

            # Common download locations
            os.path.expanduser('~/Downloads/synchformer.pt'),
            os.path.expanduser('~/Downloads/sync_model.pt'),
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"Found checkpoint at: {path}")
                return path

        logger.warning(
            "No checkpoint found automatically. Please provide checkpoint_path or download a "
            "pre-trained model to one of these locations:\n" +
            "\n".join(f"  - {p}" for p in possible_paths[:5] if p)
        )
        return None

    def preprocess_video(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess video frames for Synchformer.

        Args:
            frames: [B, T, C, H, W] or [B, C, T, H, W] - video frames

        Returns:
            frames: [B, S, Tv, C, H, W] - segmented video frames
        """
        # Ensure frames are in [B, T, C, H, W] format
        if frames.dim() == 5 and frames.shape[1] == 3:  # [B, C, T, H, W]
            frames = frames.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]

        B, T, C, H, W = frames.shape

        # Resize to expected input size if needed
        if H != self.input_size or W != self.input_size:
            frames = F.interpolate(
                frames.view(B * T, C, H, W),
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            ).view(B, T, C, self.input_size, self.input_size)

        # Normalize to [-1, 1] range expected by MotionFormer
        if frames.max() > 1.0:  # Assume [0, 255] range
            frames = frames / 255.0

        # Apply MotionFormer normalization
        self.visual_mean = self.visual_mean.to(frames.device)
        self.visual_std = self.visual_std.to(frames.device)
        frames = (frames - self.visual_mean) / self.visual_std

        # Segment video into chunks
        # We need S segments of Tv frames each
        S = self.n_segments
        Tv = self.segment_size_vframes

        # Ensure we have enough frames
        min_frames = S * Tv
        if T < min_frames:
            # Pad with last frame if needed
            pad_frames = min_frames - T
            last_frame = frames[:, -1:].expand(-1, pad_frames, -1, -1, -1)
            frames = torch.cat([frames, last_frame], dim=1)
            T = min_frames
        elif T > min_frames:
            # Truncate if too many frames
            frames = frames[:, :min_frames]
            T = min_frames

        # Reshape to segments: [B, S*Tv, C, H, W] -> [B, S, Tv, C, H, W]
        frames = frames.view(B, S, Tv, C, self.input_size, self.input_size)

        return frames

    def preprocess_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for Synchformer.

        Args:
            audio_features: Audio input - either raw waveform, MFCC, or mel spectrogram

        Returns:
            spec: [B, S, 1, F, Ta] - segmented audio spectrograms
        """
        B = audio_features.shape[0]
        device = audio_features.device

        # Handle different audio input types
        if audio_features.dim() == 2:  # [B, samples] - raw waveform
            # Convert to mel spectrogram
            if hasattr(self.mel_spec_transform, '__call__'):
                # Torchaudio transform
                spec = self.mel_spec_transform(audio_features)
            else:
                # Synchformer transform expects dict
                spec = self.mel_spec_transform({'audio': audio_features})['audio']
            spec = torch.log(spec + 1e-8)  # Log scale

        elif audio_features.dim() == 3:
            if audio_features.shape[-1] > 1000:  # Likely raw waveform [B, 1, samples]
                audio_features = audio_features.squeeze(1)  # [B, samples]
                if hasattr(self.mel_spec_transform, '__call__'):
                    # Torchaudio transform
                    spec = self.mel_spec_transform(audio_features)
                else:
                    # Synchformer transform expects dict
                    spec = self.mel_spec_transform({'audio': audio_features})['audio']
                spec = torch.log(spec + 1e-8)
            else:  # Already a spectrogram [B, F, T] or [B, T, F]
                if audio_features.shape[1] > audio_features.shape[2]:
                    # [B, F, T] format
                    spec = audio_features
                else:
                    # [B, T, F] format - transpose
                    spec = audio_features.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected audio shape: {audio_features.shape}")

        # Ensure spec is [B, F, T]
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)  # Add batch dim if needed

        # Pad or truncate time dimension
        F, Ta = spec.shape[1], spec.shape[2]
        target_Ta = self.max_spec_t * self.n_segments  # Total time needed

        if Ta < target_Ta:
            # Pad with zeros
            padding = target_Ta - Ta
            spec = F.pad(spec, (0, padding), mode='constant', value=0)
        elif Ta > target_Ta:
            # Truncate
            spec = spec[:, :, :target_Ta]

        # Apply AST normalization
        spec = (spec - self.audio_mean) / self.audio_std

        # Segment audio: [B, F, S*Ta] -> [B, S, F, Ta]
        S = self.n_segments
        Ta_per_seg = self.max_spec_t
        spec = spec.view(B, F, S, Ta_per_seg).permute(0, 2, 1, 3)  # [B, S, F, Ta]

        # Add channel dimension: [B, S, F, Ta] -> [B, S, 1, F, Ta]
        spec = spec.unsqueeze(2)

        return spec

    def compute_sync_score(
        self,
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Compute synchronization score between video and audio.

        Args:
            video_frames: Video frames [B, T, C, H, W] or [B, C, T, H, W]
            audio_features: Audio features (waveform, MFCC, or spectrogram)
            return_logits: If True, return raw logits; if False, return probability

        Returns:
            sync_score: Synchronization score (higher = better sync)
        """
        with torch.no_grad():
            # Preprocess inputs
            vis = self.preprocess_video(video_frames)  # [B, S, Tv, C, H, W]
            aud = self.preprocess_audio(audio_features)  # [B, S, 1, F, Ta]

            # Forward through Synchformer
            # The model returns (loss, logits) but we only need logits
            _, logits = self.model(vis, aud, targets=None, for_loop=False)

            if return_logits:
                return logits
            else:
                # Convert to probabilities
                # Synchformer outputs logits for offset classification
                # The center class (index 10 for 21 classes) represents perfect sync
                probs = F.softmax(logits, dim=-1)

                # Get probability of being in sync (center class)
                center_idx = logits.shape[-1] // 2  # Middle class = perfect sync
                sync_prob = probs[:, center_idx]

                return sync_prob

    def compute_sync_loss(
        self,
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
        encourage_sync: bool = True
    ) -> torch.Tensor:
        """
        Compute sync loss for training.

        Args:
            video_frames: Generated video frames [B, T, C, H, W]
            audio_features: Corresponding audio features
            encourage_sync: If True, loss encourages sync; if False, it's just evaluation

        Returns:
            loss: Sync loss to minimize
        """
        # Get sync logits
        logits = self.compute_sync_score(video_frames, audio_features, return_logits=True)

        if encourage_sync:
            # Create target labels for perfect sync (center class)
            B = logits.shape[0]
            center_idx = logits.shape[-1] // 2
            targets = torch.full((B,), center_idx, dtype=torch.long, device=logits.device)

            # Cross-entropy loss to encourage sync
            loss = F.cross_entropy(logits, targets)
        else:
            # Just return negative sync probability as loss (for monitoring)
            probs = F.softmax(logits, dim=-1)
            center_idx = logits.shape[-1] // 2
            sync_prob = probs[:, center_idx]
            loss = 1.0 - sync_prob.mean()  # Higher sync prob = lower loss

        return loss


# Compatibility class to match SyncNetInstance interface
class SynchformerInstance(SynchformerWrapper):
    """
    Compatibility wrapper to match the SyncNetInstance interface used in vasa_losses.py
    """

    def __init__(self, device='cuda', checkpoint_path=None, config_path=None):
        """
        Initialize SynchformerInstance with flexible path handling.

        Args:
            device: Device to run the model on
            checkpoint_path: Optional path to checkpoint file
            config_path: Optional path to config file
        """
        super().__init__(
            config_path=config_path,  # Will use default if None
            checkpoint_path=checkpoint_path,  # Will auto-search if None
            device=device
        )

    def forward_vid(self, frames: torch.Tensor) -> torch.Tensor:
        """Process video frames (for compatibility with SyncNet interface)"""
        # This would normally extract visual features
        # For Synchformer, preprocessing happens in compute_sync_score
        return frames

    def forward_aud(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio (for compatibility with SyncNet interface)"""
        # This would normally extract audio features
        # For Synchformer, preprocessing happens in compute_sync_score
        return audio

    def evaluate_sync(
        self,
        video_frames: torch.Tensor,
        audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate synchronization (compatible with SyncNet interface).

        Returns:
            sync_score: Synchronization confidence score
            sync_loss: Loss value (lower = better sync)
        """
        # Compute sync probability
        sync_prob = self.compute_sync_score(video_frames, audio_features, return_logits=False)

        # Convert to loss (lower = better)
        sync_loss = 1.0 - sync_prob

        return sync_prob, sync_loss