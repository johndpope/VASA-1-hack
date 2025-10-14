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
    import os

    # Save current sys.path
    original_path = sys.path.copy()

    try:
        # CRITICAL: Build a clean sys.path with Synchformer at the VERY FRONT
        # Keep all system paths (for stdlib like 'logging'), but remove VASA paths that conflict
        vasa_hack_root = os.path.abspath('/media/2TB/VASA-1-hack')

        filtered_path = []
        for p in sys.path:
            # Skip paths that could have conflicting utils modules
            if 'repos' in p or 'face_par_off' in p:
                continue
            # Skip empty string (current dir) and VASA root (has nemo/utils)
            if p == '' or os.path.abspath(p) == vasa_hack_root:
                continue
            # Skip relative paths like 'L2CS-Net', 'nemo' (they have conflicting utils)
            if p and not p.startswith('/') and not p.startswith('__editable__'):
                continue
            # Keep everything else (including Python stdlib, site-packages, etc.)
            filtered_path.append(p)

        # Add Synchformer paths at the VERY FRONT in correct order
        # This mimics what sync_model.py's `sys.path.insert(0, '.')` tries to do
        synchformer_paths = [
            synchformer_path,  # Main Synchformer directory
        ]

        for path in reversed(synchformer_paths):
            if path not in filtered_path:
                filtered_path.insert(0, path)

        # Temporarily replace sys.path
        sys.path = filtered_path

        # DEBUG: Verify Synchformer path is first
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"About to import with sys.path[0] = {sys.path[0]}")
        logger.info(f"Full sys.path (first 5): {sys.path[:5]}")
        logger.info(f"Synchformer utils exists? {os.path.exists(os.path.join(synchformer_path, 'synch_utils', 'utils.py'))}")

        # Test the import before doing it
        try:
            test_spec = __import__('synch_utils.utils', fromlist=['instantiate_from_config'])
            logger.info(f"✅ Test import of synch_utils.utils succeeded: {test_spec}")
        except Exception as e:
            logger.error(f"❌ Test import failed: {e}")
            logger.error(f"   Full sys.path:")
            for i, p in enumerate(sys.path[:15]):
                logger.error(f"     [{i}] {p!r}")

            # Check what utils Python can find
            try:
                import utils
                logger.error(f"   Python found utils at: {utils.__file__ if hasattr(utils, '__file__') else 'no __file__'}")
            except:
                logger.error(f"   Cannot even import utils")

        # Try to import the Synchformer modules
        from model.sync_model import Synchformer
        from scripts.train_utils import get_model
        from synch_utils.utils import instantiate_from_config

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

        # Restore original state after successful import
        sys.path = original_path

        return Synchformer, get_model, instantiate_from_config, AudioMelSpectrogram, AudioLog, AudioNormalizeAST, PadOrTruncate

    except ImportError as e:
        # Restore original state before raising
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
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    # Full training checkpoint with nested model
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict, strict=False)
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
            frames: [B, T, C, H, W] or [B, C, T, H, W] or [B, T, H, W, C] - video frames

        Returns:
            frames: [B, S, Tv, C, H, W] - segmented video frames
        """
        logger.debug(f"[preprocess_video] Input shape: {frames.shape}, dtype: {frames.dtype}, device: {frames.device}")

        # Ensure frames are in [B, T, C, H, W] format
        if frames.dim() == 5:
            logger.debug(f"[preprocess_video] Input is 5D: shape[1]={frames.shape[1]}, shape[-1]={frames.shape[-1]}")
            if frames.shape[1] == 3:  # [B, C, T, H, W]
                logger.debug(f"[preprocess_video] Detected [B, C, T, H, W] format, permuting to [B, T, C, H, W]")
                frames = frames.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
            elif frames.shape[-1] == 3:  # [B, T, H, W, C] - channels last (from emo_frames)
                logger.debug(f"[preprocess_video] Detected [B, T, H, W, C] format, permuting to [B, T, C, H, W]")
                frames = frames.permute(0, 1, 4, 2, 3)  # -> [B, T, C, H, W]

        logger.debug(f"[preprocess_video] After permutation: {frames.shape}")
        B, T, C, H, W = frames.shape
        logger.debug(f"[preprocess_video] Extracted dimensions: B={B}, T={T}, C={C}, H={H}, W={W}")

        # Resize to expected input size if needed
        if H != self.input_size or W != self.input_size:
            logger.debug(f"[preprocess_video] Resizing from {H}x{W} to {self.input_size}x{self.input_size}")
            frames = F.interpolate(
                frames.view(B * T, C, H, W),
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            ).view(B, T, C, self.input_size, self.input_size)
            logger.debug(f"[preprocess_video] After resize: {frames.shape}")

        # Normalize to [-1, 1] range expected by MotionFormer
        if frames.max() > 1.0:  # Assume [0, 255] range
            logger.debug(f"[preprocess_video] Normalizing from [0, 255] to [0, 1] (max={frames.max():.2f})")
            frames = frames / 255.0

        # Apply MotionFormer normalization
        logger.debug(f"[preprocess_video] Applying MotionFormer normalization")
        logger.debug(f"[preprocess_video] frames.shape before norm: {frames.shape}")
        logger.debug(f"[preprocess_video] visual_mean.shape: {self.visual_mean.shape}, visual_std.shape: {self.visual_std.shape}")

        self.visual_mean = self.visual_mean.to(frames.device)
        self.visual_std = self.visual_std.to(frames.device)

        # Reshape mean/std for [B, T, C, H, W] format instead of [B, C, T, H, W]
        # Original shape: [1, 3, 1, 1, 1] for [B, C, T, H, W]
        # Need shape: [1, 1, 3, 1, 1] for [B, T, C, H, W]
        visual_mean_btchw = self.visual_mean.view(1, 1, 3, 1, 1)
        visual_std_btchw = self.visual_std.view(1, 1, 3, 1, 1)

        logger.debug(f"[preprocess_video] Reshaped visual_mean to: {visual_mean_btchw.shape}, visual_std to: {visual_std_btchw.shape}")
        logger.debug(f"[preprocess_video] About to compute: (frames - visual_mean) / visual_std")
        logger.debug(f"[preprocess_video] frames.shape: {frames.shape}, visual_mean.shape: {visual_mean_btchw.shape}")

        frames = (frames - visual_mean_btchw) / visual_std_btchw
        logger.debug(f"[preprocess_video] After normalization: {frames.shape}")

        # Segment video into chunks
        # We need S segments of Tv frames each
        S = self.n_segments
        Tv = self.segment_size_vframes

        logger.debug(f"[preprocess_video] Segmentation: S={S}, Tv={Tv}, min_frames needed={S * Tv}")
        logger.debug(f"[preprocess_video] Current T={T}, will pad/truncate to {S * Tv}")

        # Ensure we have enough frames
        min_frames = S * Tv
        if T < min_frames:
            # Pad with last frame if needed
            pad_frames = min_frames - T
            logger.debug(f"[preprocess_video] Padding: adding {pad_frames} frames")
            last_frame = frames[:, -1:].expand(-1, pad_frames, -1, -1, -1)
            frames = torch.cat([frames, last_frame], dim=1)
            T = min_frames
        elif T > min_frames:
            # Truncate if too many frames
            logger.debug(f"[preprocess_video] Truncating: removing {T - min_frames} frames")
            frames = frames[:, :min_frames]
            T = min_frames

        logger.debug(f"[preprocess_video] After pad/truncate: {frames.shape}")
        logger.debug(f"[preprocess_video] About to reshape to [B={B}, S={S}, Tv={Tv}, C={C}, H={self.input_size}, W={self.input_size}]")

        # Reshape to segments: [B, S*Tv, C, H, W] -> [B, S, Tv, C, H, W]
        frames = frames.view(B, S, Tv, C, self.input_size, self.input_size)

        logger.debug(f"[preprocess_video] Final output shape: {frames.shape}")
        return frames

    def preprocess_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for Synchformer.

        Args:
            audio_features: Audio input - either raw waveform, MFCC, or mel spectrogram

        Returns:
            spec: [B, S, 1, F, Ta] - segmented audio spectrograms
        """
        logger.debug(f"[preprocess_audio] Input shape: {audio_features.shape}, dtype: {audio_features.dtype}")

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
        n_freq, Ta = spec.shape[1], spec.shape[2]
        target_Ta = self.max_spec_t * self.n_segments  # Total time needed

        if Ta < target_Ta:
            # Pad with zeros
            padding = target_Ta - Ta
            spec = nn.functional.pad(spec, (0, padding), mode='constant', value=0)
        elif Ta > target_Ta:
            # Truncate
            spec = spec[:, :, :target_Ta]

        # Apply AST normalization
        spec = (spec - self.audio_mean) / self.audio_std

        # Segment audio: [B, F, S*Ta] -> [B, S, F, Ta]
        S = self.n_segments
        Ta_per_seg = self.max_spec_t
        spec = spec.view(B, n_freq, S, Ta_per_seg).permute(0, 2, 1, 3)  # [B, S, F, Ta]

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