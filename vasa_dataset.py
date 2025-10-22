import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import cv2
import os
import json
from pathlib import Path
import subprocess
import random
import h5py
import hashlib
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import logging
import torchvision.transforms as transforms
import traceback
import mediapipe as mp
import matplotlib.pyplot as plt
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger
import sys
# Add fixed L2CS to path
if 'L2CS-Net' not in sys.path:
    sys.path.insert(0, 'L2CS-Net')
from l2cs import L2CS, select_device, Pipeline
import h5py
from tqdm import tqdm
from typing import *
from collections import defaultdict
from torchvision.transforms import ToTensor, ToPILImage
to_image = ToPILImage()
# Import the new chunked window cache
try:
    from window_cache import WindowCache as ChunkedWindowCache
    USE_CHUNKED_CACHE = True
    logger.info("Using ChunkedWindowCache for flexible window sizes")
except ImportError:
    ChunkedWindowCache = None
    USE_CHUNKED_CACHE = False
    logger.info("ChunkedWindowCache not available, using built-in cache")

try:
    from single_bucket_cache import SingleBucketCache
    USE_SINGLE_BUCKET = True
    logger.info("SingleBucketCache available for single-file caching")
except ImportError:
    SingleBucketCache = None
    USE_SINGLE_BUCKET = False
    logger.info("SingleBucketCache not available")

try:
    from frame_disk_cache import FrameDiskCache
    USE_FRAME_DISK_CACHE = True
    logger.info("FrameDiskCache available for MD5-indexed frame storage")
except ImportError:
    FrameDiskCache = None
    USE_FRAME_DISK_CACHE = False
    logger.info("FrameDiskCache not available")

try:
    from per_video_cache import PerVideoCache
    USE_PER_VIDEO_CACHE = True
    logger.info("PerVideoCache available for per-video H5 files")
except ImportError:
    PerVideoCache = None
    USE_PER_VIDEO_CACHE = False
    logger.info("PerVideoCache not available")
from torchvision.utils import save_image
from datetime import datetime
import hashlib
from blink_condition_handler import BlinkConditionHandler
from video_tracker import VideoEventData, VideoEvent, ProblematicVideosTracker

__all__ = ['VASAIntegratedDataset', 'WorkerState','VASADatasetMixin','SpeedEncoder']


class SpeedEncoder(nn.Module):
    """Speed bucketing using the EMO paper approach"""
    def __init__(self, num_buckets=9):
        super().__init__()
        self.num_buckets = num_buckets
        
        # Centers for speed buckets from -1.0 to 1.0 
        self.centers = torch.tensor([
            -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0
        ])
        
        # Fixed radius of 0.1 for each bucket
        self.radius = 0.1 
    def encode_speed(self, head_rotation_speed: float) -> int:
        """
        Convert speed to bucket index using tanh((w - c)/r * 3)
        Args:
            head_rotation_speed: Head rotation speed in [-1, 1] range
        Returns:
            int: Bucket index (0-8)
        """
        # Calculate distance to each bucket center 
        distances = torch.abs(
            torch.tanh((head_rotation_speed - self.centers) / self.radius * 3)
        )
        
        # Return index of closest bucket
        return torch.argmin(distances).item()

    def __call__(self, speed: float) -> int:
        """
        Convenience wrapper for encode_speed
        Args:
            speed: Head rotation speed in [-1, 1] range
        Returns:
            int: Bucket index (0-8)
        """
        return self.encode_speed(speed)

class WorkerState:
    """Manages per-worker state initialization with proper multiprocessing support"""
    _instance = None
    _initialized = False
  
    @property
    def whisper_model(self):
        """Lazy initialization of Whisper model"""
        if self._whisper_model is None:
            from transformers import WhisperModel
            logger.info("Loading Whisper model...")
            self._whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
            if torch.cuda.is_available():
                self._whisper_model = self._whisper_model.cuda()
            self._whisper_model.eval()
            logger.info(f"Whisper model loaded on device: {self._whisper_model.device}")
        return self._whisper_model

    @property  
    def whisper_processor(self):
        """Lazy initialization of Whisper processor"""
        if self._whisper_processor is None:
            from transformers import WhisperProcessor
            logger.info("Loading Whisper processor...")
            self._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            logger.info("Whisper processor loaded successfully")
        return self._whisper_processor

    def __init__(self):
        """Initialize worker state"""
        super().__init__()
        # Standard properties
        self._emotion_recognizer = None
        self._face_mesh = None
        self._modnet = None

        # L2CS properties  
        self._l2cs_model = None
        self._l2cs_device = None
        self._l2cs_pipeline = None

        # Audio properties - initialize as None but create the attributes
        self._audio_model = None  # wav2vec
        self._audio_processor = None  # wav2vec processor
        self._whisper_model = None  # whisper model
        self._whisper_processor = None  # whisper processor

        # Phoneme recognition properties
        self._phoneme_model = None  # wav2vec2 for phoneme recognition
        self._phoneme_processor = None  # phoneme processor

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance for current process"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize_worker(cls, worker_id: int):
        """Initialize worker-specific resources"""
        try:
            instance = cls.get_instance()
            if not cls._initialized:
                logger.info(f"Worker {worker_id}: Initializing resources")
                cls._initialized = True
                
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error initializing resources - {str(e)}")
            raise
    

    @property
    def l2cs_pipeline(self):
        """Lazy initialization of L2CS pipeline"""
        if self._l2cs_pipeline is None:
            try:
                # Create pipeline
                import pathlib
                # Use the existing L2CSNet_gaze360.pkl file which has 92MB
                weights_path = pathlib.Path('models/L2CSNet_gaze360.pkl')

                self._l2cs_pipeline = Pipeline(
                    weights=weights_path,
                    arch='ResNet50',
                    device='cuda',
                    include_detector=False  # We use our own face detection
                )
                logger.info("L2CS pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize L2CS pipeline: {e}")
                logger.warning("Will use default gaze values")
                return None

        return self._l2cs_pipeline



    @property
    def emotion_recognizer(self):
        """Lazy initialization of emotion recognizer"""
        if self._emotion_recognizer is None:
            self._emotion_recognizer = HSEmotionRecognizer(
                model_name='enet_b0_8_va_mtl'
            )
        return self._emotion_recognizer
    
    @property
    def audio_model(self):
        """Lazy initialization of aligned wav2vec model"""
        if self._audio_model is None:
            from wav2vec_module import AlignedWav2Vec2Model
            self._audio_model = AlignedWav2Vec2Model(
                'facebook/wav2vec2-base',
                freeze_feature_extractor=True
            ).eval()
        return self._audio_model
    
    @property
    def audio_processor(self):
        """Lazy initialization of audio processor"""
        if self._audio_processor is None:
            from transformers import Wav2Vec2Processor
            self._audio_processor = Wav2Vec2Processor.from_pretrained(
                'facebook/wav2vec2-base'
            )
        return self._audio_processor

    @property
    def phoneme_model(self):
        """Lazy initialization of phoneme recognition model"""
        if self._phoneme_model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            logger.info("Loading phoneme recognition model...")
            self._phoneme_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            self._phoneme_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            )
            if torch.cuda.is_available():
                self._phoneme_model = self._phoneme_model.cuda()
            self._phoneme_model.eval()
            logger.info(f"Phoneme model loaded on device: {next(self._phoneme_model.parameters()).device}")
        return self._phoneme_model

    @property
    def phoneme_processor(self):
        """Get phoneme processor (initialized with phoneme_model)"""
        if self._phoneme_processor is None:
            # Trigger phoneme model initialization which also sets processor
            _ = self.phoneme_model
        return self._phoneme_processor

    @property
    def face_mesh(self):
        """Lazy initialization of face mesh"""
        if self._face_mesh is None:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,  # Increased from 1 to 2 to handle partial faces
                refine_landmarks=True,
                min_detection_confidence=0.3  # Reduced from 0.5 to 0.3 for more sensitive detection
            )
        return self._face_mesh

    @property
    def modnet(self):
        """Lazy initialization of MODNet"""
        if self._modnet is None:
            from repos.MODNet.src.models.modnet import MODNet
            self._modnet = MODNet(backbone_pretrained=False)
            
            # Load pretrained weights
            state_dict = torch.load(
                '/media/oem/12TB/nemo/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt',
                map_location='cpu'
            )
            
            # Remove module prefix from state dict keys
            new_state_dict = {}
            for k in list(state_dict.keys()):
                new_k = k[7:]  # Remove 'module.' prefix
                new_state_dict[new_k] = state_dict[k]
                
            # Load state dict and move to GPU if available
            self._modnet.load_state_dict(new_state_dict)
            if torch.cuda.is_available():
                self._modnet = self._modnet.cuda()
            self._modnet.eval()
            
        return self._modnet

class VASADatasetMixin:
    """Mixin class to handle worker initialization for VASA dataset"""
    
    def __init__(self):
        self._worker_state = None
    
    @property
    def modnet(self):  # Added MODNet property
        return self.worker_state.modnet
    
    @property
    def worker_state(self):
        """Get worker state instance for current process"""
        if self._worker_state is None:
            self._worker_state = WorkerState.get_instance()
            # Set L2CS device based on dataset device
            # self._worker_state.set_l2cs_device(self.device)
        return self._worker_state
    
    @property
    def emotion_recognizer(self):
        return self.worker_state.emotion_recognizer
    
    @property
    def audio_model(self):
        return self.worker_state.audio_model
    
    @property
    def audio_processor(self):
        return self.worker_state.audio_processor

    @property
    def whisper_model(self):
        return self.worker_state.whisper_model
    
    @property
    def whisper_processor(self):
        return self.worker_state.whisper_processor
       
    @property
    def face_mesh(self):
        return self.worker_state.face_mesh
    
    def __getstate__(self):
        """Remove non-picklable attributes before serialization"""
        state = self.__dict__.copy()
        state['_worker_state'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state without initializing worker state"""
        self.__dict__.update(state)


class WindowCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, video_path: str) -> Path:
        """Generate unique cache file path for video."""
        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        return self.cache_dir / f"{video_hash}.h5"
        
    def has_cache(self, video_path: str) -> bool:
        """Check if cache exists for video."""
        cache_path = self._get_cache_path(video_path)
        return cache_path.exists()


    def load_windows(self, video_path: str) -> List[Dict[str, torch.Tensor]]:
        """Load window data from H5 file with proper tensor reshaping."""
        cache_path = self._get_cache_path(video_path)
        windows_data = []

        try:
            if not cache_path.exists():
                logger.warning(f"Cache file not found: {cache_path}")
                return windows_data

            with h5py.File(cache_path, 'r') as f:
                # Validate metadata
                if 'video_path' not in f.attrs or f.attrs['video_path'] != video_path:
                    logger.warning("Cache file metadata mismatch")
                    return windows_data

                num_windows = f.attrs['num_windows']
                
                # Load each window
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        logger.warning(f"Missing {window_key} in cache file")
                        continue
                        
                    window_group = f[window_key]
                    window_data = {}
                    
                    # Load tensors
                    for key in window_group.keys():
                        if key == 'identity_frame':
                            # Load identity frame and expand to full frames tensor
                            identity_frame_data = window_group[key][()]
                            identity_frame = torch.from_numpy(identity_frame_data)
                            # Duplicate identity frame for all frame positions (for compatibility)
                            window_data['frames'] = identity_frame.unsqueeze(0).repeat(self.window_size, 1, 1, 1)
                            continue
                        elif key == 'metadata':
                            continue

                        try:
                            # Get dataset
                            dataset = window_group[key]

                            # Load tensor data
                            data = dataset[()]

                            # Convert to tensor
                            tensor = torch.from_numpy(data)

                            # Fix tensor dtype if needed
                            if 'dtype' in dataset.attrs:
                                dtype_str = dataset.attrs['dtype']
                                if isinstance(dtype_str, tuple):
                                    dtype_str = dtype_str[0].decode('utf-8')
                                tensor = tensor.to(dtype=getattr(torch, dtype_str.split('.')[-1]))

                            # Move ALL tensors to GPU for training efficiency
                            if torch.cuda.is_available():
                                tensor = tensor.cuda()
                            
                            # Remove extra dimensions if needed
                            if key in ['theta', 'rotation', 'translation', 'expression_embed', 'scale']:
                                # These should be [B, T, ...] not [B, 1, T, ...]
                                if len(tensor.shape) > 3 and tensor.shape[1] == 1:
                                    tensor = tensor.squeeze(1)
                                
                            # Special handling for audio features
                            if key == 'audio_features' and len(tensor.shape) == 3:
                                # Add channel dim if missing: [B, T, D] -> [B, 1, T, D]
                                tensor = tensor.unsqueeze(1)
                            
                            window_data[key] = tensor
                            
                        except Exception as e:
                            logger.error(f"Error loading tensor {key}: {str(e)}")
                            continue

                    # Load metadata if present
                    if 'metadata' in window_group:
                        metadata = {}
                        for k, v in window_group['metadata'].attrs.items():
                            # Convert bytes to str if needed
                            if isinstance(v, bytes):
                                v = v.decode('utf-8')
                            metadata[k] = v
                        window_data['metadata'] = metadata

                    # Log shapes for debugging
                    logger.debug(f"\nWindow {i} tensor shapes:")
                    for k, v in window_data.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"  {k}: {v.shape}")
                    
                    windows_data.append(window_data)

                if not windows_data:
                    logger.warning("No valid windows loaded from cache")
                
                return windows_data

        except Exception as e:
            logger.error(f"Error loading cache file: {str(e)}")
            if cache_path.exists():
                logger.info(f"Removing corrupted cache file: {cache_path}")
                cache_path.unlink()
            return windows_data

    def save_windows(self, video_path: str, windows_data: List[Dict[str, torch.Tensor]]):
        """Save window data with proper metadata handling."""
        cache_path = self._get_cache_path(video_path)
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            with h5py.File(temp_path, 'w') as f:
                # Save metadata
                f.attrs['video_path'] = video_path
                f.attrs['num_windows'] = len(windows_data)
                
                # Save each window
                for i, window in enumerate(windows_data):
                    window_group = f.create_group(f'window_{i}')
                    
                    # Save tensors
                    for key, tensor in window.items():
                        if key == 'metadata':
                            continue
                            
                        if isinstance(tensor, torch.Tensor):
                            # Convert to numpy and save
                            data = tensor.cpu().numpy()
                            ds = window_group.create_dataset(
                                key,
                                data=data,
                                compression='gzip'
                            )
                            # Save tensor metadata as string
                            ds.attrs['dtype'] = str(tensor.dtype)
                            ds.attrs['shape'] = tensor.shape
                    
                    # Save metadata dict if present
                    if 'metadata' in window:
                        meta_group = window_group.create_group('metadata')
                        for k, v in window['metadata'].items():
                            # Convert any non-string values to strings
                            if not isinstance(v, (str, bytes)):
                                v = str(v)
                            meta_group.attrs[k] = v

            # Only after successful save, replace old cache
            if temp_path.exists():
                if cache_path.exists():
                    cache_path.unlink()
                temp_path.rename(cache_path)
                logger.info(f"Successfully saved cache to {cache_path}")

        except Exception as e:
            logger.error(f"Error saving cache file: {str(e)}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

class VASAIntegratedDataset(Dataset, VASADatasetMixin):

    def __init__(
        self,
        video_folder: str,
        emo_model,
        window_size: int = 50,
        stride: int = 25,
        max_batch_size: int = 20,  # Max windows to save at once
        context_size: int = 10,
        frame_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 50,
        hop_length: int = 10,
        cache_audio: bool = True,
        preextract_audio: bool = True,
        max_videos: Optional[int] = 100,
        random_seed: int = 42,
        device: str = 'cuda',
        cache_dir: Optional[str] = 'cache',
        use_single_bucket: bool = True,  # New parameter for single-bucket caching
        generate_emo_frames: bool = False,  # NEW: Whether to generate EMO frames
        emo_identity_path: str = "nemo/data/IMG_1.png",  # NEW: Identity image for EMO
        emo_keyframes_per_window: int = 5,  # NEW: Number of EMO keyframes to generate
        va_bridge = None,  # NEW: Volumetric avatar bridge for EMO generation
        auto_rebuild_expression_db: bool = False,  # NEW: Auto-rebuild expression DB after preprocessing
        expression_db_frame_stride: int = 5,  # NEW: Sample every Nth frame for expression DB
        cache_frames_to_disk: bool = False,  # NEW: Load frames from disk cache
        cache_emo_frames_to_disk: bool = False,  # NEW: Load EMO frames from disk cache
        frame_format: str = 'png',  # NEW: Frame format for disk cache
        flow_noise_level: float = 0.1,  # NEW: Noise level for Flow-DPO dispreferred samples
    ):
        VASADatasetMixin.__init__(self)

        # Basic initialization
        self.video_folder = Path(video_folder)
        self.emo_model = emo_model
        self.window_size = window_size
        self.auto_rebuild_expression_db = auto_rebuild_expression_db
        self.expression_db_frame_stride = expression_db_frame_stride
        self.stride = stride
        self.max_batch_size = max_batch_size
        self.context_size = context_size
        self.cache_audio = cache_audio
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        self.device = device
        self.model_device = next(emo_model.parameters()).device

        # Flow-DPO parameters
        self.flow_noise_level = flow_noise_level

        # Initialize LipStateAnalyzer for lip metrics computation
        self.lip_analyzer = LipStateAnalyzer()

        # EMO generation setup
        self.generate_emo_frames = generate_emo_frames
        self.emo_keyframes_per_window = emo_keyframes_per_window
        self.va_bridge = va_bridge
        self.emo_identity_image = None

        if generate_emo_frames:
            # Load identity image for EMO generation
            if os.path.exists(emo_identity_path):
                from PIL import Image
                # Note: transforms already imported at module level (line 20)

                img = Image.open(emo_identity_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ])
                self.emo_identity_image = transform(img).unsqueeze(0).to(device)
                logger.info(f"Loaded EMO identity image from {emo_identity_path}")
            else:
                logger.warning(f"EMO identity image not found at {emo_identity_path}")
                self.generate_emo_frames = False

        # Set cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path(video_folder) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_single_bucket = use_single_bucket

        # Initialize frame disk caches if enabled
        self.frame_cache = None
        self.emo_frame_cache = None
        self.cache_frames_to_disk = cache_frames_to_disk
        self.cache_emo_frames_to_disk = cache_emo_frames_to_disk

        if cache_frames_to_disk and USE_FRAME_DISK_CACHE and FrameDiskCache:
            self.frame_cache = FrameDiskCache(self.cache_dir, frame_type='frames')
            logger.info(f"✅ Frame disk cache enabled for loading at {self.frame_cache.root}")

        if cache_emo_frames_to_disk and USE_FRAME_DISK_CACHE and FrameDiskCache:
            self.emo_frame_cache = FrameDiskCache(self.cache_dir, frame_type='emo_frames')
            logger.info(f"✅ EMO frame disk cache enabled for loading at {self.emo_frame_cache.root}")

        # Choose cache implementation based on preference
        # Priority: PerVideoCache > SingleBucketCache > ChunkedCache > Built-in
        per_video_index = self.cache_dir / 'cache_index.json'
        single_bucket_h5 = self.cache_dir / 'all_windows_cache.h5'

        if per_video_index.exists() and USE_PER_VIDEO_CACHE and PerVideoCache:
            # Use per-video cache (MD5-indexed folders with separate H5 files)
            self.cache = PerVideoCache(
                cache_dir=self.cache_dir,
                compression='gzip',
                compression_level=4
            )
            self.cache_type = 'per_video'
            logger.info(f"✅ Using PerVideoCache at {self.cache_dir}")
            logger.info(f"   Per-video H5 files with MD5-indexed folders")
        elif single_bucket_h5.exists() and use_single_bucket and USE_SINGLE_BUCKET and SingleBucketCache:
            # Use single-bucket cache for all windows
            self.cache = SingleBucketCache(
                cache_dir=self.cache_dir,
                cache_name="all_windows_cache.h5",
                compression='gzip',
                compression_level=4
            )
            self.cache_type = 'single_bucket'
            logger.info(f"Using SingleBucketCache at {self.cache_dir}/all_windows_cache.h5")
        elif USE_CHUNKED_CACHE and ChunkedWindowCache:
            # Use chunked cache for flexible window support
            cache_path = Path(cache_dir) if cache_dir else Path(video_folder) / "window_cache_chunked"
            self.cache = ChunkedWindowCache(
                cache_dir=cache_path,
                chunk_size=1000,  # 1000 frames per chunk
                overlap_size=50,  # 50 frame overlap for context
                max_memory_cache=5  # Keep 5 chunks in memory
            )
            self.cache_type = 'chunked'
            logger.info(f"Initialized ChunkedWindowCache at {cache_path}")
        else:
            # Fallback to built-in cache
            self.cache = WindowCache(Path(video_folder) / "window_cache")
            self.cache_type = 'built_in'
            logger.info("Using built-in WindowCache")

        self.blink_handler = BlinkConditionHandler(window_size=sequence_length)

        self.tracker = ProblematicVideosTracker(Path("bad_videos"))


        # Set up caching
        self.audio_cache_dir = self.video_folder / "audio_cache"
        self.audio_cache_dir.mkdir(exist_ok=True)
        logger.info(f"Using audio cache directory: {self.audio_cache_dir}")
        
        self.video_folder = Path(self.video_folder)  # Ensure it's a Path object
        all_videos = [str(f) for ext in ("*.mp4", "*.mpg") for f in self.video_folder.rglob(ext)]
        logger.info(f"Found {len(all_videos)} total videos")
        
        # Sample videos if needed
        if max_videos and max_videos < len(all_videos):
            random.seed(random_seed)
            self.video_paths = random.sample(all_videos, max_videos)
            logger.info(f"Randomly sampled {max_videos} videos")
        else:
            self.video_paths = all_videos
            
        # Initialize audio status
        self.audio_status = {}
        self.audio_status_file = self.audio_cache_dir / "audio_status.json"
        
        # Load cached status
        if self.audio_status_file.exists():
            with open(self.audio_status_file, 'r') as f:
                cached_status = json.load(f)
                logger.info(f"Loaded cached status with {len(cached_status)} entries")
                # Debug each cached entry
                for k, v in cached_status.items():
                    if not isinstance(v, dict):
                        logger.error(f"Invalid cache entry for {k}: {v} (type: {type(v)})")
                # Only keep valid dictionary entries for selected videos
                self.audio_status = {
                    k: v for k, v in cached_status.items() 
                    if k in self.video_paths and isinstance(v, dict)
                }
                logger.info(f"Kept {len(self.audio_status)} valid cached entries")
                
        # Check remaining videos
        videos_to_check = [v for v in self.video_paths if v not in self.audio_status]
        if videos_to_check:
            logger.info(f"Checking audio for {len(videos_to_check)} new videos...")
            new_status = {}
            for video_path in tqdm(videos_to_check, desc="Checking audio"):
                try:
                    # Check for existing audio file
                    audio_path = self._get_audio_path(video_path)
                    has_cache = audio_path.exists()
                    
                    # Check for audio stream
                    command = [
                        'ffprobe',
                        '-loglevel', 'error',
                        '-show_streams',
                        '-select_streams', 'a',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'json',
                        video_path
                    ]
                    
                    result = subprocess.run(command, capture_output=True, text=True)
                    has_audio = False
                    
                    if result.returncode == 0:
                        data = json.loads(result.stdout)
                        has_audio = bool(data.get('streams', []))
                    
                    new_status[video_path] = {
                        'has_audio': has_audio,
                        'has_cache': has_cache,
                        'last_checked': str(datetime.now())
                    }
                    
                except Exception as e:
                    logger.error(f"Error checking {video_path}: {str(e)}")
                    new_status[video_path] = {
                        'has_audio': False,
                        'has_cache': False,
                        'error': str(e),
                        'last_checked': str(datetime.now())
                    }
                    
            # Update status
            self.audio_status.update(new_status)
            
            # Save updated status
            with open(self.audio_status_file, 'w') as f:
                json.dump(self.audio_status, f)
                
        # Verify status format before filtering
        for k, v in list(self.audio_status.items()):
            if not isinstance(v, dict):
                logger.error(f"Invalid status for {k}: {v}")
                self.audio_status[k] = {'has_audio': False, 'has_cache': False}
                
        # Filter to valid videos
        valid_videos = [
            v for v in self.video_paths
            if isinstance(self.audio_status.get(v), dict) and
            self.audio_status[v].get('has_audio', False)
        ]
        self.video_paths = valid_videos

        logger.info(f"Found {len(self.video_paths)} videos with valid audio")
        
        # Extract audio if requested
        if preextract_audio and self.video_paths:
            logger.info(f"Pre-extracting audio for {len(self.video_paths)} videos...")
            self._preextract_all_audio()
            
        # Continue with rest of initialization if we have valid videos
        if self.video_paths:
            self.speed_encoder = SpeedEncoder(num_buckets=9)
            self.centers = self.speed_encoder.centers
            self.radius = self.speed_encoder.radius
            
            # Initialize transforms
            self.preprocess_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(frame_size),
                transforms.ToTensor(),
            ])
            
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Create windows
            self.windows = self._create_window_indices()

            # NOTE: emo_frames filtering disabled because emo_frames are generated on-the-fly
            # and not always saved back to cache. Filtering would incorrectly exclude valid windows.
            # If needed in future, ensure emo_frames are saved to cache after generation.
            logger.info(f"Using all {len(self.windows)} windows (emo_frames will be generated on-the-fly if needed)")

            self.video_windows = defaultdict(list)
            for window in self.windows:
                self.video_windows[window['video_path']].append(window)

            logger.info(f"Created {len(self.windows)} total windows across {len(self.video_paths)} videos")
        else:
            logger.warning("No valid videos found with audio!")
            self.windows = []
            self.video_windows = defaultdict(list)

    def _check_audio_cache_status(self, video_paths: List[str]) -> Dict[str, Dict]:
        """Check audio cache status and validity for all videos with smart caching."""
        status = {}
        audio_status_file = self.audio_cache_dir / "audio_status.json"
        
        # Load existing cache status if available
        if audio_status_file.exists():
            with open(audio_status_file, 'r') as f:
                cached_status = json.load(f)
                logger.info(f"Loaded cached audio status for {len(cached_status)} videos")
        else:
            cached_status = {}
            logger.info("No cached audio status found")
        
        # For each video:
        # 1. If in cache and has_audio is True and has_cache is True - use cached result
        # 2. If new video or cache says has_audio but no cache file - check it
        videos_to_check = []
        for video_path in video_paths:
            if video_path in cached_status:
                video_status = cached_status[video_path]
                audio_path = self._get_audio_path(video_path)
                
                # If cache says it has audio and the file exists, use cached result
                if (video_status.get('has_audio', False) and 
                    video_status.get('has_cache', False) and 
                    audio_path.exists()):
                    status[video_path] = video_status
                    continue
            
            # Need to check this video
            videos_to_check.append(video_path)
        
        # Report stats
        logger.info(f"Using {len(status)} cached results")
        logger.info(f"Need to check {len(videos_to_check)} videos")
        
        # Only check videos that weren't in cache or need rechecking
        if videos_to_check:
            logger.info("Checking audio for videos not in cache...")
            for video_path in tqdm(videos_to_check):
                try:
                    # Check if audio file exists in cache
                    audio_path = self._get_audio_path(video_path)
                    has_cache = audio_path.exists()
                    
                    # Assume audio exists if cache file exists
                    has_audio = has_cache
                    
                    status[video_path] = {
                        'has_audio': has_audio,
                        'has_cache': has_cache,
                        'last_checked': str(datetime.now())
                    }
                    
                except Exception as e:
                    logger.error(f"Error checking audio for {video_path}: {str(e)}")
                    status[video_path] = {
                        'has_audio': False,
                        'has_cache': False,
                        'error': str(e),
                        'last_checked': str(datetime.now())
                    }
        
        # Save updated status
        # Merge with any existing status entries for videos we didn't process
        final_status = {**cached_status, **status}
        with open(audio_status_file, 'w') as f:
            json.dump(final_status, f, indent=2)
        
        # Return only status for requested videos
        return {k: final_status[k] for k in video_paths}

    
    def _compute_rotation_speed(self, curr_landmarks: np.ndarray, prev_landmarks: np.ndarray) -> float:
        """Calculate head rotation speed between frames"""
        try:
            # logger.debug("\n=== Computing Rotation Speed ===")
            # logger.debug(f"Input landmarks shapes - current: {curr_landmarks.shape}, previous: {prev_landmarks.shape}")

            def get_head_angles(landmarks, frame_label=""):
                # logger.debug(f"\nProcessing {frame_label} frame landmarks:")
                
                # Get key landmarks for angle calculation
                nose_bridge = landmarks[27]
                nose_tip = landmarks[30]
                left_eye = np.mean(landmarks[36:42], axis=0)
                right_eye = np.mean(landmarks[42:48], axis=0)
                
                # logger.debug(f"Key points:")
                # logger.debug(f"  Nose bridge: {nose_bridge}")
                # logger.debug(f"  Nose tip: {nose_tip}")
                # logger.debug(f"  Left eye center: {left_eye}")
                # logger.debug(f"  Right eye center: {right_eye}")

                # Calculate face normal
                eye_vector = right_eye - left_eye
                nose_vector = nose_tip - nose_bridge
                face_normal = np.cross(eye_vector, nose_vector)
                
                # logger.debug(f"Vectors:")
                # logger.debug(f"  Eye vector: {eye_vector}")
                # logger.debug(f"  Nose vector: {nose_vector}")
                # logger.debug(f"  Face normal (before norm): {face_normal}")

                # Normalize face normal
                normal_magnitude = np.linalg.norm(face_normal)
                if normal_magnitude < 1e-6:
                    logger.warning("Near-zero face normal magnitude detected")
                    face_normal = np.array([0., 0., 1.])
                else:
                    face_normal = face_normal / normal_magnitude
                
                # logger.debug(f"  Normalized face normal: {face_normal}")

                # Get angles
                yaw = np.arctan2(face_normal[0], face_normal[2])
                pitch = np.arctan2(-face_normal[1], np.sqrt(face_normal[0]**2 + face_normal[2]**2))
                
                angles = np.array([yaw, pitch])
                # logger.debug(f"  Computed angles (yaw, pitch): {angles} radians")
                # logger.debug(f"                               {np.degrees(angles)} degrees")
                
                return angles

            # Get angles for both frames
            curr_angles = get_head_angles(curr_landmarks, "current")
            prev_angles = get_head_angles(prev_landmarks, "previous")

            # Calculate angular velocity
            angle_diff = curr_angles - prev_angles
            # logger.debug(f"\nAngle differences:")
            # logger.debug(f"  Radians: {angle_diff}")
            # logger.debug(f"  Degrees: {np.degrees(angle_diff)}")

            speed = np.linalg.norm(angle_diff)
            # logger.debug(f"Raw speed (radians): {speed}")
            # logger.debug(f"Raw speed (degrees): {np.degrees(speed)}")

            # Normalize to [-1, 1] range
            normalized_speed = np.clip(speed / np.pi, -1.0, 1.0) * 10 
            # logger.debug(f"Normalized speed: {normalized_speed}")
            
            # logger.debug("=== Rotation Speed Computation Complete ===\n")
            return normalized_speed

        except Exception as e:
            logger.error(f"Error calculating rotation speed: {str(e)}")
            logger.error("Landmark shapes:")
            logger.error(f"  Current landmarks: {curr_landmarks.shape if curr_landmarks is not None else 'None'}")
            logger.error(f"  Previous landmarks: {prev_landmarks.shape if prev_landmarks is not None else 'None'}")
            logger.error(traceback.format_exc())
            return 0.0
            
    
    def _extract_gaze_from_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Extract gaze angles using L2CS with temporal smoothing"""
        try:
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            # Get predictions using worker state pipeline
            results = self.worker_state.l2cs_pipeline.step(frame)
            
            # If no face detected, interpolate from previous values if available
            if len(results.pitch) == 0:
                if hasattr(self, '_prev_gaze'):
                    return self._prev_gaze
                return 0.0, 0.0
                
            # Get current predictions
            pitch = float(results.pitch[0] * 180/np.pi)
            yaw = float(results.yaw[0] * 180/np.pi)
            
            # Apply temporal smoothing
            if hasattr(self, '_prev_gaze'):
                prev_pitch, prev_yaw = self._prev_gaze
                smoothing_factor = 0.7  # Adjust this value (0-1) to control smoothing
                
                pitch = smoothing_factor * prev_pitch + (1 - smoothing_factor) * pitch
                yaw = smoothing_factor * prev_yaw + (1 - smoothing_factor) * yaw
            
            # Store current values for next frame
            self._prev_gaze = (pitch, yaw)
            
            return pitch, yaw
            
        except Exception as e:
            logger.error(f"Error extracting gaze: {str(e)}")
            if hasattr(self, '_prev_gaze'):
                return self._prev_gaze
            return 0.0, 0.0
        

    def _get_emotion(self, face_crop: np.ndarray) -> tuple:
        """Get emotion label and VA (valence-arousal) values using HSEmotion MTL model

        Returns:
            tuple: (emotion_label: str, va_values: np.ndarray) where va_values is shape (2,) for [valence, arousal]
        """
        try:
            # Ensure face crop is the right size
            if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
                face_crop = cv2.resize(face_crop, (64, 64))

            # Get predictions - returns (label, scores) where scores includes VA values
            labels, scores = self.emotion_recognizer.predict_emotions(face_crop, logits=True)

            # logger.debug(f"Emotion scores: {scores}")
            # logger.debug(f"Emotion labels: {labels}")

            # Extract emotion label
            emotion_label = labels if isinstance(labels, str) else "neutral"

            # Extract VA values (last two values in scores)
            if isinstance(scores, np.ndarray) and scores.size >= 2:
                va_values = scores[-2:]  # Get last two values (valence, arousal)
                va_values = np.array(va_values, dtype=np.float32)

                # Ensure correct shape
                if va_values.shape != (2,):
                    logger.warning(f"Unexpected VA shape: {va_values.shape}")
                    return emotion_label, np.zeros(2, dtype=np.float32)

                # Apply tanh to ensure values are in [-1, 1] range
                va_values = np.tanh(va_values)

                return emotion_label, va_values

            return emotion_label, np.zeros(2, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in emotion VA extraction: {str(e)}")
            return "neutral", np.zeros(2, dtype=np.float32)  # [label, valence-arousal]
        
            
    def _compute_main_gaze_direction(self, frames: List[np.ndarray]) -> Tuple[float, float]:
        """Compute main gaze direction using histogram clustering"""
        
        # Extract gaze angles for each frame
        gaze_angles = []
        for frame in frames:
            pitch, yaw = self._extract_gaze_from_frame(frame)
            gaze_angles.append((pitch, yaw))
            
        if not gaze_angles:
            return 0.0, 0.0
            
        # Convert to numpy array
        gaze_angles = np.array(gaze_angles)
        
        # Create 2D histogram
        pitch_bins = np.linspace(-90, 90, 18)  # 10-degree bins
        yaw_bins = np.linspace(-90, 90, 18)    # 10-degree bins
        
        H, xedges, yedges = np.histogram2d(
            gaze_angles[:, 0],  # pitch
            gaze_angles[:, 1],  # yaw
            bins=[pitch_bins, yaw_bins]
        )
        
        # Find mode of histogram
        max_idx = np.unravel_index(H.argmax(), H.shape)
        main_pitch = (pitch_bins[max_idx[0]] + pitch_bins[max_idx[0] + 1]) / 2
        main_yaw = (yaw_bins[max_idx[1]] + yaw_bins[max_idx[1] + 1]) / 2
        
        return main_pitch, main_yaw


    def _check_video_length(self, video_path: str) -> Optional[Tuple[int, float]]:
        """Check if video meets minimum length requirements"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Minimum frames needed for 2 windows (window_size + stride)
            # Example: window_size=50, stride=25 -> window1=[0-49], window2=[25-74] -> need 75 frames
            min_frames = self.window_size + self.stride  # 50 + 25 = 75
            
            logger.debug(f"\nVideo length check for {video_path}:")
            logger.debug(f"  Total frames: {total_frames}")
            logger.debug(f"  FPS: {fps}")
            logger.debug(f"  Minimum frames needed: {min_frames}")
            logger.debug(f"  Window calculation:")
            logger.debug(f"    total_frames - window_size = {total_frames - self.window_size}")
            logger.debug(f"    divided by stride = {(total_frames - self.window_size) / self.stride}")
            logger.debug(f"    floor division = {(total_frames - self.window_size) // self.stride}")
            logger.debug(f"    final n_windows = {max(1, (total_frames - self.window_size) // self.stride + 1)}")
            
            if total_frames < min_frames:
                logger.warning(
                    f"Video too short: {video_path}\n"
                    f"  Has {total_frames} frames, need minimum {min_frames}\n"
                    f"  window_size={self.window_size}, context={self.context_size}, "
                    f"stride={self.stride}"
                )
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.VIDEO_TOO_SHORT,
                    details={
                        "total_frames": total_frames,
                        "min_frames": min_frames
                    }
                ))
                return None
                
            return total_frames, fps
            
        except Exception as e:
            self.tracker.dispatch(VideoEventData(
                video_path=video_path,
                event_type=VideoEvent.PROCESSING_ERROR,
                details={"error": f"Length check error: {str(e)}"}
            ))            
            logger.error(f"Error checking video {video_path}: {str(e)}")
            return None
    
    def _create_window_indices(self) -> List[Dict]:
        """Create sliding windows for videos with proper overlap."""
        all_windows = []
        videos_with_windows = 0
        
        logger.info("\n=== Creating Window Indices ===")
        logger.info(f"Window size: {self.window_size}")
        logger.info(f"Stride: {self.stride}")
        logger.info(f"Context size: {self.context_size}")
        logger.info(f"Total videos to process: {len(self.video_paths)}")
        
        for video_path in self.video_paths:
            logger.debug(f"\nProcessing video: {video_path}")

            # Skip if no audio
            if not self.audio_status.get(video_path, {}).get('has_audio', False):
                logger.debug(f"Skipping - no audio available")
                continue

            # Check video length
            video_info = self._check_video_length(video_path)
            if video_info is None:
                logger.debug("Skipping - video length check failed")
                continue
                
            total_frames, fps = video_info
            logger.debug(f"Video info - frames: {total_frames}, fps: {fps}")

            # Calculate number of complete windows with overlap
            n_windows = max(1, (total_frames - self.window_size) // self.stride + 1)
            logger.debug(f"Number of windows: {n_windows}")
            logger.debug(f"Expected overlap size: {self.window_size - self.stride}")
            
            windows_this_video = 0
            
            for window_idx in range(n_windows):
                start_frame = window_idx * self.stride
                end_frame = start_frame + self.window_size
                
                if end_frame > total_frames:
                    logger.debug(f"Window {window_idx} exceeds video length - breaking")
                    break
                
                logger.debug(f"\nWindow {window_idx}:")
                logger.debug(f"  Start frame: {start_frame}")
                logger.debug(f"  End frame: {end_frame}")
                
                if window_idx > 0:
                    overlap_with_prev = start_frame - ((window_idx-1) * self.stride + self.window_size)
                    logger.debug(f"  Overlap with previous: {abs(overlap_with_prev)} frames")
                
                window_info = {
                    'video_path': video_path,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'fps': fps,
                    'has_context': window_idx > 0,
                    'window_idx': window_idx,
                    'total_frames': total_frames
                }
                
                all_windows.append(window_info)
                windows_this_video += 1

            if windows_this_video > 0:
                videos_with_windows += 1
                logger.info(
                    f"Created {windows_this_video} windows for video {video_path}\n"
                    f"  First window: {all_windows[-windows_this_video]['start_frame']} -> {all_windows[-windows_this_video]['end_frame']}\n"
                    f"  Last window: {all_windows[-1]['start_frame']} -> {all_windows[-1]['end_frame']}"
                )

        logger.info(
            f"\nWindow Creation Summary:\n"
            f"  Valid videos: {videos_with_windows}/{len(self.video_paths)}\n"
            f"  Total windows: {len(all_windows)}\n"
            f"  Window size: {self.window_size}\n"
            f"  Stride: {self.stride}\n"
            f"  Expected overlap: {self.window_size - self.stride}"
        )

        if len(all_windows) == 0:
            raise RuntimeError("No valid windows created. Check video lengths and audio availability")

        return all_windows
        

    def convert_theta_format(self, theta):
        if theta.ndim == 4:
            # shape [B, T, 4, 4], remove the last row and pick the first time step
            return theta[:, 0, :3, :]  # => [B, 3, 4]
        elif theta.ndim == 3:
            # shape [B, 4, 4], remove the last row
            return theta[:, :3, :]     # => [B, 3, 4]
        else:
            raise ValueError(f"Unexpected shape for theta: {theta.shape}")




    def _extract_emo_features(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate target warps and features for frames - cherry-picked from calculate_target_warps."""
        with torch.no_grad():
            try:
                logger.debug("\n=== Calculate Target Warps Start ===")
                logger.debug(f"Input frames shape: {frames.shape}")
                logger.debug(f"Input frames device: {frames.device}")
                logger.debug(f"EMO model device: {next(self.emo_model.parameters()).device}")

                assert len(frames.shape) == 4, f"Expected 4D input [T,C,H,W], got shape {frames.shape}"
                assert frames.shape[1] == 3, f"Expected 3 channels, got {frames.shape[1]}"

                T = frames.shape[0]
                logger.debug(f"Processing sequence of length {T}")

                # Add batch dimension and move to device
                frames = frames.unsqueeze(0)  # [1,T,C,H,W]
                frames = frames.to(next(self.emo_model.parameters()).device)
                logger.debug(f"After adding batch dim - frames shape: {frames.shape}")

                outputs = {
                    'theta': [],            # Target pose
                    'scale': [],            # SRT scale component
                    'rotation': [],         # SRT rotation component
                    'translation': [],      # SRT translation component
                    'expression_embed': [], # Renamed from target_pose_embed as requested
                    'uv_warps': [],        # Target UV warps (main warping field)
                    'target_masks': [],    # Target face masks
                }

                # Use first frame as identity for warp generation
                identity_frame = frames[:, 0]  # [1,C,H,W]

                # Extract identity features once (similar to extract_identity_features in create_video_face_swap.py)
                with torch.no_grad():
                    # Get face mask
                    identity_mask, _, _, _ = self.emo_model.face_idt.forward(identity_frame)
                    identity_mask = (identity_mask > 0.6).float()
                    identity_mask = F.avg_pool2d(identity_mask, 3, stride=1, padding=1)

                    # Mask source image
                    masked_identity = identity_frame * identity_mask

                    # Extract identity embedding
                    idt_embed = self.emo_model.idt_embedder_nw(masked_identity)

                    # Get head pose for identity
                    identity_theta = self.emo_model.head_pose_regressor.forward(identity_frame)

                    # Prepare identity data dict
                    identity_dict = {
                        'source_img': identity_frame,
                        'source_mask': identity_mask,
                        'source_theta': identity_theta,
                        'target_img': identity_frame,
                        'target_mask': identity_mask,
                        'target_theta': identity_theta,
                        'idt_embed': idt_embed
                    }

                    # Get expression embedding for identity
                    identity_dict = self.emo_model.expression_embedder_nw(identity_dict, True, False, False)

                    # Get warp embeddings
                    source_warp_embed, _, _, embed_dict = self.emo_model.predict_embed(identity_dict)

                    # Generate XY warps for source
                    source_xy_warp, _ = self.emo_model.xy_generator_nw(source_warp_embed)

                    # Extract source volume
                    source_latents = self.emo_model.local_encoder_nw(masked_identity)
                    c = self.emo_model.args.latent_volume_channels
                    d = self.emo_model.args.latent_volume_depth
                    s = self.emo_model.args.latent_volume_size
                    source_volume = source_latents.view(1, c, d, s, s)

                    # Generate 3D grid for transformations
                    identity_grid_3d = self.emo_model.identity_grid_3d.repeat_interleave(1, dim=0)

                    # Apply inverse theta to create canonical volume
                    theta_inv = torch.zeros(1, 3, 4, device=identity_frame.device)
                    theta_inv[:, :3, :3] = identity_theta[:, :3, :3].transpose(1, 2)
                    theta_inv[:, :3, 3] = -identity_theta[:, :3, :3].transpose(1, 2).bmm(identity_theta[:, :3, 3:4]).squeeze(-1)

                    source_rotation_warp = identity_grid_3d.bmm(theta_inv[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

                    # Create canonical volume by applying inverse warps
                    canonical_volume = self.emo_model.grid_sample(
                        self.emo_model.grid_sample(source_volume, source_xy_warp),
                        source_rotation_warp
                    )

                    # Store identity info
                    identity_info = {
                        'idt_embed': idt_embed,
                        'embed_dict': embed_dict,
                        'canonical_volume': canonical_volume,
                        'source_theta': identity_theta,
                        'source_mask': identity_mask
                    }


                # Process each frame (cherry-picked from calculate_target_warps)
                for t in range(T):
                    frame = frames[:, t]  # [1,C,H,W]

                    with torch.no_grad():
                        # Get target face mask
                        target_mask, _, _, _ = self.emo_model.face_idt.forward(frame)
                        target_mask = (target_mask > 0.6).float()
                        target_mask = F.avg_pool2d(target_mask, 3, stride=1, padding=1)

                        # Get target pose with SRT components
                        target_theta, scale, rotation, translation = self.emo_model.head_pose_regressor.forward(
                            frame, return_srt=True
                        )
                        target_theta = self.convert_theta_format(target_theta)

                        # Create target data dict with source identity
                        data_dict = {
                            'source_img': frame,
                            'source_mask': target_mask,
                            'source_theta': target_theta,
                            'target_img': frame,
                            'target_mask': target_mask,
                            'target_theta': target_theta,
                            'idt_embed': identity_info['idt_embed']  # Use source identity
                        }

                        # Get aligned expression embedding
                        data_dict = self.emo_model.expression_embedder_nw(data_dict, True, False, False)
                        expression_embed = data_dict['source_pose_embed']  # This is the aligned expression

                        # Generate target warps
                        _, target_warp_embed, _, _ = self.emo_model.predict_embed(data_dict)
                        target_uv_warp, _ = self.emo_model.uv_generator_nw(target_warp_embed)

                    # Verify feature shapes
                    assert target_theta.shape == (1, 3, 4), f"Wrong theta shape: {target_theta.shape}"
                    assert scale.shape == (1, 3), f"Wrong scale shape: {scale.shape}"
                    assert rotation.shape == (1, 3), f"Wrong rotation shape: {rotation.shape}"
                    assert translation.shape == (1, 3), f"Wrong translation shape: {translation.shape}"
                    assert expression_embed.shape == (1, 128), f"Wrong expression shape: {expression_embed.shape}"
                    assert target_uv_warp.shape == (1, 16, 64, 64, 3), f"Wrong uv_warp shape: {target_uv_warp.shape}"

                    # Store outputs matching H5 cache structure
                    outputs['theta'].append(target_theta.cpu())
                    outputs['scale'].append(scale.cpu())
                    outputs['rotation'].append(rotation.cpu())
                    outputs['translation'].append(translation.cpu())
                    outputs['expression_embed'].append(expression_embed.cpu())  # Using aligned expression
                    outputs['uv_warps'].append(target_uv_warp.cpu())
                    outputs['target_masks'].append(target_mask.cpu())

                # Stack along time dimension for per-frame features
                per_frame_keys = ['theta', 'scale', 'rotation', 'translation', 'expression_embed',
                                 'uv_warps', 'target_masks']
                for k in per_frame_keys:
                    if k in outputs and isinstance(outputs[k], list) and len(outputs[k]) > 0:
                        outputs[k] = torch.stack(outputs[k], dim=1)

                # DEBUG: Check expression variation
                expr_tensor = outputs['expression_embed']  # [1, T, 128]
                expr_flat = expr_tensor.squeeze(0)  # [T, 128]
                frame_diff = torch.diff(expr_flat, dim=0)  # [T-1, 128]
                diff_norm = torch.norm(frame_diff, dim=-1)  # [T-1]
                is_constant = torch.allclose(expr_flat[0], expr_flat, atol=1e-5)

                logger.debug(f"Expression embed shape: {expr_tensor.shape}")
                logger.debug(f"Constant across frames? {is_constant}")
                logger.debug(f"Frame-to-frame diff - Mean: {diff_norm.mean():.6f}, Max: {diff_norm.max():.6f}")
                logger.debug(f"First frame values (first 5): {expr_flat[0, :5].tolist()}")
                logger.debug(f"Last frame values (first 5): {expr_flat[-1, :5].tolist()}")

                # Add identity info to outputs
                outputs['identity_info'] = identity_info

                # Verify final output shapes matching H5 cache
                logger.debug("\nFinal output shapes (matching H5):")
                d = 16  # depth
                s = 64  # spatial size
                expected_shapes = {
                    'theta': (1, T, 3, 4),  # Target pose (H5: 1, 4, 4 but model uses 3, 4)
                    'scale': (1, T, 3),  # SRT scale - matches H5
                    'rotation': (1, T, 3),  # SRT rotation - matches H5
                    'translation': (1, T, 3),  # SRT translation - matches H5
                    'expression_embed': (1, T, 128),  # Aligned expression - matches H5 target_pose_embed
                    'uv_warps': (1, T, d, s, s, 3),  # Target UV warps - matches H5
                    'target_masks': (1, T, 1, 512, 512),  # Face masks
                }

                for k, expected_shape in expected_shapes.items():
                    if k in outputs:
                        actual_shape = outputs[k].shape

                        # Check per-frame features
                        assert actual_shape == expected_shapes[k], f"Wrong {k} shape: expected {expected_shapes[k]}, got {actual_shape}"
                        logger.debug(f"  {k}: {actual_shape} on {outputs[k].device if hasattr(outputs[k], 'device') else 'CPU'}")

                logger.debug("=== Calculate Target Warps Complete ===\n")

                return outputs
                
            except Exception as e:
                logger.error(f"Error in EMO feature extraction: {str(e)}")
                logger.error(traceback.format_exc())
                logger.error(f"Input frames shape: {frames.shape}")
                logger.error(f"Input frames device: {frames.device}")
                logger.error(f"EMO model device: {next(self.emo_model.parameters()).device}")
                raise

    def _matrix_to_euler_angles(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            matrix: Rotation matrix [3, 3]
            
        Returns:
            Euler angles [3] in radians
        """
        # Handle singularity at pitch = ±90°
        # Clamp to slightly inside [-1, 1] to avoid gradient explosion in asin backward
        sin_pitch = torch.clamp(-matrix[2, 0], -0.9999, 0.9999)  # Safer bounds
        pitch = torch.asin(sin_pitch)
        
        if torch.abs(matrix[2, 0]) < 0.9999:
            # Regular case
            yaw = torch.atan2(matrix[1, 0], matrix[0, 0])
            roll = torch.atan2(matrix[2, 1], matrix[2, 2])
        else:
            # Gimbal lock case
            yaw = torch.atan2(-matrix[0, 1], matrix[1, 1])
            roll = torch.zeros(1, device=matrix.device)
            
        return torch.tensor([pitch, yaw, roll], device=matrix.device)

    def _check_videos_for_audio(self, video_paths: List[str]) -> Dict[str, Dict]:
        """Check which videos have audio streams using ffprobe"""
        audio_status = {}
        total_videos = len(video_paths)
        
        for i, video_path in enumerate(video_paths, 1):
            try:
                # Check if audio file exists in cache
                audio_path = self._get_audio_path(video_path)
                has_cache = audio_path.exists()
                
                command = [
                    'ffprobe', 
                    '-loglevel', 'error',
                    '-show_streams', 
                    '-select_streams', 'a', 
                    '-show_entries', 'stream=codec_type',
                    '-of', 'json',
                    video_path
                ]
                
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8'
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    # Check if there are any audio streams
                    has_audio = bool(data.get('streams', []))
                    if not has_audio:
                        self.tracker.dispatch(VideoEventData(
                            video_path=video_path,
                            event_type=VideoEvent.NO_AUDIO,
                            details={"error": "No audio stream found in video"}
                        ))
                else:
                    has_audio = False
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.PROCESSING_ERROR,
                        details={
                            "error": "FFprobe error checking audio",
                            "stderr": result.stderr
                        }
                    ))
                    
                audio_status[video_path] = {
                    'has_audio': has_audio,
                    'has_cache': has_cache,
                    'last_checked': str(datetime.now())
                }
                
                if i % 10 == 0:
                    logger.info(f"Checked audio for {i}/{total_videos} videos")
                
            except Exception as e:
                logger.error(f"Error checking audio in {video_path}: {str(e)}")
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.PROCESSING_ERROR,
                    details={
                        "error": f"Audio check error: {str(e)}",
                        "traceback": traceback.format_exc()
                    }
                ))
                audio_status[video_path] = {
                    'has_audio': False,
                    'has_cache': False,
                    'error': str(e),
                    'last_checked': str(datetime.now())
                }
                    
        return audio_status

    def _get_audio_path(self, video_path: str) -> Path:
        """Get the path where the audio file should be stored"""
        video_path = Path(video_path)
        if self.cache_audio:
            relative_path = video_path.relative_to(self.video_folder)
            audio_path = self.audio_cache_dir / relative_path.with_suffix('.wav')
        else:
            audio_path = video_path.with_suffix('.wav')
            
        # Ensure parent directory exists
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        return audio_path

    def get_audio_path(self, video_path: str) -> Optional[str]:
        """Public method to get audio path for testing"""
        if not self.audio_status.get(video_path, False):
            return None

        return str(self._get_audio_path(video_path))

    def _get_face_cache_path(self, video_path: str) -> Path:
        """Get the path for cached face attributes"""
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()
        return self.cache_dir / f"face_attrs_{video_hash}.h5"

    def _load_cached_window(self, video_path: str, window_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached window data including face attributes"""
        cache_path = self._get_face_cache_path(video_path)
        if not cache_path.exists():
            return None

        try:
            with h5py.File(cache_path, 'r') as f:
                window_key = f"window_{window_idx}"
                if window_key not in f:
                    return None

                window_group = f[window_key]

                # Load all cached data
                cached_data = {}
                for key in window_group.keys():
                    if key == 'identity_frame':
                        # Load identity frame and expand to full frames tensor
                        identity_frame_data = window_group[key][()]
                        identity_frame = torch.from_numpy(identity_frame_data)
                        # Duplicate identity frame for all frame positions (for compatibility)
                        cached_data['frames'] = identity_frame.unsqueeze(0).repeat(self.window_size, 1, 1, 1)
                    elif key == 'metadata':
                        # Handle metadata specially
                        cached_data['metadata'] = {
                            'video_path': window_group['metadata'].attrs['video_path'],
                            'start_frame': window_group['metadata'].attrs['start_frame'],
                            'fps': window_group['metadata'].attrs['fps'],
                            'has_context': window_group['metadata'].attrs['has_context']
                        }
                    elif key == 'lip_metrics':
                        # Handle nested lip_metrics
                        cached_data['lip_metrics'] = {}
                        for metric_key in window_group['lip_metrics'].keys():
                            cached_data['lip_metrics'][metric_key] = torch.tensor(
                                window_group['lip_metrics'][metric_key][()],
                                dtype=torch.float32
                            )
                    else:
                        # Load tensor data
                        data = window_group[key][()]
                        if key in ['speed_bucket']:
                            cached_data[key] = torch.tensor(data, dtype=torch.long)
                        else:
                            cached_data[key] = torch.tensor(data, dtype=torch.float32)

                return cached_data

        except Exception as e:
            logger.warning(f"Failed to load cached window: {str(e)}")
            return None

    def _save_window_to_cache(self, video_path: str, window_idx: int, window_data: Dict[str, Any]):
        """Save window data including face attributes to cache"""
        cache_path = self._get_face_cache_path(video_path)

        try:
            # Open in append mode to add new windows
            mode = 'a' if cache_path.exists() else 'w'
            with h5py.File(cache_path, mode) as f:
                window_key = f"window_{window_idx}"

                # Remove existing window if it exists
                if window_key in f:
                    del f[window_key]

                window_group = f.create_group(window_key)

                # Save all window data (only save identity frame)
                for key, value in window_data.items():
                    # Handle frames specially - only save first frame as identity_frame
                    if key == 'frames':
                        if isinstance(value, torch.Tensor) and len(value) > 0:
                            # Save only the first frame as identity_frame
                            window_group.create_dataset(
                                'identity_frame',
                                data=value[0].cpu().numpy(),  # Just the first frame
                                compression='gzip',
                                compression_opts=4
                            )
                            logger.debug(f"Saved identity frame for window {window_idx}")
                        continue

                    if key == 'metadata':
                        # Save metadata as attributes
                        meta_group = window_group.create_group('metadata')
                        for meta_key, meta_value in value.items():
                            meta_group.attrs[meta_key] = meta_value
                    elif key == 'lip_metrics':
                        # Save nested lip_metrics
                        lip_group = window_group.create_group('lip_metrics')
                        for metric_key, metric_value in value.items():
                            if isinstance(metric_value, torch.Tensor):
                                lip_group.create_dataset(metric_key, data=metric_value.cpu().numpy())
                            else:
                                lip_group.create_dataset(metric_key, data=metric_value)
                    else:
                        # Save tensor data
                        if isinstance(value, torch.Tensor):
                            window_group.create_dataset(key, data=value.cpu().numpy())
                        elif isinstance(value, np.ndarray):
                            window_group.create_dataset(key, data=value)
                        else:
                            window_group.create_dataset(key, data=value)

        except Exception as e:
            logger.error(f"Failed to save window to cache: {str(e)}")

    def _extract_audio(self, video_path: str, audio_path: Path) -> None:
        """Extract audio from video file using ffmpeg"""
        try:
            # Skip if video has no audio
            if not self.audio_status.get(video_path, False):
                logger.info(f"Skipping audio extraction for {video_path} - no audio stream")
                return
                
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sampling rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                str(audio_path)
            ]
            
            # Create directory if it doesn't exist
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
            if not audio_path.exists():
                raise RuntimeError(f"FFmpeg completed but audio file not created at {audio_path}")
                
            logger.info(f"Successfully extracted audio to {audio_path}")
            
        except Exception as e:
            logger.error(f"🔥 Error extracting audio from {video_path}: {str(e)}")
            raise

    def _get_or_extract_audio(self, video_path: str) -> Path:
        """Get audio file path, extracting audio if necessary"""
        audio_path = self._get_audio_path(video_path)
        
        if not audio_path.exists():
            logger.debug(f"Extracting audio for {video_path}")
            self._extract_audio(video_path, audio_path)
            
        if not audio_path.exists():
            raise RuntimeError(f"Audio file still not found after extraction attempt: {audio_path}")
            
        return audio_path




    def _extract_audio_features(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        use_whisper: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract both whisper/wav2vec and MFCC audio features."""
        try:
            video_name = Path(video_path).name
            logger.debug(f"Starting audio feature extraction for video: {video_name}")
            logger.debug(f"Processing window: start_time={start_time:.2f}s, duration={duration:.2f}s")
            logger.debug(f"Using Whisper: {use_whisper}")

            # Check if video has audio
            has_audio = self.audio_status.get(video_path, {}).get('has_audio', False)
            logger.debug(f"Audio status for video: has_audio={has_audio}")

            if not has_audio:
                logger.warning(f"No audio in video: {video_path}")
                return (
                    torch.zeros((1, self.window_size, 384 if use_whisper else 768)),
                    torch.zeros((1, self.window_size, 13))
                )

            # Get audio path
            audio_path = self._get_or_extract_audio(video_path)
            logger.debug(f"Audio path resolved to: {audio_path}")

            # Load audio
            logger.debug(f"Loading audio file from: {audio_path}")
            waveform, sample_rate = torchaudio.load(str(audio_path))
            logger.debug(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}Hz")
            
            # Calculate window parameters
            window_duration = self.window_size / 30  # 50 frames at 30fps = 1.67s
            samples_needed = int(window_duration * sample_rate)
            start_sample = int(start_time * sample_rate)
            logger.debug(f"Window parameters: duration={window_duration:.2f}s, samples_needed={samples_needed}")

            # Extract audio segment
            audio_segment = waveform[:, start_sample:start_sample + samples_needed]
            logger.debug(f"Extracted audio segment shape: {audio_segment.shape}")

            # Process with Whisper or wav2vec
            with torch.no_grad():
                if use_whisper:
                    # Process audio with Whisper
                    inputs = self.whisper_processor(
                        audio_segment.squeeze(0).numpy(),
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    )
                    
                    # Add required decoder input ids - this fixes the error
                    inputs['decoder_input_ids'] = torch.tensor([[1]]).to(self.whisper_model.device)
                    
                    # Move inputs to model device
                    inputs = {
                        k: v.to(self.whisper_model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()
                    }
                    
                    outputs = self.whisper_model(**inputs)
                    features = outputs.last_hidden_state  # [1, T, 384]
                else:
                    # Process with aligned wav2vec using JoyVASA's approach
                    inputs = self.audio_processor(
                        audio_segment.squeeze(0),
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Use aligned model with BackResample strategy (like JoyVASA)
                    # This extracts at 2x frame rate then downsamples for better temporal info
                    features = self.audio_model(
                        inputs.input_values,
                        output_fps=25,  # Target FPS
                        frame_num=self.window_size,  # Target number of frames
                        use_back_resample=True  # JoyVASA's strategy
                    )  # [1, window_size, 768]

            # Extract MFCC features for SyncNet
 
            logger.debug("\n=== Processing MFCC Features ===")
            logger.debug(f"Input audio_segment shape: {audio_segment.shape}")
            
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={'n_mels': 40}
            )
            logger.debug("Created MFCC transform")
            
            try:
                # Initial MFCC computation
                mfcc_features = mfcc_transform(audio_segment)
                logger.debug(f"Raw MFCC features shape: {mfcc_features.shape}")
                logger.debug(f"MFCC features dtype: {mfcc_features.dtype}")
                logger.debug(f"MFCC features device: {mfcc_features.device}")
                logger.debug(f"MFCC value range: [{mfcc_features.min():.3f}, {mfcc_features.max():.3f}]")
                
                # Handle potential extra dimensions
                if len(mfcc_features.shape) > 2:
                    logger.debug("Squeezing extra dimensions...")
                    mfcc_features = mfcc_features.squeeze()
                    logger.debug(f"After squeeze shape: {mfcc_features.shape}")
                
                # Ensure we have [channels, time] format
                if mfcc_features.shape[0] != 13:
                    logger.debug("Transposing to get [channels, time] format...")
                    mfcc_features = mfcc_features.transpose(0, 1)
                    logger.debug(f"After transpose shape: {mfcc_features.shape}")
                
                # Add batch dimension for interpolation
                mfcc_features = mfcc_features.unsqueeze(0)  # [1, channels, time]
                logger.debug(f"Shape after adding batch dimension: {mfcc_features.shape}")
                
                # Interpolate to match window size
                logger.debug(f"Target window size: {self.window_size}")
                mfcc_features = F.interpolate(
                    mfcc_features,
                    size=self.window_size,
                    mode='linear',
                    align_corners=False
                )
                logger.debug(f"Shape after interpolation: {mfcc_features.shape}")
                
                # Transpose to get [batch, time, channels]
                mfcc_features = mfcc_features.transpose(1, 2)
                logger.debug(f"Final MFCC features shape: {mfcc_features.shape}")
                logger.debug(f"Final value range: [{mfcc_features.min():.3f}, {mfcc_features.max():.3f}]")
                logger.debug("=== MFCC Processing Complete ===\n")
                
            except Exception as e:
                logger.error(f"Error in MFCC processing: {str(e)}")
                logger.error(f"Error location: {traceback.format_exc()}")
                # Return zero tensor with correct shape
                mfcc_features = torch.zeros(1, self.window_size, 13, device=audio_segment.device)
                logger.debug(f"Returning zero tensor with shape: {mfcc_features.shape}")

            # Extract mel spectrogram for Synchformer (128 mel bins, 50 time frames)
            logger.debug("\n=== Processing Mel Spectrogram for Synchformer ===")
            try:
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=1024,
                    hop_length=audio_segment.shape[-1] // self.window_size,  # Ensure 50 frames
                    n_mels=128,  # Standard for Synchformer
                    f_min=0.0,
                    f_max=8000.0
                )

                mel_spec = mel_transform(audio_segment)  # [1, n_mels, T]

                # Ensure exactly window_size (50) time frames
                if mel_spec.shape[-1] != self.window_size:
                    # Interpolate to exact window_size
                    mel_spec = F.interpolate(
                        mel_spec.unsqueeze(0),  # [1, 1, n_mels, T]
                        size=(128, self.window_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [1, n_mels, T]

                # Convert to log scale
                mel_spec = torch.log(mel_spec + 1e-8)

                # Transpose to [1, T, n_mels] for consistency
                mel_spec = mel_spec.transpose(1, 2)  # [1, window_size, 128]

                logger.debug(f"Mel spectrogram shape: {mel_spec.shape}")
                logger.debug(f"Mel spectrogram range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")

            except Exception as e:
                logger.error(f"Error in mel spectrogram processing: {str(e)}")
                logger.error(traceback.format_exc())
                mel_spec = torch.zeros(1, self.window_size, 128, device=audio_segment.device)

            return features, mfcc_features, audio_segment, mel_spec

        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zero tensors including audio waveform and mel spec
            samples_needed = int(self.window_size * sample_rate / fps)
            return (
                torch.zeros((1, self.window_size, 384 if use_whisper else 768)),
                torch.zeros((1, self.window_size, 13)),
                torch.zeros((1, samples_needed)),  # audio_segment with correct shape
                torch.zeros((1, self.window_size, 128))  # mel_spec [1, 50, 128]
            )

    def _extract_phoneme_sequence(
        self,
        audio_waveform: torch.Tensor,
        sample_rate: int,
        num_queries: int = 8
    ) -> torch.Tensor:
        """
        Extract phoneme sequence from audio using wav2vec2 phoneme recognition.

        Args:
            audio_waveform: Raw audio tensor [samples] or [1, samples]
            sample_rate: Audio sample rate (usually 16000 Hz)
            num_queries: Number of latent queries to align phonemes to (default: 8)

        Returns:
            Phoneme IDs tensor [num_queries] - one phoneme per latent query
        """
        try:
            # Ensure audio is 1D
            if audio_waveform.ndim > 1:
                audio_waveform = audio_waveform.squeeze(0)

            # Process audio with phoneme model
            inputs = self.worker_state.phoneme_processor(
                audio_waveform.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            with torch.no_grad():
                logits = self.worker_state.phoneme_model(inputs).logits  # [1, T_phoneme, vocab_size]

            phoneme_ids = torch.argmax(logits, dim=-1)  # [1, T_phoneme]
            phoneme_seq = phoneme_ids.squeeze(0).cpu()  # [T_phoneme]

            # Align to num_queries using average pooling
            # Handle case where phoneme_seq might be shorter than num_queries
            if len(phoneme_seq) < num_queries:
                # Pad with zeros if too short
                pooled_phoneme = torch.zeros(num_queries, dtype=torch.long)
                pooled_phoneme[:len(phoneme_seq)] = phoneme_seq
            else:
                # Average pool to match num_queries
                kernel_size = max(1, len(phoneme_seq) // num_queries)
                stride = kernel_size

                # Use max pooling instead of avg for phoneme IDs (preserves discrete values)
                pooled = torch.nn.functional.max_pool1d(
                    phoneme_seq.unsqueeze(0).unsqueeze(0).float(),
                    kernel_size=kernel_size,
                    stride=stride
                )
                pooled_phoneme = pooled.squeeze().long()[:num_queries]

                # Pad if needed (in case pooling didn't produce exactly num_queries)
                if len(pooled_phoneme) < num_queries:
                    padded = torch.zeros(num_queries, dtype=torch.long)
                    padded[:len(pooled_phoneme)] = pooled_phoneme
                    pooled_phoneme = padded

            logger.debug(f"Extracted phoneme sequence: {phoneme_seq.shape} -> {pooled_phoneme.shape}")
            return pooled_phoneme  # [num_queries]

        except Exception as e:
            logger.error(f"Error extracting phoneme sequence: {str(e)}")
            logger.error(traceback.format_exc())
            # Return zeros on error
            return torch.zeros(num_queries, dtype=torch.long)

    def _preextract_all_audio(self):
        """Pre-extract audio with progress bar and error handling"""
        for i, video_path in enumerate(tqdm(self.video_paths, desc="Extracting audio")):
            try:
                if self.audio_status.get(video_path, {}).get('has_audio', False):
                    audio_path = self._get_or_extract_audio(video_path)
                    if i % 10 == 0:
                        logger.info(f"Processed {i+1}/{len(self.video_paths)} videos")
            except Exception as e:
                logger.error(f"Failed to extract audio for {video_path}: {e}")
                continue


    def _get_audio_path(self, video_path: str) -> Path:
        """Get the path where the audio file should be stored"""
        video_path = Path(video_path)
        if self.cache_audio:
            # Use cache directory with original filename structure
            relative_path = video_path.relative_to(self.video_folder)
            audio_path = self.audio_cache_dir / relative_path.with_suffix('.wav')
            audio_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Use same directory as video
            audio_path = video_path.with_suffix('.wav')
        return audio_path

    def _extract_audio(self, video_path: str, audio_path: Path) -> None:
        """Extract audio from video file using ffmpeg"""
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sampling rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                str(audio_path)
            ]
            
            # Create directory if it doesn't exist
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
            if not audio_path.exists():
                raise RuntimeError(f"FFmpeg completed but audio file not created at {audio_path}")
                
            logger.info(f"Successfully extracted audio to {audio_path}")
            
        except Exception as e:
            logger.error(f"🔥 Error extracting audio from {video_path}: {str(e)}")
            raise

    def _get_or_extract_audio(self, video_path: str) -> Path:
        """Get audio file path, extracting audio if necessary"""
        audio_path = self._get_audio_path(video_path)
        
        if not audio_path.exists():
            logger.info(f"Extracting audio for {video_path}")
            self._extract_audio(video_path, audio_path)
            
        if not audio_path.exists():
            raise RuntimeError(f"Audio file still not found after extraction attempt: {audio_path}")
            
        return audio_path


    def get_audio_path(self, video_path: str) -> str:
        """Public method to get audio path for testing"""
        return str(self._get_or_extract_audio(video_path))

    def __len__(self) -> int:
        return len(self.windows)

    def save_cache(self):
        """Explicitly save any pending cache data."""
        if hasattr(self, 'cache_type') and self.cache_type == 'single_bucket' and hasattr(self, '_pending_windows'):
            self._save_pending_windows()
            logger.info("Cache saved successfully")

    def __del__(self):
        """Cleanup - save any pending windows before destruction."""
        if hasattr(self, 'cache_type') and self.cache_type == 'single_bucket' and hasattr(self, '_pending_windows'):
            self._save_pending_windows()



            
    def _extract_face_attributes(self, frame: np.ndarray,video_path) -> Optional[Dict[str, np.ndarray]]:
        """Extract face attributes using MediaPipe and L2CS"""
        try:
            # Ensure frame is in RGB format and uint8 [0,255]
            if frame.dtype == np.float32:
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            # Handle different color channel arrangements
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.shape[2] == 3:  # Assume BGR if not explicitly RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get image dimensions
            height, width = frame.shape[:2]

            # Get face landmarks with protobuf error handling
            try:
                results = self.face_mesh.process(frame)
            except (AttributeError, TypeError) as e:
                if "SymbolDatabase" in str(e) or "GetPrototype" in str(e):
                    # Protobuf compatibility issue - return None to skip this frame
                    logger.error(f"MediaPipe protobuf issue for {video_path}: {str(e)}")
                    return None
                else:
                    raise

            if not results.multi_face_landmarks:
                # Get video path from filename if part of error context
                logger.error(f"frame.shape: {frame.shape}")
                logger.error(f"No faces detected in frame from video: {video_path}")
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.FACE_DETECTION_FAILED,
                    details={
                        "frame_shape": frame.shape,
                        "error": "No face landmarks detected"
                    }
                ))
                return None

                
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array with correct scale
            landmarks = np.array([
                [lm.x * width, lm.y * height, lm.z * width]
                for lm in face_landmarks.landmark
            ], dtype=np.float32)

            # Map landmarks to 68-point format
            landmarks_68 = self._map_to_68_landmarks(landmarks)
            
            # Calculate bounding box
            x_min, y_min = np.min(landmarks_68[:, :2], axis=0)
            x_max, y_max = np.max(landmarks_68[:, :2], axis=0)
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

            # Extract face region for emotion recognition
            margin = 0.2
            h, w = y_max - y_min, x_max - x_min
            x1 = max(0, int(x_min - margin * w))
            x2 = min(width, int(x_max + margin * w))
            y1 = max(0, int(y_min - margin * h))
            y2 = min(height, int(y_max + margin * h))
            
            if x1 >= x2 or y1 >= y2:
                logger.warning("Invalid face crop region")
                return None
                
            face_crop = frame[y1:y2, x1:x2]

            # Extract L2CS gaze
            try:
                # Skip L2CS if pipeline not available or has protobuf issues
                if hasattr(self.worker_state, 'l2cs_pipeline') and self.worker_state.l2cs_pipeline is not None:
                    try:
                        # Ensure frame is in correct format for L2CS
                        if len(frame.shape) == 2:
                            frame_l2cs = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2] == 4:
                            frame_l2cs = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        else:
                            frame_l2cs = frame

                        # Get predictions using worker state pipeline
                        results = self.worker_state.l2cs_pipeline.step(frame_l2cs)

                        # Get pitch and yaw (converting radians to degrees)
                        if len(results.pitch) > 0:
                            pitch = float(results.pitch[0] * 180/np.pi)
                            yaw = float(results.yaw[0] * 180/np.pi)
                        else:
                            pitch = 0.0
                            yaw = 0.0

                        gaze = np.array([pitch, yaw], dtype=np.float32)
                    except (AttributeError, ImportError) as e:
                        # Protobuf or L2CS issue - just use default values
                        if "SymbolDatabase" in str(e) or "GetPrototype" in str(e):
                            logger.debug("L2CS protobuf issue detected, using default gaze values")
                        else:
                            logger.debug(f"L2CS error: {str(e)}")
                        gaze = np.zeros(2, dtype=np.float32)
                else:
                    gaze = np.zeros(2, dtype=np.float32)

            except Exception as e:
                logger.debug(f"Error in L2CS gaze extraction: {str(e)}")
                gaze = np.zeros(2, dtype=np.float32)

            # Calculate face size/distance
            right_eye = np.mean(landmarks_68[42:48], axis=0)  # Right eye center
            left_eye = np.mean(landmarks_68[36:42], axis=0)   # Left eye center
            face_width = np.linalg.norm(right_eye - left_eye)
            face_size = face_width / width
            distance = np.array([face_size], dtype=np.float32)

            # Get emotion using existing emotion recognizer
            emotion_label, emotion_va = self._get_emotion(face_crop)

            return {
                'landmarks': landmarks_68,
                'emotion': emotion_va,  # VA values [valence, arousal]
                'emotion_label': emotion_label,  # String label like "sad", "happy", etc.
                'gaze': gaze,  # L2CS gaze results
                'head_distance': distance,
                'bbox': bbox
            }

        except Exception as e:
            logger.error(f"Error in face attribute extraction: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
                
    def _map_to_68_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Helper method to map MediaPipe landmarks to 68-point format"""
        # MediaPipe to 68-point mapping (move the mapping code here)
        FACIAL_LANDMARKS_68_MAPPING = {
            "jaw": list(range(0, 17)),
            "right_eyebrow": list(range(17, 22)),
            "left_eyebrow": list(range(22, 27)),
            "nose_bridge": list(range(27, 31)),
            "nose_tip": list(range(31, 36)),
            "right_eye": list(range(36, 42)),
            "left_eye": list(range(42, 48)),
            "outer_mouth": list(range(48, 60)),
            "inner_mouth": list(range(60, 68))
        }

        MEDIAPIPE_TO_68_MAPPING = {
            "jaw": [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361],
            "right_eyebrow": [70, 63, 105, 66, 107],
            "left_eyebrow": [336, 296, 334, 293, 300],
            "nose_bridge": [168, 6, 197, 195],
            "nose_tip": [4, 242, 141, 94, 370],
            "right_eye": [33, 7, 163, 144, 145, 153],
            "left_eye": [362, 382, 381, 380, 374, 373],
            "outer_mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409],
            "inner_mouth": [78, 95, 88, 178, 87, 14, 317, 402]
        }

        landmarks_68 = np.zeros((68, 3), dtype=np.float32)
        current_index = 0

        for region, target_indices in FACIAL_LANDMARKS_68_MAPPING.items():
            mediapipe_indices = MEDIAPIPE_TO_68_MAPPING[region]
            num_points = len(target_indices)
            
            if num_points != len(mediapipe_indices):
                logger.warning(f"Mismatch in number of points for {region}")
                continue
                
            for idx, mp_idx in zip(target_indices, mediapipe_indices):
                if idx < 68 and mp_idx < len(landmarks):  # Bounds checking
                    landmarks_68[idx] = landmarks[mp_idx]
                    current_index += 1

        return landmarks_68
    
            
    def _extract_face_landmarks(self, frame: np.ndarray, video_path: str = "") -> Optional[Dict[str, np.ndarray]]:
        """
        Extract facial landmarks using MediaPipe.
        This is the SHARED helper used by both dataset and loss computation.
        """
        try:
            # Ensure frame is in RGB format and uint8 [0,255]
            if frame.dtype == np.float32:
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            # Handle different color channel arrangements
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.shape[2] == 3:  # Assume BGR if not explicitly RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get image dimensions
            height, width = frame.shape[:2]

            # Get face landmarks
            results = self.face_mesh.process(frame)

            if not results.multi_face_landmarks:
                if video_path:
                    logger.error(f"No faces detected in frame from video: {video_path}")
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]

            # Convert landmarks to numpy array
            landmarks = np.array([
                [lm.x * width, lm.y * height, lm.z * width]
                for lm in face_landmarks.landmark
            ], dtype=np.float32)

            # Define landmark indices
            landmark_indices = {
                'lips': list(range(61, 69)) + list(range(48, 60)),  # 20 points
                'right_eye': [33, 133, 157, 158, 159, 160, 161, 173],  # 8 points
                'left_eye': [362, 385, 386, 387, 388, 466, 263],  # 7 points
                'jaw': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409],  # 10 points
                'nose': [1, 2, 98, 327]  # 4 points
            }

            # Extract landmark groups
            extracted_landmarks = {}
            for key, indices in landmark_indices.items():
                extracted_landmarks[key] = np.array([landmarks[i] for i in indices])

            # Validate shapes
            expected_shapes = {
                'lips': (20, 3),
                'right_eye': (8, 3),
                'left_eye': (7, 3),
                'jaw': (10, 3),
                'nose': (4, 3)
            }

            # Verify and pad if necessary
            for key, expected_shape in expected_shapes.items():
                current_shape = extracted_landmarks[key].shape
                if current_shape != expected_shape:
                    logger.warning(f"Shape mismatch for {key}: got {current_shape}, expected {expected_shape}")
                    # Pad with zeros if we have too few points
                    if current_shape[0] < expected_shape[0]:
                        padding = np.zeros((expected_shape[0] - current_shape[0], 3), dtype=np.float32)
                        extracted_landmarks[key] = np.vstack([extracted_landmarks[key], padding])
                    # Truncate if we have too many points
                    else:
                        extracted_landmarks[key] = extracted_landmarks[key][:expected_shape[0]]

            # Add blink detection
            def get_eye_aspect_ratio(eye_points: np.ndarray) -> float:
                if len(eye_points) < 6:  # Need at least 6 points for EAR
                    return 0.3  # Default open value

                # Compute the euclidean distances
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                C = np.linalg.norm(eye_points[0] - eye_points[3])

                # Compute the eye aspect ratio
                ear = (A + B) / (2.0 * C) if C > 0 else 0.3
                return ear

            # Get EAR for both eyes
            left_ear = get_eye_aspect_ratio(extracted_landmarks['left_eye'])
            right_ear = get_eye_aspect_ratio(extracted_landmarks['right_eye'])

            # Determine blink state
            EAR_THRESHOLD = 0.2
            left_openness = min(max(left_ear / 0.3, 0), 1)  # Normalize to [0,1]
            right_openness = min(max(right_ear / 0.3, 0), 1)

            # Add blink state
            extracted_landmarks['blink_state'] = np.array([
                0 if (left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD) else 2,  # Phase
                left_openness,   # Left eye openness
                right_openness   # Right eye openness
            ])

            return extracted_landmarks

        except Exception as e:
            logger.error(f"Error extracting landmarks: {str(e)}")
            logger.error(traceback.format_exc())
            return None





    
        
    def _emotion_logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert emotion logits to probabilities using softmax"""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities back to logits"""
        return np.log(np.clip(probs, 1e-7, 1.0))

    def _verify_attributes(self, attrs: Optional[Dict[str, np.ndarray]]) -> bool:
        """Verify that extracted attributes are valid"""
        if attrs is None:
            return False
            
        try:
            required_shapes = {
                'landmarks': (68, 3),
                'emotion': (2,),  # Changed to 2 for valence-arousal
                'gaze': (2,),
                'head_distance': (1,),
                'bbox': (4,)
            }
            
            required_dtypes = {
                'landmarks': np.float32,
                'emotion': np.float32,
                'gaze': np.float32,
                'head_distance': np.float32,
                'bbox': np.int32
            }
            
            for key, expected_shape in required_shapes.items():
                if key not in attrs:
                    logger.error(f"Missing {key} in attributes")
                    return False
                if not isinstance(attrs[key], np.ndarray):
                    logger.error(f"{key} is not a numpy array")
                    return False
                if attrs[key].shape != expected_shape:
                    logger.error(f"Wrong shape for {key}: expected {expected_shape}, got {attrs[key].shape}")
                    return False
                if attrs[key].dtype != required_dtypes[key]:
                    logger.error(f"Wrong dtype for {key}: expected {required_dtypes[key]}, got {attrs[key].dtype}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"🔥 Error verifying attributes: {str(e)}")
            return False




    def _extract_frames(
            self, 
            video_path: str,
            start_frame: int,
            num_frames: int
        ) -> Tuple[List[torch.Tensor], List[int]]:
            """Extract frames with proper validation."""
            frames = []
            frame_indices = []
            
            try:
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Validate frame range
                if start_frame >= total_frames:
                    logger.error(f"Start frame {start_frame} exceeds video length {total_frames}")
                    cap.release()
                    return [], []
                    
                # Adjust end frame if needed
                end_frame = min(start_frame + num_frames, total_frames)
                actual_frames = end_frame - start_frame
                
                if actual_frames < num_frames:
                    logger.warning(
                        f"Not enough frames in video: needed {num_frames}, "
                        f"got {actual_frames} (frames {start_frame} to {end_frame})"
                    )
                    cap.release()
                    return [], []

                # Seek to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Read frames
                for i in range(actual_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Convert and normalize frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.preprocess_transform(frame)
                    
                    frames.append(frame_tensor)
                    frame_indices.append(start_frame + i)
                    
                cap.release()
                
                # Validate frame count
                if len(frames) != num_frames:
                    logger.warning(
                        f"Incorrect frame count: expected {num_frames}, got {len(frames)}"
                    )
                    return [], []
                    
                return frames, frame_indices
                
            except Exception as e:
                logger.error(f"Error extracting frames: {str(e)}")
                cap.release()
                return [], []
            

    def visualize_sample(self, sample: Dict[str, torch.Tensor]):
        """Visualize a single sample including EMO features"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4)
        
        # Plot frames
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            frame = sample['frames'][i].permute(1, 2, 0).numpy()
            # Proper normalization handling
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame = (frame - frame_min) / (frame_max - frame_min)
            else:
                frame = np.zeros_like(frame)
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(f'Frame {i}')
        
        # Plot canonical volume visualization
        ax = fig.add_subplot(gs[1, 0])
        # Take mean across channels for visualization
        volume_slice = sample['canonical_volume'][0, :, 
            sample['canonical_volume'].shape[2]//2].mean(dim=0).numpy()
        im = ax.imshow(volume_slice, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title('Canonical Volume (Mid Slice)')
        
        # Plot emotion probabilities
        ax = fig.add_subplot(gs[1, 1])
        emotions = F.softmax(sample['emotion'][0], dim=-1)
        emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 
                        'Fear', 'Disgust', 'Anger', 'Contempt']
        ax.bar(range(len(emotion_labels)), emotions.numpy())
        ax.set_xticks(range(len(emotion_labels)))
        ax.set_xticklabels(emotion_labels, rotation=45)
        ax.set_title('Emotions')
        
        # Plot gaze trajectories
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(sample['gaze'][:, 0].numpy(), label='Yaw')
        ax.plot(sample['gaze'][:, 1].numpy(), label='Pitch')
        ax.legend()
        ax.set_title('Gaze Trajectories')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (rad)')
        
        # Plot identity embeddings over sequence - FIXED
        ax = fig.add_subplot(gs[1, 3])
        # Properly reshape id_embed for visualization
        id_embed = sample['id_embed'][0]  # Take first frame
        id_embed_flat = id_embed.view(-1).numpy()  # Flatten to 1D
        ax.plot(id_embed_flat)  # Plot flattened embedding
        ax.set_title('Identity Embedding (First Frame)')
        ax.set_xlabel('Embedding Dimension')
        
        # Plot audio features
        ax = fig.add_subplot(gs[2, :2])
        audio_feat = sample['audio_features'][0].numpy()
        im = ax.imshow(audio_feat.T, aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title('Audio Features')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature Dimension')
        
        # Plot head distance over time
        ax = fig.add_subplot(gs[2, 2:])
        ax.plot(sample['head_distance'].squeeze().numpy())
        ax.set_title('Head Distance')
        ax.set_xlabel('Frame')
        ax.set_ylabel('head_distance')
        
        plt.tight_layout()
        plt.savefig(self.sample_dir / 'sample_visualization.png')
        plt.close()
            
    def _get_lip_motion_sequence(self, lip_landmarks: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Calculate lip motion sequence from a list of lip landmarks.
        
        Args:
            lip_landmarks: List of lip landmark arrays, each of shape [20, 3]
            
        Returns:
            Lip motion sequence array of shape [T, 20, 3] or None if invalid
        """
        try:
            if not lip_landmarks or len(lip_landmarks) < 2:
                logger.warning("Insufficient lip landmarks for motion calculation")
                return None
                
            lip_sequences = []
            prev_landmarks = lip_landmarks[0]
            
            # First frame has zero motion
            lip_sequences.append(np.zeros_like(prev_landmarks))
            
            # Calculate motion vectors for subsequent frames
            for curr_landmarks in lip_landmarks[1:]:
                lip_motion = curr_landmarks - prev_landmarks
                lip_sequences.append(lip_motion)
                prev_landmarks = curr_landmarks
                
            # Stack into sequence
            lip_motion_sequence = np.stack(lip_sequences)  # [T, 20, 3]
            
            # Normalize motion vectors
            max_motion = np.abs(lip_motion_sequence).max()
            if max_motion > 0:
                lip_motion_sequence = lip_motion_sequence / max_motion
                
            return lip_motion_sequence
            
        except Exception as e:
            logger.error(f"Error calculating lip motion: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        
    def _extract_face_landmarks(self, frame: np.ndarray, video_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract facial landmarks using MediaPipe."""
        try:
            # Ensure frame is in RGB format and uint8 [0,255]
            if frame.dtype == np.float32:
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
                    
            # Handle different color channel arrangements
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.shape[2] == 3:  # Assume BGR if not explicitly RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
            # Get image dimensions
            height, width = frame.shape[:2]
            
            # Get face landmarks
            results = self.face_mesh.process(frame)
            # logger.info(f"MediaPipe results: {results}")
            # logger.info(f"Processing frame from self.face_mesh: {self.face_mesh}, frame shape: {frame.shape}")
            
            if not results.multi_face_landmarks:
                logger.error(f"No faces detected in frame from video: {video_path}")
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            landmarks = np.array([
                [lm.x * width, lm.y * height, lm.z * width]
                for lm in face_landmarks.landmark
            ], dtype=np.float32)

            # Extract specific landmark groups
            landmarks = {
                'lips': np.array([landmarks[i] for i in list(range(61, 69)) + list(range(48, 60))]),
                'right_eye': np.array([landmarks[i] for i in [33, 133, 157, 158, 159, 160, 161, 173]]),
                'left_eye': np.array([landmarks[i] for i in [362, 385, 386, 387, 388, 466, 263]]),
                'jaw': np.array([landmarks[i] for i in [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]]),
                'nose': np.array([landmarks[i] for i in [1, 2, 98, 327]])
            }
            
               # Add blink detection using eye aspect ratio (EAR)
            def get_eye_aspect_ratio(eye_points: np.ndarray) -> float:
                # Compute the euclidean distances between the vertical eye landmarks
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                
                # Compute the euclidean distance between the horizontal eye landmarks
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                
                # Compute the eye aspect ratio
                ear = (A + B) / (2.0 * C)
                return ear
                
            # Get EAR for both eyes
            left_ear = get_eye_aspect_ratio(landmarks['left_eye'])
            right_ear = get_eye_aspect_ratio(landmarks['right_eye'])
            
            # Determine blink state
            EAR_THRESHOLD = 0.2
            left_openness = min(max(left_ear / 0.3, 0), 1)  # Normalize to [0,1]
            right_openness = min(max(right_ear / 0.3, 0), 1)
            
            # Add blink state
            landmarks['blink_state'] = np.array([
                0 if (left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD) else 2,  # Phase
                left_openness,   # Left eye openness
                right_openness   # Right eye openness
            ])
            
        
            return landmarks

        except Exception as e:
            logger.error(f"Error extracting landmarks: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single window by index with proper audio feature handling"""
        try:
            # Get window info for this index
            if idx >= len(self.windows):
                raise IndexError(f"Index {idx} out of range for {len(self.windows)} windows")
            
            window = self.windows[idx]
            video_path = window['video_path']
            # logger.debug(f"Processing window {idx} from video: {video_path}")
            
            # Check if we have cached data for this specific window
            if self.cache_type == 'per_video':
                # For per-video cache, load by video path and window index
                cached_data = self.cache.load_window(
                    video_path=video_path,
                    window_idx=window['window_idx'],
                    load_frames=True  # Load frames from disk
                )
                if cached_data is not None:
                    logger.info(f"👽 Getting cached window {window['window_idx']} from per-video cache for {Path(video_path).name}")
                    # Ensure metadata contains required fields from the window
                    if 'metadata' not in cached_data:
                        cached_data['metadata'] = {}
                    cached_data['metadata'].update({
                        'video_path': str(video_path),
                        'start_frame': window['start_frame'],
                        'window_idx': window['window_idx'],
                        'fps': window.get('fps', 30),
                        'has_context': window.get('has_context', False)
                    })

                    # Ensure emotion_label exists (add default if missing from old cache)
                    if 'emotion_label' not in cached_data:
                        # Get sequence length from any tensor in the data
                        seq_len = 50  # default
                        for key in ['theta', 'expression_embed', 'emotion']:
                            if key in cached_data and isinstance(cached_data[key], torch.Tensor):
                                seq_len = cached_data[key].shape[0]
                                break
                        # Create default neutral labels for all frames
                        cached_data['emotion_label'] = ['neutral'] * seq_len
                        logger.debug(f"Added default emotion_label for window {window['window_idx']} (length: {seq_len})")

                    return cached_data
            elif self.cache_type == 'single_bucket':
                # For single-bucket cache, load by index directly
                cached_data = self.cache.load_window(idx)
                if cached_data is not None:
                    logger.info(f"👽 Getting cached window {idx} from H5 cache")
                    # Ensure metadata contains required fields from the window
                    if 'metadata' not in cached_data:
                        cached_data['metadata'] = {}
                    cached_data['metadata'].update({
                        'video_path': str(video_path),
                        'start_frame': window['start_frame'],
                        'window_idx': window['window_idx'],
                        'fps': window.get('fps', 30),
                        'has_context': window.get('has_context', False)
                    })

                    # Load frames from disk cache if enabled and not in H5
                    if self.cache_frames_to_disk and self.frame_cache and 'frames' not in cached_data:
                        frames = self.frame_cache.load_frames(
                            video_path=video_path,
                            window_idx=window['window_idx'],
                            as_tensor=True
                        )
                        if frames is not None:
                            cached_data['frames'] = frames
                            logger.info(f"📀 Loaded frames from disk cache for window {idx}")
                        else:
                            # RECOVERY: Regenerate frames from video on-the-fly
                            logger.warning(f"⚠️ Frames missing from disk for window {idx}, regenerating from video...")
                            start_frame = cached_data['metadata'].get('start_frame', 0)
                            regenerated_frames, _ = self._extract_frames(video_path, start_frame, self.sequence_length)

                            if regenerated_frames and len(regenerated_frames) > 0:
                                frames_tensor = torch.stack(regenerated_frames)
                                cached_data['frames'] = frames_tensor
                                # Save to disk cache for future use
                                try:
                                    self.frame_cache.save_frames(video_path, window['window_idx'], frames_tensor, format='png')
                                    logger.info(f"✅ Regenerated and saved {len(regenerated_frames)} frames for window {idx}")
                                except Exception as e:
                                    logger.warning(f"Failed to save regenerated frames: {e}")
                            else:
                                logger.error(f"❌ Could not regenerate frames for window {idx} from video {video_path}")
                                # Return None to skip this window
                                return None
                    elif 'frames' not in cached_data:
                        logger.error(f"❌ CRITICAL: frames not in cached_data and frame loading disabled!")
                        logger.error(f"   cache_frames_to_disk={self.cache_frames_to_disk}, frame_cache={self.frame_cache is not None}")

                    # Load emo_frames from disk cache if enabled and not in H5
                    if self.cache_emo_frames_to_disk and self.emo_frame_cache and 'emo_frames' not in cached_data:
                        emo_frames = self.emo_frame_cache.load_frames(
                            video_path=video_path,
                            window_idx=window['window_idx'],
                            as_tensor=True
                        )
                        if emo_frames is not None:
                            cached_data['emo_frames'] = emo_frames
                            logger.info(f"📀 Loaded emo_frames from disk cache for window {idx}")
                        else:
                            # FAIL HARD: emo_frames are REQUIRED for training
                            logger.error(f"❌ CRITICAL: emo_frames missing from disk for window {idx}")
                            logger.error(f"   Video: {video_path}, window_idx: {window['window_idx']}")
                            logger.error(f"   Expected at: {self.emo_frame_cache.get_window_dir(video_path, window['window_idx'])}")
                            logger.error(f"   This window CANNOT be used for training without emo_frames!")
                            logger.error(f"   ACTION: Re-run preprocessing with --cache-emo-frames to generate missing emo_frames")
                            # Return None to exclude this window from training
                            return None

                    # Ensure emotion_label exists (add default if missing from old cache)
                    if 'emotion_label' not in cached_data:
                        # Get sequence length from any tensor in the data
                        seq_len = 50  # default
                        for key in ['theta', 'expression_embed', 'emotion']:
                            if key in cached_data and isinstance(cached_data[key], torch.Tensor):
                                seq_len = cached_data[key].shape[0]
                                break
                        # Create default neutral labels for all frames
                        cached_data['emotion_label'] = ['neutral'] * seq_len
                        logger.debug(f"Added default emotion_label for window {idx} (length: {seq_len})")

                   

                    return cached_data
            elif self.cache_type == 'chunked':
                # For chunked cache (WindowCache), load from chunk
                chunk_idx = window['start_frame'] // self.cache.chunk_size
                chunk = self.cache.load_chunk(video_path, chunk_idx)
                if chunk is not None and f"window_{window['window_idx']}" in chunk:
                    cached_data = chunk[f"window_{window['window_idx']}"]
                    logger.info(f"👽 Getting cached window {window['window_idx']} from chunk {chunk_idx} for {Path(video_path).name}")
                    # Ensure metadata contains required fields from the window
                    if 'metadata' not in cached_data:
                        cached_data['metadata'] = {}
                    cached_data['metadata'].update({
                        'video_path': str(video_path),
                        'start_frame': window['start_frame'],
                        'window_idx': window['window_idx'],
                        'fps': window.get('fps', 30),
                        'has_context': window.get('has_context', False)
                    })

                    # Ensure emotion_label exists (add default if missing from old cache)
                    if 'emotion_label' not in cached_data:
                        # Get sequence length from any tensor in the data
                        seq_len = 50  # default
                        for key in ['theta', 'expression_embed', 'emotion']:
                            if key in cached_data and isinstance(cached_data[key], torch.Tensor):
                                seq_len = cached_data[key].shape[0]
                                break
                        # Create default neutral labels for all frames
                        cached_data['emotion_label'] = ['neutral'] * seq_len
                        logger.debug(f"Added default emotion_label for window {window['window_idx']} (length: {seq_len})")

                   
                    return cached_data
            else:
                # For built-in cache
                cached_data = self._load_cached_window(video_path, window['window_idx'])
                if cached_data is not None:
                    logger.info(f"👽 Getting cached window {window['window_idx']} for video {Path(video_path).name}")
                    # Ensure metadata contains required fields from the window
                    if 'metadata' not in cached_data:
                        cached_data['metadata'] = {}
                    cached_data['metadata'].update({
                        'video_path': str(video_path),
                        'start_frame': window['start_frame'],
                        'window_idx': window['window_idx'],
                        'fps': window.get('fps', 30),
                        'has_context': window.get('has_context', False)
                    })

                    # Ensure emotion_label exists (add default if missing from old cache)
                    if 'emotion_label' not in cached_data:
                        # Get sequence length from any tensor in the data
                        seq_len = 50  # default
                        for key in ['theta', 'expression_embed', 'emotion']:
                            if key in cached_data and isinstance(cached_data[key], torch.Tensor):
                                seq_len = cached_data[key].shape[0]
                                break
                        # Create default neutral labels for all frames
                        cached_data['emotion_label'] = ['neutral'] * seq_len
                        logger.debug(f"Added default emotion_label for window {window['window_idx']} (length: {seq_len})")

                   
                    return cached_data

            # Process the single window
            try:
                # Extract frames
                frames, frame_indices = self._extract_frames(
                    video_path,
                    window['start_frame'],
                    self.sequence_length
                )
                
                if not frames:
                    return self._get_zero_sample()

                # Extract EMO features
                frames_tensor = torch.stack(frames)
                with torch.no_grad():
                    emo_features = self._extract_emo_features(frames_tensor)
                    if not emo_features:
                        return self._get_zero_sample()

                    # Extract both types of audio features + mel spectrogram
                    wav2vec_features, mfcc_features, audio_segment, mel_spec = self._extract_audio_features(
                        video_path,
                        start_time=window['start_frame'] / window['fps'],
                        duration=self.window_size / window['fps']
                    )

                    # Extract phoneme sequence for self-supervised phoneme prediction
                    # num_queries=8 matches TalkVidAudioProjection's default
                    phoneme_gt = self._extract_phoneme_sequence(
                        audio_waveform=audio_segment,
                        sample_rate=16000,  # Standard sample rate for wav2vec2
                        num_queries=8
                    )

                    # Process face attributes
                    gaze_angles = []
                    emotion_logits = []
                    emotion_labels = []  # Store emotion labels (e.g., "sad", "happy")
                    distances = []
                    landmarks_list = []
                    speed_buckets = []


                    # Initialize lists for landmark groups
                    lips_landmarks = []
                    right_eye_landmarks = []
                    left_eye_landmarks = []
                    jaw_landmarks = []
                    nose_landmarks = []
                    blink_states = []
                    
                    
                    for i, frame in enumerate(frames):
                        try:
                            # Extract face attributes for gaze and emotion
                            attrs = self._extract_face_attributes(
                                frame.numpy().transpose(1, 2, 0),
                                video_path
                            )
                            
                            if attrs is None:
                                self.tracker.dispatch(VideoEventData(
                                    video_path=video_path,
                                    event_type=VideoEvent.LANDMARK_DETECTION_FAILED,
                                    details={"error": "Failed to extract valid landmarks"}
                                ))
                                return None
                            # Extract facial landmarks
                            landmarks = self._extract_face_landmarks(
                                frame.numpy().transpose(1, 2, 0),
                                video_path
                            )
                            
                            if attrs and landmarks:
                                # Store basic attributes
                                gaze_angles.append(attrs['gaze'])
                                emotion_logits.append(attrs['emotion'])
                                emotion_labels.append(attrs.get('emotion_label', 'neutral'))  # Store label
                                distances.append(attrs['head_distance'])
                                
                                # Store landmarks in correct order
                                lips_landmarks.append(landmarks['lips'])
                                right_eye_landmarks.append(landmarks['right_eye'])
                                left_eye_landmarks.append(landmarks['left_eye'])
                                jaw_landmarks.append(landmarks['jaw'])
                                nose_landmarks.append(landmarks['nose'])
                                blink_states.append(landmarks['blink_state'])
                                
                                # Compute speed bucket
                                if i == 0:
                                    speed_buckets.append(4)  # Middle bucket for first frame
                                    prev_landmarks = attrs['landmarks']
                                else:
                                    speed = self._compute_rotation_speed(attrs['landmarks'], prev_landmarks)
                                    bucket_idx = self.speed_encoder.encode_speed(speed)
                                    speed_buckets.append(bucket_idx)
                                    prev_landmarks = attrs['landmarks']
                                    
                            else:
                                # Add zero-filled arrays for missing data
                                gaze_angles.append(np.zeros(2, dtype=np.float32))
                                emotion_logits.append(np.zeros(2, dtype=np.float32))
                                emotion_labels.append('neutral')  # Default label
                                distances.append(np.array([0.5], dtype=np.float32))
                                speed_buckets.append(4)  # Middle bucket
                                
                                # Add zero-filled landmarks
                                lips_landmarks.append(np.zeros((20, 3), dtype=np.float32))
                                right_eye_landmarks.append(np.zeros((8, 3), dtype=np.float32))
                                left_eye_landmarks.append(np.zeros((7, 3), dtype=np.float32))
                                jaw_landmarks.append(np.zeros((10, 3), dtype=np.float32))
                                nose_landmarks.append(np.zeros((4, 3), dtype=np.float32))
                                blink_states.append(np.array([0, 1.0, 1.0], dtype=np.float32))  # Default open state
                                
                        except Exception as e:
                            logger.error(f"Error processing frame {i}: {str(e)}")
                            import traceback
                            logger.debug(f"Traceback: {traceback.format_exc()}")
                            continue

                    lip_motion_sequence = self._get_lip_motion_sequence(lips_landmarks)
                    if lip_motion_sequence is None:
                        logger.warning(f"Skipping window - no valid lip motion")
                        return self._get_zero_sample()

                    # Compute lip metrics using LipStateAnalyzer
                    lip_metrics = self.lip_analyzer.analyze_sequence(np.array(lips_landmarks))

                    # Create window data with correct key names
                    # Squeeze batch dimension from EMO features (they come as [1, T, ...])
                    # NOTE: Frames will be saved to disk cache and excluded from H5 cache

                    # Extract motion parameters (needed for velocity computation)
                    theta_gt = emo_features['theta'].squeeze(0)  # [1, T, 3, 4] -> [T, 3, 4]
                    expression_gt = emo_features['expression_embed'].squeeze(0)  # [1, T, 128] -> [T, 128]

                    window_data = {
                        'frames': torch.stack(frames),  # Include frames so they can be saved to disk cache
                        'theta': theta_gt,  # [T, 3, 4]
                        'scale': emo_features['scale'].squeeze(0),  # [1, T, 3] -> [T, 3]
                        'rotation': emo_features['rotation'].squeeze(0),  # [1, T, 3] -> [T, 3]
                        'translation': emo_features['translation'].squeeze(0),  # [1, T, 3] -> [T, 3]
                        'expression_embed': expression_gt,  # [T, 128]
                        'audio_features': wav2vec_features.squeeze(0) if wav2vec_features.ndim == 3 else wav2vec_features,  # [1, T, 768] -> [T, 768]
                        'audio_mfcc': mfcc_features.squeeze(0) if mfcc_features.ndim == 3 else mfcc_features,  # [1, T, 13] -> [T, 13]
                        'audio_waveform': audio_segment.squeeze(0),  # [1, samples] -> [samples] - raw audio for legacy
                        'audio_mel_spec': mel_spec.squeeze(0),  # [1, T, 128] -> [T, 128] - mel spectrogram for Synchformer
                        'phoneme_gt': phoneme_gt,  # [num_queries=8] - phoneme IDs for self-supervised phoneme prediction
                        'gaze': torch.tensor(np.stack(gaze_angles), dtype=torch.float32),
                        'emotion': torch.tensor(np.stack(emotion_logits), dtype=torch.float32),
                        'emotion_label': emotion_labels,  # List of strings like ["sad", "happy", "neutral", ...]
                        'head_distance': torch.tensor(np.stack(distances), dtype=torch.float32),
                        'speed_bucket': torch.tensor(speed_buckets, dtype=torch.long).unsqueeze(-1),

                        # Store landmarks with correct keys
                        'lips': torch.tensor(np.stack(lips_landmarks), dtype=torch.float32),
                        'right_eye': torch.tensor(np.stack(right_eye_landmarks), dtype=torch.float32),
                        'left_eye': torch.tensor(np.stack(left_eye_landmarks), dtype=torch.float32),
                        'jaw': torch.tensor(np.stack(jaw_landmarks), dtype=torch.float32),
                        'nose': torch.tensor(np.stack(nose_landmarks), dtype=torch.float32),

                        'lip_motion': torch.tensor(lip_motion_sequence, dtype=torch.float32),

                        'blink_state': torch.tensor(np.stack(blink_states), dtype=torch.float32),

                        # Add UV warps from calculate_target_warps (matching H5 cache)
                        # Note: warps have shape [1, T, ...] so we squeeze the batch dimension
                        'uv_warps': emo_features.get('uv_warps', torch.zeros(1, self.sequence_length, 16, 64, 64, 3)).squeeze(0),
                        'target_masks': emo_features.get('target_masks', torch.zeros(1, self.sequence_length, 1, 512, 512)).squeeze(0),

                        # Add identity info for reconstruction
                        'identity_info': emo_features.get('identity_info', {}),

                        # Add lip metrics for audio-lip correlation loss
                        'lip_metrics': {
                            'openness': torch.tensor(lip_metrics['openness'], dtype=torch.float32),
                            'symmetry': torch.tensor(lip_metrics['symmetry'], dtype=torch.float32),
                            'aspect_ratio': torch.tensor(lip_metrics['aspect_ratio'], dtype=torch.float32),
                            'area': torch.tensor(lip_metrics['area'], dtype=torch.float32),
                            'perimeter': torch.tensor(lip_metrics['perimeter'], dtype=torch.float32),
                        },

                        'metadata': {
                            'video_path': str(video_path),
                            'start_frame': window['start_frame'],
                            'fps': window['fps'],
                            'has_context': window['has_context'],
                            'window_idx': window['window_idx']
                        }
                    }

                   
                    # Verify shapes
                    expected_shapes = {
                        'lips': (self.sequence_length, 20, 3),
                        'right_eye': (self.sequence_length, 8, 3),
                        'left_eye': (self.sequence_length, 7, 3),
                        'jaw': (self.sequence_length, 10, 3),
                        'nose': (self.sequence_length, 4, 3),
                        'blink_state': (self.sequence_length, 3)
                    }
                    
                    # Validate and correct shapes if needed
                    for key, expected_shape in expected_shapes.items():
                        current_shape = window_data[key].shape
                        if current_shape != expected_shape:
                            logger.warning(f"Shape mismatch for {key}: expected {expected_shape}, got {current_shape}")
                            # Create correct shape with zeros
                            window_data[key] = torch.zeros(expected_shape, dtype=torch.float32)
                    
                    # Save to cache before returning
                    if self.cache_type == 'chunked':
                        # For chunked cache (WindowCache), use save_chunk
                        chunk_idx = window['start_frame'] // self.cache.chunk_size
                        self.cache.save_chunk(
                            video_path=video_path,
                            chunk_idx=chunk_idx,
                            chunk_data={f"window_{window['window_idx']}": window_data},
                            start_frame=window['start_frame'],
                            end_frame=window['start_frame'] + self.sequence_length
                        )
                        logger.info(f"Saved window {window['window_idx']} to chunk {chunk_idx} for {Path(video_path).name}")
                    elif self.cache_type == 'single_bucket':
                        # For single-bucket cache, append the window
                        # Store the window with index for later batch saving
                        if not hasattr(self, '_pending_windows'):
                            self._pending_windows = []
                        self._pending_windows.append((idx, window_data))

                        # Batch save periodically to avoid memory issues
                        if len(self._pending_windows) >= self.max_batch_size:
                            self._save_pending_windows()

                    elif self.cache_type == 'built_in':
                        # For built-in cache, use the original save method
                        self._save_window_to_cache(video_path, window['window_idx'], window_data)

                    # Generate EMO frames if enabled AND not already in window_data (from cache)
                    if self.generate_emo_frames and self.va_bridge is not None and self.emo_identity_image is not None:
                        if 'emo_frames' in window_data:
                            logger.debug(f"✅ Using cached EMO frames for window {idx} ({len(window_data['emo_frames'])} keyframes)")
                        else:
                            try:
                                with torch.no_grad():
                                    # Clear VA bridge cache to ensure fresh embeddings for EMO identity
                                    if hasattr(self.va_bridge, 'clear_cache'):
                                        self.va_bridge.clear_cache()

                                    # Select keyframe indices
                                    T = window_data['theta'].shape[0]
                                    keyframe_indices = np.linspace(0, T-1, self.emo_keyframes_per_window, dtype=int)

                                    emo_frames = []
                                    for frame_idx in keyframe_indices:
                                        # Extract motion for this frame and ensure all on same device
                                        frame_motion = {
                                            'theta': window_data['theta'][frame_idx:frame_idx+1].unsqueeze(0).to(self.device),  # [1, 1, 3, 4]
                                            'expression_embed': window_data['expression_embed'][frame_idx:frame_idx+1].unsqueeze(0).to(self.device),  # [1, 1, 128]
                                            'uv_warps': window_data['uv_warps'][frame_idx:frame_idx+1].unsqueeze(0).to(self.device) if 'uv_warps' in window_data else None,  # [1, 1, 16, 64, 64, 3]
                                        }

                                        # Check if we have uv_warps (required for EMO generation)
                                        if frame_motion['uv_warps'] is None:
                                            if frame_idx == 0:  # Log once per window
                                                logger.error(f"❌ Window {idx}: uv_warps is None - EMO frames will be black!")
                                                logger.error(f"   'uv_warps' in window_data: {'uv_warps' in window_data}")
                                                if 'uv_warps' in window_data:
                                                    logger.error(f"   uv_warps shape: {window_data['uv_warps'].shape}")
                                                    logger.error(f"   uv_warps range: [{window_data['uv_warps'].min():.6f}, {window_data['uv_warps'].max():.6f}]")
                                            emo_frames.append(torch.zeros(3, 512, 512, device=self.device))
                                            continue

                                        # Generate EMO frame using va_bridge
                                        # Output will be [1, 1, C, H, W]
                                        emo_identity = self.emo_identity_image.to(self.device)

                                        # DEBUG: Save identity image once to verify it's correct
                                        if frame_idx == 0 and idx % 100 == 0:
                                            import torchvision
                                            torchvision.utils.save_image(emo_identity[0], f'debug_emo_identity_window_{idx}.png')
                                            logger.info(f"Saved debug EMO identity image for window {idx}")

                                        emo_output, _ = self.va_bridge.generate_frames_from_motion(
                                            motion_outputs=frame_motion,
                                            source_img=emo_identity,
                                            use_black_background=True  # Use black background to ensure clean EMO render
                                        )

                                        if emo_output is not None:
                                            # Extract the single frame [1, 1, C, H, W] -> [C, H, W]
                                            frame = emo_output[0, 0]
                                            emo_frames.append(frame)
                                        else:
                                            # Add blank frame if generation failed
                                            emo_frames.append(torch.zeros(3, 512, 512, device=self.device))

                                    # Stack EMO frames [num_keyframes, C, H, W]
                                    if len(emo_frames) == 0:
                                        raise RuntimeError(f"EMO generation is enabled but produced no frames for window {idx}")

                                    window_data['emo_frames'] = torch.stack(emo_frames, dim=0)
                                    window_data['emo_keyframe_indices'] = torch.tensor(keyframe_indices, dtype=torch.long)
                                    logger.info(f"✅ Generated {len(emo_frames)} EMO frames for window {idx}")

                                    # QUALITY CHECK: Detect bad UV warps immediately after generation
                                    if 'uv_warps' in window_data:
                                        uv_magnitude = window_data['uv_warps'].abs().mean().item()
                                        uv_std = window_data['uv_warps'].std().item()

                                        # Check for collapsed/bad UV warps
                                        if uv_magnitude < 0.15 or uv_std < 0.01:
                                            logger.error(f"❌ BAD UV WARPS detected for window {idx} from video {video_path}")
                                            logger.error(f"   UV magnitude: {uv_magnitude:.6f} (threshold: 0.15)")
                                            logger.error(f"   UV std: {uv_std:.6f} (threshold: 0.01)")

                                            # Dispatch event to mark video as bad
                                            self.tracker.dispatch(VideoEventData(
                                                video_path=video_path,
                                                event_type=VideoEvent.BAD_UV_WARPS,
                                                details={
                                                    "window_idx": idx,
                                                    "uv_magnitude": uv_magnitude,
                                                    "uv_std": uv_std,
                                                    "reason": f"UV warps collapsed (magnitude={uv_magnitude:.4f}, std={uv_std:.6f})"
                                                }
                                            ))

                                            # Return zero sample to skip this window
                                            logger.warning(f"⚠️ Returning zero sample for window {idx} due to bad UV warps")
                                            return self._get_zero_sample()

                            except Exception as e:
                                logger.error(f"❌ FAILED to generate EMO frames for window {idx}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                # EMO frames are REQUIRED - re-raise the exception
                                raise RuntimeError(f"EMO frame generation is mandatory but failed: {e}") from e

                    # Return the single window data directly
                    return window_data

            except Exception as e:
                logger.error(f"Error processing window: {str(e)}")
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.PROCESSING_ERROR,
                    details={"error": f"Face attribute error: {str(e)}"}
                ))
                import traceback
                logger.error(traceback.format_exc())
                return self._get_zero_sample()

        except Exception as e:
            logger.error(f"Error in __getitem__: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_zero_sample()

    def _save_pending_windows(self):
        """Save any pending windows to SingleBucketCache."""
        if not hasattr(self, '_pending_windows') or not self._pending_windows:
            return

        try:
            # Sort windows by index
            self._pending_windows.sort(key=lambda x: x[0])

            # Prepare windows list for appending
            windows_to_save = [window_data for _, window_data in self._pending_windows]

            # Append to cache
            self.cache.append_windows(windows_to_save)
            logger.info(f"Saved {len(self._pending_windows)} windows to SingleBucketCache")

            # Clear pending windows and force garbage collection
            self._pending_windows = []
            del windows_to_save

            # Force memory cleanup
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Error saving pending windows: {str(e)}")

    def _compute_velocity(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute ground-truth velocity flows from motion parameters for Flow-DPO.
        Concatenates frame differences in theta and expression.

        Args:
            motion: Dict with 'theta' [T, 3, 4] and 'expression_embed' [T, 128]

        Returns:
            velocity: [T, flow_dim] where flow_dim = 12 + 128 = 140
        """
        theta = motion['theta']  # [T, 3, 4]
        expr = motion['expression_embed']  # [T, 128]

        # Velocity: frame differences
        theta_vel = theta[1:] - theta[:-1]  # [T-1, 3, 4]
        expr_vel = expr[1:] - expr[:-1]  # [T-1, 128]

        # Pad to T with zeros at the beginning (first frame has zero velocity)
        theta_vel = torch.cat([torch.zeros(1, 3, 4, dtype=theta.dtype, device=theta.device), theta_vel], dim=0)  # [T, 3, 4]
        expr_vel = torch.cat([torch.zeros(1, 128, dtype=expr.dtype, device=expr.device), expr_vel], dim=0)  # [T, 128]

        # Flatten theta and concatenate
        theta_flat = theta_vel.reshape(theta_vel.shape[0], -1)  # [T, 12]
        velocity = torch.cat([theta_flat, expr_vel], dim=-1)  # [T, 140]

        return velocity.float()

    def _generate_dispreferred_motion(self, motion: Dict[str, torch.Tensor], noise_level: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Generate dispreferred motion samples by adding controlled noise to theta and expression.
        Used for Flow-DPO preference learning.

        Args:
            motion: Dict with 'theta' and 'expression_embed'
            noise_level: Standard deviation of Gaussian noise (default 0.1 = 10% of signal)

        Returns:
            dispreferred_motion: Dict with noisy theta and expression_embed
        """
        dispreferred = {}

        # Add noise to theta [T, 3, 4]
        theta_noise = torch.randn_like(motion['theta']) * noise_level
        dispreferred['theta'] = motion['theta'] + theta_noise

        # Add noise to expression [T, 128]
        expr_noise = torch.randn_like(motion['expression_embed']) * noise_level
        dispreferred['expression_embed'] = motion['expression_embed'] + expr_noise

        return dispreferred

    def _get_zero_sample(self) -> Dict[str, torch.Tensor]:
        """Return a zero-filled sample with all required features including landmarks and lip motion"""
        return {
            'frames': torch.zeros((self.sequence_length, 3, *self.frame_size)),
            'theta': torch.zeros((self.sequence_length, 3, 4)),
            'scale': torch.zeros((self.sequence_length, 3)),  
            'rotation': torch.zeros((self.sequence_length, 3)),
            'translation': torch.zeros((self.sequence_length, 3)),
            'audio_features': torch.zeros((self.sequence_length, 768)),   # wav2vec - no batch dim
            'audio_mfcc': torch.zeros((self.sequence_length, 13)),       # mfcc for syncnet - no batch dim
            'audio_waveform': torch.zeros(self.sequence_length * 640),    # raw audio at 16kHz, ~40ms per frame
            'audio_mel_spec': torch.zeros((self.sequence_length, 128)),   # mel spectrogram for Synchformer
            'phoneme_gt': torch.zeros(8, dtype=torch.long),               # phoneme IDs for self-supervised phoneme prediction
            'gaze': torch.zeros((self.sequence_length, 2)),
            'head_distance': torch.zeros((self.sequence_length, 1)),
            'emotion': torch.zeros((self.sequence_length, 2)),
            'speed_bucket': torch.zeros((self.sequence_length, 1), dtype=torch.long),
            'expression_embed': torch.zeros((self.sequence_length, 128)),  # No batch dim
            
            # Facial landmarks
            'lips': torch.zeros((self.sequence_length, 20, 3)),
            'right_eye': torch.zeros((self.sequence_length, 8, 3)),
            'left_eye': torch.zeros((self.sequence_length, 7, 3)),
            'jaw': torch.zeros((self.sequence_length, 10, 3)),
            'nose': torch.zeros((self.sequence_length, 4, 3)),
            
            # Lip motion sequence - matches lips shape but represents motion vectors
            'lip_motion': torch.zeros((self.sequence_length, 20, 3)),
            
            # Blink state: [phase, left_openness, right_openness]
            'blink_state': torch.zeros((self.sequence_length, 3)),

            # Per-frame warps
            'xy_warps': torch.zeros((self.sequence_length, 16, 64, 64, 3)),
            'rigid_warps': torch.zeros((self.sequence_length, 16, 64, 64, 3)),
            'uv_warps': torch.zeros((self.sequence_length, 16, 64, 64, 3)),
            'source_theta_warp': torch.zeros((self.sequence_length, 3, 4)),

            # Lip metrics for audio-lip correlation
            'lip_metrics': {
                'openness': torch.zeros(self.sequence_length),
                'symmetry': torch.ones(self.sequence_length),  # Default to perfect symmetry
                'aspect_ratio': torch.ones(self.sequence_length),  # Default to 1.0
                'area': torch.zeros(self.sequence_length),
                'perimeter': torch.zeros(self.sequence_length),
            },

            'metadata': {
                'video_path': '',
                'video_name': '',
                'start_frame': 0,
                'fps': 0,
                'has_context': False
            }
        }
    
    
    

                
    def _compute_gaze(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute gaze direction from facial landmarks
        Returns (θ,φ) in radians
        """
        try:
            # Get eye landmarks
            left_eye = landmarks[36:42].mean(axis=0)   # Left eye center
            right_eye = landmarks[42:48].mean(axis=0)  # Right eye center
            eye_center = (left_eye + right_eye) / 2
            
            # Get nose tip and other reference points
            nose_tip = landmarks[30]
            nose_bridge = landmarks[27]
            
            # Compute direction vectors
            forward = nose_tip - nose_bridge
            gaze = nose_tip - eye_center
            
            # Normalize vectors
            forward = forward / np.linalg.norm(forward)
            gaze = gaze / np.linalg.norm(gaze)
            
            # Calculate angles
            theta = np.arctan2(gaze[0], gaze[2])  # Yaw
            phi = np.arctan2(gaze[1], gaze[2])    # Pitch
            
            return np.array([theta, phi])
            
        except Exception as e:
            logger.error(f"🔥 Error computing gaze: {str(e)}")
            return np.array([0.0, 0.0])

    def compute_emotion_offset(self, emotion_va: torch.Tensor) -> torch.Tensor:
        """
        Compute emotion offset from sequence of VA values
        Args:
            emotion_va: Tensor of shape (sequence_length, 2) containing per-frame VA values
        Returns:
            Tensor of shape (2,) containing averaged VA coefficients
        """
        # Simply average the VA values over the sequence
        avg_emotion = torch.mean(emotion_va, dim=0)
        
        # Ensure values are in [-1, 1] range
        avg_emotion = torch.clamp(avg_emotion, -1.0, 1.0)
        
        return avg_emotion
        
    def _compute_distance(
        self, 
        landmarks: np.ndarray, 
        bbox: np.ndarray, 
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Compute normalized head distance using facial landmarks
        """
        try:
            # Use outer eye corners and nose tip for stable distance estimate
            left_corner = landmarks[36]   # Left eye outer corner
            right_corner = landmarks[45]  # Right eye outer corner
            nose_tip = landmarks[30]      # Nose tip
            
            # Compute face size metrics
            eye_distance = np.linalg.norm(right_corner - left_corner)
            nose_height = np.linalg.norm(nose_tip - (left_corner + right_corner) / 2)
            
            # Compute face area relative to frame
            face_size = eye_distance * nose_height
            frame_area = frame_shape[0] * frame_shape[1]
            
            # Normalize
            normalized_distance = np.clip(face_size / frame_area, 0, 1)
            
            return normalized_distance
            
        except Exception as e:
            logger.error(f"🔥 Error computing distance: {str(e)}")
            return 0.5  # Return middle distance on error

    def _compute_face_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute face rotation angles from landmarks
        Returns [yaw, pitch, roll] in radians
        """
        try:
            # Get key landmarks
            nose_bridge = landmarks[27]
            nose_tip = landmarks[30]
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
            
            # Compute face normal
            face_normal = np.cross(right_eye - left_eye, nose_tip - nose_bridge)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Compute angles
            yaw = np.arctan2(face_normal[0], face_normal[2])
            pitch = np.arctan2(-face_normal[1], np.sqrt(face_normal[0]**2 + face_normal[2]**2))
            
            # Compute roll using mouth corners
            mouth_vector = right_mouth - left_mouth
            roll = np.arctan2(mouth_vector[1], mouth_vector[0])
            
            return np.array([yaw, pitch, roll])
            
        except Exception as e:
            logger.error(f"🔥 Error computing face angles: {str(e)}")
            return np.array([0.0, 0.0, 0.0])

    def _normalize_landmarks(
        self, 
        landmarks: np.ndarray, 
        bbox: np.ndarray
    ) -> np.ndarray:
        """
        Normalize landmarks to [-1, 1] range relative to face bbox
        """
        try:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Center and scale landmarks
            norm_landmarks = landmarks.copy()
            norm_landmarks[:, 0] = (norm_landmarks[:, 0] - x1) / w * 2 - 1
            norm_landmarks[:, 1] = (norm_landmarks[:, 1] - y1) / h * 2 - 1
            
            return norm_landmarks
            
        except Exception as e:
            logger.error(f"🔥 Error normalizing landmarks: {str(e)}")
            return landmarks
        


class VideoValidation:
    """Handles video validation for both audio and face landmark detection."""
    def __init__(
        self, 
        video_folder: Path,
        min_valid_faces: float = 0.8,  # Minimum ratio of frames that must have valid faces
        min_valid_landmarks: float = 0.8,  # Minimum ratio of frames that must have valid landmarks
        cache_audio: bool = True,
        num_sample_frames: int = 10  # Number of frames to sample for validation
    ):
        self.video_folder = Path(video_folder)
        self.min_valid_faces = min_valid_faces
        self.min_valid_landmarks = min_valid_landmarks
        self.cache_audio = cache_audio
        self.num_sample_frames = num_sample_frames
        
        # Set up cache directories
        self.cache_dir = self.video_folder / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.audio_cache_dir = self.cache_dir / "audio"
        self.audio_cache_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.status_file = self.cache_dir / "video_status.json"

    def validate_videos(self, video_paths: List[str], face_mesh) -> Dict[str, Dict]:
        """Run validation on videos checking both audio and face landmarks."""
        status = {}
        
        # Load cached status if available
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                logger.info(f"Loaded cached status for {len(status)} videos")
            
            # Filter out videos that need revalidation
            videos_to_check = []
            for video_path in video_paths:
                if video_path not in status:
                    videos_to_check.append(video_path)
                    continue
                    
                video_status = status[video_path]
                needs_check = (
                    not video_status.get('is_valid', False) or
                    not video_status.get('audio_checked', False) or
                    not video_status.get('face_checked', False) or
                    not video_status.get('landmark_checked', False)  # Add landmark check
                )
                if needs_check:
                    videos_to_check.append(video_path)
                    
            logger.info(f"Found {len(videos_to_check)} videos needing validation")
        else:
            videos_to_check = video_paths
            logger.info(f"Validating all {len(videos_to_check)} videos")
            
        # Process videos needing validation
        for video_path in tqdm(videos_to_check, desc="Validating videos"):
            try:
                # Check audio first
                audio_status = self._check_audio_status(video_path)
                
                # Only check faces and landmarks if audio is valid
                if audio_status['has_audio']:
                    # Check face detection
                    face_status = self._check_faces(video_path, face_mesh)
                    
                    # Only check landmarks if face detection passes
                    if face_status['is_valid']:
                        landmark_status = self._check_landmarks(video_path, face_mesh)
                    else:
                        landmark_status = {
                            'is_valid': False,
                            'valid_ratio': 0.0,
                            'error': 'Face detection failed'
                        }
                else:
                    face_status = {
                        'is_valid': False,
                        'valid_ratio': 0.0,
                        'error': 'No audio available'
                    }
                    landmark_status = {
                        'is_valid': False,
                        'valid_ratio': 0.0,
                        'error': 'No audio available'
                    }
                
                # Combine status
                status[video_path] = {
                    'is_valid': (
                        audio_status['has_audio'] and 
                        face_status['is_valid'] and 
                        landmark_status['is_valid']
                    ),
                    'audio_status': audio_status,
                    'face_status': face_status,
                    'landmark_status': landmark_status,
                    'audio_checked': True,
                    'face_checked': True,
                    'landmark_checked': True,
                    'last_checked': str(datetime.now())
                }
                
            except Exception as e:
                logger.error(f"Error validating {video_path}: {str(e)}")
                status[video_path] = {
                    'is_valid': False,
                    'error': str(e),
                    'last_checked': str(datetime.now())
                }
                
        # Save updated status
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
            
        # Log validation results
        valid_videos = [v for v in status.values() if v.get('is_valid', False)]
        logger.info(f"\nValidation Results:")
        logger.info(f"  Total videos: {len(status)}")
        logger.info(f"  Valid videos: {len(valid_videos)}")
        logger.info(f"  Invalid videos: {len(status) - len(valid_videos)}")
        
        return status

    def _check_landmarks(
        self, 
        video_path: str, 
        face_mesh,
        required_landmarks: List[str] = ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']
    ) -> Dict:
        """Check if facial landmarks can be consistently detected."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'is_valid': False, 'error': 'Failed to open video'}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.num_sample_frames:
                return {'is_valid': False, 'error': 'Video too short'}
                
            # Sample frames evenly
            sample_indices = np.linspace(0, total_frames-1, self.num_sample_frames, dtype=int)
            valid_landmarks = 0
            landmark_stats = defaultdict(int)
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Extract facial landmarks using MediaPipe
                    results = face_mesh.process(frame)
                    if not results.multi_face_landmarks:
                        continue

                    # Get first face landmarks
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Convert landmarks to numpy array
                    height, width = frame.shape[:2]
                    landmarks = np.array([
                        [lm.x * width, lm.y * height, lm.z * width]
                        for lm in face_landmarks.landmark
                    ], dtype=np.float32)

                    # Check if we can extract all required landmark groups
                    all_landmarks_valid = True
                    for group in required_landmarks:
                        if group == 'lips':
                            points = landmarks[list(range(61, 69)) + list(range(48, 60))]
                        elif group == 'right_eye':
                            points = landmarks[[33, 133, 157, 158, 159, 160, 161, 173]]
                        elif group == 'left_eye':
                            points = landmarks[[362, 385, 386, 387, 388, 466, 263]]
                        elif group == 'jaw':
                            points = landmarks[[61, 185, 40, 39, 37, 0, 267, 269, 270, 409]]
                        elif group == 'nose':
                            points = landmarks[[1, 2, 98, 327]]
                            
                        if len(points) == 0:
                            all_landmarks_valid = False
                            landmark_stats[group] += 1
                            break

                    if all_landmarks_valid:
                        valid_landmarks += 1
                        
                except Exception as e:
                    logger.debug(f"Error extracting landmarks from frame {frame_idx}: {str(e)}")
                    continue
                    
            cap.release()
            
            # Calculate success ratio
            valid_ratio = valid_landmarks / self.num_sample_frames
            is_valid = valid_ratio >= self.min_valid_landmarks
            
            return {
                'is_valid': is_valid,
                'valid_ratio': valid_ratio,
                'samples_checked': self.num_sample_frames,
                'failed_landmarks': dict(landmark_stats),
                'last_checked': str(datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error in landmark detection: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'valid_ratio': 0.0
            }

    def _check_audio_status(self, video_paths: List[str]) -> Dict[str, Dict]:
        """Check audio status for all videos."""
        status = {}
        
        for video_path in tqdm(video_paths, desc="Checking audio"):
            try:
                audio_path = self._get_audio_path(video_path)
                has_cache = audio_path.exists()
                command = [
                    'ffprobe',
                    '-loglevel', 'error',
                    '-show_streams',
                    '-select_streams', 'a',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'json',
                    video_path
                ]
                
                result = subprocess.run(command, capture_output=True, text=True)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    has_audio = bool(data.get('streams', []))
                    
                    if not has_audio:
                        self.tracker.dispatch(VideoEventData(
                            video_path=video_path,
                            event_type=VideoEvent.NO_AUDIO,
                            details={"error": "No audio stream found"}
                        ))
                else:
                    has_audio = False
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.PROCESSING_ERROR,
                        details={"error": "FFprobe error checking audio"}
                    ))
                    
                status[video_path] = {
                    'has_audio': has_audio,
                    'has_cache': has_cache,
                    'last_checked': str(datetime.now())
                }
                
            except Exception as e:
                self.tracker.dispatch(VideoEventData(
                    video_path=video_path,
                    event_type=VideoEvent.PROCESSING_ERROR,
                    details={"error": f"Audio check error: {str(e)}"}
                ))
                status[video_path] = {'has_audio': False, 'has_cache': False}
                
        return status
    def _check_faces(
        self, 
        video_path: str, 
        face_mesh,
    ) -> Dict:
        """Check face detection on sampled frames."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'is_valid': False, 'error': 'Failed to open video'}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.num_sample_frames:
                return {'is_valid': False, 'error': 'Video too short'}
                
            # Sample frames evenly
            sample_indices = np.linspace(0, total_frames-1, self.num_sample_frames, dtype=int)
            valid_faces = 0
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                
                if results.multi_face_landmarks:
                    valid_faces += 1
                    
            cap.release()
            
            # Calculate success ratio
            valid_ratio = valid_faces / self.num_sample_frames
            is_valid = valid_ratio >= self.min_valid_faces
            
            return {
                'is_valid': is_valid,
                'valid_ratio': valid_ratio,
                'samples_checked': self.num_sample_frames,
                'last_checked': str(datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'valid_ratio': 0.0
            }
            
    def _get_audio_path(self, video_path: str) -> Path:
        """Get path for cached audio file."""
        video_path = Path(video_path)
        if self.cache_audio:
            relative_path = video_path.relative_to(self.video_folder)
            audio_path = self.audio_cache_dir / relative_path.with_suffix('.wav')
        else:
            audio_path = video_path.with_suffix('.wav')
        return audio_path
        
    def _extract_audio(self, video_path: str, audio_path: Path):
        """Extract audio from video using ffmpeg."""
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ]
            
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
            if not audio_path.exists():
                raise RuntimeError("FFmpeg completed but audio file not created")
                
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

        
    def validate_videos(self, video_paths: List[str], face_mesh) -> Dict[str, Dict]:
        """Run validation on videos and return combined status."""
        status = {}
        
        # Load cached status if available
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                logger.info(f"Loaded cached status for {len(status)} videos")
            
            # Filter out videos that need revalidation
            videos_to_check = []
            for video_path in video_paths:
                if video_path not in status:
                    videos_to_check.append(video_path)
                    continue
                    
                video_status = status[video_path]
                needs_check = (
                    not video_status.get('is_valid', False) or
                    not video_status.get('audio_checked', False) or
                    not video_status.get('face_checked', False)
                )
                if needs_check:
                    videos_to_check.append(video_path)
                    
            logger.info(f"Found {len(videos_to_check)} videos needing validation")
        else:
            videos_to_check = video_paths
            logger.info(f"Validating all {len(videos_to_check)} videos")
            
        # Process videos needing validation
        for video_path in tqdm(videos_to_check, desc="Validating videos"):
            try:
                # Check audio first
                audio_status = self._check_audio(video_path)
                
                # Only check faces if audio is valid
                if audio_status['has_audio']:
                    face_status = self._check_faces(video_path, face_mesh)
                else:
                    face_status = {
                        'is_valid': False,
                        'valid_ratio': 0.0,
                        'error': 'No audio available'
                    }
                
                # Combine status
                status[video_path] = {
                    'is_valid': audio_status['has_audio'] and face_status['is_valid'],
                    'audio_status': audio_status,
                    'face_status': face_status,
                    'audio_checked': True,
                    'face_checked': True,
                    'last_checked': str(datetime.now())
                }
                
            except Exception as e:
                logger.error(f"Error validating {video_path}: {str(e)}")
                status[video_path] = {
                    'is_valid': False,
                    'error': str(e),
                    'last_checked': str(datetime.now())
                }
                
        # Save updated status
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
            
        # Log validation results
        valid_videos = [v for v in status.values() if v.get('is_valid', False)]
        logger.info(f"\nValidation Results:")
        logger.info(f"  Total videos: {len(status)}")
        logger.info(f"  Valid videos: {len(valid_videos)}")
        logger.info(f"  Invalid videos: {len(status) - len(valid_videos)}")
        
        return status
        
    def _check_audio(self, video_path: str) -> Dict:
        """Check if video has valid audio."""
        try:
            # Check audio cache first
            audio_path = self._get_audio_path(video_path)
            if audio_path.exists():
                return {
                    'has_audio': True,
                    'has_cache': True,
                    'last_checked': str(datetime.now())
                }
                
            # Check for audio stream using ffprobe
            command = [
                'ffprobe',
                '-loglevel', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                video_path
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                has_audio = bool(data.get('streams', []))
                
                # Extract audio if found
                if has_audio and self.cache_audio:
                    self._extract_audio(video_path, audio_path)
                    
                return {
                    'has_audio': has_audio,
                    'has_cache': audio_path.exists(),
                    'last_checked': str(datetime.now())
                }
            else:
                return {
                    'has_audio': False,
                    'error': 'ffprobe error',
                    'last_checked': str(datetime.now())
                }
                
        except Exception as e:
            logger.error(f"Error checking audio: {str(e)}")
            return {
                'has_audio': False,
                'error': str(e),
                'last_checked': str(datetime.now())
            }
            
    def _check_faces(
        self, 
        video_path: str, 
        face_mesh,
        num_samples: int = 10
    ) -> Dict:
        """Check face detection on sampled frames."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'is_valid': False, 'error': 'Failed to open video'}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < num_samples:
                num_samples = total_frames
                
            # Sample frames evenly
            sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
            valid_faces = 0
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                
                if results.multi_face_landmarks:
                    valid_faces += 1
                    
            cap.release()
            
            # Calculate success ratio
            valid_ratio = valid_faces / num_samples
            is_valid = valid_ratio >= self.min_valid_faces
            
            return {
                'is_valid': is_valid,
                'valid_ratio': valid_ratio,
                'samples_checked': num_samples,
                'last_checked': str(datetime.now())
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'valid_ratio': 0.0
            }
            
    def _get_audio_path(self, video_path: str) -> Path:
        """Get path for cached audio file."""
        video_path = Path(video_path)
        if self.cache_audio:
            relative_path = video_path.relative_to(self.video_folder)
            audio_path = self.audio_cache_dir / relative_path.with_suffix('.wav')
        else:
            audio_path = video_path.with_suffix('.wav')
        return audio_path
        
    def _extract_audio(self, video_path: str, audio_path: Path):
        """Extract audio from video using ffmpeg."""
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ]
            
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

class LipStateAnalyzer:
    """Analyzes lip state using geometric features"""
    def __init__(self):
        # Define thresholds
        self.openness_thresholds = {
            'closed': 0.05,
            'slightly_open': 0.1,
            'open': 0.2,
            'wide_open': float('inf')
        }

    def analyze_lip_metrics(self, lips: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive lip metrics for a single frame
        Args:
            lips: Lip landmarks array of shape [N, 3]
        Returns:
            Dictionary of lip metrics
        """
        try:
            # 1. Calculate openness
            upper_middle = np.mean(lips[:len(lips)//2], axis=0)
            lower_middle = np.mean(lips[len(lips)//2:], axis=0)
            openness = np.abs(upper_middle[1] - lower_middle[1])

            # 2. Calculate symmetry
            left_side = lips[:len(lips)//2]
            right_side = np.flip(lips[len(lips)//2:], axis=0)
            symmetry = 1.0 - np.mean(np.linalg.norm(left_side - right_side, axis=1))

            # 3. Calculate aspect ratio (width/height)
            width = np.linalg.norm(lips[0] - lips[-1])
            height = openness + 1e-6  # Avoid division by zero
            aspect_ratio = width / height

            # 4. Calculate area using shoelace formula
            x = lips[:, 0]
            y = lips[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            # 5. Calculate perimeter
            perimeter = np.sum(np.linalg.norm(lips[1:] - lips[:-1], axis=1))

            # Determine mouth state
            state = 'closed'
            for state_name, threshold in self.openness_thresholds.items():
                if openness <= threshold:
                    state = state_name
                    break

            return {
                'openness': float(openness),
                'symmetry': float(symmetry),
                'aspect_ratio': float(aspect_ratio),
                'area': float(area),
                'perimeter': float(perimeter),
                'state': state
            }

        except Exception as e:
            logger.error(f"Error calculating lip metrics: {str(e)}")
            return {
                'openness': 0.0,
                'symmetry': 1.0,
                'aspect_ratio': 1.0,
                'area': 0.0,
                'perimeter': 0.0,
                'state': 'closed'
            }

    def analyze_sequence(self, lip_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze a sequence of lip landmarks
        Args:
            lip_sequence: Array of shape [T, N, 3] containing lip landmarks
        Returns:
            Dictionary of metric sequences and state sequence
        """
        try:
            T = len(lip_sequence)
            metrics_sequence = []
            
            for t in range(T):
                frame_metrics = self.analyze_lip_metrics(lip_sequence[t])
                metrics_sequence.append(frame_metrics)

            # Combine metrics into arrays
            metrics = {
                'openness': np.array([m['openness'] for m in metrics_sequence]),
                'symmetry': np.array([m['symmetry'] for m in metrics_sequence]),
                'aspect_ratio': np.array([m['aspect_ratio'] for m in metrics_sequence]),
                'area': np.array([m['area'] for m in metrics_sequence]),
                'perimeter': np.array([m['perimeter'] for m in metrics_sequence]),
                'states': [m['state'] for m in metrics_sequence]
            }

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing lip sequence: {str(e)}")
            # Return zero-filled arrays of correct length
            return {
                'openness': np.zeros(T),
                'symmetry': np.ones(T),
                'aspect_ratio': np.ones(T),
                'area': np.zeros(T),
                'perimeter': np.zeros(T),
                'states': ['closed'] * T
            } 
