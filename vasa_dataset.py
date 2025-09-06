import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import os
import json
from pathlib import Path
import subprocess
import random
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
from l2cs import L2CS, select_device, Pipeline
import h5py
from tqdm import tqdm
from typing import *
from collections import defaultdict

# Import the new chunked window cache
try:
    from window_cache import WindowCache as ChunkedWindowCache
    USE_CHUNKED_CACHE = True
    logger.info("Using ChunkedWindowCache for flexible window sizes")
except ImportError:
    ChunkedWindowCache = None
    USE_CHUNKED_CACHE = False
    logger.info("ChunkedWindowCache not available, using built-in cache")
from torchvision.utils import save_image
from datetime import datetime
import hashlib
from vasa_model import BlinkConditionHandler
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
            # Initialize device if not already set
            # if self._l2cs_device is None:
            #     self._l2cs_device = select_device('cpu', batch_size=1)
                
            # Create pipeline
            import pathlib
            # Use the existing L2CSNet_gaze360.pkl file which has 92MB
            weights_path = pathlib.Path('models/L2CSNet_gaze360.pkl')
            
            self._l2cs_pipeline =  Pipeline(
                weights=weights_path,
                arch='ResNet50',
                device='cuda'
            )
            
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
        """Lazy initialization of audio model"""
        if self._audio_model is None:
            from transformers import Wav2Vec2Model
            self._audio_model = Wav2Vec2Model.from_pretrained(
                'facebook/wav2vec2-base'
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
    def face_mesh(self):
        """Lazy initialization of face mesh"""
        if self._face_mesh is None:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
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
                        if key == 'metadata':
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
    ):
        VASADatasetMixin.__init__(self)
        
        # Basic initialization
        self.video_folder = Path(video_folder)
        self.emo_model = emo_model
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size
        self.cache_audio = cache_audio
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        self.device = device
        self.model_device = next(emo_model.parameters()).device
        
        # Use chunked cache if available for flexible window support
        if USE_CHUNKED_CACHE and ChunkedWindowCache:
            cache_path = Path(cache_dir) if cache_dir else Path(video_folder) / "window_cache_chunked"
            self.cache = ChunkedWindowCache(
                cache_dir=cache_path,
                chunk_size=1000,  # 1000 frames per chunk
                overlap_size=50,  # 50 frame overlap for context
                max_memory_cache=5  # Keep 5 chunks in memory
            )
            logger.info(f"Initialized ChunkedWindowCache at {cache_path}")
        else:
            # Fallback to built-in cache
            self.cache = WindowCache(Path(video_folder) / "window_cache")
            logger.info("Using built-in WindowCache")

        self.blink_handler = BlinkConditionHandler(window_size=sequence_length)

        self.tracker = ProblematicVideosTracker(Path("bad_videos"))


        # Set up caching
        self.audio_cache_dir = self.video_folder / "audio_cache"
        self.audio_cache_dir.mkdir(exist_ok=True)
        logger.info(f"Using audio cache directory: {self.audio_cache_dir}")
        
        # Get all videos
        all_videos = [str(f) for f in self.video_folder.rglob("*.mp4")]
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
        

    def _get_emotion(self, face_crop: np.ndarray) -> np.ndarray:
        """Get emotion VA (valence-arousal) values using HSEmotion MTL model"""
        try:
            # Ensure face crop is the right size
            if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
                face_crop = cv2.resize(face_crop, (64, 64))
            
            # Get predictions - returns (label, scores) where scores includes VA values
            _, scores = self.emotion_recognizer.predict_emotions(face_crop, logits=True)
            
            # Extract VA values (last two values in scores)
            if isinstance(scores, np.ndarray) and scores.size >= 2:
                va_values = scores[-2:]  # Get last two values (valence, arousal)
                va_values = np.array(va_values, dtype=np.float32)
                
                # Ensure correct shape
                if va_values.shape != (2,):
                    logger.warning(f"Unexpected VA shape: {va_values.shape}")
                    return np.zeros(2, dtype=np.float32)
                    
                # Apply tanh to ensure values are in [-1, 1] range
                va_values = np.tanh(va_values)
                
                return va_values
                
            return np.zeros(2, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error in emotion VA extraction: {str(e)}")
            return np.zeros(2, dtype=np.float32)  # [valence, arousal]
        
            
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

            # Minimum frames needed for window + context
            min_frames = self.window_size + self.context_size + self.stride  # 50 + 10 + 25 = 85
            
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
        """Extract EMO features with proper batch and sequence dimensions"""
        with torch.no_grad():
            try:
                logger.debug("\n=== EMO Feature Extraction Start ===")
                logger.debug(f"Input frames shape: {frames.shape}")
                logger.debug(f"Input frames device: {frames.device}")
                logger.debug(f"EMO model device: {next(self.emo_model.parameters()).device}")

                assert len(frames.shape) == 4, f"Expected 4D input [T,C,H,W], got shape {frames.shape}"
                assert frames.shape[1] == 3, f"Expected 3 channels, got {frames.shape[1]}"
                
                T = frames.shape[0]
                assert T == 50, f"Expected sequence length 50, got {T}"

                # Add batch dimension and move to device
                frames = frames.unsqueeze(0)  # [1,T,C,H,W] 
                frames = frames.to(next(self.emo_model.parameters()).device)
                logger.debug(f"After adding batch dim - frames shape: {frames.shape}")
                
                outputs = {
                    'theta': [],
                    'scale': [],
                    'rotation': [],
                    'translation': [],
                    'expression_embed': []
                }

                logger.debug("\nProcessing frames in sequence...")
                
                for t in range(T):
                    frame = frames[:, t]  # [1,C,H,W]
                    logger.debug(f"\nFrame {t}:")
                    logger.debug(f"  Current frame shape: {frame.shape}")

                    input_dict = {
                        'source_img': frame,
                        'target_img': frame.clone(),
                        'source_mask': torch.ones_like(frame[:, :1]),
                        'target_mask': torch.ones_like(frame[:, :1]),
                        'crop': False
                    }

                    # Get theta AND scale/rotation/translation 
                    theta, scale, rotation, translation = self.emo_model.head_pose_regressor.forward(
                        input_dict['source_img'],
                        return_srt=True
                    )
                    theta = self.convert_theta_format(theta)
                    expression_embed = self.emo_model.expression_embedder_nw.net_face(
                        input_dict['source_img']
                    )[0]

                    # Verify feature shapes
                    assert theta.shape == (1, 3, 4), f"Wrong theta shape: {theta.shape}" 
                    assert scale.shape == (1, 3), f"Wrong scale shape: {scale.shape}"
                    assert rotation.shape == (1, 3), f"Wrong rotation shape: {rotation.shape}" 
                    assert translation.shape == (1, 3), f"Wrong translation shape: {translation.shape}"
                    assert expression_embed.shape == (1, 128), f"Wrong expression shape: {expression_embed.shape}"
                    
                    logger.debug("  Feature shapes for current frame:")
                    logger.debug(f"    theta: {theta.shape} on {theta.device}")
                    logger.debug(f"    scale: {scale.shape} on {scale.device}")
                    logger.debug(f"    rotation: {rotation.shape} on {rotation.device}")
                    logger.debug(f"    translation: {translation.shape} on {translation.device}")
                    logger.debug(f"    expression_embed: {expression_embed.shape} on {expression_embed.device}")
                    
                    # Store outputs
                    outputs['theta'].append(theta)
                    outputs['scale'].append(scale)
                    outputs['rotation'].append(rotation)
                    outputs['translation'].append(translation)
                    outputs['expression_embed'].append(expression_embed)

                # Stack along time dimension 
                outputs = {
                    k: torch.stack(v, dim=1)  # [B=1, T=50, ...]
                    for k, v in outputs.items()
                }

                # Verify final output shapes
                logger.debug("\nFinal output shapes:")
                expected_shapes = {
                    'theta': (1, T, 3, 4), # [B, T, 3, 4] 
                    'scale': (1, T, 3),
                    'rotation': (1, T, 3),
                    'translation': (1, T, 3),
                    'expression_embed': (1, T, 128)
                }

                for k, expected_shape in expected_shapes.items():
                    actual_shape = outputs[k].shape
                    assert actual_shape == expected_shape, f"Wrong {k} shape: expected {expected_shape}, got {actual_shape}"
                    logger.debug(f"  {k}: {actual_shape} on {outputs[k].device}")

                logger.debug("=== EMO Feature Extraction Complete ===\n")
                
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
        pitch = torch.asin(torch.clamp(-matrix[2, 0], -1, 1))
        
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
            logger.debug(f"Starting audio feature extraction for {video_path}")
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
                    # Process with wav2vec
                    inputs = self.audio_processor(
                        audio_segment.squeeze(0),
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                        padding=True
                    )
                    outputs = self.audio_model(**inputs)
                    features = outputs.last_hidden_state  # [1, T, 768]

                # Ensure exactly window_size features through interpolation
                features = F.interpolate(
                    features.transpose(1, 2),
                    size=self.window_size,
                    mode='linear'
                ).transpose(1, 2)

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
            

            return features, mfcc_features

        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            logger.error(traceback.format_exc())
            return (
                torch.zeros((1, self.window_size, 384 if use_whisper else 768)),
                torch.zeros((1, self.window_size, 13))
            )
                    
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
        return len(self.video_paths)



            
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
            
            # Get face landmarks
            results = self.face_mesh.process(frame)
            
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
                
            except Exception as e:
                import traceback
                logger.error(f"Error in L2CS gaze extraction: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                gaze = np.zeros(2, dtype=np.float32)

            # Calculate face size/distance
            right_eye = np.mean(landmarks_68[42:48], axis=0)  # Right eye center
            left_eye = np.mean(landmarks_68[36:42], axis=0)   # Left eye center
            face_width = np.linalg.norm(right_eye - left_eye)
            face_size = face_width / width
            distance = np.array([face_size], dtype=np.float32)

            # Get emotion using existing emotion recognizer
            emotion_logits = self._get_emotion(face_crop)

            return {
                'landmarks': landmarks_68,
                'emotion': emotion_logits,
                'gaze': gaze,  # L2CS gaze results
                'head_distance': distance,
                'bbox': bbox
            }

        except Exception as e:
            logger.error(f"Error in face attribute extraction: {str(e)}")
            logger.error(traceback.format_exc())
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
            
            if not results.multi_face_landmarks:
                logger.error(f"No faces detected in frame from video: {video_path}")
                # return None
                raise Exception("No faces detected")

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
        """Get all windows for a video with proper audio feature handling"""
        try:
            # Get video path for this index
            video_path = self.video_paths[idx]
            # logger.debug(f"Processing video: {video_path}")
            # Check cache first
            if self.cache.has_cache(video_path):
                windows_data = self.cache.load_windows(video_path)
                return {
                    'windows': windows_data,
                    'video_path': video_path,
                    'num_windows': len(windows_data)
                }

            # Process each window
            windows_data = []
            for window in self.video_windows[video_path]:
                try:
                    # Extract frames
                    frames, frame_indices = self._extract_frames(
                        video_path,
                        window['start_frame'],
                        self.sequence_length
                    )
                    
                    if not frames:
                        continue

                    # Extract EMO features
                    frames_tensor = torch.stack(frames)
                    with torch.no_grad():
                        emo_features = self._extract_emo_features(frames_tensor)
                        if not emo_features:
                            continue

                    # Extract both types of audio features
                    wav2vec_features, mfcc_features = self._extract_audio_features(
                        video_path,
                        start_time=window['start_frame'] / window['fps'],
                        duration=self.window_size / window['fps']
                    )


                    # Process face attributes
                    gaze_angles = []
                    emotion_logits = []
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
                            continue

                    lip_motion_sequence = self._get_lip_motion_sequence(lips_landmarks)
                    if lip_motion_sequence is None:
                        logger.warning(f"Skipping window - no valid lip motion")
                        continue

                    # Create window data with correct key names
                    window_data = {
                        'frames': torch.stack(frames),
                        'theta': emo_features['theta'],
                        'scale': emo_features['scale'],
                        'rotation': emo_features['rotation'],
                        'translation': emo_features['translation'],
                        'expression_embed': emo_features['expression_embed'],
                        'scale': emo_features['scale'],
                        'audio_features': wav2vec_features,
                        'audio_mfcc': mfcc_features,
                        'gaze': torch.tensor(np.stack(gaze_angles), dtype=torch.float32),
                        'emotion': torch.tensor(np.stack(emotion_logits), dtype=torch.float32),
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
                        
                        'metadata': {
                            'video_path': str(video_path),
                            'start_frame': window['start_frame'],
                            'fps': window['fps'],
                            'has_context': window['has_context']
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
                    
                    windows_data.append(window_data)

                except Exception as e:
                    logger.error(f"Error processing window: {str(e)}")
                    self.tracker.dispatch(VideoEventData(
                        video_path=video_path,
                        event_type=VideoEvent.PROCESSING_ERROR,
                        details={"error": f"Face attribute error: {str(e)}"}
                    ))
                    logger.error(traceback.format_exc())
                    continue

            if not windows_data:
                logger.warning(f"No valid windows for video {video_path}")
                return self._get_zero_sample()

            # Cache the results
            self.cache.save_windows(video_path, windows_data)
        
            return {
                'windows': windows_data,
                'video_path': video_path,
                'num_windows': len(windows_data)
            }

        except Exception as e:
            logger.error(f"Error in __getitem__: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_zero_sample()

    def _get_zero_sample(self) -> Dict[str, torch.Tensor]:
        """Return a zero-filled sample with all required features including landmarks and lip motion"""
        return {
            'frames': torch.zeros((self.sequence_length, 3, *self.frame_size)),
            'theta': torch.zeros((self.sequence_length, 3, 4)),
            'scale': torch.zeros((self.sequence_length, 3)),  
            'rotation': torch.zeros((self.sequence_length, 3)),
            'translation': torch.zeros((self.sequence_length, 3)),
            'audio_features': torch.zeros((1, self.sequence_length, 768)),   # wav2vec
            'audio_mfcc': torch.zeros((1, self.sequence_length, 13)),       # mfcc for syncnet
            'gaze': torch.zeros((self.sequence_length, 2)),
            'head_distance': torch.zeros((self.sequence_length, 1)),
            'emotion': torch.zeros((self.sequence_length, 2)),
            'speed_bucket': torch.zeros((self.sequence_length, 1), dtype=torch.long),
            'expression_embed': torch.zeros((1, self.sequence_length, 128)),
            'scale': torch.zeros((1, self.sequence_length, 3)),
            
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
