import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
import os
from pathlib import Path
import subprocess
import random
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import logging
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VASADataset(Dataset):
    """
    Dataset for VASA training with enhanced emotion extraction and audio handling
    """
    def __init__(
        self, 
        video_folder: str,
        frame_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 25,
        hop_length: int = 10,
        cache_audio: bool = True,
        preextract_audio: bool = False,
        max_videos: Optional[int] = None,  # New parameter
        random_seed: int = 42  # New parameter for reproducibility
    ):
        self.video_folder = Path(video_folder)
        self.cache_audio = cache_audio
        random.seed(random_seed)
        
        # Create audio cache directory if needed
        self.audio_cache_dir = self.video_folder / "audio_cache"
        if self.cache_audio:
            self.audio_cache_dir.mkdir(exist_ok=True)
        
        # Get all video files from folder
        all_videos = [str(f) for f in self.video_folder.rglob("*.mp4")]
        logger.info(f"Found {len(all_videos)} total videos")
        
        # Randomly sample max_videos if specified
        if max_videos is not None and max_videos < len(all_videos):
            self.video_paths = random.sample(all_videos, max_videos)
            logger.info(f"Randomly sampled {max_videos} videos for processing")
        else:
            self.video_paths = all_videos
            
        logger.info(f"Using {len(self.video_paths)} videos for dataset")
        
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        
        # Initialize face analyzer
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize emotion recognizer
        model_name = 'enet_b0_8_va_mtl'
        self.emotion_recognizer = HSEmotionRecognizer(model_name=model_name)
        
        # Initialize audio feature extractor
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
        self.audio_model.eval()
        
        # Add transforms
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(frame_size[0]),
            transforms.CenterCrop(frame_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True
            )
        ])

        # Pre-extract audio if requested
        if preextract_audio:
            self._preextract_all_audio()

    def _preextract_all_audio(self):
        """Pre-extract audio from all videos"""
        logger.info(f"Pre-extracting audio for {len(self.video_paths)} videos...")
        for i, video_path in enumerate(self.video_paths, 1):
            try:
                self._get_or_extract_audio(video_path)
                if i % 100 == 0:  # Log progress every 100 videos
                    logger.info(f"Processed {i}/{len(self.video_paths)} videos")
            except Exception as e:
                logger.error(f"Failed to extract audio for {video_path}: {e}")


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
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
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

    def _extract_audio_features(
        self,
        video_path: str,
        start_time: float,
        duration: float
    ) -> torch.Tensor:
        """Extract audio features using Wav2Vec2"""
        try:
            audio_path = self._get_or_extract_audio(video_path)
            
            # Load audio segment
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert start_time and duration to samples
            start_sample = int(start_time * sample_rate)
            duration_samples = int(duration * sample_rate)
            
            # Handle case where requested duration exceeds file length
            if start_sample >= waveform.shape[1]:
                start_sample = 0
            if start_sample + duration_samples > waveform.shape[1]:
                duration_samples = waveform.shape[1] - start_sample
            
            # Extract segment
            audio_segment = waveform[:, start_sample:start_sample + duration_samples]
            
            # Ensure minimum length (pad if necessary)
            min_samples = int(0.1 * sample_rate)  # Minimum 100ms
            if audio_segment.shape[1] < min_samples:
                audio_segment = F.pad(audio_segment, (0, min_samples - audio_segment.shape[1]))
            
            # Process through Wav2Vec2
            with torch.no_grad():
                inputs = self.audio_processor(
                    audio_segment,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                outputs = self.audio_model(**inputs)
                features = outputs.last_hidden_state
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features from {video_path}: {str(e)}")
            raise

    def get_audio_path(self, video_path: str) -> str:
        """Public method to get audio path for testing"""
        return str(self._get_or_extract_audio(video_path))

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        while True:
            try:
                video_path = self.video_paths[idx]
                
                # Ensure audio is available
                audio_path = self._get_or_extract_audio(video_path)
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Random sequence start point
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = np.random.randint(0, total_frames - self.sequence_length)
                
                frames = []
                attributes = []
                valid_frames = 0
                
                # Extract frames and attributes
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                while valid_frames < self.sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Preprocess frame
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Extract face attributes
                    attrs = self._extract_face_attributes(frame)
                    if attrs is None:
                        continue
                        
                    frames.append(frame)
                    attributes.append(attrs)
                    valid_frames += 1
                
                cap.release()
                
                if valid_frames < self.sequence_length:
                    while len(frames) < self.sequence_length:
                        frames.append(frames[-1])
                        attributes.append(attributes[-1])
                
                # Extract audio features
                start_time = start_frame / fps
                duration = self.sequence_length / fps
                audio_features = self._extract_audio_features(
                    video_path,
                    start_time,
                    duration
                )
                
                # Prepare final tensors
                frames_tensor = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1) / 255.0 
                    for f in frames
                ])
                
                # Apply transformations
                frames_tensor = self.pixel_transforms(frames_tensor)
                
                gaze_tensor = torch.stack([
                    torch.from_numpy(a['gaze']).float() 
                    for a in attributes
                ])
                
                distance_tensor = torch.stack([
                    torch.tensor([a['distance']]).float() 
                    for a in attributes
                ])
                
                emotion_tensor = torch.stack([
                    torch.from_numpy(a['emotion']).float() 
                    for a in attributes
                ])
                
                return {
                    'frames': frames_tensor,
                    'audio_features': audio_features,
                    'gaze': gaze_tensor,
                    'distance': distance_tensor,
                    'emotion': emotion_tensor,
                    'metadata': {
                        'video_path': video_path,
                        'audio_path': str(audio_path),
                        'video_name': os.path.basename(video_path),
                        'start_frame': start_frame,
                        'fps': fps
                    }
                }
                
            except Exception as e:
                logger.error(f"Error processing video {self.video_paths[idx]}: {e}")
                idx = np.random.randint(0, len(self.video_paths))