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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VASADataset(Dataset):
    def __init__(
        self, 
        video_folder: str,
        frame_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 25,
        hop_length: int = 10,
        cache_audio: bool = True,
        preextract_audio: bool = False,
        max_videos: Optional[int] = None,
        random_seed: int = 42
    ):
        self.video_folder = Path(video_folder)
        self.cache_audio = cache_audio
        random.seed(random_seed)
        
        # Create audio cache directory within the video folder
        if self.cache_audio:
            self.audio_cache_dir = self.video_folder / "audio_cache"
            self.audio_cache_dir.mkdir(exist_ok=True)
            logger.info(f"Using audio cache directory: {self.audio_cache_dir}")
        
        # Get all video files from folder
        all_videos = [str(f) for f in self.video_folder.rglob("*.mp4")]
        logger.info(f"Found {len(all_videos)} total videos")
        
        # Try to load cached audio status
        self.audio_status_file = self.audio_cache_dir / "audio_status.json"
        if self.audio_status_file.exists():
            with open(self.audio_status_file, 'r') as f:
                self.audio_status = json.load(f)
            logger.info("Loaded cached audio status")
        else:
            # Check videos for audio streams
            self.audio_status = self._check_videos_for_audio(all_videos)
            # Save audio status
            with open(self.audio_status_file, 'w') as f:
                json.dump(self.audio_status, f)
            logger.info("Saved new audio status cache")
        
        videos_with_audio = [v for v, has_audio in self.audio_status.items() if has_audio]
        logger.info(f"Found {len(videos_with_audio)} videos with audio")
        
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

    def _check_videos_for_audio(self, video_paths: List[str]) -> Dict[str, bool]:
        """Check which videos have audio streams using ffprobe"""
        audio_status = {}
        total_videos = len(video_paths)
        
        for i, video_path in enumerate(video_paths, 1):
            try:
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
                else:
                    has_audio = False
                    
                audio_status[video_path] = has_audio
                
                if i % 100 == 0:
                    logger.info(f"Checked audio for {i}/{total_videos} videos")
                
            except Exception as e:
                logger.error(f"Error checking audio in {video_path}: {str(e)}")
                audio_status[video_path] = False
                
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
            logger.error(f"Error extracting audio from {video_path}: {str(e)}")
            raise

    def _get_or_extract_audio(self, video_path: str) -> Optional[Path]:
        """Get audio file path, extracting audio if necessary"""
        # Check if video has audio
        if not self.audio_status.get(video_path, False):
            return None
            
        audio_path = self._get_audio_path(video_path)
        
        if not audio_path.exists():
            logger.info(f"Extracting audio for {video_path}")
            self._extract_audio(video_path, audio_path)
            
        return audio_path

    def _extract_audio_features(
        self,
        video_path: str,
        start_time: float,
        duration: float
    ) -> torch.Tensor:
        """Extract audio features using Wav2Vec2"""
        try:
            # Return zero tensor for videos without audio
            if not self.audio_status.get(video_path, False):
                return torch.zeros((1, self.sequence_length, 768))
                
            audio_path = self._get_or_extract_audio(video_path)
            if audio_path is None:
                return torch.zeros((1, self.sequence_length, 768))
                
            # Load audio segment
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
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
            
            # Reshape audio to match wav2vec2 requirements
            # Should be [batch_size, sequence_length]
            audio_segment = audio_segment.squeeze(0)  # Remove channel dim
            
            # Process through Wav2Vec2
            with torch.no_grad():
                inputs = self.audio_processor(
                    audio_segment,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True
                )
                outputs = self.audio_model(**inputs)
                features = outputs.last_hidden_state
                
            # Ensure consistent sequence length through adaptive pooling
            features = F.adaptive_avg_pool1d(
                features.transpose(1, 2),  # [B, C, T]
                self.sequence_length
            ).transpose(1, 2)  # Back to [B, T, C]
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features from {video_path}: {str(e)}")
            return torch.zeros((1, self.sequence_length, 768))


    def _preextract_all_audio(self):
        """Pre-extract audio from all videos"""
        logger.info(f"Pre-extracting audio for {len(self.video_paths)} videos...")
        for i, video_path in enumerate(self.video_paths, 1):
            try:
                if self.audio_status.get(video_path, False):
                    self._get_or_extract_audio(video_path)
                if i % 10 == 0:
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


    def get_audio_path(self, video_path: str) -> str:
        """Public method to get audio path for testing"""
        return str(self._get_or_extract_audio(video_path))

    def __len__(self) -> int:
        return len(self.video_paths)



        
    def _extract_face_attributes(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract face attributes including landmarks, emotion, gaze, and distance
        """
        try:
            logger.info("Starting face attribute extraction...")
            
            # Detect faces using InsightFace
            faces = self.face_analyzer.get(frame)
            if not faces:
                logger.info("No faces detected in frame")
                return None
                
            logger.info(f"Detected {len(faces)} faces")
            
            # Get largest face if multiple detected
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Convert landmarks and bbox to numpy arrays
            try:
                landmarks = np.array(face.landmark_3d_68, dtype=np.float32)
                logger.info(f"Landmarks shape: {landmarks.shape}, dtype: {landmarks.dtype}")
            except Exception as e:
                logger.error(f"Error converting landmarks: {str(e)}")
                return None
                
            try:
                bbox = np.array(face.bbox, dtype=np.int32)
                logger.info(f"Bbox shape: {bbox.shape}, dtype: {bbox.dtype}")
            except Exception as e:
                logger.error(f"Error converting bbox: {str(e)}")
                return None
            
            # Extract face region for emotion recognition
            x1, y1, x2, y2 = bbox
            margin = 0.2
            h, w = y2 - y1, x2 - x1
            x1 = max(0, int(x1 - margin * w))
            x2 = min(frame.shape[1], int(x2 + margin * w))
            y1 = max(0, int(y1 - margin * h))
            y2 = min(frame.shape[0], int(y2 + margin * h))
            
            logger.info(f"Face crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            if x1 >= x2 or y1 >= y2:
                logger.warning("Invalid crop region")
                return None
                
            face_crop = frame[y1:y2, x1:x2]
            logger.info(f"Face crop shape: {face_crop.shape}")
            
            if face_crop.size == 0:
                logger.warning("Empty face crop")
                return None
            
            # Resize for emotion recognition if needed
            original_size = face_crop.shape[:2]
            if face_crop.shape[0] < 64 or face_crop.shape[1] < 64:
                face_crop = cv2.resize(face_crop, (64, 64))
                logger.info(f"Resized face from {original_size} to {face_crop.shape[:2]}")
            
            logger.info("Starting face attribute extraction...")

            # Get emotion logits
            try:
                logger.info("Starting emotion recognition...")
                emotions = self.emotion_recognizer.predict_emotions(face_crop, logits=True)
                logger.info(f"Raw emotion output type: {type(emotions)}")
                logger.info(f"Raw emotion output value: {emotions}")
                
                # Initialize zero array
                emotion_logits = np.zeros(8, dtype=np.float32)
                
                # Map emotion indices to standard order
                emotion_map = {
                    'Neutral': 0,
                    'Happiness': 1,
                    'Sadness': 2,
                    'Surprise': 3,
                    'Fear': 4,
                    'Disgust': 5,
                    'Anger': 6,
                    'Contempt': 7
                }
                
                if isinstance(emotions, tuple) and len(emotions) == 2:
                    logger.info("Processing tuple emotion output")
                    emotion_label, logits_array = emotions
                    
                    # Map the logits array to our standard emotion ordering
                    # The logits array from HSEmotion has 10 values, we need to map the relevant ones
                    hs_emotion_map = {
                        'Neutral': 0,
                        'Happiness': 1,
                        'Sadness': 2,
                        'Surprise': 3,
                        'Fear': 4,
                        'Disgust': 5,
                        'Anger': 6,
                        'Contempt': 7,
                    }
                    
                    # Convert logits to proper format
                    logits_array = np.array(logits_array, dtype=np.float32)
                    logger.info(f"Logits array shape: {logits_array.shape}")
                    
                    # Take the first 8 elements (corresponding to the basic emotions)
                    # and apply softmax to get probabilities
                    emotion_probs = np.exp(logits_array[:8])
                    emotion_probs = emotion_probs / np.sum(emotion_probs)
                    
                    # Convert probabilities back to logits for consistency
                    emotion_logits = np.log(np.clip(emotion_probs, 1e-7, 1.0))
                    
                    logger.info(f"Processed emotion logits: {emotion_logits}")
                    
                elif isinstance(emotions, dict):
                    logger.info("Processing dictionary emotion output")
                    for emotion, value in emotions.items():
                        if emotion in emotion_map:
                            idx = emotion_map[emotion]
                            emotion_logits[idx] = float(value)
                            
                elif isinstance(emotions, (list, np.ndarray)):
                    logger.info(f"Processing array-like emotion output: shape={np.array(emotions).shape}")
                    emotions = np.array(emotions, dtype=np.float32)
                    if emotions.size == 8:
                        emotion_logits = emotions
                else:
                    logger.error(f"Unexpected emotion output type: {type(emotions)}")
                    
                logger.info(f"Final emotion_logits shape: {emotion_logits.shape}, dtype: {emotion_logits.dtype}")
                logger.info(f"Final emotion_logits values: {emotion_logits}")
                
                # Verify the output is valid
                assert emotion_logits.shape == (8,), f"Wrong emotion shape: {emotion_logits.shape}"
                assert not np.isnan(emotion_logits).any(), "NaN values in emotion logits"
                assert not np.isinf(emotion_logits).any(), "Inf values in emotion logits"
                
            except Exception as e:
                logger.error(f"Error in emotion recognition: {str(e)}")
                logger.error(f"Emotion recognition error details: {str(e)}", exc_info=True)
                emotion_logits = np.zeros(8, dtype=np.float32)


            # Calculate gaze direction
            gaze = np.zeros(2, dtype=np.float32)
            try:
                left_eye = landmarks[36:42].mean(axis=0)
                right_eye = landmarks[42:48].mean(axis=0)
                eye_center = (left_eye + right_eye) / 2
                nose_tip = landmarks[30]
                
                direction = nose_tip - eye_center
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                
                gaze[0] = np.arctan2(direction[0], direction[2])
                gaze[1] = np.arctan2(direction[1], direction[2])
                logger.info(f"Computed gaze angles: {gaze}")
                
            except Exception as e:
                logger.error(f"Error calculating gaze: {str(e)}")
            
            # Calculate distance
            distance = np.array([0.5], dtype=np.float32)
            try:
                left_corner = landmarks[36]
                right_corner = landmarks[45]
                nose_tip = landmarks[30]
                
                face_width = np.linalg.norm(right_corner - left_corner)
                face_height = np.linalg.norm(nose_tip - (left_corner + right_corner) / 2)
                face_size = face_width * face_height
                frame_area = frame.shape[0] * frame.shape[1]
                
                if frame_area > 0:
                    distance[0] = np.clip(face_size / frame_area, 0, 1)
                logger.info(f"Computed face distance: {distance[0]}")
                
            except Exception as e:
                logger.error(f"Error calculating distance: {str(e)}")
            
            # Verify outputs
            try:
                assert isinstance(landmarks, np.ndarray) and landmarks.dtype == np.float32, \
                    f"Invalid landmarks: type={type(landmarks)}, dtype={landmarks.dtype}"
                assert isinstance(emotion_logits, np.ndarray) and emotion_logits.dtype == np.float32, \
                    f"Invalid emotions: type={type(emotion_logits)}, dtype={emotion_logits.dtype}"
                assert isinstance(gaze, np.ndarray) and gaze.dtype == np.float32, \
                    f"Invalid gaze: type={type(gaze)}, dtype={gaze.dtype}"
                assert isinstance(distance, np.ndarray) and distance.dtype == np.float32, \
                    f"Invalid distance: type={type(distance)}, dtype={distance.dtype}"
                assert isinstance(bbox, np.ndarray) and bbox.dtype == np.int32, \
                    f"Invalid bbox: type={type(bbox)}, dtype={bbox.dtype}"
                
                assert landmarks.shape == (68, 3), f"Wrong landmarks shape: {landmarks.shape}"
                assert emotion_logits.shape == (8,), f"Wrong emotion shape: {emotion_logits.shape}"
                assert gaze.shape == (2,), f"Wrong gaze shape: {gaze.shape}"
                assert distance.shape == (1,), f"Wrong distance shape: {distance.shape}"
                assert bbox.shape == (4,), f"Wrong bbox shape: {bbox.shape}"
                
                logger.info("All attribute shapes and types verified successfully")
                
            except Exception as e:
                logger.error(f"Verification failed: {str(e)}")
                return None
            
            return {
                'landmarks': landmarks,
                'emotion': emotion_logits,
                'gaze': gaze,
                'distance': distance,
                'bbox': bbox
            }
                
        except Exception as e:
            logger.error(f"Error in face attribute extraction: {str(e)}")
            return None
    

    def _emotion_logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert emotion logits to probabilities using softmax"""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities back to logits"""
        return np.log(np.clip(probs, 1e-7, 1.0))

    def _verify_attributes(self, attrs: Optional[Dict[str, np.ndarray]]) -> bool:
        """
        Verify that extracted attributes are valid
        """
        if attrs is None:
            return False
            
        try:
            required_shapes = {
                'landmarks': (68, 3),
                'emotion': (8,),
                'gaze': (2,),
                'distance': (1,),
                'bbox': (4,)
            }
            
            required_dtypes = {
                'landmarks': np.float32,
                'emotion': np.float32,
                'gaze': np.float32,
                'distance': np.float32,
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
            logger.error(f"Error verifying attributes: {str(e)}")
            return False
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            try:
                video_path = self.video_paths[idx]
                
                # Ensure audio is available
                audio_path = self._get_or_extract_audio(video_path)
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Random sequence start point
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = np.random.randint(0, max(1, total_frames - self.sequence_length))
                
                frames = []
                attributes = []
                valid_frames = 0
                
                # Extract frames and attributes
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                consecutive_failures = 0
                
                while valid_frames < self.sequence_length and consecutive_failures < 30:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Preprocess frame
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Extract face attributes
                    attrs = self._extract_face_attributes(frame)
                    if attrs is None:
                        consecutive_failures += 1
                        continue
                        
                    frames.append(frame)
                    attributes.append(attrs)
                    valid_frames += 1
                    consecutive_failures = 0
                
                cap.release()
                
                # If we don't have enough valid frames, try another video
                if valid_frames < self.sequence_length:
                    attempt += 1
                    idx = random.randint(0, len(self.video_paths) - 1)
                    continue
                
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
                    torch.from_numpy(a['distance']).float() 
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
                        'audio_path': str(audio_path) if audio_path else None,
                        'video_name': os.path.basename(video_path),
                        'start_frame': start_frame,
                        'fps': fps
                    }
                }
                
            except Exception as e:
                logger.error(f"Error processing video {self.video_paths[idx]}: {e}")
                attempt += 1
                idx = random.randint(0, len(self.video_paths) - 1)
        
        # If all attempts fail, return a zero-filled sample
        return self._get_zero_sample()

    def _get_zero_sample(self) -> Dict[str, torch.Tensor]:
        """Return a zero-filled sample as fallback"""
        return {
            'frames': torch.zeros((self.sequence_length, 3, *self.frame_size)),
            'audio_features': torch.zeros((1, self.sequence_length, 768)),
            'gaze': torch.zeros((self.sequence_length, 2)),
            'distance': torch.zeros((self.sequence_length, 1)),
            'emotion': torch.zeros((self.sequence_length, 8)),
            'metadata': {
                'video_path': '',
                'audio_path': None,
                'video_name': '',
                'start_frame': 0,
                'fps': 0
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
            logger.error(f"Error computing gaze: {str(e)}")
            return np.array([0.0, 0.0])

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
            logger.error(f"Error computing distance: {str(e)}")
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
            logger.error(f"Error computing face angles: {str(e)}")
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
            logger.error(f"Error normalizing landmarks: {str(e)}")
            return landmarks
