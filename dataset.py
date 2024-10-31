
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from insightface.app import FaceAnalysis
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer



class EmotionExtractor:
    """
    Emotion extraction using HSEmotion as specified in VASA paper
    """
    def __init__(self, face_size: Tuple[int, int] = (224, 224)):
        # Initialize HSEmotion with ResNet-34 backend as used in paper
        self.recognizer = HSEmotionRecognizer(
            model_path=None,  # Will use default ResNet34 model
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.face_size = face_size
        
        # Emotion categories from HSEmotion
        self.emotion_categories = [
            'Neutral', 'Happiness', 'Sadness', 'Surprise', 
            'Fear', 'Disgust', 'Anger', 'Contempt'
        ]

class VASADataset(Dataset):
    """
    Dataset for VASA training with enhanced emotion extraction
    """
    def __init__(
        self, 
        video_paths: List[str],
        frame_size: Tuple[int, int] = (512, 512),
        sequence_length: int = 25,
        hop_length: int = 10
    ):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.hop_length = hop_length
        
        # Initialize face analyzer
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize emotion recognizer
        self.emotion_recognizer = HSEmotionRecognizer(
            model_name='enet_b2_8',  # EfficientNet-B2 backbone
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize audio feature extractor
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.audio_model.eval()

    def _extract_face_attributes(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract face attributes including emotion from frame
        Following VASA paper's methodology
        """
        # Detect face
        faces = self.face_analyzer.get(frame)
        if not faces:
            return None
            
        face = faces[0]
        landmarks = face.landmark_3d_68
        bbox = face.bbox
        
        # Extract face region for emotion recognition
        x1, y1, x2, y2 = map(int, bbox)
        face_crop = frame[y1:y2, x1:x2]
        
        try:
            # Get emotion probabilities using HSEmotion
            emotion_probs = self.emotion_recognizer.predict_emotions(
                face_crop,
                logits=True  # Get raw logits as mentioned in paper
            )
            
            # Calculate gaze direction (θ,φ)
            gaze = self._compute_gaze_direction(landmarks)
            
            # Calculate head distance (normalized)
            distance = self._compute_head_distance(landmarks)
            
            return {
                'gaze': gaze,
                'distance': distance,
                'emotion': emotion_probs,  # Raw emotion logits
                'bbox': bbox,
                'landmarks': landmarks
            }
        except Exception as e:
            print(f"Error in emotion extraction: {str(e)}")
            return None

    def _compute_gaze_direction(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute gaze direction from facial landmarks
        Returns: (θ,φ) in radians
        """
        # Get eye landmarks
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Compute gaze direction
        eye_center = (left_eye + right_eye) / 2
        nose_tip = landmarks[30]
        
        # Calculate angles
        direction = nose_tip - eye_center
        theta = np.arctan2(direction[0], direction[2])  # yaw
        phi = np.arctan2(direction[1], direction[2])    # pitch
        
        return np.array([theta, phi])

    def _compute_head_distance(self, landmarks: np.ndarray) -> float:
        """
        Compute normalized head distance from facial landmarks
        """
        # Use outer eye corners and nose tip
        left_corner = landmarks[36]
        right_corner = landmarks[45]
        nose_tip = landmarks[30]
        
        # Compute face size from landmarks
        face_width = np.linalg.norm(right_corner - left_corner)
        face_height = np.linalg.norm(nose_tip - (left_corner + right_corner) / 2)
        face_size = face_width * face_height
        
        # Normalize by frame size
        normalized_distance = face_size / (self.frame_size[0] * self.frame_size[1])
        return normalized_distance

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        video_path = self.video_paths[idx]
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
        
        # Handle cases where we couldn't get enough valid frames
        if valid_frames < self.sequence_length:
            # Repeat last valid frame
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
                attributes.append(attributes[-1])
        
        # Extract corresponding audio features
        start_time = start_frame / fps
        duration = self.sequence_length / fps
        audio_features = self._extract_audio_features(
            video_path.replace('.mp4', '.wav'),
            start_time,
            duration
        )
        
        # Prepare final tensors
        frames_tensor = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1) / 255.0 
            for f in frames
        ])
        
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
                'start_frame': start_frame,
                'fps': fps
            }
        }