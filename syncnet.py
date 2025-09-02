import os
import torch
import numpy as np
import subprocess
import glob
import cv2
import gdown
import python_speech_features
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from torch import nn
from urllib.parse import urlparse
import traceback
import urllib.request
import subprocess
import torchvision.transforms as transforms
import torch.nn.functional as F
from logger import logger
from typing import Union

__all__ = ['SyncNetModel', 'SyncNetInstance']

class ModelConfig:
    """Configuration for SyncNet model paths and parameters"""
    MODEL_URL = "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
    MODEL_PATH = "data/syncnet_v2.model"
    TEMP_DIR = "data/tmp"
    @classmethod
    def initialize(cls):
        """Create necessary directories and download model if needed"""
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        
        # Download model if needed
        if not os.path.exists(cls.MODEL_PATH):
            print(f"Downloading SyncNet model to {cls.MODEL_PATH}...")
            urllib.request.urlretrieve(cls.MODEL_URL, cls.MODEL_PATH)
            
            
        print("Downloads complete.")
class SyncNetModel(nn.Module):
    """Neural network model for audio-visual synchronization"""
    
    def __init__(self, num_layers_in_fc_layers: int = 1024):
        super().__init__()
        logger.info("=== Initializing SyncNetModel ===")
        logger.info(f"FC layers size: {num_layers_in_fc_layers}")
        
        # Initialize network components
        self.netcnnaud = self._build_audio_encoder()
        self.netcnnlip = self._build_visual_encoder()
        # FC layers with fixed input size
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers)
        )
        
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers)
        )

        
        logger.info("Network components initialized")

    def _build_audio_encoder(self) -> nn.Sequential:
        """Build the audio CNN encoder"""
        logger.info("Building audio encoder...")
        encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 512, kernel_size=(5,4), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Add adaptive pooling to get fixed size output
            nn.AdaptiveAvgPool2d((1, 1))
        )
        logger.info("Audio encoder built")
        return encoder

    def _build_visual_encoder(self) -> nn.Sequential:
        """Build the visual CNN encoder"""
        logger.info("Building visual encoder...")
        encoder = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            
            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            
            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            
            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        logger.info("Visual encoder built")
        return encoder

    def _build_fc_layers(self, num_layers: int) -> nn.Sequential:
        """Build fully connected layers"""
        logger.info(f"Building FC layers with output size {num_layers}")
        layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers),
        )
        logger.info("FC layers built")
        return layers

    def forward_aud(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio features with consistent shapes and devices"""
        try:
            logger.info("\n=== Audio Forward Pass ===")
            logger.info(f"Input audio shape: {x.shape}")
            
            # Check for empty tensor
            if x.numel() == 0:
                logger.warning("Received empty audio tensor - returning zero features")
                return torch.zeros(x.shape[0], 1024, device=x.device)
                
            # Handle different input shapes
            if len(x.shape) == 3:  # [B, T, D]
                x = x.unsqueeze(1)  # Add channel dim [B, 1, T, D]
            
            # Process through CNN
            logger.info("Processing through audio CNN...")
            mid = self.netcnnaud(x)  # Output will be [B, 512, 1, 1]
            logger.info(f"After CNN shape: {mid.shape}")
            
            # Flatten properly
            mid = mid.flatten(1)  # Reshape to [B, 512]
            logger.info(f"After flatten shape: {mid.shape}")
            
            # Process through FC
            logger.info("Processing through FC layers...")
            out = self.netfcaud(mid)
            logger.info(f"Final output shape: {out.shape}")
            
            return out

        except Exception as e:
            logger.error(f"Error in audio forward pass: {str(e)}")
            logger.error(f"Input tensor shape: {x.shape}")
            logger.error(f"Input tensor dtype: {x.dtype}")
            logger.error(f"Input tensor device: {x.device}")
            # Return zero features on same device
            return torch.zeros(x.shape[0], 1024, device=x.device)

    def forward_lip(self, x: torch.Tensor) -> torch.Tensor:
        """Process visual features"""
        try:
            logger.info("\n=== Visual Forward Pass ===")
            logger.info(f"Input visual shape: {x.shape}")
            
            # Process through 3D CNN
            logger.info("Processing through visual CNN...")
            mid = self.netcnnlip(x)
            logger.info(f"After CNN shape: {mid.shape}")
            
            # Flatten
            mid = mid.reshape(mid.size(0), -1)
            logger.info(f"After flatten shape: {mid.shape}")
            
            # Process through FC
            logger.info("Processing through FC layers...")
            out = self.netfclip(mid)
            logger.info(f"Final output shape: {out.shape}")
            
            # Validate output
            if torch.isnan(out).any():
                logger.error("NaN values detected in output")
            if torch.isinf(out).any():
                logger.error("Inf values detected in output")
                
            return out

        except Exception as e:
            logger.error(f"Error in visual forward pass: {str(e)}")
            logger.error(f"Input tensor shape: {x.shape}")
            logger.error(f"Input tensor dtype: {x.dtype}")
            logger.error(f"Input tensor device: {x.device}")
            logger.error("Stack trace:")
            logger.error(traceback.format_exc())
            raise
    
class SyncNetInstance(nn.Module):
    """Main SyncNet class for audio-visual synchronization"""
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.model = SyncNetModel().to(device)
        ModelConfig.initialize()
        self.load_model()
        
        # Define target dimensions for frames
        self.target_size = (224, 224)  # Standard size for face recognition models
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load pre-trained model weights"""
        state_dict = torch.load(ModelConfig.MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate(
        self,
        frames: torch.Tensor,  # [B, C, H, W] or [B, T, C, H, W]
        audio_features: torch.Tensor,  # [B, T, D] or [B, 1, T, D]
        batch_size: int = 20
    ) -> Tuple[float, torch.Tensor]:
        """Evaluate synchronization between frames and audio."""
        try:
            with torch.no_grad():
                # Get batch size and device
                B = frames.shape[0]
                device = frames.device
                logger.debug(f"Starting evaluation - batch size: {B}, device: {device}")
                
                # Check for empty audio features
                if audio_features.numel() == 0:
                    logger.warning("Empty audio features - returning default values")
                    return 0.0, torch.zeros(B, device=device)
                    
                # Move tensors to correct device
                frames = frames.to(device)
                audio_features = audio_features.to(device)
                
                # Process frames
                logger.debug(f"Input frames shape: {frames.shape}")
                frames = self.preprocess_frames(frames)
                logger.debug(f"Preprocessed frames shape: {frames.shape}")
                
                # Get visual features
                v_feats = []
                for i in range(0, B, batch_size):
                    batch = frames[i:i+batch_size]
                    v_out = self.model.forward_lip(batch)
                    v_feats.append(v_out)
                v_feats = torch.cat(v_feats, 0)
                logger.debug(f"Visual features shape: {v_feats.shape}")
                
                # Process audio features
                if audio_features.dim() == 3:  # [B, T, D]
                    audio_features = audio_features.unsqueeze(1)
                logger.debug(f"Audio features shape: {audio_features.shape}")
                
                # Get audio features
                a_feats = []
                for i in range(0, B, batch_size):
                    batch = audio_features[i:i+batch_size]
                    a_out = self.model.forward_aud(batch)
                    a_feats.append(a_out)
                a_feats = torch.cat(a_feats, 0)
                logger.debug(f"Audio features shape: {a_feats.shape}")
                
                # Validate shapes
                if v_feats.shape != a_feats.shape:
                    logger.error(f"Shape mismatch: visual={v_feats.shape}, audio={a_feats.shape}")
                    return 0.0, torch.zeros(B, device=device)
                
                # Calculate synchronization
                dists = F.pairwise_distance(v_feats, a_feats)
                confidence = torch.exp(-dists)
                offset = torch.argmin(dists).item()
                
                logger.debug(f"Computed offset: {offset}, average confidence: {confidence.mean().item():.4f}")
                
                return offset, confidence

        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0, torch.zeros(B, device=frames.device)

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Preprocess frames for SyncNet model."""
        try:
            logger.debug(f"Preprocessing frames with shape: {frames.shape}")
            
            # Handle different input formats
            if frames.dim() == 4:  # [B, C, H, W]
                B, C, H, W = frames.shape
                frames = frames.unsqueeze(2)  # Add time dim [B, C, 1, H, W]
                logger.debug(f"Added time dimension, new shape: {frames.shape}")
            elif frames.dim() == 5 and frames.shape[1] == 3:  # [B, C, T, H, W]
                pass  # Already in correct format
            elif frames.dim() == 5:  # [B, T, C, H, W]
                frames = frames.transpose(1, 2)  # [B, C, T, H, W]
                logger.debug(f"Transposed time dimension, new shape: {frames.shape}")
            
            # Get current dimensions
            B, C, T, H, W = frames.shape
            
            # Resize if needed
            if H != self.target_size[0] or W != self.target_size[1]:
                frames = frames.reshape(B * C * T, 1, H, W)
                frames = F.interpolate(
                    frames,
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                )
                frames = frames.reshape(B, C, T, *self.target_size)
                logger.debug(f"Resized to {self.target_size}, new shape: {frames.shape}")
            
            # Normalize if needed
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
                logger.debug("Normalized uint8 to float")
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 3, 1, 1, 1)
            frames = (frames - mean) / std
            
            return frames
            
        except Exception as e:
            logger.error(f"Error in preprocess_frames: {str(e)}")
            logger.error(f"Input frames shape: {frames.shape}")
            raise

    def process_batch(
        self,
        frames: torch.Tensor,      # [B, T, C, H, W]
        audio_features: torch.Tensor,  # [B, T, audio_dim]
        batch_size: int = 20
    ) -> torch.Tensor:
        """Process a batch of frames and audio."""
        try:
            B, T = frames.shape[:2]
            device = frames.device
            logger.debug(f"Processing batch with shape: {frames.shape}")
            
            confidence = torch.zeros(B, T, device=device)
            
            # Check for empty audio features
            if audio_features.numel() == 0:
                logger.warning("Empty audio features - returning zero confidence")
                return confidence
            
            # Process sequence
            for t in range(T):
                try:
                    _, conf = self.evaluate(
                        frames=frames[:, t],  # [B, C, H, W]
                        audio_features=audio_features[:, t:t+1],  # [B, 1, audio_dim]
                        batch_size=batch_size
                    )
                    confidence[:, t] = conf
                except Exception as e:
                    logger.error(f"Error processing timestep {t}: {str(e)}")
                    continue
                    
            return confidence
            
        except Exception as e:
            logger.error(f"Error in process_batch: {str(e)}")
            return torch.zeros(B, T, device=frames.device)



class SyncNetEvaluator:
    def __init__(self, config: dict):
        """Initialize SyncNet evaluator.
        
        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing SyncNet Evaluator")
        self.config = config
        
        # Initialize SyncNet model
        self.syncnet = SyncNetInstance(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set evaluation mode
        self.syncnet.eval()
        
        # Set temperature for loss computation
        self.temperature = config.loss.get('sync_temperature', 0.1)
        logger.info(f"Using temperature {self.temperature} for sync loss")

    def compute_syncnet_loss(
        self,
        generated_frames: torch.Tensor,  # [B, T, C, H, W]
        audio_features: torch.Tensor,    # [B, T, audio_dim]
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute synchronization loss between generated frames and audio.
        
        Args:
            generated_frames: Generated video frames
            audio_features: Audio features corresponding to frames
            return_metrics: Whether to return additional metrics
            
        Returns:
            Loss tensor and optionally metrics dictionary
        """
        try:
            logger.info("\n=== Computing SyncNet Loss ===")
            logger.debug(f"Input shapes - frames: {generated_frames.shape}, audio: {audio_features.shape}")
            
            # Process batches through SyncNet
            with torch.no_grad():
                confidence = self.syncnet.process_batch(
                    frames=generated_frames,
                    audio_features=audio_features,
                    batch_size=self.config.train.batch_size
                )  # [B, T]
                
                # Get batch and sequence dimensions
                B, T = confidence.shape
                logger.debug(f"Got confidence scores: {confidence.shape}")
                
                # Initialize metrics
                metrics = defaultdict(float)
                total_loss = 0
                
                # Compute loss for each batch
                for b in range(B):
                    # Get confidence scores for current sequence
                    seq_conf = confidence[b]  # [T]
                    
                    # Compute positive and negative pairs
                    pos_conf = seq_conf.mean()  # Average confidence for aligned pairs
                    
                    # Roll sequence to get misaligned pairs
                    neg_conf = []
                    for offset in range(1, T):
                        rolled = torch.roll(seq_conf, offset)
                        neg_conf.append(rolled)
                    neg_conf = torch.stack(neg_conf)  # [T-1, T]
                    neg_conf = neg_conf.mean(dim=0)  # [T]
                    
                    # Compute contrastive loss with temperature scaling
                    loss = -torch.log(
                        torch.exp(pos_conf / self.temperature) /
                        (torch.exp(pos_conf / self.temperature) + 
                         torch.exp(neg_conf / self.temperature).sum())
                    )
                    
                    total_loss += loss
                    
                    if return_metrics:
                        # Track metrics for this sequence
                        metrics['positive_confidence'] += pos_conf.item()
                        metrics['negative_confidence'] += neg_conf.mean().item()
                        metrics['sync_accuracy'] += (pos_conf > neg_conf.max()).float().item()
                
                # Average loss and metrics across batch
                total_loss = total_loss / B
                
                if return_metrics:
                    # Average metrics
                    metrics = {k: v/B for k, v in metrics.items()}
                    logger.info(f"Sync metrics: {metrics}")
                    return total_loss, metrics
                    
                return total_loss

        except Exception as e:
            logger.error(f"Error computing sync loss: {str(e)}")
            logger.error(traceback.format_exc())
            if return_metrics:
                return torch.tensor(0.0).to(generated_frames.device), {}
            return torch.tensor(0.0).to(generated_frames.device)