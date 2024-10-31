import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from torchmetrics.image import FrechetInceptionDistance
import cv2
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from mysixdrepnet import SixDRepNet_Detector

def extract_pose_sequences(video: torch.Tensor, use_6d_rotation: bool = True) -> torch.Tensor:
    """
    Extract pose sequences from video frames using SixDRepNet or other pose estimator
    Args:
        video: Video tensor of shape [B, T, C, H, W]
        use_6d_rotation: Whether to use 6D rotation representation
    Returns:
        pose_sequences: Pose parameters [B, T, pose_dim]
    """
    B, T, C, H, W = video.shape
    device = video.device
    
    # Initialize pose estimator
    pose_estimator = SixDRepNet_Detector().to(device)
    pose_estimator.eval()
    
    pose_sequences = []
    
    with torch.no_grad():
        # Process each batch and frame
        for b in range(B):
            batch_poses = []
            for t in range(T):
                frame = video[b, t].cpu().numpy().transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)  # Normalize if needed
                
                # Get pose parameters
                pose_params = pose_estimator(frame)
                
                if use_6d_rotation:
                    # Convert to 6D rotation representation
                    rot_matrix = Rotation.from_euler('xyz', pose_params[:3]).as_matrix()
                    rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_matrix))
                    pose = torch.cat([rot_6d, torch.from_numpy(pose_params[3:])], dim=0)
                else:
                    pose = torch.from_numpy(pose_params)
                
                batch_poses.append(pose)
            
            # Stack temporal sequence
            batch_poses = torch.stack(batch_poses)
            pose_sequences.append(batch_poses)
    
    # Stack batches
    pose_sequences = torch.stack(pose_sequences).to(device)
    return pose_sequences

def compute_pose_intensity(pose_sequences: torch.Tensor) -> float:
    """
    Compute pose variation intensity from pose sequences
    Following VASA paper's methodology
    Args:
        pose_sequences: Pose parameters [B, T, pose_dim]
    Returns:
        intensity: Average pose variation score
    """
    B, T, D = pose_sequences.shape
    device = pose_sequences.device
    
    # Split rotation and translation
    rotation = pose_sequences[..., :6]  # 6D rotation
    translation = pose_sequences[..., 6:]
    
    # Compute rotation differences
    rot_diff = []
    for t in range(T-1):
        # Convert 6D rotation to matrices
        rot1 = rotation_6d_to_matrix(rotation[:, t])
        rot2 = rotation_6d_to_matrix(rotation[:, t+1])
        
        # Compute geodesic distance between rotations
        R_rel = torch.bmm(rot1.transpose(1, 2), rot2)
        theta = torch.acos(torch.clamp(
            (torch.diagonal(R_rel, dim1=1, dim2=2).sum(1) - 1) / 2,
            -1 + 1e-6,
            1 - 1e-6
        ))
        rot_diff.append(theta)
    
    rot_diff = torch.stack(rot_diff, dim=1)  # [B, T-1]
    
    # Compute translation differences
    trans_diff = torch.norm(
        translation[:, 1:] - translation[:, :-1],
        dim=-1
    )  # [B, T-1]
    
    # Combine rotation and translation variations
    pose_diff = rot_diff + trans_diff
    
    # Compute statistics
    mean_intensity = pose_diff.mean().item()
    std_intensity = pose_diff.std().item()
    max_intensity = pose_diff.max().item()
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'max_intensity': max_intensity,
        'total_intensity': mean_intensity + std_intensity
    }

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation
    From Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    """
    return matrix[:2, :].flatten()

def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix"""
    x = rotation_6d[..., :3]
    y = rotation_6d[..., 3:]
    
    x = F.normalize(x, dim=-1)
    z = torch.cross(x, y)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x)
    
    matrix = torch.stack([x, y, z], dim=-2)
    return matrix

class VideoFrechetDistance:
    """
    Compute Fréchet Video Distance (FVD) between real and generated videos
    Implementation follows the paper's methodology
    """
    def __init__(
        self,
        feature_extractor: str = "i3d",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize FID computer
        self.fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True
        ).to(self.device)
        
        # Initialize video feature extractor (I3D or similar)
        if feature_extractor == "i3d":
            self.feature_extractor = I3DFeatureExtractor().to(self.device)
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    @torch.no_grad()
    def compute_fvd(
        self,
        generated_videos: torch.Tensor,
        real_videos: torch.Tensor,
        batch_size: int = 8
    ) -> float:
        """
        Compute FVD between real and generated videos
        Args:
            generated_videos: Generated videos [B, T, C, H, W]
            real_videos: Real videos [B, T, C, H, W]
            batch_size: Batch size for feature extraction
        Returns:
            fvd_score: Fréchet Video Distance
        """
        # Extract features
        gen_features = self._extract_features(generated_videos, batch_size)
        real_features = self._extract_features(real_videos, batch_size)
        
        # Update FID computer
        self.fid.update(real_features, real=True)
        self.fid.update(gen_features, real=False)
        
        # Compute FID
        fvd_score = self.fid.compute().item()
        
        # Reset FID computer
        self.fid.reset()
        
        return fvd_score

    def _extract_features(
        self,
        videos: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Extract features from videos in batches"""
        features = []
        num_batches = (len(videos) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Extracting features"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(videos))
            batch = videos[start_idx:end_idx].to(self.device)
            
            # Extract features
            batch_features = self.feature_extractor(batch)
            features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0)

#  NOT used 
# class I3DFeatureExtractor(nn.Module):
#     """I3D network for video feature extraction"""
#     def __init__(self, pretrained: bool = True):
#         super().__init__()
#         # Initialize I3D network
#         # This would typically load pretrained I3D weights
#         # Actual implementation would depend on available I3D implementation
#         pass

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Extract features from video"""
#         # Implementation would depend on I3D network
#         pass

def compute_fvd(generated_video: torch.Tensor, real_video: torch.Tensor) -> float:
    """
    Wrapper function to compute FVD between videos
    Args:
        generated_video: Generated video [B, T, C, H, W]
        real_video: Real video [B, T, C, H, W]
    Returns:
        fvd_score: Fréchet Video Distance
    """
    fvd_computer = VideoFrechetDistance()
    return fvd_computer.compute_fvd(generated_video, real_video)


from syncnet import SyncNet



class CAPPScore(nn.Module):
    """
    Contrastive Audio and Pose Pretraining (CAPP) score implementation
    """
    def __init__(self, pose_dim: int, audio_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # Pose encoder (6-layer transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=6),
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Audio encoder (initialized from Wav2Vec2)
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.audio_proj = nn.Linear(768, hidden_dim)  # Wav2Vec2 dim to hidden_dim
        
        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, pose_sequences: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Compute CAPP score for pose-audio pairs
        """
        # Encode pose sequences
        pose_embeddings = self.pose_encoder(pose_sequences)
        pose_embeddings = F.normalize(pose_embeddings, dim=-1)
        
        # Encode audio features
        audio_embeddings = self.audio_encoder(audio_features).last_hidden_state
        audio_embeddings = self.audio_proj(audio_embeddings)
        audio_embeddings = audio_embeddings.mean(dim=1)  # Global pooling
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(pose_embeddings, audio_embeddings.T) / self.temperature
        
        return similarity



class Evaluator:
    """
    Evaluation metrics for VASA
    """
    def __init__(self):
        self.syncnet = SyncNet()  # Load pretrained SyncNet
        self.capp_scorer = CAPPScore(pose_dim=6, audio_dim=768)
        
    @torch.no_grad()
    def compute_metrics(self,
                       generated_video: torch.Tensor,
                       audio_features: torch.Tensor,
                       real_video: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics for generated video
        """
        metrics = {}
        
        # Compute SyncNet confidence and distance
        sync_conf, sync_dist = self.syncnet(generated_video, audio_features)
        metrics['sync_confidence'] = sync_conf.mean().item()
        metrics['sync_distance'] = sync_dist.mean().item()
        
        # Compute CAPP score
        pose_sequences = extract_pose_sequences(generated_video)  # Extract pose from video
        capp_similarity = self.capp_scorer(pose_sequences, audio_features)
        metrics['capp_score'] = capp_similarity.diagonal().mean().item()
        
        # Compute pose variation intensity
        metrics['pose_intensity'] = compute_pose_intensity(pose_sequences)
        
        # Compute FVD if real video is provided
        if real_video is not None:
            metrics['fvd'] = compute_fvd(generated_video, real_video)
        
        return metrics
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

# # Initialize metrics
# metrics = VASAMetrics(device)

# # Compute metrics
# results = metrics.compute_metrics(
#     generated_video=generated_video,
#     audio_features=audio_features
# )

# # Print results
# print(f"Sync Confidence: {results['sync_confidence']:.4f}")
# print(f"CAPP Score: {results['capp_score']:.4f}")
# print(f"Pose Intensity: {results['pose_intensity']:.4f}")

class VASAMetrics:
    """
    Metrics implementation following VASA paper
    Key metrics:
    1. Lip sync (SyncNet)
    2. CAPP score (Audio-pose alignment)
    3. Pose variation intensity (ΔP)
    """
    def __init__(self, device: torch.device):
        self.device = device
        
        # Initialize SyncNet
        self.syncnet = SyncNet().to(device)
        
        # Initialize CAPP scorer
        self.capp_scorer = CAPPScore(
            pose_dim=6,  # 3 rotation + 3 translation
            audio_dim=768,  # Wav2Vec2 feature dimension
            hidden_dim=512
        ).to(device)
        
        # Initialize pose extractor
        self.pose_extractor = SixDRepNet_Detector().to(device)

    @torch.no_grad()
    def compute_metrics(
        self,
        generated_video: torch.Tensor,
        audio_features: torch.Tensor,
        real_video: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics following paper's methodology
        Args:
            generated_video: Generated video frames [B, T, C, H, W]
            audio_features: Audio features [B, T, C]
            real_video: Optional ground truth video
        """
        metrics = {}
        
        # 1. Audio-Visual Sync Metrics (SyncNet)
        sync_conf, sync_dist = self.compute_sync_metrics(
            generated_video,
            audio_features
        )
        metrics['sync_confidence'] = sync_conf.mean().item()
        metrics['sync_distance'] = sync_dist.mean().item()
        
        # 2. CAPP Score (Audio-Pose Alignment)
        pose_sequences = self.extract_pose_sequences(generated_video)
        capp_score = self.compute_capp_score(
            pose_sequences,
            audio_features
        )
        metrics['capp_score'] = capp_score.mean().item()
        
        # 3. Pose Variation Intensity
        intensity_metrics = self.compute_pose_intensity(pose_sequences)
        metrics.update(intensity_metrics)
        
        return metrics

    def compute_sync_metrics(
        self,
        video: torch.Tensor,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SyncNet confidence and distance
        """
        return self.syncnet(video, audio)

    def extract_pose_sequences(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract pose sequences from video frames
        Returns: [B, T, 6] tensor (3 rotation + 3 translation)
        """
        B, T, C, H, W = video.shape
        poses = []
        
        for b in range(B):
            batch_poses = []
            for t in range(T):
                frame = video[b, t].cpu().numpy().transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)
                
                # Extract pose using SixDRepNet
                pose = self.pose_extractor(frame)
                batch_poses.append(torch.from_numpy(pose))
            
            poses.append(torch.stack(batch_poses))
        
        return torch.stack(poses).to(self.device)

    def compute_capp_score(
        self,
        pose_sequences: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CAPP score as described in paper
        """
        return self.capp_scorer(pose_sequences, audio_features)

    def compute_pose_intensity(
        self,
        pose_sequences: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute ΔP (pose variation intensity) as defined in paper
        """
        # Split into rotation and translation
        rotation = pose_sequences[..., :3]
        translation = pose_sequences[..., 3:]
        
        # Compute frame-to-frame differences
        rot_diff = torch.norm(
            rotation[:, 1:] - rotation[:, :-1],
            dim=-1
        ).mean()
        
        trans_diff = torch.norm(
            translation[:, 1:] - translation[:, :-1],
            dim=-1
        ).mean()
        
        return {
            'pose_intensity': (rot_diff + trans_diff).item() / 2,
            'rotation_intensity': rot_diff.item(),
            'translation_intensity': trans_diff.item()
        }