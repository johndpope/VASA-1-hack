"""
VASA Loss Module - Handles all loss computations for VASA training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import traceback
import wandb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger


from syncnet import SyncNetInstance
from synchformer_wrapper import SynchformerInstance


def safe_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Safe extraction of Euler angles from rotation matrix with clamping to avoid NaN in backward.
    
    Args:
        R: Rotation matrix of shape [..., 3, 3]
    
    Returns:
        Euler angles of shape [..., 3] (pitch, yaw, roll)
    """
    # Store original shape
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    # Check for gimbal lock
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    singular = sy < 1e-6
    
    # Safe extraction with clamping to avoid gradient explosion
    # The backward of asin(x) is 1/sqrt(1-x^2) which explodes at x=±1
    sin_pitch = torch.clamp(-R[:, 2, 0], -0.9999, 0.9999)
    
    pitch = torch.asin(sin_pitch)
    
    # Non-singular case
    yaw = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    roll = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    
    # Singular case (gimbal lock)
    yaw_singular = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    roll_singular = torch.zeros_like(roll)
    
    # Select based on singularity
    yaw = torch.where(singular.unsqueeze(-1), yaw_singular.unsqueeze(-1), yaw.unsqueeze(-1)).squeeze(-1)
    roll = torch.where(singular.unsqueeze(-1), roll_singular.unsqueeze(-1), roll.unsqueeze(-1)).squeeze(-1)
    
    # Stack and reshape
    euler = torch.stack([pitch, yaw, roll], dim=-1)
    return euler.reshape(*original_shape, 3)


class SpeedLossHandler:
    """Handler for speed bucket classification loss."""
    
    def __init__(self, num_buckets: int = 9):
        self.num_buckets = num_buckets
        self.speed_buckets = torch.linspace(0, 1, num_buckets)
    
    def compute_speed_loss(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_buckets: torch.Tensor,
        lambda_speed: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute speed classification loss.
        
        Args:
            pred_motion: Predicted motion parameters
            target_buckets: Target speed bucket indices [B, T, 1]
            lambda_speed: Weight for speed loss
            device: Computation device
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        try:
            metrics = {}
            
            # Compute motion speed from predicted parameters
            if 'theta' not in pred_motion:
                return torch.tensor(0.0, device=device), metrics
            
            # Calculate frame-to-frame motion magnitude
            theta_diff = pred_motion['theta'][:, 1:] - pred_motion['theta'][:, :-1]
            
            # Compute Frobenius norm of the difference
            motion_speed = torch.norm(theta_diff.reshape(theta_diff.shape[0], theta_diff.shape[1], -1), dim=-1)
            
            # Normalize to [0, 1] range
            motion_speed = motion_speed / (motion_speed.max() + 1e-8)
            
            # Assign to speed buckets
            B, T_minus_1 = motion_speed.shape
            pred_buckets = torch.zeros(B, T_minus_1, device=device, dtype=torch.long)
            
            # Create logits for cross_entropy (distance to each bucket center)
            # Shape: [B, T_minus_1, num_buckets]
            logits = torch.zeros(B, T_minus_1, self.num_buckets, device=device)
            
            for i in range(self.num_buckets):
                # Use negative distance as logit (closer = higher score)
                bucket_center = self.speed_buckets[i] if i < len(self.speed_buckets) else 1.0
                logits[:, :, i] = -torch.abs(motion_speed - bucket_center)
                
                # Also assign discrete buckets for accuracy calculation
                if i == 0:
                    mask = motion_speed <= self.speed_buckets[i]
                elif i == self.num_buckets - 1:
                    mask = motion_speed > self.speed_buckets[i-1]
                else:
                    mask = (motion_speed > self.speed_buckets[i-1]) & (motion_speed <= self.speed_buckets[i])
                pred_buckets[mask] = i
            
            # Ensure target_buckets has the right shape
            target_buckets = target_buckets.squeeze(-1)  # Remove last dimension if present
            if target_buckets.shape[1] > T_minus_1:
                target_buckets = target_buckets[:, :T_minus_1]
            
            # Classification loss using logits
            speed_loss = F.cross_entropy(
                logits.reshape(-1, self.num_buckets),  # [B*T, num_buckets]
                target_buckets.long().reshape(-1)       # [B*T]
            )
            
            # Compute accuracy
            accuracy = (pred_buckets == target_buckets).float().mean()
            
            # Store metrics
            metrics['speed_loss'] = speed_loss.item()
            metrics['speed_accuracy'] = accuracy.item()
            
            return speed_loss * lambda_speed, metrics
            
        except Exception as e:
            logger.error(f"Error in speed loss: {str(e)}")
            return torch.tensor(0.0, device=device), {}


class VASALossModule:
    """Loss module for VASA training, focusing on control signal adherence."""
    
    def __init__(
        self,
        volumetric_avatar: nn.Module,
        config: Dict,
        device: str = 'cuda',
        model: Optional[nn.Module] = None
    ):
        self.volumetric_avatar = volumetric_avatar
        self.config = config
        self.device = device
        self.model = model  # Store reference to VASAModel for Flow-DPO
        

        self.speed_handler = SpeedLossHandler(num_buckets=9)

 # Initialize sync evaluator (SyncNet or Synchformer)
        use_synchformer = config.loss.get('use_synchformer', False)
        if use_synchformer:
            logger.info("Using Synchformer for audio-visual synchronization")
            self.syncnet = SynchformerInstance(device=device)
        else:
            logger.info("Using SyncNet for audio-visual synchronization")
            self.syncnet = SyncNetInstance(device=device)
        self.syncnet.eval()
        
        self.batch_size =  config.train.batch_size
        # Extract loss weights from config
           # Extract loss weights from config
        self.lambda_pose = config.loss.lambda_pose
        self.lambda_dynamics = config.loss.lambda_dynamics
        self.lambda_scale = getattr(config.loss, 'lambda_scale', 1.0)
        self.lambda_rotation = getattr(config.loss, 'lambda_rotation', 1.0)
        self.lambda_translation = getattr(config.loss, 'lambda_translation', 1.0)
        self.lambda_gaze_direction = config.loss.lambda_gaze_direction
        self.lambda_distance = config.loss.lambda_head_distance
        self.lambda_emotion = config.loss.lambda_emotion
        self.lambda_speed = config.loss.lambda_speed

        self.lambda_temporal = config.loss.lambda_temporal

        # Extract loss weights from config
        self.lambda_sync = config.loss.lambda_sync  # Weight for sync loss
        
        # New landmark loss weights
        self.lambda_lips = config.loss.lambda_lips  # Loss weight for lip motion
        self.lambda_nonlip = config.loss.lambda_nonlip  # Loss weight for other facial landmarks

        self.lambda_blink = config.loss.lambda_blink

      # Initialize transform for verification images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        import lpips
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.vis_freq = config.vis.vis_freq
        
        # New VASA-1 style disentanglement loss weights
        self.lambda_consist = getattr(config.loss, 'lambda_consist', 1.0)
        self.lambda_cross_id = getattr(config.loss, 'lambda_cross_id', 0.1)
        self.lambda_velocity = getattr(config.loss, 'lambda_velocity', 1e-4)
        self.lambda_expression_l1 = getattr(config.loss, 'lambda_expression_l1', 0.5)  # Balance between L1 and L2
        self.lambda_smoothness = getattr(config.loss, 'lambda_smoothness', 1e-4)

        # Audio-lip correlation loss weight
        self.lambda_audio_lip = getattr(config.loss, 'lambda_audio_lip', 2.0)
        self.lambda_mouth_openness = getattr(config.loss, 'lambda_mouth_openness', 10.0)
        self.lambda_aux_phoneme = getattr(config.loss, 'lambda_aux_phoneme', 0.05)

        # Flow-DPO loss weight (VideoReward framework - Liu et al., 2025)
        self.lambda_flow_dpo = getattr(config.loss, 'lambda_flow_dpo', 0.5)
        self.flow_dpo_start_epoch = getattr(config.loss, 'flow_dpo_start_epoch', 20)
        self.flow_beta_scale = getattr(config.loss, 'flow_beta_scale', 1.0)



        # Check if using derived warps (disable warp loss if true)
        self.use_derived_warps = getattr(config.model, 'use_derived_warps', False)

        # Initialize identity feature extractor for cross-id loss
        try:
            from facenet_pytorch import InceptionResnetV1
            self.id_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            for param in self.id_extractor.parameters():
                param.requires_grad = False
            logger.info("Initialized InceptionResnetV1 for identity extraction")
        except Exception as e:
            logger.warning(f"Could not initialize identity extractor: {e}")
            self.id_extractor = None

    def _extract_mouth_masks(self, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Extract mouth region masks from frames using MediaPipe landmarks.

        Args:
            frames: Video frames [B, C, H, W] in range [-1, 1] or [0, 1]
            device: Target device for output tensor

        Returns:
            Mouth masks [B, 1, H, W] with values 0-1 (1 = mouth region)
        """
        try:
            import mediapipe as mp
            import cv2

            B, C, H, W = frames.shape

            # Convert frames to numpy [0, 255] range for MediaPipe
            frames_np = frames.detach().cpu().numpy()
            if frames_np.min() < 0:  # If in [-1, 1]
                frames_np = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            else:  # If in [0, 1]
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

            # Convert from [B, C, H, W] to [B, H, W, C]
            frames_np = np.transpose(frames_np, (0, 2, 3, 1))

            # Initialize MediaPipe Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

            # Process each frame
            mouth_masks = []
            for b in range(B):
                # Convert RGB to BGR for MediaPipe
                frame_bgr = cv2.cvtColor(frames_np[b], cv2.COLOR_RGB2BGR)

                # Detect landmarks
                results = face_mesh.process(frame_bgr)

                if results.multi_face_landmarks:
                    # Get mouth landmarks (MediaPipe lip indices: 61-68, 291-308)
                    landmarks = results.multi_face_landmarks[0].landmark

                    # Inner mouth: indices 78, 191, 80, 81, 82, 13, 312, 311, 310, 415
                    # Outer mouth: indices 61, 146, 91, 181, 84, 17, 314, 405, 321, 375
                    mouth_indices = list(range(61, 69)) + list(range(291, 309))  # All mouth landmarks

                    # Extract mouth points
                    mouth_points = []
                    for idx in mouth_indices:
                        lm = landmarks[idx]
                        x, y = int(lm.x * W), int(lm.y * H)
                        mouth_points.append([x, y])

                    mouth_points = np.array(mouth_points, dtype=np.int32)

                    # Create mouth mask
                    mask = np.zeros((H, W), dtype=np.float32)
                    cv2.fillConvexPoly(mask, mouth_points, 1.0)

                    # Dilate mask slightly to include surrounding area
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    mask = cv2.dilate(mask, kernel, iterations=1)

                    mouth_masks.append(mask)
                else:
                    # No face detected - use center region as fallback
                    mask = np.zeros((H, W), dtype=np.float32)
                    center_y, center_x = H // 2 + H // 6, W // 2  # Slightly below center
                    mouth_h, mouth_w = H // 6, W // 4
                    mask[center_y - mouth_h // 2:center_y + mouth_h // 2,
                         center_x - mouth_w // 2:center_x + mouth_w // 2] = 1.0
                    mouth_masks.append(mask)

            # Stack and convert to tensor [B, 1, H, W]
            masks_np = np.stack(mouth_masks, axis=0)[:, np.newaxis, :, :]
            masks_tensor = torch.from_numpy(masks_np).float().to(device)

            face_mesh.close()
            return masks_tensor

        except Exception as e:
            logger.warning(f"Error extracting mouth masks, using center region: {e}")
            # Fallback: simple center region mask
            B, C, H, W = frames.shape
            mask = torch.zeros(B, 1, H, W, device=device)
            center_y, center_x = H // 2 + H // 6, W // 2
            mouth_h, mouth_w = H // 6, W // 4
            mask[:, :, center_y - mouth_h // 2:center_y + mouth_h // 2,
                 center_x - mouth_w // 2:center_x + mouth_w // 2] = 1.0
            return mask

    def _compute_blink_loss(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_blinks: torch.Tensor,
        lambda_blink: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss to ensure generated motion follows blink patterns.

        Args:
            pred_motion: Dictionary containing predicted motion parameters OR
                         extracted blink states [B, T, 3] from generated frames
            target_blinks: Target blink states [B, T, 3] tensor where:
                - Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
                - Channel 1: Left eye openness (0-1)
                - Channel 2: Right eye openness (0-1)
            lambda_blink: Weight for blink loss
            device: Computation device

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        metrics = {}

        # Extract predicted blink states
        if 'blink_state' in pred_motion:
            pred_blinks = pred_motion['blink_state']  # [B, T, 3]
        else:
            # No blink states available
            return torch.tensor(0.0, device=device), {}

        # Ensure shapes match
        if pred_blinks.shape != target_blinks.shape:
            logger.warning(f"Blink shape mismatch: pred={pred_blinks.shape}, target={target_blinks.shape}")
            return torch.tensor(0.0, device=device), {}

        # Move to device
        pred_blinks = pred_blinks.to(device).float()
        target_blinks = target_blinks.to(device).float()

        # 1. Eye openness loss (channels 1-2: continuous values 0-1)
        pred_openness = pred_blinks[:, :, 1:]  # [B, T, 2] (left, right eye openness)
        target_openness = target_blinks[:, :, 1:]

        openness_loss = F.mse_loss(pred_openness, target_openness)
        metrics['blink_openness_loss'] = openness_loss

        # 2. Blink phase loss (channel 0: categorical 0-3)
        # Treat as regression for simplicity (could use cross-entropy but phases are ordered)
        pred_phase = pred_blinks[:, :, 0]  # [B, T]
        target_phase = target_blinks[:, :, 0]

        phase_loss = F.mse_loss(pred_phase, target_phase)
        metrics['blink_phase_loss'] = phase_loss

        # 3. Combined loss
        total_loss = (openness_loss + phase_loss) * lambda_blink

        # 4. Compute accuracy metrics
        with torch.no_grad():
            # Phase accuracy (within 0.5 of correct phase)
            phase_correct = (torch.abs(pred_phase - target_phase) < 0.5).float().mean()
            metrics['blink_phase_accuracy'] = phase_correct

            # Openness MAE
            openness_mae = torch.abs(pred_openness - target_openness).mean()
            metrics['blink_openness_mae'] = openness_mae

        return total_loss, metrics
    
    

    def plot_to_wandb_image(self,fig):
  
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from PIL import Image

        """Convert matplotlib figure to wandb image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return wandb.Image(Image.open(buf))

    
    
    
    


    def _compute_landmark_losses(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """Compute losses for lip and non-lip facial landmarks."""
            try:
                losses = {}
                
                # 1. Lip Motion Loss
                if 'lips' in targets:
                    pred_lips = outputs['lips']
                    target_lips = targets['lips']

                    # Compute positional loss for lips
                    lips_pos_loss = F.mse_loss(pred_lips, target_lips)

                    # Compute velocity loss for smoother lip motion (only if we have multiple frames)
                    if pred_lips.shape[1] > 1:
                        lips_vel_pred = pred_lips[:, 1:] - pred_lips[:, :-1]
                        lips_vel_target = target_lips[:, 1:] - target_lips[:, :-1]
                        lips_vel_loss = F.mse_loss(lips_vel_pred, lips_vel_target)
                        # Combined lip loss
                        lips_loss = self.lambda_lips * (lips_pos_loss + 0.5 * lips_vel_loss)
                    else:
                        lips_vel_loss = torch.tensor(0.0, device=pred_lips.device)
                        lips_loss = self.lambda_lips * lips_pos_loss

                    losses['lips_pos_loss'] = lips_pos_loss
                    losses['lips_vel_loss'] = lips_vel_loss
                    losses['lips_total'] = lips_loss
                    
                # 2. Non-lip Facial Motion Losses
                nonlip_losses = []
                
                # Process each non-lip landmark group
                landmark_groups = {
                    'right_eye': 'right_eye',
                    'left_eye': 'left_eye',
                    'jaw': 'jaw',
                    'nose': 'nose'
                }
                
                for group_name, key in landmark_groups.items():
                    if key in targets:
                        pred = outputs[key]
                        target = targets[key]
                        
                        # Position loss
                        pos_loss = F.mse_loss(pred, target)
                        losses[f'{group_name}_pos_loss'] = pos_loss
                        
                        # Velocity loss for smooth motion
                        if pred.shape[1] > 1:  # Only if we have multiple frames
                            vel_pred = pred[:, 1:] - pred[:, :-1]
                            vel_target = target[:, 1:] - target[:, :-1]
                            vel_loss = F.mse_loss(vel_pred, vel_target)
                            losses[f'{group_name}_vel_loss'] = vel_loss
                            
                            # Combined loss for this landmark group
                            group_loss = pos_loss + 0.5 * vel_loss
                        else:
                            group_loss = pos_loss
                            
                        losses[f'{group_name}_total'] = group_loss
                        nonlip_losses.append(group_loss)
                
                # Combine non-lip losses
                if nonlip_losses:
                    nonlip_total = sum(nonlip_losses) * self.lambda_nonlip
                    losses['nonlip_total'] = nonlip_total
                    
                    # Add to total facial motion loss
                    if 'lips_total' in losses:
                        losses['facial_motion_total'] = losses['lips_total'] + nonlip_total
                    else:
                        losses['facial_motion_total'] = nonlip_total
                
                return losses
                
            except Exception as e:
                logger.error(f"Error computing landmark losses: {str(e)}")
                logger.error(traceback.format_exc())
                return {}




    def compute_audio_lip_correlation(self, pred_motion, audio_features, lip_metrics, generated_frames=None):
        """
        Compute audio-lip correlation loss to enforce synchronization between
        audio energy and lip motion.

        Args:
            pred_motion: Predicted motion parameters (kept for compatibility)
            audio_features: Audio features [B, T, D] from wav2vec2
            lip_metrics: Dictionary containing lip motion metrics from dataset (fallback)
            generated_frames: Generated frames [B, T, C, H, W] to extract lips from (preferred)

        Returns:
            Audio-lip correlation loss scaled by lambda_audio_lip
        """
        try:
            logger.debug("\n=== Audio-Lip Correlation Loss Computation ===")

            # Extract lip openness metric from dataset
            # This should be provided by the dataset (e.g., computed from landmarks)
            lip_openness = lip_metrics.get('openness')  # [B, T]

            logger.debug(f"Lip metrics keys available: {lip_metrics.keys() if isinstance(lip_metrics, dict) else 'Not a dict'}")

            if lip_openness is None:
                logger.debug("Lip openness not directly provided, attempting to compute from landmarks")
                # If lip openness not provided, try to compute from lip landmarks
                if 'lips' in lip_metrics:
                    # Compute openness as vertical distance between upper and lower lips
                    lips = lip_metrics['lips']  # Expected shape: [B, T, num_lip_points, 2]
                    logger.debug(f"Lips shape: {lips.shape}")
                    # Simple approximation: use mean vertical distance
                    upper_lips = lips[:, :, :lips.shape[2]//2, 1]  # Upper lip y-coords
                    lower_lips = lips[:, :, lips.shape[2]//2:, 1]  # Lower lip y-coords
                    lip_openness = (lower_lips.mean(dim=-1) - upper_lips.mean(dim=-1)).abs()
                    logger.debug(f"Computed lip openness from landmarks, shape: {lip_openness.shape}")
                else:
                    # No lip metrics available, return zero loss
                    logger.warning("No lip metrics available, returning zero loss")
                    return torch.tensor(0.0, device=audio_features.device)
            else:
                logger.debug(f"Using provided lip openness, shape: {lip_openness.shape}")

            # Log lip openness statistics
            logger.debug(f"Lip openness stats - min: {lip_openness.min():.6f}, max: {lip_openness.max():.6f}, "
                        f"mean: {lip_openness.mean():.6f}, std: {lip_openness.std():.6f}")

            # Compute audio energy/magnitude
            audio_energy = torch.norm(audio_features, dim=-1)  # [B, T] audio magnitude
            logger.debug(f"Audio features shape: {audio_features.shape}, Audio energy shape: {audio_energy.shape}")
            logger.debug(f"Audio energy stats - min: {audio_energy.min():.6f}, max: {audio_energy.max():.6f}, "
                        f"mean: {audio_energy.mean():.6f}, std: {audio_energy.std():.6f}")

            # Check for silence vs speech
            silence_threshold = 0.1
            is_silent = audio_energy.mean() < silence_threshold
            logger.debug(f"Audio type: {'SILENT' if is_silent else 'SPEECH'} (mean energy: {audio_energy.mean():.6f})")

            # Normalize both signals for better correlation
            # Use L2 normalization (like SyncNet) instead of min-max for better gradient flow
            # Reshape to [B*T, 1] for normalization, then reshape back
            B, T = lip_openness.shape
            lip_openness_norm = F.normalize(lip_openness.reshape(-1, 1), p=2, dim=0).reshape(B, T)
            audio_energy_norm = F.normalize(audio_energy.reshape(-1, 1), p=2, dim=0).reshape(B, T)

            logger.debug(f"Normalized lip openness - min: {lip_openness_norm.min():.6f}, max: {lip_openness_norm.max():.6f}, "
                        f"mean: {lip_openness_norm.mean():.6f}")
            logger.debug(f"Normalized audio energy - min: {audio_energy_norm.min():.6f}, max: {audio_energy_norm.max():.6f}, "
                        f"mean: {audio_energy_norm.mean():.6f}")

            # Compute MSE loss between normalized signals
            correlation_loss = F.mse_loss(lip_openness_norm, audio_energy_norm)
            logger.debug(f"Raw correlation loss (MSE): {correlation_loss.item():.6f}")

            # Scale by lambda weight
            scaled_loss = correlation_loss * self.lambda_audio_lip
            logger.debug(f"Scaled correlation loss (lambda={self.lambda_audio_lip}): {scaled_loss.item():.6f}")

            # Additional debug: Check correlation
            if lip_openness_norm.numel() > 0 and audio_energy_norm.numel() > 0:
                # Flatten tensors for correlation
                lip_flat = lip_openness_norm.flatten()
                audio_flat = audio_energy_norm.flatten()
                if len(lip_flat) > 1:
                    # Compute Pearson correlation
                    vx = lip_flat - torch.mean(lip_flat)
                    vy = audio_flat - torch.mean(audio_flat)
                    correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                    logger.debug(f"Pearson correlation between lip openness and audio energy: {correlation.item():.4f}")

            logger.debug("=== End Audio-Lip Correlation Loss ===\n")

            return scaled_loss

        except Exception as e:
            logger.warning(f"Error computing audio-lip correlation loss: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return torch.tensor(0.0, device=audio_features.device)

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        return_metrics: bool = True,
        current_epoch: Optional[int] = None,
        step: Optional[int] = None,
        generated_frames: Optional[torch.Tensor] = None,
        target_frames: Optional[torch.Tensor] = None,
        source_identity: Optional[torch.Tensor] = None,
        dataset = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses including expression verification and perceptual loss."""
        try:
            logger.debug("\n=== Computing Losses ===")
            losses = {}
            metrics = {}
            device = outputs['theta'].device

            # 1. Reconstruction losses
            logger.debug("\nComputing reconstruction losses:")
            recon_losses = self._compute_reconstruction_losses(outputs, targets, noise, step)
            losses.update(recon_losses)

            # Log reconstruction losses at INFO level to track training progress
            total_recon_loss = sum(v.item() for v in recon_losses.values() if isinstance(v, torch.Tensor))
            logger.info(f"📊 Total reconstruction loss: {total_recon_loss:.4f}")

            # Diagnostic: Check if predictions are collapsing to same value
            if 'expression_embed' in outputs and 'expression_embed' in targets:
                pred_expr_std = outputs['expression_embed'].std().item()
                target_expr_std = targets['expression_embed'].std().item()
                expr_diff = (outputs['expression_embed'] - targets['expression_embed']).abs().mean().item()

                logger.info(f"🔍 Expression diagnostics:")
                logger.info(f"  Predicted std: {pred_expr_std:.6f}, Target std: {target_expr_std:.6f}")
                logger.info(f"  Pred-Target diff: {expr_diff:.6f}")

                # Add to metrics for wandb logging
                metrics['expression_std/predicted'] = pred_expr_std
                metrics['expression_std/target'] = target_expr_std
                metrics['expression_std/diff'] = expr_diff

                if pred_expr_std < 1e-4:
                    logger.error(f"🚨 PREDICTION COLLAPSE: expression_embed has near-zero variance ({pred_expr_std:.8f})!")
                    logger.error(f"  Model is outputting constant values - gradients may be vanishing!")
                    metrics['diagnostics/expression_collapse'] = 1.0
                else:
                    metrics['diagnostics/expression_collapse'] = 0.0

            logger.debug("Reconstruction losses:")
            for k, v in recon_losses.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: {v.item():.6f}")

            # 1.5 Motion Diversity Loss - Encourage variance to prevent collapse
            # logger.debug("\nComputing motion diversity loss:")
            # if 'expression_embed' in outputs:
            #     expr = outputs['expression_embed']  # [B, T, D]
            #     B, T, D = expr.shape

            #     # Compute entropy of expression embeddings to encourage diversity
            #     # Higher entropy = more diverse expressions
            #     expr_flat = expr.view(-1, D)  # Flatten to [B*T, D]

            #     # Normalize to probabilities using softmax
            #     expr_norm = F.softmax(expr_flat.abs(), dim=-1)  # Use abs to handle negative values

            #     # Compute entropy: -sum(p * log(p))
            #     entropy = -(expr_norm * torch.log(expr_norm + 1e-8)).sum(-1).mean()

            #     # We want HIGH entropy (diverse), so we minimize negative entropy
            #     diversity_loss = -entropy * 0.1  # Small weight to encourage diversity
            #     losses['motion_diversity'] = diversity_loss

            #     logger.debug(f"  Expression entropy: {entropy.item():.6f}")

            #     # 1.6 Audio-Expression Direct Coupling - Force expression to follow audio energy
            #     audio_key = 'audio_features' if 'audio_features' in conditions else 'audio' if 'audio' in conditions else None
            #     if audio_key is not None:
            #         audio = conditions[audio_key]  # [B, T_audio, 768]

            #         # Align audio sequence length with expression sequence length
            #         B_expr, T_expr, D_expr = expr.shape
            #         B_audio, T_audio, D_audio = audio.shape

            #         if T_audio != T_expr:
            #             logger.debug(f"  Interpolating audio from {T_audio} to {T_expr} frames for audio-expr coupling")
            #             # Permute to [B, D, T] for interpolation, then back to [B, T, D]
            #             audio = audio.permute(0, 2, 1)  # [B, 768, T]
            #             audio = F.interpolate(audio, size=T_expr, mode='linear', align_corners=False)
            #             audio = audio.permute(0, 2, 1)  # [B, T_expr, 768]

            #         # Compute audio energy/magnitude
            #         audio_energy = torch.norm(audio, dim=-1, keepdim=True)  # [B, T_expr, 1]
            #         # Normalize to [0, 1]
            #         audio_energy_norm = (audio_energy - audio_energy.min()) / (audio_energy.max() - audio_energy.min() + 1e-8)

            #         # Compute expression magnitude
            #         expr_magnitude = torch.norm(expr, dim=-1, keepdim=True)  # [B, T_expr, 1]
            #         # Normalize to [0, 1]
            #         expr_magnitude_norm = (expr_magnitude - expr_magnitude.min()) / (expr_magnitude.max() - expr_magnitude.min() + 1e-8)

            #         # MSE loss: Expression magnitude should correlate with audio energy
            #         lambda_audio_expr_coupling = getattr(self.config.loss, 'lambda_audio_expr_coupling', 5.0)
            #         audio_expr_coupling_loss = F.mse_loss(expr_magnitude_norm, audio_energy_norm) * lambda_audio_expr_coupling
            #         losses['audio_expression_coupling'] = audio_expr_coupling_loss
            #         logger.debug(f"  Audio-Expression coupling loss: {audio_expr_coupling_loss.item():.6f}")
            #     logger.debug(f"  Motion diversity loss: {diversity_loss.item():.6f}")

            #     # 1.7 Direct GT Expression Matching Loss - CRITICAL for overfitting
            #     if 'expression_embed' in targets:
            #         gt_expr = targets['expression_embed']  # [B, T, D]
            #         pred_expr = outputs['expression_embed']  # [B, T, D]

            #         # Normalize both for cosine similarity
            #         pred_expr_norm = F.normalize(pred_expr, p=2, dim=-1)  # [B, T, D]
            #         gt_expr_norm = F.normalize(gt_expr, p=2, dim=-1)  # [B, T, D]

            #         # Cosine similarity loss: minimize 1 - cosine_similarity
            #         # Higher cosine similarity = better match (closer to 1)
            #         cosine_sim = (pred_expr_norm * gt_expr_norm).sum(dim=-1).mean()  # Scalar
            #         expression_cosine_loss = 1.0 - cosine_sim

            #         # Weight and add to losses
            #         lambda_expression_cosine = getattr(self.config.loss, 'lambda_expression_cosine', 0.0)
            #         if lambda_expression_cosine > 0:
            #             losses['expression_cosine'] = expression_cosine_loss * lambda_expression_cosine
            #             logger.debug(f"  Direct GT expression cosine loss: {expression_cosine_loss.item():.6f} (weighted: {(expression_cosine_loss * lambda_expression_cosine).item():.6f})")
            #             logger.debug(f"  Cosine similarity: {cosine_sim.item():.6f}")

            #             # Also compute L2 distance for monitoring
            #             l2_dist = (pred_expr - gt_expr).pow(2).sum(dim=-1).sqrt().mean()
            #             metrics['expression_gt/l2_distance'] = l2_dist.item()
            #             metrics['expression_gt/cosine_similarity'] = cosine_sim.item()
            #             logger.debug(f"  L2 distance to GT: {l2_dist.item():.6f}")

            #     # Also compute standard deviation as a metric
            #     expr_std = expr.std(dim=-1).mean()
            #     losses['expression_std'] = expr_std  # Just for monitoring
            #     logger.debug(f"  Expression std: {expr_std.item():.6f}")


            # 3. Expression Verification Loss
            logger.debug("\nChecking verification loss conditions:")
            should_compute_verify = (
                current_epoch is not None and 
                current_epoch >= self.config.train.control_start_epoch and  # Start verification with control
                hasattr(self.config.loss, 'use_verification') and 
                self.config.loss.use_verification
            )
            logger.debug(f"Should compute verification loss: {should_compute_verify}")

            if should_compute_verify:
    
                logger.debug("Computing verification loss...")
                try:
                    # Initialize LPIPS for perceptual loss
    

                    # Load reference image
                    data_dir = self.config.paths.data_dir if hasattr(self.config, 'paths') else "data"
                    test_img = Image.open(f"{data_dir}/A.png").convert('RGB')
                    test_tensor = self.transform(test_img).unsqueeze(0).to(device)
                    
                    # Get current expression embed
                    curr_expression = outputs['expression_embed']  # [B, T, 128]
                    B, T = curr_expression.shape[:2]
                    
                    verification_loss = 0.0
                    num_samples = min(T, 1)  # Check up to 4 frames to save compute
                    sample_indices = torch.linspace(0, T-1, num_samples).long()
                    
                    logger.debug(f"Verifying {num_samples} expression samples from sequence")
                    
                    for idx in sample_indices:
                        # Create data dict for current expression
                        source_tensor = test_tensor.repeat(B, 1, 1, 1)
                        source_mask = self.volumetric_avatar.face_idt.forward(source_tensor)[0]
                        source_mask = (source_mask > 0.6).float()
                        
                        data_dict = {
                            'source_img': source_tensor,
                            'source_mask': source_mask,
                            'target_img': source_tensor,
                            'target_mask': source_mask,
                            'source_theta': outputs['theta'][:, idx:idx+1],
                            'target_theta': outputs['theta'][:, idx:idx+1],
                            'idt_embed': self.volumetric_avatar.idt_embedder_nw.forward_image(
                                source_tensor * source_mask
                            )
                        }
                        
                        # Set current expression for verification
                        data_dict['target_pose_embed'] = curr_expression[:, idx:idx+1]
                        
                        # Generate reconstruction through full pipeline
                        frame = self._generate_verification_frame(data_dict)
                        save_image(frame, f"{data_dir}/verification_{idx}.png")

                        # Compute perceptual loss
                        verify_loss = self.loss_fn_alex(frame, source_tensor)
                        verification_loss += verify_loss.mean()
                        
                        logger.debug(f"  Sample {idx} verification loss: {verify_loss.mean().item():.6f}")
                    
                    # Average and scale verification loss
                    verification_loss = verification_loss / num_samples * self.config.loss.lambda_verification
                    losses['verification'] = verification_loss
                    logger.debug(f"Total verification loss: {verification_loss.item():.6f}")

                    # Log visualizations periodically
                    if step is not None and step % 100 == 0:
                        wandb.log({
                            'verification/source': wandb.Image(test_tensor[0].cpu()),
                            'verification/reconstruction': wandb.Image(frame[0].cpu()),
                            'verification/loss': verification_loss.item()
                        }, step=step)

                except Exception as e:
                    logger.error(f"Error in expression verification: {str(e)}")
                    logger.error(traceback.format_exc())
                    losses['verification'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("Skipping verification loss")
                losses['verification'] = torch.tensor(0.0, device=device)

            # 3. Control losses
            logger.debug("\nChecking control loss conditions:")
            logger.debug(f"Current epoch: {current_epoch}")
            logger.debug(f"Control start epoch: {self.config.train.control_start_epoch}")
            
            should_compute_control = (
                current_epoch is not None and 
                current_epoch >= self.config.train.control_start_epoch
            )
            logger.debug(f"Should compute control losses: {should_compute_control}")

            if should_compute_control:
                logger.debug("Computing control losses...")
                control_losses = self._compute_control_losses(
                    generated_frames=generated_frames,
                    conditions=conditions,
                    epoch=current_epoch,
                    pred_motion=outputs,  # Keep for backward compatibility with gaze/distance
                    dataset=dataset
                )
                losses.update(control_losses)
                logger.debug("Control losses:")
                for k, v in control_losses.items():
                    if isinstance(v, torch.Tensor):
                        logger.debug(f"  {k}: {v.item():.6f}")
            else:
                logger.debug("Skipping control losses")
                losses.update(self._get_zero_losses())

            # 4. SyncNet losses
            logger.debug("\nChecking sync loss conditions:")
            logger.debug(f"Use sync loss: {self.config.loss.use_sync_loss}")
            
            if self.config.loss.use_sync_loss and generated_frames is not None:
                logger.debug("Computing sync loss...")
                sync_loss = self._compute_sync_loss(generated_frames, targets)
                losses['sync_loss'] = sync_loss * self.lambda_sync
                logger.debug(f"Sync loss: {losses['sync_loss'].item():.6f}")
            else:
                logger.debug("Skipping sync loss")
                losses['sync_loss'] = torch.tensor(0.0, device=device)

            # 5. VASA-1 Disentanglement losses (Section 3.1)
            logger.debug("\nComputing VASA disentanglement losses:")
            
            # For VASA disentanglement, we need motion outputs with pose and dynamics
            # Check if we have the required outputs
            has_pose = 'theta' in outputs
            has_dynamics = 'expression' in outputs or 'expression_embed' in outputs
            
            # 5. VASA-1 Disentanglement losses (Section 3.1)
            # Check if we should compute disentanglement losses based on frequency
            compute_disentangle = False
            disentangle_freq = getattr(self.config.loss, 'disentangle_compute_freq', 1)
            
            # JP - 🤷 unplug this because is it making the motion predictions blur to sameness - https://wandb.ai/snoozie/vasa-overfitting/runs/9d0di1su?nw=nwusersnoozie
            # if step is not None and step % disentangle_freq == 0:
            #     compute_disentangle = True
            # elif step is None:  # Always compute during validation
            #     compute_disentangle = True
                
            if compute_disentangle and has_pose and has_dynamics:
                # Get source identity from target frames (first frame)
                source_identity = target_frames[:, 0] if target_frames is not None else None
                
                # Pass the correct parameters
                disentangle_loss, l_consist, l_cross_id = self.compute_disentanglement_loss(
                    motion_outputs=outputs,
                    target_frames=target_frames,
                    generated_frames=generated_frames,  # This should be the actual generated frames
                    source_identity=source_identity,     # Pass the source identity explicitly
                    step=step                            # Pass step for visualization
                )
            else:
                # Either skipped due to frequency or missing required outputs
                if not compute_disentangle:
                    logger.debug(f"DISENTANGLE: Skipping (step {step} not divisible by freq {disentangle_freq})")
                else:
                    logger.error(f"DISENTANGLE: Missing required outputs (pose={has_pose}, dynamics={has_dynamics})")
                disentangle_loss = torch.tensor(0.0, device=device)
                l_consist = torch.tensor(0.0, device=device)
                l_cross_id = torch.tensor(0.0, device=device)


            # Add disentanglement losses to both losses dict and metrics
            losses['l_consist'] = l_consist
            losses['l_cross_id'] = l_cross_id
            losses['disentangle_total'] = disentangle_loss
            
            # Also add to metrics for logging
            metrics['l_consist'] = l_consist.item() if torch.is_tensor(l_consist) else l_consist
            metrics['l_cross_id'] = l_cross_id.item() if torch.is_tensor(l_cross_id) else l_cross_id
            
            # 6. Velocity and smoothness regularization
            logger.debug("\nComputing velocity/smoothness losses:")
            vel_smooth_loss = self.compute_velocity_smoothness_loss(outputs, targets)
            losses['velocity_smoothness'] = vel_smooth_loss
            logger.debug(f"Velocity/smoothness loss: {vel_smooth_loss.item():.6f}")

            # Final loss aggregation
            # Log loss weights
            logger.debug("\nLoss weights:")
            logger.debug(f"  lambda_reconstruction: {self.config.loss.lambda_reconstruction}")
            logger.debug(f"  lambda_verification: {self.config.loss.lambda_verification}")
            logger.debug(f"  lambda_control: {self.config.loss.lambda_control}")
            logger.debug(f"  lambda_sync: {self.config.loss.lambda_sync}")
            logger.debug(f"  lambda_consist: {self.lambda_consist}")
            logger.debug(f"  lambda_cross_id: {self.lambda_cross_id}")
            logger.debug(f"  lambda_velocity: {self.lambda_velocity}")
            logger.debug(f"  lambda_smoothness: {self.lambda_smoothness}")

            # Compute total loss
            logger.debug("\nComputing total loss:")
            recon_term = self.config.loss.lambda_reconstruction * losses['reconstruction']
            verify_term = losses['verification']  # Already scaled in computation
            control_term = self.config.loss.lambda_control * losses.get('control_total', torch.tensor(0.0, device=device))
            sync_term = self.lambda_sync * losses.get('sync_loss', torch.tensor(0.0, device=device))
            disentangle_term = losses.get('disentangle_total', torch.tensor(0.0, device=device))
            vel_smooth_term = losses.get('velocity_smoothness', torch.tensor(0.0, device=device))

            # Aggregate warp losses (UV warps is the main one now)
            warp_term = (
                losses.get('uv_warp_loss', torch.tensor(0.0, device=device)) +
                losses.get('uv_warp_smooth', torch.tensor(0.0, device=device)) +
                losses.get('xy_warp_loss', torch.tensor(0.0, device=device)) +
                losses.get('xy_warp_smooth', torch.tensor(0.0, device=device)) +
                losses.get('rigid_warp_loss', torch.tensor(0.0, device=device)) +
                losses.get('rigid_warp_smooth', torch.tensor(0.0, device=device)) +
                losses.get('source_theta_warp_loss', torch.tensor(0.0, device=device)) +
                losses.get('warp_temporal_consistency', torch.tensor(0.0, device=device))
            )

            logger.debug(f"  Reconstruction term: {recon_term.item():.6f}")
            logger.debug(f"  Verification term: {verify_term.item():.6f}")
            logger.debug(f"  Control term: {control_term.item():.6f}")
            logger.debug(f"  Sync term: {sync_term.item():.6f}")
            logger.debug(f"  Disentangle term: {disentangle_term.item():.6f}")
            logger.debug(f"  Velocity/smoothness term: {vel_smooth_term.item():.6f}")
            logger.debug(f"  Warp term (UV + smoothness): {warp_term.item():.6f}")
            
            # Get diversity term
            diversity_term = losses.get('motion_diversity', torch.tensor(0.0, device=device))
            logger.debug(f"  Diversity term: {diversity_term.item():.6f}")
            
            # Compute audio-lip correlation loss
            # DISABLED: Old audio-lip correlation using dataset lip_metrics
            # Moved to control losses section after line 2757 to use extracted landmarks from generated frames
            # This ensures we correlate audio with ACTUAL generated lips, not dataset targets
            audio_lip_term = torch.tensor(0.0, device=device)
            losses['audio_lip_correlation'] = audio_lip_term  # Will be computed later in control losses
            losses['mouth_openness_direct'] = torch.tensor(0.0, device=device)  # Will be computed later

            # Compute LPIPS perceptual loss (VASA paper Section 3.3)
            perceptual_term = torch.tensor(0.0, device=device)
            if generated_frames is not None and target_frames is not None:
                logger.debug("Computing LPIPS perceptual loss...")
                try:
                    # Ensure frames are in correct shape [B*T, C, H, W]
                    if generated_frames.dim() == 5:  # [B, T, C, H, W]
                        B, T, C, H, W = generated_frames.shape
                        gen_flat = generated_frames.view(B * T, C, H, W)
                        tgt_flat = target_frames.view(B * T, C, H, W)
                    else:
                        gen_flat = generated_frames
                        tgt_flat = target_frames
                    
                    # Normalize to [-1, 1] for LPIPS (expects this range)
                    if gen_flat.max() > 1.0:
                        gen_flat = gen_flat / 127.5 - 1.0
                        tgt_flat = tgt_flat / 127.5 - 1.0
                    elif gen_flat.min() >= 0:  # If in [0, 1], convert to [-1, 1]
                        gen_flat = gen_flat * 2.0 - 1.0
                        tgt_flat = tgt_flat * 2.0 - 1.0
                    
                    # Compute LPIPS loss
                    perceptual_loss = self.loss_fn_alex(gen_flat, tgt_flat).mean()

                    # Apply lambda_perceptual weight
                    lambda_perceptual = self.config.loss.get('lambda_perceptual', 0.5)
                    perceptual_term = perceptual_loss * lambda_perceptual
                    losses['perceptual'] = perceptual_term

                    logger.debug(f"  Perceptual loss (LPIPS): {perceptual_loss.item():.6f}")
                    logger.debug(f"  Weighted perceptual term: {perceptual_term.item():.6f}")

                    # Compute mouth-focused perceptual loss (TalkVid style)
                    lambda_mouth_perceptual = self.config.loss.get('lambda_mouth_perceptual', 0.0)
                    if lambda_mouth_perceptual > 0:
                        try:
                            # Extract mouth masks from MediaPipe landmarks
                            mouth_masks = self._extract_mouth_masks(gen_flat, device)

                            # Compute per-pixel LPIPS loss (no reduction)
                            lpips_per_pixel = self.loss_fn_alex(gen_flat, tgt_flat)  # [B*T, 1, H, W]

                            # Apply mouth weighting: mouth_region × 100 + non_mouth × 1
                            # This matches TalkVid: loss *= ((100 - 1) * mouth_mask + 1)
                            weight_map = (lambda_mouth_perceptual - 1) * mouth_masks + 1.0

                            # Weight and average
                            weighted_lpips = (lpips_per_pixel * weight_map).mean()
                            losses['mouth_perceptual'] = weighted_lpips

                            logger.debug(f"  Mouth perceptual loss: {weighted_lpips.item():.6f}")
                        except Exception as mouth_err:
                            logger.warning(f"Could not compute mouth perceptual loss: {mouth_err}")
                            losses['mouth_perceptual'] = torch.tensor(0.0, device=device)
                    else:
                        losses['mouth_perceptual'] = torch.tensor(0.0, device=device)

                except Exception as e:
                    logger.warning(f"Could not compute perceptual loss: {e}")
                    losses['perceptual'] = torch.tensor(0.0, device=device)
                    losses['mouth_perceptual'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No frames provided for perceptual loss")
                losses['perceptual'] = torch.tensor(0.0, device=device)

            # Add audio-expression coupling term
            audio_expr_term = losses.get('audio_expression_coupling', torch.tensor(0.0, device=device))

            # Add mouth-focused perceptual term
            mouth_perceptual_term = losses.get('mouth_perceptual', torch.tensor(0.0, device=device))

            # 10. Flow-DPO Loss (VideoReward framework - Liu et al., 2025)
            flow_dpo_term = torch.tensor(0.0, device=device)

            # 11. Phoneme Prediction Loss (Self-Supervised Auxiliary Task)
            # This helps the Perceiver audio projection explicitly learn phoneme-related features
            # which are critical for lip sync (different phonemes → specific lip shapes)
            # CRITICAL: This loss MUST be computed for proper training
            aux_phoneme_term = torch.tensor(0.0, device=device)

            # HARD ASSERTION: Fail training if phoneme data is missing
            assert 'aux_predictions' in outputs, \
                "❌ FATAL: aux_predictions missing from model outputs! Phoneme loss cannot be computed."

            aux = outputs['aux_predictions']

            assert 'phoneme_pred' in aux, \
                "❌ FATAL: phoneme_pred missing from aux_predictions! Model not configured for phoneme prediction."

            assert 'phoneme_gt' in aux, \
                "❌ FATAL: phoneme_gt missing from aux_predictions! Cache does not have phoneme ground truth. Run ./upsert_phoneme.sh first!"

            phoneme_pred = aux['phoneme_pred']  # [B, num_queries, vocab_size]
            phoneme_gt = aux['phoneme_gt']      # [B, num_queries]

            # Get vocab size from predictions (may be > 50 if using full wav2vec2 vocab)
            vocab_size = phoneme_pred.size(-1)

            # CRITICAL: Clamp phoneme_gt to valid range [0, vocab_size-1]
            # Some phoneme IDs from wav2vec2 may exceed our expected 50 classes
            phoneme_gt_clamped = torch.clamp(phoneme_gt, min=0, max=vocab_size - 1)

            # Log if clamping occurred (indicates vocab size mismatch)
            num_clamped = (phoneme_gt != phoneme_gt_clamped).sum().item()
            if num_clamped > 0:
                logger.warning(f"⚠️ Clamped {num_clamped} phoneme IDs to range [0, {vocab_size-1}]")
                logger.warning(f"   Original range: [{phoneme_gt.min().item()}, {phoneme_gt.max().item()}]")
                logger.warning(f"   This suggests vocab_size mismatch. Consider updating phoneme_head output_dim.")

            # Cross-entropy loss (flatten for CE)
            aux_phoneme_term = F.cross_entropy(
                phoneme_pred.view(-1, vocab_size),  # [B*num_queries, vocab_size]
                phoneme_gt_clamped.view(-1).long()  # [B*num_queries] - ensure long type
            )
            losses['aux_phoneme'] = aux_phoneme_term

            logger.debug(f"  ✅ Auxiliary phoneme loss: {aux_phoneme_term.item():.6f}")

            # ASSERT: Log what keys are actually in targets for debugging
            logger.info(f"🔍 LOSS FUNCTION - Checking targets keys: {sorted(targets.keys())}")

            total_loss = recon_term + verify_term + control_term + sync_term + disentangle_term + vel_smooth_term + warp_term + diversity_term + perceptual_term + mouth_perceptual_term + audio_lip_term + audio_expr_term + flow_dpo_term + aux_phoneme_term * self.lambda_aux_phoneme
            losses['total'] = total_loss
            logger.debug(f"Total loss (with perceptual): {total_loss.item():.6f}")

            # Return results
            if return_metrics:
                metrics.update({k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
                return losses, metrics

            return losses

        except Exception as e:
            logger.error(f"Error in compute_losses: {str(e)}")
            logger.error(f"Outputs keys: {outputs.keys() if outputs else 'None'}")
            logger.error(f"Targets keys: {targets.keys() if targets else 'None'}")
            logger.error(f"Conditions keys: {conditions.keys() if conditions else 'None'}")
            logger.error(f"Current epoch: {current_epoch}, Step: {step}")
            logger.error(traceback.format_exc())
            # Return both losses and metrics to match expected return signature
            # Use self.device as fallback if device not yet defined
            error_device = device if 'device' in locals() else self.device
            error_losses = {'total': torch.tensor(1.0, device=error_device)}
            if return_metrics:
                return error_losses, {}
            return error_losses

        

    def generate_frames_from_motion(
        self,
        motion: Dict[str, torch.Tensor],
        source_image: torch.Tensor,
        max_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate frames from motion parameters using volumetric avatar.
        
        Args:
            motion: Dictionary with 'theta', 'expression', etc.
            source_image: Source identity image [B, C, H, W]
            max_frames: Maximum number of frames to generate (for memory)
            
        Returns:
            Generated frames [B, T, C, H, W]
        """
        try:
            with torch.no_grad():
                B = source_image.shape[0]
                T = motion['theta'].shape[1] if 'theta' in motion else 1
                
                # Limit frames for memory
                if max_frames is not None and T > max_frames:
                    T = max_frames
                    motion = {k: v[:, :T] if v.dim() > 1 else v for k, v in motion.items()}
                
                device = source_image.device
                frames = []
                
                # Get source identity embeddings
                source_mask = self.volumetric_avatar.face_idt.forward(source_image)[0]
                source_mask = (source_mask > 0.6).float()
                source_masked = source_image * source_mask
                
                # Get identity embedding
                idt_embed = self.volumetric_avatar.idt_embedder_nw(source_masked)
                
                # Get canonical volume
                source_latents = self.volumetric_avatar.local_encoder_nw(source_masked)
                c = self.volumetric_avatar.args.latent_volume_channels
                d = self.volumetric_avatar.args.latent_volume_depth
                s = self.volumetric_avatar.args.latent_volume_size
                
                source_volume = source_latents.view(B, c, d, s, s)
                if hasattr(self.volumetric_avatar, 'volume_source_nw'):
                    source_volume = self.volumetric_avatar.volume_source_nw(source_volume)
                canonical_volume = self.volumetric_avatar.volume_process_nw(source_volume)
                
                # Generate each frame
                for t in range(T):
                    # Extract frame parameters
                    frame_theta = motion['theta'][:, t] if 'theta' in motion else torch.eye(3, 4).unsqueeze(0).to(device)
                    frame_expr = motion['expression'][:, t] if 'expression' in motion else torch.zeros(B, 256).to(device)
                    
                    # Create data dict for this frame
                    data_dict = {
                        'source_img': source_image,
                        'target_img': source_image,  # Dummy
                        'source_theta': frame_theta,
                        'target_theta': frame_theta,
                        'source_pose_embed': frame_expr.unsqueeze(1),
                        'target_pose_embed': frame_expr.unsqueeze(1),
                    }
                    
                    # Generate frame through decoder
                    try:
                        # Get embeddings
                        source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = \
                            self.volumetric_avatar.predict_embed(data_dict)
                        
                        # Apply motion warping
                        grid = self.volumetric_avatar.identity_grid_3d.repeat_interleave(B, dim=0)
                        target_rotation_warp = grid.bmm(frame_theta[:, :3].transpose(1, 2)).view(B, d, s, s, 3)
                        
                        # Warp canonical volume
                        warped_volume = self.volumetric_avatar.grid_sample(canonical_volume, target_rotation_warp)
                        target_latent_feats = warped_volume.view(B, c * d, s, s)
                        
                        # Generate frame
                        frame, _, _, _ = self.volumetric_avatar.decoder_nw(
                            data_dict,
                            embed_dict,
                            target_latent_feats,
                            None
                        )
                        frames.append(frame)
                        
                    except Exception as e:
                        logger.warning(f"Error generating frame {t}: {e}")
                        # Use source image as fallback
                        frames.append(source_image)
                
                # Stack frames [B, T, C, H, W]
                if frames:
                    return torch.stack(frames, dim=1)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error in generate_frames_from_motion: {e}")
            return None
    
    def _generate_verification_frame(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper method to generate verification frame through EMO pipeline with memory optimizations."""
        try:
            logger.info("\n=== Starting Memory-Optimized Verification Frame Generation ===")
            
            # 1. Memory Optimization: Clear cache and freeze model
            torch.cuda.empty_cache()
            
            # Store original states to restore later if needed
            training_state = self.volumetric_avatar.training
            
            # Set model to eval mode and disable gradients
            self.volumetric_avatar.eval()
            
            with torch.no_grad():
                # 2. Memory Optimization: Use dimension reduction early
                data_dict['source_theta'] = data_dict['source_theta'].squeeze(1)
                data_dict['target_theta'] = data_dict['target_theta'].squeeze(1)
                data_dict['target_pose_embed'] = data_dict['target_pose_embed'].squeeze(1)
                
                # 3. Memory Optimization: Process in smaller batches if needed
                B = data_dict['source_img'].shape[0]
                if B > 4:
                    logger.info(f"Large batch size {B} detected, processing in chunks")
                    frames = []
                    for i in range(0, B, 4):
                        batch_dict = {k: v[i:i+4] if torch.is_tensor(v) else v 
                                    for k, v in data_dict.items()}
                        frame = self._process_single_batch(batch_dict)
                        frames.append(frame)
                        torch.cuda.empty_cache()
                    result = torch.cat(frames, dim=0)
                else:
                    result = self._process_single_batch(data_dict)
                

                    
                return result

        except Exception as e:

            logger.error(f"Error in frame generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _process_single_batch(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single batch with memory optimizations."""
        try:
            with torch.no_grad():  # Extra safety to ensure no gradients
                # 1. Generate source pose embedding
                source_data = {
                    'source_img': data_dict['source_img'],
                    'source_mask': data_dict['source_mask'],
                    'target_img': data_dict['source_img'],
                    'target_mask': data_dict['source_mask'],
                    'source_theta': data_dict['source_theta'],
                    'target_theta': data_dict['source_theta'],
                    'idt_embed': data_dict['idt_embed']
                }
                
                # 2. Memory Optimization: Process without storing gradients
                source_data = self.volumetric_avatar.expression_embedder_nw(source_data, True, False)
                data_dict['source_pose_embed'] = source_data['source_pose_embed'].detach()
                del source_data
                torch.cuda.empty_cache()
                
                # 3. Predict embeddings without gradient computation
                source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = \
                    self.volumetric_avatar.predict_embed(data_dict)
                
                # Detach embeddings to ensure no gradient history
                embed_dict = {k: v.detach() if torch.is_tensor(v) else v 
                            for k, v in embed_dict.items()}
                
                # 4. Generate UV warp
                target_uv_warp, _ = self.volumetric_avatar.uv_generator_nw(target_warp_embed_dict)
                target_uv_warp = target_uv_warp.detach()
                del target_warp_embed_dict
                torch.cuda.empty_cache()
                
                if self.volumetric_avatar.resize_warp:
                    target_uv_warp = F.avg_pool3d(
                        target_uv_warp.permute(0, 4, 1, 2, 3),
                        kernel_size=self.volumetric_avatar.warp_resize_stride,
                        stride=self.volumetric_avatar.warp_resize_stride
                    ).permute(0, 2, 3, 4, 1)
                
                # 5. Process volume with no gradients
                B = data_dict['source_img'].shape[0]
                source_latents = self.volumetric_avatar.local_encoder_nw(
                    data_dict['source_img'] * data_dict['source_mask']
                ).detach()
                
                source_volume = source_latents.view(B, -1,
                    self.volumetric_avatar.args.latent_volume_depth,
                    self.volumetric_avatar.args.latent_volume_size,
                    self.volumetric_avatar.args.latent_volume_size)
                del source_latents
                torch.cuda.empty_cache()
                
                if self.volumetric_avatar.args.source_volume_num_blocks > 0:
                    source_volume = self.volumetric_avatar.volume_source_nw(source_volume).detach()
                
                canonical_volume = self.volumetric_avatar.volume_process_nw(source_volume).detach()
                del source_volume
                torch.cuda.empty_cache()
                
                # 6. Process grid and rotation
                grid = self.volumetric_avatar.identity_grid_3d.repeat_interleave(B, dim=0)
                rotation_warp = grid.bmm(data_dict['target_theta'][:, :3].transpose(1, 2)).view(
                    B, self.volumetric_avatar.args.latent_volume_depth,
                    self.volumetric_avatar.args.latent_volume_size,
                    self.volumetric_avatar.args.latent_volume_size, 3)
                rotation_warp = rotation_warp.detach()
                del grid
                torch.cuda.empty_cache()
                
                # 7. Grid sampling
                aligned_volume = self.volumetric_avatar.grid_sample(
                    self.volumetric_avatar.grid_sample(canonical_volume, target_uv_warp),
                    rotation_warp
                ).detach()
                del canonical_volume, target_uv_warp, rotation_warp
                torch.cuda.empty_cache()
                
                # 8. Generate final frame
                frame, _, _, _ = self.volumetric_avatar.decoder_nw(
                    data_dict,
                    embed_dict,
                    aligned_volume.view(B, -1,
                        self.volumetric_avatar.args.latent_volume_size,
                        self.volumetric_avatar.args.latent_volume_size),
                    False,
                    stage_two=True
                )

                # Apply green chroma key removal and composite onto gray background
                frame = self._remove_green_background(frame)

                return frame.detach()  # Ensure final output is detached

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _remove_green_background(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Remove green chroma key background and composite onto gray background.
        Matches the implementation in frame_disk_cache.py for consistency.

        Args:
            frame: Generated frame tensor [B, C, H, W] in range [0, 1] or [-1, 1]

        Returns:
            Frame with green background replaced by gray (128/255)
        """
        try:
            import cv2

            # Ensure frame is in correct format and range
            device = frame.device
            B, C, H, W = frame.shape

            # Convert to [B, H, W, C] numpy for OpenCV processing
            frame_np = frame.detach().cpu().numpy()
            frame_np = np.transpose(frame_np, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]

            # Convert to uint8 [0, 255] range
            frame_min, frame_max = frame_np.min(), frame_np.max()
            if frame_min >= 0.0 and frame_max <= 1.0:
                # [0, 1] range
                frame_np = (frame_np * 255).astype(np.uint8)
            elif frame_min >= -1.0 and frame_max <= 1.0:
                # [-1, 1] range
                frame_np = ((frame_np + 1.0) * 127.5).astype(np.uint8)
            else:
                # Normalize to [0, 255]
                frame_np = ((frame_np - frame_min) / (frame_max - frame_min + 1e-8) * 255).astype(np.uint8)

            # Process each batch item
            processed_frames = []
            for b in range(B):
                frame_rgb = frame_np[b]  # [H, W, C]

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Convert to HSV for better green detection
                frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                # Define green color range (matching frame_disk_cache.py)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])

                # Create mask for green pixels
                green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

                # Create alpha channel (255 where NOT green, 0 where green)
                alpha = 255 - green_mask

                # Convert back to RGB
                frame_rgb_out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Create solid gray background (128 = middle gray)
                gray_background = np.full_like(frame_rgb_out, 128, dtype=np.uint8)

                # Composite: blend foreground over gray using alpha
                alpha_float = alpha.astype(np.float32) / 255.0
                alpha_3ch = np.stack([alpha_float, alpha_float, alpha_float], axis=-1)

                # Composite: fg * alpha + bg * (1 - alpha)
                composited = (frame_rgb_out * alpha_3ch + gray_background * (1 - alpha_3ch)).astype(np.uint8)

                processed_frames.append(composited)

            # Stack back to batch
            processed_np = np.stack(processed_frames, axis=0)  # [B, H, W, C]

            # Convert back to torch tensor [B, C, H, W] in [0, 1] range
            processed_tensor = torch.from_numpy(processed_np).float() / 255.0
            processed_tensor = processed_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            return processed_tensor.to(device)

        except Exception as e:
            logger.warning(f"Error in green background removal: {str(e)}, returning original frame")
            return frame

    def _compute_temporal_offset(
        self,
        generated_frames: torch.Tensor,  # [B, T, C, H, W]
        audio_features: torch.Tensor,    # [B, T, D] - can be MFCC or wav2vec features
        window_size: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal offset between audio and video using sliding window correlation.

        Returns:
            offset_magnitude: Magnitude of temporal offset (frames)
            offset_timestamp: Timestamp where maximum misalignment occurs
        """
        # Validate inputs
        if generated_frames.dim() != 5:
            raise ValueError(f"Expected generated_frames to have 5 dims [B, T, C, H, W], got {generated_frames.shape}")

        B, T, C, H, W = generated_frames.shape
        device = generated_frames.device

        # Ensure audio has batch dimension
        if audio_features.dim() == 2:  # [T, D]
            audio_features = audio_features.unsqueeze(0).expand(B, -1, -1)
        elif audio_features.dim() != 3:
            raise ValueError(f"Expected audio_features to have 2 or 3 dims, got {audio_features.shape}")

        # Ensure batch sizes match
        if audio_features.shape[0] != B:
            if audio_features.shape[0] == 1:
                audio_features = audio_features.expand(B, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: frames {B} vs audio {audio_features.shape[0]}")

        # If using Synchformer, it can directly predict offsets
        if isinstance(self.syncnet, SynchformerInstance):
            # Get offset predictions from Synchformer
            logger.info(f"[_compute_temporal_offset] About to call compute_sync_score")
            logger.info(f"[_compute_temporal_offset] generated_frames.shape: {generated_frames.shape}")
            logger.info(f"[_compute_temporal_offset] audio_features.shape: {audio_features.shape}")
            with torch.no_grad():
                logits = self.syncnet.compute_sync_score(
                    generated_frames, audio_features, return_logits=True
                )
                # Synchformer outputs 21 classes for offsets from -2 to +2 seconds
                # Convert logits to offset predictions
                offset_probs = F.softmax(logits, dim=-1)

                # Create offset grid (21 classes from -10 to +10 frames at 25fps)
                offset_grid = torch.linspace(-10, 10, 21, device=device)

                # Weighted average to get predicted offset
                offset_magnitude = (offset_probs * offset_grid).sum(dim=-1)

                # Find timestamp of maximum misalignment (highest entropy in predictions)
                entropy = -(offset_probs * torch.log(offset_probs + 1e-8)).sum(dim=-1)
                offset_timestamp = torch.zeros(B, device=device)  # Simplified: use frame 0

                return offset_magnitude.abs(), offset_timestamp

        # Fallback: Use cross-correlation based offset detection
        offsets = []
        timestamps = []

        for b in range(B):
            max_offset = 0
            max_timestamp = 0
            max_correlation = -float('inf')

            # Slide window through the sequence
            for t in range(0, T - window_size + 1):
                window_frames = generated_frames[b:b+1, t:t+window_size]
                window_audio = audio_features[b:b+1, t:t+window_size]

                # Compute correlation at different offsets
                for offset in range(-2, 3):  # Check offsets from -2 to +2 frames
                    if t + offset < 0 or t + offset + window_size > T:
                        continue

                    offset_audio = audio_features[b:b+1, t+offset:t+offset+window_size]

                    # Simple correlation metric (can be replaced with more sophisticated sync evaluation)
                    with torch.no_grad():
                        if hasattr(self.syncnet, 'evaluate'):
                            _, confidence = self.syncnet.evaluate(
                                window_frames.transpose(1, 2),  # [B, C, T, H, W]
                                offset_audio.unsqueeze(1),      # [B, 1, T, D]
                                batch_size=1
                            )
                            correlation = confidence.item()
                        else:
                            # Simple correlation fallback
                            correlation = 0.0

                    if correlation > max_correlation:
                        max_correlation = correlation
                        max_offset = abs(offset)
                        max_timestamp = t

            offsets.append(max_offset)
            timestamps.append(max_timestamp)

        offset_magnitude = torch.tensor(offsets, dtype=torch.float32, device=device)
        offset_timestamp = torch.tensor(timestamps, dtype=torch.float32, device=device)

        return offset_magnitude, offset_timestamp

    def _compute_sync_loss(
        self,
        generated_frames: torch.Tensor,  # [B, T, C, H, W]
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute sync loss as described in the paper using temporal offset detection.

        L_sync = (Δt_p - Δt_gt)² + (t_p - t_gt)²

        where:
        - Δt_p, Δt_gt: predicted and ground truth temporal offset magnitudes
        - t_p, t_gt: timestamps where misalignment occurs
        """
        try:
            logger.info(f"[_compute_sync_loss] Entry: generated_frames.shape = {generated_frames.shape}")
            logger.info(f"[_compute_sync_loss] Available audio keys in targets: {[k for k in targets.keys() if 'audio' in k.lower()]}")

            # Get audio features - prioritize mel spectrogram for Synchformer (50 frames x 128 mel bins)
            if 'audio_mel_spec' in targets:
                audio_features = targets['audio_mel_spec']
                logger.info(f"✅ Using mel spectrogram for sync loss: {audio_features.shape}")
            elif 'audio_waveform' in targets:
                audio_features = targets['audio_waveform']
                logger.warning(f"⚠️ Using raw audio waveform (will need conversion): {audio_features.shape}")
            elif 'mfcc' in targets:
                audio_features = targets['mfcc']
                logger.warning(f"Using MFCC features for sync loss (not ideal for Synchformer): {audio_features.shape}")
            elif 'audio_mfcc' in targets:
                audio_features = targets['audio_mfcc']
                logger.warning(f"Using audio_mfcc features for sync loss (not ideal for Synchformer): {audio_features.shape}")
            elif 'audio_features' in targets:
                # wav2vec features - not usable with Synchformer
                logger.warning(f"❌ Only wav2vec features available (shape: {targets['audio_features'].shape}), but Synchformer needs raw audio. Skipping sync loss.")
                logger.warning(f"   Available keys: {list(targets.keys())}")
                return torch.tensor(0.0, device=generated_frames.device)
            else:
                logger.warning(f"❌ No audio features in targets, returning zero sync loss. Available keys: {list(targets.keys())}")
                return torch.tensor(0.0, device=generated_frames.device)

            logger.info(f"[_compute_sync_loss] Using audio with shape = {audio_features.shape}")

            # Get ground truth frames if available
            gt_frames = targets.get('frames', None)
            if gt_frames is not None:
                logger.info(f"[_compute_sync_loss] gt_frames.shape = {gt_frames.shape}")

            # Ensure proper shapes
            B, T = generated_frames.shape[:2]
            logger.info(f"[_compute_sync_loss] Extracted B={B}, T={T} from generated_frames")

            # Only compute sync loss if we have enough frames
            if T < 5:
                logger.debug(f"Not enough frames for sync loss (T={T})")
                return torch.tensor(0.0, device=generated_frames.device)

            # Ensure audio features have the right shape [B, T, D]
            if audio_features.dim() == 2:  # [T, D]
                audio_features = audio_features.unsqueeze(0)  # Add batch dim
            elif audio_features.dim() == 4:  # [B, 1, T, D]
                audio_features = audio_features.squeeze(1)  # Remove channel dim

            # Compute temporal offsets for generated frames
            delta_t_pred, t_pred = self._compute_temporal_offset(
                generated_frames, audio_features
            )

            # Compute temporal offsets for ground truth if available
            if gt_frames is not None:
                delta_t_gt, t_gt = self._compute_temporal_offset(
                    gt_frames, audio_features
                )
            else:
                # If no ground truth, assume perfect sync (offset=0, timestamp=0)
                delta_t_gt = torch.zeros_like(delta_t_pred)
                t_gt = torch.zeros_like(t_pred)

            # Compute sync loss as per the paper equation:
            # L_sync = (Δt_p - Δt_gt)² + (t_p - t_gt)²
            offset_loss = F.mse_loss(delta_t_pred, delta_t_gt)
            timestamp_loss = F.mse_loss(t_pred, t_gt)

            # Normalize timestamp loss by sequence length to make it scale-invariant
            timestamp_loss = timestamp_loss / T

            # Combine the two components
            sync_loss = offset_loss + timestamp_loss

            logger.debug(f"Sync loss - Offset: {offset_loss.item():.4f}, Timestamp: {timestamp_loss.item():.4f}, Total: {sync_loss.item():.4f}")
            
            return sync_loss
            
        except Exception as e:
            logger.error(f"Error computing sync loss: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=generated_frames.device)
        
    def evaluate_sync_quality(
        self,
        generated_frames: torch.Tensor,    # [B, T, C, H, W]
        audio_features: torch.Tensor,      # [B, T, D] or [B, 1, T, D]
        audio_mfcc: torch.Tensor,         # [B, T, 13] MFCC features for SyncNet
        window_size: int = 5              # Size of evaluation window
    ) -> Dict[str, float]:
        """
        Evaluate sync quality using SyncNet with MFCC features.
        
        Args:
            generated_frames: Generated video frames
            audio_features: Original wav2vec features (not used by SyncNet)
            audio_mfcc: MFCC features for SyncNet
            window_size: Size of sliding window
            
        Returns:
            Dictionary of sync quality metrics
        """
        try:
            logger.debug("\n=== Evaluating Sync Quality ===")
            logger.debug(f"Generated frames shape: {generated_frames.shape}")
            logger.debug(f"Audio MFCC shape: {audio_mfcc.shape}")
            
            # Get batch size and verify inputs
            B, T = generated_frames.shape[:2]
            device = generated_frames.device
            logger.debug(f"Processing batch size: {B}, sequence length: {T}")
            
            # Move tensors to appropriate device
            audio_mfcc = audio_mfcc.to(device)
            
            # Initialize metrics
            metrics = {
                'avg_sync_confidence': 0.0,
                'avg_sync_offset': 0.0,
                'min_confidence': float('inf'),
                'max_confidence': float('-inf')
            }
            
            # Process sequence in windows
            all_confidences = []
            all_offsets = []
            
            for start_idx in range(0, T - window_size + 1, window_size):
                try:
                    end_idx = min(start_idx + window_size, T)
                    
                    # Get window tensors
                    frame_window = generated_frames[:, start_idx:end_idx]
                    mfcc_window = audio_mfcc[:, start_idx:end_idx]
                    
                    # Evaluate sync for this window
                    offset, confidence = self.syncnet.evaluate(
                        frames=frame_window,
                        audio_features=mfcc_window,
                        batch_size=self.batch_size
                    )
                    
                    all_confidences.append(confidence)
                    all_offsets.append(offset)
                    
                except Exception as e:
                    logger.error(f"Error evaluating window {start_idx}:{end_idx}: {str(e)}")
                    continue
            
            # Aggregate metrics
            if all_confidences:
                confidences = torch.stack(all_confidences)
                metrics.update({
                    'avg_sync_confidence': confidences.mean().item(),
                    'avg_sync_offset': float(sum(all_offsets)) / len(all_offsets),
                    'min_confidence': confidences.min().item(),
                    'max_confidence': confidences.max().item()
                })
            
            logger.debug("Evaluation complete")
            logger.debug(f"Final metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in sync quality evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'avg_sync_confidence': 0.0,
                'avg_sync_offset': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }

    def evaluate_window(
        self,
        frames: torch.Tensor,      # [B, T, C, H, W]
        audio: torch.Tensor,       # [B, T, D] or [B, 1, T, D]
        batch_size: int = 20
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Evaluate synchronization for a single window of frames and audio.
        
        Args:
            frames: Video frames for the window
            audio: Audio features for the window
            batch_size: Processing batch size
            
        Returns:
            Dictionary containing offset and confidence
        """
        try:
            logger.debug("\n=== Evaluating Window ===")
            logger.debug(f"Window frames shape: {frames.shape}")
            logger.debug(f"Window audio shape: {audio.shape}")
            
            # Ensure tensors are on same device
            device = frames.device
            audio = audio.to(device)
            
            # Prepare frames
            if len(frames.shape) == 5:  # [B, T, C, H, W]
                frames = frames.transpose(1, 2)  # -> [B, C, T, H, W]
            
            # Prepare audio
            if len(audio.shape) == 3:  # [B, T, D]
                audio = audio.unsqueeze(1)  # -> [B, 1, T, D]
            
            # Get predictions from SyncNet
            offset, confidence = self.syncnet.evaluate(
                frames=frames,
                audio_features=audio,
                batch_size=batch_size
            )
            
            logger.debug(f"Window evaluation results:")
            logger.debug(f"  Offset: {offset}")
            logger.debug(f"  Confidence: {confidence.mean().item():.4f}")
            
            return {
                'offset': offset,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error evaluating window: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'offset': 0.0,
                'confidence': torch.zeros(1, device=frames.device)
            }



    def _compute_reconstruction_losses(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        try:
            logger.debug("\n=== Computing Reconstruction Losses ===")
            
            # Get device and initialize losses
            device = pred['theta'].device
            losses = {}
            
            # VASA-1 style score matching: predict clean signal directly
            use_score_matching = getattr(self.config.loss, 'use_score_matching', True)
            
            if use_score_matching and noise is not None:
                # VASA-1: Direct MSE on clean signal prediction ||X0 - H(Xt, t, C)||^2
                comparison_target = target  # Compare with clean target, not noise
                is_training = True
                logger.debug(f"\nMode: VASA-1 Score Matching (predicting X0)")
                
                # Optional: Add timestep weighting for early denoising emphasis
                timestep_weight = 1.0
                if step is not None:
                    # Weight early timesteps more (when t is small, denoising is crucial)
                    max_steps = getattr(self, 'num_steps', 1000)
                    timestep_weight = 1.0 + (1.0 - step / max_steps) * 0.5  # 1.5x weight at t=0
                    logger.debug(f"Timestep weight: {timestep_weight:.3f} (step {step}/{max_steps})")
            else:
                # Original noise prediction mode
                comparison_target = noise if noise is not None else target
                is_training = noise is not None
                timestep_weight = 1.0
                logger.debug(f"\nMode: {'training (noise prediction)' if is_training else 'validation'}")


            # 1. Theta (pose matrix) loss
            # Skip if using identity theta (no theta prediction/loss in this mode)
            use_identity_theta = self.config.dataset.get('use_identity_theta', False)
            use_gt_theta = getattr(self.config.train, 'use_gt_theta', False)
            if 'theta' in pred and not use_identity_theta and not use_gt_theta:
                if is_training:
                    # During training, compare predicted noise to target noise
                    pred_flat = pred['theta'].view(pred['theta'].shape[0], -1, 12)
                    target_flat = comparison_target['theta'].view(comparison_target['theta'].shape[0], -1, 12)
                    losses['theta_loss'] = F.mse_loss(pred_flat, target_flat)
                else:
                    # During validation, use pose matrix loss
                    losses['theta_loss'] = self._compute_pose_matrix_loss(
                        pred['theta'],
                        target['theta']
                    )

            # SRT losses - compare predicted scale, rotation, translation to ground truth
            # Skip SRT losses if using GT values (no point computing loss on GT)
            use_gt_scale = getattr(self.config.train, 'use_gt_scale', False)
            use_gt_rotation = getattr(self.config.train, 'use_gt_rotation', False)
            use_gt_translation = getattr(self.config.train, 'use_gt_translation', False)

            if 'scale' in pred and 'scale' in target and not use_gt_scale:
                losses['scale_loss'] = F.mse_loss(pred['scale'], target['scale'])

            if 'rotation' in pred and 'rotation' in target and not use_gt_rotation:
                losses['rotation_loss'] = F.mse_loss(pred['rotation'], target['rotation'])

            if 'translation' in pred and 'translation' in target and not use_gt_translation:
                losses['translation_loss'] = F.mse_loss(pred['translation'], target['translation'])

            # 5. Expression loss with variance preservation
            if 'expression_embed' in pred:
                # L1 loss for expression embeddings (prevents collapse, encourages sparsity)
                losses['expression_loss'] = F.l1_loss(
                    pred['expression_embed'],
                    comparison_target['expression_embed']
                ) * self.lambda_expression_l1

                # Also add MSE for smoothness
                losses['expression_mse'] = F.mse_loss(
                    pred['expression_embed'],
                    comparison_target['expression_embed']
                ) * (1.0 - self.lambda_expression_l1)  # Balance between L1 and L2

                # Add variance preservation loss to prevent collapse
                pred_std = pred['expression_embed'].std(dim=-1).mean()  # Std across features, mean across batch/time
                target_std = comparison_target['expression_embed'].std(dim=-1).mean()
                variance_loss = F.mse_loss(pred_std, target_std)
                lambda_expression_variance = getattr(self.config.loss, 'lambda_expression_variance', 5.0)
                losses['expression_variance_loss'] = variance_loss * lambda_expression_variance

                # Add temporal variation loss to encourage dynamics
                if pred['expression_embed'].shape[1] > 1:  # If we have temporal dimension
                    pred_temporal_diff = (pred['expression_embed'][:, 1:] - pred['expression_embed'][:, :-1]).abs().mean()
                    target_temporal_diff = (comparison_target['expression_embed'][:, 1:] - comparison_target['expression_embed'][:, :-1]).abs().mean()
                    temporal_loss = F.mse_loss(pred_temporal_diff, target_temporal_diff)
                    lambda_expression_temporal = getattr(self.config.loss, 'lambda_expression_temporal', 2.0)
                    losses['expression_temporal_loss'] = temporal_loss * lambda_expression_temporal

                # Log statistics to detect collapse (every step for debugging)
                pred_std_scalar = pred['expression_embed'].std().item()
                target_std_scalar = comparison_target['expression_embed'].std().item()
                pred_mean = pred['expression_embed'].mean().item()
                target_mean = comparison_target['expression_embed'].mean().item()

                # Log every step
                logger.info(f"Expression stats - Pred: mean={pred_mean:.3f}, std={pred_std_scalar:.3f} | Target: mean={target_mean:.3f}, std={target_std_scalar:.3f}")
                logger.info(f"  Variance loss: {losses['expression_variance_loss'].item():.6f}, Temporal loss: {losses.get('expression_temporal_loss', torch.tensor(0.0)).item():.6f}")

                # Warn if prediction variance is collapsing
                if pred_std_scalar < target_std_scalar * 0.5:
                    logger.warning(f"⚠️ Prediction variance collapse detected! pred_std={pred_std_scalar:.3f} << target_std={target_std_scalar:.3f}")
         
                should_visualize = step is not None and step > 0 and step % self.vis_freq == 0
                if should_visualize:
                    # Use comparison_target which is the actual target being used for loss
                    self.visualize_sequence(pred['expression_embed'], comparison_target['expression_embed'], step, 0)


            # Apply timestep weighting for VASA-1 score matching
            if use_score_matching and timestep_weight != 1.0:
                for k in losses:
                    losses[k] = losses[k] * timestep_weight

            # Ensure losses have gradients when needed
            if is_training:
                for k, v in losses.items():
                    if not v.requires_grad:
                        losses[k] = v.clone().requires_grad_(True)

            # Combine into major loss components with proper scaling
            # Direct theta loss (no wrapper)
            theta_loss = losses.get('theta_loss', torch.tensor(0.0, device=device)) * self.lambda_pose

            # SRT losses
            scale_loss = losses.get('scale_loss', torch.tensor(0.0, device=device)) * self.lambda_scale
            rotation_loss = losses.get('rotation_loss', torch.tensor(0.0, device=device)) * self.lambda_rotation
            translation_loss = losses.get('translation_loss', torch.tensor(0.0, device=device)) * self.lambda_translation

            # Aggregate all expression-related losses
            expression_total = losses.get('expression_loss', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_mse', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_variance_loss', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_temporal_loss', torch.tensor(0.0, device=device))

            dynamics_loss = expression_total * self.lambda_dynamics


            # Motion smoothness loss if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            seq_len = pred.get('theta', pred.get('expression_embed')).shape[1] if pred else 0
            if seq_len > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred) * self.lambda_temporal

            # Combined reconstruction loss (theta + SRT + dynamics + motion)
            reconstruction_loss = theta_loss + scale_loss + rotation_loss + translation_loss + dynamics_loss + motion_loss

            # Return all losses including SRT
            return {
                'reconstruction': reconstruction_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': theta_loss,
                'scale_loss': scale_loss,
                'rotation_loss': rotation_loss,
                'translation_loss': translation_loss,
                'expression_loss': losses.get('expression_loss', torch.tensor(0.0, device=device))
            }

        except Exception as e:
            logger.error(f"Error computing reconstruction losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device, requires_grad=True),
                'dynamics_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'motion_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'theta_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'expression_loss': torch.tensor(0.0, device=device, requires_grad=True)
            }
        
    def compute_motion_losses(
        self,
        pred_motion: Dict[str, torch.Tensor],
        target_motion: Dict[str, torch.Tensor],
        noise: Optional[Dict[str, torch.Tensor]] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all motion parameters with visualization and wandb logging.
        """
        try:
            B = pred_motion['theta'].shape[0]
            device = pred_motion['theta'].device
            losses = {}

            # Determine if we're in training (noise prediction) or validation mode
            is_training = noise is not None
            comparison_target = noise if is_training else target_motion
            mode_prefix = 'train' if is_training else 'val'
            
            logger.debug(f"\nMode: {'training' if is_training else 'validation'}")

            # Process predictions and targets (SRT removed - only theta and expression)
            param_dims = {
                'theta': 12,      # 3x4 matrix flattened
                'expression_embed': 128
            }

            # Initialize losses and logging metrics
            losses = {f'{param}_loss': torch.tensor(0.0, device=device) for param in param_dims.keys()}
            metrics = {}

            # Process each parameter and compute losses
            # Skip theta if using identity theta mode
            use_identity_theta = self.config.dataset.get('use_identity_theta', False)

            for param, dim in param_dims.items():
                try:
                    if param == 'theta':
                        # Skip theta loss if using identity theta
                        if use_identity_theta:
                            losses[f'{param}_loss'] = torch.tensor(0.0, device=device)
                            continue

                        pred_flat = pred_motion[param].view(B, -1, 12)
                        target_flat = comparison_target[param].view(B, -1, 12)

                        if not is_training:
                            losses[f'{param}_loss'] = self._compute_pose_matrix_loss(
                                pred_motion[param],
                                comparison_target[param]
                            )
                        else:
                            losses[f'{param}_loss'] = F.mse_loss(pred_flat, target_flat)

                        # Log theta statistics
                        if wandb.run is not None:
                            metrics.update({
                                f'{mode_prefix}/theta/mean': pred_flat.mean().item(),
                                f'{mode_prefix}/theta/std': pred_flat.std().item(),
                                f'{mode_prefix}/theta/min': pred_flat.min().item(),
                                f'{mode_prefix}/theta/max': pred_flat.max().item(),
                                f'{mode_prefix}/theta/loss': losses[f'{param}_loss'].item()
                            })

                    elif param == 'expression_embed':
                        losses['expression_loss'] = F.mse_loss(
                            pred_motion[param],
                            comparison_target[param]
                        )

                        # Log expression statistics
                        if wandb.run is not None:
                            expr_pred = pred_motion[param]
                            metrics.update({
                                f'{mode_prefix}/expression/mean': expr_pred.mean().item(),
                                f'{mode_prefix}/expression/std': expr_pred.std().item(),
                                f'{mode_prefix}/expression/min': expr_pred.min().item(),
                                f'{mode_prefix}/expression/max': expr_pred.max().item(),
                                f'{mode_prefix}/expression/loss': losses['expression_loss'].item()
                            })

                except Exception as e:
                    logger.error(f"Error processing {param}: {str(e)}")
                    continue

            # Compute combined losses (direct theta loss, no wrapper)
            theta_loss = losses['theta_loss'] * self.lambda_pose

            # Aggregate all expression-related losses (same as compute_reconstruction_loss)
            expression_total = losses.get('expression_loss', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_mse', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_variance_loss', torch.tensor(0.0, device=device))
            expression_total += losses.get('expression_temporal_loss', torch.tensor(0.0, device=device))

            dynamics_loss = expression_total * self.lambda_dynamics

            # Compute motion smoothness if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            if pred_motion['theta'].shape[1] > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred_motion)

            # Combined reconstruction loss
            reconstruction_loss = theta_loss + dynamics_loss + motion_loss

            # Log combined losses and detailed parameter statistics
           # Log combined losses and detailed parameter statistics
            if wandb.run is not None:
                # Base losses
                metrics.update({
                    f'{mode_prefix}/loss/theta': theta_loss.item(),
                    f'{mode_prefix}/loss/dynamics': dynamics_loss.item(),
                    f'{mode_prefix}/loss/motion': motion_loss.item(),
                    f'{mode_prefix}/loss/total': reconstruction_loss.item()
                })
                
                # Add detailed statistics for core motion parameters (SRT removed)
                core_params = ['theta', 'expression_embed']
                for param in core_params:
                    if param in pred_motion and param in comparison_target:
                        try:
                            # Safely process tensors
                            pred_tensor = pred_motion[param]
                            target_tensor = comparison_target[param]
                            
                            # Ensure tensors are detached and on CPU
                            pred_np = pred_tensor.detach().cpu().numpy()
                            target_np = target_tensor.detach().cpu().numpy()
                            
                            # Compute differences and statistics
                            diff = pred_np - target_np
                            
                            # Handle potential NaN or Inf values
                            max_diff = np.nan_to_num(np.abs(diff).max())
                            mean_diff = np.nan_to_num(np.abs(diff).mean())
                            
                            # Safely compute correlation
                            try:
                                correlation = np.nan_to_num(np.corrcoef(
                                    pred_np.flatten(), 
                                    target_np.flatten()
                                )[0,1])
                            except ValueError:
                                correlation = 0.0
                            
                            metrics.update({
                                f'{mode_prefix}/{param}/max_diff': float(max_diff),
                                f'{mode_prefix}/{param}/mean_diff': float(mean_diff),
                                f'{mode_prefix}/{param}/correlation': float(correlation),
                                f'{mode_prefix}/{param}/pred_range_min': float(np.nan_to_num(pred_np.min())),
                                f'{mode_prefix}/{param}/pred_range_max': float(np.nan_to_num(pred_np.max())),
                                f'{mode_prefix}/{param}/target_range_min': float(np.nan_to_num(target_np.min())),
                                f'{mode_prefix}/{param}/target_range_max': float(np.nan_to_num(target_np.max()))
                            })
                        except Exception as e:
                            logger.error(f"Error processing statistics for {param}: {str(e)}")
                            continue

                # Log all metrics to wandb
                wandb.log(metrics, step=step)

            # Return dictionary of all losses (pose_loss removed)
            return {
                'reconstruction': reconstruction_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': theta_loss,
                'expression_loss': losses['expression_loss']
            }

        except Exception as e:
            logger.error(f"Error computing motion losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device),
                'dynamics_loss': torch.tensor(0.0, device=device),
                'motion_loss': torch.tensor(0.0, device=device),
                'theta_loss': torch.tensor(0.0, device=device),
                'expression_loss': torch.tensor(0.0, device=device)
            }
        
    def _compute_motion_smoothness_loss(
            self, 
            pred: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            """
            Compute temporal smoothness loss for motion parameters.
            
            Args:
                pred: Dictionary of predicted motion parameters
                
            Returns:
                Motion smoothness loss
            """
            loss = 0.0

            # Compute velocity (first derivative) - only for theta and expression (SRT removed)
            velocities = {}
            accelerations = {}

            if 'theta' in pred:
                velocities['theta'] = torch.diff(pred['theta'], dim=1)
            if 'expression_embed' in pred:
                velocities['expression'] = torch.diff(pred['expression_embed'], dim=1)

            # Compute acceleration (second derivative) for available components
            T = pred.get('theta', pred.get('expression_embed')).shape[1]
            if T > 2:
                for key, vel in velocities.items():
                    accelerations[key] = torch.diff(vel, dim=1)
            else:
                for key, vel in velocities.items():
                    accelerations[key] = torch.zeros_like(vel)

            # Velocity smoothness - only for available components
            for key, vel in velocities.items():
                loss += F.mse_loss(vel, torch.zeros_like(vel))

            # Acceleration smoothness - only for available components
            for key, acc in accelerations.items():
                loss += 0.5 * F.mse_loss(acc, torch.zeros_like(acc))

            return loss


    def create_expression_comparison_plot(self, pred: torch.Tensor, target: torch.Tensor, step: int, frame_losses=None):
        """Create visualization comparing predicted and target expressions with normalized difference plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from PIL import Image
        
        # Ensure we're working with numpy arrays on CPU
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Handle sequence vs single frame
        if len(pred_np.shape) > 1 and pred_np.shape[0] > 1:
            logger.debug(f"Got sequence of shape {pred_np.shape}, extracting first frame")
            pred_np = pred_np[0]
            target_np = target_np[0]
        
        # Ensure we have shape [128]
        pred_np = pred_np.reshape(-1)[:128]
        target_np = target_np.reshape(-1)[:128]
        
        # Compute normalized difference relative to the value range
        value_range = max(pred_np.max(), target_np.max()) - min(pred_np.min(), target_np.min())
        diff = (pred_np - target_np) / (value_range + 1e-8)  # Normalize by value range
        
        # Find global min/max for consistent scale
        global_min = min(pred_np.min(), target_np.min())
        global_max = max(pred_np.max(), target_np.max())
        
        # Create figure with plots
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5])
        
        # Plot predicted embedding
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.imshow(pred_np.reshape(1, -1), aspect='auto', 
                        cmap='viridis', vmin=global_min, vmax=global_max)
        ax1.set_title(f'Predicted Expression Embedding (Step {step})')
        plt.colorbar(im1, ax=ax1)
        ax1.set_yticks([])
        ax1.set_xlabel('Dimension (128)')
        
        # Plot target embedding
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.imshow(target_np.reshape(1, -1), aspect='auto',
                        cmap='plasma', vmin=global_min, vmax=global_max)
        ax2.set_title('Target Expression Embedding')
        plt.colorbar(im2, ax=ax2)
        ax2.set_yticks([])
        ax2.set_xlabel('Dimension (128)')
        
        # Plot normalized difference with white-centered colormap
        ax3 = fig.add_subplot(gs[2])
        # Use smaller scale for difference to make it more sensitive
        diff_scale = 0.2  # This means differences > 20% of value range will be full color
        im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto',
                        cmap='RdGy_r', vmin=-diff_scale, vmax=diff_scale)
        ax3.set_title('Normalized Difference (Pred - Target) / Range')
        plt.colorbar(im3, ax=ax3)
        ax3.set_yticks([])
        ax3.set_xlabel('Dimension (128)')
        
        # Plot value distributions
        ax4 = fig.add_subplot(gs[3])
        bins = np.linspace(global_min, global_max, 50)
        n_pred, _, _ = ax4.hist(pred_np.flatten(), bins=bins, alpha=0.5, 
                            label='Predicted', density=True, color='cornflowerblue')
        n_target, _, _ = ax4.hist(target_np.flatten(), bins=bins, alpha=0.5,
                                label='Target', density=True, color='orange')
        ax4.set_title('Value Distributions')
        ax4.legend()
        ax4.grid(True)
        
        # Compute correlation
        correlation = np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]
        
        # Add statistics with normalized differences
        stats_text = (
            f'Max Diff (% of range): {np.abs(diff).max()*100:.2f}%\n'
            f'Mean Diff (% of range): {np.abs(diff).mean()*100:.2f}%\n'
            f'Correlation: {correlation:.4f}\n'
            f'Value Range (Pred): [{pred_np.min():.2f}, {pred_np.max():.2f}]\n'
            f'Value Range (Target): [{target_np.min():.2f}, {target_np.max():.2f}]'
        )
        
        # Add frame losses if provided
        if frame_losses is not None:
            if isinstance(frame_losses, torch.Tensor):
                frame_losses_np = frame_losses.detach().cpu().numpy()
                stats_text += f'\nMean Frame Loss: {frame_losses_np.mean():.4f}'
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
        plt.tight_layout()

        # Convert to wandb image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return wandb.Image(Image.open(buf))

    def create_motion_comparison_plot(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        frame_idx: int,
        step: int,
        is_noise: bool = False
    ) -> wandb.Image:
        """
        Create visualization comparing predicted and target motion parameters for a single frame.
        
        Args:
            pred: Dictionary of predicted motion tensors
            target: Dictionary of target motion tensors
            frame_idx: Index of frame to visualize
            step: Current training step
            is_noise: Whether we're visualizing noise prediction
            
        Returns:
            wandb.Image with the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            from PIL import Image

            # Extract frame data for each parameter - only include components that exist
            param_data = {}

            if 'theta' in pred:
                param_data['theta'] = {
                    'pred': pred['theta'][:, frame_idx].reshape(-1),
                    'target': target['theta'][:, frame_idx].reshape(-1),
                    'title': 'Pose Matrix (θ)',
                    'dim': 12
                }

            if 'scale' in pred:
                param_data['scale'] = {
                    'pred': pred['scale'][:, frame_idx],
                    'target': target['scale'][:, frame_idx],
                    'title': 'Scale',
                    'dim': 3
                }

            if 'rotation' in pred:
                param_data['rotation'] = {
                    'pred': pred['rotation'][:, frame_idx],
                    'target': target['rotation'][:, frame_idx],
                    'title': 'Rotation',
                    'dim': 3
                }

            if 'translation' in pred:
                param_data['translation'] = {
                    'pred': pred['translation'][:, frame_idx],
                    'target': target['translation'][:, frame_idx],
                    'title': 'Translation',
                    'dim': 3
                }

            if 'expression_embed' in pred:
                param_data['expression'] = {
                    'pred': pred['expression_embed'][:, frame_idx],
                    'target': target['expression_embed'][:, frame_idx],
                    'title': 'Expression',
                    'dim': 128
                }

            # Create figure with subplots for each parameter
            fig = plt.figure(figsize=(20, 25))
            gs = plt.GridSpec(len(param_data), 4, height_ratios=[1] * len(param_data))
            
            # Create title for the entire figure
            prefix = 'Predicted Noise' if is_noise else 'Predicted'
            fig.suptitle(f'{prefix} vs Target Motion Parameters (Frame {frame_idx}, Step {step})', 
                        fontsize=16, y=0.95)

            # Process each parameter
            for i, (param_name, data) in enumerate(param_data.items()):
                # Convert tensors to numpy
                pred_np = data['pred'].detach().cpu().numpy()
                target_np = data['target'].detach().cpu().numpy()
                
                # Compute difference
                diff = pred_np - target_np
                
                # Find global min/max for consistent scale
                global_min = min(pred_np.min(), target_np.min())
                global_max = max(pred_np.max(), target_np.max())
                
                # Create parameter title text
                param_title = f"{data['title']} (dim={data['dim']})"
                
                # 1. Plot predicted values
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.imshow(pred_np.reshape(1, -1), aspect='auto',
                            cmap='viridis', vmin=global_min, vmax=global_max)
                ax1.set_title(f'Predicted {param_title}')
                plt.colorbar(im1, ax=ax1)
                ax1.set_yticks([])
                
                # 2. Plot target values
                ax2 = fig.add_subplot(gs[i, 1])
                im2 = ax2.imshow(target_np.reshape(1, -1), aspect='auto',
                            cmap='plasma', vmin=global_min, vmax=global_max)
                ax2.set_title(f'Target {param_title}')
                plt.colorbar(im2, ax=ax2)
                ax2.set_yticks([])
                
                # 3. Plot difference
                max_diff = max(abs(diff.min()), abs(diff.max()))
                ax3 = fig.add_subplot(gs[i, 2])
                im3 = ax3.imshow(diff.reshape(1, -1), aspect='auto',
                            cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
                ax3.set_title('Difference (Predicted - Target)')
                plt.colorbar(im3, ax=ax3)
                ax3.set_yticks([])
                
                # 4. Plot distributions
                ax4 = fig.add_subplot(gs[i, 3])
                bins = np.linspace(global_min, global_max, 50)
                ax4.hist(pred_np.flatten(), bins=bins, alpha=0.5,
                        label='Predicted', density=True, color='cornflowerblue')
                ax4.hist(target_np.flatten(), bins=bins, alpha=0.5,
                        label='Target', density=True, color='orange')
                ax4.set_title('Value Distributions')
                ax4.legend()
                ax4.grid(True)
                
                # Add statistics text for this parameter
                stats_text = (
                    f'Max Diff: {np.abs(diff).max():.4f}\n'
                    f'Mean Diff: {np.abs(diff).mean():.4f}\n'
                    f'Correlation: {np.corrcoef(pred_np.flatten(), target_np.flatten())[0,1]:.4f}\n'
                    f'Value Range (Pred): [{pred_np.min():.2f}, {pred_np.max():.2f}]\n'
                    f'Value Range (Target): [{target_np.min():.2f}, {target_np.max():.2f}]'
                )
                ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Convert to wandb image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plt.close(fig)
            return wandb.Image(Image.open(buf))

        except Exception as e:
            logger.error(f"Error creating motion comparison plot: {str(e)}")
            logger.error(traceback.format_exc())
            plt.close()  # Ensure figure is closed even on error
            return None

    def visualize_sequence(
        self, 
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        step: int,
        is_noise: bool = False,
        b:int = 0 
    ) -> None:
        """
        Create visualizations for a sequence of frames.
        
        Args:
            pred: Dictionary of predicted motion tensors
            target: Dictionary of target motion tensors
            step: Current training step
            is_noise: Whether we're visualizing noise prediction
            max_frames: Maximum number of frames to visualize
        """
        try:
          
          
            wandb_img = self.create_expression_comparison_plot(
                pred[b].detach(),
                target[b].detach(),
                step,
                b
            )
            wandb.log({
                f"expression/comparison_batch_{b}": wandb_img,
            }, step=step)
            
                
        except Exception as e:
            logger.error(f"Error in sequence visualization: {str(e)}")
            logger.error(traceback.format_exc())

    def _extract_features_from_frames(
        self,
        generated_frames: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        device: torch.device,
        dataset=None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features from generated frames for control loss computation.

        Extracts:
        - Landmarks (lips, eyes, jaw, nose) via MediaPipe
        - Blink states via Eye Aspect Ratio (EAR)
        - Emotions via HSEmotion recognizer
        - Gaze via L2CS

        Args:
            generated_frames: [B, T, C, H, W] generated video frames
            conditions: Dictionary of conditioning signals (to check what to extract)
            device: Device for tensor operations
            dataset: VASAIntegratedDataset to access extractors from (required)

        Returns:
            Dictionary with extracted features:
                - 'landmarks': dict of {key: tensor[T_sampled, N_points, 3]}
                - 'blink_states': tensor[T_sampled, 3]
                - 'emotions': tensor[T_sampled, emotion_dim]
                - 'gazes': tensor[T_sampled, 2]
        """
        extracted = {}

        if dataset is None:
            logger.warning("No dataset provided to _extract_features_from_frames, cannot extract features")
            return extracted

        try:
            # Log incoming generated_frames stats
            B, T = generated_frames.shape[:2]
            frames_min = generated_frames.min().item()
            frames_max = generated_frames.max().item()
            frames_mean = generated_frames.mean().item()
            frames_std = generated_frames.std().item()

            logger.info(f"📥 Input generated_frames: shape={generated_frames.shape}, min={frames_min:.3f}, max={frames_max:.3f}, mean={frames_mean:.3f}, std={frames_std:.3f}")

            if frames_std < 0.01:
                logger.error(f"🚨 Generated frames have VERY LOW variance ({frames_std:.5f}) - likely all same value!")
            if abs(frames_mean) < 0.01 and frames_max < 0.1:
                logger.error(f"🚨 Generated frames appear EMPTY/BLACK - all values near zero!")

            # Sample frames for extraction - process all frames for accurate control losses
            num_samples = T  # Process all frames (was limited to 5)
            sample_indices = torch.arange(0, T, dtype=torch.long)  # All frame indices

            landmark_keys = ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']
            pred_landmarks = {k: [] for k in landmark_keys}
            pred_blink_states = []
            pred_emotions = []
            pred_gazes = []  # For gaze estimation from frames

            logger.debug(f"Extracting features from {num_samples} frames at indices: {sample_indices.tolist()}")

            # Optional: Save generated frames for debugging (configurable)
            save_debug_frames = getattr(self.config.loss, 'save_debug_frames', False)
            if save_debug_frames:
                debug_dir = "debug_generated_frames"
                import os
                os.makedirs(debug_dir, exist_ok=True)
                for i, t_idx in enumerate(sample_indices):
                    frame_debug = generated_frames[0, t_idx].detach().cpu()
                    if frame_debug.shape[0] == 3:  # [C, H, W]
                        from torchvision.utils import save_image

                        # Detailed frame statistics
                        frame_min = frame_debug.min().item()
                        frame_max = frame_debug.max().item()
                        frame_mean = frame_debug.mean().item()
                        frame_std = frame_debug.std().item()

                        # Check if frame is nearly black/empty (very low variance or all near zero)
                        is_black = frame_std < 0.01 or (frame_max - frame_min) < 0.05
                        is_nearly_zero = abs(frame_mean) < 0.01

                        logger.info(f"📊 Frame {i} (t={t_idx}) stats: min={frame_min:.3f}, max={frame_max:.3f}, mean={frame_mean:.3f}, std={frame_std:.3f}")
                        if is_black:
                            logger.warning(f"⚠️ Frame {i} appears BLACK (low variance: {frame_std:.4f})")
                        if is_nearly_zero:
                            logger.warning(f"⚠️ Frame {i} appears EMPTY (near-zero mean: {frame_mean:.4f})")

                        # If frame is in [-1, 1] range, convert to [0, 1]
                        if frame_min < 0:
                            frame_debug = (frame_debug + 1.0) / 2.0

                        # Remove green screen background (chroma key)
                        # Convert to numpy for processing [C, H, W] -> [H, W, C]
                        frame_np = frame_debug.permute(1, 2, 0).numpy()

                        # Convert to uint8 for OpenCV
                        import cv2
                        frame_bgr = (frame_np * 255).clip(0, 255).astype(np.uint8)
                        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
                        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

                        # Green color range in HSV for chroma keying
                        lower_green = np.array([35, 40, 40])   # Lower bound for green
                        upper_green = np.array([85, 255, 255]) # Upper bound for green

                        # Create mask for green pixels
                        green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)

                        # Create alpha channel (255 where not green, 0 where green)
                        alpha = 255 - green_mask

                        # Convert back to RGB
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        # Add alpha channel [H, W, 4]
                        frame_rgba = np.dstack([frame_rgb, alpha])

                        # Convert back to tensor [4, H, W] and normalize to [0, 1]
                        frame_debug = torch.from_numpy(frame_rgba).permute(2, 0, 1).float() / 255.0

                        save_path = f"{debug_dir}/frame_{i}_idx{t_idx}.png"
                        save_image(frame_debug, save_path)

                        # Log file size to correlate with quality
                        file_size = os.path.getsize(save_path) / 1024  # KB
                        logger.info(f"💾 Saved {save_path} ({file_size:.1f} KB) with alpha channel")

            # Track which indices successfully extracted features
            successful_indices = []

            # Extract features from each sampled frame
            for i, t_idx in enumerate(sample_indices):
                # Get frame [B, C, H, W] - take first batch item
                frame = generated_frames[0, t_idx].detach().cpu()

                # Convert to numpy [H, W, C] in range [0, 255]
                if frame.shape[0] == 3:  # [C, H, W]
                    frame = frame.permute(1, 2, 0)

                # Normalize frame if it's in [-1, 1] range (convert to [0, 1])
                if frame.min() < 0:
                    frame = (frame + 1.0) / 2.0

                frame_np = frame.numpy()

                # Ensure frame is in [0, 255] range for feature extractors
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)

                # Extract landmarks and blink state using dataset's method
                landmarks = dataset._extract_face_landmarks(frame_np, "")

                # Only process this frame if we successfully extracted landmarks
                if landmarks is not None and 'lips' in landmarks:
                    # Validate that we have valid landmark arrays
                    valid_frame = True
                    for key in landmark_keys:
                        if key in landmarks:
                            if not isinstance(landmarks[key], np.ndarray) or landmarks[key].size == 0:
                                logger.debug(f"Invalid {key} landmark array in frame {t_idx}")
                                valid_frame = False
                                break

                    if valid_frame:
                        successful_indices.append(i)
                        logger.info(f"✅ Successfully processed frame {t_idx} - face detected and features extracted")

                        for key in landmark_keys:
                            if key in landmarks:
                                pred_landmarks[key].append(landmarks[key])
                        if 'blink_state' in landmarks:
                            pred_blink_states.append(landmarks['blink_state'])

                        # Extract emotion if recognizer available (only for valid frames)
                        if dataset.emotion_recognizer is not None and 'emotion' in conditions:
                            try:
                                # Use dataset's _get_emotion method which returns (label, va_values)
                                emotion_label, emotion_va = dataset._get_emotion(frame_np)
                                pred_emotions.append(emotion_va)  # Only append VA values [valence, arousal]
                            except Exception as e:
                                logger.warning(f"Could not extract emotion from frame {t_idx}: {e}")

                        # Extract gaze if L2CS pipeline available (only for valid frames)
                        if dataset.worker_state.l2cs_pipeline is not None and 'gaze' in conditions:
                            try:
                                frame_uint8 = (frame_np * 255).astype(np.uint8) if frame_np.dtype != np.uint8 else frame_np
                                # Ensure BGR format for L2CS
                                if len(frame_uint8.shape) == 2:
                                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
                                elif frame_uint8.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGBA2BGR)
                                else:
                                    frame_bgr = frame_uint8

                                # Get gaze predictions
                                results = dataset.worker_state.l2cs_pipeline.step(frame_bgr)
                                if len(results.pitch) > 0:
                                    # Convert to degrees
                                    pitch = float(results.pitch[0] * 180 / np.pi)
                                    yaw = float(results.yaw[0] * 180 / np.pi)
                                    pred_gazes.append([pitch, yaw])
                                else:
                                    logger.debug(f"No face detected in frame {t_idx} for gaze")
                            except Exception as e:
                                logger.warning(f"Could not extract gaze from frame {t_idx}: {e}")
                    else:
                        logger.debug(f"Frame {t_idx} has invalid landmark arrays, skipping")
                else:
                    logger.warning(f"❌ No face/landmarks detected in frame {t_idx}, skipping")

            # Convert to tensors only if we have successful extractions
            if pred_landmarks['lips'] and successful_indices:
                landmarks_dict = {}
                for key in landmark_keys:
                    if pred_landmarks[key]:
                        try:
                            # Check all arrays have the same shape before stacking
                            shapes = [arr.shape for arr in pred_landmarks[key]]
                            if len(set(shapes)) == 1:  # All shapes are identical
                                landmarks_dict[key] = torch.tensor(
                                    np.stack(pred_landmarks[key]),
                                    device=device,
                                    dtype=torch.float32
                                ).unsqueeze(0)  # [1, T_successful, N_points, 3]
                            else:
                                logger.warning(f"Inconsistent shapes for {key}: {shapes}, skipping")
                        except Exception as e:
                            logger.warning(f"Could not stack {key} landmarks: {e}")

                if landmarks_dict:  # Only add if we successfully stacked some landmarks
                    extracted['landmarks'] = landmarks_dict
                    # Store only the successful sample indices for target sampling
                    extracted['sample_indices'] = sample_indices[successful_indices]
                    logger.info(f"✅ Feature extraction summary: {len(successful_indices)}/{len(sample_indices)} frames with valid faces")
                    logger.debug(f"Extracted landmarks for: {list(landmarks_dict.keys())} from {len(successful_indices)}/{len(sample_indices)} frames")
                else:
                    logger.warning(f"❌ No landmarks could be stacked due to shape inconsistencies")

            if pred_blink_states and len(pred_blink_states) == len(successful_indices):
                try:
                    # Validate blink_states shapes
                    blink_shapes = [np.array(b).shape for b in pred_blink_states]
                    logger.debug(f"Blink state shapes: {blink_shapes}")
                    if len(set(blink_shapes)) == 1:  # All same shape
                        extracted['blink_states'] = torch.tensor(
                            np.stack(pred_blink_states),
                            device=device,
                            dtype=torch.float32
                        ).unsqueeze(0)  # [1, T_successful, 3]
                        logger.debug(f"Extracted {len(pred_blink_states)} blink states")
                    else:
                        logger.warning(f"Inconsistent blink state shapes: {blink_shapes}")
                except Exception as e:
                    logger.warning(f"Could not stack blink states: {e}")

            if pred_emotions and len(pred_emotions) == len(successful_indices):
                try:
                    # Validate emotion shapes
                    emotion_shapes = [np.array(e).shape for e in pred_emotions]
                    logger.debug(f"Emotion shapes: {emotion_shapes}")
                    if len(set(emotion_shapes)) == 1:  # All same shape
                        extracted['emotions'] = torch.tensor(
                            np.stack(pred_emotions),
                            device=device,
                            dtype=torch.float32
                        ).unsqueeze(0)  # [1, T_successful, emotion_dim]
                        logger.debug(f"Extracted {len(pred_emotions)} emotion predictions")
                    else:
                        logger.warning(f"Inconsistent emotion shapes: {emotion_shapes}")
                except Exception as e:
                    logger.warning(f"Could not stack emotions: {e}")

            if pred_gazes and len(pred_gazes) == len(successful_indices):
                try:
                    # Validate gaze shapes
                    gaze_shapes = [np.array(g).shape if isinstance(g, (list, np.ndarray)) else len(g) for g in pred_gazes]
                    logger.debug(f"Gaze shapes: {gaze_shapes}, gazes: {pred_gazes[:3]}")  # Show first 3
                    if len(set([str(s) for s in gaze_shapes])) == 1:  # All same shape
                        extracted['gazes'] = torch.tensor(
                            pred_gazes,
                            device=device,
                            dtype=torch.float32
                        ).unsqueeze(0)  # [1, T_successful, 2] (pitch, yaw)
                        logger.debug(f"Extracted {len(pred_gazes)} gaze predictions")
                    else:
                        logger.warning(f"Inconsistent gaze shapes: {gaze_shapes}, gazes: {pred_gazes}")
                except Exception as e:
                    logger.warning(f"Could not stack gazes: {e}, gazes: {pred_gazes}")

        except Exception as e:
            logger.warning(f"Error extracting features from frames: {e}")
            import traceback
            tb = traceback.format_exc()
            logger.warning(f"Traceback:\n{tb}")
            # Log what we collected so far
            logger.warning(f"pred_landmarks counts: {[(k, len(v)) for k, v in pred_landmarks.items()]}")
            logger.warning(f"pred_blink_states count: {len(pred_blink_states)}")
            logger.warning(f"pred_emotions count: {len(pred_emotions)}")
            logger.warning(f"pred_gazes count: {len(pred_gazes)}")
            logger.warning(f"successful_indices: {successful_indices}")

        return extracted

    def _compute_control_losses(
        self,
        generated_frames: Optional[torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        epoch: int,
        pred_motion: Optional[Dict[str, torch.Tensor]] = None,
        dataset = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute control signal losses from generated frames.

        Args:
            generated_frames: Generated video frames [B, T, C, H, W]
            conditions: Dictionary of conditioning signals
            epoch: Current training epoch
            pred_motion: Motion parameters (for backward compatibility with gaze/distance extraction)

        Returns:
            Dictionary of control losses
        """
        try:
            logger.debug("\n=== Computing Control Losses ===")
            logger.debug(f"Current epoch: {epoch}, Control start epoch: {self.config.train.control_start_epoch}")

            # Initialize loss dict and get device
            device = generated_frames.device if generated_frames is not None else pred_motion['theta'].device
            losses = {}
            total_loss = torch.tensor(0.0, device=device)
            
            # Log available control signals
            logger.debug("\nChecking available control signals:")
            control_signals = ['gaze', 'head_distance', 'emotion', 'speed_bucket', 'lips', 'blink_state']
            for signal in control_signals:
                if signal in conditions and conditions[signal] is not None:
                    logger.debug(f"  Found {signal}: shape={conditions[signal].shape}")
                else:
                    logger.debug(f"  Missing or None: {signal}")

            logger.debug(f"\nUsing device: {device}")

            # STEP 1: Extract features from generated frames
            # This includes landmarks (lips, eyes, jaw, nose), blink states, and emotions
            extracted_features = {}

            # Only extract from frames after epoch 5 when quality should be better
            # Early training uses motion parameters only
            use_frame_extraction = epoch >= 5 and generated_frames is not None and dataset is not None

            if use_frame_extraction:
                logger.debug("\n=== Extracting features from generated frames ===")
                try:
                    extracted_features = self._extract_features_from_frames(
                        generated_frames=generated_frames,
                        conditions=conditions,
                        device=device,
                        dataset=dataset
                    )
                    logger.debug(f"Extracted features: {list(extracted_features.keys())}")

                    # If extraction failed completely (no faces detected), skip control losses entirely
                    # to prevent unstable training from bad feature extractions
                    if not extracted_features or 'landmarks' not in extracted_features:
                        logger.warning("⚠️ No faces detected in generated frames - SKIPPING control losses for this window to prevent instability")
                        return self._get_zero_losses()
                except Exception as e:
                    logger.warning(f"⚠️ Frame extraction failed: {e} - SKIPPING control losses for this window")
                    return self._get_zero_losses()
            else:
                logger.debug(f"Skipping frame extraction (epoch {epoch} < 5, no dataset, or no generated frames)")
                # Before epoch 5, return zero control losses - we don't have reliable features yet
                if epoch < 5:
                    logger.debug("Early training epoch - returning zero control losses")
                    return self._get_zero_losses()

            # STEP 2: Compute control losses from extracted features
            # 1. Gaze Loss - from extracted features from generated frames only
            logger.debug("\nComputing Gaze Loss:")
            if 'gaze' in conditions and conditions['gaze'] is not None and 'gazes' in extracted_features:
                try:
                    pred_gaze = extracted_features['gazes']  # [1, T_sampled, 2]
                    sample_indices = extracted_features['sample_indices']
                    target_gaze = conditions['gaze'][:, sample_indices].to(device).float()  # [B, T_sampled, 2]

                    logger.debug(f"  Predicted gaze shape: {pred_gaze.shape}")
                    logger.debug(f"  Target gaze shape: {target_gaze.shape}")

                    # Convert degrees to radians for cosine similarity
                    pred_gaze_rad = pred_gaze * (np.pi / 180.0)
                    target_gaze_rad = target_gaze * (np.pi / 180.0)

                    # Compute gaze loss with NaN protection
                    diff = pred_gaze_rad - target_gaze_rad
                    diff = torch.clamp(diff, -3.14, 3.14)
                    cos_sim = torch.cos(diff)

                    if torch.isnan(cos_sim).any() or torch.isinf(cos_sim).any():
                        logger.warning("NaN/Inf in gaze cosine similarity, using zero loss")
                        gaze_loss = torch.tensor(0.0, device=device)
                    else:
                        gaze_loss = (1 - cos_sim).mean()

                    if torch.isnan(gaze_loss) or torch.isinf(gaze_loss):
                        logger.warning("NaN/Inf in final gaze loss, using zero")
                        gaze_loss = torch.tensor(0.0, device=device)

                    losses['control_gaze'] = gaze_loss * self.lambda_gaze_direction
                    total_loss = total_loss + losses['control_gaze']
                    logger.debug(f"  Gaze loss from generated frames: {losses['control_gaze'].item():.6f}")
                except Exception as e:
                    logger.warning(f"Error computing gaze loss from extracted features: {e}")
                    losses['control_gaze'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No gaze signal or extracted features")
                losses['control_gaze'] = torch.tensor(0.0, device=device)

            # 2. Audio-Lip Correlation Loss - from extracted landmarks in generated frames
            logger.debug("\nComputing Audio-Lip Correlation Loss:")
            audio_lip_term = torch.tensor(0.0, device=device)

            audio_key = 'audio_features' if 'audio_features' in conditions else 'audio' if 'audio' in conditions else None

            if audio_key and 'landmarks' in extracted_features and 'lips' in extracted_features['landmarks']:
                try:
                    # Extract lip landmarks from generated frames
                    pred_lips = extracted_features['landmarks']['lips']  # [1, T_sampled, N_points, 3]
                    sample_indices = extracted_features['sample_indices']

                    logger.debug(f"  Extracted lip landmarks shape: {pred_lips.shape}")

                    # Compute lip openness from extracted landmarks
                    # lips shape: [1, T, 20, 3] where 20 points, upper half = upper lip, lower half = lower lip
                    upper_lips = pred_lips[:, :, :pred_lips.shape[2]//2, 1]  # Upper lip y-coords
                    lower_lips = pred_lips[:, :, pred_lips.shape[2]//2:, 1]  # Lower lip y-coords
                    lip_openness = (lower_lips.mean(dim=-1) - upper_lips.mean(dim=-1)).abs()  # [1, T_sampled]

                    # Get corresponding audio features
                    audio_features = conditions[audio_key][:, sample_indices].to(device).float()  # [B, T_sampled, D]
                    audio_energy = torch.norm(audio_features, dim=-1)  # [B, T_sampled]

                    logger.debug(f"  Lip openness range: [{lip_openness.min():.4f}, {lip_openness.max():.4f}]")
                    logger.debug(f"  Audio energy range: [{audio_energy.min():.4f}, {audio_energy.max():.4f}]")

                    # Normalize both signals
                    lip_openness_norm = (lip_openness - lip_openness.min()) / (lip_openness.max() - lip_openness.min() + 1e-8)
                    audio_energy_norm = (audio_energy - audio_energy.min()) / (audio_energy.max() - audio_energy.min() + 1e-8)

                    # MSE loss between normalized signals
                    audio_lip_loss = F.mse_loss(lip_openness_norm, audio_energy_norm)
                    audio_lip_term = audio_lip_loss * self.lambda_audio_lip

                    losses['audio_lip_correlation'] = audio_lip_term
                    total_loss = total_loss + audio_lip_term
                    logger.debug(f"  Audio-lip correlation loss: {audio_lip_term.item():.6f}")

                    # 2b. Direct mouth openness supervision (using extracted lips)
                    # Direct supervision: mouth should be open when audio is strong
                    mouth_openness_loss = F.mse_loss(lip_openness_norm, audio_energy_norm) * self.lambda_mouth_openness
                    losses['mouth_openness_direct'] = mouth_openness_loss
                    total_loss = total_loss + mouth_openness_loss
                    logger.debug(f"  Mouth openness direct loss: {mouth_openness_loss.item():.6f} (lambda={self.lambda_mouth_openness})")

                except Exception as e:
                    logger.warning(f"Could not compute audio-lip correlation from extracted features: {e}")
                    losses['audio_lip_correlation'] = torch.tensor(0.0, device=device)
                    losses['mouth_openness_direct'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  Skipping audio-lip correlation (no audio or no extracted lips)")
                losses['audio_lip_correlation'] = torch.tensor(0.0, device=device)
                losses['mouth_openness_direct'] = torch.tensor(0.0, device=device)

            # 3. Head Distance Loss
            logger.debug("\nComputing Head Distance Loss:")
            if 'head_distance' in conditions and conditions['head_distance'] is not None:
                pred_distance = self._extract_distance_from_motion(pred_motion)
                logger.debug(f"  Predicted distance shape: {pred_distance.shape}")
                
                target_distance = conditions['head_distance'].to(device).float()
                logger.debug(f"  Target distance shape: {target_distance.shape}")
                logger.debug(f"  Target distance range: [{target_distance.min():.3f}, {target_distance.max():.3f}]")
                
                # Ensure distance has sequence dimension
                if len(target_distance.shape) == 2:  # [B, 1]
                    target_distance = target_distance.unsqueeze(1).expand(-1, pred_distance.shape[1], -1)
                    logger.debug(f"  Expanded target distance shape: {target_distance.shape}")
                
                pred_norm = pred_distance / (pred_distance.mean(dim=1, keepdim=True) + 1e-6)
                target_norm = target_distance / (target_distance.mean(dim=1, keepdim=True) + 1e-6)
                
                distance_loss = F.mse_loss(pred_norm, target_norm)
                losses['control_distance'] = distance_loss * self.lambda_distance
                total_loss = total_loss + losses['control_distance']
                logger.debug(f"  Distance loss: {losses['control_distance'].item():.6f}")
            else:
                logger.debug("  No head_distance signal found")
                losses['control_distance'] = torch.tensor(0.0, device=device)

            # 3. Emotion Loss - only from motion parameters if frame extraction is disabled
            # (Frame-based emotion is computed later in the landmark section)
            if not use_frame_extraction:
                logger.debug("\nComputing Emotion Loss from motion parameters:")
                if 'emotion' in conditions and conditions['emotion'] is not None and pred_motion is not None:
                    pred_emotion = self._extract_emotion_from_motion(pred_motion)
                    logger.debug(f"  Predicted emotion shape: {pred_emotion.shape}")

                    target_emotion = conditions['emotion'].to(device).float()
                    logger.debug(f"  Target emotion shape: {target_emotion.shape}")
                    logger.debug(f"  Target emotion range: [{target_emotion.min():.3f}, {target_emotion.max():.3f}]")

                    # Ensure emotion has sequence dimension
                    if len(target_emotion.shape) == 2:  # [B, 2]
                        target_emotion = target_emotion.unsqueeze(1).expand(-1, pred_emotion.shape[1], -1)
                        logger.debug(f"  Expanded target emotion shape: {target_emotion.shape}")

                    emotion_loss = F.mse_loss(pred_emotion, target_emotion)
                    losses['control_emotion'] = emotion_loss * self.lambda_emotion
                    total_loss = total_loss + losses['control_emotion']
                    logger.debug(f"  Emotion loss from motion: {losses['control_emotion'].item():.6f}")
                else:
                    logger.debug("  No emotion signal found")
                    losses['control_emotion'] = torch.tensor(0.0, device=device)

             # 4. Speed Loss 
            logger.debug("\nComputing Speed Loss:")
            if 'speed_bucket' in conditions:
                speed_loss, speed_metrics = self.speed_handler.compute_speed_loss(
                    pred_motion=pred_motion,
                    target_buckets=conditions['speed_bucket'],
                    lambda_speed=self.lambda_speed,
                    device=self.device
                )
                losses['control_speed'] = speed_loss
                total_loss = total_loss + speed_loss

                # Convert metrics to tensors and add to losses
                for k, v in speed_metrics.items():
                    losses[f'speed_{k}'] = torch.tensor(v, device=self.device)
                
            else:
                logger.debug("  No speed_bucket signal found")
                losses['control_speed'] = torch.tensor(0.0, device=self.device)
                losses['speed_loss'] = torch.tensor(0.0, device=self.device)
                losses['speed_accuracy'] = torch.tensor(0.0, device=self.device)


            # STEP 3: Compute losses from extracted features
            # 4. Landmark Losses (lips, eyes, jaw, nose) - from extracted features
            logger.debug("\nComputing Landmark Losses:")
            landmark_keys = ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']
            landmark_conditions = {k: v for k, v in conditions.items() if k in landmark_keys and v is not None}

            if landmark_conditions and 'landmarks' in extracted_features:
                try:
                    pred_landmarks_dict = extracted_features['landmarks']
                    sample_indices = extracted_features['sample_indices']

                    # Get corresponding target landmarks
                    target_landmarks_dict = {}
                    for key in landmark_keys:
                        if key in landmark_conditions:
                            # Sample same frames from targets
                            target = landmark_conditions[key]  # [B, T, N, 3]
                            target_landmarks_dict[key] = target[:, sample_indices]

                    # Compute landmark losses
                    landmark_losses = self._compute_landmark_losses(pred_landmarks_dict, target_landmarks_dict)
                    losses.update(landmark_losses)

                    # Add landmark losses to total control loss
                    if 'facial_motion_total' in landmark_losses:
                        total_loss = total_loss + landmark_losses['facial_motion_total']
                        logger.debug(f"  Landmark losses computed from extracted features")

                except Exception as e:
                    logger.warning(f"Could not compute landmark losses from generated frames: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            else:
                logger.debug("  No landmark conditions or extracted landmarks")

            # 5. Blink Loss - from extracted features
            logger.debug("\nComputing Blink Loss:")
            if 'blink_states' in extracted_features and 'blink_state' in conditions and conditions['blink_state'] is not None:
                try:
                    pred_blink_tensor = extracted_features['blink_states']
                    sample_indices = extracted_features['sample_indices']

                    # Sample target blink states
                    target_blink = conditions['blink_state'][:, sample_indices]  # [B, T_sampled, 3]

                    # Compute blink loss
                    blink_loss, blink_metrics = self._compute_blink_loss(
                        {'blink_state': pred_blink_tensor},  # Wrap in dict like pred_motion
                        target_blink,
                        lambda_blink=self.lambda_blink,
                        device=device
                    )
                    losses.update(blink_metrics)
                    losses['control_blink'] = blink_loss
                    total_loss = total_loss + blink_loss
                    logger.debug(f"  Blink loss from generated frames: {blink_loss.item():.6f}")
                except Exception as e:
                    logger.warning(f"Could not compute blink loss from extracted features: {e}")
                    losses['control_blink'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No blink states or conditions")
                losses['control_blink'] = torch.tensor(0.0, device=device)

            # 6. Emotion Loss - from extracted features (overrides earlier computation from motion)
            logger.debug("\nComputing Emotion Loss from generated frames:")
            if 'emotions' in extracted_features and 'emotion' in conditions and conditions['emotion'] is not None:
                try:
                    pred_emotion_tensor = extracted_features['emotions']
                    sample_indices = extracted_features['sample_indices']

                    # Sample target emotions
                    target_emotion = conditions['emotion'][:, sample_indices]  # [B, T_sampled, emotion_dim]

                    # Compute emotion loss (MSE between logits)
                    emotion_loss = F.mse_loss(pred_emotion_tensor, target_emotion) * self.lambda_emotion
                    losses['control_emotion'] = emotion_loss
                    total_loss = total_loss + emotion_loss
                    logger.debug(f"  Emotion loss from generated frames: {emotion_loss.item():.6f}")
                except Exception as e:
                    logger.warning(f"Could not compute emotion loss from extracted features: {e}")
                    if 'control_emotion' not in losses:
                        losses['control_emotion'] = torch.tensor(0.0, device=device)

            # Set final control total
            losses['control_total'] = total_loss

            # Log all losses, ensuring they're tensors
            logger.debug("\nControl Loss Summary:")
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: {v.item():.6f}")
                else:
                    logger.debug(f"  {k}: {v:.6f}")
                    losses[k] = torch.tensor(v, device=self.device)

            return losses

        except Exception as e:
            logger.error(f"Error computing control losses: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_zero_losses(device=self.device)
            
   

    def _get_zero_losses(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Return dictionary of zero losses."""
        try:
            # Get device from input or model parameters
            if device is None:
                device = self.device
                logger.debug(f"Using model device: {device}")

            logger.debug("Creating zero losses")
            return {
                'lips_total': torch.tensor(0.0, device=device),
                'nonlip_total': torch.tensor(0.0, device=device),
                'facial_motion_total': torch.tensor(0.0, device=device),
                'control_total': torch.tensor(0.0, device=device),
                'control_gaze': torch.tensor(0.0, device=device),
                'control_distance': torch.tensor(0.0, device=device),
                'control_emotion': torch.tensor(0.0, device=device),
                'control_speed': torch.tensor(0.0, device=device)
            }
        except Exception as e:
            logger.error(f"Error creating zero losses: {str(e)}")
            # Default to CPU if all else fails
            logger.warning("Defaulting to CPU device for zero losses")
            return {
                'control_total': torch.tensor(0.0),
                'control_gaze': torch.tensor(0.0),
                'control_distance': torch.tensor(0.0),
                'control_emotion': torch.tensor(0.0),
                'control_speed': torch.tensor(0.0)
            }

    
    def _extract_gaze_from_motion(
        self,
        motion: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Extract gaze angles from motion parameters using safe Euler extraction."""
        try:
            theta = motion['theta']  # [B, T, 3, 4]
            R = theta[..., :3, :3]  # Extract rotation part [B, T, 3, 3]
            
            # Use safe Euler extraction to avoid NaN in backward pass
            euler_angles = safe_matrix_to_euler(R)  # [B, T, 3] (pitch, yaw, roll)
            
            # Extract pitch and yaw for gaze (ignore roll)
            pitch = euler_angles[..., 0]
            yaw = euler_angles[..., 1]
            
            # Clamp angles to reasonable ranges
            # Pitch: -90 to 90 degrees (-π/2 to π/2)
            # Yaw: -180 to 180 degrees (-π to π)
            pitch_clamped = torch.clamp(pitch, -1.57, 1.57)
            yaw_clamped = torch.clamp(yaw, -3.14, 3.14)
            
            # Stack gaze angles
            gaze = torch.stack([pitch_clamped, yaw_clamped], dim=-1)  # [B, T, 2]
            
            # Validate outputs
            if torch.isnan(gaze).any() or torch.isinf(gaze).any():
                logger.warning("Invalid values in gaze angles after safe extraction - returning zeros")
                return torch.zeros_like(gaze)
                
            return gaze
            
        except Exception as e:
            logger.error(f"Error extracting gaze: {str(e)}")
            return None

    def _extract_distance_from_motion(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract head distance from translation."""
        # Use z-translation as proxy for head distance
        if 'translation' not in motion:
            # If translation not available (e.g., derived warps mode), return default
            B = motion.get('theta', motion.get('expression_embed')).shape[0]
            T = motion.get('theta', motion.get('expression_embed')).shape[1]
            device = motion.get('theta', motion.get('expression_embed')).device
            return torch.ones(B, T, 1, device=device) * 0.5  # Default distance

        translation = motion['translation']  # [B, T, 3]
        z_dist = translation[..., 2:3]  # Get z-component

        # Normalize to [0, 1]
        distance = torch.sigmoid(z_dist)

        return distance

    def _extract_emotion_from_motion(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract emotion values from expression embeddings."""
        # Project first two dimensions as valence-arousal
        expression = motion['expression_embed']  # [B, T, 128]
        emotion = expression[..., :2]  # [B, T, 2]
        
        # Normalize to [-1, 1]
        emotion = torch.tanh(emotion)
        
        return emotion



    def _compute_angular_loss(
        self,
        pred_angles: torch.Tensor,
        target_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular difference loss considering periodicity.
        Args:
            pred_angles: Predicted angles in radians
            target_angles: Target angles in radians
        Returns:
            Angular difference loss
        """
        # Normalize angles to [-π, π]
        pred_norm = torch.atan2(torch.sin(pred_angles), torch.cos(pred_angles))
        target_norm = torch.atan2(torch.sin(target_angles), torch.cos(target_angles))
        
        # Compute shortest angular distance
        diff = pred_norm - target_norm
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        return torch.mean(diff ** 2)

    def _compute_pose_matrix_loss(
        self,
        pred_theta: torch.Tensor,  # [B, T, 3, 4] or [B, 3, 4]
        target_theta: torch.Tensor,  # [B, T, 3, 4] or [B, 3, 4]
        separate_rotation: bool = True
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target pose matrices with proper handling
        of rotation and translation components.
        
        Args:
            pred_theta: Predicted transformation matrices [B, T, 3, 4] or [B, 3, 4]
            target_theta: Target transformation matrices [B, T, 3, 4] or [B, 3, 4]
            separate_rotation: Whether to compute rotation and translation losses separately
            
        Returns:
            Combined pose matrix loss (scalar tensor)
        """
        try:
            # Add time dimension if not present
            if pred_theta.dim() == 3:
                pred_theta = pred_theta.unsqueeze(1)
                target_theta = target_theta.unsqueeze(1)
                
            B, T = pred_theta.shape[:2]
            device = pred_theta.device
            
            if separate_rotation:
                # Extract rotation matrices (3x3)
                pred_R = pred_theta[..., :3, :3]
                target_R = target_theta[..., :3, :3]
                
                # Extract translation vectors
                pred_t = pred_theta[..., :3, 3]
                target_t = target_theta[..., :3, 3]
                
                # Compute rotation loss using multiple metrics for better gradients

                # 1. Geodesic distance (angular error)
                R_diff = torch.matmul(pred_R.transpose(-2, -1), target_R)
                trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
                cos_theta = (trace - 1) / 2
                cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)  # Numerical stability
                geodesic_loss = torch.acos(cos_theta).mean()

                # 2. Frobenius norm (provides better gradients when rotations are small)
                frobenius_loss = torch.norm(pred_R - target_R, dim=(-2, -1)).mean()

                # 3. 6D rotation representation loss (for better optimization)
                # Convert to 6D representation (first two columns of rotation matrix)
                pred_6d = pred_R[..., :, :2].reshape(B * T, 6)
                target_6d = target_R[..., :, :2].reshape(B * T, 6)
                rotation_6d_loss = F.mse_loss(pred_6d, target_6d)

                # Combine rotation losses with adaptive weighting
                rotation_loss = 0.5 * geodesic_loss + 0.3 * frobenius_loss + 0.2 * rotation_6d_loss

                # Compute translation loss (L2)
                translation_loss = F.mse_loss(pred_t, target_t)

                # Add stronger weighting to rotation to ensure head movement
                total_loss = 2.0 * rotation_loss + translation_loss
                
            else:
                # Direct matrix comparison using Frobenius norm
                matrix_diff = pred_theta - target_theta
                total_loss = torch.norm(matrix_diff.view(B * T, -1), p='fro').mean()


            # Add velocity consistency
            # pred_vel = pred_theta[:, 1:] - pred_theta[:, :-1]
            # target_vel = target_theta[:, 1:] - target_theta[:, :-1]
            # velocity_loss = F.mse_loss(pred_vel, target_vel) * 0.1
            
            # total_loss = rotation_loss + 0.5 * translation_loss + velocity_loss
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing pose matrix loss: {str(e)}")
            logger.error("\nDebug info:")
            logger.error(f"pred_theta shape: {pred_theta.shape}")
            logger.error(f"target_theta shape: {target_theta.shape}")
            logger.error(f"pred_theta device: {pred_theta.device}")
            logger.error(f"target_theta device: {target_theta.device}")
            # Return default loss
            return torch.tensor(0.0, device=device if 'device' in locals() else 'cpu')

    def _compute_geodesic_loss(
        self,
        R1: torch.Tensor,  # [..., 3, 3]
        R2: torch.Tensor   # [..., 3, 3]
    ) -> torch.Tensor:
        """
        Helper function to compute geodesic distance between rotation matrices.
        
        Args:
            R1, R2: Rotation matrices [..., 3, 3]
            
        Returns:
            Geodesic distance loss
        """
        # Compute R1 @ R2.T
        R_diff = torch.matmul(R1, R2.transpose(-2, -1))
        
        # Get trace
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        
        # Compute angle (clamp for numerical stability)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1, 1)
        theta = torch.acos(cos_theta)
        
        return theta.mean()

    def _validate_rotation_matrix(
        self,
        R: torch.Tensor,  # [..., 3, 3]
        eps: float = 1e-6
    ) -> bool:
        """
        Helper function to validate rotation matrix properties.
        
        Args:
            R: Rotation matrix to validate
            eps: Tolerance for numerical comparisons
            
        Returns:
            True if valid rotation matrix
        """
        # Check orthogonality
        I = torch.eye(3, device=R.device).expand_as(R)
        orth_error = torch.norm(
            torch.matmul(R, R.transpose(-2, -1)) - I
        )
        
        # Check determinant
        det = torch.linalg.det(R)
        det_error = torch.abs(det - 1)
        
        return orth_error < eps and det_error < eps


    def compute_disentanglement_loss(
        self, 
        motion_outputs: Dict[str, torch.Tensor],
        target_frames: Optional[torch.Tensor] = None,
        generated_frames: Optional[torch.Tensor] = None,
        source_identity: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VASA-1 style disentanglement losses as described in Section 3.1.
        
        Following VASA paper:
        - l_consist: Pairwise head pose and facial dynamics transfer loss
        - l_cross_id: Face identity similarity loss for cross-identity transfer
        
        Args:
            motion_outputs: Dictionary containing predicted motion parameters
            target_frames: Optional ground truth frames for reference [B, T, C, H, W]
            generated_frames: Optional generated frames from motion [B, T, C, H, W]
            source_identity: Optional source identity image [B, C, H, W]
        """
        try:
            device = self.device
            
            # Initialize losses
            l_consist = torch.tensor(0.0, device=device)
            l_cross_id = torch.tensor(0.0, device=device)
            
            # Check if we have the required outputs
            if 'theta' not in motion_outputs:
                logger.debug("DISENTANGLE: No theta (pose) in outputs, skipping")
                return l_consist, l_consist, l_consist
            
            B, T = motion_outputs['theta'].shape[:2]
            
            if T < 2:
                logger.debug(f"DISENTANGLE: Sequence too short (T={T}), need at least 2 frames")
                return l_consist, l_consist, l_consist
            
            # 1. L_consist: Pairwise transfer loss (VASA Section 3.1)
            # Transfer pose from frame i to j and dynamics from j to i
            # Get dynamics - could be 'expression', 'expression_embed', or individual components
            dynamics_key = None
            if 'expression_embed' in motion_outputs:
                dynamics_key = 'expression_embed'
            elif 'expression' in motion_outputs:
                dynamics_key = 'expression'
            
            if dynamics_key:
                # Get config for sampling strategy
                num_pairs = getattr(self.config.loss, 'disentangle_num_pairs', 3)
                sampling_strategy = getattr(self.config.loss, 'disentangle_sampling', 'random')
                
                # Sample multiple frame pairs for more robust loss
                l_consist_accum = 0.0
                actual_pairs = min(num_pairs, T // 2)  # Don't sample more pairs than possible
                
                for pair_idx in range(actual_pairs):
                    if sampling_strategy == 'uniform':
                        # Uniform sampling across sequence
                        idx_i = pair_idx * T // actual_pairs
                        idx_j = min((pair_idx + 1) * T // actual_pairs, T - 1)
                    elif sampling_strategy == 'random':
                        # Random sampling
                        indices = torch.randperm(T)[:2]
                        idx_i, idx_j = indices[0].item(), indices[1].item()
                    else:
                        # Default: first and last frames
                        idx_i = 0
                        idx_j = T - 1
                    
                    # Get pose (theta) and dynamics at different times
                    pose_i = motion_outputs['theta'][:, idx_i]  # [B, 3, 4]
                    pose_j = motion_outputs['theta'][:, idx_j]
                    dyn_i = motion_outputs[dynamics_key][:, idx_i]  # [B, D]
                    dyn_j = motion_outputs[dynamics_key][:, idx_j]
                    
                    # Normalize by their respective scales for fair comparison
                    pose_scale = torch.std(motion_outputs['theta'].view(B, -1), dim=1, keepdim=True) + 1e-6
                    dyn_scale = torch.std(motion_outputs[dynamics_key].view(B, -1), dim=1, keepdim=True) + 1e-6
                    
                    pose_i_norm = pose_i.view(B, -1) / pose_scale
                    pose_j_norm = pose_j.view(B, -1) / pose_scale
                    dyn_i_norm = dyn_i.view(B, -1) / dyn_scale
                    dyn_j_norm = dyn_j.view(B, -1) / dyn_scale
                    
                    # Following VASA Eq in Section 3.1:
                    # The discrepancy between swapped combinations should be minimized
                    # This encourages disentanglement between pose and dynamics
                    pose_diff = F.mse_loss(pose_i_norm, pose_j_norm.detach())
                    dyn_diff = F.mse_loss(dyn_i_norm, dyn_j_norm.detach())
                    
                    l_consist_accum += (pose_diff + dyn_diff)
                    
                    if pair_idx == 0:
                        logger.debug(f"DISENTANGLE: First pair ({idx_i},{idx_j}) - pose_diff={pose_diff.item():.4f}, dyn_diff={dyn_diff.item():.4f}")
                
                # Average over pairs and apply weight
                l_consist = (l_consist_accum / actual_pairs) * self.lambda_consist
                logger.debug(f"DISENTANGLE: l_consist computed = {l_consist.item():.6f} (averaged over {actual_pairs} pairs)")
            else:
                logger.debug("DISENTANGLE: No dynamics (expression) found, l_consist = 0")

            # 2. L_cross_id: Identity preservation loss (VASA Section 3.1)
            # Ensures identity is preserved when transferring motion
            if self.id_extractor is not None:
                try:
                    # Get config for identity loss computation
                    num_frames = getattr(self.config.loss, 'cross_id_num_frames', 3)
                    normalize_frames = getattr(self.config.loss, 'cross_id_normalize', True)
                    
                    # Determine what frames to use for identity comparison
                    source_frames = None
                    generated_frames_to_compare = None
                    
                    # Priority 1: Use provided source identity and generated frames
                    if source_identity is not None and generated_frames is not None:
                        # Expand source_identity to match number of frames if needed
                        if source_identity.dim() == 4:  # [B, C, H, W]
                            source_frames = source_identity.unsqueeze(1).expand(-1, min(num_frames, generated_frames.shape[1]), -1, -1, -1)
                        else:
                            source_frames = source_identity[:, :num_frames]  # Already has time dimension
                        generated_frames_to_compare = generated_frames[:, :min(num_frames, generated_frames.shape[1])]
                        logger.debug(f"Using provided source_identity and generated_frames ({generated_frames_to_compare.shape[1]} frames)")
                    
                    # Priority 2: Use target frames as source identity reference
                    elif target_frames is not None and generated_frames is not None:
                        num_available = min(num_frames, target_frames.shape[1], generated_frames.shape[1])
                        source_frames = target_frames[:, :num_available]  # Use multiple frames
                        generated_frames_to_compare = generated_frames[:, :num_available]
                        logger.debug(f"Using target_frames as source identity ({num_available} frames)")
                    
                    # Priority 3: Fall back to comparing different time points in target frames
                    elif target_frames is not None and target_frames.shape[1] >= 2:
                        # Sample frames uniformly across the sequence
                        indices = torch.linspace(0, target_frames.shape[1]-1, min(num_frames, target_frames.shape[1])).long()
                        source_frames = target_frames[:, indices[:len(indices)//2]]   # First half
                        generated_frames_to_compare = target_frames[:, indices[len(indices)//2:]]  # Second half
                        logger.debug(f"Fallback: comparing target_frames across time ({source_frames.shape[1]} vs {generated_frames_to_compare.shape[1]} frames)")
                    
                    if source_frames is not None and generated_frames_to_compare is not None:
                        # Ensure we have same number of frames to compare
                        min_frames = min(source_frames.shape[1], generated_frames_to_compare.shape[1])
                        source_frames = source_frames[:, :min_frames]
                        generated_frames_to_compare = generated_frames_to_compare[:, :min_frames]
                        
                        # Normalize frames if configured
                        if normalize_frames:
                            # Normalize to [-1, 1] range for better feature extraction
                            def normalize_frame(x):
                                # Assume frames are in [0, 255] or [0, 1] range
                                if x.max() > 1.0:
                                    x = x / 127.5 - 1.0
                                elif x.min() >= 0:
                                    x = x * 2.0 - 1.0
                                return x
                            
                            source_frames = normalize_frame(source_frames)
                            generated_frames_to_compare = normalize_frame(generated_frames_to_compare)
                        
                        # Flatten batch and time dimensions for processing
                        B, T_src = source_frames.shape[:2]
                        source_flat = source_frames.view(B * T_src, *source_frames.shape[2:])
                        generated_flat = generated_frames_to_compare.view(B * T_src, *generated_frames_to_compare.shape[2:])
                        
                        # Enhanced high-res identity extraction
                        # Get configuration values
                        highres_scale = getattr(self.config.loss, 'identity_highres_scale', 2)
                        sharpening_alpha = getattr(self.config.loss, 'identity_sharpening_alpha', 0.5)
                        use_antialias = getattr(self.config.loss, 'identity_use_antialias', True)
                        
                        original_h, original_w = source_flat.shape[-2:]
                        
                        # Step 1: Upscale to higher resolution with bicubic for smooth interpolation
                        source_highres = F.interpolate(
                            source_flat, 
                            size=(original_h * highres_scale, original_w * highres_scale),
                            mode='bicubic', 
                            align_corners=False
                        )
                        generated_highres = F.interpolate(
                            generated_flat,
                            size=(original_h * highres_scale, original_w * highres_scale), 
                            mode='bicubic',
                            align_corners=False
                        )
                        
                        # Step 2: Apply sharpening filter to enhance facial features
                        # Simple unsharp mask: image + alpha * (image - blurred)
                        blur_kernel_size = 3
                        alpha = sharpening_alpha  # Sharpening strength from config
                        
                        # Apply Gaussian blur for unsharp mask
                        source_blurred = F.avg_pool2d(
                            F.pad(source_highres, (1, 1, 1, 1), mode='reflect'),
                            kernel_size=blur_kernel_size, stride=1
                        )
                        generated_blurred = F.avg_pool2d(
                            F.pad(generated_highres, (1, 1, 1, 1), mode='reflect'),
                            kernel_size=blur_kernel_size, stride=1
                        )
                        
                        # Apply unsharp mask
                        source_sharpened = source_highres + alpha * (source_highres - source_blurred)
                        generated_sharpened = generated_highres + alpha * (generated_highres - generated_blurred)
                        
                        # Step 3: Resize to identity extractor input size (160x160)
                        # Use Lanczos (approximated by bicubic) for high-quality downsampling
                        source_resized = F.interpolate(
                            source_sharpened, 
                            size=(160, 160),
                            mode='bicubic',
                            align_corners=False,
                            antialias=use_antialias
                        )
                        generated_resized = F.interpolate(
                            generated_sharpened,
                            size=(160, 160),
                            mode='bicubic', 
                            align_corners=False,
                            antialias=use_antialias
                        )
                        
                        # Clamp values to valid range after processing
                        source_resized = torch.clamp(source_resized, -1, 1)
                        generated_resized = torch.clamp(generated_resized, -1, 1)
                        
                        with torch.no_grad():
                            # Extract identity features for all frames
                            id_feat_source = self.id_extractor(source_resized)
                            id_feat_generated = self.id_extractor(generated_resized)
                        
                        # Reshape back to [B, T, D]
                        id_feat_source = id_feat_source.view(B, T_src, -1)
                        id_feat_generated = id_feat_generated.view(B, T_src, -1)
                        
                        # Compute similarity across all frame pairs
                        similarities = []
                        for t in range(T_src):
                            sim = F.cosine_similarity(id_feat_source[:, t], id_feat_generated[:, t], dim=1)
                            similarities.append(sim)
                        
                        # Average similarity across frames
                        avg_similarity = torch.stack(similarities).mean()
                        
                        # Following VASA: maximize cosine similarity to preserve identity
                        # Use 1 - cosine_similarity as loss (minimize to maximize similarity)
                        l_cross_id = (1 - avg_similarity) * self.lambda_cross_id
                        
                        logger.debug(f"DISENTANGLE: l_cross_id computed = {l_cross_id.item():.6f} (avg_similarity={avg_similarity.item():.4f} over {T_src} frames)")
                    else:
                        logger.debug("DISENTANGLE: No suitable frames for l_cross_id computation")
                        
                except Exception as e:
                    logger.error(f"Cross-id loss computation failed: {e}")
                    logger.error(f"Source frame shape: {source_frame.shape if source_frame is not None else 'None'}")
                    logger.error(f"Generated frame shape: {generated_frame.shape if generated_frame is not None else 'None'}")
                    l_cross_id = torch.tensor(0.0, device=device)
            else:
                logger.debug(f"DISENTANGLE: id_extractor not available")
            
            # Visualization of disentanglement (if configured and at visualization interval)
            if step is not None and hasattr(self, 'volumetric_avatar') and generated_frames is not None:
                vis_freq = getattr(self.config.loss, 'disentangle_vis_freq', 100)
                if step % vis_freq == 0 and step > 0:
                    try:
                        logger.debug("Creating disentanglement visualization...")
                        # Visualize the swapped combinations for first sample in batch
                        if dynamics_key and 'theta' in motion_outputs:
                            # Get a pair of frames for visualization
                            idx_i = 0
                            idx_j = min(T - 1, T // 2)  # Use middle frame if available
                            
                            # Original combinations
                            pose_i = motion_outputs['theta'][0:1, idx_i]  # [1, 3, 4]
                            pose_j = motion_outputs['theta'][0:1, idx_j]
                            dyn_i = motion_outputs[dynamics_key][0:1, idx_i]  # [1, D]
                            dyn_j = motion_outputs[dynamics_key][0:1, idx_j]
                            
                            # Log the swapped frame visualization
                            import wandb
                            wandb.log({
                                'disentangle/frame_i': wandb.Image(generated_frames[0, idx_i].cpu() if generated_frames.shape[1] > idx_i else target_frames[0, idx_i].cpu()),
                                'disentangle/frame_j': wandb.Image(generated_frames[0, idx_j].cpu() if generated_frames.shape[1] > idx_j else target_frames[0, idx_j].cpu()),
                                'disentangle/l_consist': l_consist.item(),
                                'disentangle/l_cross_id': l_cross_id.item(),
                                'disentangle/pose_diff_norm': torch.std(motion_outputs['theta'][0]).item(),
                                'disentangle/dyn_diff_norm': torch.std(motion_outputs[dynamics_key][0]).item()
                            }, step=step)
                            
                            logger.debug(f"Logged disentanglement visualization at step {step}")
                    except Exception as e:
                        logger.warning(f"Could not create disentanglement visualization: {e}")
            
            # Return both the total and individual components
            total_disentangle = l_consist + l_cross_id
            
            logger.debug(
                f"VASA Disentanglement - l_consist: {l_consist.item():.6f}, "
                f"l_cross_id: {l_cross_id.item():.6f}, "
                f"total: {total_disentangle.item():.6f}"
            )
            
            return total_disentangle, l_consist, l_cross_id
            
        except Exception as e:
            import traceback
            logger.error(f"Error in VASA disentanglement loss: {str(e)}")
            logger.error(f"Motion outputs keys: {motion_outputs.keys() if motion_outputs else 'None'}")
            logger.error(f"Target frames shape: {target_frames.shape if target_frames is not None else 'None'}")
            logger.error(f"Generated frames shape: {generated_frames.shape if generated_frames is not None else 'None'}")
            logger.error(f"Source identity shape: {source_identity.shape if source_identity is not None else 'None'}")
            logger.error(f"Step: {step}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            zero_tensor = torch.tensor(0.0, device=device)
            return zero_tensor, zero_tensor, zero_tensor
        
    def compute_velocity_smoothness_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute velocity and smoothness regularization losses for temporal consistency.
        SMOOTHNESS LOSS DISABLED to avoid numerical instability.
        """
        try:
            device = self.device
            vel_loss = torch.tensor(0.0, device=device)
            smooth_loss = torch.tensor(0.0, device=device)  # Disabled
            
            # Process each motion component
            for key in ['theta', 'expression', 'scale', 'rotation', 'translation']:
                if key in pred and key in target and pred[key].shape[1] > 1:  # Need at least 2 frames for velocity
                    pred_val = pred[key]
                    target_val = target[key]
                    
                    # Special handling for theta (rotation matrix) - use safe Euler extraction
                    if key == 'theta' and pred_val.shape[-2:] == (3, 4):
                        # Extract rotation part and convert to Euler angles for stable gradients
                        pred_R = pred_val[..., :3, :3]
                        target_R = target_val[..., :3, :3]
                        
                        # Use safe Euler extraction
                        pred_euler = safe_matrix_to_euler(pred_R)  # [B, T, 3]
                        target_euler = safe_matrix_to_euler(target_R)  # [B, T, 3]
                        
                        # Also handle translation part
                        pred_trans = pred_val[..., :3, 3]  # [B, T, 3]
                        target_trans = target_val[..., :3, 3]  # [B, T, 3]
                        
                        # Compute velocity on Euler angles (more stable than matrix differences)
                        pred_euler_vel = pred_euler[:, 1:] - pred_euler[:, :-1]
                        target_euler_vel = target_euler[:, 1:] - target_euler[:, :-1]
                        vel_loss += F.mse_loss(pred_euler_vel, target_euler_vel) * self.lambda_velocity
                        
                        # Compute velocity on translation
                        pred_trans_vel = pred_trans[:, 1:] - pred_trans[:, :-1]
                        target_trans_vel = target_trans[:, 1:] - target_trans[:, :-1]
                        vel_loss += F.mse_loss(pred_trans_vel, target_trans_vel) * self.lambda_velocity
                        
                    else:
                        # For other parameters, flatten if needed
                        if len(pred_val.shape) > 3:
                            pred_val = pred_val.view(pred_val.shape[0], pred_val.shape[1], -1)
                            target_val = target_val.view(target_val.shape[0], target_val.shape[1], -1)
                        
                        # Velocity matching loss (first derivative)
                        pred_vel = pred_val[:, 1:] - pred_val[:, :-1]
                        target_vel = target_val[:, 1:] - target_val[:, :-1]
                        vel_loss += F.mse_loss(pred_vel, target_vel) * self.lambda_velocity
                    
                    # SMOOTHNESS LOSS DISABLED - causes numerical instability
                    # Second derivatives can amplify noise and cause gradient explosions
                    # if pred_val.shape[1] > 2:
                    #     pred_acc = pred_val[:, 2:] - 2*pred_val[:, 1:-1] + pred_val[:, :-2]
                    #     target_acc = target_val[:, 2:] - 2*target_val[:, 1:-1] + target_val[:, :-2]
                    #     smooth_loss += F.mse_loss(pred_acc, target_acc) * self.lambda_smoothness
            
            # Return only velocity loss (smoothness is disabled)
            return vel_loss  # smooth_loss is always 0
            
        except Exception as e:
            import traceback
            logger.error(f"Error in velocity/smoothness loss: {str(e)}")
            logger.error(f"Pred keys: {pred.keys() if pred else 'None'}")
            logger.error(f"Target keys: {target.keys() if target else 'None'}")
            if pred and 'theta' in pred:
                logger.error(f"Pred theta shape: {pred['theta'].shape}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return torch.tensor(0.0, device=device)



