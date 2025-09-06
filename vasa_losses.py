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
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger
from syncnet import SyncNetInstance


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
        device: str = 'cuda'
    ):
        self.volumetric_avatar = volumetric_avatar
        self.config = config
        self.device = device
        

        self.speed_handler = SpeedLossHandler(num_buckets=9)

 # Initialize SyncNet evaluator
        self.syncnet = SyncNetInstance(device=device)
        self.syncnet.eval()
        
        self.batch_size =  config.train.batch_size
        # Extract loss weights from config
           # Extract loss weights from config
        self.lambda_pose = config.loss.lambda_pose
        self.lambda_dynamics = config.loss.lambda_dynamics
        self.lambda_gaze = config.loss.lambda_gaze_direction
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
        self.lambda_smoothness = getattr(config.loss, 'lambda_smoothness', 1e-4)
        
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
            pred_motion: Dictionary containing predicted motion parameters
            target_blinks: Target blink states [B, T, 3] tensor where:
                - Channel 0: Blink phase (0=open, 1=closing, 2=closed, 3=opening)
                - Channel 1: Left eye openness (0-1)
                - Channel 2: Right eye openness (0-1)
            lambda_blink: Weight for blink loss
            device: Computation device
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        try:
            metrics = {}
            
            # Extract predicted eye landmarks
            left_eye = pred_motion['left_eye']   # [B, T, 7, 3]  
            right_eye = pred_motion['right_eye'] # [B, T, 8, 3]
            
            B, T = left_eye.shape[:2]

            # Compute eye aspect ratios for predictions
            def compute_ear(eye_landmarks: torch.Tensor) -> torch.Tensor:
                """Compute Eye Aspect Ratio from landmarks."""
                # For left eye: vertical distances between landmarks
                A = torch.norm(eye_landmarks[..., 1, :] - eye_landmarks[..., 5, :], dim=-1)
                B = torch.norm(eye_landmarks[..., 2, :] - eye_landmarks[..., 4, :], dim=-1)
                
                # Horizontal distance
                C = torch.norm(eye_landmarks[..., 0, :] - eye_landmarks[..., 3, :], dim=-1)
                
                # Compute EAR with stability term
                return (A + B) / (2.0 * C.clamp(min=1e-6))

            # Get predicted eye openness
            left_ear = compute_ear(left_eye)    # [B, T]
            right_ear = compute_ear(right_eye)   # [B, T]
            
            # Normalize to [0,1] range
            left_openness = torch.clamp(left_ear / 0.3, 0, 1)   # [B, T]
            right_openness = torch.clamp(right_ear / 0.3, 0, 1) # [B, T]

            # Get target values
            target_phase = target_blinks[..., 0]      # [B, T]
            target_left = target_blinks[..., 1]       # [B, T]
            target_right = target_blinks[..., 2]      # [B, T]

            # Phase matching loss
            pred_phase = torch.zeros((B, T, 4), device=device)  # 4 phases
            
            # Fix: Pad the diff results to match sequence length
            is_closed = (left_openness < 0.2) | (right_openness < 0.2)
            
            # Compute diffs and pad
            left_diff = F.pad(left_openness.diff(dim=1), (0, 1))  # Pad right
            right_diff = F.pad(right_openness.diff(dim=1), (0, 1))  # Pad right
            
            is_opening = ~is_closed & (left_diff > 0.1)
            is_closing = ~is_closed & (left_diff < -0.1)

            pred_phase[..., 0] = ~(is_closed | is_opening | is_closing)  # Open
            pred_phase[..., 1] = is_closing
            pred_phase[..., 2] = is_closed
            pred_phase[..., 3] = is_opening
            
            phase_loss = F.cross_entropy(
                pred_phase.view(-1, 4),
                target_phase.long().view(-1)
            )
            
            # Openness matching loss
            left_loss = F.mse_loss(left_openness, target_left)
            right_loss = F.mse_loss(right_openness, target_right)
            
            # Symmetry loss to encourage eyes to blink together
            symmetry_loss = F.mse_loss(left_openness, right_openness)
            
            # Temporal smoothness loss
            if T > 1:
                temp_loss = F.mse_loss(
                    left_openness[:, 1:] - left_openness[:, :-1],
                    target_left[:, 1:] - target_left[:, :-1]
                ) + F.mse_loss(
                    right_openness[:, 1:] - right_openness[:, :-1],
                    target_right[:, 1:] - target_right[:, :-1]
                )
            else:
                temp_loss = torch.tensor(0.0, device=device)
            
            # Combine losses
            total = (
                phase_loss * 0.3 +
                (left_loss + right_loss) * 0.3 +
                symmetry_loss * 0.2 +
                temp_loss * 0.2
            ) * lambda_blink
            
            # Store metrics
            metrics['blink_phase_loss'] = phase_loss
            metrics['blink_left_loss'] = left_loss
            metrics['blink_right_loss'] = right_loss
            metrics['blink_symmetry_loss'] = symmetry_loss
            metrics['blink_temporal_loss'] = temp_loss
            metrics['blink_total'] = total
            
            return total, metrics
            
        except Exception as e:
            logger.error(f"Error in blink loss: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=device), {}
    
    def visualize_sequence(self, pred_embed, target_embed, step, window_idx):
        """Visualize embedding sequences for debugging"""
        try:
            if step % self.vis_freq != 0:
                return
                
            # Convert tensors to numpy and get first sample
            pred = pred_embed[0].detach().cpu()  # [T, 128]
            target = target_embed[0].detach().cpu()  # [T, 128]
            
            # Create visualization
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # Plot predicted sequence
            axes[0].imshow(pred.T, aspect='auto', cmap='viridis')
            axes[0].set_title(f'Predicted Expression Embed (Step {step}, Window {window_idx})')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Dimension')
            
            # Plot target sequence
            axes[1].imshow(target.T, aspect='auto', cmap='viridis')
            axes[1].set_title('Target Expression Embed')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Dimension')
            
            # Plot difference
            diff = (pred - target).abs()
            im = axes[2].imshow(diff.T, aspect='auto', cmap='hot')
            axes[2].set_title('Absolute Difference')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Dimension')
            
            plt.colorbar(im, ax=axes[2])
            plt.tight_layout()
            
            # Save to wandb
            if wandb.run is not None:
                wandb.log({
                    f"expression_sequence/window_{window_idx}": wandb.Image(fig),
                    "step": step
                })
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not visualize sequence: {e}")

    # Continue with rest of VASALossModule methods...
    # [The rest of the VASALossModule class methods would continue here]
    # Due to length constraints, I'll create a continuation pattern            right_ear = compute_ear(right_eye)   # [B, T]
            
            # Normalize to [0,1] range
            left_openness = torch.clamp(left_ear / 0.3, 0, 1)   # [B, T]
            right_openness = torch.clamp(right_ear / 0.3, 0, 1) # [B, T]

            # Get target values
            target_phase = target_blinks[..., 0]      # [B, T]
            target_left = target_blinks[..., 1]       # [B, T]
            target_right = target_blinks[..., 2]      # [B, T]

            # Phase matching loss
            pred_phase = torch.zeros((B, T, 4), device=device)  # 4 phases
            
            # Fix: Pad the diff results to match sequence length
            is_closed = (left_openness < 0.2) | (right_openness < 0.2)
            
            # Compute diffs and pad
            left_diff = F.pad(left_openness.diff(dim=1), (0, 1))  # Pad right
            right_diff = F.pad(right_openness.diff(dim=1), (0, 1))  # Pad right
            
            is_opening = ~is_closed & (left_diff > 0.1)
            is_closing = ~is_closed & (left_diff < -0.1)

            pred_phase[..., 0] = ~(is_closed | is_opening | is_closing)  # Open
            pred_phase[..., 1] = is_closing
            pred_phase[..., 2] = is_closed
            pred_phase[..., 3] = is_opening
            
            phase_loss = F.cross_entropy(
                pred_phase.view(-1, 4),
                target_phase.long().view(-1)
            )
            
            # Openness matching loss
            left_loss = F.mse_loss(left_openness, target_left)
            right_loss = F.mse_loss(right_openness, target_right)
            
            # Symmetry loss to encourage eyes to blink together
            symmetry_loss = F.mse_loss(left_openness, right_openness)
            
            # Temporal smoothness loss
            if T > 1:
                temp_loss = F.mse_loss(
                    left_openness[:, 1:] - left_openness[:, :-1],
                    target_left[:, 1:] - target_left[:, :-1]
                ) + F.mse_loss(
                    right_openness[:, 1:] - right_openness[:, :-1],
                    target_right[:, 1:] - target_right[:, :-1]
                )
            else:
                temp_loss = torch.tensor(0.0, device=device)
                
            # Combine losses
            total_loss = (
                phase_loss + 
                left_loss + 
                right_loss + 
                0.5 * symmetry_loss +
                0.2 * temp_loss
            ) * lambda_blink
            
            # Record metrics
            metrics.update({
                'blink_phase_loss': phase_loss.item(),
                'blink_left_loss': left_loss.item(),
                'blink_right_loss': right_loss.item(),
                'blink_symmetry_loss': symmetry_loss.item(),
                'blink_temporal_loss': temp_loss.item(),
                'blink_total': total_loss.item()
            })
            
            return total_loss, metrics
            
        except Exception as e:
            logger.error(f"Error computing blink loss: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=device), {
                'blink_total': 0.0,
                'blink_error': str(e)
            }
    

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
                    
                    # Compute velocity loss for smoother lip motion
                    lips_vel_pred = pred_lips[:, 1:] - pred_lips[:, :-1]
                    lips_vel_target = target_lips[:, 1:] - target_lips[:, :-1]
                    lips_vel_loss = F.mse_loss(lips_vel_pred, lips_vel_target)
                    
                    # Combined lip loss
                    lips_loss = self.lambda_lips * (lips_pos_loss + 0.5 * lips_vel_loss)
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
        target_frames: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses including expression verification and perceptual loss."""
        try:
            logger.info("\n=== Computing Losses ===")
            losses = {}
            metrics = {}
            device = outputs['theta'].device

            # 1. Reconstruction losses
            logger.debug("\nComputing reconstruction losses:")
            recon_losses = self._compute_reconstruction_losses(outputs, targets, noise, step)
            losses.update(recon_losses)
            logger.debug("Reconstruction losses:")
            for k, v in recon_losses.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  {k}: {v.item():.6f}")

            # 2. Expression Verification Loss
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
                control_losses = self._compute_control_losses(outputs, conditions, current_epoch)
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
            
            if self.config.loss.use_sync_loss:
                logger.debug("Computing sync loss...")
                sync_loss = self._compute_sync_loss(outputs, targets)
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
            
            if has_pose and has_dynamics:
                # Pass motion outputs for disentanglement computation
                disentangle_loss, l_consist, l_cross_id = self.compute_disentanglement_loss(
                    motion_outputs=outputs,
                    target_frames=target_frames
                )
            else:
                # Missing required outputs for disentanglement
                logger.info(f"DISENTANGLE: Missing required outputs (pose={has_pose}, dynamics={has_dynamics})")
                disentangle_loss = torch.tensor(0.0, device=device)
                l_consist = torch.tensor(0.0, device=device)
                l_cross_id = torch.tensor(0.0, device=device)
            losses['disentangle'] = disentangle_loss
            losses['l_consist'] = l_consist  # Store individual components
            losses['l_cross_id'] = l_cross_id  # Store individual components
            logger.debug(f"Disentanglement loss: {disentangle_loss.item():.6f}")
            logger.debug(f"  L_consist: {l_consist.item():.6f}")
            logger.debug(f"  L_cross_id: {l_cross_id.item():.6f}")
            logger.info(f"DISENTANGLE METRICS - l_consist: {l_consist.item():.6f}, l_cross_id: {l_cross_id.item():.6f}")
            
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
            disentangle_term = losses.get('disentangle', torch.tensor(0.0, device=device))
            vel_smooth_term = losses.get('velocity_smoothness', torch.tensor(0.0, device=device))

            logger.debug(f"  Reconstruction term: {recon_term.item():.6f}")
            logger.debug(f"  Verification term: {verify_term.item():.6f}")
            logger.debug(f"  Control term: {control_term.item():.6f}")
            logger.debug(f"  Sync term: {sync_term.item():.6f}")
            logger.debug(f"  Disentangle term: {disentangle_term.item():.6f}")
            logger.debug(f"  Velocity/smoothness term: {vel_smooth_term.item():.6f}")
            
            # Get diversity term
            diversity_term = losses.get('motion_diversity', torch.tensor(0.0, device=device))
            logger.debug(f"  Diversity term: {diversity_term.item():.6f}")
            
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
                    
                except Exception as e:
                    logger.warning(f"Could not compute perceptual loss: {e}")
                    losses['perceptual'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No frames provided for perceptual loss")
                losses['perceptual'] = torch.tensor(0.0, device=device)

            total_loss = recon_term + verify_term + control_term + sync_term + disentangle_term + vel_smooth_term + diversity_term + perceptual_term
            losses['total'] = total_loss
            logger.debug(f"Total loss (with perceptual): {total_loss.item():.6f}")

            # Return results
            if return_metrics:
                metrics.update({k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
                return losses, metrics

            return losses

        except Exception as e:
            logger.error(f"Error in compute_losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {'total': torch.tensor(1.0, device=device)}

        

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
                
                return frame.detach()  # Ensure final output is detached

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
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
            if 'theta' in pred:
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

            # 2. Scale loss
            if 'scale' in pred:
                losses['scale_loss'] = F.mse_loss(
                    pred['scale'],
                    comparison_target['scale']
                )

            # 3. Rotation loss
            if 'rotation' in pred:
                losses['rotation_loss'] = F.mse_loss(
                    pred['rotation'],
                    comparison_target['rotation']
                )

            # 4. Translation loss
            if 'translation' in pred:
                losses['translation_loss'] = F.mse_loss(
                    pred['translation'],
                    comparison_target['translation']
                )

            # 5. Expression loss 
            if 'expression_embed' in pred:
                # Apply dimension weights

                losses['expression_loss'] = F.mse_loss(
                            pred['expression_embed'],
                            comparison_target['expression_embed']
                        )
         
                should_visualize = step > 0 and step % self.vis_freq == 0
                if should_visualize:
                    self.visualize_sequence(pred['expression_embed'],target['expression_embed'],step,0)


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
            pose_loss = (
                losses.get('theta_loss', torch.tensor(0.0, device=device)) +
                losses.get('scale_loss', torch.tensor(0.0, device=device)) +
                losses.get('rotation_loss', torch.tensor(0.0, device=device)) +
                losses.get('translation_loss', torch.tensor(0.0, device=device))
            ) * self.lambda_pose

            dynamics_loss = losses['expression_loss'] * self.lambda_dynamics


            # Motion smoothness loss if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            if pred['theta'].shape[1] > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred) * self.lambda_temporal

            # Combined reconstruction loss
            reconstruction_loss = pose_loss + dynamics_loss + motion_loss

            # Return all losses
            return {
                'reconstruction': reconstruction_loss,
                'pose_loss': pose_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': losses.get('theta_loss', torch.tensor(0.0, device=device)),
                'scale_loss': losses.get('scale_loss', torch.tensor(0.0, device=device)),
                'rotation_loss': losses.get('rotation_loss', torch.tensor(0.0, device=device)),
                'translation_loss': losses.get('translation_loss', torch.tensor(0.0, device=device)),
                'expression_loss': losses.get('expression_loss', torch.tensor(0.0, device=device))
            }

        except Exception as e:
            logger.error(f"Error computing reconstruction losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device, requires_grad=True),
                'pose_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'dynamics_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'motion_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'theta_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'scale_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'rotation_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'translation_loss': torch.tensor(0.0, device=device, requires_grad=True),
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

            # Process predictions and targets
            param_dims = {
                'theta': 12,      # 3x4 matrix flattened
                'scale': 3,    
                'rotation': 3,    
                'translation': 3,
                'expression_embed': 128
            }

            # Initialize losses and logging metrics
            losses = {f'{param}_loss': torch.tensor(0.0, device=device) for param in param_dims.keys()}
            metrics = {}
            
            # Process each parameter and compute losses
            for param, dim in param_dims.items():
                try:
                    if param == 'theta':
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
                            
                    else:
                        # Handle rotation and translation
                        losses[f'{param}_loss'] = F.mse_loss(
                            pred_motion[param],
                            comparison_target[param]
                        )
                        
                        # Log parameter statistics
                        if wandb.run is not None:
                            param_tensor = pred_motion[param]
                            metrics.update({
                                f'{mode_prefix}/{param}/mean': param_tensor.mean().item(),
                                f'{mode_prefix}/{param}/std': param_tensor.std().item(),
                                f'{mode_prefix}/{param}/min': param_tensor.min().item(),
                                f'{mode_prefix}/{param}/max': param_tensor.max().item(),
                                f'{mode_prefix}/{param}/loss': losses[f'{param}_loss'].item()
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing {param}: {str(e)}")
                    continue

            # Compute combined losses
            pose_loss = (
                losses['theta_loss'] +
                losses['rotation_loss'] + 
                losses['translation_loss'] +
                losses['scale_loss']
            ) * self.lambda_pose

            dynamics_loss = losses['expression_loss'] * self.lambda_dynamics

            # Compute motion smoothness if sequence length > 1
            motion_loss = torch.tensor(0.0, device=device)
            if pred_motion['theta'].shape[1] > 1:
                motion_loss = self._compute_motion_smoothness_loss(pred_motion)

            # Combined reconstruction loss
            reconstruction_loss = pose_loss + dynamics_loss + motion_loss

            # Log combined losses and detailed parameter statistics
           # Log combined losses and detailed parameter statistics
            if wandb.run is not None:
                # Base losses
                metrics.update({
                    f'{mode_prefix}/loss/pose': pose_loss.item(),
                    f'{mode_prefix}/loss/dynamics': dynamics_loss.item(),
                    f'{mode_prefix}/loss/motion': motion_loss.item(),
                    f'{mode_prefix}/loss/total': reconstruction_loss.item()
                })
                
                # Add detailed statistics for core motion parameters
                core_params = ['theta', 'rotation', 'scale', 'translation', 'expression_embed']
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

            # Return dictionary of all losses
            return {
                'reconstruction': reconstruction_loss,
                'pose_loss': pose_loss,
                'dynamics_loss': dynamics_loss,
                'motion_loss': motion_loss,
                'theta_loss': losses['theta_loss'],
                'rotation_loss': losses['rotation_loss'],
                'scale_loss': losses['scale_loss'],
                'translation_loss': losses['translation_loss'],
                'expression_loss': losses['expression_loss']
            }

        except Exception as e:
            logger.error(f"Error computing motion losses: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'reconstruction': torch.tensor(1.0, device=device),
                'pose_loss': torch.tensor(0.0, device=device),
                'dynamics_loss': torch.tensor(0.0, device=device),
                'motion_loss': torch.tensor(0.0, device=device),
                'theta_loss': torch.tensor(0.0, device=device),
                'rotation_loss': torch.tensor(0.0, device=device),
                'scale_loss': torch.tensor(0.0, device=device),
                'translation_loss': torch.tensor(0.0, device=device),
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
            
            # Compute velocity (first derivative)
            theta_vel = torch.diff(pred['theta'], dim=1)
            rotation_vel = torch.diff(pred['rotation'], dim=1)
            scale_vel = torch.diff(pred['scale'], dim=1)
            translation_vel = torch.diff(pred['translation'], dim=1)
            expression_vel = torch.diff(pred['expression_embed'], dim=1)
            
            # Compute acceleration (second derivative)
            if pred['theta'].shape[1] > 2:
                theta_acc = torch.diff(theta_vel, dim=1)
                rotation_acc = torch.diff(rotation_vel, dim=1)
                scale_acc = torch.diff(scale_vel, dim=1)
                translation_acc = torch.diff(translation_vel, dim=1)
                # expression_acc = torch.diff(expression_vel, dim=1)
            else:
                theta_acc = torch.zeros_like(theta_vel)
                rotation_acc = torch.zeros_like(rotation_vel)
                scale_acc = torch.zeros_like(scale_vel)
                translation_acc = torch.zeros_like(translation_vel)
                # expression_acc = torch.zeros_like(expression_vel)
            
            # Velocity smoothness
            loss += F.mse_loss(theta_vel, torch.zeros_like(theta_vel))
            loss += F.mse_loss(rotation_vel, torch.zeros_like(rotation_vel))
            loss += F.mse_loss(scale_vel, torch.zeros_like(scale_vel))
            loss += F.mse_loss(translation_vel, torch.zeros_like(translation_vel))
            # loss += F.mse_loss(expression_vel, torch.zeros_like(expression_vel))
            
            # Acceleration smoothness
            loss += 0.5 * F.mse_loss(theta_acc, torch.zeros_like(theta_acc))
            loss += 0.5 * F.mse_loss(rotation_acc, torch.zeros_like(rotation_acc))
            loss += 0.5 * F.mse_loss(scale_acc, torch.zeros_like(scale_acc))
            loss += 0.5 * F.mse_loss(translation_acc, torch.zeros_like(translation_acc))
            # loss += 0.5 * F.mse_loss(expression_acc, torch.zeros_like(expression_acc))
            
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
            logger.info(f"Got sequence of shape {pred_np.shape}, extracting first frame")
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

            # Extract frame data for each parameter
            param_data = {
                'theta': {
                    'pred': pred['theta'][:, frame_idx].reshape(-1),
                    'target': target['theta'][:, frame_idx].reshape(-1),
                    'title': 'Pose Matrix (θ)',
                    'dim': 12
                },
                'scale': {
                    'pred': pred['scale'][:, frame_idx],
                    'target': target['scale'][:, frame_idx],
                    'title': 'Scale',
                    'dim': 3
                },
                'rotation': {
                    'pred': pred['rotation'][:, frame_idx],
                    'target': target['rotation'][:, frame_idx],
                    'title': 'Rotation',
                    'dim': 3
                },
                'translation': {
                    'pred': pred['translation'][:, frame_idx],
                    'target': target['translation'][:, frame_idx],
                    'title': 'Translation',
                    'dim': 3
                },
                'expression': {
                    'pred': pred['expression_embed'][:, frame_idx],
                    'target': target['expression_embed'][:, frame_idx],
                    'title': 'Expression',
                    'dim': 128
                }
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
            
   
    def _compute_control_losses(
        self,
        pred_motion: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Compute control signal losses with improved signal checking."""
        try:
            logger.debug("\n=== Computing Control Losses ===")
            logger.debug(f"Current epoch: {epoch}, Control start epoch: {self.config.train.control_start_epoch}")
            
            # Initialize loss dict and get device
            device = pred_motion['theta'].device
            losses = {}
            total_loss = torch.tensor(0.0, device=device)
            
            # Log available control signals
            logger.debug("\nChecking available control signals:")
            control_signals = ['gaze', 'head_distance', 'emotion', 'speed_bucket']
            logger.debug("\nChecking available control signals:")
            for signal in control_signals:
                if signal in conditions and conditions[signal] is not None:
                    logger.debug(f"  Found {signal}: shape={conditions[signal].shape}")
                else:
                    logger.debug(f"  Missing or None: {signal}")

            logger.debug(f"\nUsing device: {device}")

            # 1. Gaze Loss
            logger.debug("\nComputing Gaze Loss:")
            if 'gaze' in conditions and conditions['gaze'] is not None:
                pred_gaze = self._extract_gaze_from_motion(pred_motion)
                if pred_gaze is not None:
                    logger.debug(f"  Predicted gaze shape: {pred_gaze.shape}")
                    target_gaze = conditions['gaze'].to(device).float()
                    logger.debug(f"  Target gaze shape: {target_gaze.shape}")
                    logger.debug(f"  Target gaze range: [{target_gaze.min():.3f}, {target_gaze.max():.3f}]")
                    
                    # Ensure gaze has sequence dimension
                    if len(target_gaze.shape) == 2:  # [B, 2]
                        target_gaze = target_gaze.unsqueeze(1).expand(-1, pred_gaze.shape[1], -1)
                        logger.debug(f"  Expanded target gaze shape: {target_gaze.shape}")
                    
                    gaze_loss = (1 - torch.cos(pred_gaze - target_gaze)).mean()
                    losses['control_gaze'] = gaze_loss * self.lambda_gaze
                    total_loss = total_loss + losses['control_gaze']
                    logger.debug(f"  Gaze loss: {losses['control_gaze'].item():.6f}")
                else:
                    logger.debug("  Failed to extract gaze - using zero loss")
                    losses['control_gaze'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No gaze signal found")
                losses['control_gaze'] = torch.tensor(0.0, device=device)

            # 2. Head Distance Loss
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

            # 3. Emotion Loss
            logger.debug("\nComputing Emotion Loss:")
            if 'emotion' in conditions and conditions['emotion'] is not None:
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
                logger.debug(f"  Emotion loss: {losses['control_emotion'].item():.6f}")
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


            # 4. Blink Loss 
            logger.debug("\nComputing Blink Loss:")
            if 'blink_state' in conditions and conditions['blink_state'] is not None:
                try:
                    blink_loss, blink_metrics = self._compute_blink_loss(
                        pred_motion,
                        conditions['blink_state'].to(device),
                        lambda_blink=self.lambda_blink,
                        device=device
                    )
                    losses.update(blink_metrics)
                    total_loss = total_loss + blink_loss
                    losses['control_blink'] = blink_loss
                    logger.debug(f"  Blink loss: {blink_loss.item():.6f}")
                except Exception as e:
                    logger.error(f"Error computing blink loss: {str(e)}")
                    logger.error(traceback.format_exc())
                    losses['control_blink'] = torch.tensor(0.0, device=device)
            else:
                logger.debug("  No blink_state signal found")
                losses['control_blink'] = torch.tensor(0.0, device=device)
     

            losses['control_total'] = total_loss


            # Add landmark losses if we have landmark conditions
            landmark_conditions = {k: v for k, v in conditions.items() if 'landmarks' in k}
            if landmark_conditions:
                landmark_losses = self._compute_landmark_losses(pred_motion, landmark_conditions)
                losses.update(landmark_losses)
                
                # Add landmark losses to total control loss
                if 'facial_motion_total' in landmark_losses:
                    losses['control_total'] = losses['control_total'] + landmark_losses['facial_motion_total']
            

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
            
    def _compute_motion_speed(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute motion speed logits from motion parameters."""
        B = motion['theta'].shape[0]
        T = motion['theta'].shape[1]
        device = motion['theta'].device
        
        # Get consecutive frame differences
        theta_diff = motion['theta'][:, 1:] - motion['theta'][:, :-1]
        speed = torch.norm(theta_diff.view(B, T-1, -1), dim=-1)  # [B, T-1]
        
        # Add zero for first frame
        speed = torch.cat([
            torch.zeros(B, 1, device=device),
            speed
        ], dim=1)  # [B, T]
        
        # Normalize to [-1, 1] range
        speed = torch.tanh(speed)
        
        # Convert to logits using bucket centers and radius
        centers = torch.tensor([-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0], device=device)
        radius = 0.1
        
        # Calculate distances to bucket centers
        speed = speed.unsqueeze(-1)  # [B, T, 1]
        centers = centers.view(1, 1, -1)  # [1, 1, num_buckets]
        
        # Convert to logits
        logits = -((speed - centers) / radius) ** 2
        
        return logits  # [B, T, num_buckets]

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
        """Extract gaze angles from motion parameters."""
        try:
            theta = motion['theta']  # [B, T, 3, 4]
            R = theta[..., :3, :3]  # Extract rotation part [B, T, 3, 3]
            
            # Compute pitch (x-axis rotation)
            pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1, 1))
            
            # Compute yaw (y-axis rotation)
            cos_pitch = torch.cos(pitch)
            yaw = torch.atan2(R[..., 2, 1] / cos_pitch, R[..., 2, 2] / cos_pitch)
            
            # Stack gaze angles
            gaze = torch.stack([pitch, yaw], dim=-1)  # [B, T, 2]
            
            # Validate outputs
            if torch.isnan(gaze).any() or torch.isinf(gaze).any():
                logger.warning("Invalid values in gaze angles")
                return None
                
            return gaze
            
        except Exception as e:
            logger.error(f"Error extracting gaze: {str(e)}")
            return None

    def _extract_distance_from_motion(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract head distance from translation."""
        # Use z-translation as proxy for head distance
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

    def _compute_motion_speed(self, motion: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute motion speed from consecutive frames."""
        # Compute pose difference between consecutive frames
        pose_diff = torch.norm(
            motion['theta'][:, 1:] - motion['theta'][:, :-1],
            dim=(-1, -2)
        )
        
        # Add dummy dimension for first frame
        speed = torch.cat([
            torch.zeros_like(pose_diff[:, :1]),
            pose_diff
        ], dim=1)
        
        return speed

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
                
                # Compute rotation loss using geodesic distance
                R_diff = torch.matmul(pred_R, target_R.transpose(-2, -1))
                trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
                cos_theta = (trace - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1, 1)  # Numerical stability
                rotation_loss = torch.acos(cos_theta).mean()
                
                # Compute translation loss (L2)
                translation_loss = F.mse_loss(pred_t, target_t)
                
                # Combine losses with weighting
                total_loss = rotation_loss + 0.5 * translation_loss
                
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
        target_frames: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VASA-1 style disentanglement losses as described in Section 3.1.
        
        Following VASA paper:
        - l_consist: Pairwise head pose and facial dynamics transfer loss
        - l_cross_id: Face identity similarity loss for cross-identity transfer
        """
        try:
            device = self.device
            
            # Initialize losses
            l_consist = torch.tensor(0.0, device=device)
            l_cross_id = torch.tensor(0.0, device=device)
            
            # Check if we have the required outputs
            if 'theta' not in motion_outputs:
                logger.info("DISENTANGLE: No theta (pose) in outputs, skipping")
                return l_consist, l_consist, l_consist
            
            B, T = motion_outputs['theta'].shape[:2]
            
            if T < 2:
                logger.info(f"DISENTANGLE: Sequence too short (T={T}), need at least 2 frames")
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
                # Sample two different time points
                idx_i = 0  # First frame
                idx_j = T - 1  # Last frame
                
                # Get pose (theta) and dynamics at different times
                pose_i = motion_outputs['theta'][:, idx_i]  # [B, 3, 4]
                pose_j = motion_outputs['theta'][:, idx_j]
                dyn_i = motion_outputs[dynamics_key][:, idx_i]  # [B, D]
                dyn_j = motion_outputs[dynamics_key][:, idx_j]
                
                # Following VASA Eq in Section 3.1:
                # The discrepancy between swapped combinations should be minimized
                # This encourages disentanglement between pose and dynamics
                pose_diff = F.mse_loss(pose_i, pose_j.detach())
                dyn_diff = F.mse_loss(dyn_i, dyn_j.detach())
                
                l_consist = (pose_diff + dyn_diff) * self.lambda_consist
                logger.info(f"DISENTANGLE: l_consist computed = {l_consist.item():.6f} (pose_diff={pose_diff.item():.4f}, dyn_diff={dyn_diff.item():.4f})")
            else:
                logger.info("DISENTANGLE: No dynamics (expression) found, l_consist = 0")
            
            # 2. L_cross_id: Identity preservation loss (VASA Section 3.1)
            # Ensures identity is preserved when transferring motion
            if self.id_extractor is not None and target_frames is not None:
                try:
                    # For cross-identity transfer, we need frames from different time points
                    # In VASA, this ensures identity is preserved during motion transfer
                    if target_frames.shape[1] >= 2:
                        frames_i = target_frames[:, 0]   # Frame at time i
                        frames_j = target_frames[:, -1]  # Frame at time j
                        
                        # Resize frames for identity extractor (expects 160x160)
                        frames_i_resized = F.interpolate(
                            frames_i, size=(160, 160), 
                            mode='bilinear', align_corners=False
                        )
                        frames_j_resized = F.interpolate(
                            frames_j, size=(160, 160),
                            mode='bilinear', align_corners=False
                        )
                        
                        with torch.no_grad():
                            # Extract identity features
                            id_feat_i = self.id_extractor(frames_i_resized)
                            id_feat_j = self.id_extractor(frames_j_resized)
                        
                        # Following VASA: maximize cosine similarity to preserve identity
                        # Use 1 - cosine_similarity as loss (minimize to maximize similarity)
                        similarity = F.cosine_similarity(id_feat_i, id_feat_j, dim=1).mean()
                        l_cross_id = (1 - similarity) * self.lambda_cross_id
                        
                        logger.info(f"DISENTANGLE: l_cross_id computed = {l_cross_id.item():.6f} (similarity={similarity.item():.4f})")
                    else:
                        logger.info(f"DISENTANGLE: Not enough frames for l_cross_id (shape={target_frames.shape})")
                    
                except Exception as e:
                    logger.debug(f"Cross-id loss computation skipped: {e}")
                    l_cross_id = torch.tensor(0.0, device=device)
            else:
                logger.info(f"DISENTANGLE: id_extractor={self.id_extractor is not None}, target_frames={target_frames is not None}")
            
            # Return both the total and individual components
            total_disentangle = l_consist + l_cross_id
            
            logger.info(
                f"VASA Disentanglement - l_consist: {l_consist.item():.6f}, "
                f"l_cross_id: {l_cross_id.item():.6f}, "
                f"total: {total_disentangle.item():.6f}"
            )
            
            return total_disentangle, l_consist, l_cross_id
            
        except Exception as e:
            logger.error(f"Error in VASA disentanglement loss: {str(e)}")
            zero_tensor = torch.tensor(0.0, device=device)
            return zero_tensor, zero_tensor, zero_tensor
    
    def compute_velocity_smoothness_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute velocity and smoothness regularization losses for temporal consistency.
        """
        try:
            device = self.device
            vel_loss = torch.tensor(0.0, device=device)
            smooth_loss = torch.tensor(0.0, device=device)
            
            # Process each motion component
            for key in ['theta', 'expression', 'scale', 'rotation', 'translation']:
                if key in pred and key in target and pred[key].shape[1] > 2:
                    pred_val = pred[key]
                    target_val = target[key]
                    
                    # Flatten if needed
                    if len(pred_val.shape) > 3:
                        pred_val = pred_val.view(pred_val.shape[0], pred_val.shape[1], -1)
                        target_val = target_val.view(target_val.shape[0], target_val.shape[1], -1)
                    
                    # Velocity matching loss (first derivative)
                    pred_vel = pred_val[:, 1:] - pred_val[:, :-1]
                    target_vel = target_val[:, 1:] - target_val[:, :-1]
                    vel_loss += F.mse_loss(pred_vel, target_vel) * self.lambda_velocity
                    
                    # Smoothness loss (second derivative/acceleration)
                    if pred_val.shape[1] > 2:
                        pred_acc = pred_val[:, 2:] - 2*pred_val[:, 1:-1] + pred_val[:, :-2]
                        target_acc = target_val[:, 2:] - 2*target_val[:, 1:-1] + target_val[:, :-2]
                        smooth_loss += F.mse_loss(pred_acc, target_acc) * self.lambda_smoothness
            
            return vel_loss + smooth_loss
            
        except Exception as e:
            logger.error(f"Error in velocity/smoothness loss: {str(e)}")
            return torch.tensor(0.0, device=device)



