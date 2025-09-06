#!/usr/bin/env python3
"""
TDD Progressive Loss System for VASA
=====================================
Losses unlock progressively as training milestones are achieved.
Each loss has acceptance criteria that must be met before the next unlocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LossStage(Enum):
    """Progressive loss stages"""
    FOUNDATION = "foundation"  # Basic reconstruction
    BLINKING = "blinking"      # Eye blinking control
    EYE_GAZE = "eye_gaze"      # Eye gaze direction
    HEAD_POSE = "head_pose"    # Head pose control
    EMOTION = "emotion"        # Emotional expression
    LIP_SYNC = "lip_sync"      # Audio-visual sync
    FINE_TUNE = "fine_tune"    # All losses active


@dataclass
class LossThreshold:
    """Defines when a loss component should activate"""
    name: str
    stage: LossStage
    prerequisite_loss: Optional[str] = None
    activation_threshold: float = 0.1  # Previous loss must be below this
    min_epoch: int = 0  # Minimum epoch before activation
    weight_start: float = 0.0  # Initial weight when activated
    weight_target: float = 1.0  # Target weight after ramp-up
    ramp_epochs: int = 5  # Epochs to ramp up weight
    test_metric: str = ""  # Metric to test for success
    test_threshold: float = 0.0  # Threshold for test metric
    is_active: bool = False
    current_weight: float = 0.0
    activation_epoch: Optional[int] = None


class TDDProgressiveLoss(nn.Module):
    """
    Progressive loss system that unlocks components based on training progress.
    Implements TDD principles with clear acceptance criteria.
    """
    
    def __init__(self, config: dict, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.metrics_history = []
        
        # Define loss progression
        self.loss_stages = self._define_loss_stages()
        
        # Track current stage
        self.current_stage = LossStage.FOUNDATION
        
        # Eye landmark indices for MediaPipe (468 landmarks)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]  # Right eye landmarks
        
    def _define_loss_stages(self) -> Dict[str, LossThreshold]:
        """Define progressive loss thresholds"""
        return {
            # Stage 1: Foundation
            'reconstruction': LossThreshold(
                name='reconstruction',
                stage=LossStage.FOUNDATION,
                prerequisite_loss=None,
                activation_threshold=None,  # Always active
                min_epoch=0,
                weight_start=1.0,
                weight_target=1.0,
                ramp_epochs=0,
                test_metric='l1_error',
                test_threshold=0.05,
                is_active=True,
                current_weight=1.0
            ),
            
            'dynamics': LossThreshold(
                name='dynamics',
                stage=LossStage.FOUNDATION,
                prerequisite_loss=None,
                activation_threshold=None,
                min_epoch=0,
                weight_start=10.0,
                weight_target=10.0,
                ramp_epochs=0,
                test_metric='expression_variance',
                test_threshold=0.01,
                is_active=True,
                current_weight=10.0
            ),
            
            # Stage 2: Blinking Control
            'blink_control': LossThreshold(
                name='blink_control',
                stage=LossStage.BLINKING,
                prerequisite_loss='reconstruction',
                activation_threshold=0.1,  # Reconstruction loss must be < 0.1
                min_epoch=5,
                weight_start=0.1,
                weight_target=2.0,
                ramp_epochs=10,
                test_metric='blink_accuracy',
                test_threshold=0.8,  # 80% blink detection accuracy
                is_active=False,
                current_weight=0.0
            ),
            
            'blink_naturalness': LossThreshold(
                name='blink_naturalness',
                stage=LossStage.BLINKING,
                prerequisite_loss='blink_control',
                activation_threshold=0.2,
                min_epoch=10,
                weight_start=0.05,
                weight_target=1.0,
                ramp_epochs=5,
                test_metric='blink_duration',
                test_threshold=0.15,  # 150ms average blink duration
                is_active=False,
                current_weight=0.0
            ),
            
            # Stage 3: Eye Gaze
            'eye_gaze_direction': LossThreshold(
                name='eye_gaze_direction',
                stage=LossStage.EYE_GAZE,
                prerequisite_loss='blink_control',
                activation_threshold=0.15,
                min_epoch=15,
                weight_start=0.1,
                weight_target=1.5,
                ramp_epochs=10,
                test_metric='gaze_accuracy',
                test_threshold=0.7,  # 70% gaze direction accuracy
                is_active=False,
                current_weight=0.0
            ),
            
            'eye_coordination': LossThreshold(
                name='eye_coordination',
                stage=LossStage.EYE_GAZE,
                prerequisite_loss='eye_gaze_direction',
                activation_threshold=0.2,
                min_epoch=20,
                weight_start=0.05,
                weight_target=0.5,
                ramp_epochs=5,
                test_metric='eye_sync',
                test_threshold=0.9,  # 90% eye synchronization
                is_active=False,
                current_weight=0.0
            ),
            
            # Stage 4: Head Pose
            'head_pose': LossThreshold(
                name='head_pose',
                stage=LossStage.HEAD_POSE,
                prerequisite_loss='eye_gaze_direction',
                activation_threshold=0.2,
                min_epoch=25,
                weight_start=0.1,
                weight_target=1.0,
                ramp_epochs=10,
                test_metric='pose_accuracy',
                test_threshold=0.8,
                is_active=False,
                current_weight=0.0
            ),
            
            # Stage 5: Emotion
            'emotion': LossThreshold(
                name='emotion',
                stage=LossStage.EMOTION,
                prerequisite_loss='head_pose',
                activation_threshold=0.25,
                min_epoch=35,
                weight_start=0.05,
                weight_target=0.8,
                ramp_epochs=10,
                test_metric='emotion_accuracy',
                test_threshold=0.6,
                is_active=False,
                current_weight=0.0
            ),
            
            # Stage 6: Lip Sync
            'lip_sync': LossThreshold(
                name='lip_sync',
                stage=LossStage.LIP_SYNC,
                prerequisite_loss='emotion',
                activation_threshold=0.3,
                min_epoch=45,
                weight_start=0.1,
                weight_target=2.0,
                ramp_epochs=15,
                test_metric='sync_accuracy',
                test_threshold=0.75,
                is_active=False,
                current_weight=0.0
            ),
        }
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Update epoch and check for loss activation"""
        self.current_epoch = epoch
        self.metrics_history.append(metrics)
        
        # Check each loss threshold
        for name, threshold in self.loss_stages.items():
            if not threshold.is_active:
                # Check if we should activate this loss
                if self._should_activate_loss(threshold, metrics):
                    self._activate_loss(threshold)
                    logger.info(f"🎯 Activated {name} loss at epoch {epoch}")
                    logger.info(f"   Stage: {threshold.stage.value}")
                    logger.info(f"   Initial weight: {threshold.weight_start}")
            else:
                # Update weight if in ramp-up period
                if threshold.activation_epoch is not None:
                    epochs_since_activation = epoch - threshold.activation_epoch
                    if epochs_since_activation < threshold.ramp_epochs:
                        progress = epochs_since_activation / threshold.ramp_epochs
                        threshold.current_weight = (
                            threshold.weight_start + 
                            (threshold.weight_target - threshold.weight_start) * progress
                        )
                    else:
                        threshold.current_weight = threshold.weight_target
    
    def _should_activate_loss(self, threshold: LossThreshold, metrics: Dict[str, float]) -> bool:
        """Check if a loss should be activated"""
        # Check minimum epoch
        if self.current_epoch < threshold.min_epoch:
            return False
        
        # Check prerequisite loss
        if threshold.prerequisite_loss:
            prereq = self.loss_stages.get(threshold.prerequisite_loss)
            if not prereq or not prereq.is_active:
                return False
            
            # Check if prerequisite loss is below threshold
            prereq_metric = metrics.get(threshold.prerequisite_loss, float('inf'))
            if prereq_metric > threshold.activation_threshold:
                return False
        
        return True
    
    def _activate_loss(self, threshold: LossThreshold):
        """Activate a loss component"""
        threshold.is_active = True
        threshold.activation_epoch = self.current_epoch
        threshold.current_weight = threshold.weight_start
        
        # Update current stage
        for stage in LossStage:
            stage_losses = [l for l in self.loss_stages.values() 
                          if l.stage == stage and l.is_active]
            if stage_losses:
                self.current_stage = stage
    
    def compute_blink_loss(self, pred_landmarks: torch.Tensor, 
                          target_blink: torch.Tensor) -> torch.Tensor:
        """
        Compute blink control loss using Eye Aspect Ratio (EAR)
        
        Args:
            pred_landmarks: Predicted facial landmarks [B, T, 468, 3]
            target_blink: Target blink states [B, T] (0=open, 1=closed)
        """
        B, T = target_blink.shape
        
        # Calculate Eye Aspect Ratio for predicted landmarks
        pred_ear = self._calculate_ear(pred_landmarks)  # [B, T]
        
        # Convert blink states to expected EAR values
        # Open eye: EAR ~0.3, Closed eye: EAR ~0.1
        target_ear = torch.where(target_blink > 0.5, 
                                torch.ones_like(target_blink) * 0.1,
                                torch.ones_like(target_blink) * 0.3)
        
        # L1 loss between predicted and target EAR
        blink_loss = F.l1_loss(pred_ear, target_ear)
        
        # Add temporal consistency loss for smooth blinking
        if T > 1:
            ear_diff = pred_ear[:, 1:] - pred_ear[:, :-1]
            temporal_loss = torch.abs(ear_diff).mean()
            blink_loss = blink_loss + 0.5 * temporal_loss
        
        return blink_loss
    
    def _calculate_ear(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate Eye Aspect Ratio (EAR) from landmarks
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        B, T, _, _ = landmarks.shape
        
        # Extract eye landmarks
        left_eye = landmarks[:, :, self.left_eye_indices, :]
        right_eye = landmarks[:, :, self.right_eye_indices, :]
        
        # Calculate EAR for left eye
        left_ear = self._ear_for_eye(left_eye)
        
        # Calculate EAR for right eye  
        right_ear = self._ear_for_eye(right_eye)
        
        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        return ear
    
    def _ear_for_eye(self, eye_points: torch.Tensor) -> torch.Tensor:
        """Calculate EAR for a single eye"""
        # Vertical distances
        A = torch.norm(eye_points[:, :, 1] - eye_points[:, :, 5], dim=-1)
        B = torch.norm(eye_points[:, :, 2] - eye_points[:, :, 4], dim=-1)
        
        # Horizontal distance
        C = torch.norm(eye_points[:, :, 0] - eye_points[:, :, 3], dim=-1)
        
        # EAR formula
        ear = (A + B) / (2.0 * C + 1e-6)
        
        return ear
    
    def compute_gaze_loss(self, pred_gaze: torch.Tensor, 
                         target_gaze: torch.Tensor) -> torch.Tensor:
        """
        Compute eye gaze direction loss
        
        Args:
            pred_gaze: Predicted gaze vectors [B, T, 2] (yaw, pitch)
            target_gaze: Target gaze vectors [B, T, 2]
        """
        # Angular loss
        gaze_loss = F.mse_loss(pred_gaze, target_gaze)
        
        # Add smoothness constraint
        if pred_gaze.shape[1] > 1:
            gaze_velocity = pred_gaze[:, 1:] - pred_gaze[:, :-1]
            smoothness_loss = torch.abs(gaze_velocity).mean()
            gaze_loss = gaze_loss + 0.3 * smoothness_loss
        
        return gaze_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute progressive losses
        
        Returns:
            Dictionary of individual losses and total weighted loss
        """
        losses = {}
        
        # Always compute foundation losses
        if 'reconstruction' in self.loss_stages and self.loss_stages['reconstruction'].is_active:
            if 'pred_img' in predictions and 'target_img' in targets:
                losses['reconstruction'] = F.l1_loss(
                    predictions['pred_img'], 
                    targets['target_img']
                )
        
        if 'dynamics' in self.loss_stages and self.loss_stages['dynamics'].is_active:
            if 'expression' in predictions:
                # Expression variance loss
                expr_var = torch.var(predictions['expression'], dim=1).mean()
                losses['dynamics'] = F.relu(0.01 - expr_var)  # Penalize low variance
        
        # Compute blink loss if active
        if self.loss_stages['blink_control'].is_active:
            if 'landmarks' in predictions and 'blink_state' in targets:
                losses['blink_control'] = self.compute_blink_loss(
                    predictions['landmarks'],
                    targets['blink_state']
                )
        
        # Compute gaze loss if active
        if self.loss_stages['eye_gaze_direction'].is_active:
            if 'gaze' in predictions and 'target_gaze' in targets:
                losses['eye_gaze_direction'] = self.compute_gaze_loss(
                    predictions['gaze'],
                    targets['target_gaze']
                )
        
        # Compute weighted total
        total_loss = torch.zeros(1, device=self.device)
        for name, loss_value in losses.items():
            if name in self.loss_stages:
                weight = self.loss_stages[name].current_weight
                total_loss = total_loss + weight * loss_value
                losses[f'{name}_weighted'] = weight * loss_value
        
        losses['total'] = total_loss.squeeze()
        
        # Add stage info
        losses['current_stage'] = self.current_stage.value
        losses['active_losses'] = sum(1 for l in self.loss_stages.values() if l.is_active)
        
        return losses
    
    def get_status(self) -> Dict[str, any]:
        """Get current status of loss progression"""
        status = {
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage.value,
            'active_losses': [],
            'pending_losses': [],
            'loss_weights': {}
        }
        
        for name, threshold in self.loss_stages.items():
            if threshold.is_active:
                status['active_losses'].append(name)
                status['loss_weights'][name] = threshold.current_weight
            else:
                status['pending_losses'].append({
                    'name': name,
                    'stage': threshold.stage.value,
                    'min_epoch': threshold.min_epoch,
                    'prerequisite': threshold.prerequisite_loss,
                    'threshold': threshold.activation_threshold
                })
        
        return status
    
    def test_criteria(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Test if current metrics meet acceptance criteria"""
        results = {}
        
        for name, threshold in self.loss_stages.items():
            if threshold.is_active and threshold.test_metric:
                metric_value = metrics.get(threshold.test_metric, float('inf'))
                passed = metric_value <= threshold.test_threshold
                results[name] = {
                    'passed': passed,
                    'metric': threshold.test_metric,
                    'value': metric_value,
                    'threshold': threshold.test_threshold
                }
        
        return results