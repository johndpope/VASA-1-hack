#!/usr/bin/env python3
"""
TDD-Driven Loss Module for VASA
================================
All losses are defined by tests that specify expected behavior.
Each loss component has acceptance criteria that must be met.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCriteria:
    """Defines acceptance criteria for a loss component"""
    name: str
    min_value: float = 0.0
    max_value: float = float('inf')
    target_value: Optional[float] = None
    tolerance: float = 0.1
    weight: float = 1.0
    must_pass: bool = False  # If True, training stops if test fails
    
    def evaluate(self, value: float) -> Tuple[bool, str]:
        """Check if value meets criteria"""
        if value < self.min_value:
            return False, f"{self.name}: {value:.4f} < min {self.min_value}"
        if value > self.max_value:
            return False, f"{self.name}: {value:.4f} > max {self.max_value}"
        if self.target_value is not None:
            if abs(value - self.target_value) > self.tolerance:
                return False, f"{self.name}: {value:.4f} not within {self.tolerance} of target {self.target_value}"
        return True, f"{self.name}: PASS ({value:.4f})"


class TDDLossModule(nn.Module):
    """
    Test-Driven Development Loss Module
    Each loss is driven by test criteria that define expected behavior
    """
    
    def __init__(self, config: dict, volumetric_avatar=None, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self.volumetric_avatar = volumetric_avatar
        
        # Define test criteria for each loss component
        self.test_criteria = self._define_test_criteria()
        
        # Track test results
        self.test_history = []
        self.failure_counts = {}
        
    def _define_test_criteria(self) -> Dict[str, TestCriteria]:
        """Define acceptance criteria for all loss components"""
        return {
            # Motion Quality Tests
            'motion_variance': TestCriteria(
                name='motion_variance',
                min_value=0.01,  # Must have some motion
                max_value=0.5,   # But not too much
                target_value=0.1,
                tolerance=0.05,
                weight=10.0,
                must_pass=False
            ),
            
            'frame_difference': TestCriteria(
                name='frame_difference',
                min_value=0.001,  # Frames must differ
                max_value=0.3,    # But remain coherent
                target_value=0.05,
                weight=5.0
            ),
            
            'optical_flow': TestCriteria(
                name='optical_flow',
                min_value=0.5,   # Minimum motion detected
                max_value=20.0,  # Maximum reasonable motion
                target_value=3.0,
                weight=8.0
            ),
            
            # Expression Tests
            'expression_variation': TestCriteria(
                name='expression_variation',
                min_value=1.0,   # Expressions must vary
                max_value=10.0,
                target_value=3.0,
                weight=7.0
            ),
            
            'expression_smoothness': TestCriteria(
                name='expression_smoothness',
                min_value=0.0,
                max_value=0.5,   # Smooth transitions
                target_value=0.1,
                weight=3.0
            ),
            
            # Pose Quality Tests
            'rotation_variance': TestCriteria(
                name='rotation_variance',
                min_value=0.001,
                max_value=0.3,
                target_value=0.05,
                weight=6.0
            ),
            
            'pose_stability': TestCriteria(
                name='pose_stability',
                min_value=0.0,
                max_value=0.1,   # Stable but not static
                target_value=0.02,
                weight=4.0
            ),
            
            # Temporal Coherence Tests
            'temporal_consistency': TestCriteria(
                name='temporal_consistency',
                min_value=0.7,   # High correlation between frames
                max_value=1.0,
                target_value=0.9,
                weight=5.0
            ),
            
            'velocity_smoothness': TestCriteria(
                name='velocity_smoothness',
                min_value=0.0,
                max_value=0.2,
                target_value=0.05,
                weight=3.0
            ),
            
            # Audio Sync Tests
            'audio_correlation': TestCriteria(
                name='audio_correlation',
                min_value=0.3,   # Must correlate with audio
                max_value=1.0,
                target_value=0.7,
                weight=8.0
            ),
            
            'lip_sync_accuracy': TestCriteria(
                name='lip_sync_accuracy',
                min_value=0.5,
                max_value=1.0,
                target_value=0.8,
                weight=10.0,
                must_pass=False  # Critical for quality
            ),
            
            # Reconstruction Quality Tests
            'reconstruction_error': TestCriteria(
                name='reconstruction_error',
                min_value=0.0,
                max_value=0.1,
                target_value=0.02,
                weight=15.0
            ),
            
            'perceptual_quality': TestCriteria(
                name='perceptual_quality',
                min_value=0.0,
                max_value=0.5,
                target_value=0.1,
                weight=12.0
            ),
        }
    
    def compute_motion_variance(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Motion parameters should vary over time"""
        if outputs['theta'].shape[1] <= 1:
            return 0.0
        
        # Compute variance across time dimension
        theta_var = torch.var(outputs['theta'], dim=1).mean()
        rotation_var = torch.var(outputs['rotation'], dim=1).mean()
        
        return (theta_var + rotation_var).item()
    
    def compute_frame_difference(self, frames: torch.Tensor) -> float:
        """Test: Consecutive frames should differ (no static video)"""
        if frames.shape[1] <= 1:
            return 0.0
        
        # Compute differences between consecutive frames
        frame_diffs = []
        for i in range(frames.shape[1] - 1):
            diff = torch.abs(frames[:, i] - frames[:, i+1]).mean()
            frame_diffs.append(diff)
        
        return torch.stack(frame_diffs).mean().item()
    
    def compute_optical_flow(self, frames: torch.Tensor) -> float:
        """Test: Optical flow should show motion"""
        if frames.shape[1] <= 1:
            return 0.0
        
        B, T, C, H, W = frames.shape
        total_flow = 0
        
        for b in range(B):
            for t in range(T - 1):
                frame1 = frames[b, t].cpu().numpy().transpose(1, 2, 0)
                frame2 = frames[b, t+1].cpu().numpy().transpose(1, 2, 0)
                
                # Convert to grayscale
                gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                total_flow += magnitude.mean()
        
        return total_flow / (B * (T - 1))
    
    def compute_expression_variation(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Facial expressions should vary throughout sequence"""
        if 'expression_embed' not in outputs:
            return 0.0
        
        expr = outputs['expression_embed']
        if expr.shape[1] <= 1:
            return 0.0
        
        # Compute pairwise distances between expressions
        expr_diffs = []
        for i in range(expr.shape[1] - 1):
            diff = torch.norm(expr[:, i] - expr[:, i+1], dim=-1).mean()
            expr_diffs.append(diff)
        
        return torch.stack(expr_diffs).mean().item()
    
    def compute_expression_smoothness(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Expression changes should be smooth (not jittery)"""
        if 'expression_embed' not in outputs or outputs['expression_embed'].shape[1] <= 2:
            return 0.0
        
        expr = outputs['expression_embed']
        
        # Compute second-order differences (acceleration)
        velocity = torch.diff(expr, dim=1)
        acceleration = torch.diff(velocity, dim=1)
        
        # Lower is smoother
        return torch.norm(acceleration, dim=-1).mean().item()
    
    def compute_rotation_variance(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Head rotation should vary (not static head)"""
        if 'rotation' not in outputs or outputs['rotation'].shape[1] <= 1:
            return 0.0
        
        return torch.var(outputs['rotation'], dim=1).mean().item()
    
    def compute_pose_stability(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Pose should be stable but not completely static"""
        if outputs['theta'].shape[1] <= 1:
            return 0.0
        
        # Compute frame-to-frame pose changes
        theta_changes = torch.diff(outputs['theta'], dim=1)
        
        # Want small but non-zero changes
        return torch.norm(theta_changes, dim=-1).mean().item()
    
    def compute_temporal_consistency(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Temporal coherence between frames"""
        if outputs['theta'].shape[1] <= 2:
            return 1.0
        
        # Compute correlation between consecutive frames
        correlations = []
        for key in ['theta', 'rotation', 'translation']:
            if key in outputs:
                data = outputs[key].reshape(outputs[key].shape[0], outputs[key].shape[1], -1)
                for i in range(data.shape[1] - 1):
                    # Compute correlation
                    frame1 = data[:, i].flatten()
                    frame2 = data[:, i+1].flatten()
                    if frame1.numel() > 0:
                        corr = torch.corrcoef(torch.stack([frame1, frame2]))[0, 1]
                        if not torch.isnan(corr):
                            correlations.append(corr)
        
        return torch.stack(correlations).mean().item() if correlations else 0.0
    
    def compute_velocity_smoothness(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Test: Velocity should be smooth (low acceleration)"""
        if outputs['theta'].shape[1] <= 2:
            return 0.0
        
        # Compute acceleration from theta
        velocity = torch.diff(outputs['theta'], dim=1)
        acceleration = torch.diff(velocity, dim=1)
        
        return torch.norm(acceleration, dim=-1).mean().item()
    
    def compute_audio_correlation(
        self, 
        outputs: Dict[str, torch.Tensor],
        audio_features: torch.Tensor
    ) -> float:
        """Test: Motion should correlate with audio energy"""
        if audio_features is None or outputs['theta'].shape[1] <= 1:
            return 0.0
        
        # Compute audio energy
        audio_energy = torch.norm(audio_features, dim=-1)  # [B, T]
        
        # Compute motion magnitude
        motion = torch.diff(outputs['theta'], dim=1)
        motion_energy = torch.norm(motion.reshape(motion.shape[0], motion.shape[1], -1), dim=-1)
        
        # Pad motion energy to match audio
        if motion_energy.shape[1] < audio_energy.shape[1]:
            motion_energy = F.pad(motion_energy, (0, 1), value=motion_energy[:, -1])
        
        # Compute correlation
        correlations = []
        for b in range(audio_energy.shape[0]):
            if audio_energy[b].numel() > 1:
                corr = torch.corrcoef(torch.stack([
                    audio_energy[b].flatten(),
                    motion_energy[b].flatten()
                ]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.abs())  # Absolute correlation
        
        return torch.stack(correlations).mean().item() if correlations else 0.0
    
    def compute_lip_sync_accuracy(
        self,
        outputs: Dict[str, torch.Tensor],
        audio_features: torch.Tensor
    ) -> float:
        """Test: Lip motion should sync with audio"""
        # Simplified version - in practice you'd use specialized lip-sync metrics
        if 'expression_embed' not in outputs or audio_features is None:
            return 0.0
        
        # Use lower part of expression embedding as proxy for lip motion
        lip_motion = outputs['expression_embed'][:, :, :32]  # First 32 dims for lips
        
        # Compute correlation with audio amplitude
        audio_amp = torch.norm(audio_features, dim=-1)
        lip_changes = torch.norm(torch.diff(lip_motion, dim=1), dim=-1)
        
        if lip_changes.shape[1] > 0 and audio_amp.shape[1] > 1:
            # Simple correlation as sync measure
            min_len = min(lip_changes.shape[1], audio_amp.shape[1] - 1)
            sync_score = F.cosine_similarity(
                lip_changes[:, :min_len].flatten(),
                audio_amp[:, :min_len].flatten(),
                dim=0
            )
            return (sync_score.abs().item() + 1.0) / 2.0  # Normalize to [0, 1]
        
        return 0.0
    
    def compute_reconstruction_error(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> float:
        """Test: Reconstruction should be accurate"""
        total_error = 0
        count = 0
        
        for key in ['theta', 'rotation', 'translation', 'scale']:
            if key in outputs and key in targets:
                error = F.mse_loss(outputs[key], targets[key])
                total_error += error
                count += 1
        
        return (total_error / count).item() if count > 0 else 0.0
    
    def compute_perceptual_quality(
        self,
        generated_frames: Optional[torch.Tensor],
        target_frames: Optional[torch.Tensor]
    ) -> float:
        """Test: Perceptual quality should be high"""
        if generated_frames is None or target_frames is None:
            return 0.0
        
        # Simple L1 + gradient loss as perceptual metric
        l1_loss = F.l1_loss(generated_frames, target_frames)
        
        # Compute gradient loss for sharpness
        if generated_frames.shape[1] > 1:
            gen_grad_x = torch.abs(generated_frames[:, :, :, 1:] - generated_frames[:, :, :, :-1])
            tgt_grad_x = torch.abs(target_frames[:, :, :, 1:] - target_frames[:, :, :, :-1])
            grad_loss = F.l1_loss(gen_grad_x, tgt_grad_x)
        else:
            grad_loss = 0
        
        return (l1_loss + grad_loss * 0.5).item()
    
    def run_tests(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        generated_frames: Optional[torch.Tensor] = None,
        target_frames: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, float], Dict[str, bool], List[str]]:
        """
        Run all tests and return metrics, pass/fail status, and messages
        """
        metrics = {}
        test_results = {}
        messages = []
        
        # Run motion tests
        metrics['motion_variance'] = self.compute_motion_variance(outputs)
        metrics['rotation_variance'] = self.compute_rotation_variance(outputs)
        metrics['pose_stability'] = self.compute_pose_stability(outputs)
        metrics['temporal_consistency'] = self.compute_temporal_consistency(outputs)
        metrics['velocity_smoothness'] = self.compute_velocity_smoothness(outputs)
        
        # Run expression tests
        metrics['expression_variation'] = self.compute_expression_variation(outputs)
        metrics['expression_smoothness'] = self.compute_expression_smoothness(outputs)
        
        # Run audio sync tests if audio provided
        if 'audio_features' in conditions and conditions['audio_features'] is not None:
            metrics['audio_correlation'] = self.compute_audio_correlation(
                outputs, conditions['audio_features']
            )
            metrics['lip_sync_accuracy'] = self.compute_lip_sync_accuracy(
                outputs, conditions['audio_features']
            )
        
        # Run reconstruction tests
        metrics['reconstruction_error'] = self.compute_reconstruction_error(outputs, targets)
        
        # Run frame-based tests if frames provided
        if generated_frames is not None:
            metrics['frame_difference'] = self.compute_frame_difference(generated_frames)
            metrics['optical_flow'] = self.compute_optical_flow(generated_frames)
            if target_frames is not None:
                metrics['perceptual_quality'] = self.compute_perceptual_quality(
                    generated_frames, target_frames
                )
        
        # Evaluate against criteria
        for name, value in metrics.items():
            if name in self.test_criteria:
                passed, message = self.test_criteria[name].evaluate(value)
                test_results[name] = passed
                messages.append(message)
                
                # Track failures
                if not passed:
                    self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
        
        return metrics, test_results, messages
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        noise: Dict[str, torch.Tensor],
        generated_frames: Optional[torch.Tensor] = None,
        target_frames: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        current_epoch: int = 0,
        step: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compute losses based on test results
        Each failing test contributes to the loss
        """
        # Run all tests
        metrics, test_results, messages = self.run_tests(
            outputs, targets, conditions, generated_frames, target_frames
        )
        
        losses = {}
        
        # Convert test results to losses
        for name, value in metrics.items():
            if name in self.test_criteria:
                criteria = self.test_criteria[name]
                
                # Compute loss based on distance from target/acceptable range
                if criteria.target_value is not None:
                    # L2 distance from target
                    loss = (value - criteria.target_value) ** 2
                elif value < criteria.min_value:
                    # Penalize being below minimum
                    loss = (criteria.min_value - value) ** 2 * 10  # High penalty
                elif value > criteria.max_value:
                    # Penalize being above maximum
                    loss = (value - criteria.max_value) ** 2 * 10  # High penalty
                else:
                    # Within acceptable range - small loss to encourage target
                    if criteria.target_value:
                        loss = abs(value - criteria.target_value) * 0.1
                    else:
                        loss = 0.0
                
                losses[name] = torch.tensor(loss, device=self.device) * criteria.weight
        
        # Special handling for must-pass tests
        for name, passed in test_results.items():
            if not passed and name in self.test_criteria:
                if self.test_criteria[name].must_pass:
                    # Critical failure - add large penalty
                    losses[f'{name}_critical'] = torch.tensor(100.0, device=self.device)
                    logger.error(f"CRITICAL TEST FAILURE: {name}")
        
        # Compute total loss
        losses['total'] = sum(losses.values())
        
        # Log test results periodically
        if step % 100 == 0:
            logger.info("\n=== TDD Test Results ===")
            passed_count = sum(test_results.values())
            total_count = len(test_results)
            logger.info(f"Tests: {passed_count}/{total_count} passed")
            
            for message in messages[:10]:  # Show first 10 messages
                if "PASS" in message:
                    logger.info(f"  ✓ {message}")
                else:
                    logger.warning(f"  ✗ {message}")
            
            # Log failure statistics
            if self.failure_counts:
                logger.info("\nTop failures:")
                sorted_failures = sorted(
                    self.failure_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for name, count in sorted_failures:
                    logger.info(f"  {name}: {count} failures")
        
        if return_metrics:
            return losses, {
                'test_metrics': metrics,
                'test_results': test_results,
                'passed_ratio': sum(test_results.values()) / len(test_results) if test_results else 0
            }
        
        return losses, {}
    
    def get_test_summary(self) -> str:
        """Generate a summary of test performance"""
        if not self.test_history:
            return "No tests run yet"
        
        recent = self.test_history[-100:]  # Last 100 test runs
        
        # Calculate pass rates
        pass_rates = {}
        for run in recent:
            for name, passed in run.items():
                if name not in pass_rates:
                    pass_rates[name] = []
                pass_rates[name].append(passed)
        
        summary = "\n=== TDD Test Summary ===\n"
        for name, results in pass_rates.items():
            pass_rate = sum(results) / len(results) * 100
            summary += f"{name}: {pass_rate:.1f}% pass rate\n"
        
        return summary