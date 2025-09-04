#!/usr/bin/env python3
"""
Differentiable TDD Loss Module
===============================
Uses differentiable operations to compute test-based losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DifferentiableTDDLoss(nn.Module):
    """
    Fully differentiable TDD loss computation.
    Each test contributes to loss based on how far it is from passing.
    """
    
    def __init__(self, config: dict, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Define target values and weights for each test
        self.register_buffer('motion_var_target', torch.tensor(0.1))
        self.register_buffer('motion_var_min', torch.tensor(0.01))
        self.register_buffer('motion_var_max', torch.tensor(0.5))
        
        self.register_buffer('rotation_var_target', torch.tensor(0.05))
        self.register_buffer('rotation_var_min', torch.tensor(0.001))
        self.register_buffer('rotation_var_max', torch.tensor(0.3))
        
        self.register_buffer('expr_var_target', torch.tensor(3.0))
        self.register_buffer('expr_var_min', torch.tensor(1.0))
        self.register_buffer('expr_var_max', torch.tensor(10.0))
        
        self.register_buffer('temporal_consistency_target', torch.tensor(0.9))
        self.register_buffer('temporal_consistency_min', torch.tensor(0.7))
        
        self.register_buffer('reconstruction_target', torch.tensor(0.02))
        self.register_buffer('reconstruction_max', torch.tensor(0.1))
        
        # Weights for each loss component
        self.weights = {
            'motion_variance': 10.0,
            'rotation_variance': 6.0,
            'expression_variation': 7.0,
            'temporal_consistency': 5.0,
            'velocity_smoothness': 3.0,
            'reconstruction': 15.0,
            'audio_sync': 8.0,
            'static_penalty': 20.0  # High penalty for static output
        }
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compute differentiable losses based on TDD principles.
        """
        losses = {}
        metrics = {}
        
        # 1. Motion Variance Loss - penalize if too static or too wild
        if outputs['theta'].shape[1] > 1:
            theta_var = torch.var(outputs['theta'], dim=1).mean()
            rotation_var = torch.var(outputs['rotation'], dim=1).mean()
            
            # Penalize being outside target range
            motion_var_loss = (
                F.relu(self.motion_var_min - theta_var) * 100 +  # Penalty for too static
                F.relu(theta_var - self.motion_var_max) * 10 +   # Penalty for too wild
                (theta_var - self.motion_var_target).abs()       # Distance from target
            )
            
            rotation_var_loss = (
                F.relu(self.rotation_var_min - rotation_var) * 100 +
                F.relu(rotation_var - self.rotation_var_max) * 10 +
                (rotation_var - self.rotation_var_target).abs()
            )
            
            losses['motion_variance'] = motion_var_loss * self.weights['motion_variance']
            losses['rotation_variance'] = rotation_var_loss * self.weights['rotation_variance']
            
            metrics['theta_var'] = theta_var.item()
            metrics['rotation_var'] = rotation_var.item()
        
        # 2. Static Penalty - strongly penalize no motion
        if outputs['theta'].shape[1] > 1:
            # Compute frame-to-frame differences
            theta_diff = torch.diff(outputs['theta'], dim=1)
            motion_magnitude = torch.norm(theta_diff, dim=-1).mean()
            
            # Heavy penalty if motion is near zero
            static_penalty = torch.exp(-motion_magnitude * 100)  # Exponential penalty for static
            losses['static_penalty'] = static_penalty * self.weights['static_penalty']
            
            metrics['motion_magnitude'] = motion_magnitude.item()
        
        # 3. Expression Variation Loss
        if 'expression_embed' in outputs and outputs['expression_embed'].shape[1] > 1:
            expr_diff = torch.diff(outputs['expression_embed'], dim=1)
            expr_var = torch.norm(expr_diff, dim=-1).mean()
            
            expr_var_loss = (
                F.relu(self.expr_var_min - expr_var) * 50 +
                F.relu(expr_var - self.expr_var_max) * 5 +
                (expr_var - self.expr_var_target).abs()
            )
            
            losses['expression_variation'] = expr_var_loss * self.weights['expression_variation']
            metrics['expr_var'] = expr_var.item()
        
        # 4. Temporal Consistency Loss
        if outputs['theta'].shape[1] > 2:
            # First-order smoothness (velocity should be consistent)
            velocity = torch.diff(outputs['theta'], dim=1)
            acceleration = torch.diff(velocity, dim=1)
            
            # Penalize high acceleration (jittery motion)
            smoothness_loss = torch.norm(acceleration, dim=-1).mean()
            losses['velocity_smoothness'] = smoothness_loss * self.weights['velocity_smoothness']
            
            # Ensure consecutive frames are similar but not identical
            frame_similarity = F.cosine_similarity(
                outputs['theta'][:, :-1].reshape(outputs['theta'].shape[0], outputs['theta'].shape[1]-1, -1),
                outputs['theta'][:, 1:].reshape(outputs['theta'].shape[0], outputs['theta'].shape[1]-1, -1),
                dim=-1
            ).mean()
            
            # Penalize if similarity is too low
            consistency_loss = F.relu(self.temporal_consistency_min - frame_similarity) * 10
            losses['temporal_consistency'] = consistency_loss * self.weights['temporal_consistency']
            
            metrics['frame_similarity'] = frame_similarity.item()
        
        # 5. Reconstruction Loss
        recon_loss = F.mse_loss(outputs['theta'], targets['theta'])
        for key in ['rotation', 'translation', 'scale']:
            if key in outputs and key in targets:
                recon_loss += F.mse_loss(outputs[key], targets[key])
        
        # Penalize high reconstruction error
        reconstruction_penalty = (
            F.relu(recon_loss - self.reconstruction_max) * 10 +
            (recon_loss - self.reconstruction_target).abs()
        )
        losses['reconstruction'] = reconstruction_penalty * self.weights['reconstruction']
        metrics['recon_error'] = recon_loss.item()
        
        # 6. Audio Synchronization Loss
        if 'audio_features' in conditions and conditions['audio_features'] is not None:
            # Compute audio energy
            audio_energy = torch.norm(conditions['audio_features'], dim=-1)
            
            # Motion should correlate with audio energy
            if outputs['theta'].shape[1] > 1:
                motion_energy = torch.norm(torch.diff(outputs['theta'], dim=1), dim=-1).mean(dim=-1)
                
                # Normalize energies
                audio_norm = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)
                motion_norm = (motion_energy - motion_energy.mean()) / (motion_energy.std() + 1e-6)
                
                # Correlation loss (want high correlation)
                min_len = min(audio_norm.shape[-1], motion_norm.shape[-1])
                correlation = (audio_norm[..., :min_len] * motion_norm[..., :min_len]).mean()
                audio_sync_loss = 1.0 - correlation  # Convert to loss
                
                losses['audio_sync'] = audio_sync_loss * self.weights['audio_sync']
                metrics['audio_correlation'] = correlation.item()
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        # Compute test pass/fail metrics
        test_results = self._evaluate_tests(metrics)
        
        return losses, {
            'metrics': metrics,
            'test_results': test_results,
            'passed_ratio': sum(test_results.values()) / len(test_results) if test_results else 0
        }
    
    def _evaluate_tests(self, metrics: Dict) -> Dict[str, bool]:
        """Evaluate which tests pass based on metrics"""
        results = {}
        
        if 'theta_var' in metrics:
            results['motion_variance'] = (
                metrics['theta_var'] >= 0.01 and 
                metrics['theta_var'] <= 0.5
            )
        
        if 'rotation_var' in metrics:
            results['rotation_variance'] = (
                metrics['rotation_var'] >= 0.001 and
                metrics['rotation_var'] <= 0.3
            )
        
        if 'expr_var' in metrics:
            results['expression_variation'] = (
                metrics['expr_var'] >= 1.0 and
                metrics['expr_var'] <= 10.0
            )
        
        if 'frame_similarity' in metrics:
            results['temporal_consistency'] = metrics['frame_similarity'] >= 0.7
        
        if 'recon_error' in metrics:
            results['reconstruction'] = metrics['recon_error'] <= 0.1
        
        if 'audio_correlation' in metrics:
            results['audio_sync'] = metrics['audio_correlation'] >= 0.3
        
        if 'motion_magnitude' in metrics:
            results['not_static'] = metrics['motion_magnitude'] >= 0.001
        
        return results


def train_with_differentiable_tdd():
    """Quick training script using differentiable TDD losses"""
    import sys
    from pathlib import Path
    from omegaconf import OmegaConf
    
    sys.path.insert(0, 'nemo')
    from vasa_model import VASAModel
    import importlib
    
    print("\n" + "="*60)
    print("Differentiable TDD Training")
    print("="*60)
    
    # Load config and models
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda().eval()
    
    model = VASAModel(config, volumetric_avatar).cuda().train()
    
    # Use differentiable TDD loss
    tdd_loss = DifferentiableTDDLoss(config, device='cuda')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    B, T = 2, 30
    best_score = 0
    
    print(f"\nTraining with B={B}, T={T}")
    print("-" * 40)
    
    for step in range(500):
        optimizer.zero_grad()
        
        # Create targets with realistic motion
        t = torch.linspace(0, 4*3.14159, T).cuda()
        targets = {
            'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda() * 0.3
        }
        
        # Add realistic motion
        targets['rotation'][:, :, 0] = torch.sin(t * 0.5) * 0.15  # Pitch
        targets['rotation'][:, :, 1] = torch.cos(t * 0.7) * 0.2   # Yaw
        targets['translation'][:, :, 2] = torch.sin(t * 0.3) * 0.02  # Z motion
        
        # Add motion to theta
        for b in range(B):
            for ti in range(T):
                rot_y = targets['rotation'][b, ti, 1]
                targets['theta'][b, ti, 0, 0] = torch.cos(rot_y)
                targets['theta'][b, ti, 0, 2] = torch.sin(rot_y)
                targets['theta'][b, ti, 2, 0] = -torch.sin(rot_y)
                targets['theta'][b, ti, 2, 2] = torch.cos(rot_y)
        
        # Audio with energy variations
        audio_features = torch.randn(B, T, 768).cuda()
        audio_features *= (1 + torch.sin(t * 2)).unsqueeze(0).unsqueeze(-1) * 0.5
        
        conditions = {
            'audio_features': audio_features,
            'gaze': torch.randn(B, T, 2).cuda() * 0.1,
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda() * 0.1
        }
        
        # Minimal noise
        noise_level = torch.ones(B, device='cuda') * 0.2
        noise = {key: torch.randn_like(val) * 0.05 for key, val in targets.items()}
        
        noisy_inputs = {}
        for key in targets.keys():
            if len(targets[key].shape) == 4:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1, 1)
            elif len(targets[key].shape) == 3:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1)
        
        # Forward pass
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Compute differentiable TDD losses
        losses, test_info = tdd_loss.compute_losses(
            outputs=outputs,
            targets=targets,
            conditions=conditions
        )
        
        # Backward pass
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track progress
        pass_rate = test_info['passed_ratio']
        if pass_rate > best_score:
            best_score = pass_rate
            print(f"Step {step:3d}: Loss={losses['total'].item():7.2f}, Tests={pass_rate:5.1%} ⬆️ NEW BEST")
            
            # Save if good
            if pass_rate > 0.5:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'test_score': best_score,
                    'step': step
                }
                Path("checkpoints/tdd").mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, "checkpoints/tdd/differentiable_tdd.pth")
        elif step % 50 == 0:
            metrics = test_info.get('metrics', {})
            print(f"Step {step:3d}: Loss={losses['total'].item():7.2f}, Tests={pass_rate:5.1%}")
            if metrics:
                print(f"  Motion: {metrics.get('motion_magnitude', 0):.4f}, "
                      f"Recon: {metrics.get('recon_error', 0):.4f}")
    
    print(f"\n✅ Training complete!")
    print(f"Best test pass rate: {best_score:.1%}")
    
    # Show final test results
    if test_info.get('test_results'):
        print("\nFinal test results:")
        for name, passed in test_info['test_results'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")


if __name__ == "__main__":
    train_with_differentiable_tdd()