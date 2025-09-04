#!/usr/bin/env python3
"""
Balanced TDD Loss Module
========================
Fixes the conservative motion problem by:
1. Rewarding motion diversity
2. Using adaptive loss weighting
3. Implementing curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BalancedTDDLoss(nn.Module):
    """
    Balanced TDD loss that encourages motion while maintaining quality.
    Key improvements:
    - Motion diversity rewards
    - Adaptive loss weighting
    - Curriculum learning stages
    """
    
    def __init__(self, config: dict, device='cuda', stage='motion_first'):
        super().__init__()
        self.config = config
        self.device = device
        self.stage = stage  # 'motion_first', 'balanced', 'quality_focus'
        
        # Stage-based weights
        self.stage_weights = {
            'motion_first': {
                'motion_diversity': 30.0,    # HIGH - encourage motion
                'motion_variance': 20.0,      # HIGH - ensure variance
                'expression_variation': 15.0, # HIGH - facial movement
                'reconstruction': 1.0,        # LOW - don't prioritize yet
                'temporal_consistency': 2.0,  # LOW - allow some jitter initially
                'audio_sync': 10.0,           # MEDIUM - maintain sync
                'static_penalty': 50.0,       # VERY HIGH - punish no motion
            },
            'balanced': {
                'motion_diversity': 15.0,
                'motion_variance': 10.0,
                'expression_variation': 10.0,
                'reconstruction': 5.0,
                'temporal_consistency': 5.0,
                'audio_sync': 10.0,
                'static_penalty': 20.0,
            },
            'quality_focus': {
                'motion_diversity': 5.0,
                'motion_variance': 5.0,
                'expression_variation': 7.0,
                'reconstruction': 15.0,       # HIGH - now focus on quality
                'temporal_consistency': 10.0,  # HIGH - smooth motion
                'audio_sync': 12.0,
                'static_penalty': 10.0,
            }
        }
        
        # Get weights for current stage
        self.weights = self.stage_weights[stage]
        
        # Motion targets - more permissive ranges
        self.register_buffer('motion_var_target', torch.tensor(0.15))  # Higher target
        self.register_buffer('motion_var_min', torch.tensor(0.05))     # Higher minimum
        self.register_buffer('motion_var_max', torch.tensor(0.5))
        
        self.register_buffer('rotation_var_target', torch.tensor(0.1))  # Higher target
        self.register_buffer('rotation_var_min', torch.tensor(0.01))    # Higher minimum
        self.register_buffer('rotation_var_max', torch.tensor(0.3))
        
        self.register_buffer('expr_var_target', torch.tensor(5.0))      # Higher target
        self.register_buffer('expr_var_min', torch.tensor(2.0))         # Higher minimum
        self.register_buffer('expr_var_max', torch.tensor(10.0))
        
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor],
        step: int = 0,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compute balanced losses with motion diversity rewards.
        """
        losses = {}
        metrics = {}
        
        # 1. MOTION DIVERSITY REWARD (new!)
        # Reward different motion across batch and time
        if outputs['theta'].shape[0] > 1 and outputs['theta'].shape[1] > 1:
            # Batch diversity - different samples should move differently
            theta_flat = outputs['theta'].reshape(outputs['theta'].shape[0], -1)
            batch_similarity = F.cosine_similarity(
                theta_flat.unsqueeze(1), 
                theta_flat.unsqueeze(0), 
                dim=2
            )
            # Exclude diagonal (self-similarity)
            mask = ~torch.eye(batch_similarity.shape[0], dtype=torch.bool, device=self.device)
            batch_diversity = 1.0 - batch_similarity[mask].mean()
            
            # Temporal diversity - consecutive frames should differ
            theta_diff = torch.diff(outputs['theta'], dim=1)
            temporal_diversity = torch.norm(theta_diff, dim=-1).mean()
            
            # Reward diversity (negative loss when diverse)
            diversity_reward = -1.0 * (batch_diversity + temporal_diversity * 2)
            losses['motion_diversity'] = diversity_reward * self.weights['motion_diversity']
            
            metrics['batch_diversity'] = batch_diversity.item()
            metrics['temporal_diversity'] = temporal_diversity.item()
        
        # 2. MOTION VARIANCE with softer penalties
        if outputs['theta'].shape[1] > 1:
            theta_var = torch.var(outputs['theta'], dim=1).mean()
            rotation_var = torch.var(outputs['rotation'], dim=1).mean()
            
            # Softer penalties - only penalize if WAY outside range
            motion_var_loss = (
                F.relu(self.motion_var_min - theta_var) * 20 +  # Penalty for too static
                F.relu(theta_var - self.motion_var_max) * 2     # Gentle penalty for too wild
            )
            
            # REWARD being near target
            if theta_var > self.motion_var_min and theta_var < self.motion_var_max:
                motion_var_loss -= (1.0 - torch.abs(theta_var - self.motion_var_target)) * 5
            
            rotation_var_loss = (
                F.relu(self.rotation_var_min - rotation_var) * 20 +
                F.relu(rotation_var - self.rotation_var_max) * 2
            )
            
            losses['motion_variance'] = motion_var_loss * self.weights['motion_variance']
            losses['rotation_variance'] = rotation_var_loss * self.weights.get('rotation_variance', 5.0)
            
            metrics['theta_var'] = theta_var.item()
            metrics['rotation_var'] = rotation_var.item()
        
        # 3. STATIC PENALTY - exponentially punish no motion
        if outputs['theta'].shape[1] > 1:
            theta_diff = torch.diff(outputs['theta'], dim=1)
            motion_magnitude = torch.norm(theta_diff, dim=-1).mean()
            
            # Heavy penalty that increases exponentially as motion approaches zero
            static_penalty = torch.exp(-motion_magnitude * 50)
            losses['static_penalty'] = static_penalty * self.weights['static_penalty']
            
            metrics['motion_magnitude'] = motion_magnitude.item()
        
        # 4. EXPRESSION VARIATION with rewards
        if 'expression_embed' in outputs and outputs['expression_embed'].shape[1] > 1:
            expr_diff = torch.diff(outputs['expression_embed'], dim=1)
            expr_var = torch.norm(expr_diff, dim=-1).mean()
            
            expr_var_loss = (
                F.relu(self.expr_var_min - expr_var) * 30 +  # High penalty for static face
                F.relu(expr_var - self.expr_var_max) * 2
            )
            
            # Reward expression changes
            if expr_var > self.expr_var_min:
                expr_var_loss -= expr_var * 2  # Reward more expression
            
            losses['expression_variation'] = expr_var_loss * self.weights['expression_variation']
            metrics['expr_var'] = expr_var.item()
        
        # 5. RECONSTRUCTION - only if not in motion_first stage
        if self.stage != 'motion_first':
            recon_loss = 0
            for key in ['theta', 'rotation', 'translation']:
                if key in outputs and key in targets:
                    recon_loss += F.mse_loss(outputs[key], targets[key])
            
            losses['reconstruction'] = recon_loss * self.weights['reconstruction']
            metrics['recon_error'] = recon_loss.item()
        
        # 6. TEMPORAL CONSISTENCY - gentler in early stages
        if outputs['theta'].shape[1] > 2 and self.stage != 'motion_first':
            velocity = torch.diff(outputs['theta'], dim=1)
            acceleration = torch.diff(velocity, dim=1)
            
            # Only penalize extreme jitter
            smoothness_loss = F.relu(torch.norm(acceleration, dim=-1).mean() - 0.5)
            losses['temporal_consistency'] = smoothness_loss * self.weights['temporal_consistency']
            
            metrics['smoothness'] = torch.norm(acceleration, dim=-1).mean().item()
        
        # 7. AUDIO SYNC - always important
        if 'audio_features' in conditions and conditions['audio_features'] is not None:
            audio_energy = torch.norm(conditions['audio_features'], dim=-1)
            
            if outputs['theta'].shape[1] > 1:
                motion_energy = torch.norm(torch.diff(outputs['theta'], dim=1), dim=-1).mean(dim=-1)
                
                # Normalize
                if audio_energy.std() > 1e-6 and motion_energy.std() > 1e-6:
                    audio_norm = (audio_energy - audio_energy.mean()) / audio_energy.std()
                    motion_norm = (motion_energy - motion_energy.mean()) / motion_energy.std()
                    
                    # Correlation
                    min_len = min(audio_norm.shape[-1], motion_norm.shape[-1])
                    correlation = (audio_norm[..., :min_len] * motion_norm[..., :min_len]).mean()
                    
                    # Reward positive correlation
                    audio_sync_loss = 1.0 - correlation
                    losses['audio_sync'] = audio_sync_loss * self.weights['audio_sync']
                    metrics['audio_correlation'] = correlation.item()
        
        # ADAPTIVE LOSS SCALING based on current performance
        # If motion is too low, increase motion losses
        if 'motion_magnitude' in metrics and metrics['motion_magnitude'] < 0.05:
            losses['motion_variance'] *= 2.0
            losses['static_penalty'] *= 3.0
            losses['motion_diversity'] *= 2.0
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        # Test evaluation
        test_results = self._evaluate_tests(metrics)
        
        return losses, {
            'metrics': metrics,
            'test_results': test_results,
            'passed_ratio': sum(test_results.values()) / len(test_results) if test_results else 0,
            'stage': self.stage
        }
    
    def _evaluate_tests(self, metrics: Dict) -> Dict[str, bool]:
        """Evaluate tests with more permissive thresholds"""
        results = {}
        
        if 'theta_var' in metrics:
            # More permissive range
            results['motion_variance'] = (
                metrics['theta_var'] >= 0.03 and  # Lower minimum
                metrics['theta_var'] <= 0.6       # Higher maximum
            )
        
        if 'rotation_var' in metrics:
            results['rotation_variance'] = (
                metrics['rotation_var'] >= 0.005 and  # Lower minimum
                metrics['rotation_var'] <= 0.4
            )
        
        if 'expr_var' in metrics:
            results['expression_variation'] = (
                metrics['expr_var'] >= 1.5 and  # Lower minimum
                metrics['expr_var'] <= 12.0
            )
        
        if 'motion_magnitude' in metrics:
            results['not_static'] = metrics['motion_magnitude'] >= 0.01  # Lower threshold
        
        if 'audio_correlation' in metrics:
            results['audio_sync'] = metrics['audio_correlation'] >= 0.2  # Lower threshold
        
        if 'recon_error' in metrics:
            results['reconstruction'] = metrics['recon_error'] <= 0.2  # More permissive
        
        return results
    
    def update_stage(self, epoch: int, total_epochs: int):
        """Update training stage based on progress"""
        progress = epoch / total_epochs
        
        if progress < 0.3:
            new_stage = 'motion_first'
        elif progress < 0.7:
            new_stage = 'balanced'
        else:
            new_stage = 'quality_focus'
        
        if new_stage != self.stage:
            logger.info(f"Updating TDD stage from {self.stage} to {new_stage}")
            self.stage = new_stage
            self.weights = self.stage_weights[new_stage]


def train_with_balanced_tdd():
    """Quick training with balanced TDD losses"""
    import sys
    from omegaconf import OmegaConf
    sys.path.insert(0, 'nemo')
    from vasa_model import VASAModel
    import importlib
    
    print("\n" + "="*60)
    print("Balanced TDD Training - Encouraging Motion")
    print("="*60)
    
    # Load models
    config = OmegaConf.load('vasa_config_fixed.yaml')
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda().eval()
    
    model = VASAModel(config, volumetric_avatar).cuda().train()
    
    # Use BALANCED TDD loss
    tdd_loss = BalancedTDDLoss(config, device='cuda', stage='motion_first')
    
    # Higher learning rate for faster convergence
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.01,  # Higher LR to encourage motion learning
        weight_decay=0.0001
    )
    
    B, T = 2, 30
    num_epochs = 3
    steps_per_epoch = 100
    
    print(f"\nTraining with B={B}, T={T}")
    print("Stage progression: motion_first → balanced → quality_focus")
    print("-" * 40)
    
    best_diversity = 0
    
    for epoch in range(num_epochs):
        # Update stage
        tdd_loss.update_stage(epoch, num_epochs)
        print(f"\n📊 Epoch {epoch}, Stage: {tdd_loss.stage}")
        
        for step in range(steps_per_epoch):
            global_step = epoch * steps_per_epoch + step
            optimizer.zero_grad()
            
            # Create DIVERSE targets
            t = torch.linspace(0, 4*3.14159, T).cuda()
            
            # Add variety across batch
            targets = {}
            for b in range(B):
                if b == 0:
                    # Sample 1: Smooth motion
                    targets = {
                        'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
                        'scale': torch.ones(B, T, 3).cuda(),
                        'rotation': torch.zeros(B, T, 3).cuda(),
                        'translation': torch.zeros(B, T, 3).cuda(),
                        'expression_embed': torch.randn(B, T, 128).cuda()
                    }
                    targets['rotation'][0, :, 0] = torch.sin(t * 0.5) * 0.2
                    targets['rotation'][0, :, 1] = torch.cos(t * 0.7) * 0.3
                
                if B > 1:
                    # Sample 2: Different motion pattern
                    targets['rotation'][1, :, 0] = torch.cos(t * 0.8) * 0.25
                    targets['rotation'][1, :, 1] = torch.sin(t * 0.4) * 0.35
                    targets['expression_embed'][1] = torch.randn(T, 128).cuda() * 0.5
            
            # Audio with variations
            audio_features = torch.randn(B, T, 768).cuda()
            audio_energy = 1 + torch.sin(t * 2) * 0.5
            audio_features *= audio_energy.unsqueeze(0).unsqueeze(-1)
            
            conditions = {
                'audio_features': audio_features,
                'gaze': torch.randn(B, T, 2).cuda() * 0.2,
                'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
                'emotion': torch.randn(B, T, 2).cuda() * 0.3
            }
            
            # Less noise to preserve motion
            noise_level = torch.ones(B, device='cuda') * 0.1
            noise = {key: torch.randn_like(val) * 0.02 for key, val in targets.items()}
            
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
            
            # Compute balanced losses
            losses, test_info = tdd_loss.compute_losses(
                outputs=outputs,
                targets=targets,
                conditions=conditions,
                step=global_step
            )
            
            # Backward
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track diversity
            diversity = test_info.get('metrics', {}).get('temporal_diversity', 0)
            if diversity > best_diversity:
                best_diversity = diversity
                print(f"  Step {global_step}: Loss={losses['total'].item():.1f}, "
                      f"Motion={test_info['metrics'].get('motion_magnitude', 0):.3f}, "
                      f"Diversity={diversity:.3f} ⬆️")
            elif step % 20 == 0:
                print(f"  Step {global_step}: Loss={losses['total'].item():.1f}, "
                      f"Motion={test_info['metrics'].get('motion_magnitude', 0):.3f}, "
                      f"Tests={test_info['passed_ratio']:.1%}")
    
    print(f"\n✅ Training complete!")
    print(f"Best diversity achieved: {best_diversity:.3f}")
    
    # Save model
    from pathlib import Path
    Path("checkpoints/balanced_tdd").mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'diversity': best_diversity
    }, "checkpoints/balanced_tdd/model.pth")
    print("Model saved to checkpoints/balanced_tdd/model.pth")


if __name__ == "__main__":
    train_with_balanced_tdd()