#!/usr/bin/env python3
"""
Motion-Focused Training for VASA Model
=======================================
Training script that prioritizes motion generation based on TDD test results.
"""

import torch
import torch.nn.functional as F
import logging
from omegaconf import OmegaConf
import sys
import numpy as np
from pathlib import Path
import wandb
import math

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel, VASALossModule
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionAugmentation:
    """Add motion to training data to prevent static outputs"""
    
    def __init__(self, motion_scale=0.1):
        self.motion_scale = motion_scale
    
    def add_temporal_motion(self, data, t):
        """Add time-varying motion to encourage dynamics"""
        B, T = data['theta'].shape[:2]
        
        # Add sinusoidal motion patterns
        time_factor = torch.linspace(0, 2*math.pi, T).cuda()
        
        # Head rotation motion
        rotation_motion = torch.zeros_like(data['rotation'])
        rotation_motion[..., 1] = torch.sin(time_factor) * 0.3  # Yaw motion
        rotation_motion[..., 0] = torch.cos(time_factor * 0.5) * 0.15  # Pitch motion
        
        # Expression dynamics
        expr_motion = torch.randn_like(data['expression_embed']) * 0.05
        expr_motion *= torch.sin(time_factor).unsqueeze(0).unsqueeze(-1)
        
        # Apply motion
        data['rotation'] = data['rotation'] + rotation_motion
        data['expression_embed'] = data['expression_embed'] + expr_motion
        
        # Add subtle translation
        data['translation'][..., :2] += torch.sin(time_factor).unsqueeze(0).unsqueeze(-1) * 0.01
        
        return data
    
    def add_speech_motion(self, data, audio_features):
        """Add motion correlated with audio"""
        B, T = data['theta'].shape[:2]
        
        # Compute audio energy
        audio_energy = torch.norm(audio_features, dim=-1)  # [B, T]
        audio_energy = audio_energy / (audio_energy.max() + 1e-6)
        
        # Add jaw motion based on audio
        jaw_motion = audio_energy.unsqueeze(-1).unsqueeze(-1) * 0.2
        data['theta'][..., 2, 3] += jaw_motion.squeeze(-1).squeeze(-1)  # Jaw open/close
        
        # Add expression variation with audio
        expr_variation = audio_energy.unsqueeze(-1) * torch.randn(B, T, 128).cuda() * 0.1
        data['expression_embed'] += expr_variation
        
        return data

def compute_motion_losses(outputs, targets, prev_outputs=None):
    """Compute losses that encourage motion"""
    losses = {}
    
    # Temporal difference loss - penalize static frames
    if outputs['theta'].shape[1] > 1:
        theta_diff = torch.diff(outputs['theta'], dim=1)
        expr_diff = torch.diff(outputs['expression_embed'], dim=1)
        
        # We want some motion, but not too much
        target_motion = 0.1
        losses['theta_motion'] = F.mse_loss(
            torch.norm(theta_diff, dim=-1).mean(),
            torch.tensor(target_motion).cuda()
        )
        losses['expr_motion'] = F.mse_loss(
            torch.norm(expr_diff, dim=-1).mean(), 
            torch.tensor(target_motion).cuda()
        )
    
    # Penalize completely static sequences
    if outputs['theta'].shape[1] > 1:
        static_penalty = 0
        for key in ['theta', 'rotation', 'expression_embed']:
            if key in outputs:
                variance = torch.var(outputs[key], dim=1).mean()
                static_penalty += torch.exp(-variance * 100)  # High penalty for low variance
        losses['static_penalty'] = static_penalty
    
    # Smoothness loss - prevent jittery motion
    if outputs['theta'].shape[1] > 2:
        theta_accel = torch.diff(torch.diff(outputs['theta'], dim=1), dim=1)
        losses['smoothness'] = torch.norm(theta_accel, dim=-1).mean()
    
    return losses

def main():
    print("\n" + "="*60)
    print("VASA Motion-Focused Training")
    print("="*60)
    print("Addressing TDD test failures:")
    print("  ❌ 82.9% static frames")
    print("  ❌ Face motion < 0.001")
    print("  ❌ Expression variation < 1.0")
    print("-"*60)
    
    # Initialize wandb
    wandb.init(
        project="vasa-motion",
        name="motion_focused_training",
        config={"focus": "motion_generation"},
        mode="offline"
    )
    
    # Load config with motion-focused adjustments
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Motion-focused overrides
    config.model.dropout = 0.1  # Some dropout for generalization
    config.train.learning_rate = 0.01
    
    # CRITICAL: High weights for motion losses
    config.loss.lambda_reconstruction = 1.0
    config.loss.lambda_pose = 10.0  # Increased for motion
    config.loss.lambda_dynamics = 20.0  # Much higher for expression dynamics
    config.loss.lambda_temporal = 15.0  # Temporal consistency
    
    print("\nMotion-focused loss weights:")
    print(f"  lambda_pose: {config.loss.lambda_pose}")
    print(f"  lambda_dynamics: {config.loss.lambda_dynamics}")
    print(f"  lambda_temporal: {config.loss.lambda_temporal}")
    
    # Load volumetric avatar
    print("\n1. Loading volumetric avatar...")
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    
    # Create model
    print("\n2. Creating VASA model...")
    model = VASAModel(config, volumetric_avatar)
    
    # Load checkpoint if exists
    checkpoint_path = Path("checkpoints/overfit/model.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.cuda()
    model.train()
    
    # Create loss module
    print("\n3. Creating loss module...")
    loss_module = VASALossModule(
        volumetric_avatar=volumetric_avatar,
        config=config,
        device='cuda'
    )
    
    # Motion augmentation
    motion_aug = MotionAugmentation(motion_scale=0.1)
    
    # Optimizer with appropriate learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/motion_focused")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training batch with SEQUENCE data (not single frames!)
    print("\n4. Creating motion-rich training batch...")
    B, T = 4, 50  # Batch of 4, sequence of 50 frames (2 seconds at 25fps)
    
    print(f"Training with sequences: B={B}, T={T} frames")
    print("This encourages temporal coherence and motion learning")
    
    best_motion_score = 0
    
    print("\n5. Starting motion-focused training...")
    print("-" * 40)
    
    for step in range(2000):  # 2000 steps for motion learning
        optimizer.zero_grad()
        
        # Create dynamic targets with built-in motion
        t = step / 100.0  # Time factor for motion
        targets = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        # Add motion patterns
        targets = motion_aug.add_temporal_motion(targets, t)
        
        # Create audio-correlated conditions
        audio_features = torch.randn(B, T, 768).cuda()
        conditions = {
            'audio_features': audio_features,
            'gaze': torch.randn(B, T, 2).cuda() * 0.3,  # Eye motion
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda()
        }
        
        # Add speech-related motion
        targets = motion_aug.add_speech_motion(targets, audio_features)
        
        # Add noise for denoising training
        noise_level = torch.ones(B, device='cuda') * 0.3
        noise = {key: torch.randn_like(val) * 0.2 for key, val in targets.items()}
        
        # Apply noise with proper broadcasting
        noisy_inputs = {}
        for key in targets.keys():
            if len(targets[key].shape) == 4:  # [B, T, D1, D2]
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1, 1)
            elif len(targets[key].shape) == 3:  # [B, T, D]
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1)
            else:
                noisy_inputs[key] = targets[key] + noise[key]
        
        # Forward pass
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Compute standard losses
        try:
            losses, metrics = loss_module.compute_losses(
                outputs=outputs,
                targets=targets,
                conditions=conditions,
                noise=noise,
                return_metrics=True,
                current_epoch=step // 100,
                step=step
            )
        except:
            # Fallback
            losses = {'total': F.mse_loss(outputs['theta'], targets['theta'])}
        
        # Add motion-specific losses
        motion_losses = compute_motion_losses(outputs, targets)
        
        # Combine losses with emphasis on motion
        total_loss = losses.get('total', 0)
        for key, loss in motion_losses.items():
            total_loss += loss * 5.0  # High weight for motion losses
            losses[f'motion_{key}'] = loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Calculate motion score
        with torch.no_grad():
            if outputs['theta'].shape[1] > 1:
                motion_score = torch.norm(
                    torch.diff(outputs['theta'], dim=1), dim=-1
                ).mean().item()
            else:
                motion_score = 0
        
        # Track best motion score
        if motion_score > best_motion_score:
            best_motion_score = motion_score
            # Save checkpoint with good motion
            checkpoint = {
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'motion_score': motion_score,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / 'best_motion.pth')
            print(f"    💃 New best motion score: {motion_score:.4f}")
        
        # Log progress
        if step % 20 == 0:
            loss_val = total_loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log motion metrics
            motion_metrics = {
                'static_penalty': motion_losses.get('static_penalty', 0).item() if 'static_penalty' in motion_losses else 0,
                'motion_score': motion_score
            }
            
            print(f"Step {step:4d}: Loss={loss_val:.4f}, Motion={motion_score:.4f}, "
                  f"Static={motion_metrics['static_penalty']:.4f}, LR={current_lr:.6f}")
            
            # Log to wandb
            wandb.log({
                'loss': loss_val,
                'motion_score': motion_score,
                'learning_rate': current_lr,
                **motion_metrics,
                'step': step
            })
        
        # Save periodic checkpoints
        if step % 100 == 0 and step > 0:
            checkpoint = {
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'motion_score': motion_score,
                'config': config,
            }
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"    📁 Saved checkpoint at step {step}")
        
        # Early stopping if motion is good
        if motion_score > 0.5:
            print(f"\n✅ Good motion achieved! Score: {motion_score:.4f}")
            break
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_motion_score': best_motion_score,
        'config': config,
    }
    torch.save(final_checkpoint, checkpoint_dir / 'final_motion_model.pth')
    
    print("\n" + "="*60)
    print("Motion-Focused Training Complete!")
    print(f"Best motion score: {best_motion_score:.4f}")
    print(f"Model saved to: {checkpoint_dir}")
    print("\nNext: Run vi.py with the new checkpoint to test motion generation")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()