#!/usr/bin/env python3
"""
Pose-Aware Training for VASA Model
===================================
Training that specifically targets head pose motion generation.
Based on understanding of head_pose_regressor.py
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

def generate_realistic_head_motion(B, T, device='cuda'):
    """Generate realistic head motion parameters based on head_pose_regressor understanding"""
    
    # Time axis for smooth motion
    t = torch.linspace(0, 4*math.pi, T).to(device)
    
    # Generate theta (3x4 transformation matrix) with realistic motion
    theta = torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).to(device)
    
    # Add realistic head rotation patterns
    for b in range(B):
        # Nodding motion (pitch)
        pitch_amplitude = 0.1 + torch.rand(1).item() * 0.1
        pitch_freq = 0.5 + torch.rand(1).item() * 0.3
        pitch = torch.sin(t * pitch_freq) * pitch_amplitude
        
        # Head shake (yaw)
        yaw_amplitude = 0.15 + torch.rand(1).item() * 0.1
        yaw_freq = 0.3 + torch.rand(1).item() * 0.2
        yaw = torch.cos(t * yaw_freq) * yaw_amplitude
        
        # Slight tilt (roll)
        roll_amplitude = 0.05 + torch.rand(1).item() * 0.05
        roll_freq = 0.2 + torch.rand(1).item() * 0.1
        roll = torch.sin(t * roll_freq * 0.5) * roll_amplitude
        
        # Apply rotations to theta matrix
        for ti in range(T):
            # Create rotation matrices
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(pitch[ti]), -torch.sin(pitch[ti])],
                [0, torch.sin(pitch[ti]), torch.cos(pitch[ti])]
            ], device=device)
            
            Ry = torch.tensor([
                [torch.cos(yaw[ti]), 0, torch.sin(yaw[ti])],
                [0, 1, 0],
                [-torch.sin(yaw[ti]), 0, torch.cos(yaw[ti])]
            ], device=device)
            
            Rz = torch.tensor([
                [torch.cos(roll[ti]), -torch.sin(roll[ti]), 0],
                [torch.sin(roll[ti]), torch.cos(roll[ti]), 0],
                [0, 0, 1]
            ], device=device)
            
            # Combined rotation
            R = Rz @ Ry @ Rx
            theta[b, ti, :3, :3] = R
    
    # Generate rotation vector (Euler angles)
    rotation = torch.zeros(B, T, 3).to(device)
    for b in range(B):
        # Natural head rotation patterns
        rotation[b, :, 0] = torch.sin(t * 0.5) * 0.2  # Pitch
        rotation[b, :, 1] = torch.cos(t * 0.3) * 0.3  # Yaw (larger range)
        rotation[b, :, 2] = torch.sin(t * 0.2) * 0.1  # Roll (smaller)
    
    # Generate scale (usually stays close to 1.0)
    scale = torch.ones(B, T, 3).to(device)
    # Add subtle breathing/pulsing effect
    scale_variation = 1.0 + torch.sin(t * 0.8).unsqueeze(0).unsqueeze(-1) * 0.02
    scale = scale * scale_variation
    
    # Generate translation (head position shifts)
    translation = torch.zeros(B, T, 3).to(device)
    for b in range(B):
        # Subtle forward/back motion (z)
        translation[b, :, 2] = torch.sin(t * 0.4) * 0.05
        # Slight side-to-side (x)
        translation[b, :, 1] = torch.cos(t * 0.3) * 0.03
        # Minimal up/down (y)
        translation[b, :, 0] = torch.sin(t * 0.6) * 0.02
    
    # Generate expression variations
    expression_embed = torch.randn(B, T, 128).to(device) * 0.3
    # Add temporal coherence to expressions
    for b in range(B):
        # Smooth transitions between expressions
        base_expr = torch.randn(128).to(device)
        for ti in range(T):
            blend = 0.7 + 0.3 * torch.sin(t[ti] * 0.5)
            expression_embed[b, ti] = base_expr * blend + torch.randn(128).to(device) * (1-blend) * 0.2
    
    return {
        'theta': theta,
        'rotation': rotation,
        'scale': scale,
        'translation': translation,
        'expression_embed': expression_embed
    }

def compute_pose_aware_losses(outputs, targets):
    """Compute losses that encourage realistic pose changes"""
    losses = {}
    
    # Standard reconstruction losses
    losses['theta_recon'] = F.mse_loss(outputs['theta'], targets['theta'])
    losses['rotation_recon'] = F.mse_loss(outputs['rotation'], targets['rotation'])
    losses['scale_recon'] = F.mse_loss(outputs['scale'], targets['scale'])
    losses['translation_recon'] = F.mse_loss(outputs['translation'], targets['translation'])
    
    # Motion consistency loss - ensure smooth transitions
    if outputs['theta'].shape[1] > 1:
        # First-order smoothness (velocity)
        theta_vel = torch.diff(outputs['theta'], dim=1)
        target_vel = torch.diff(targets['theta'], dim=1)
        losses['velocity_consistency'] = F.mse_loss(theta_vel, target_vel)
        
        # Second-order smoothness (acceleration) 
        if outputs['theta'].shape[1] > 2:
            theta_acc = torch.diff(theta_vel, dim=1)
            target_acc = torch.diff(target_vel, dim=1)
            losses['acceleration_consistency'] = F.mse_loss(theta_acc, target_acc)
    
    # Ensure rotation stays in valid range
    rotation_magnitude = torch.norm(outputs['rotation'], dim=-1)
    losses['rotation_regularization'] = torch.mean(F.relu(rotation_magnitude - math.pi))
    
    # Scale should stay close to 1.0
    scale_deviation = torch.abs(outputs['scale'] - 1.0)
    losses['scale_regularization'] = torch.mean(scale_deviation)
    
    # Translation should be bounded
    translation_magnitude = torch.norm(outputs['translation'], dim=-1)
    losses['translation_regularization'] = torch.mean(F.relu(translation_magnitude - 0.5))
    
    return losses

def main():
    print("\n" + "="*60)
    print("VASA Pose-Aware Training")
    print("="*60)
    print("Training with realistic head pose motion patterns")
    print("Based on head_pose_regressor.py analysis")
    print("-"*60)
    
    # Initialize wandb
    wandb.init(
        project="vasa-pose",
        name="pose_aware_training",
        config={"focus": "head_pose_motion"},
        mode="offline"
    )
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Adjust for pose-focused training
    config.model.dropout = 0.1
    config.train.learning_rate = 0.005
    config.loss.lambda_pose = 20.0  # High weight for pose
    config.loss.lambda_dynamics = 10.0
    config.loss.lambda_temporal = 15.0
    
    print("\nPose-focused loss weights:")
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
    
    # Load checkpoint if available
    checkpoint_path = Path("checkpoints/overfit/model.pth")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded pre-trained weights")
    
    model = model.cuda()
    model.train()
    
    # Create loss module
    print("\n3. Creating loss module...")
    loss_module = VASALossModule(
        volumetric_avatar=volumetric_avatar,
        config=config,
        device='cuda'
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5
    )
    
    # Checkpoint directory
    checkpoint_dir = Path("checkpoints/pose_aware")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    B, T = 2, 50  # Smaller batch, full sequence
    print(f"\n4. Training with B={B}, T={T} (full sequences)")
    
    best_pose_score = float('inf')
    
    print("\n5. Starting pose-aware training...")
    print("-" * 40)
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # Generate realistic head motion targets
        targets = generate_realistic_head_motion(B, T)
        
        # Create conditions that correlate with motion
        audio_features = torch.randn(B, T, 768).cuda()
        # Make audio correlate with head motion
        audio_energy = torch.norm(audio_features, dim=-1, keepdim=True)
        audio_features = audio_features * (1 + targets['rotation'][:, :, 1:2].abs())  # Correlate with yaw
        
        conditions = {
            'audio_features': audio_features,
            'gaze': targets['rotation'][:, :, :2] * 0.3,  # Gaze follows head rotation
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda() * 0.5
        }
        
        # Add noise for denoising
        noise_level = torch.ones(B, device='cuda') * 0.2
        noise = {}
        for key, val in targets.items():
            noise[key] = torch.randn_like(val) * 0.1
        
        # Apply noise properly
        noisy_inputs = {}
        for key in targets.keys():
            if len(targets[key].shape) == 4:  # theta
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1, 1)
            elif len(targets[key].shape) == 3:  # others
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1)
            else:
                noisy_inputs[key] = targets[key] + noise[key]
        
        # Forward pass
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Compute losses
        pose_losses = compute_pose_aware_losses(outputs, targets)
        
        # Try standard losses too
        try:
            standard_losses, metrics = loss_module.compute_losses(
                outputs=outputs,
                targets=targets,
                conditions=conditions,
                noise=noise,
                return_metrics=True,
                current_epoch=step // 100,
                step=step
            )
            total_loss = standard_losses.get('total', 0)
        except:
            total_loss = 0
            for loss in pose_losses.values():
                total_loss += loss
        
        # Add pose losses with high weight
        for key, loss in pose_losses.items():
            total_loss += loss * (10.0 if 'recon' in key else 2.0)
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate pose quality metrics
        with torch.no_grad():
            # Measure rotation variation
            rotation_var = torch.var(outputs['rotation'], dim=1).mean().item()
            # Measure translation motion
            trans_motion = torch.norm(torch.diff(outputs['translation'], dim=1), dim=-1).mean().item()
            # Measure theta changes
            theta_motion = torch.norm(
                torch.diff(outputs['theta'].view(B, T, -1), dim=1), dim=-1
            ).mean().item()
            
            pose_score = pose_losses['theta_recon'].item()
        
        # Track best model
        if pose_score < best_pose_score:
            best_pose_score = pose_score
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pose_score': pose_score,
                'rotation_variance': rotation_var,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / 'best_pose.pth')
            print(f"    🎯 New best pose score: {pose_score:.4f}")
        
        # Log progress
        if step % 10 == 0:
            print(f"Step {step:4d}: Loss={total_loss.item():.4f}, "
                  f"Pose={pose_score:.4f}, RotVar={rotation_var:.4f}, "
                  f"ThetaMotion={theta_motion:.4f}")
            
            wandb.log({
                'loss': total_loss.item(),
                'pose_score': pose_score,
                'rotation_variance': rotation_var,
                'translation_motion': trans_motion,
                'theta_motion': theta_motion,
                'step': step
            })
        
        # Save periodic checkpoint
        if step % 100 == 0 and step > 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_{step}.pth')
            print(f"    📁 Saved checkpoint at step {step}")
        
        # Early stopping if good pose motion achieved
        if rotation_var > 0.01 and theta_motion > 0.1:
            print(f"\n✅ Good pose motion achieved!")
            print(f"  Rotation variance: {rotation_var:.4f}")
            print(f"  Theta motion: {theta_motion:.4f}")
            break
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_pose_score': best_pose_score,
        'config': config,
    }
    torch.save(final_checkpoint, checkpoint_dir / 'final_pose_model.pth')
    
    print("\n" + "="*60)
    print("Pose-Aware Training Complete!")
    print(f"Best pose score: {best_pose_score:.4f}")
    print(f"Model saved to: {checkpoint_dir}")
    print("\nNext: Test with vi.py using checkpoints/pose_aware/best_pose.pth")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()