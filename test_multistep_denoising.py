#!/usr/bin/env python3
"""
Multi-step Denoising Test
=========================
Tests if using multiple denoising steps improves quality.
"""

import torch
import sys
sys.path.insert(0, 'nemo')
from vasa_model import VASAModel
from omegaconf import OmegaConf
import importlib
import numpy as np

print("\n" + "="*60)
print("MULTI-STEP DENOISING TEST")
print("="*60)

# Load model
config = OmegaConf.load('vasa_config_fixed.yaml')
volumetric_config = OmegaConf.load(config.paths.volumetric_config)
volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(volumetric_config, training=False)
model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
volumetric_avatar.load_state_dict(model_dict, strict=False)
volumetric_avatar = volumetric_avatar.cuda().eval()

model = VASAModel(config, volumetric_avatar).cuda().eval()
checkpoint = torch.load('checkpoints/tdd_wandb/best_model.pth', map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Test parameters
B, T = 1, 10

# Target motion (what we want to generate)
target_motion = {
    'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
    'scale': torch.ones(B, T, 3).cuda(),
    'rotation': torch.zeros(B, T, 3).cuda(),
    'translation': torch.zeros(B, T, 3).cuda(),
    'expression_embed': torch.randn(B, T, 128).cuda() * 0.5
}

# Add some motion
t = torch.linspace(0, 2*3.14159, T).cuda()
target_motion['rotation'][0, :, 1] = torch.sin(t) * 0.2

conditions = {
    'audio_features': torch.randn(B, T, 768).cuda(),
    'gaze': torch.zeros(B, T, 2).cuda(),
    'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
    'emotion': torch.zeros(B, T, 2).cuda()
}

print("\nTest 1: Single-step denoising (current approach)")
print("-" * 50)

with torch.no_grad():
    # Single step at t=0
    outputs_single = model.forward(
        motion_data=target_motion,
        noise_level=torch.zeros(B, device='cuda'),
        conditions=conditions
    )
    
    motion_mag_single = torch.norm(torch.diff(outputs_single['theta'], dim=1)).item()
    expr_var_single = torch.var(outputs_single['expression_embed']).item()
    
    print(f"Motion magnitude: {motion_mag_single:.3f}")
    print(f"Expression variance: {expr_var_single:.3f}")

print("\nTest 2: Multi-step DDIM denoising")
print("-" * 50)

# DDIM sampling with multiple steps
num_inference_steps = 20  # Use 20 steps instead of 1000 for speed
ddim_timesteps = torch.linspace(999, 0, num_inference_steps).long().cuda()

# Start with pure noise
current_motion = {
    key: torch.randn_like(val) for key, val in target_motion.items()
}

print(f"Using {num_inference_steps} denoising steps...")

# DDIM sampling loop
for i, t in enumerate(ddim_timesteps):
    with torch.no_grad():
        # Predict noise at timestep t
        noise_pred = model.forward(
            motion_data=current_motion,
            noise_level=t.unsqueeze(0).expand(B),
            conditions=conditions
        )
        
        # DDIM update step (simplified)
        alpha = 1.0 - (t.float() / 1000.0)  # Simple linear schedule
        
        # Denoise
        for key in current_motion.keys():
            if key in noise_pred:
                # Remove predicted noise
                current_motion[key] = (current_motion[key] - (1 - alpha) * noise_pred[key]) / alpha.clamp(min=0.01)
        
        if i % 5 == 0:
            motion_mag = torch.norm(torch.diff(current_motion['theta'], dim=1)).item()
            print(f"  Step {i:2d} (t={t:3d}): motion_mag={motion_mag:.3f}")

outputs_multi = current_motion

motion_mag_multi = torch.norm(torch.diff(outputs_multi['theta'], dim=1)).item()
expr_var_multi = torch.var(outputs_multi['expression_embed']).item()

print(f"\nFinal motion magnitude: {motion_mag_multi:.3f}")
print(f"Final expression variance: {expr_var_multi:.3f}")

print("\nTest 3: Different number of steps")
print("-" * 50)

step_counts = [1, 5, 10, 50]
results = []

for num_steps in step_counts:
    # Start with noise
    current = {key: torch.randn_like(val) for key, val in target_motion.items()}
    
    # Create timestep schedule
    if num_steps == 1:
        timesteps = torch.tensor([0]).cuda()
    else:
        timesteps = torch.linspace(999, 0, num_steps).long().cuda()
    
    # Denoise
    for t in timesteps:
        with torch.no_grad():
            output = model.forward(
                motion_data=current,
                noise_level=t.unsqueeze(0).expand(B),
                conditions=conditions
            )
            
            # Simple update (not proper DDIM, just for testing)
            alpha = 0.1  # Update rate
            for key in current.keys():
                if key in output:
                    current[key] = (1 - alpha) * current[key] + alpha * output[key]
    
    motion_mag = torch.norm(torch.diff(current['theta'], dim=1)).item()
    results.append(motion_mag)
    print(f"  {num_steps:2d} steps: motion_mag={motion_mag:.3f}")

print("\n" + "="*60)
print("RESULTS:")
print("-" * 60)

print("\nSingle-step (current):")
print(f"  Motion: {motion_mag_single:.3f}")
print(f"  Expression variance: {expr_var_single:.3f}")

print("\nMulti-step DDIM:")
print(f"  Motion: {motion_mag_multi:.3f}")
print(f"  Expression variance: {expr_var_multi:.3f}")

print("\nRECOMMENDATION:")
if motion_mag_multi > motion_mag_single * 1.2:
    print("✅ Multi-step denoising improves motion quality!")
    print("   Consider implementing proper DDIM sampling")
else:
    print("⚠️  Multi-step doesn't significantly help")
    print("   The issue may be in the model training")

print("\nTo improve output quality:")
print("1. Implement proper DDIM scheduler")
print("2. Use classifier-free guidance")
print("3. Fine-tune the noise schedule")
print("="*60)