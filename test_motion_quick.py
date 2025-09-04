#!/usr/bin/env python3
"""
Quick test to verify improved motion in TDD model
"""

import torch
import sys
sys.path.insert(0, 'nemo')
from vi import VASAInference
import numpy as np
from PIL import Image
import time

print("\n" + "="*60)
print("Quick Motion Test - TDD Model")
print("="*60)

# Initialize with TDD model
inferencer = VASAInference(
    checkpoint_path="checkpoints/tdd_wandb/best_model.pth",
    config_path='vasa_config_fixed.yaml'
)

print("\n📊 Generating short test sequence...")

# Generate just a few frames to test motion
with torch.no_grad():
    # Use simple inputs for quick test
    B, T = 1, 10  # Just 10 frames
    
    # Create motion data
    motion_data = {
        'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
        'scale': torch.ones(B, T, 3).cuda(),
        'rotation': torch.zeros(B, T, 3).cuda(),
        'translation': torch.zeros(B, T, 3).cuda(),
        'expression_embed': torch.randn(B, T, 128).cuda()
    }
    
    # Add some rotation for testing
    t = torch.linspace(0, 2*3.14159, T).cuda()
    motion_data['rotation'][0, :, 1] = torch.sin(t) * 0.3  # Yaw motion
    
    # Audio features
    conditions = {
        'audio_features': torch.randn(B, T, 768).cuda(),
        'gaze': torch.zeros(B, T, 2).cuda(),
        'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
        'emotion': torch.zeros(B, T, 2).cuda()
    }
    
    # Generate sequence
    outputs = inferencer.model.forward(
        motion_data=motion_data,
        noise_level=torch.zeros(B, device='cuda'),
        conditions=conditions
    )
    
    # Analyze motion
    theta_diff = torch.diff(outputs['theta'], dim=1)
    motion_magnitude = torch.norm(theta_diff, dim=-1).mean()
    
    rotation_var = torch.var(outputs['rotation'], dim=1).mean()
    expr_diff = torch.diff(outputs['expression_embed'], dim=1)
    expr_variation = torch.norm(expr_diff, dim=-1).mean()
    
    print(f"\n✅ Motion Analysis:")
    print(f"   Motion magnitude: {motion_magnitude.item():.3f}")
    print(f"   Rotation variance: {rotation_var.item():.3f}")
    print(f"   Expression variation: {expr_variation.item():.3f}")
    
    # Compare with expected conservative values
    print(f"\n📈 Quality Assessment:")
    if motion_magnitude > 1.0:
        print(f"   ✅ Good motion (>{1.0})")
    elif motion_magnitude > 0.1:
        print(f"   ⚠️  Moderate motion (>{0.1})")
    else:
        print(f"   ❌ Too static (<{0.1})")
    
    if expr_variation > 1.0:
        print(f"   ✅ Good expression variation (>{1.0})")
    elif expr_variation > 0.5:
        print(f"   ⚠️  Moderate expression (>{0.5})")
    else:
        print(f"   ❌ Static expression (<{0.5})")
    
    # Save a comparison image of first and last frame theta values
    first_theta = outputs['theta'][0, 0].cpu().numpy()
    last_theta = outputs['theta'][0, -1].cpu().numpy()
    
    # Compute difference
    theta_change = np.linalg.norm(last_theta - first_theta)
    print(f"\n   Total theta change (first→last): {theta_change:.3f}")
    
print("\n" + "="*60)
print("Test Complete!")
print("The improved TDD model should show:")
print("- Motion magnitude > 1.0")
print("- Expression variation > 1.0")
print("- Significant theta changes between frames")
print("="*60)