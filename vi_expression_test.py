#!/usr/bin/env python3
"""
Expression Preservation Test
============================
Tests if we can preserve exact expressions through the pipeline.
"""

import torch
import sys
sys.path.insert(0, 'nemo')
from vasa_model import VASAModel
from omegaconf import OmegaConf
import importlib
import numpy as np

print("\n" + "="*60)
print("EXPRESSION PRESERVATION TEST")
print("="*60)

# Load models
config = OmegaConf.load('vasa_config_fixed.yaml')
volumetric_config = OmegaConf.load(config.paths.volumetric_config)
volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(volumetric_config, training=False)
model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
volumetric_avatar.load_state_dict(model_dict, strict=False)
volumetric_avatar = volumetric_avatar.cuda().eval()

model = VASAModel(config, volumetric_avatar).cuda().eval()

# Load checkpoint
checkpoint = torch.load('checkpoints/tdd_wandb/best_model.pth', map_location='cuda', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print("\nTest 1: Check if model modifies expressions at noise_level=0")
print("-" * 40)

B, T = 1, 10

# Create a distinctive expression pattern
expression_pattern = torch.zeros(B, T, 128).cuda()
for t in range(T):
    # Each frame has a unique pattern
    expression_pattern[0, t, t*10:(t+1)*10] = 1.0  # Different features active per frame

motion_data = {
    'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
    'scale': torch.ones(B, T, 3).cuda(),
    'rotation': torch.zeros(B, T, 3).cuda(),
    'translation': torch.zeros(B, T, 3).cuda(),
    'expression_embed': expression_pattern
}

conditions = {
    'audio_features': torch.zeros(B, T, 768).cuda(),  # No audio influence
    'gaze': torch.zeros(B, T, 2).cuda(),
    'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
    'emotion': torch.zeros(B, T, 2).cuda()
}

# Test with zero noise (should preserve perfectly)
with torch.no_grad():
    outputs_zero_noise = model.forward(
        motion_data=motion_data,
        noise_level=torch.zeros(B, device='cuda'),
        conditions=conditions
    )

# Check preservation
expr_diff = torch.norm(outputs_zero_noise['expression_embed'] - expression_pattern).item()
print(f"Expression difference at noise=0: {expr_diff:.6f}")

if expr_diff < 0.01:
    print("✅ Expressions preserved at zero noise!")
else:
    print("❌ Model modifies expressions even with no noise!")

print("\nTest 2: Check model architecture")
print("-" * 40)

# The issue is likely that the model ALWAYS generates new expressions
# Let's check if we can bypass the generation

# Direct pass through motion transformer
motion_flat = torch.cat([
    motion_data['theta'].reshape(B, T, -1),
    motion_data['rotation'],
    motion_data['translation'], 
    motion_data['scale'],
    motion_data['expression_embed']
], dim=-1)

print(f"Motion input shape: {motion_flat.shape}")
print(f"Expression portion: columns {12+3+3+3} to {12+3+3+3+128}")

# The model is likely PREDICTING expressions from audio/conditions
# rather than preserving input expressions

print("\nTest 3: Expression generation vs preservation")
print("-" * 40)

# Test with different audio to see if expressions change
audio1 = torch.randn(B, T, 768).cuda() * 0.5
audio2 = torch.randn(B, T, 768).cuda() * 2.0

conditions1 = conditions.copy()
conditions1['audio_features'] = audio1

conditions2 = conditions.copy() 
conditions2['audio_features'] = audio2

with torch.no_grad():
    outputs1 = model.forward(
        motion_data=motion_data,
        noise_level=torch.zeros(B, device='cuda'),
        conditions=conditions1
    )
    
    outputs2 = model.forward(
        motion_data=motion_data,
        noise_level=torch.zeros(B, device='cuda'),
        conditions=conditions2
    )

audio_influence = torch.norm(outputs1['expression_embed'] - outputs2['expression_embed']).item()
print(f"Expression change from different audio: {audio_influence:.6f}")

if audio_influence > 1.0:
    print("❌ Model generates expressions from audio, ignoring input!")
else:
    print("✅ Audio has minimal influence on expressions")

print("\n" + "="*60)
print("DIAGNOSIS:")
if expr_diff > 0.1:
    print("The model is GENERATING expressions, not preserving them.")
    print("This is why expressions drift from the original.")
    print("\nSOLUTION:")
    print("1. Modify the model to have an 'expression preservation' mode")
    print("2. Or extract expressions frame-by-frame from original")
    print("3. Or train a separate expression encoder/decoder")
else:
    print("Model can preserve expressions - just need to use them properly!")
print("="*60)