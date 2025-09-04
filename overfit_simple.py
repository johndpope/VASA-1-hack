#!/usr/bin/env python3
"""
Simple Overfitting Test
========================
Minimal script to overfit VASA on a single batch.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
import sys
import logging

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from logger import logger
import importlib

logging.basicConfig(level=logging.INFO)

def create_dummy_batch(B=1, T=5, device='cuda'):
    """Create a dummy batch for overfitting."""
    batch = {
        'theta': torch.randn(B, T, 3, 4).to(device),
        'scale': torch.ones(B, T, 3).to(device),
        'rotation': torch.zeros(B, T, 3).to(device),
        'translation': torch.zeros(B, T, 3).to(device),
        'expression_embed': torch.randn(B, T, 128).to(device),
        'audio_features': torch.randn(B, T, 768).to(device),
        'gaze': torch.randn(B, T, 2).to(device),
        'head_distance': torch.ones(B, T, 1).to(device) * 0.5,
        'emotion': torch.randn(B, T, 2).to(device),
        'frames': torch.randn(B, T, 3, 256, 256).to(device),
    }
    return batch

def main():
    print("\n" + "="*60)
    print("VASA Simple Overfitting Test")
    print("="*60)
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Override for overfitting
    config.model.dropout = 0.0
    config.train.learning_rate = 0.01  # High LR for fast overfitting
    
    # Load volumetric avatar first
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
    model = VASAModel(config.model, volumetric_avatar)
    model = model.cuda()
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create single batch to overfit on
    print("\n3. Creating dummy batch...")
    batch = create_dummy_batch()
    
    # Store original batch as target
    targets = {
        'theta': batch['theta'].clone(),
        'scale': batch['scale'].clone(),
        'rotation': batch['rotation'].clone(),
        'translation': batch['translation'].clone(),
        'expression_embed': batch['expression_embed'].clone()
    }
    
    print("\n4. Starting overfitting loop...")
    print("-" * 40)
    
    best_loss = float('inf')
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # Add noise and denoise
        noise_level = torch.ones(1, device='cuda') * 0.5
        
        # Add noise to inputs
        noisy_inputs = {}
        for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
            noise = torch.randn_like(targets[key]) * 0.1
            noisy_inputs[key] = targets[key] + noise
        
        # Forward pass - denoise
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions={
                'audio_features': batch['audio_features'],
                'gaze': batch['gaze'],
                'head_distance': batch['head_distance'],
                'emotion': batch['emotion']
            }
        )
        
        # Compute simple MSE loss
        loss = 0
        for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
            if key in outputs and key in targets:
                loss += F.mse_loss(outputs[key], targets[key])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Log progress
        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            
        if step % 10 == 0:
            print(f"Step {step:4d}: Loss = {loss_val:.6f}, Best = {best_loss:.6f}")
            
        # Check for successful overfitting
        if loss_val < 0.001:
            print(f"\n✅ Overfitting successful at step {step}!")
            print(f"Final loss: {loss_val:.6f}")
            break
            
        if step == 999:
            print(f"\n⚠️ Failed to overfit after 1000 steps")
            print(f"Final loss: {loss_val:.6f}")
    
    print("\n" + "="*60)
    print("Testing reconstruction without noise...")
    print("-" * 40)
    
    # Test without noise
    model.eval()
    with torch.no_grad():
        # No noise, just reconstruction
        outputs = model.forward(
            motion_data=targets,
            noise_level=torch.zeros(1, device='cuda'),
            conditions={
                'audio_features': batch['audio_features'],
                'gaze': batch['gaze'],
                'head_distance': batch['head_distance'],
                'emotion': batch['emotion']
            }
        )
        
        # Compute reconstruction loss
        recon_loss = 0
        for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
            if key in outputs and key in targets:
                key_loss = F.mse_loss(outputs[key], targets[key])
                recon_loss += key_loss
                print(f"  {key}: {key_loss.item():.6f}")
        
        print(f"\nTotal reconstruction loss: {recon_loss.item():.6f}")
        
        if recon_loss.item() < 0.01:
            print("✅ Model successfully memorized the batch!")
        else:
            print("⚠️ Model failed to memorize the batch perfectly")
    
    print("="*60)

if __name__ == "__main__":
    main()