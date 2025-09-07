#!/usr/bin/env python3
"""Comprehensive test to verify motion variance and prevent mode collapse"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'nemo')
from vasa_trainer import VASATrainer
import yaml
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_motion_differences(motion_dict, save_path="motion_variance.png"):
    """Visualize frame-to-frame differences"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for key, tensor in motion_dict.items():
        if isinstance(tensor, torch.Tensor) and plot_idx < 6:
            if len(tensor.shape) >= 2 and tensor.shape[1] > 1:
                # Compute frame differences
                diff = (tensor[:, 1:] - tensor[:, :-1]).abs()
                
                # Plot variance over time
                ax = axes[plot_idx]
                frame_vars = diff.var(dim=-1).squeeze().cpu().numpy()
                ax.plot(frame_vars)
                ax.set_title(f"{key} Frame-to-Frame Variance")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Variance")
                ax.grid(True, alpha=0.3)
                
                # Add warning line for low variance
                ax.axhline(y=1e-4, color='r', linestyle='--', alpha=0.5, label='Low variance threshold')
                
                plot_idx += 1
    
    # Remove unused subplots
    for i in range(plot_idx, 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved motion variance plot to {save_path}")
    plt.close()

def analyze_checkpoint_gradients(checkpoint_path):
    """Analyze gradients and optimizer state for signs of collapse"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\n=== Gradient Flow Analysis ===")
    if 'optimizer_state_dict' in checkpoint:
        optimizer = checkpoint['optimizer_state_dict']
        if 'state' in optimizer:
            # Check Adam momentum terms
            param_momentums = []
            for param_id, state in optimizer['state'].items():
                if 'exp_avg' in state:  # First moment (momentum)
                    momentum_norm = state['exp_avg'].norm().item()
                    param_momentums.append(momentum_norm)
            
            if param_momentums:
                avg_momentum = np.mean(param_momentums)
                std_momentum = np.std(param_momentums)
                print(f"Average momentum norm: {avg_momentum:.6f}")
                print(f"Std momentum norm: {std_momentum:.6f}")
                
                if avg_momentum < 1e-5:
                    print("⚠️  WARNING: Very low momentum - gradients may be vanishing")

def test_with_varying_conditions():
    """Test inference with varying conditions to force motion"""
    
    print("Loading config...")
    config = OmegaConf.load('overfit_config.yaml')
    
    # Check critical config values
    print(f"\n=== Config Check ===")
    print(f"eta: {config.inference.eta}")
    print(f"lambda_dynamics: {config.loss.lambda_dynamics}")
    print(f"CFG scales: {config.train.cfg_scales}")
    
    print("\nLoading checkpoint...")
    checkpoint_path = "checkpoints_overfit/best_checkpoint.pt"
    
    # Analyze checkpoint first
    analyze_checkpoint_gradients(checkpoint_path)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        
        # Create test batch with strong temporal variation
        B, T = 1, 30  # More frames for better analysis
        device = 'cuda'
        
        # Create sinusoidal variations to force motion
        t_vals = torch.linspace(0, 4*np.pi, T).unsqueeze(0)  # 2 full cycles
        
        test_batch = {
            'theta': torch.randn(B, T, 3, 4).to(device),
            'rotation': torch.stack([
                torch.sin(t_vals) * 0.2,  # Sinusoidal head rotation
                torch.cos(t_vals) * 0.1,
                torch.sin(t_vals * 2) * 0.05
            ], dim=-1).to(device),
            'translation': torch.stack([
                torch.zeros_like(t_vals),
                torch.sin(t_vals * 0.5) * 0.02,  # Slow vertical motion
                torch.zeros_like(t_vals)
            ], dim=-1).to(device),
            'scale': torch.ones(B, T, 3).to(device),
            'expression': torch.randn(B, T, 256).to(device),
            'expression_embed': torch.randn(B, T, 128).to(device),
            'audio_features': torch.randn(B, T, 128).to(device) * (1 + torch.sin(t_vals).unsqueeze(-1).to(device)),  # Modulated audio
            'gaze': torch.stack([
                torch.sin(t_vals * 3) * 0.1,  # Fast eye movement
                torch.cos(t_vals * 3) * 0.1
            ], dim=-1).to(device),
            'emotion': torch.stack([
                torch.sin(t_vals * 0.5),  # Slow emotion change
                torch.cos(t_vals * 0.5)
            ], dim=-1).to(device),
        }
        
        # Load model
        from vasa_model import VASAModel
        import importlib
        
        volumetric_config = OmegaConf.load(config.paths.volumetric_config)
        VolumetricAvatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model
        volumetric_avatar = VolumetricAvatar(volumetric_config, training=False)
        model_dict = torch.load(config.paths.volumetric_model, map_location='cuda', weights_only=False)
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda().eval()
        
        model = VASAModel(config=config, volumetric_avatar=volumetric_avatar, device='cuda')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded model weights")
        
        model.eval()
        
        # Test with different eta values
        print("\n=== Testing Different Eta Values ===")
        eta_values = [0.0, 0.1, 0.3, 0.5, 1.0]
        
        results = {}
        for eta in eta_values:
            print(f"\nTesting eta={eta}")
            
            # Override eta in config
            original_eta = config.inference.eta
            config.inference.eta = eta
            model.config.inference.eta = eta
            
            with torch.no_grad():
                # Use generate_sequence_inference for testing
                initial_pose = {
                    'theta': test_batch['theta'][:, 0:1],
                    'rotation': test_batch['rotation'][:, 0:1],
                    'translation': test_batch['translation'][:, 0:1],
                    'scale': test_batch['scale'][:, 0:1],
                }
                
                output = model.generate_sequence_inference(
                    initial_pose=initial_pose,
                    initial_dynamics=test_batch['expression_embed'][:, 0],  # Use expression_embed (128 dims)
                    conditions={
                        'audio_features': test_batch['audio_features'],
                        'gaze': test_batch['gaze'],
                        'emotion': test_batch['emotion'],
                    }
                )
            
            # Analyze variance
            for key in ['theta', 'expression', 'rotation', 'translation']:
                if key in output:
                    tensor = output[key]
                    temporal_var = tensor.var(dim=1).mean().item()
                    frame_diff = (tensor[:, 1:] - tensor[:, :-1]).abs().mean().item()
                    
                    print(f"  {key:15s}: temporal_var={temporal_var:.6f}, frame_diff={frame_diff:.6f}")
                    
                    if temporal_var < 1e-4:
                        print(f"    ⚠️  LOW VARIANCE DETECTED!")
            
            results[eta] = output
            
            # Restore original eta
            config.inference.eta = original_eta
        
        # Visualize best result (eta=0.1)
        if 0.1 in results:
            print("\n=== Visualizing Motion with eta=0.1 ===")
            visualize_motion_differences(results[0.1])
        
        # Compare variances across eta values
        print("\n=== Variance Summary Across Eta Values ===")
        print("Key         | eta=0.0  | eta=0.1  | eta=0.3  | eta=0.5  | eta=1.0")
        print("-" * 70)
        
        for key in ['theta', 'expression', 'rotation', 'translation']:
            variances = []
            for eta in eta_values:
                if eta in results and key in results[eta]:
                    var = results[eta][key].var(dim=1).mean().item()
                    variances.append(f"{var:.6f}")
                else:
                    variances.append("N/A     ")
            
            print(f"{key:12s}| {' | '.join(variances)}")
        
        print("\n=== Recommendations ===")
        print("1. If all eta values show low variance → Model has collapsed to static output")
        print("2. If eta>0 shows more variance → Stochasticity is helping")
        print("3. If variance increases with eta → DDIM sampling is working correctly")
        print("4. Optimal eta is typically 0.1-0.3 for balance between quality and diversity")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_varying_conditions()