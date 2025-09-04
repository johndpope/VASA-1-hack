#!/usr/bin/env python3
"""
Test Loss Computation Directly
==============================
Find out why the loss computation is failing.
"""

import torch
import sys
import traceback
from omegaconf import OmegaConf
import importlib

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel, VASALossModule
from logger import logger
import logging

# Set debug logging
logging.basicConfig(level=logging.DEBUG)


def test_loss_computation():
    """Test the loss computation directly."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    try:
        # Load config
        config = OmegaConf.load('vasa_config_fixed.yaml')
        
        # Load volumetric model
        print("\n1. Loading volumetric avatar...")
        model_path = config.paths.volumetric_model
        emo_config = OmegaConf.load(config.paths.volumetric_config)
        volumetric_avatar = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)
        
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        
        # Create loss module
        print("\n2. Creating loss module...")
        loss_module = VASALossModule(
            volumetric_avatar=volumetric_avatar,
            config=config,
            device='cuda'
        )
        
        # Create dummy data
        print("\n3. Creating test data...")
        B, T = 1, 5
        
        outputs = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda(),
            'noise': {}
        }
        
        targets = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda() * 1.1,  # Slightly different
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        conditions = {
            'audio_features': torch.randn(B, T, 768).cuda(),
            'gaze': torch.randn(B, T, 2).cuda(),
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda()
        }
        
        # Add noise
        outputs['noise'] = {
            k: torch.randn_like(v) * 0.01 for k, v in targets.items()
        }
        
        # Test loss computation
        print("\n4. Computing losses...")
        losses, metrics = loss_module.compute_losses(
            outputs=outputs,
            targets=targets,
            conditions=conditions,
            noise=outputs['noise'],
            return_metrics=True,
            current_epoch=0,
            step=0
        )
        
        print("\n5. Loss results:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"   {k}: {v.item():.6f} (requires_grad: {v.requires_grad})")
        
        print("\n6. Checking if losses are reasonable...")
        if losses['total'].item() == 1.0:
            print("   ❌ Total loss is exactly 1.0 - likely using default error value!")
        elif losses['total'].item() > 0 and losses['total'].item() < 100:
            print(f"   ✅ Total loss is reasonable: {losses['total'].item():.6f}")
        else:
            print(f"   ⚠️ Total loss seems unusual: {losses['total'].item():.6f}")
        
        # Test backward pass
        print("\n7. Testing backward pass...")
        try:
            losses['total'].backward()
            print("   ✅ Backward pass successful!")
            
            # Check if any gradients were computed
            has_grads = False
            for name, param in loss_module.__dict__.items():
                if isinstance(param, torch.nn.Module):
                    for p in param.parameters():
                        if p.grad is not None:
                            has_grads = True
                            break
            
            if has_grads:
                print("   ✅ Gradients computed!")
            else:
                print("   ⚠️ No gradients found in loss module")
                
        except Exception as e:
            print(f"   ❌ Backward pass failed: {str(e)}")
        
        return losses['total'].item() != 1.0
        
    except Exception as e:
        print(f"\n❌ Error in loss computation: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        return False


def test_reconstruction_loss_directly():
    """Test just the reconstruction loss computation."""
    print("\n" + "="*60)
    print("Testing Reconstruction Loss Directly")
    print("="*60)
    
    try:
        # Simple MSE test
        print("\n1. Testing simple MSE loss...")
        pred = torch.randn(2, 5, 3, 4).cuda()
        target = torch.randn(2, 5, 3, 4).cuda()
        
        loss = torch.nn.functional.mse_loss(pred, target)
        print(f"   MSE loss: {loss.item():.6f}")
        
        if loss.item() > 0:
            print("   ✅ MSE loss computed successfully!")
            return True
        else:
            print("   ❌ MSE loss is zero or negative!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def main():
    """Run all loss tests."""
    results = {
        'Simple MSE': test_reconstruction_loss_directly(),
        'Full Loss Module': test_loss_computation()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✅ Loss computation is working!")
    else:
        print("\n❌ Loss computation has issues that need fixing.")


if __name__ == "__main__":
    main()