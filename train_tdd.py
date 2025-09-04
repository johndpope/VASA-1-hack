#!/usr/bin/env python3
"""
TDD Training Script
===================
Simplified script to train VASA using TDD loss module
"""

import torch
from omegaconf import OmegaConf
import sys
from pathlib import Path

if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from tdd_loss_module import TDDLossModule
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*60)
    print("VASA Training with Test-Driven Development")
    print("="*60)
    print("All losses computed from test criteria")
    print("-"*60)
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
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
    model = model.cuda()
    model.train()
    
    # Create TDD loss module
    print("\n3. Creating TDD loss module...")
    tdd_loss = TDDLossModule(config, volumetric_avatar, device='cuda')
    
    # Print test criteria
    print("\n4. Test criteria for losses:")
    for name, criteria in tdd_loss.test_criteria.items():
        print(f"\n  {name}:")
        print(f"    Range: [{criteria.min_value:.3f}, {criteria.max_value:.3f}]")
        if criteria.target_value:
            print(f"    Target: {criteria.target_value:.3f}")
        print(f"    Weight: {criteria.weight:.1f}")
        if criteria.must_pass:
            print(f"    CRITICAL: Must pass!")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Training parameters
    B, T = 2, 50
    print(f"\n5. Training with B={B}, T={T}")
    
    # Checkpoint directory
    checkpoint_dir = Path("checkpoints/tdd")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_test_score = 0.0
    
    print("\n6. Starting TDD training loop...")
    print("-" * 40)
    
    for step in range(1000):
        optimizer.zero_grad()
        
        # Create motion targets
        targets = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.randn(B, T, 3).cuda() * 0.3,
            'translation': torch.randn(B, T, 3).cuda() * 0.1,
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        # Add realistic motion patterns
        t = torch.linspace(0, 4*3.14159, T).cuda()
        targets['rotation'][:, :, 0] = torch.sin(t) * 0.2  # Pitch
        targets['rotation'][:, :, 1] = torch.cos(t * 0.7) * 0.3  # Yaw
        targets['rotation'][:, :, 2] = torch.sin(t * 0.5) * 0.1  # Roll
        
        # Create conditions
        audio_features = torch.randn(B, T, 768).cuda()
        conditions = {
            'audio_features': audio_features,
            'gaze': torch.randn(B, T, 2).cuda() * 0.5,
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda()
        }
        
        # Add noise for denoising
        noise_level = torch.ones(B, device='cuda') * 0.3
        noise = {key: torch.randn_like(val) * 0.1 for key, val in targets.items()}
        
        # Apply noise
        noisy_inputs = {}
        for key in targets.keys():
            if len(targets[key].shape) == 4:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1, 1)
            elif len(targets[key].shape) == 3:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1)
            else:
                noisy_inputs[key] = targets[key] + noise[key]
        
        # Forward pass
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Compute TDD losses
        losses, test_metrics = tdd_loss.compute_losses(
            outputs=outputs,
            targets=targets,
            conditions=conditions,
            noise=noise,
            generated_frames=None,  # Skip frame generation for speed
            target_frames=None,
            return_metrics=True,
            current_epoch=step // 100,
            step=step
        )
        
        # Backward pass
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track test pass rate
        if 'test_results' in test_metrics:
            test_results = test_metrics['test_results']
            pass_rate = sum(test_results.values()) / len(test_results)
            
            if pass_rate > best_test_score:
                best_test_score = pass_rate
                # Save best model
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'test_score': best_test_score,
                    'step': step
                }
                torch.save(checkpoint, checkpoint_dir / 'best_tdd_model.pth')
                print(f"  🎯 New best test score: {best_test_score:.1%}")
        
        # Log progress
        if step % 20 == 0:
            test_pass_rate = test_metrics.get('passed_ratio', 0)
            print(f"Step {step:4d}: Loss={losses['total'].item():.4f}, "
                  f"Tests={test_pass_rate:.1%}")
            
            # Show which tests are failing
            if step % 100 == 0 and 'test_results' in test_metrics:
                print("\n  Test Results:")
                for name, passed in test_results.items():
                    status = "✓" if passed else "✗"
                    metric_value = test_metrics.get('test_metrics', {}).get(name, 0)
                    print(f"    {status} {name}: {metric_value:.3f}")
                print()
        
        # Early stopping if all tests pass
        if test_metrics.get('passed_ratio', 0) > 0.95:
            print(f"\n✅ 95% of tests passing! Training successful.")
            break
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'final_test_score': test_metrics.get('passed_ratio', 0),
        'config': config
    }
    torch.save(final_checkpoint, checkpoint_dir / 'final_tdd_model.pth')
    
    # Print final test summary
    print("\n" + "="*60)
    print("TDD Training Complete!")
    print(tdd_loss.get_test_summary())
    print(f"Best test score: {best_test_score:.1%}")
    print(f"Model saved to: {checkpoint_dir}")
    print("="*60)

if __name__ == "__main__":
    main()