#!/usr/bin/env python3
"""
Minimal Overfitting Test
=========================
Simplified script to test if VASA model can overfit on dummy data.
"""

import torch
import torch.nn.functional as F
import logging
from omegaconf import OmegaConf
import sys
import numpy as np
from pathlib import Path
import wandb

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

# Imports
from vasa_model import VASAModel, VASALossModule
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*60)
    print("VASA Minimal Overfitting Test")
    print("="*60)
    
    # Initialize wandb
    wandb.init(
        project="vasa-overfit",
        name="aggressive_overfit",
        config={"mode": "overfitting", "batch_size": 16},
        mode="offline"  # Run offline to avoid network issues
    )
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Override for fast overfitting
    config.model.dropout = 0.0
    config.train.learning_rate = 0.02
    config.loss.lambda_reconstruction = 10.0
    config.loss.lambda_pose = 5.0
    config.loss.lambda_dynamics = 5.0
    
    # Disable noise for pure overfitting test
    use_noise = False  # Set to True to test with denoising
    
    print("\nLoss weights:")
    print(f"  lambda_reconstruction: {config.loss.lambda_reconstruction}")
    print(f"  lambda_pose: {config.loss.lambda_pose}")
    print(f"  lambda_dynamics: {config.loss.lambda_dynamics}")
    
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
    
    # Create loss module
    print("\n3. Creating loss module...")
    loss_module = VASALossModule(
        volumetric_avatar=volumetric_avatar,
        config=config,
        device='cuda'
    )
    
    # Create optimizer with aggressive learning rate for 32GB VRAM
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)  # Even higher LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5  # More aggressive reduction
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/overfit")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create larger batch to use more VRAM (32GB available, using ~6GB currently)
    print("\n4. Creating training batch (maximizing VRAM usage)...")
    B, T = 32, 30  # Max batch for 32GB VRAM - will use ~20-25GB
    
    # Create targets (what we want to learn)
    targets = {
        'theta': torch.randn(B, T, 3, 4).cuda(),
        'scale': torch.ones(B, T, 3).cuda(),
        'rotation': torch.zeros(B, T, 3).cuda(),
        'translation': torch.zeros(B, T, 3).cuda(),
        'expression_embed': torch.randn(B, T, 128).cuda()
    }
    
    # Create conditions
    conditions = {
        'audio_features': torch.randn(B, T, 768).cuda(),
        'gaze': torch.randn(B, T, 2).cuda(),
        'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
        'emotion': torch.randn(B, T, 2).cuda()
    }
    
    print("\n5. Starting overfitting loop...")
    print("-" * 40)
    
    best_loss = float('inf')
    losses_history = []
    
    for step in range(10000):  # Extended training to 10000 steps
        optimizer.zero_grad()
        
        if use_noise:
            # Sample noise level (or use fixed for debugging)
            if step < 500:
                # Longer period with no noise for better initial learning
                noise_level = torch.zeros(B, device='cuda')  # Match batch size
            else:
                # Very gradually introduce noise
                noise_level = torch.ones(B, device='cuda') * min(0.3, (step - 500) / 1500)  # Match batch size
            
            # Create noise
            noise = {key: torch.randn_like(val) * 0.1 for key, val in targets.items()}
            
            # Add noise to targets
            noisy_inputs = {}
            for key in targets.keys():
                noisy_inputs[key] = targets[key] + noise[key] * noise_level
        else:
            # No noise - pure reconstruction task
            noise_level = torch.zeros(B, device='cuda')  # Match batch size
            noise = {key: torch.zeros_like(val) for key, val in targets.items()}
            noisy_inputs = targets
        
        # Forward pass - denoise
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Compute losses
        try:
            losses, metrics = loss_module.compute_losses(
                outputs=outputs,
                targets=targets,
                conditions=conditions,
                noise=noise,
                return_metrics=True,
                current_epoch=0,
                step=step
            )
            
            total_loss = losses['total']
            
        except Exception as e:
            print(f"\nError computing loss: {str(e)}")
            # Fallback to simple MSE
            total_loss = 0
            for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
                if key in outputs and key in targets:
                    total_loss += F.mse_loss(outputs[key], targets[key])
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track loss
        loss_val = total_loss.item()
        losses_history.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            # Save best checkpoint
            checkpoint = {
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'config': config,
            }
            checkpoint_path = checkpoint_dir / 'model.pth'  # Always same filename
            torch.save(checkpoint, checkpoint_path)
            print(f"    💾 Saved model (loss={loss_val:.6f}) to {checkpoint_path}")
        
        # Update scheduler
        scheduler.step(loss_val)
        
        # Log progress
        if step % 10 == 0:
            recent_avg = np.mean(losses_history[-10:]) if len(losses_history) >= 10 else loss_val
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step:4d}: Loss = {loss_val:.6f}, Best = {best_loss:.6f}, Avg(10) = {recent_avg:.6f}, LR = {current_lr:.6f}")
            
            # Log to wandb
            wandb.log({
                "loss": loss_val,
                "best_loss": best_loss,
                "learning_rate": current_lr,
                "recent_avg_loss": recent_avg,
                "step": step
            })
            
        # Save periodic checkpoints (overwrite same file)
        if step % 100 == 0 and step > 0:
            checkpoint = {
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_val,
                'config': config,
            }
            checkpoint_path = checkpoint_dir / 'model.pth'  # Always same filename
            torch.save(checkpoint, checkpoint_path)
            print(f"    📁 Saved checkpoint at step {step} to {checkpoint_path}")
            
            # Check if we're improving
            if step > 100 and len(losses_history) > 100:
                old_avg = np.mean(losses_history[-100:-50])
                new_avg = np.mean(losses_history[-50:])
                if new_avg > old_avg * 0.95:  # Not improving by at least 5%
                    print("⚠️ Warning: Loss plateau detected!")
        
        # Check for successful overfitting
        if loss_val < 0.01:
            print(f"\n✅ Overfitting successful at step {step}!")
            print(f"Final loss: {loss_val:.6f}")
            break
    
    else:
        if best_loss < 0.1:
            print(f"\n✅ Partial success - best loss: {best_loss:.6f}")
        elif best_loss < 1.0:
            print(f"\n⚠️ Some learning occurred - best loss: {best_loss:.6f}")
        else:
            print(f"\n❌ Failed to overfit - best loss: {best_loss:.6f}")
    
    # Save final checkpoint
    final_checkpoint = {
        'epoch': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses_history[-1] if losses_history else float('inf'),
        'best_loss': best_loss,
        'config': config,
        'losses_history': losses_history
    }
    final_path = checkpoint_dir / 'model.pth'  # Always same filename
    torch.save(final_checkpoint, final_path)
    print(f"\n💾 Saved final model to {final_path}")
    
    print("\n" + "="*60)
    print("Testing reconstruction without noise...")
    print("-" * 40)
    
    # Test without noise
    model.eval()
    with torch.no_grad():
        # Direct reconstruction
        outputs = model.forward(
            motion_data=targets,
            noise_level=torch.zeros(1, device='cuda'),
            conditions=conditions
        )
        
        # Compute reconstruction loss
        recon_loss = 0
        for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
            if key in outputs and key in targets:
                key_loss = F.mse_loss(outputs[key], targets[key])
                recon_loss += key_loss
                print(f"  {key}: {key_loss.item():.6f}")
        
        print(f"\nTotal reconstruction loss: {recon_loss.item():.6f}")
        
        if recon_loss.item() < 0.1:
            print("✅ Model successfully memorized the batch!")
        else:
            print("⚠️ Model has not fully memorized the batch")
    
    # Plot loss curve if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses_history)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig('overfit_loss_curve.png')
        print(f"\nLoss curve saved to overfit_loss_curve.png")
    except:
        pass
    
    print("="*60)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()