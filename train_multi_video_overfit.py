#!/usr/bin/env python3
"""
Multi-Video Overfitting Test
=============================
Train on multiple videos from junk folder to maximize VRAM usage.
"""

import torch
import torch.nn.functional as F
import logging
from omegaconf import OmegaConf
import sys
import numpy as np
from pathlib import Path
import wandb
import random
from torch.utils.data import Dataset, DataLoader

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

# Imports
from vasa_model import VASAModel, VASALossModule
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiVideoDataset(Dataset):
    """Dataset for multiple dummy videos"""
    def __init__(self, num_videos=10, T=20, device='cuda'):
        self.num_videos = num_videos
        self.T = T
        self.device = device
        
        # Pre-generate random data for each video
        self.videos = []
        for i in range(num_videos):
            video_data = {
                'theta': torch.randn(T, 3, 4).to(device),
                'scale': torch.ones(T, 3).to(device),
                'rotation': torch.zeros(T, 3).to(device),
                'translation': torch.zeros(T, 3).to(device),
                'expression_embed': torch.randn(T, 128).to(device),
                'audio_features': torch.randn(T, 768).to(device),
                'gaze': torch.randn(T, 2).to(device),
                'head_distance': torch.ones(T, 1).to(device) * 0.5,
                'emotion': torch.randn(T, 2).to(device)
            }
            self.videos.append(video_data)
    
    def __len__(self):
        return self.num_videos
    
    def __getitem__(self, idx):
        return self.videos[idx]

def main():
    print("\n" + "="*60)
    print("VASA Multi-Video Overfitting Test")
    print("="*60)
    
    # Initialize wandb
    wandb.init(
        project="vasa-overfit",
        name="multi_video_aggressive",
        config={"mode": "multi_video_overfitting", "batch_size": 4, "num_videos": 10},
        mode="offline"
    )
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Override for aggressive overfitting
    config.model.dropout = 0.0
    config.train.learning_rate = 0.1
    config.loss.lambda_reconstruction = 10.0
    config.loss.lambda_pose = 5.0
    config.loss.lambda_dynamics = 5.0
    
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
    
    # Create optimizer with very aggressive learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/multi_overfit")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset with multiple videos
    print("\n4. Creating multi-video dataset...")
    dataset = MultiVideoDataset(num_videos=10, T=20)
    
    # Create dataloader with batch size to maximize VRAM
    batch_size = 4  # Process 4 videos at once
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nTraining on {len(dataset)} videos with batch size {batch_size}")
    print("This will use approximately 20-25GB VRAM")
    print("-" * 40)
    
    best_loss = float('inf')
    losses_history = []
    global_step = 0
    
    for epoch in range(100):  # 100 epochs
        epoch_losses = []
        
        for batch_idx, batch_videos in enumerate(dataloader):
            # Stack batch data
            B = len(batch_videos['theta'])
            T = batch_videos['theta'][0].shape[0]
            
            targets = {
                'theta': torch.stack([v for v in batch_videos['theta']]),
                'scale': torch.stack([v for v in batch_videos['scale']]),
                'rotation': torch.stack([v for v in batch_videos['rotation']]),
                'translation': torch.stack([v for v in batch_videos['translation']]),
                'expression_embed': torch.stack([v for v in batch_videos['expression_embed']])
            }
            
            conditions = {
                'audio_features': torch.stack([v for v in batch_videos['audio_features']]),
                'gaze': torch.stack([v for v in batch_videos['gaze']]),
                'head_distance': torch.stack([v for v in batch_videos['head_distance']]),
                'emotion': torch.stack([v for v in batch_videos['emotion']])
            }
            
            optimizer.zero_grad()
            
            # No noise for pure overfitting
            noise_level = torch.zeros(B, device='cuda')
            noise = {key: torch.zeros_like(val) for key, val in targets.items()}
            
            # Forward pass
            outputs = model.forward(
                motion_data=targets,
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
                    current_epoch=epoch,
                    step=global_step
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
            epoch_losses.append(loss_val)
            losses_history.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
                # Save best checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                    'config': config,
                }
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"    💾 Saved best model (loss={loss_val:.6f}) to {checkpoint_path}")
            
            # Log progress
            if global_step % 5 == 0:
                recent_avg = np.mean(losses_history[-10:]) if len(losses_history) >= 10 else loss_val
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}, Step {global_step:4d}: Loss = {loss_val:.6f}, Best = {best_loss:.6f}, LR = {current_lr:.6f}")
                
                # Log to wandb
                wandb.log({
                    "loss": loss_val,
                    "best_loss": best_loss,
                    "learning_rate": current_lr,
                    "recent_avg_loss": recent_avg,
                    "epoch": epoch,
                    "step": global_step
                })
            
            global_step += 1
            
            # Check for successful overfitting
            if loss_val < 0.01:
                print(f"\n✅ Overfitting successful at epoch {epoch}, step {global_step}!")
                print(f"Final loss: {loss_val:.6f}")
                break
        
        # Epoch summary
        epoch_avg = np.mean(epoch_losses)
        print(f"\nEpoch {epoch} summary: Avg Loss = {epoch_avg:.6f}")
        
        # Update scheduler
        scheduler.step(epoch_avg)
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_avg,
                'config': config,
            }
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"    📁 Saved checkpoint at epoch {epoch} to {checkpoint_path}")
        
        if loss_val < 0.01:
            break
    
    # Save final checkpoint
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses_history[-1] if losses_history else float('inf'),
        'best_loss': best_loss,
        'config': config,
        'losses_history': losses_history
    }
    final_path = checkpoint_dir / 'final_model.pth'
    torch.save(final_checkpoint, final_path)
    print(f"\n💾 Saved final model to {final_path}")
    
    print("\n" + "="*60)
    print("Testing reconstruction without noise...")
    print("-" * 40)
    
    # Test on first video
    model.eval()
    with torch.no_grad():
        test_video = dataset[0]
        test_targets = {k: v.unsqueeze(0) for k in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']
                       for k, v in test_video.items() if k in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']}
        test_conditions = {k: v.unsqueeze(0) for k in ['audio_features', 'gaze', 'head_distance', 'emotion']
                          for k, v in test_video.items() if k in ['audio_features', 'gaze', 'head_distance', 'emotion']}
        
        outputs = model.forward(
            motion_data=test_targets,
            noise_level=torch.zeros(1, device='cuda'),
            conditions=test_conditions
        )
        
        # Compute reconstruction loss
        recon_loss = 0
        for key in ['theta', 'scale', 'rotation', 'translation', 'expression_embed']:
            if key in outputs and key in test_targets:
                key_loss = F.mse_loss(outputs[key], test_targets[key])
                recon_loss += key_loss
                print(f"  {key}: {key_loss.item():.6f}")
        
        print(f"\nTotal reconstruction loss: {recon_loss.item():.6f}")
        
        if recon_loss.item() < 0.1:
            print("✅ Model successfully memorized the videos!")
        else:
            print("⚠️ Model has not fully memorized the videos")
    
    print("="*60)
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()