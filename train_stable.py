#!/usr/bin/env python3
"""
Stable Training Configuration
==============================
Fixes negative loss issue and ensures proper convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from omegaconf import OmegaConf

sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableTDDLoss(nn.Module):
    """Fixed TDD loss without negative values"""
    
    def __init__(self):
        super().__init__()
        
        # Positive-only loss weights
        self.weights = {
            'reconstruction': 1.0,      # L2 reconstruction
            'expression': 1.5,          # Expression preservation  
            'motion_smoothness': 0.3,   # Temporal smoothness
            'pose_consistency': 0.5,    # Pose accuracy
            'perceptual': 0.2,         # Perceptual quality
        }
        
        # Motion targets (not rewards)
        self.motion_target = 20.0  # Target motion magnitude
        self.expression_target = 0.95  # Target expression similarity
        
    def forward(self, outputs, targets, conditions=None):
        losses = {}
        
        # 1. Reconstruction Loss (always positive)
        if 'theta' in outputs and 'theta' in targets:
            recon_loss = F.mse_loss(outputs['theta'], targets['theta'])
            losses['reconstruction'] = recon_loss * self.weights['reconstruction']
        
        # 2. Expression Preservation (always positive)
        if 'expression_embed' in outputs and 'expression_embed' in targets:
            expr_loss = F.mse_loss(outputs['expression_embed'], targets['expression_embed'])
            
            # Add cosine similarity loss (1 - similarity, so always positive)
            cos_sim = F.cosine_similarity(
                outputs['expression_embed'].reshape(-1, outputs['expression_embed'].size(-1)),
                targets['expression_embed'].reshape(-1, targets['expression_embed'].size(-1)),
                dim=-1
            ).mean()
            
            expr_loss = expr_loss + (1.0 - cos_sim) * 0.5
            losses['expression'] = expr_loss * self.weights['expression']
        
        # 3. Motion Smoothness (always positive)
        if 'theta' in outputs:
            if outputs['theta'].shape[1] > 1:
                motion_diff = torch.diff(outputs['theta'], dim=1)
                smoothness_loss = torch.norm(motion_diff, dim=-1).mean()
                losses['motion_smoothness'] = smoothness_loss * self.weights['motion_smoothness']
        
        # 4. Motion Target Loss (distance from desired motion, always positive)
        if 'theta' in outputs and outputs['theta'].shape[1] > 1:
            # Flatten theta to compute motion
            theta_flat = outputs['theta'].reshape(outputs['theta'].shape[0], outputs['theta'].shape[1], -1)
            actual_motion = torch.norm(torch.diff(theta_flat, dim=1), dim=-1).mean()
            motion_distance = F.mse_loss(actual_motion, torch.tensor(self.motion_target).cuda())
            losses['motion_target'] = motion_distance * 0.1
        
        # 5. Pose Consistency (always positive)
        if 'scale' in outputs and 'scale' in targets:
            scale_loss = F.mse_loss(outputs['scale'], targets['scale'])
            losses['pose_consistency'] = scale_loss * self.weights['pose_consistency']
        
        # Total loss (guaranteed positive)
        total_loss = sum(losses.values())
        
        # Add small epsilon to prevent exactly zero
        total_loss = total_loss + 1e-6
        
        # Compute test metrics
        tests = {}
        if 'expression_embed' in outputs and 'expression_embed' in targets:
            with torch.no_grad():
                cos_sim_val = F.cosine_similarity(
                    outputs['expression_embed'].reshape(-1, outputs['expression_embed'].size(-1)),
                    targets['expression_embed'].reshape(-1, targets['expression_embed'].size(-1)),
                    dim=-1
                ).mean().item()
                tests['expression_similarity'] = cos_sim_val > 0.9
                losses['metric_expr_sim'] = cos_sim_val
        
        if 'theta' in outputs and outputs['theta'].shape[1] > 1:
            with torch.no_grad():
                theta_flat = outputs['theta'].reshape(outputs['theta'].shape[0], outputs['theta'].shape[1], -1)
                motion_mag = torch.norm(torch.diff(theta_flat, dim=1), dim=-1).mean().item()
                tests['has_motion'] = motion_mag > 0.1
                losses['metric_motion'] = motion_mag
        
        return total_loss, losses, tests


def train_stable():
    """Train with stable, positive-only losses"""
    
    print("\n" + "="*70)
    print("STABLE VASA TRAINING")
    print("Fixed: No negative losses, proper convergence")
    print("="*70)
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Initialize W&B
    wandb.init(
        project="vasa-stable",
        name="stable_training",
        config=OmegaConf.to_container(config)
    )
    
    # Load volumetric avatar
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda().eval()
    
    # Initialize VASA model
    model = VASAModel(config, volumetric_avatar).cuda()
    
    # Load checkpoint if exists
    checkpoint_path = Path('checkpoints/stable/best_model.pth')
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        start_step = 0
        best_loss = float('inf')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # No need for noise scheduler - handled differently
    
    # Loss module
    loss_module = StableTDDLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=2000,
        eta_min=1e-5
    )
    
    # Create simple test batch
    batch_size = 1
    seq_len = 50
    
    print(f"\nStarting training from step {start_step}")
    print("Loss weights:", loss_module.weights)
    print("-" * 70)
    
    for step in range(start_step, 2000):
        model.train()
        
        # Create synthetic batch (for testing)
        batch = {
            'theta': torch.randn(batch_size, seq_len, 3, 4).cuda(),  # [B, T, 3, 4]
            'expression_embed': torch.randn(batch_size, seq_len, 128).cuda(),
            'scale': torch.ones(batch_size, seq_len, 3).cuda(),  # [B, T, 3]
            'rotation': torch.randn(batch_size, seq_len, 3).cuda(),  # [B, T, 3]
            'translation': torch.randn(batch_size, seq_len, 3).cuda(),  # [B, T, 3]
            'audio_features': torch.randn(batch_size, seq_len, 768).cuda()
        }
        
        # Add controlled noise that decays over time
        noise_level = torch.ones(batch_size, device='cuda') * (0.3 * (1 - step / 2000))
        
        # Forward pass
        outputs = model(
            motion_data=batch,
            noise_level=noise_level,
            conditions={
                'audio_features': batch['audio_features'],
                'gaze': torch.zeros(batch_size, seq_len, 2).cuda(),
                'head_distance': torch.ones(batch_size, seq_len, 1).cuda() * 0.5,
                'emotion': torch.zeros(batch_size, seq_len, 2).cuda()
            }
        )
        
        # Compute loss
        total_loss, losses, test_results = loss_module(outputs, batch)
        
        # Ensure loss is positive
        if total_loss.item() < 0:
            logger.error(f"Negative loss detected: {total_loss.item()}")
            total_loss = torch.abs(total_loss) + 1.0
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()
        
        # Logging
        if step % 50 == 0:
            test_pass_rate = sum(test_results.values()) / len(test_results) if test_results else 0
            
            print(f"Step {step}: Loss={total_loss.item():.2f}, Tests={test_pass_rate*100:.1f}%, LR={lr_scheduler.get_last_lr()[0]:.6f}")
            
            if test_results:
                print("\n  Test Results:")
                for name, passed in test_results.items():
                    metric_name = f"metric_{name.replace('_', '')}"
                    metric_val = losses.get(metric_name, 'N/A')
                    status = "✅" if passed else "❌"
                    print(f"    {status} {name}: {metric_val}")
            
            # Log to W&B
            log_dict = {
                'step': step,
                'loss/total': total_loss.item(),
                'lr': lr_scheduler.get_last_lr()[0],
                'test_pass_rate': test_pass_rate
            }
            
            for name, value in losses.items():
                if not name.startswith('metric_'):
                    log_dict[f'loss/{name}'] = value.item() if torch.is_tensor(value) else value
                else:
                    log_dict[f'metric/{name[7:]}'] = value
            
            wandb.log(log_dict)
        
        # Save checkpoint
        if step % 500 == 0 and step > 0:
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                    'best_loss': best_loss
                }
                
                torch.save(checkpoint, checkpoint_path)
                print(f"\n✅ Saved best model at step {step} (loss: {best_loss:.4f})\n")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print("Model saved to:", checkpoint_path)
    print("="*70)
    
    wandb.finish()


if __name__ == "__main__":
    train_stable()