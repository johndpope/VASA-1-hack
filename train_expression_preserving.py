#!/usr/bin/env python3
"""
Training with Expression Preservation
======================================
Uses strong expression reconstruction loss to maintain consistency.
"""

import torch
import sys
import os
import wandb
from pathlib import Path
from tqdm import tqdm
import logging

sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from tdd_expression_preserving import ExpressionPreservingTDDLoss
from omegaconf import OmegaConf
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_expression_preservation():
    """Train VASA with strong expression preservation"""
    
    print("\n" + "="*70)
    print("VASA Training with Expression Preservation")
    print("="*70)
    
    # Load config
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    # Initialize W&B
    wandb.init(
        project="vasa-expression",
        name="expression_preserving",
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
    
    # Initialize model
    model = VASAModel(config, volumetric_avatar).cuda()
    
    # Load checkpoint if exists
    checkpoint_path = Path('checkpoints/expression_preserving/best_model.pth')
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        start_epoch = 0
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    dataset = VASAIntegratedDataset(
        video_folder=config.paths.video_folder,
        emo_model=volumetric_avatar,
        window_size=config.motion.window_size,
        stride=config.motion.stride,
        sequence_length=config.dataset.sequence_length,
        cache_dir=config.paths.cache_dir
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize expression-preserving loss
    loss_module = ExpressionPreservingTDDLoss(
        expression_weight=2.5,  # Very strong expression preservation
        expression_codebook_weight=2.0,
        expression_temporal_weight=1.5,
        motion_weight=0.5,  # Lower motion weight
        lip_sync_weight=1.0
    )
    
    # Optimizer with different LR for expression parameters
    expr_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'expression' in name.lower():
            expr_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': expr_params, 'lr': 2e-3},  # Higher LR for expression
        {'params': other_params, 'lr': 5e-4}   # Lower LR for others
    ], weight_decay=0.01)
    
    # Curriculum stages
    curriculum_stages = [
        {'epochs': 50, 'name': 'expression_focus', 'expr_mult': 3.0},
        {'epochs': 100, 'name': 'balanced', 'expr_mult': 2.0},
        {'epochs': 200, 'name': 'fine_tuning', 'expr_mult': 1.5}
    ]
    
    current_stage_idx = 0
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}")
    print("-" * 70)
    
    best_expression_score = 0.0
    
    for epoch in range(start_epoch, config.train.num_epochs):
        # Update curriculum stage
        total_epochs = 0
        for stage_idx, stage in enumerate(curriculum_stages):
            total_epochs += stage['epochs']
            if epoch < total_epochs:
                if stage_idx != current_stage_idx:
                    current_stage_idx = stage_idx
                    print(f"\n📚 Curriculum Stage: {stage['name']}")
                    print(f"   Expression multiplier: {stage['expr_mult']}x")
                    
                    # Update loss weights
                    loss_module.weights['expression_reconstruction'] = 2.0 * stage['expr_mult']
                    loss_module.weights['expression_codebook'] = 1.5 * stage['expr_mult']
                break
        
        model.train()
        epoch_losses = []
        epoch_expr_scores = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to GPU
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].cuda()
            
            # Forward pass
            outputs = model(
                motion_data=batch,
                noise_level=torch.zeros(batch['theta'].shape[0], device='cuda'),
                conditions={
                    'audio_features': batch.get('audio_features'),
                    'gaze': batch.get('gaze', torch.zeros(1, 50, 2).cuda()),
                    'head_distance': batch.get('head_distance', torch.ones(1, 50, 1).cuda() * 0.5),
                    'emotion': batch.get('emotion', torch.zeros(1, 50, 2).cuda())
                }
            )
            
            # Compute expression-preserving loss
            total_loss, losses, test_results = loss_module(
                outputs=outputs,
                targets=batch,
                conditions={'audio_features': batch.get('audio_features')},
                stage='train'
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(total_loss.item())
            
            # Calculate expression score
            if 'expression_consistency' in losses:
                expr_score = losses.get('metric_expression_consistency', 0.0)
                epoch_expr_scores.append(expr_score)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'expr': f"{expr_score:.3f}" if epoch_expr_scores else "N/A"
            })
            
            # Log to W&B
            if batch_idx % 10 == 0:
                log_dict = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': total_loss.item(),
                    'stage': curriculum_stages[current_stage_idx]['name']
                }
                
                # Add individual losses
                for name, value in losses.items():
                    if torch.is_tensor(value):
                        log_dict[f'loss/{name}'] = value.item()
                    else:
                        log_dict[f'metric/{name}'] = value
                
                # Add test results
                passed_tests = sum(test_results.values()) if test_results else 0
                total_tests = len(test_results) if test_results else 1
                log_dict['test_pass_rate'] = passed_tests / total_tests
                
                wandb.log(log_dict)
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_expr_score = sum(epoch_expr_scores) / len(epoch_expr_scores) if epoch_expr_scores else 0
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Expression Score: {avg_expr_score:.3f}")
        
        # Save checkpoint if expression score improved
        if avg_expr_score > best_expression_score:
            best_expression_score = avg_expr_score
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'expression_score': avg_expr_score,
                'stage': curriculum_stages[current_stage_idx]['name']
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✅ Saved best model (expression score: {avg_expr_score:.3f})")
        
        # Validation every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_expr_scores = []
                
                for batch in dataloader:
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].cuda()
                    
                    outputs = model(
                        motion_data=batch,
                        noise_level=torch.zeros(batch['theta'].shape[0], device='cuda'),
                        conditions={
                            'audio_features': batch.get('audio_features'),
                            'gaze': batch.get('gaze', torch.zeros(1, 50, 2).cuda()),
                            'head_distance': batch.get('head_distance', torch.ones(1, 50, 1).cuda() * 0.5),
                            'emotion': batch.get('emotion', torch.zeros(1, 50, 2).cuda())
                        }
                    )
                    
                    total_loss, losses, test_results = loss_module(
                        outputs=outputs,
                        targets=batch,
                        conditions={'audio_features': batch.get('audio_features')},
                        stage='val'
                    )
                    
                    val_losses.append(total_loss.item())
                    if 'metric_expression_consistency' in losses:
                        val_expr_scores.append(losses['metric_expression_consistency'])
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_expr = sum(val_expr_scores) / len(val_expr_scores) if val_expr_scores else 0
                
                print(f"\n📊 Validation Results:")
                print(f"  Loss: {avg_val_loss:.4f}")
                print(f"  Expression Score: {avg_val_expr:.3f}")
                
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/expression_score': avg_val_expr,
                    'epoch': epoch
                })
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Expression Score: {best_expression_score:.3f}")
    print("="*70)
    
    wandb.finish()


if __name__ == "__main__":
    train_with_expression_preservation()