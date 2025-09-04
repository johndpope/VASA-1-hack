#!/usr/bin/env python3
"""
Debug VASA Training Issues
==========================
This script helps identify why the model isn't training.
"""

import torch
import sys
import traceback
from pathlib import Path
from omegaconf import OmegaConf
import importlib

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from vasa_trainer import collate_vasa_batch
from logger import logger
import torch.nn.functional as F


def test_model_forward():
    """Test if the model can do a forward pass."""
    print("\n" + "="*60)
    print("Testing VASA Model Forward Pass")
    print("="*60)
    
    try:
        # Load config
        config = OmegaConf.load('vasa_config_fixed.yaml')
        config.train.turn_off_noise = True  # No noise for debugging
        
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
        print("   ✓ Volumetric avatar loaded")
        
        # Create VASA model
        print("\n2. Creating VASA model...")
        model = VASAModel(
            config=config,
            volumetric_avatar=volumetric_avatar,
            device='cuda'
        )
        model = model.cuda()
        model.eval()
        print("   ✓ VASA model created")
        
        # Create dummy data
        print("\n3. Creating dummy data...")
        B, T = 2, 5  # Batch size 2, sequence length 5
        
        motion_data = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),  # Fixed: should be [B, T, 3]
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()  # Fixed: should be 128
        }
        
        conditions = {
            'audio_features': torch.randn(B, T, 768).cuda(),
            'gaze': torch.randn(B, T, 2).cuda(),
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda()
        }
        
        print(f"   Motion data shapes:")
        for k, v in motion_data.items():
            print(f"     {k}: {v.shape}")
        
        # Test forward pass
        print("\n4. Testing forward pass...")
        with torch.no_grad():
            # Create noise
            noise = {k: torch.randn_like(v) for k, v in motion_data.items()}
            noise_level = torch.zeros(B, dtype=torch.long, device='cuda')
            
            # Add noise (minimal for t=0)
            noised_motion = model._add_noise_to_motion(
                motion_data=motion_data,
                noise=noise,
                noise_level=noise_level
            )
            
            # Forward pass
            outputs = model(
                motion_data=noised_motion,
                noise_level=noise_level,
                conditions=conditions,
                noise=noise
            )
            
            print(f"   Output keys: {outputs.keys()}")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"     {k}: shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
        
        # Test loss computation
        print("\n5. Testing loss computation...")
        loss_module = model.loss_module if hasattr(model, 'loss_module') else None
        
        if loss_module:
            # Compute reconstruction loss manually
            recon_losses = {}
            for key in ['theta', 'rotation', 'translation', 'expression_embed']:
                if key in outputs and key in motion_data:
                    loss = F.mse_loss(outputs[key], motion_data[key])
                    recon_losses[key] = loss.item()
                    print(f"   {key}_loss: {loss.item():.6f}")
            
            total_recon = sum(recon_losses.values())
            print(f"   Total reconstruction: {total_recon:.6f}")
        else:
            print("   No loss module found")
        
        # Check if model parameters require gradients
        print("\n6. Checking gradient flow...")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        # Check specific components
        components = [
            'motion_transformer',
            'motion_projections',
            'condition_embeddings'
        ]
        
        for comp_name in components:
            if hasattr(model, comp_name):
                comp = getattr(model, comp_name)
                comp_trainable = sum(p.numel() for p in comp.parameters() if p.requires_grad)
                print(f"   {comp_name}: {comp_trainable:,} trainable params")
        
        print("\n✅ Model forward pass successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in forward pass: {str(e)}")
        print(traceback.format_exc())
        return False


def test_data_pipeline():
    """Test if the data pipeline works correctly."""
    print("\n" + "="*60)
    print("Testing Data Pipeline")
    print("="*60)
    
    try:
        # Load config
        config = OmegaConf.load('vasa_config_fixed.yaml')
        
        # Load volumetric model
        print("\n1. Loading volumetric avatar for dataset...")
        model_path = config.paths.volumetric_model
        emo_config = OmegaConf.load(config.paths.volumetric_config)
        volumetric_avatar = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)
        
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        
        # Create dataset
        print("\n2. Creating dataset...")
        dataset = VASAIntegratedDataset(
            video_folder=config.paths.video_folder,
            emo_model=volumetric_avatar,
            max_videos=1,
            frame_size=(512, 512),
            sequence_length=5,
            cache_audio=False,
            preextract_audio=False,
            random_seed=42
        )
        print(f"   Dataset size: {len(dataset)}")
        
        # Get a sample
        print("\n3. Getting a sample...")
        sample = dataset[0]
        
        if sample is None:
            print("   ❌ Sample is None!")
            return False
        
        print(f"   Sample keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
        
        if 'windows' in sample:
            print(f"   Number of windows: {len(sample['windows'])}")
            if sample['windows']:
                window = sample['windows'][0]
                print(f"   First window keys: {window.keys()}")
                for k, v in window.items():
                    if isinstance(v, torch.Tensor):
                        print(f"     {k}: shape={v.shape}")
        
        # Test collate function
        print("\n4. Testing collate function...")
        batch = collate_vasa_batch([sample])
        
        if batch is None:
            print("   ❌ Batch is None!")
            return False
        
        print(f"   Batch keys: {batch.keys()}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"     {k}: shape={v.shape}, dtype={v.dtype}")
        
        print("\n✅ Data pipeline working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in data pipeline: {str(e)}")
        print(traceback.format_exc())
        return False


def test_loss_computation():
    """Test loss computation directly."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    try:
        # Create simple tensors
        print("\n1. Creating test tensors...")
        B, T = 2, 5
        
        pred = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        target = {
            'theta': torch.randn(B, T, 3, 4).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda()
        }
        
        # Compute MSE loss
        print("\n2. Computing MSE losses...")
        losses = {}
        for key in pred.keys():
            loss = F.mse_loss(pred[key], target[key])
            losses[key] = loss
            print(f"   {key}_loss: {loss.item():.6f}")
        
        total_loss = sum(losses.values())
        print(f"   Total loss: {total_loss.item():.6f}")
        
        # Check if loss is reasonable
        if total_loss.item() > 0 and total_loss.item() < 100:
            print("\n✅ Loss computation working!")
            return True
        else:
            print(f"\n⚠️ Loss value unusual: {total_loss.item()}")
            return False
        
    except Exception as e:
        print(f"\n❌ Error in loss computation: {str(e)}")
        print(traceback.format_exc())
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("VASA Training Diagnostics")
    print("="*60)
    
    results = {
        'Model Forward': test_model_forward(),
        'Data Pipeline': test_data_pipeline(),
        'Loss Computation': test_loss_computation()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✅ All tests passed! The model should be able to train.")
        print("\nPossible issues to check:")
        print("1. Learning rate might be too low/high")
        print("2. Gradient clipping might be too aggressive")
        print("3. Loss weights might need adjustment")
    else:
        print("\n❌ Some tests failed. Fix these issues before training.")
        print("\nRecommended actions:")
        if not results['Model Forward']:
            print("- Check model architecture and initialization")
        if not results['Data Pipeline']:
            print("- Check dataset paths and preprocessing")
        if not results['Loss Computation']:
            print("- Check loss function implementation")


if __name__ == "__main__":
    main()