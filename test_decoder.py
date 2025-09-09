#!/usr/bin/env python3
"""Test script for the refactored TransformerDecoder-based VASAModel."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import traceback

# Import the refactored model
from vasa_model import VASAModel, HolisticMotionTransformer

def test_decoder_architecture():
    """Test the decoder-based transformer architecture."""
    
    print("Loading configuration...")
    # Load overfit config for testing
    config = OmegaConf.load('overfit_config.yaml')
    
    # Create dummy volumetric avatar
    class DummyVolumetricAvatar(nn.Module):
        def __init__(self):
            super().__init__()
            self.args = type('Args', (), {
                'latent_volume_channels': 32,
                'latent_volume_depth': 16,
                'latent_volume_size': 16
            })()
    
    print(f"Creating model with {config.model.n_layers} transformer layers...")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    volumetric_avatar = DummyVolumetricAvatar()
    model = VASAModel(config, volumetric_avatar, device)
    model = model.to(device)
    
    print(f"Model created successfully on {device}")
    print(f"Transformer config:")
    print(f"  - Layers: {config.model.n_layers}")
    print(f"  - Heads: {config.model.n_heads}")
    print(f"  - Hidden dim: {config.model.hidden_dim}")
    print(f"  - Feedforward dim: {config.model.dim_feedforward}")
    
    # Test forward pass
    B = 2  # Batch size
    T = config.motion.window_size  # Sequence length
    
    print(f"\nTesting forward pass with B={B}, T={T}...")
    
    # Create dummy motion data
    motion_data = {
        'theta': torch.randn(B, T, 3, 4).to(device),
        'scale': torch.randn(B, T, 3).to(device),
        'rotation': torch.randn(B, T, 3).to(device),
        'translation': torch.randn(B, T, 3).to(device),
        'expression_embed': torch.randn(B, T, 128).to(device)
    }
    
    # Create noise level
    noise_level = torch.randint(0, config.diffusion.num_steps, (B,)).to(device)
    
    # Create conditions
    conditions = {
        'audio_features': torch.randn(B, T, 768).to(device),  # Wav2Vec features
        'gaze': torch.randn(B, T, 2).to(device),
        'head_distance': torch.randn(B, T, 1).to(device),
        'emotion': torch.randn(B, T, 2).to(device),
        'speed_bucket': torch.randn(B, T, 1).to(device),
    }
    
    # Test with context
    if config.motion.context_size > 0:
        K = config.motion.context_size
        prev_context = {
            'theta': torch.randn(B, K, 3, 4).to(device),
            'rotation': torch.randn(B, K, 3).to(device),
            'translation': torch.randn(B, K, 3).to(device),
            'expression_embed': torch.randn(B, K, 128).to(device)
        }
        print(f"  with context K={K}")
    else:
        prev_context = None
        print(f"  without context")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(
                motion_data=motion_data,
                noise_level=noise_level,
                conditions=conditions,
                prev_context=prev_context
            )
        
        print("\nForward pass successful!")
        print("Output keys:", list(output.keys()))
        
        # Check output shapes
        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Test decoder mask generation
        print("\nTesting causal mask generation...")
        transformer = model.motion_transformer
        total_seq_len = T + (K if prev_context else 0)
        
        # Generate mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(total_seq_len).to(device)
        print(f"Causal mask shape: {tgt_mask.shape}")
        print(f"Mask values (first 5x5):\n{tgt_mask[:5, :5]}")
        
        if prev_context:
            # Test custom mask for context
            custom_mask = torch.zeros(total_seq_len, total_seq_len, device=device)
            custom_mask[K:, :K] = 0  # Current can attend to all context
            causal_part = torch.triu(torch.ones(T, T, device=device), diagonal=1) * float('-inf')
            custom_mask[K:, K:] = causal_part
            print(f"\nCustom mask with context (first 8x8):\n{custom_mask[:8, :8]}")
        
        print("\n✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_decoder_architecture()
    sys.exit(0 if success else 1)