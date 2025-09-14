#!/usr/bin/env python3
"""Debug script to check expression variation in dataset"""

import torch
import numpy as np
import sys
sys.path.append('nemo')
from vasa_dataset import VASAIntegratedDataset
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from volumetric_avatar.src.generator import Generator

def debug_expression_variation():
    print("Loading models...")

    # Load config
    config = OmegaConf.load('overfit_config.yaml')

    # Load volumetric avatar model
    model_path = '/media/oem/12TB/VASA-1-hack/pretrained_weights/insightface_models/liveportrait_ckpts/base_models/warping_module.pth'
    volumetric_avatar = Generator(model='spade_generator')
    model_dict = torch.load(model_path, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    print("Loading dataset...")
    # Create dataset
    dataset = VASAIntegratedDataset(
        video_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/",
        emo_model=volumetric_avatar,
        max_videos=1,
        window_size=50,
        stride=50,
        context_size=10,
        random_seed=42,
        cache_dir='cache'
    )

    print(f"Dataset has {len(dataset)} windows")

    # Get first window
    window_data = dataset[0]

    # Check expression embeddings
    if 'expression_embed' in window_data:
        expr = window_data['expression_embed']
        print(f"\nExpression embed shape: {expr.shape}")
        print(f"Expression dtype: {expr.dtype}")

        # Calculate statistics
        mean_per_frame = expr.mean(dim=-1)  # Mean across 128 dims for each frame
        std_per_frame = expr.std(dim=-1)    # Std across 128 dims for each frame

        print(f"\nPer-frame statistics:")
        print(f"Mean range: [{mean_per_frame.min():.4f}, {mean_per_frame.max():.4f}]")
        print(f"Std range: [{std_per_frame.min():.4f}, {std_per_frame.max():.4f}]")

        # Check variation between frames
        frame_diff = torch.diff(expr, dim=0)  # Difference between consecutive frames
        diff_norm = torch.norm(frame_diff, dim=-1)  # L2 norm of differences

        print(f"\nFrame-to-frame variation:")
        print(f"Difference norm - Mean: {diff_norm.mean():.6f}, Std: {diff_norm.std():.6f}")
        print(f"Difference norm - Min: {diff_norm.min():.6f}, Max: {diff_norm.max():.6f}")

        # Check if expressions are constant
        is_constant = torch.allclose(expr[0], expr, atol=1e-5)
        print(f"\nAre all frames identical? {is_constant}")

        if is_constant:
            print("WARNING: Expression embeddings are constant across frames!")

            # Check if they're all zeros
            is_zero = torch.allclose(expr, torch.zeros_like(expr), atol=1e-5)
            print(f"Are they all zeros? {is_zero}")

            # Print first few values
            print(f"\nFirst frame values (first 10 dims): {expr[0, :10].tolist()}")
        else:
            # Plot variation
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(diff_norm.numpy())
            plt.title('Frame-to-frame Expression Difference (L2 norm)')
            plt.xlabel('Frame pair')
            plt.ylabel('L2 norm of difference')

            plt.subplot(1, 2, 2)
            # Plot first 5 expression dimensions over time
            for i in range(min(5, expr.shape[-1])):
                plt.plot(expr[:, i].numpy(), label=f'Dim {i}')
            plt.title('First 5 Expression Dimensions Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.legend()

            plt.tight_layout()
            plt.savefig('expression_variation_debug.png')
            print("\nPlot saved to expression_variation_debug.png")

    # Also check theta (pose) variation for comparison
    if 'theta' in window_data:
        theta = window_data['theta']
        print(f"\n--- THETA (pose) for comparison ---")
        print(f"Theta shape: {theta.shape}")

        theta_flat = theta.view(theta.shape[0], -1)  # Flatten to [T, 12]
        theta_diff = torch.diff(theta_flat, dim=0)
        theta_diff_norm = torch.norm(theta_diff, dim=-1)

        print(f"Theta frame-to-frame variation:")
        print(f"Difference norm - Mean: {theta_diff_norm.mean():.6f}, Std: {theta_diff_norm.std():.6f}")

        is_theta_constant = torch.allclose(theta[0], theta, atol=1e-5)
        print(f"Is theta constant? {is_theta_constant}")

if __name__ == "__main__":
    debug_expression_variation()