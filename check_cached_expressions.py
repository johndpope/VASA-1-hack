#!/usr/bin/env python3
"""Check cached expression embeddings for variation"""

import torch
import os
import glob

def check_cached_expressions():
    # Find cached expression files
    cache_dir = "cache/motion_features"
    expr_files = glob.glob(os.path.join(cache_dir, "*_expressions.pt"))

    if not expr_files:
        print(f"No cached expression files found in {cache_dir}")
        return

    print(f"Found {len(expr_files)} cached expression files")

    for expr_file in expr_files[:3]:  # Check first 3 files
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(expr_file)}")
        print('='*60)

        # Load expressions
        expr_data = torch.load(expr_file, map_location='cpu')

        if 'expression_embed' in expr_data:
            expr = expr_data['expression_embed']
            print(f"Expression shape: {expr.shape}")

            # Remove batch dimension if present
            if len(expr.shape) == 3 and expr.shape[0] == 1:
                expr = expr.squeeze(0)

            # Calculate variation
            if len(expr.shape) == 2:  # [T, 128]
                # Check if all frames are identical
                is_constant = torch.allclose(expr[0], expr, atol=1e-5)
                print(f"All frames identical? {is_constant}")

                if not is_constant:
                    # Calculate frame-to-frame differences
                    frame_diff = torch.diff(expr, dim=0)
                    diff_norm = torch.norm(frame_diff, dim=-1)

                    print(f"\nFrame-to-frame variation:")
                    print(f"  Mean diff: {diff_norm.mean():.6f}")
                    print(f"  Std diff: {diff_norm.std():.6f}")
                    print(f"  Max diff: {diff_norm.max():.6f}")
                    print(f"  Min diff: {diff_norm.min():.6f}")

                    # Show some actual values
                    print(f"\nFirst 5 values of frame 0: {expr[0, :5].tolist()}")
                    print(f"First 5 values of frame 25: {expr[min(25, len(expr)-1), :5].tolist()}")
                    print(f"First 5 values of last frame: {expr[-1, :5].tolist()}")
                else:
                    print("\n⚠️ WARNING: Expression embeddings are constant!")
                    print(f"First 10 values: {expr[0, :10].tolist()}")

                    # Check if they're zeros
                    is_zero = torch.allclose(expr, torch.zeros_like(expr), atol=1e-5)
                    print(f"All zeros? {is_zero}")
            else:
                print(f"Unexpected shape: {expr.shape}")
        else:
            print("No 'expression_embed' key found in cached data")
            print(f"Available keys: {list(expr_data.keys())}")

if __name__ == "__main__":
    check_cached_expressions()