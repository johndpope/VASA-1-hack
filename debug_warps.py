#!/usr/bin/env python3
"""Debug script to check warp values and generated frames"""

import h5py
import numpy as np
import torch
from PIL import Image
import cv2

# Check warp statistics
print("Checking motion warps in H5 file...")
with h5py.File('motion_warps.h5', 'r') as f:
    num_frames = f.attrs['num_frames']
    print(f"Total frames: {num_frames}")

    # Check if warps are identity-like (which would be bad for self-driving)
    for i in range(min(3, num_frames)):
        frame_key = f'frame_{i:04d}'
        if frame_key in f:
            uv_warps = f[frame_key]['uv_warps'][:]
            theta = f[frame_key]['theta'][:]

            # Check if UV warps are close to identity
            # Identity warp would have values close to grid coordinates
            d, s = 16, 64  # from args
            identity_grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, d),
                torch.linspace(-1, 1, s),
                torch.linspace(-1, 1, s),
                indexing='ij'
            ), dim=-1).numpy()

            # UV warps shape: [1, 16, 64, 64, 3]
            warp_diff = np.abs(uv_warps[0] - identity_grid).mean()

            print(f"\n{frame_key}:")
            print(f"  UV warp deviation from identity: {warp_diff:.4f}")
            print(f"  Theta rotation matrix:")
            print(theta[0, :3, :3])
            print(f"  Theta translation: {theta[0, :3, 3]}")

# Check generated frames
print("\nChecking generated video frames...")
cap = cv2.VideoCapture('junk/output_driven.mp4')
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Check first frame statistics
        print(f"First frame shape: {frame.shape}")
        print(f"First frame range: [{frame.min()}, {frame.max()}]")
        print(f"First frame mean: {frame.mean():.2f}")

        # Save first frame for inspection
        cv2.imwrite('junk/debug_first_frame.jpg', frame)
        print("Saved first frame to junk/debug_first_frame.jpg")
    cap.release()

# Also check the source image
source = Image.open('junk/source.jpg')
print(f"\nSource image size: {source.size}")
print(f"Source image mode: {source.mode}")