#!/usr/bin/env python3
"""
Check if there's confusion about which image is the identity.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import torch

# Load IMG_1.png
img1 = Image.open("nemo/data/IMG_1.png").convert('RGB')
img1_np = np.array(img1)

print(f"IMG_1.png original size: {img1.size}")
print(f"IMG_1.png array shape: {img1_np.shape}")

# Resize for display
img1_small = img1.resize((512, 512), Image.LANCZOS)

# Check cached identity
with h5py.File("proper_face_attributes.h5", 'r') as f:
    # Print what's in the cache
    print("\nCache contents:")
    print(f"Keys: {list(f.keys())}")
    print(f"Attributes: {list(f.attrs.keys())}")

    if 'identity_frame' in f:
        cached_identity = f['identity_frame'][:]
        print(f"\nCached identity shape: {cached_identity.shape}")
        print(f"Cached identity range: [{cached_identity.min()}, {cached_identity.max()}]")

        # Convert to displayable
        if len(cached_identity.shape) == 4:
            cached_identity = cached_identity[0]
        if cached_identity.shape[0] == 3:
            cached_identity = np.transpose(cached_identity, (1, 2, 0))
        if cached_identity.min() < 0:
            cached_identity = (cached_identity + 1) / 2
        cached_identity = np.clip(cached_identity, 0, 1)

        # Create comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img1_small)
        axes[0].set_title("IMG_1.png from disk\n(Should be a MAN)", fontsize=12, weight='bold', color='blue')
        axes[0].axis('off')

        axes[1].imshow(cached_identity)
        axes[1].set_title("Cached 'identity_frame'\n(What is this?)", fontsize=12, weight='bold', color='red')
        axes[1].axis('off')

        # Check a random cached frame
        if 'frame_0005' in f:
            frame = f['frame_0005/frame'][:]
            if len(frame.shape) == 4:
                frame = frame[0]
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.min() < 0:
                frame = (frame + 1) / 2
            frame = np.clip(frame, 0, 1)

            axes[2].imshow(frame)
            axes[2].set_title("Cached frame_0005\n(Video frame)", fontsize=12)
            axes[2].axis('off')

        plt.suptitle("Identity Confusion Check", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("identity_confusion_check.png", dpi=150, bbox_inches='tight')
        plt.close()

        print("\nSaved identity_confusion_check.png")

        # Also save them individually for clarity
        img1_small.save("check_img1_from_disk.png")
        cached_img = Image.fromarray((cached_identity * 255).astype(np.uint8))
        cached_img.save("check_cached_identity.png")

        print("Saved individual images:")
        print("  - check_img1_from_disk.png (should be a man)")
        print("  - check_cached_identity.png (what's in the cache)")

print("\n" + "="*60)
print("IMPORTANT: Check if the cached identity is the WRONG person!")
print("The cache might have been created with a different image!")
print("="*60)