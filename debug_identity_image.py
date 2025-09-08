#!/usr/bin/env python3
"""Debug script to check identity image processing."""

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_identity_image():
    # Load the identity image the same way as in training
    identity_path = "nemo/data/IMG_1.png"
    
    logger.info(f"Loading identity image from: {identity_path}")
    img = Image.open(identity_path).convert('RGB')
    
    # Check original image
    img_array = np.array(img)
    logger.info(f"Original image shape: {img_array.shape}")
    logger.info(f"Original image range: [{img_array.min()}, {img_array.max()}]")
    
    # Apply the same transform as in training
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    identity_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    
    logger.info(f"\nAfter transform:")
    logger.info(f"Tensor shape: {identity_tensor.shape}")
    logger.info(f"Tensor range: [{identity_tensor.min():.3f}, {identity_tensor.max():.3f}]")
    logger.info(f"Tensor mean: {identity_tensor.mean():.3f}")
    logger.info(f"Tensor std: {identity_tensor.std():.3f}")
    
    # Check what volumetric avatar expects
    logger.info("\n=== Volumetric Avatar Expectations ===")
    logger.info("The volumetric avatar expects images in range [-1, 1]")
    logger.info("Current normalization: (x - 0.5) / 0.5 maps [0,1] to [-1,1] ✓")
    
    # Test different normalizations
    logger.info("\n=== Testing Alternative Normalizations ===")
    
    # No normalization (just ToTensor)
    transform_no_norm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    tensor_no_norm = transform_no_norm(img).unsqueeze(0)
    logger.info(f"No normalization range: [{tensor_no_norm.min():.3f}, {tensor_no_norm.max():.3f}]")
    
    # ImageNet normalization
    transform_imagenet = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_imagenet = transform_imagenet(img).unsqueeze(0)
    logger.info(f"ImageNet norm range: [{tensor_imagenet.min():.3f}, {tensor_imagenet.max():.3f}]")
    
    return identity_tensor

if __name__ == "__main__":
    identity_tensor = check_identity_image()
    
    # Save a sample to visualize
    from torchvision.utils import save_image
    
    # Denormalize for saving
    denorm = identity_tensor * 0.5 + 0.5
    save_image(denorm, "debug_identity_normalized.png")
    logger.info("\nSaved normalized identity image to debug_identity_normalized.png")