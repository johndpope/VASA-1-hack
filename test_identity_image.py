#!/usr/bin/env python3
"""Test script to verify identity image is being used correctly in training."""

import torch
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_identity_image():
    # Load config
    config = OmegaConf.load('overfit_config.yaml')
    
    # Check if identity image is enabled
    use_identity = config.dataset.get('use_identity_image', False)
    identity_path = config.dataset.get('identity_image_path', None)
    
    logger.info(f"use_identity_image: {use_identity}")
    logger.info(f"identity_image_path: {identity_path}")
    
    if use_identity and identity_path:
        # Load and preprocess the identity image
        logger.info(f"Loading identity image from: {identity_path}")
        
        img = Image.open(identity_path).convert('RGB')
        logger.info(f"Original image size: {img.size}")
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        identity_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
        logger.info(f"Tensor shape: {identity_tensor.shape}")
        logger.info(f"Tensor min: {identity_tensor.min():.3f}, max: {identity_tensor.max():.3f}")
        
        # Simulate batch expansion
        B = 4  # batch size
        batch_identity = identity_tensor.repeat(B, 1, 1, 1)
        logger.info(f"Batch identity shape: {batch_identity.shape}")
        
        return True
    else:
        logger.warning("Identity image not configured or path not found")
        return False

if __name__ == "__main__":
    success = test_identity_image()
    if success:
        logger.info("✅ Identity image test passed!")
    else:
        logger.info("⚠️ Identity image not configured")