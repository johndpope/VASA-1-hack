#!/usr/bin/env python3
"""Test thumbnail generation"""

import torch
import numpy as np
from thumbnail_generator import generate_simple_thumbnail, create_debug_thumbnail
from PIL import Image

def test_thumbnail_generation():
    """Test the thumbnail generation functions"""
    
    print("Testing thumbnail generation...")
    
    # Create dummy motion data
    B, T = 1, 20
    motion_data = {
        'theta': torch.randn(B, T, 3, 4),
        'rotation': torch.randn(B, T, 3) * 0.1,
        'translation': torch.randn(B, T, 3) * 0.05,
        'expression': torch.randn(B, T, 256),
        'gaze': torch.randn(B, T, 2) * 0.1,
    }
    
    # Add some motion to make it interesting
    for t in range(T):
        motion_data['theta'][0, t] += torch.randn(3, 4) * 0.01 * t
    
    # Test simple thumbnail
    print("\n1. Testing simple thumbnail (motion stats only)...")
    try:
        thumbnail = generate_simple_thumbnail(motion_data, size=(512, 512))
        print(f"   ✓ Generated thumbnail shape: {thumbnail.shape}")
        
        # Save it
        img = Image.fromarray(thumbnail)
        img.save("test_simple_thumbnail.png")
        print("   ✓ Saved to test_simple_thumbnail.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test debug thumbnail with overlay
    print("\n2. Testing debug thumbnail with overlay...")
    try:
        # Create a dummy frame
        dummy_frame = torch.rand(3, 256, 256)
        
        thumbnail = create_debug_thumbnail(
            dummy_frame,
            motion_params=motion_data,
            size=(512, 512),
            add_overlay=True
        )
        print(f"   ✓ Generated thumbnail shape: {thumbnail.shape}")
        
        # Save it
        img = Image.fromarray(thumbnail)
        img.save("test_debug_thumbnail.png")
        print("   ✓ Saved to test_debug_thumbnail.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test with static motion (should show red indicator)
    print("\n3. Testing with static motion (should show warning)...")
    try:
        # Create static motion
        static_motion = {
            'theta': torch.ones(B, T, 3, 4) * 0.5,  # No variation
            'rotation': torch.zeros(B, T, 3),
            'translation': torch.zeros(B, T, 3),
            'expression': torch.ones(B, T, 256) * 0.1,
        }
        
        thumbnail = generate_simple_thumbnail(static_motion, size=(512, 512))
        print(f"   ✓ Generated static thumbnail shape: {thumbnail.shape}")
        
        # Save it
        img = Image.fromarray(thumbnail)
        img.save("test_static_thumbnail.png")
        print("   ✓ Saved to test_static_thumbnail.png (should show red/warning)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Thumbnail generation test complete!")
    print("Check the generated PNG files to verify the visualizations.")

if __name__ == "__main__":
    test_thumbnail_generation()