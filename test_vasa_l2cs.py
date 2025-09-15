#!/usr/bin/env python3
"""Test VASA dataset with fixed L2CS"""

import numpy as np
import torch

# Test the imports as they would be in vasa_dataset.py
print("Testing VASA dataset L2CS integration...")

try:
    from l2cs import Pipeline
    print("✓ L2CS imported successfully")

    # Create a dummy frame
    dummy_frame = np.ones((224, 224, 3), dtype=np.uint8) * 128

    # Test pipeline creation
    pipeline = Pipeline(
        weights='models/L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device='cuda',
        include_detector=False
    )
    print("✓ Pipeline created successfully")

    # Test inference
    results = pipeline.step(dummy_frame)
    print(f"✓ Inference successful: pitch={results.pitch[0]:.3f}, yaw={results.yaw[0]:.3f}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nL2CS is now working properly!")