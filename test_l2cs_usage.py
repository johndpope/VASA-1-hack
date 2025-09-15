#!/usr/bin/env python3
"""Test L2CS actual usage to identify the protobuf error"""

import sys
import traceback
import numpy as np
import cv2

sys.path.insert(0, 'L2CS-Net')

print("Testing L2CS pipeline with actual inference...")

try:
    from l2cs.pipeline import Pipeline

    # Create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image[:, :] = [128, 128, 128]  # Gray image

    print("\n1. Creating Pipeline with detector...")
    pipeline = Pipeline(
        weights='models/L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device='cuda',
        include_detector=True
    )
    print("✓ Pipeline with detector created")

    print("\n2. Running inference on dummy image...")
    results = pipeline.step(dummy_image)
    print(f"✓ Inference completed. Results: pitch shape={results.pitch.shape}, yaw shape={results.yaw.shape}")

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

    # Check if it's the SymbolDatabase error
    if "SymbolDatabase" in str(e) or "GetPrototype" in str(e):
        print("\n⚠️  This is the protobuf SymbolDatabase error!")
        print("The error occurs during runtime, not import.")

print("\nNow testing without detector...")

try:
    from l2cs.pipeline import Pipeline

    pipeline = Pipeline(
        weights='models/L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device='cuda',
        include_detector=False
    )
    print("✓ Pipeline without detector created")

    # Prepare a face crop (224x224 RGB)
    face_crop = np.ones((224, 224, 3), dtype=np.uint8) * 128

    print("Running inference on face crop...")
    results = pipeline.step(face_crop)
    print(f"✓ Inference completed. Results: pitch={results.pitch}, yaw={results.yaw}")

except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()