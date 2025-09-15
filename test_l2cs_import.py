#!/usr/bin/env python3
"""Test L2CS import to identify the exact error"""

import sys
import traceback

print("Python version:", sys.version)
print("\nAttempting to import face_detection...")

try:
    from face_detection import RetinaFace
    print("✓ face_detection imported successfully")
except Exception as e:
    print(f"✗ face_detection import failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\nAttempting to import L2CS components...")

try:
    # First try importing L2CS utils
    sys.path.insert(0, 'L2CS-Net')
    from l2cs.utils import select_device
    print("✓ l2cs.utils imported successfully")
except Exception as e:
    print(f"✗ l2cs.utils import failed: {e}")
    traceback.print_exc()

try:
    from l2cs.model import L2CS
    print("✓ l2cs.model imported successfully")
except Exception as e:
    print(f"✗ l2cs.model import failed: {e}")
    traceback.print_exc()

try:
    from l2cs.pipeline import Pipeline
    print("✓ l2cs.pipeline imported successfully")
except Exception as e:
    print(f"✗ l2cs.pipeline import failed: {e}")
    traceback.print_exc()

print("\nAttempting to create Pipeline...")
try:
    pipeline = Pipeline(
        weights=None,
        arch='ResNet50',
        device='cpu',
        include_detector=False  # Skip detector to isolate issue
    )
    print("✓ Pipeline created successfully (without detector)")
except Exception as e:
    print(f"✗ Pipeline creation failed: {e}")
    traceback.print_exc()

print("\nTest complete.")