#!/usr/bin/env python3
"""Test script to verify vi.py setup without needing a checkpoint."""

import os
import sys
from pathlib import Path

print("=" * 60)
print("VI.PY Setup Test")
print("=" * 60)

# Test 1: Check if config exists
configs = ['config_stage2.yaml', 'vasa_config.yaml']
for config in configs:
    if Path(config).exists():
        print(f"✓ {config} exists")
    else:
        print(f"✗ {config} not found")

# Test 2: Check symlinks
symlinks = {
    'repos': 'nemo/repos',
    'models': 'nemo/models', 
    'logs': 'nemo/logs'
}

print("\nSymlinks:")
for link, target in symlinks.items():
    if Path(link).exists():
        if Path(link).is_symlink():
            actual_target = os.readlink(link)
            print(f"  ✓ {link} -> {actual_target}")
        else:
            print(f"  ⚠ {link} exists but is not a symlink")
    else:
        print(f"  ✗ {link} symlink missing (should point to {target})")

# Test 3: Check for volumetric model
vol_model = "logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth"
if Path(vol_model).exists():
    print(f"\n✓ Volumetric model found: {vol_model}")
else:
    print(f"\n✗ Volumetric model not found: {vol_model}")

# Test 4: Check for checkpoints
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if checkpoints:
        print(f"\n✓ Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints[:5]:  # Show first 5
            print(f"  - {cp.name}")
    else:
        print("\n⚠ Checkpoints directory exists but is empty")
        print("  vi.py requires a trained checkpoint to run inference")
else:
    print("\n✗ Checkpoints directory not found")

# Test 5: Check test video
test_video = "junk/ovs-GiY_848_1.mp4"
if Path(test_video).exists():
    print(f"\n✓ Test video found: {test_video}")
else:
    print(f"\n⚠ Test video not found: {test_video}")
    videos = list(Path("junk").glob("*.mp4")) if Path("junk").exists() else []
    if videos:
        print(f"  Available videos: {[v.name for v in videos[:3]]}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

can_run = False
if not Path("checkpoints").exists() or not list(Path("checkpoints").glob("*.pt")):
    print("✗ vi.py CANNOT run - no trained checkpoints available")
    print("\nTo use vi.py for inference:")
    print("1. Train a model using: python vasa_trainer.py")
    print("2. Or copy a checkpoint from EMOPortraits to checkpoints/")
    print("3. Update epoch number in vi.py to match your checkpoint")
else:
    can_run = True
    print("✓ vi.py can potentially run inference")
    print("\nTo run vi.py:")
    print("1. Edit vi.py to set the correct epoch number")
    print("2. Run: python vi.py")

print("=" * 60)