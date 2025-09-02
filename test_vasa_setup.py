#!/usr/bin/env python3
"""Test script to verify VASA setup is working correctly."""

import sys
import os
from pathlib import Path

# Add nemo to path
sys.path.insert(0, 'nemo')

print("=" * 60)
print("VASA Setup Test")
print("=" * 60)

# Test 1: Check config loading
try:
    from omegaconf import OmegaConf
    config = OmegaConf.load('vasa_config.yaml')
    print("✓ Config loaded successfully")
    print(f"  - Volumetric model: {config.paths.volumetric_model}")
    print(f"  - Volumetric config: {config.paths.volumetric_config}")
    print(f"  - Data dir: {config.paths.data_dir}")
    print(f"  - Video folder: {config.paths.video_folder}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

# Test 2: Check if paths exist
paths_to_check = [
    ('Volumetric config', config.paths.volumetric_config),
    ('Volumetric model', config.paths.volumetric_model),
    ('Data directory', config.paths.data_dir),
    ('Video folder', config.paths.video_folder),
    ('Cache directory', config.paths.cache_dir),
    ('Checkpoint directory', config.paths.checkpoint_dir),
]

print("\nChecking paths:")
all_paths_ok = True
for name, path in paths_to_check:
    path_obj = Path(path)
    if path_obj.exists():
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: {path} (not found)")
        all_paths_ok = False

# Test 3: Try importing VASA modules
print("\nImporting VASA modules:")
modules_to_import = [
    'vasa_model',
    'vasa_dataset',
    'vasa_trainer',
    'vasa_scheduler',
    'expression_normalizer',
]

for module in modules_to_import:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError as e:
        print(f"  ✗ {module}: {e}")

# Test 4: Try importing nemo modules
print("\nImporting nemo modules:")
nemo_modules = [
    'logger',
    'mem',
    'models.stage_1.volumetric_avatar.va'
]

for module in nemo_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError as e:
        print(f"  ✗ {module}: {e}")

print("\n" + "=" * 60)
if all_paths_ok:
    print("✓ Setup looks good! You can now run vasa_trainer.py")
    print("\nTo run the trainer:")
    print("  python vasa_trainer.py")
else:
    print("⚠ Some paths are missing. Please check the setup.")
print("=" * 60)