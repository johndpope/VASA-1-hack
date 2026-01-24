#!/usr/bin/env python3
"""Test script to verify nemo/EMOPortraits and VASA setup is working correctly."""

import sys
import os
import platform
from pathlib import Path

# Add nemo to path
sys.path.insert(0, 'nemo')

print("=" * 60)
print("Nemo/EMOPortraits & VASA Setup Test")
print("=" * 60)

# =============================================================================
# SECTION 0: Platform and Device Detection
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 0: Platform & Device Detection")
print("=" * 60)

print(f"\n[0.1] System Info:")
print(f"  - Platform: {platform.system()} {platform.release()}")
print(f"  - Python: {sys.version.split()[0]}")
print(f"  - Architecture: {platform.machine()}")

print("\n[0.2] PyTorch Device Detection:")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  ✓ MPS (Apple Silicon) available")
        device = 'mps'
    else:
        print(f"  ⚠ No GPU acceleration - using CPU (training will be slow)")
        device = 'cpu'

    print(f"  - Default device: {device}")

    # Test basic tensor operation on device
    try:
        test_tensor = torch.zeros(1).to(device)
        print(f"  ✓ Device tensor test passed")
    except Exception as e:
        print(f"  ✗ Device tensor test failed: {e}")
        device = 'cpu'

except ImportError as e:
    print(f"  ✗ PyTorch not installed: {e}")
    device = None
except Exception as e:
    print(f"  ✗ PyTorch error: {e}")
    device = None

# Track overall test status
all_tests_passed = True
nemo_ok = True

# =============================================================================
# SECTION 1: NEMO/EMOPortraits Tests (run first)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 1: Nemo/EMOPortraits Tests")
print("=" * 60)

# Test 1.1: Check if nemo directory exists and has content
print("\n[1.1] Checking nemo directory:")
nemo_path = Path('nemo')
if nemo_path.exists():
    nemo_files = list(nemo_path.glob('*.py'))
    if len(nemo_files) > 0:
        print(f"  ✓ nemo directory exists with {len(nemo_files)} Python files")
    else:
        print("  ✗ nemo directory exists but appears empty")
        print("    Run: git submodule update --init --recursive")
        nemo_ok = False
else:
    print("  ✗ nemo directory not found")
    print("    Run: git submodule update --init --recursive")
    nemo_ok = False

# Test 1.2: Check essential nemo files
if nemo_ok:
    print("\n[1.2] Checking essential nemo files:")
    essential_files = [
        'nemo/main.py',
        'nemo/infer.py',
        'nemo/train.py',
        'nemo/logger.py',
        'nemo/mem.py',
        'nemo/models/stage_1/volumetric_avatar/va.py',
        'nemo/networks/volumetric_avatar/face_parcing.py',  # Note: typo in original repo
        'nemo/losses/perceptual.py',
    ]

    for filepath in essential_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (not found)")
            nemo_ok = False

# Test 1.3: Import core nemo modules
if nemo_ok:
    print("\n[1.3] Importing core nemo modules:")
    nemo_core_modules = [
        ('logger', 'Logging infrastructure'),
        ('mem', 'Memory management'),
    ]

    for module, desc in nemo_core_modules:
        try:
            __import__(module)
            print(f"  ✓ {module} - {desc}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            nemo_ok = False
        except Exception as e:
            print(f"  ⚠ {module}: {e} (non-import error)")

# Test 1.4: Import VolumetricAvatar model
if nemo_ok:
    print("\n[1.4] Importing VolumetricAvatar model:")
    try:
        from models.stage_1.volumetric_avatar.va import VolumetricAvatar
        print("  ✓ VolumetricAvatar imported successfully")
    except ImportError as e:
        print(f"  ✗ VolumetricAvatar import failed: {e}")
        nemo_ok = False
    except Exception as e:
        print(f"  ⚠ VolumetricAvatar: {e} (non-import error, may need config)")

# Test 1.5: Import WarpGenerator
if nemo_ok:
    print("\n[1.5] Importing WarpGenerator:")
    try:
        from models.stage_1.volumetric_avatar.va import WarpGenerator
        print("  ✓ WarpGenerator imported successfully")
    except ImportError as e:
        print(f"  ✗ WarpGenerator import failed: {e}")
        nemo_ok = False
    except Exception as e:
        print(f"  ⚠ WarpGenerator: {e} (non-import error)")

# Test 1.6: Check for model weights/config
if nemo_ok:
    print("\n[1.6] Checking nemo model weights/configs:")
    # Common locations for EMOPortraits weights
    weight_locations = [
        'nemo/checkpoints',
        'nemo/weights',
        'nemo/pretrained',
        'weights',
        'checkpoints/volumetric',
    ]

    found_weights = False
    for loc in weight_locations:
        loc_path = Path(loc)
        if loc_path.exists() and any(loc_path.glob('*.pt')) or any(loc_path.glob('*.pth')) or any(loc_path.glob('*.ckpt')):
            print(f"  ✓ Found weights in: {loc}")
            found_weights = True
            break

    if not found_weights:
        print("  ⚠ No pre-trained weights found (may need to download)")
        print("    Check README for weight download instructions")

# Test 1.7: Import additional nemo components
if nemo_ok:
    print("\n[1.7] Importing additional nemo components:")
    additional_modules = [
        ('networks.volumetric_avatar.face_parcing', 'Face parsing/segmentation'),
        ('losses.perceptual', 'Perceptual loss'),
    ]

    for module, desc in additional_modules:
        try:
            __import__(module)
            print(f"  ✓ {module} - {desc}")
        except ImportError as e:
            print(f"  ⚠ {module}: {e} (optional)")
        except Exception as e:
            print(f"  ⚠ {module}: {e} (non-import error)")

# Test 1.8: Check L2CS-Net dependency
print("\n[1.8] Checking L2CS-Net dependency:")
l2cs_path = Path('L2CS-Net')
if l2cs_path.exists() and list(l2cs_path.glob('*.py')):
    print("  ✓ L2CS-Net submodule present")
    sys.path.insert(0, 'L2CS-Net')
    try:
        # Try importing L2CS if available
        from l2cs import Pipeline
        print("  ✓ L2CS Pipeline importable")
    except ImportError:
        print("  ⚠ L2CS Pipeline not importable (may need setup)")
    except Exception as e:
        print(f"  ⚠ L2CS: {e}")
else:
    print("  ⚠ L2CS-Net not found or empty")
    print("    Run: git submodule update --init --recursive")

# Nemo summary
print("\n" + "-" * 60)
if nemo_ok:
    print("✓ Nemo/EMOPortraits setup looks good!")
else:
    print("✗ Nemo/EMOPortraits has issues - fix before proceeding to VASA")
    all_tests_passed = False
print("-" * 60)

# =============================================================================
# SECTION 2: VASA Tests (only run if nemo is OK)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: VASA Setup Tests")
print("=" * 60)

if not nemo_ok:
    print("\n⚠ Skipping VASA tests - fix nemo/EMOPortraits first")
else:
    # Test 2.1: Check config loading
    print("\n[2.1] Loading VASA config:")
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load('vasa_config.yaml')
        print("  ✓ Config loaded successfully")
        print(f"    - Volumetric model: {config.paths.volumetric_model}")
        print(f"    - Volumetric config: {config.paths.volumetric_config}")
        print(f"    - Data dir: {config.paths.data_dir}")
        print(f"    - Video folder: {config.paths.video_folder}")
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        all_tests_passed = False

    # Test 2.2: Check if paths exist
    print("\n[2.2] Checking VASA paths:")
    try:
        paths_to_check = [
            ('Volumetric config', config.paths.volumetric_config),
            ('Volumetric model', config.paths.volumetric_model),
            ('Data directory', config.paths.data_dir),
            ('Video folder', config.paths.video_folder),
            ('Cache directory', config.paths.cache_dir),
            ('Checkpoint directory', config.paths.checkpoint_dir),
        ]

        for name, path in paths_to_check:
            path_obj = Path(path)
            if path_obj.exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: {path} (not found)")
                all_tests_passed = False
    except Exception as e:
        print(f"  ✗ Error checking paths: {e}")
        all_tests_passed = False

    # Test 2.3: Try importing VASA modules
    print("\n[2.3] Importing VASA modules:")
    vasa_modules = [
        'vasa_model',
        'vasa_dataset',
        'vasa_trainer',
        'vasa_scheduler',
        'expression_normalizer',
    ]

    for module in vasa_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_tests_passed = False
        except Exception as e:
            print(f"  ⚠ {module}: {e} (non-import error)")

    # Test 2.4: Try importing nemo modules from VASA context
    print("\n[2.4] Importing nemo modules (VASA integration):")
    nemo_vasa_modules = [
        'logger',
        'mem',
        'models.stage_1.volumetric_avatar.va'
    ]

    for module in nemo_vasa_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_tests_passed = False
        except Exception as e:
            print(f"  ⚠ {module}: {e} (non-import error)")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

if all_tests_passed and nemo_ok:
    print("\n✓ All tests passed! Setup is complete.")
    print("\nYou can now run:")
    print("  python vasa_trainer.py        # Full training")
    print("  python train_overfit.py       # Overfitting test")
    print("  python infer.py               # Inference")
else:
    print("\n✗ Some tests failed. Please fix the issues above.")
    if not nemo_ok:
        print("\nTo fix nemo/EMOPortraits:")
        print("  git submodule update --init --recursive")
    print("\nCheck the README for additional setup instructions.")

print("=" * 60)
