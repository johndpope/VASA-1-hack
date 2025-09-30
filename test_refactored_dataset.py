#!/usr/bin/env python3
"""Test the refactored vasa_dataset.py with aligned warp extraction."""

import torch
import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Add paths
sys.path.insert(0, '/media/2TB/VASA-1-hack')
sys.path.insert(0, '/media/2TB/VASA-1-hack/nemo')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_warps():
    """Test that the refactored dataset extracts warps correctly."""

    from vasa_dataset import VASAIntegratedDataset
    import yaml
    from omegaconf import OmegaConf

    # Load config
    config_path = '/media/2TB/VASA-1-hack/overfit_config.yaml'
    with open(config_path) as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # Load EMO model first
    from create_video_face_swap import load_volumetric_model
    emo_model = load_volumetric_model()

    # Create dataset with hardcoded values
    logger.info("Creating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder='/media/2TB/VASA-1-hack/data/overfit_video',  # Hardcoded path
        emo_model=emo_model,
        window_size=50,
        stride=25,
        context_size=10,
        frame_size=(512, 512),
        max_videos=1,  # Just test with one video
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_single_bucket=False
    )

    # Get a sample
    logger.info("Getting sample from dataset...")
    sample = dataset[0]

    # Check the motion data structure
    logger.info("Checking motion data structure...")
    motion_data = sample['motion']

    expected_keys = [
        'theta', 'scale', 'rotation', 'translation', 'expression_embed',
        'uv_warps', 'target_pose_embed', 'xy_warps', 'rigid_warps',
        'identity_info'
    ]

    missing_keys = [k for k in expected_keys if k not in motion_data]
    if missing_keys:
        logger.error(f"Missing keys in motion data: {missing_keys}")
        return False

    logger.info("✅ All expected keys present in motion data")

    # Check identity_info
    identity_info = motion_data.get('identity_info')
    if identity_info is None:
        logger.error("identity_info is None!")
        return False

    identity_keys = ['idt_embed', 'embed_dict', 'canonical_volume', 'source_theta', 'source_mask']
    missing_identity = [k for k in identity_keys if k not in identity_info]
    if missing_identity:
        logger.error(f"Missing keys in identity_info: {missing_identity}")
        return False

    logger.info("✅ Identity info structure correct")

    # Check shapes
    B, T = motion_data['theta'].shape[:2]
    logger.info(f"Batch size: {B}, Sequence length: {T}")

    shape_checks = {
        'theta': (B, T, 3, 4),
        'uv_warps': (B, T, 16, 64, 64, 3),
        'target_pose_embed': (B, T, 512),
        'xy_warps': (B, T, 16, 64, 64, 3),
        'rigid_warps': (B, T, 16, 64, 64, 3),
    }

    for key, expected_shape in shape_checks.items():
        actual_shape = motion_data[key].shape
        if actual_shape != expected_shape:
            logger.error(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
            return False
        logger.info(f"✅ {key}: {actual_shape}")

    # Check canonical volume
    canonical_shape = identity_info['canonical_volume'].shape
    if len(canonical_shape) != 5 or canonical_shape[0] != B:
        logger.error(f"Canonical volume wrong shape: {canonical_shape}")
        return False
    logger.info(f"✅ Canonical volume: {canonical_shape}")

    return True

def test_decoder_integration():
    """Test the new decode_frame_with_warps method."""

    from vasa_model import VASAModel
    import yaml
    from omegaconf import OmegaConf

    # Load config
    config_path = '/media/2TB/VASA-1-hack/overfit_config.yaml'
    with open(config_path) as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # Load volumetric model
    logger.info("Loading volumetric model...")
    from create_video_face_swap import load_volumetric_model
    volumetric_model = load_volumetric_model()

    # Create VASA model
    logger.info("Creating VASA model...")
    model = VASAModel(config, volumetric_model)
    model.eval()

    # Create dummy data
    B = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    identity_info = {
        'canonical_volume': torch.randn(B, 96, 16, 64, 64, device=device),
        'embed_dict': {'key': torch.randn(B, 512, device=device)},
        'idt_embed': torch.randn(B, 512, device=device)
    }

    warp_data = {
        'uv_warp': torch.randn(B, 16, 64, 64, 3, device=device),
        'theta': torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1),
        'target_pose_embed': torch.randn(B, 512, device=device)
    }

    # Test decoder
    logger.info("Testing decode_frame_with_warps...")
    try:
        with torch.no_grad():
            frame = model.decode_frame_with_warps(identity_info, warp_data)

        if frame.shape != (B, 3, 512, 512):
            logger.error(f"Wrong output shape: {frame.shape}")
            return False

        logger.info(f"✅ Generated frame shape: {frame.shape}")

        # Save test image
        img = frame[0].cpu().permute(1, 2, 0).numpy()
        img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
        Image.fromarray(img).save('test_decoder_output.png')
        logger.info("✅ Saved test output to test_decoder_output.png")

        return True

    except Exception as e:
        logger.error(f"Decoder test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("=" * 60)
    logger.info("Testing Refactored Implementation")
    logger.info("=" * 60)

    # Test 1: Dataset warps
    logger.info("\nTest 1: Dataset Warp Extraction")
    logger.info("-" * 40)
    dataset_ok = test_dataset_warps()

    # Test 2: Decoder integration
    logger.info("\nTest 2: Decoder Integration")
    logger.info("-" * 40)
    decoder_ok = test_decoder_integration()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  Dataset warps: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    logger.info(f"  Decoder integration: {'✅ PASS' if decoder_ok else '❌ FAIL'}")

    if dataset_ok and decoder_ok:
        logger.info("\n✅ All tests passed!")
        return 0
    else:
        logger.info("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())