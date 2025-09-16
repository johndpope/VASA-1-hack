#!/usr/bin/env python3
"""Test script to verify per-frame warp extraction in VASAIntegratedDataset"""

import torch
import logging
from vasa_dataset import VASAIntegratedDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_warp_extraction():
    """Test that per-frame warps are properly extracted"""

    # Load EMO model first
    import sys
    sys.path.append('nemo')
    from infer import InferenceWrapper

    project_dir = 'nemo'
    emo_model = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        model_file_name='328_model.pth',
        project_dir=project_dir,
        folder='logs',
        args_overwrite={'l1_vol_rgb': 0}
    )

    # Create dataset with small window for testing
    dataset = VASAIntegratedDataset(
        video_folder='vasa_train_data/',
        emo_model=emo_model,
        window_size=4,
        sequence_length=4,
        cache_dir='cache_single_bucket/',
        use_single_bucket=True,
        max_videos=1  # Just test with one video
    )

    logger.info(f"Dataset created with {len(dataset)} windows")

    # Get a sample
    sample = dataset[0]

    # Check if warps are present and have correct shapes
    warp_keys = ['xy_warps', 'rigid_warps', 'uv_warps', 'canonical_volume', 'source_theta_warp']

    logger.info("\n=== Checking Per-Frame Warps ===")
    for key in warp_keys:
        if key in sample:
            shape = sample[key].shape
            logger.info(f"{key}: shape={shape}, dtype={sample[key].dtype}")

            # Check if it's not just zeros
            is_zero = torch.allclose(sample[key], torch.zeros_like(sample[key]))
            if is_zero:
                logger.warning(f"  -> {key} is all zeros (may indicate warps couldn't be extracted)")
            else:
                logger.info(f"  -> {key} contains non-zero values ✓")

            # Expected shapes (assuming sequence_length=4)
            expected_shapes = {
                'xy_warps': (4, 16, 64, 64, 3),
                'rigid_warps': (4, 16, 64, 64, 3),
                'uv_warps': (4, 16, 64, 64, 3),
                'canonical_volume': (4, 96, 16, 64, 64),
                'source_theta_warp': (4, 3, 4)
            }

            if key in expected_shapes:
                expected = expected_shapes[key]
                if shape == expected:
                    logger.info(f"  -> Shape matches expected {expected} ✓")
                else:
                    logger.error(f"  -> Shape mismatch! Expected {expected}, got {shape}")
        else:
            logger.error(f"{key} not found in sample!")

    # Check other features for comparison
    logger.info("\n=== Other Features for Comparison ===")
    other_keys = ['frames', 'theta', 'expression_embed', 'audio_features']
    for key in other_keys:
        if key in sample:
            logger.info(f"{key}: shape={sample[key].shape}")

    logger.info("\n=== Test Complete ===")
    return sample

if __name__ == "__main__":
    try:
        sample = test_warp_extraction()
        logger.info("✓ Warp extraction test completed successfully")
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()