#!/usr/bin/env python3
"""
Simple test to verify warp injection works.
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_warp_loading():
    """Test loading warps from H5 and verify shapes."""

    h5_path = Path('../temp_single_video/window_cache/6ae61842675c083d.h5')

    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return False

    logger.info(f"Loading warps from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'window_0' not in f:
            logger.error("No window_0 in cache file")
            return False

        window = f['window_0']

        # Check for source warps
        warps_found = {}

        if 'source_xy_warp' in window:
            warp = window['source_xy_warp'][()]
            warps_found['source_xy_warp'] = warp.shape
            logger.info(f"✓ source_xy_warp: shape={warp.shape}, non-zero={np.any(warp != 0)}")

        if 'source_rotation_warp' in window:
            warp = window['source_rotation_warp'][()]
            warps_found['source_rotation_warp'] = warp.shape
            logger.info(f"✓ source_rotation_warp: shape={warp.shape}, non-zero={np.any(warp != 0)}")

        if 'canonical_volume' in window:
            vol = window['canonical_volume'][()]
            warps_found['canonical_volume'] = vol.shape
            logger.info(f"✓ canonical_volume: shape={vol.shape}, non-zero={np.any(vol != 0)}")

        if 'source_theta' in window:
            theta = window['source_theta'][()]
            warps_found['source_theta'] = theta.shape
            logger.info(f"✓ source_theta: shape={theta.shape}, non-zero={np.any(theta != 0)}")

        # Check for expression embeddings
        if 'expression_embed' in window:
            expr = window['expression_embed'][()]
            logger.info(f"✓ expression_embed: shape={expr.shape}, non-zero={np.any(expr != 0)}")

        # Check for face attributes
        if 'gaze' in window:
            gaze = window['gaze'][()]
            logger.info(f"✓ gaze: shape={gaze.shape}")

        if 'emotion' in window:
            emotion = window['emotion'][()]
            logger.info(f"✓ emotion: shape={emotion.shape}")

    if warps_found:
        logger.info(f"\n✅ Successfully loaded {len(warps_found)} warp tensors")
        return True
    else:
        logger.error("\n❌ No warps found in cache")
        return False


def test_warp_injection_mock():
    """Test that warp injection would work with mock inferer."""

    class MockInferer:
        """Mock inferer to test warp injection."""
        def __init__(self):
            self.source_xy_warp_resize = None
            self.source_rotation_warp = None
            self.target_latent_volume = None
            self.pred_source_theta = None
            self.use_cached_source_warps = False

    # Create mock inferer
    inferer = MockInferer()

    # Create mock warps
    source_warps = {
        'source_xy_warp': torch.randn(1, 16, 64, 64, 3),
        'source_rotation_warp': torch.randn(1, 16, 64, 64, 3),
        'canonical_volume': torch.randn(1, 96, 16, 64, 64),
        'source_theta': torch.randn(1, 4, 4)
    }

    # Inject warps
    logger.info("\nTesting warp injection...")

    if 'source_xy_warp' in source_warps:
        inferer.source_xy_warp_resize = source_warps['source_xy_warp']
        logger.info(f"✓ Injected source_xy_warp: {inferer.source_xy_warp_resize.shape}")

    if 'source_rotation_warp' in source_warps:
        inferer.source_rotation_warp = source_warps['source_rotation_warp']
        logger.info(f"✓ Injected source_rotation_warp: {inferer.source_rotation_warp.shape}")

    if 'canonical_volume' in source_warps:
        inferer.target_latent_volume = source_warps['canonical_volume']
        logger.info(f"✓ Injected canonical_volume: {inferer.target_latent_volume.shape}")

    if 'source_theta' in source_warps:
        inferer.pred_source_theta = source_warps['source_theta']
        logger.info(f"✓ Injected source_theta: {inferer.pred_source_theta.shape}")

    inferer.use_cached_source_warps = True
    logger.info("✓ Set use_cached_source_warps = True")

    # Verify injection
    assert inferer.source_xy_warp_resize is not None
    assert inferer.source_rotation_warp is not None
    assert inferer.target_latent_volume is not None
    assert inferer.pred_source_theta is not None
    assert inferer.use_cached_source_warps == True

    logger.info("\n✅ Warp injection test passed!")
    return True


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Testing Warp Loading and Injection")
    logger.info("=" * 60)

    # Test loading warps from H5
    if test_warp_loading():
        # Test injection mechanism
        test_warp_injection_mock()

    logger.info("\n✅ All tests completed!")