#!/usr/bin/env python3
"""
Test script to verify per-video cache integration with VASAIntegratedDataset.

This verifies that:
1. Dataset auto-detects per-video cache
2. Windows can be loaded correctly
3. Frames and emo_frames are loaded from disk
4. All expected data fields are present
"""

import torch
from pathlib import Path
import logging
from vasa_dataset import VASAIntegratedDataset
from per_video_cache import PerVideoCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_per_video_cache_detection():
    """Test that dataset correctly detects per-video cache."""
    logger.info("="*80)
    logger.info("TEST 1: Per-Video Cache Detection")
    logger.info("="*80)

    cache_dir = Path('cache_per_video')

    # Check if per-video cache exists
    if not (cache_dir / 'cache_index.json').exists():
        logger.error(f"❌ Per-video cache not found at {cache_dir}")
        logger.error("   Run migration first: python convert_to_per_video_cache.py")
        return False

    # Get cache stats
    cache = PerVideoCache(cache_dir)
    stats = cache.get_cache_stats()

    logger.info(f"✅ Per-video cache found:")
    logger.info(f"   Videos: {stats['total_videos']}")
    logger.info(f"   Windows: {stats['total_windows']}")
    logger.info(f"   Metadata size: {stats['total_metadata_size_mb']:.2f} MB")

    return True


def test_dataset_initialization():
    """Test that dataset initializes with per-video cache."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Dataset Initialization")
    logger.info("="*80)

    # Mock EMO model (not needed for cache loading test)
    class MockEmoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def to(self, device):
            return self

        def parameters(self):
            return [self.dummy]

    emo_model = MockEmoModel().cuda()

    try:
        dataset = VASAIntegratedDataset(
            video_folder='s1',
            emo_model=emo_model,
            window_size=50,
            stride=25,
            max_videos=100,
            cache_dir='cache_per_video',
            use_single_bucket=False,  # Prefer per-video cache
            generate_emo_frames=False,  # Don't generate - use cached
            cache_frames_to_disk=False,  # Frames in per-video cache
            cache_emo_frames_to_disk=False,  # EMO frames in per-video cache
        )

        logger.info(f"✅ Dataset initialized successfully")
        logger.info(f"   Cache type: {dataset.cache_type}")
        logger.info(f"   Total windows: {len(dataset)}")

        if dataset.cache_type != 'per_video':
            logger.error(f"❌ Expected cache_type='per_video', got '{dataset.cache_type}'")
            return False, None

        return True, dataset

    except Exception as e:
        logger.error(f"❌ Failed to initialize dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def test_window_loading(dataset):
    """Test loading a window from per-video cache."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Window Loading")
    logger.info("="*80)

    if dataset is None:
        logger.error("❌ Dataset not initialized, skipping window loading test")
        return False

    try:
        # Load first window
        logger.info(f"Loading window 0 from dataset...")
        window_data = dataset[0]

        if window_data is None:
            logger.error("❌ Window data is None")
            return False

        logger.info(f"✅ Window loaded successfully")

        # Check expected fields
        required_fields = [
            'frames', 'emo_frames', 'theta', 'expression_embed',
            'audio_features', 'metadata'
        ]

        missing_fields = []
        for field in required_fields:
            if field not in window_data:
                missing_fields.append(field)
                logger.warning(f"⚠️  Missing field: {field}")
            else:
                if isinstance(window_data[field], torch.Tensor):
                    logger.info(f"   {field}: {window_data[field].shape}")
                elif isinstance(window_data[field], dict):
                    logger.info(f"   {field}: dict with {len(window_data[field])} keys")
                else:
                    logger.info(f"   {field}: {type(window_data[field])}")

        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return False

        # Check frames shape
        if 'frames' in window_data:
            frames_shape = window_data['frames'].shape
            logger.info(f"\n✅ Frames loaded from disk:")
            logger.info(f"   Shape: {frames_shape}")
            logger.info(f"   Expected: [T, C, H, W] = [50, 3, 512, 512]")

            if frames_shape != torch.Size([50, 3, 512, 512]):
                logger.warning(f"⚠️  Unexpected frames shape: {frames_shape}")

        # Check emo_frames shape
        if 'emo_frames' in window_data:
            emo_frames_shape = window_data['emo_frames'].shape
            logger.info(f"\n✅ EMO frames loaded from disk:")
            logger.info(f"   Shape: {emo_frames_shape}")
            logger.info(f"   Expected: [T, C, H, W] = [50, 3, 512, 512]")

            if emo_frames_shape != torch.Size([50, 3, 512, 512]):
                logger.warning(f"⚠️  Unexpected emo_frames shape: {emo_frames_shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load window: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_multiple_windows(dataset, num_windows=5):
    """Test loading multiple windows."""
    logger.info("\n" + "="*80)
    logger.info(f"TEST 4: Loading Multiple Windows (n={num_windows})")
    logger.info("="*80)

    if dataset is None:
        logger.error("❌ Dataset not initialized, skipping multiple window test")
        return False

    try:
        for i in range(min(num_windows, len(dataset))):
            logger.info(f"\nLoading window {i}...")
            window_data = dataset[i]

            if window_data is None:
                logger.error(f"❌ Window {i} is None")
                return False

            video_path = window_data['metadata'].get('video_path', 'unknown')
            window_idx = window_data['metadata'].get('window_idx', -1)
            logger.info(f"✅ Window {i}: {Path(video_path).name}, window_idx={window_idx}")

        logger.info(f"\n✅ Successfully loaded {num_windows} windows")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load multiple windows: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("PER-VIDEO CACHE INTEGRATION TESTS")
    logger.info("="*80)

    results = {}

    # Test 1: Cache detection
    results['cache_detection'] = test_per_video_cache_detection()

    if not results['cache_detection']:
        logger.error("\n❌ TESTS FAILED: Per-video cache not found")
        return

    # Test 2: Dataset initialization
    results['dataset_init'], dataset = test_dataset_initialization()

    if not results['dataset_init']:
        logger.error("\n❌ TESTS FAILED: Dataset initialization failed")
        return

    # Test 3: Window loading
    results['window_loading'] = test_window_loading(dataset)

    # Test 4: Multiple windows
    results['multiple_windows'] = test_multiple_windows(dataset, num_windows=5)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    if all_passed:
        logger.info("\n✅ ALL TESTS PASSED")
        logger.info("   Per-video cache integration is working correctly!")
        logger.info("   You can now use cache_per_video/ for training.")
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        logger.error("   Review the errors above before using per-video cache for training.")


if __name__ == "__main__":
    main()
