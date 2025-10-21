#!/usr/bin/env python3
"""
Test script for MD5-indexed frame disk cache
"""

import torch
from pathlib import Path
from frame_disk_cache import FrameDiskCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_frame_cache():
    """Test frame cache operations"""

    # Create test cache
    cache_dir = Path("test_cache")
    cache_dir.mkdir(exist_ok=True)

    frame_cache = FrameDiskCache(cache_dir, frame_type='frames')

    # Create test frames [T, C, H, W]
    test_frames = torch.rand(50, 3, 512, 512)

    # Create a dummy video file for MD5 hashing
    test_video_dir = Path("test_cache")
    test_video_dir.mkdir(exist_ok=True)
    test_video = test_video_dir / "test_video.mp4"
    test_video.write_bytes(b"dummy video content for testing")

    test_video = str(test_video)

    logger.info("=" * 80)
    logger.info("Testing Frame Disk Cache")
    logger.info("=" * 80)

    # Test 1: Save frames
    logger.info("\n📝 Test 1: Saving frames...")
    frame_cache.save_frames(
        video_path=test_video,
        window_idx=0,
        frames=test_frames,
        format='png'
    )
    logger.info("✅ Frames saved successfully")

    # Test 2: Check if frames exist
    logger.info("\n🔍 Test 2: Checking frame existence...")
    exists = frame_cache.has_frames(test_video, 0)
    assert exists, "Frames should exist"
    logger.info(f"✅ Frame existence check passed: {exists}")

    # Test 3: Load frames
    logger.info("\n📖 Test 3: Loading frames...")
    loaded_frames = frame_cache.load_frames(
        video_path=test_video,
        window_idx=0,
        as_tensor=True
    )
    assert loaded_frames is not None, "Frames should load"
    assert loaded_frames.shape == test_frames.shape, f"Shape mismatch: {loaded_frames.shape} vs {test_frames.shape}"
    logger.info(f"✅ Frames loaded: {loaded_frames.shape}")

    # Test 4: Verify MD5 hashing
    logger.info("\n🔐 Test 4: Testing MD5 hashing...")
    video_hash = frame_cache.get_video_hash(test_video)
    logger.info(f"✅ Video hash: {video_hash}")

    # Test 5: Get cache stats
    logger.info("\n📊 Test 5: Cache statistics...")
    stats = frame_cache.get_cache_stats()
    logger.info(f"  Total videos: {stats['total_videos']}")
    logger.info(f"  Total windows: {stats['total_windows']}")
    logger.info(f"  Total frames: {stats['total_frames']}")
    logger.info(f"  Total size: {stats['total_size_gb']:.4f} GB")
    logger.info("✅ Statistics retrieved")

    # Test 6: Multiple windows
    logger.info("\n🪟 Test 6: Testing multiple windows...")
    for window_idx in range(1, 3):
        frame_cache.save_frames(test_video, window_idx, test_frames, 'png')
    stats = frame_cache.get_cache_stats()
    assert stats['total_windows'] == 3, "Should have 3 windows"
    logger.info(f"✅ Multiple windows saved: {stats['total_windows']} windows")

    # Test 7: Clean up
    logger.info("\n🧹 Test 7: Cleanup...")
    frame_cache.delete_video(test_video)
    stats = frame_cache.get_cache_stats()
    assert stats['total_videos'] == 0, "Should have 0 videos after cleanup"

    # Remove test video file and cache dir
    Path(test_video).unlink(missing_ok=True)
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    logger.info("✅ Cleanup successful")

    logger.info("\n" + "=" * 80)
    logger.info("✅ All tests passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_frame_cache()
