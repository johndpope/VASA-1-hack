#!/usr/bin/env python3
"""
Test Chunked Window Cache with Varying Window Sizes
====================================================
"""

import torch
import numpy as np
from pathlib import Path
from window_cache import WindowCache
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import traceback

def create_dummy_video_data(total_frames: int = 1500) -> dict:
    """Create dummy video data for testing."""
    return {
        'frames': torch.randn(total_frames, 3, 512, 512),
        'theta': torch.randn(total_frames, 3, 4),
        'scale': torch.randn(total_frames, 3),
        'rotation': torch.randn(total_frames, 3),
        'translation': torch.randn(total_frames, 3),
        'expression_embed': torch.randn(total_frames, 128),
        'audio_features': torch.randn(total_frames, 768),
        'gaze': torch.randn(total_frames, 2),
        'emotion': torch.randn(total_frames, 2),
        'head_distance': torch.randn(total_frames, 1),
        'speed_bucket': torch.randint(0, 9, (total_frames, 1)),
        'lips': torch.randn(total_frames, 20, 3),
        'right_eye': torch.randn(total_frames, 8, 3),
        'left_eye': torch.randn(total_frames, 7, 3),
        'jaw': torch.randn(total_frames, 10, 3),
        'nose': torch.randn(total_frames, 4, 3),
        'blink_state': torch.randn(total_frames, 3)
    }

def test_chunking():
    """Test basic chunking functionality."""
    logger.info("\n=== Testing Basic Chunking ===")
    
    cache_dir = Path("test_cache")
    cache = WindowCache(cache_dir, chunk_size=500, overlap_size=50)
    
    # Create test data
    total_frames = 1500
    video_path = "test_video.mp4"
    video_data = create_dummy_video_data(total_frames)
    metadata = {
        'total_frames': total_frames,
        'fps': 30.0,
        'window_params': {'test': True}
    }
    
    # Test chunking
    t1 = time.time()
    num_chunks = cache.generate_chunks(video_path, video_data, metadata)
    t2 = time.time()
    
    logger.info(f"Created {num_chunks} chunks in {t2-t1:.2f}s")
    logger.info(f"Expected chunks: {(total_frames - 50) // (500 - 50) + 1}")
    
    # Verify chunks exist
    for i in range(num_chunks):
        chunk_path = cache._get_cache_path(video_path, i)
        assert chunk_path.exists(), f"Chunk {i} not found"
        
    # Load a chunk
    chunk = cache._load_chunk(video_path, 0)
    assert chunk is not None, "Failed to load chunk 0"
    logger.info(f"Chunk 0 shape (theta): {chunk.get('theta', torch.empty(0)).shape}")
    
    # Clean up
    cache.clear_cache(video_path)
    logger.info("✅ Basic chunking test passed")
    
    return cache_dir

def test_varying_window_sizes(cache_dir: Path):
    """Test generating windows with different sizes."""
    logger.info("\n=== Testing Varying Window Sizes ===")
    
    cache = WindowCache(cache_dir, chunk_size=500, overlap_size=50)
    
    # Create and cache test data
    total_frames = 100  # Smaller for this test
    video_path = "test_video_small.mp4"
    video_data = create_dummy_video_data(total_frames)
    metadata = {
        'total_frames': total_frames,
        'fps': 30.0,
        'window_params': {}
    }
    
    # Generate chunks
    num_chunks = cache.generate_chunks(video_path, video_data, metadata)
    logger.info(f"Created {num_chunks} chunks for {total_frames} frames")
    
    # Test different window sizes
    test_configs = [
        (20, 10),  # Small window, small stride
        (30, 15),  # Medium window
        (50, 25),  # Original size
        (40, 20),  # Custom size
    ]
    
    for window_size, stride in test_configs:
        logger.info(f"\nTesting window_size={window_size}, stride={stride}")
        
        t1 = time.time()
        windows = cache.generate_windows(video_path, window_size, stride, context_size=10)
        t2 = time.time()
        
        expected_windows = max(0, (total_frames - window_size) // stride + 1)
        logger.info(f"  Generated {len(windows)} windows in {t2-t1:.2f}s")
        logger.info(f"  Expected: {expected_windows} windows")
        
        if len(windows) > 0:
            # Check first window
            first_window = windows[0]
            for key in ['theta', 'scale', 'rotation']:
                if key in first_window:
                    actual_size = first_window[key].shape[0]
                    logger.info(f"  {key} shape: {first_window[key].shape}")
                    assert actual_size == window_size, f"Window size mismatch for {key}: {actual_size} != {window_size}"
            
            # Check metadata
            assert first_window['metadata']['start_frame'] == 0
            assert first_window['metadata']['window_idx'] == 0
            assert first_window['metadata']['total_windows'] == len(windows)
    
    # Clean up
    cache.clear_cache(video_path)
    logger.info("✅ Varying window sizes test passed")

def test_cross_chunk_windows(cache_dir: Path):
    """Test windows that span multiple chunks."""
    logger.info("\n=== Testing Cross-Chunk Windows ===")
    
    cache = WindowCache(cache_dir, chunk_size=100, overlap_size=20)
    
    # Create test data
    total_frames = 250
    video_path = "test_video_cross.mp4"
    video_data = create_dummy_video_data(total_frames)
    metadata = {
        'total_frames': total_frames,
        'fps': 30.0,
        'window_params': {}
    }
    
    # Generate chunks
    num_chunks = cache.generate_chunks(video_path, video_data, metadata)
    logger.info(f"Created {num_chunks} chunks with size=100, overlap=20")
    
    # Generate windows that will span chunks
    window_size = 60  # Will span across chunks
    stride = 40
    
    windows = cache.generate_windows(video_path, window_size, stride)
    logger.info(f"Generated {len(windows)} windows of size {window_size}")
    
    # Check window at chunk boundary
    if len(windows) > 2:
        # Window that should span chunks 0 and 1
        boundary_window = windows[2]  # Start around frame 80
        logger.info(f"Boundary window start: {boundary_window['metadata']['start_frame']}")
        
        # Verify data continuity
        theta = boundary_window.get('theta')
        if theta is not None:
            assert theta.shape[0] == window_size, f"Cross-chunk window size incorrect: {theta.shape[0]}"
            logger.info(f"Cross-chunk window theta shape: {theta.shape}")
    
    # Clean up
    cache.clear_cache(video_path)
    logger.info("✅ Cross-chunk windows test passed")

def test_memory_efficiency(cache_dir: Path):
    """Test memory cache management."""
    logger.info("\n=== Testing Memory Efficiency ===")
    
    # Small memory cache to test eviction
    cache = WindowCache(cache_dir, chunk_size=100, overlap_size=20, max_memory_cache=2)
    
    # Create test data
    total_frames = 300
    video_path = "test_video_memory.mp4"
    video_data = create_dummy_video_data(total_frames)
    metadata = {
        'total_frames': total_frames,
        'fps': 30.0,
        'window_params': {}
    }
    
    # Generate chunks
    num_chunks = cache.generate_chunks(video_path, video_data, metadata)
    logger.info(f"Created {num_chunks} chunks")
    
    # Load multiple chunks to test eviction
    for i in range(num_chunks):
        chunk = cache._load_chunk(video_path, i)
        logger.info(f"Loaded chunk {i}, memory cache size: {len(cache.chunk_cache)}")
        assert len(cache.chunk_cache) <= cache.max_memory_cache, "Memory cache exceeded limit"
    
    # Check cache stats
    stats = cache.get_cache_stats()
    logger.info(f"Cache stats: {stats}")
    
    # Clean up
    cache.clear_cache()
    logger.info("✅ Memory efficiency test passed")

def test_cache_persistence(cache_dir: Path):
    """Test that cache persists across sessions."""
    logger.info("\n=== Testing Cache Persistence ===")
    
    video_path = "test_video_persist.mp4"
    
    # First session - create cache
    cache1 = WindowCache(cache_dir, chunk_size=200, overlap_size=30)
    video_data = create_dummy_video_data(400)
    metadata = {'total_frames': 400, 'fps': 30.0, 'window_params': {}}
    
    num_chunks = cache1.generate_chunks(video_path, video_data, metadata)
    logger.info(f"Session 1: Created {num_chunks} chunks")
    
    # Second session - load existing cache
    cache2 = WindowCache(cache_dir, chunk_size=200, overlap_size=30)
    
    # Check if cache exists
    assert cache2.has_cache(video_path), "Cache not found in new session"
    
    # Load metadata
    loaded_metadata = cache2.load_metadata(video_path)
    assert loaded_metadata is not None, "Failed to load metadata"
    assert loaded_metadata['total_frames'] == 400, "Metadata mismatch"
    
    # Generate windows from cached data
    windows = cache2.generate_windows(video_path, window_size=50, stride=25)
    logger.info(f"Session 2: Generated {len(windows)} windows from cache")
    
    # Clean up
    cache2.clear_cache()
    logger.info("✅ Cache persistence test passed")

def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("CHUNKED WINDOW CACHE TESTS")
    logger.info("="*70)
    
    try:
        # Create test cache directory
        cache_dir = Path("test_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Run tests
        test_chunking()
        test_varying_window_sizes(cache_dir)
        test_cross_chunk_windows(cache_dir)
        test_memory_efficiency(cache_dir)
        test_cache_persistence(cache_dir)
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED ✅")
        logger.info("="*70)
        
        # Clean up test directory
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Cleaned up test cache directory")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()