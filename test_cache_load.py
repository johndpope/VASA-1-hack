#!/usr/bin/env python3
"""Test loading from the single-bucket cache."""

from single_bucket_cache import SingleBucketCache
from pathlib import Path
import time

cache_path = Path("cache_overfit")
cache = SingleBucketCache(cache_dir=cache_path, cache_name="all_windows_cache.h5")

# Get cache info
info = cache.get_cache_info()
print(f"Cache info:")
print(f"  Number of windows: {info['num_windows']}")
print(f"  File size: {info['file_size_mb']:.2f} MB")
print(f"  Compression ratio: ~{2704/info['file_size_mb']:.1f}:1")
print()

# Test loading a window
print("Testing window loading...")
start = time.time()
window_data = cache.load_window(0)
load_time = time.time() - start

if window_data:
    print(f"✓ Successfully loaded window 0 in {load_time:.3f}s")
    print(f"  Keys in window: {list(window_data.keys())}")
    print(f"  Frames shape: {window_data['frames'].shape}")
    print(f"  Audio features shape: {window_data['audio_features'].shape}")
else:
    print("✗ Failed to load window")

# Test loading multiple windows
print("\nBenchmarking random access...")
import random
indices = random.sample(range(18), 5)
start = time.time()
for idx in indices:
    window = cache.load_window(idx)
end = time.time()
print(f"✓ Loaded 5 random windows in {end-start:.3f}s ({(end-start)/5:.3f}s per window)")