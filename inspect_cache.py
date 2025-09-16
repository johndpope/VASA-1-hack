#!/usr/bin/env python3
"""Inspect the single-bucket cache file to understand its contents and size."""

import h5py
from pathlib import Path
import numpy as np

def inspect_cache(cache_path):
    cache_path = Path(cache_path)

    if not cache_path.exists():
        print(f"Cache file not found: {cache_path}")
        return

    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Cache file: {cache_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print("=" * 60)

    with h5py.File(cache_path, 'r') as f:
        # Check metadata
        if 'metadata' in f.attrs:
            metadata = f.attrs['metadata']
            print(f"Metadata: {metadata}")
            print("=" * 60)

        # List all groups
        print(f"Number of windows: {len(f.keys())}")
        print("=" * 60)

        # Inspect first window
        if len(f.keys()) > 0:
            first_window = list(f.keys())[0]
            print(f"\nInspecting window '{first_window}':")
            window_group = f[first_window]

            total_size = 0
            for key in window_group.keys():
                item = window_group[key]
                if isinstance(item, h5py.Group):
                    # Handle nested groups
                    print(f"  {key:30s} | (nested group)")
                    for subkey in item.keys():
                        dataset = item[subkey]
                        shape = dataset.shape
                        dtype = dataset.dtype
                        size_mb = dataset.nbytes / (1024 * 1024)
                        total_size += size_mb
                        print(f"    {subkey:28s} | Shape: {str(shape):20s} | Type: {str(dtype):10s} | Size: {size_mb:8.2f} MB")
                else:
                    # Handle datasets
                    dataset = item
                    shape = dataset.shape
                    dtype = dataset.dtype
                    size_mb = dataset.nbytes / (1024 * 1024)
                    total_size += size_mb
                    print(f"  {key:30s} | Shape: {str(shape):20s} | Type: {str(dtype):10s} | Size: {size_mb:8.2f} MB")

            print(f"\nTotal size per window: {total_size:.2f} MB")
            print(f"Expected total for {len(f.keys())} windows: {total_size * len(f.keys()):.2f} MB")

            # Check compression
            for key in window_group.keys():
                dataset = window_group[key]
                compression = dataset.compression
                if compression:
                    print(f"  {key}: compression={compression}")

if __name__ == "__main__":
    inspect_cache("cache_overfit/all_windows_cache.h5")