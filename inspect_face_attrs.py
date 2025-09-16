#!/usr/bin/env python3
"""Inspect face_attrs cache files to understand their contents."""

import h5py
from pathlib import Path
import numpy as np

def inspect_face_attrs(cache_dir="cache_overfit"):
    cache_dir = Path(cache_dir)

    # Find all face_attrs files
    face_attrs_files = list(cache_dir.glob("face_attrs_*.h5"))

    if not face_attrs_files:
        print("No face_attrs files found")
        return

    print(f"Found {len(face_attrs_files)} face_attrs cache files")
    print("=" * 60)

    # Inspect the first file to understand structure
    sample_file = face_attrs_files[0]
    print(f"\nInspecting: {sample_file.name}")
    print(f"File size: {sample_file.stat().st_size / (1024*1024):.2f} MB")

    with h5py.File(sample_file, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")

        # Check if it's storing video-level data
        if 'video_path' in f.attrs:
            print(f"Video: {f.attrs['video_path']}")

        # Look at structure
        for key in list(f.keys())[:5]:  # First 5 keys
            item = f[key]
            if isinstance(item, h5py.Group):
                print(f"\n{key}/ (group):")
                for subkey in list(item.keys())[:10]:
                    subitem = item[subkey]
                    if isinstance(subitem, h5py.Dataset):
                        print(f"  {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
            elif isinstance(item, h5py.Dataset):
                print(f"\n{key}: shape={item.shape}, dtype={item.dtype}")

        # Check what face attributes are stored
        print("\n" + "=" * 60)
        print("Face attributes stored (from first frame group):")

        # Try to find a frame group
        frame_keys = [k for k in f.keys() if k.startswith('frame_')]
        if frame_keys:
            first_frame = f[frame_keys[0]]
            if isinstance(first_frame, h5py.Group):
                attrs_found = list(first_frame.keys())
                print(f"Attributes: {attrs_found}")

                # Show details of each attribute
                print("\nAttribute details:")
                for attr in attrs_found:
                    if attr in first_frame:
                        data = first_frame[attr]
                        if isinstance(data, h5py.Dataset):
                            print(f"  {attr:20s}: shape={data.shape}, dtype={data.dtype}")
                            # Sample first value
                            if data.shape[0] > 0:
                                sample = data[0] if len(data.shape) == 1 else data[0, :5]
                                print(f"    Sample: {sample}")

if __name__ == "__main__":
    inspect_face_attrs()