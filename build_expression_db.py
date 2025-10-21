#!/usr/bin/env python3
"""
Build expression embedding database for cosine similarity loss.
Extracts expression embeddings every 5th frame from all videos.
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import gc

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_dataset import VASAIntegratedDataset
from logger import logger

logging.basicConfig(level=logging.INFO)


def build_from_single_bucket_cache(
    cache_path: str,
    output_path: str,
    frame_stride: int = 5
):
    """
    Build expression database directly from single bucket cache (FAST!).

    Args:
        cache_path: Path to single bucket cache H5 file (e.g., 'cache_single_bucket/all_windows_cache.h5')
        output_path: Where to save the expression database H5 file
        frame_stride: Sample every Nth frame (default: 5, use 1 for all frames)
    """
    logger.info(f"🚀 Building expression database from single bucket cache")
    logger.info(f"Cache: {cache_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Frame stride: {frame_stride} (every {frame_stride}th frame)")

    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    all_embeddings = []

    with h5py.File(cache_path, 'r') as f:
        num_windows = f.attrs.get('num_windows', 0)
        logger.info(f"Found {num_windows} windows in cache")

        for i in tqdm(range(num_windows), desc="Extracting expressions"):
            window_key = f'window_{i}'
            if window_key not in f:
                continue

            window_group = f[window_key]

            # Get expression embeddings
            if 'expression_embed' not in window_group:
                logger.warning(f"No expression_embed in {window_key}")
                continue

            expr_embeds = np.array(window_group['expression_embed'])  # [T, 128]

            # Sample every Nth frame
            sampled_indices = list(range(0, expr_embeds.shape[0], frame_stride))
            for idx in sampled_indices:
                expr_embed = expr_embeds[idx]  # [128]
                if expr_embed.shape == (128,):
                    all_embeddings.append(expr_embed)

    logger.info(f"Collected {len(all_embeddings)} expression embeddings")

    # Save to H5
    save_expression_database(all_embeddings, output_path, frame_stride, num_windows)

    return np.stack(all_embeddings, axis=0)


def save_expression_database(embeddings_list, output_path, frame_stride, num_windows):
    """Save expression embeddings to H5 database."""
    embeddings_array = np.stack(embeddings_list, axis=0)  # [N, 128]

    logger.info(f"Saving {len(embeddings_array)} embeddings to {output_path}...")
    logger.info(f"Database shape: {embeddings_array.shape}")
    logger.info(f"Database size: {embeddings_array.nbytes / 1024**2:.2f} MB")

    with h5py.File(output_path, 'w') as f:
        # Save with compression
        f.create_dataset(
            'expression_embeddings',
            data=embeddings_array,
            compression='gzip',
            compression_opts=4
        )

        # Save metadata
        f.attrs['num_embeddings'] = len(embeddings_array)
        f.attrs['embedding_dim'] = 128
        f.attrs['frame_stride'] = frame_stride
        f.attrs['num_windows'] = num_windows

    logger.info(f"✅ Expression database saved to {output_path}")


def build_expression_database(
    motion_dir: str = "./cache_single_bucket",
    output_path: str = "expression_embeddings.h5",
    frame_stride: int = 5
):
    """
    Build database of expression embeddings from motion_attributes H5 files.

    DEPRECATED: Use build_from_single_bucket_cache() instead for faster builds.

    Args:
        motion_dir: Directory containing motion_attributes H5 files
        output_path: Where to save the H5 database
        frame_stride: Sample every Nth frame (default: 5)
    """

    logger.info(f"🚀 Building expression embedding database")
    logger.info(f"Motion directory: {motion_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Frame stride: {frame_stride} (every {frame_stride}th frame)")

    motion_path = Path(motion_dir)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")

    # Find all motion H5 files
    motion_files = sorted(motion_path.glob("*.h5"))
    logger.info(f"Found {len(motion_files)} motion H5 files")

    # Collect all expression embeddings
    all_embeddings = []
    total_frames = 0

    with h5py.File(output_path, 'w') as f:
        # Process each motion file
        for file_idx, motion_file in enumerate(tqdm(motion_files, desc="Processing motion files")):
            try:
                # Skip if this is the expression_embeddings.h5 file (output file)
                if motion_file.name == Path(output_path).name:
                    logger.debug(f"Skipping output file: {motion_file.name}")
                    continue

                with h5py.File(motion_file, 'r') as motion_h5:
                    # Get all window keys
                    window_keys = [k for k in motion_h5.keys() if k.startswith('window_')]

                    logger.debug(f"File {file_idx}: {motion_file.name}, {len(window_keys)} windows")

                    # Process each window
                    for window_key in window_keys:
                        window_data = motion_h5[window_key]

                        # Get expression embeddings from window
                        if 'expression_embed' not in window_data:
                            logger.warning(f"No expression_embed in {window_key}")
                            continue

                        expr_embeds = np.array(window_data['expression_embed'])  # [T, 128]

                        # Sample every Nth frame from this window
                        num_frames = expr_embeds.shape[0]
                        sampled_indices = list(range(0, num_frames, frame_stride))

                        for idx in sampled_indices:
                            expr_embed = expr_embeds[idx]  # [128]

                            # Validate shape
                            if expr_embed.shape == (128,):
                                all_embeddings.append(expr_embed)
                                total_frames += 1
                            else:
                                logger.warning(f"Invalid shape for {motion_file.name} {window_key} frame {idx}: {expr_embed.shape}")

                # Periodically log progress
                if (file_idx + 1) % 10 == 0:
                    logger.info(f"Processed {file_idx + 1}/{len(motion_files)} files, {total_frames} embeddings")

                    # Clear memory
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing file {motion_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save using helper function
        logger.info(f"Converting {len(all_embeddings)} embeddings to array...")
        save_expression_database(all_embeddings, output_path, frame_stride, len(motion_files))

        logger.info(f"📊 Statistics:")
        logger.info(f"  Files processed: {len(motion_files)}")
        logger.info(f"  Embeddings per file (avg): {len(all_embeddings) / len(motion_files):.1f}")

    return np.stack(all_embeddings, axis=0)


def verify_database(db_path: str):
    """Verify the database can be loaded and has correct structure."""
    logger.info(f"Verifying database: {db_path}")

    with h5py.File(db_path, 'r') as f:
        embeddings = f['expression_embeddings'][:]

        logger.info(f"Database shape: {embeddings.shape}")
        logger.info(f"Metadata:")
        for key, value in f.attrs.items():
            logger.info(f"  {key}: {value}")

        # Test loading a subset
        logger.info("Testing random access...")
        indices = np.random.choice(len(embeddings), size=min(100, len(embeddings)), replace=False)
        sample = embeddings[indices]
        logger.info(f"Sampled {len(sample)} embeddings: shape={sample.shape}")
        logger.info(f"Mean: {sample.mean():.6f}, Std: {sample.std():.6f}")

        # Test GPU transfer
        if torch.cuda.is_available():
            logger.info("Testing GPU transfer...")
            tensor = torch.from_numpy(embeddings).float().cuda()
            logger.info(f"GPU tensor shape: {tensor.shape}, device: {tensor.device}")
            logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    logger.info("✅ Database verification complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build expression embedding database")
    parser.add_argument('--motion-dir', type=str, default='/media/2TB/VASA-1-hack/cache_single_bucket',
                       help='Directory containing motion_attributes H5 files')
    parser.add_argument('--output', type=str, default='expression_embeddings.h5',
                       help='Output H5 file path')
    parser.add_argument('--frame-stride', type=int, default=1,
                       help='Sample every Nth frame (default: 5)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing database instead of building')

    args = parser.parse_args()

    if args.verify:
        verify_database(args.output)
    else:
        embeddings = build_expression_database(
            motion_dir=args.motion_dir,
            output_path=args.output,
            frame_stride=args.frame_stride
        )

        # Auto-verify after building
        verify_database(args.output)
