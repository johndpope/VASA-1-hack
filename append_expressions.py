#!/usr/bin/env python3
"""
Append new expression embeddings to existing database from cache files.
Extracts expressions from cache_per_video and appends them to expression_embeddings.h5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from logger import logger
from expression_db import ExpressionDatabase

def extract_expressions_from_cache(cache_dir: Path, frame_stride: int = 1) -> np.ndarray:
    """
    Extract expression embeddings from cache_per_video directory.

    Args:
        cache_dir: Path to cache_per_video directory
        frame_stride: Sample every Nth frame (1=all frames, 5=every 5th frame)

    Returns:
        expressions: [N, 128] numpy array of expression embeddings
    """
    expressions = []

    # Find all metadata.h5 files
    cache_files = sorted(cache_dir.glob("*/metadata.h5"))

    if not cache_files:
        logger.error(f"No metadata.h5 files found in {cache_dir}")
        return np.array([])

    logger.info(f"Found {len(cache_files)} cache files in {cache_dir}")

    for cache_file in tqdm(cache_files, desc="Extracting expressions"):
        try:
            with h5py.File(cache_file, 'r') as f:
                # Check if this cache has windows
                if 'num_windows' not in f.attrs:
                    logger.warning(f"Skipping {cache_file.parent.name} - no windows found")
                    continue

                num_windows = f.attrs['num_windows']

                # Extract expressions from each window
                for i in range(num_windows):
                    window_key = f'window_{i}'
                    if window_key not in f:
                        continue

                    window_group = f[window_key]

                    # Check if expression_embed exists
                    if 'expression_embed' not in window_group:
                        continue

                    # Load expression embeddings [T, 128]
                    expr = window_group['expression_embed'][:]

                    # Sample with stride
                    if frame_stride > 1:
                        expr = expr[::frame_stride]

                    expressions.append(expr)

        except Exception as e:
            logger.warning(f"Failed to read {cache_file.name}: {e}")
            continue

    if not expressions:
        logger.error("No expressions extracted from cache!")
        return np.array([])

    # Concatenate all expressions [N, 128]
    expressions = np.concatenate(expressions, axis=0)
    logger.info(f"Extracted {len(expressions)} expression embeddings (stride={frame_stride})")

    return expressions


def append_to_database(
    db_path: Path,
    cache_dir: Path,
    frame_stride: int = 1,
    device: str = 'cuda'
):
    """
    Append expressions from cache to existing database.

    Args:
        db_path: Path to existing expression_embeddings.h5
        cache_dir: Path to cache_per_video directory
        frame_stride: Sample every Nth frame
        device: Device to load database on
    """
    # Check if database exists
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        logger.info("Run build_expression_db.py first to create initial database")
        return

    # Check if cache directory exists
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return

    # Load existing database
    logger.info(f"Loading existing database from {db_path}")
    db = ExpressionDatabase(str(db_path), device=device)
    logger.info(f"Current database: {len(db)} embeddings")

    # Extract new expressions from cache
    logger.info(f"Extracting expressions from {cache_dir}")
    new_expressions = extract_expressions_from_cache(cache_dir, frame_stride=frame_stride)

    if len(new_expressions) == 0:
        logger.error("No new expressions to append!")
        return

    # Append to database
    logger.info(f"Appending {len(new_expressions)} new expressions...")
    db.append_embeddings(new_expressions)

    logger.info("✅ Done!")
    logger.info(f"Updated database: {len(db)} embeddings total")
    logger.info(f"Database saved to: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Append expressions to database from cache")
    parser.add_argument(
        '--db',
        type=str,
        default='cache_single_bucket/expression_embeddings.h5',
        help='Path to expression database H5 file'
    )
    parser.add_argument(
        '--cache',
        type=str,
        default='cache_per_video',
        help='Path to cache_per_video directory'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Sample every Nth frame (1=all frames, 5=every 5th)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to load database on (cuda/cpu)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Expression Database Append Tool")
    logger.info("=" * 80)
    logger.info(f"Database: {args.db}")
    logger.info(f"Cache: {args.cache}")
    logger.info(f"Frame stride: {args.stride}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    append_to_database(
        db_path=Path(args.db),
        cache_dir=Path(args.cache),
        frame_stride=args.stride,
        device=args.device
    )


if __name__ == '__main__':
    main()
