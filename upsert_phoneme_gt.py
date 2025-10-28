#!/usr/bin/env python3
"""
Upsert phoneme_gt to existing cache files without reprocessing everything.

This script:
1. Loads existing cached windows
2. Extracts phoneme sequences from cached audio_waveform
3. Updates cache files with phoneme_gt field
"""

import torch
import numpy as np
from pathlib import Path
import logging
import argparse
import h5py
import traceback
from tqdm import tqdm
from per_video_cache import PerVideoCache
from vasa_dataset import WorkerState
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_phoneme_sequence(
    audio_waveform: torch.Tensor,
    sample_rate: int,
    num_queries: int,
    worker_state: WorkerState
) -> torch.Tensor:
    """
    Extract phoneme sequence from audio using wav2vec2 phoneme recognition.

    Args:
        audio_waveform: Raw audio tensor [samples] or [1, samples]
        sample_rate: Audio sample rate (usually 16000 Hz)
        num_queries: Number of latent queries to align phonemes to (default: 8)
        worker_state: WorkerState instance with phoneme model

    Returns:
        Phoneme IDs tensor [num_queries] - one phoneme per latent query
    """
    try:
        # Ensure audio is 1D
        if audio_waveform.ndim > 1:
            audio_waveform = audio_waveform.squeeze(0)

        # Process audio with phoneme model
        inputs = worker_state.phoneme_processor(
            audio_waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_values

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        with torch.no_grad():
            logits = worker_state.phoneme_model(inputs).logits  # [1, T_phoneme, vocab_size]

        phoneme_ids = torch.argmax(logits, dim=-1)  # [1, T_phoneme]
        phoneme_seq = phoneme_ids.squeeze(0).cpu()  # [T_phoneme]

        # Align to num_queries using max pooling
        if len(phoneme_seq) < num_queries:
            # Pad with zeros if too short
            pooled_phoneme = torch.zeros(num_queries, dtype=torch.long)
            pooled_phoneme[:len(phoneme_seq)] = phoneme_seq
        else:
            # Max pool to match num_queries
            kernel_size = max(1, len(phoneme_seq) // num_queries)
            stride = kernel_size

            pooled = torch.nn.functional.max_pool1d(
                phoneme_seq.unsqueeze(0).unsqueeze(0).float(),
                kernel_size=kernel_size,
                stride=stride
            )
            pooled_phoneme = pooled.squeeze().long()[:num_queries]

            # Pad if needed
            if len(pooled_phoneme) < num_queries:
                padded = torch.zeros(num_queries, dtype=torch.long)
                padded[:len(pooled_phoneme)] = pooled_phoneme
                pooled_phoneme = padded

        return pooled_phoneme  # [num_queries]

    except Exception as e:
        logger.error(f"Error extracting phoneme sequence: {str(e)}")
        logger.error(traceback.format_exc())
        # Return zeros on error
        return torch.zeros(num_queries, dtype=torch.long)


def upsert_phoneme_to_per_video_cache(
    cache_dir: Path,
    num_queries: int = 8,
    sample_rate: int = 16000,
    dry_run: bool = False
):
    """
    Upsert phoneme_gt to all windows in per-video cache.

    Args:
        cache_dir: Directory with per-video cache (MD5 folders)
        num_queries: Number of latent queries for phoneme alignment
        sample_rate: Audio sample rate
        dry_run: If True, don't save changes
    """
    logger.info("="*80)
    logger.info("UPSERT PHONEME_GT TO PER-VIDEO CACHE")
    logger.info("="*80)
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Number of queries: {num_queries}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Dry run: {dry_run}")

    # Initialize worker state (loads phoneme model)
    logger.info("\n📦 Loading phoneme model...")
    worker_state = WorkerState.get_instance()
    _ = worker_state.phoneme_model  # Trigger lazy loading
    logger.info("✅ Phoneme model loaded")

    # Initialize cache
    cache = PerVideoCache(cache_dir=cache_dir)

    # Rebuild index to get all videos
    logger.info("\n📋 Scanning cache directory...")
    index = cache.rebuild_index()

    total_videos = len(index)
    total_windows_processed = 0
    total_windows_updated = 0
    failed_windows = []

    logger.info(f"Found {total_videos} cached videos\n")

    # Process each video
    for video_idx, (video_md5, video_info) in enumerate(tqdm(index.items(), desc="Processing videos")):
        video_path = video_info['video_path']
        video_name = Path(video_path).name
        num_windows = video_info['num_windows']

        try:
            logger.info(f"\n📹 Video {video_idx + 1}/{total_videos}: {video_name}")
            logger.info(f"   MD5: {video_md5}")
            logger.info(f"   Windows: {num_windows}")

            # Load metadata H5 file
            h5_path = cache.cache_dir / video_md5 / 'metadata.h5'

            if not h5_path.exists():
                logger.warning(f"⚠️  No metadata.h5 found for {video_name}, skipping")
                continue

            # Open H5 file in read/write mode
            with h5py.File(h5_path, 'r+' if not dry_run else 'r') as h5f:
                windows_updated = 0

                # Process each window in this video
                for window_key in h5f.keys():
                    try:
                        window_group = h5f[window_key]

                        # Check if phoneme_gt already exists
                        if 'phoneme_gt' in window_group:
                            logger.debug(f"   {window_key}: phoneme_gt already exists, skipping")
                            total_windows_processed += 1
                            continue

                        # Get audio_waveform
                        if 'audio_waveform' not in window_group:
                            logger.warning(f"   {window_key}: No audio_waveform, skipping")
                            failed_windows.append((video_name, window_key, "No audio_waveform"))
                            total_windows_processed += 1
                            continue

                        audio_waveform = torch.from_numpy(window_group['audio_waveform'][:])

                        # Extract phoneme sequence
                        phoneme_gt = extract_phoneme_sequence(
                            audio_waveform=audio_waveform,
                            sample_rate=sample_rate,
                            num_queries=num_queries,
                            worker_state=worker_state
                        )

                        # Save to H5 file (unless dry run)
                        if not dry_run:
                            window_group.create_dataset(
                                'phoneme_gt',
                                data=phoneme_gt.numpy(),
                                dtype='int64',
                                compression='gzip',
                                compression_opts=4
                            )

                        windows_updated += 1
                        total_windows_updated += 1
                        total_windows_processed += 1

                        logger.debug(f"   {window_key}: ✅ Added phoneme_gt {phoneme_gt.tolist()}")

                    except Exception as e:
                        logger.error(f"   {window_key}: ❌ Error - {str(e)}")
                        failed_windows.append((video_name, window_key, str(e)))
                        total_windows_processed += 1
                        continue

                logger.info(f"   Updated {windows_updated}/{num_windows} windows")

        except Exception as e:
            logger.error(f"❌ Error processing video {video_name}: {e}")
            logger.error(traceback.format_exc())
            continue

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ UPSERT COMPLETE")
    logger.info("="*80)
    logger.info(f"Videos processed: {total_videos}")
    logger.info(f"Windows processed: {total_windows_processed}")
    logger.info(f"Windows updated: {total_windows_updated}")
    logger.info(f"Windows failed: {len(failed_windows)}")

    if dry_run:
        logger.info("\n⚠️  DRY RUN MODE - No changes were saved")
        logger.info(f"   Would have updated {total_windows_updated} windows")

    if failed_windows:
        logger.warning(f"\n❌ Failed windows: {len(failed_windows)}")
        for video_name, window_key, error in failed_windows[:10]:
            logger.warning(f"   {video_name}/{window_key}: {error}")
        if len(failed_windows) > 10:
            logger.warning(f"   ... and {len(failed_windows) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Upsert phoneme_gt to existing per-video cache files'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache_per_video',
        help='Per-video cache directory (contains MD5 folders)'
    )
    parser.add_argument(
        '--num_queries',
        type=int,
        default=8,
        help='Number of latent queries for phoneme alignment (default: 8)'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Run without saving changes (for testing)'
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        return

    upsert_phoneme_to_per_video_cache(
        cache_dir=cache_dir,
        num_queries=args.num_queries,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
