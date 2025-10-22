#!/usr/bin/env python3
"""
Diagnose phoneme predictions for cached windows.

This script:
1. Loads cached windows with phoneme_gt
2. Runs model to get phoneme predictions
3. Compares predictions vs ground truth
4. Shows phoneme sequences with human-readable labels
"""

import torch
import numpy as np
from pathlib import Path
import logging
import argparse
import h5py
from tqdm import tqdm
from per_video_cache import PerVideoCache
from vasa_dataset import VASAIntegratedDataset, WorkerState
import torch.nn.functional as F
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IPA phoneme mapping (from wav2vec2-xlsr-53-espeak-cv-ft)
# These are the most common phonemes in the espeak vocabulary
PHONEME_LABELS = {
    0: '<pad>',
    1: 'a', 2: 'aɪ', 3: 'aʊ', 4: 'b', 5: 'd', 6: 'e', 7: 'eɪ', 8: 'f',
    9: 'g', 10: 'h', 11: 'i', 12: 'iː', 13: 'j', 14: 'k', 15: 'l',
    16: 'm', 17: 'n', 18: 'o', 19: 'oʊ', 20: 'p', 21: 'r', 22: 's',
    23: 't', 24: 'u', 25: 'uː', 26: 'v', 27: 'w', 28: 'z', 29: 'ð',
    30: 'ŋ', 31: 'ɔ', 32: 'ɔɪ', 33: 'ə', 34: 'ɛ', 35: 'ɪ', 36: 'ʃ',
    37: 'ʊ', 38: 'ʌ', 39: 'ʒ', 40: 'θ', 41: 'æ', 42: 'ɑ', 43: 'ɜ',
    44: 'ɡ', 45: 'ɹ', 46: 'tʃ', 47: 'dʒ', 48: 'ʔ', 49: 'ˈ'
}

def phoneme_ids_to_labels(phoneme_ids):
    """Convert phoneme IDs to human-readable labels."""
    return [PHONEME_LABELS.get(int(pid), f'<{pid}>') for pid in phoneme_ids]


def load_model(config_path, checkpoint_path):
    """Load VASA model from checkpoint."""
    logger.info("Loading model...")

    # Load config
    config = OmegaConf.load(config_path)

    # Load model
    from vasa_model import VASAModel
    model = VASAModel(
        d_model=config.model.hidden_dim,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        expression_dim=config.model.expression_dim,
        max_seq_len=config.motion.window_size,
        use_talkvid_audio_projection=config.model.get('use_talkvid_audio_projection', True)
    )

    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Using untrained model")

    model = model.cuda()
    model.eval()

    return model


def get_phoneme_predictions(model, window_data):
    """
    Run model forward pass and extract phoneme predictions.

    Args:
        model: VASAModel instance
        window_data: Dict with window data from cache

    Returns:
        phoneme_pred: [8, 50] predicted probabilities
        phoneme_gt: [8] ground truth phoneme IDs
    """
    device = 'cuda'

    # Prepare motion data
    motion_data = {
        'theta': window_data['theta'].unsqueeze(0).to(device),  # [1, T, 3, 4]
        'expression_embed': window_data['expression_embed'].unsqueeze(0).to(device),  # [1, T, 128]
        'scale': window_data['scale'].unsqueeze(0).to(device),
        'rotation': window_data['rotation'].unsqueeze(0).to(device),
        'translation': window_data['translation'].unsqueeze(0).to(device),
    }

    # Prepare conditions
    conditions = {
        'audio_features': window_data['audio_features'].unsqueeze(0).to(device),  # [1, T, 768]
        'gaze': window_data.get('gaze', torch.zeros(1, 50, 2)).unsqueeze(0).to(device),
        'head_distance': window_data.get('head_distance', torch.zeros(1, 50, 1)).unsqueeze(0).to(device),
        'emotion': window_data.get('emotion', torch.zeros(1, 50, 2)).unsqueeze(0).to(device),
        'phoneme_gt': window_data.get('phoneme_gt', torch.zeros(8, dtype=torch.long)).unsqueeze(0).to(device),  # [1, 8]
    }

    # Add landmarks if available
    for key in ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']:
        if key in window_data:
            conditions[key] = window_data[key].unsqueeze(0).to(device)

    # Noise level (for diffusion model)
    noise_level = torch.zeros(1, device=device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            motion_data=motion_data,
            noise_level=noise_level,
            conditions=conditions
        )

    # Extract phoneme predictions
    if 'aux_predictions' in outputs:
        aux = outputs['aux_predictions']
        if 'phoneme_pred' in aux:
            phoneme_pred = aux['phoneme_pred'].squeeze(0)  # [8, 50]
            phoneme_gt = aux.get('phoneme_gt', conditions['phoneme_gt']).squeeze(0)  # [8]
            return phoneme_pred, phoneme_gt

    return None, None


def diagnose_video(
    cache_dir: Path,
    video_path: str,
    model=None,
    max_windows: int = 10
):
    """
    Diagnose phoneme predictions for a specific video.

    Args:
        cache_dir: Per-video cache directory
        video_path: Path to video file
        model: VASAModel instance (optional, for predictions)
        max_windows: Maximum number of windows to analyze
    """
    cache = PerVideoCache(cache_dir=cache_dir)
    video_md5 = cache.get_video_hash(video_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"Diagnosing video: {Path(video_path).name}")
    logger.info(f"MD5: {video_md5}")
    logger.info(f"{'='*80}\n")

    # Load metadata H5 file
    h5_path = cache.cache_dir / video_md5 / 'metadata.h5'

    if not h5_path.exists():
        logger.error(f"No metadata.h5 found at {h5_path}")
        return

    results = []

    with h5py.File(h5_path, 'r') as h5f:
        window_keys = sorted(h5f.keys(), key=lambda x: int(x.split('_')[1]))[:max_windows]

        for window_key in tqdm(window_keys, desc="Analyzing windows"):
            window_group = h5f[window_key]

            # Check if phoneme_gt exists
            if 'phoneme_gt' not in window_group:
                logger.warning(f"{window_key}: No phoneme_gt found")
                continue

            # Load window data
            window_data = {}
            for key in window_group.keys():
                data = window_group[key][:]
                window_data[key] = torch.from_numpy(data)

            phoneme_gt = window_data['phoneme_gt']  # [8]
            phoneme_gt_labels = phoneme_ids_to_labels(phoneme_gt)

            result = {
                'window': window_key,
                'phoneme_gt': phoneme_gt.numpy(),
                'phoneme_gt_labels': phoneme_gt_labels,
            }

            # Get predictions if model provided
            if model is not None:
                phoneme_pred, _ = get_phoneme_predictions(model, window_data)

                if phoneme_pred is not None:
                    # Get predicted phoneme IDs (argmax)
                    phoneme_pred_ids = torch.argmax(phoneme_pred, dim=-1)  # [8]
                    phoneme_pred_labels = phoneme_ids_to_labels(phoneme_pred_ids)

                    # Get prediction confidence (softmax probabilities)
                    phoneme_probs = F.softmax(phoneme_pred, dim=-1)  # [8, 50]
                    confidence = torch.max(phoneme_probs, dim=-1)[0]  # [8]

                    # Compute accuracy
                    accuracy = (phoneme_pred_ids == phoneme_gt).float().mean().item()

                    result.update({
                        'phoneme_pred': phoneme_pred_ids.cpu().numpy(),
                        'phoneme_pred_labels': phoneme_pred_labels,
                        'confidence': confidence.cpu().numpy(),
                        'accuracy': accuracy,
                    })

            results.append(result)

    # Print results
    print(f"\n{'='*80}")
    print(f"PHONEME DIAGNOSIS REPORT")
    print(f"Video: {Path(video_path).name}")
    print(f"Windows analyzed: {len(results)}")
    print(f"{'='*80}\n")

    for i, result in enumerate(results):
        print(f"Window {result['window']}:")
        print(f"  Ground Truth:")
        print(f"    IDs:     {result['phoneme_gt']}")
        print(f"    Phonemes: {' '.join(result['phoneme_gt_labels'])}")

        if 'phoneme_pred' in result:
            print(f"  Prediction:")
            print(f"    IDs:     {result['phoneme_pred']}")
            print(f"    Phonemes: {' '.join(result['phoneme_pred_labels'])}")
            print(f"    Confidence: {result['confidence']}")
            print(f"    Accuracy: {result['accuracy']:.2%}")

            # Highlight mismatches
            mismatches = []
            for j, (gt, pred) in enumerate(zip(result['phoneme_gt_labels'], result['phoneme_pred_labels'])):
                if gt != pred:
                    mismatches.append(f"Query {j}: {gt} → {pred}")

            if mismatches:
                print(f"  Mismatches:")
                for mismatch in mismatches:
                    print(f"    {mismatch}")

        print()

    # Summary statistics
    if model is not None and results:
        accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
        avg_accuracy = np.mean(accuracies)
        avg_confidence = np.mean([r['confidence'].mean() for r in results if 'confidence' in r])

        print(f"{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Average Accuracy: {avg_accuracy:.2%}")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print(f"{'='*80}\n")


def diagnose_cache(
    cache_dir: Path,
    model=None,
    max_videos: int = 5,
    max_windows_per_video: int = 5
):
    """
    Diagnose phoneme predictions across multiple videos in cache.
    """
    cache = PerVideoCache(cache_dir=cache_dir)
    index = cache.rebuild_index()

    logger.info(f"Found {len(index)} videos in cache")

    video_list = list(index.items())[:max_videos]

    for video_md5, video_info in video_list:
        video_path = video_info['video_path']
        diagnose_video(
            cache_dir=cache_dir,
            video_path=video_path,
            model=model,
            max_windows=max_windows_per_video
        )


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose phoneme predictions for cached windows'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache_per_video',
        help='Per-video cache directory'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Specific video file to analyze (optional)'
    )
    parser.add_argument(
        '--max_videos',
        type=int,
        default=5,
        help='Maximum number of videos to analyze (if no specific video)'
    )
    parser.add_argument(
        '--max_windows',
        type=int,
        default=10,
        help='Maximum windows per video'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='overfit_config.yaml',
        help='Model config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints_overfit/best_checkpoint.pt',
        help='Model checkpoint file (optional, for predictions)'
    )
    parser.add_argument(
        '--no_model',
        action='store_true',
        help='Skip model loading (only show ground truth)'
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        return

    # Load model (optional)
    model = None
    if not args.no_model and Path(args.checkpoint).exists():
        model = load_model(args.config, args.checkpoint)
    elif not args.no_model:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
        logger.warning("Will only show ground truth phonemes")

    # Analyze specific video or scan cache
    if args.video:
        diagnose_video(
            cache_dir=cache_dir,
            video_path=args.video,
            model=model,
            max_windows=args.max_windows
        )
    else:
        diagnose_cache(
            cache_dir=cache_dir,
            model=model,
            max_videos=args.max_videos,
            max_windows_per_video=args.max_windows
        )


if __name__ == "__main__":
    main()
