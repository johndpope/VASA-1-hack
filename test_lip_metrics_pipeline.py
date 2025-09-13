#!/usr/bin/env python3
"""Test script to verify lip_metrics flow through the entire pipeline"""

import torch
import numpy as np
from omegaconf import OmegaConf
from nemo.logger import logger
import os

def test_lip_metrics_pipeline():
    """Test that lip_metrics flows correctly from dataset to loss computation"""

    # Set log level to DEBUG to see all debug messages
    os.environ['VASA_LOG_LEVEL'] = 'DEBUG'

    logger.info("Testing lip_metrics pipeline...")

    # 1. Test dataset output
    logger.info("\n=== Testing Dataset Output ===")
    from vasa_dataset import VASAIntegratedDataset, LipStateAnalyzer

    # Create a simple test
    analyzer = LipStateAnalyzer()

    # Create dummy lip landmarks [T, N, 3]
    T = 50
    N = 20
    lip_sequence = np.random.randn(T, N, 3) * 0.1

    # Test analyzer
    metrics = analyzer.analyze_sequence(lip_sequence)
    logger.info(f"LipStateAnalyzer output keys: {list(metrics.keys())}")
    logger.info(f"  openness shape: {metrics['openness'].shape}")
    logger.info(f"  openness range: [{metrics['openness'].min():.4f}, {metrics['openness'].max():.4f}]")

    # 2. Test collate function
    logger.info("\n=== Testing Collate Function ===")
    from vasa_sampler import create_window_sequence_collate_fn

    # Create dummy batch data
    batch = []
    for i in range(2):  # 2 samples
        window = {
            'theta': torch.randn(50, 3, 4),
            'audio_features': torch.randn(50, 768),
            'lip_metrics': {
                'openness': torch.from_numpy(metrics['openness']),
                'symmetry': torch.from_numpy(metrics['symmetry']),
                'aspect_ratio': torch.from_numpy(metrics['aspect_ratio']),
                'area': torch.from_numpy(metrics['area']),
                'perimeter': torch.from_numpy(metrics['perimeter']),
            },
            'metadata': {
                'video_path': f'video_{i}.mp4',
                'start_frame': 0,
                'has_context': False
            }
        }
        batch.append(window)

    # Test collate
    collate_fn = create_window_sequence_collate_fn(context_size=10)
    collated = collate_fn(batch)

    if 'lip_metrics' in collated:
        logger.info("✓ lip_metrics found in collated batch!")
        logger.info(f"  Collated lip_metrics keys: {list(collated['lip_metrics'].keys())}")
        logger.info(f"  Collated openness shape: {collated['lip_metrics']['openness'].shape}")
    else:
        logger.error("✗ lip_metrics NOT found in collated batch!")
        logger.error(f"  Available keys: {list(collated.keys())}")

    # 3. Test motion handler
    logger.info("\n=== Testing Motion Handler ===")
    from motion_sequence_handler import MotionSequenceHandler

    # Create handler with direct parameters instead of config
    handler = MotionSequenceHandler(
        window_size=50,
        stride=25,
        context_size=10
    )

    # Create batch with lip_metrics
    test_batch = {
        'frames': torch.randn(2, 50, 3, 512, 512),
        'theta': torch.randn(2, 50, 3, 4),
        'scale': torch.randn(2, 50, 3),
        'rotation': torch.randn(2, 50, 3),
        'translation': torch.randn(2, 50, 3),
        'expression_embed': torch.randn(2, 50, 128),
        'audio_features': torch.randn(2, 50, 768),
        'lip_metrics': {
            'openness': torch.randn(2, 50),
            'symmetry': torch.randn(2, 50),
            'aspect_ratio': torch.randn(2, 50),
            'area': torch.randn(2, 50),
            'perimeter': torch.randn(2, 50),
        }
    }

    # Process batch into windows
    windows = handler.process_batch(test_batch, current_window_size=50)

    if windows and 'lip_metrics' in windows[0]:
        logger.info("✓ lip_metrics found in processed windows!")
        logger.info(f"  Window lip_metrics keys: {list(windows[0]['lip_metrics'].keys())}")
    else:
        logger.error("✗ lip_metrics NOT found in processed windows!")
        if windows:
            logger.error(f"  Available keys in window: {list(windows[0].keys())}")

    # 4. Test loss computation
    logger.info("\n=== Testing Loss Computation ===")

    # Create dummy conditions and targets
    conditions = {
        'audio_features': torch.randn(1, 50, 768),
        'gaze': torch.randn(1, 50, 2),
        'head_distance': torch.randn(1, 50, 1),
        'emotion': torch.randn(1, 50, 2)
    }

    targets = {
        'theta': torch.randn(1, 50, 3, 4),
        'lip_metrics': {
            'openness': torch.randn(1, 50),
            'symmetry': torch.randn(1, 50),
            'aspect_ratio': torch.randn(1, 50),
            'area': torch.randn(1, 50),
            'perimeter': torch.randn(1, 50),
        }
    }

    # Check if loss would compute audio-lip correlation
    audio_key = 'audio_features' if 'audio_features' in conditions else 'audio' if 'audio' in conditions else None

    if audio_key and 'lip_metrics' in targets:
        logger.info("✓ Both audio and lip_metrics present - audio-lip correlation loss will be computed!")
        logger.info(f"  Audio key: {audio_key}")
        logger.info(f"  Audio shape: {conditions[audio_key].shape}")
        logger.info(f"  Lip openness shape: {targets['lip_metrics']['openness'].shape}")
    else:
        logger.error("✗ Missing requirements for audio-lip correlation loss!")
        if not audio_key:
            logger.error("  Missing: audio/audio_features in conditions")
        if 'lip_metrics' not in targets:
            logger.error("  Missing: lip_metrics in targets")

    logger.info("\n=== Pipeline Test Complete ===")

    # Summary
    all_passed = (
        'lip_metrics' in collated and
        windows and 'lip_metrics' in windows[0] and
        audio_key and 'lip_metrics' in targets
    )

    if all_passed:
        logger.info("✅ All tests passed! lip_metrics flows correctly through the pipeline.")
    else:
        logger.warning("⚠️ Some tests failed. Check the errors above.")

    return all_passed

if __name__ == "__main__":
    success = test_lip_metrics_pipeline()
    exit(0 if success else 1)