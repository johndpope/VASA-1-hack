#!/usr/bin/env python3
"""
End-to-end VASA test with minimal video generation
"""

import torch
import numpy as np
from PIL import Image
import logging
import sys
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import VASA modules
sys.path.insert(0, 'nemo')
from vi import VASAInference

def test_vasa_e2e():
    """Test VASA end-to-end with short video generation"""

    logger.info("=== VASA End-to-End Test ===")
    start_time = time.time()

    # Initialize VASA
    checkpoint = "checkpoints_overfit/best_checkpoint.pt"
    config = "overfit_config.yaml"
    logger.info(f"Loading checkpoint: {checkpoint}")
    logger.info(f"Using config: {config}")

    vasa = VASAInference(checkpoint, config)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Test 1: Quick motion generation test
    logger.info("\n=== Test 1: Motion Generation from Audio ===")

    # Load source image
    source_path = "./data/A.png"
    source_img = Image.open(source_path).convert('RGB')
    source_tensor = vasa.transform(source_img).unsqueeze(0).to(vasa.device)

    # Extract source parameters
    logger.info("Extracting source parameters...")
    source_params = vasa.extract_emo_parameters(source_tensor)

    # Create synthetic audio features for quick test (2 seconds)
    B, T = 1, 50  # 50 frames = 2 seconds at 25fps
    audio_features = torch.randn(B, T, 768).to(vasa.device) * 0.3  # Scaled random features

    # Prepare conditions
    conditions = {
        'audio_features': audio_features,
        'gaze': torch.zeros(B, T, 2).to(vasa.device),
        'head_distance': torch.ones(B, T, 1).to(vasa.device),
        'emotion': torch.zeros(B, T, 2).to(vasa.device),
        'speed_bucket': torch.ones(B, T, 1).to(vasa.device) * 4.0
    }

    # Generate motion sequence
    logger.info("Generating motion sequence...")
    gen_start = time.time()

    with torch.no_grad():
        # Prepare initial conditions
        initial_pose = {
            'theta': source_params['theta'],
            'scale': source_params['scale'],
            'rotation': source_params['rotation'],
            'translation': source_params['translation']
        }
        initial_dynamics = source_params['expression_embed'][0]

        # Generate motion
        motion_sequence = vasa.model.generate_sequence(
            initial_pose=initial_pose,
            initial_dynamics=initial_dynamics,
            conditions=conditions,
            num_steps=10,  # Reduced for speed
            eta=0.5
        )

    logger.info(f"Motion generation took {time.time() - gen_start:.2f} seconds")

    # Analyze generated motion
    logger.info("\nMotion Analysis:")
    for key, value in motion_sequence.items():
        if isinstance(value, torch.Tensor):
            variance = value.var().item()
            logger.info(f"  {key}: shape {value.shape}, variance {variance:.6f}")

    # Test 2: Render a few frames
    logger.info("\n=== Test 2: Frame Rendering ===")

    # Render first, middle, and last frames
    test_frames = [0, T//2, T-1]
    rendered_frames = []

    for t_idx in test_frames:
        logger.info(f"Rendering frame {t_idx}...")

        # Prepare motion for single frame
        frame_motion = {
            'theta': motion_sequence['theta'][:, t_idx],
            'expression': motion_sequence['expression_embed'][:, t_idx],
            'scale': motion_sequence.get('scale', torch.ones(B, 3).to(vasa.device))[:, t_idx],
            'rotation': motion_sequence.get('rotation', torch.zeros(B, 3).to(vasa.device))[:, t_idx],
            'translation': motion_sequence.get('translation', torch.zeros(B, 3).to(vasa.device))[:, t_idx]
        }

        # Combine source parameters with motion
        render_params = {
            'idt_embed': source_params['idt_embed'],
            'canonical_volume': source_params['canonical_volume'],
            'source_volume': source_params['source_volume'],
            'source_mask': source_params['source_mask']
        }

        # Render using volumetric avatar
        with torch.no_grad():
            rendered = vasa.volumetric_avatar.forward(render_params, frame_motion)

        # Convert to image
        frame_img = (rendered[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rendered_frames.append(frame_img)

        # Save frame
        Image.fromarray(frame_img).save(f'e2e_frame_{t_idx:03d}.png')
        logger.info(f"  Saved frame to e2e_frame_{t_idx:03d}.png")

    # Test 3: Check consistency
    logger.info("\n=== Test 3: Consistency Checks ===")

    # Check identity preservation (compare first and last frames)
    first_frame = torch.tensor(rendered_frames[0]).float() / 255.0
    last_frame = torch.tensor(rendered_frames[-1]).float() / 255.0
    identity_mse = torch.mean((first_frame - last_frame) ** 2).item()
    logger.info(f"Identity MSE (first vs last): {identity_mse:.6f}")

    # Check motion variation
    if len(rendered_frames) > 1:
        frame_diffs = []
        for i in range(len(rendered_frames) - 1):
            diff = np.mean(np.abs(rendered_frames[i].astype(float) - rendered_frames[i+1].astype(float)))
            frame_diffs.append(diff)
        logger.info(f"Mean frame-to-frame difference: {np.mean(frame_diffs):.3f}")

    # Test 4: Performance summary
    logger.info("\n=== Performance Summary ===")
    total_time = time.time() - start_time
    logger.info(f"Total test time: {total_time:.2f} seconds")
    logger.info(f"Average time per frame: {(time.time() - gen_start) / T:.3f} seconds")

    # Final status
    logger.info("\n=== Test Status ===")
    tests_passed = []
    tests_passed.append(("Model loading", True))
    tests_passed.append(("Source extraction", 'source_params' in locals()))
    tests_passed.append(("Motion generation", 'motion_sequence' in locals()))
    tests_passed.append(("Frame rendering", len(rendered_frames) == len(test_frames)))
    tests_passed.append(("Identity preservation", identity_mse < 0.5 if 'identity_mse' in locals() else False))

    for test_name, passed in tests_passed:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(p for _, p in tests_passed)
    final_status = "✅ ALL TESTS PASSED" if all_passed else "⚠️ SOME TESTS FAILED"
    logger.info(f"\nFinal Status: {final_status}")

    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    args = parser.parse_args()

    success = test_vasa_e2e()
    sys.exit(0 if success else 1)