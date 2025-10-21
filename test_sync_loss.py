#!/usr/bin/env python3
"""
Test sync loss to verify it's correctly configured and returns expected values.

Tests:
1. Perfect sync (ground truth video + audio) -> Should return near 0
2. Misaligned video (shifted frames) -> Should return high loss
3. Random frames -> Should return very high loss
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import cv2
import numpy as np
from logger import logger
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vasa_losses import VASALossModule
from vasa_dataset import VASAIntegratedDataset

logging.basicConfig(level=logging.INFO)


def load_test_video_and_audio(video_path: str, cache_dir: str = 'cache_per_video'):
    """
    Load a ground truth video with its audio from the dataset.

    Returns:
        frames: [1, T, 3, 512, 512]
        audio_mel: [1, T, 128]
        audio_wav2vec: [1, T, 768]
    """
    from vasa_dataset import VASAIntegratedDataset
    import yaml

    # Load config
    with open('overfit_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Mock EMO model (not needed for audio/frame loading)
    class MockEmoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

    emo_model = MockEmoModel().cuda()

    # Create dataset
    dataset = VASAIntegratedDataset(
        video_folder='s1',
        emo_model=emo_model,
        window_size=50,
        stride=25,
        max_videos=1,  # Just load one video
        cache_dir=cache_dir,
        use_single_bucket=False,
        generate_emo_frames=False,
        cache_frames_to_disk=False,
        cache_emo_frames_to_disk=False,
    )

    # Get first window
    logger.info(f"Loading window 0 from dataset (total windows: {len(dataset)})")
    window_data = dataset[0]

    if window_data is None:
        raise ValueError("Failed to load window data")

    # Extract data
    frames = window_data['frames'].unsqueeze(0).cuda()  # [1, T, 3, 512, 512]
    audio_features = window_data['audio_features'].unsqueeze(0).cuda()  # [1, T, 768]

    # Get mel spectrogram if available
    audio_mel = window_data.get('audio_mel_spec', None)
    if audio_mel is not None:
        audio_mel = audio_mel.unsqueeze(0).cuda()  # [1, T, 128]

    logger.info(f"Loaded data:")
    logger.info(f"  Frames: {frames.shape}")
    logger.info(f"  Audio (wav2vec): {audio_features.shape}")
    if audio_mel is not None:
        logger.info(f"  Audio (mel): {audio_mel.shape}")

    return frames, audio_features, audio_mel


def create_misaligned_video(frames: torch.Tensor, shift_frames: int = 5):
    """
    Create a misaligned version of the video by shifting frames.

    Args:
        frames: [B, T, C, H, W]
        shift_frames: Number of frames to shift (positive = delay video)

    Returns:
        shifted_frames: [B, T, C, H, W]
    """
    B, T, C, H, W = frames.shape

    if shift_frames > 0:
        # Delay video: pad at start, truncate at end
        pad = frames[:, :1].repeat(1, shift_frames, 1, 1, 1)  # Repeat first frame
        shifted = torch.cat([pad, frames[:, :-shift_frames]], dim=1)
    elif shift_frames < 0:
        # Advance video: truncate at start, pad at end
        shift_frames = abs(shift_frames)
        pad = frames[:, -1:].repeat(1, shift_frames, 1, 1, 1)  # Repeat last frame
        shifted = torch.cat([frames[:, shift_frames:], pad], dim=1)
    else:
        shifted = frames

    return shifted


def test_sync_loss_on_perfect_sync(loss_module, frames, audio_mel):
    """Test 1: Perfect sync (ground truth) should return near-zero loss."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Perfect Sync (Ground Truth)")
    logger.info("="*80)

    # Prepare targets
    targets = {
        'frames': frames.clone(),  # Ground truth frames
        'audio_mel_spec': audio_mel,
    }

    # Compute sync loss
    with torch.no_grad():
        sync_loss = loss_module._compute_sync_loss(frames, targets)

    logger.info(f"✅ Sync loss on PERFECT sync: {sync_loss.item():.6f}")
    logger.info(f"   Expected: ~0.0 (near zero)")
    logger.info(f"   Status: {'PASS ✅' if sync_loss.item() < 0.5 else 'FAIL ❌'}")

    return sync_loss.item()


def test_sync_loss_on_misaligned(loss_module, frames, audio_mel, shift=5):
    """Test 2: Misaligned video should return high loss."""
    logger.info("\n" + "="*80)
    logger.info(f"TEST 2: Misaligned Video (shifted by {shift} frames)")
    logger.info("="*80)

    # Create misaligned version
    misaligned_frames = create_misaligned_video(frames, shift_frames=shift)

    # Prepare targets (audio is correct, but video is shifted)
    targets = {
        'frames': frames.clone(),  # Ground truth (correctly aligned)
        'audio_mel_spec': audio_mel,
    }

    # Compute sync loss on misaligned frames
    with torch.no_grad():
        sync_loss = loss_module._compute_sync_loss(misaligned_frames, targets)

    logger.info(f"✅ Sync loss on MISALIGNED video ({shift} frames): {sync_loss.item():.6f}")
    logger.info(f"   Expected: >1.0 (higher than perfect sync)")
    logger.info(f"   Status: {'PASS ✅' if sync_loss.item() > 1.0 else 'FAIL ❌'}")

    return sync_loss.item()


def test_sync_loss_on_random(loss_module, frames, audio_mel):
    """Test 3: Random frames should return very high loss."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Random Frames (No Sync)")
    logger.info("="*80)

    # Create random frames
    B, T, C, H, W = frames.shape
    random_frames = torch.rand(B, T, C, H, W, device=frames.device)

    # Prepare targets
    targets = {
        'frames': frames.clone(),  # Ground truth
        'audio_mel_spec': audio_mel,
    }

    # Compute sync loss on random frames
    with torch.no_grad():
        sync_loss = loss_module._compute_sync_loss(random_frames, targets)

    logger.info(f"✅ Sync loss on RANDOM frames: {sync_loss.item():.6f}")
    logger.info(f"   Expected: >>1.0 (much higher than misaligned)")
    logger.info(f"   Status: {'PASS ✅' if sync_loss.item() > 5.0 else 'FAIL ❌'}")

    return sync_loss.item()


def test_sync_loss_sensitivity(loss_module, frames, audio_mel):
    """Test 4: Test sensitivity to different shift amounts."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Sensitivity Analysis (Different Shift Amounts)")
    logger.info("="*80)

    shifts = [0, 1, 2, 3, 5, 10, 15]
    losses = []

    targets = {
        'frames': frames.clone(),
        'audio_mel_spec': audio_mel,
    }

    logger.info(f"\n{'Shift (frames)':>15} | {'Sync Loss':>12} | {'Status':>10}")
    logger.info(f"{'-'*15}-+-{'-'*12}-+-{'-'*10}")

    for shift in shifts:
        if shift == 0:
            test_frames = frames
        else:
            test_frames = create_misaligned_video(frames, shift_frames=shift)

        with torch.no_grad():
            sync_loss = loss_module._compute_sync_loss(test_frames, targets)

        losses.append(sync_loss.item())

        # Determine status
        if shift == 0:
            status = "Perfect ✅" if sync_loss.item() < 0.5 else "Bad ❌"
        elif shift <= 3:
            status = "Minor ⚠️"
        else:
            status = "Poor ❌"

        logger.info(f"{shift:15d} | {sync_loss.item():12.6f} | {status:>10}")

    # Check if loss increases monotonically with shift
    is_monotonic = all(losses[i] <= losses[i+1] for i in range(len(losses)-1))
    logger.info(f"\n✅ Loss increases with misalignment: {'PASS ✅' if is_monotonic else 'FAIL ❌ (not monotonic)'}")

    return losses


def test_without_ground_truth(loss_module, frames, audio_mel):
    """Test 5: Test behavior when no ground truth frames provided."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: No Ground Truth Frames (Training Mode)")
    logger.info("="*80)

    # Prepare targets WITHOUT ground truth frames
    targets = {
        'audio_mel_spec': audio_mel,
        # No 'frames' key - simulates training where we only have audio
    }

    # Test on perfect sync
    with torch.no_grad():
        sync_loss_perfect = loss_module._compute_sync_loss(frames, targets)

    logger.info(f"✅ Sync loss on perfect video (no GT): {sync_loss_perfect.item():.6f}")

    # Test on misaligned
    misaligned = create_misaligned_video(frames, shift_frames=5)
    with torch.no_grad():
        sync_loss_misaligned = loss_module._compute_sync_loss(misaligned, targets)

    logger.info(f"✅ Sync loss on misaligned video (no GT): {sync_loss_misaligned.item():.6f}")

    # Without GT, loss assumes perfect sync (offset=0), so misaligned should have higher loss
    logger.info(f"   Difference: {abs(sync_loss_misaligned.item() - sync_loss_perfect.item()):.6f}")
    logger.info(f"   Status: {'PASS ✅' if sync_loss_misaligned.item() > sync_loss_perfect.item() else 'FAIL ❌'}")


def main():
    logger.info("\n" + "="*80)
    logger.info("SYNC LOSS DIAGNOSTIC SUITE")
    logger.info("="*80)
    logger.info("Purpose: Verify sync loss returns expected values for different scenarios")
    logger.info("")

    try:
        # Load loss module with Synchformer
        logger.info("Initializing VASALosses with Synchformer...")

        # Mock volumetric avatar (not needed for sync loss testing)
        class MockVolumetricAvatar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

        # Load actual config to get all required attributes
        from omegaconf import OmegaConf

        config = OmegaConf.load('overfit_config.yaml')

        # Ensure sync loss is enabled
        config.loss.use_synchformer = True
        config.loss.use_sync_loss = True
        volumetric_avatar = MockVolumetricAvatar().cuda()
        loss_module = VASALossModule(volumetric_avatar, config, device='cuda')
        logger.info("✅ VASALossModule initialized")

        # Load test data
        logger.info("\nLoading ground truth video and audio...")
        frames, audio_wav2vec, audio_mel = load_test_video_and_audio('s1', cache_dir='cache_per_video')

        if audio_mel is None:
            logger.error("❌ No mel spectrogram available! Sync loss needs mel_spec.")
            logger.error("   Run preprocessing with mel spectrogram extraction enabled.")
            return

        logger.info("✅ Test data loaded")

        # Run tests
        perfect_loss = test_sync_loss_on_perfect_sync(loss_module, frames, audio_mel)
        misaligned_loss = test_sync_loss_on_misaligned(loss_module, frames, audio_mel, shift=5)
        random_loss = test_sync_loss_on_random(loss_module, frames, audio_mel)

        sensitivity_losses = test_sync_loss_sensitivity(loss_module, frames, audio_mel)

        test_without_ground_truth(loss_module, frames, audio_mel)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)

        logger.info(f"\nLoss Values:")
        logger.info(f"  Perfect sync:      {perfect_loss:.6f}")
        logger.info(f"  Misaligned (5f):   {misaligned_loss:.6f}")
        logger.info(f"  Random frames:     {random_loss:.6f}")

        # Determine overall status
        tests_passed = 0
        tests_total = 4

        if perfect_loss < 0.5:
            logger.info(f"\n✅ TEST 1 PASSED: Perfect sync has low loss ({perfect_loss:.6f} < 0.5)")
            tests_passed += 1
        else:
            logger.error(f"\n❌ TEST 1 FAILED: Perfect sync has high loss ({perfect_loss:.6f} >= 0.5)")
            logger.error(f"   This means sync loss is NOT working correctly!")
            logger.error(f"   Even ground truth videos return high sync loss.")

        if misaligned_loss > perfect_loss:
            logger.info(f"✅ TEST 2 PASSED: Misaligned has higher loss than perfect ({misaligned_loss:.6f} > {perfect_loss:.6f})")
            tests_passed += 1
        else:
            logger.error(f"❌ TEST 2 FAILED: Misaligned doesn't have higher loss")

        if random_loss > misaligned_loss:
            logger.info(f"✅ TEST 3 PASSED: Random has highest loss ({random_loss:.6f} > {misaligned_loss:.6f})")
            tests_passed += 1
        else:
            logger.error(f"❌ TEST 3 FAILED: Random doesn't have highest loss")

        # Check monotonicity
        is_monotonic = all(sensitivity_losses[i] <= sensitivity_losses[i+1] for i in range(len(sensitivity_losses)-1))
        if is_monotonic:
            logger.info(f"✅ TEST 4 PASSED: Loss increases monotonically with misalignment")
            tests_passed += 1
        else:
            logger.error(f"❌ TEST 4 FAILED: Loss does not increase monotonically")

        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL RESULT: {tests_passed}/{tests_total} tests passed")

        if tests_passed == tests_total:
            logger.info("✅ SYNC LOSS IS WORKING CORRECTLY!")
            logger.info("   It returns low loss for perfect sync and high loss for misalignment.")
            logger.info("   The high loss during training (8.17) indicates poor audio-visual sync.")
        else:
            logger.error("❌ SYNC LOSS IS NOT WORKING CORRECTLY!")
            logger.error("   There's a configuration or implementation issue.")
            logger.error("   The loss values don't match expected behavior.")

        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
