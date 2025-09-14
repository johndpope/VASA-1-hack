#!/usr/bin/env python3
"""Test that audio variance is preserved with JoyVASA-aligned approach"""

import torch
import numpy as np
from omegaconf import OmegaConf
from vasa_model import EfficientConditionEmbedding
from nemo.logger import logger
import os

def test_audio_variance_preservation():
    """Test that removing LayerNorm preserves audio variance"""

    # Set debug logging
    os.environ['VASA_LOG_LEVEL'] = 'DEBUG'

    print("\n=== Testing Audio Variance Preservation (JoyVASA Aligned) ===\n")

    # Initialize the condition embedding module
    cond_emb = EfficientConditionEmbedding(model_dim=512, max_seq_len=60)
    cond_emb.eval()

    B, T = 2, 50  # Batch size, sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cond_emb = cond_emb.to(device)

    # Test 1: Silent audio (zeros)
    print("Test 1: Silent Audio")
    print("-" * 40)
    silent_audio = torch.zeros(B, T, 768, device=device)
    conditions_silent = {
        'audio_features': silent_audio,
        'gaze': torch.randn(B, T, 2, device=device),
        'head_distance': torch.randn(B, T, 1, device=device),
        'emotion': torch.randn(B, T, 2, device=device),
        'blink_state': torch.randn(B, T, 3, device=device)
    }

    with torch.no_grad():
        output_silent = cond_emb(conditions_silent)

    silent_var = output_silent.var().item()
    print(f"Silent audio output variance: {silent_var:.6f}")

    # Test 2: Speech audio (random values)
    print("\nTest 2: Speech Audio")
    print("-" * 40)
    speech_audio = torch.randn(B, T, 768, device=device) * 0.5
    conditions_speech = {
        'audio_features': speech_audio,
        'gaze': torch.randn(B, T, 2, device=device),
        'head_distance': torch.randn(B, T, 1, device=device),
        'emotion': torch.randn(B, T, 2, device=device),
        'blink_state': torch.randn(B, T, 3, device=device)
    }

    with torch.no_grad():
        output_speech = cond_emb(conditions_speech)

    speech_var = output_speech.var().item()
    speech_audio_var = speech_audio.var().item()
    print(f"Speech audio input variance: {speech_audio_var:.6f}")
    print(f"Speech audio output variance: {speech_var:.6f}")

    # Test 3: Mixed batch (silent + speech)
    print("\nTest 3: Mixed Batch (Silent + Speech)")
    print("-" * 40)
    mixed_audio = torch.zeros(B, T, 768, device=device)
    mixed_audio[0] = torch.randn(T, 768, device=device) * 0.5  # Speech in first sample
    # Second sample remains silent (zeros)

    conditions_mixed = {
        'audio_features': mixed_audio,
        'gaze': torch.randn(B, T, 2, device=device),
        'head_distance': torch.randn(B, T, 1, device=device),
        'emotion': torch.randn(B, T, 2, device=device),
        'blink_state': torch.randn(B, T, 3, device=device)
    }

    with torch.no_grad():
        output_mixed = cond_emb(conditions_mixed)

    # Check variance per sample
    sample1_var = output_mixed[0].var().item()  # Speech
    sample2_var = output_mixed[1].var().item()  # Silent

    print(f"Sample 1 (speech) output variance: {sample1_var:.6f}")
    print(f"Sample 2 (silent) output variance: {sample2_var:.6f}")
    print(f"Variance ratio (speech/silent): {sample1_var/(sample2_var+1e-8):.2f}x")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY - JoyVASA Aligned (No LayerNorm):")
    print("=" * 50)

    variance_preserved = speech_var > silent_var * 2  # Speech should have significantly more variance
    distinct_samples = sample1_var > sample2_var * 2  # Mixed batch should maintain differences

    print(f"✓ Variance Preservation: {'PASSED' if variance_preserved else 'FAILED'}")
    print(f"  - Silent variance: {silent_var:.6f}")
    print(f"  - Speech variance: {speech_var:.6f}")
    print(f"  - Ratio: {speech_var/(silent_var+1e-8):.2f}x")

    print(f"\n✓ Sample Distinction: {'PASSED' if distinct_samples else 'FAILED'}")
    print(f"  - Speech sample variance: {sample1_var:.6f}")
    print(f"  - Silent sample variance: {sample2_var:.6f}")
    print(f"  - Ratio: {sample1_var/(sample2_var+1e-8):.2f}x")

    if variance_preserved and distinct_samples:
        print("\n🎉 SUCCESS: Audio variance is properly preserved!")
        print("The JoyVASA-aligned approach maintains the distinction between silent and speech audio.")
    else:
        print("\n⚠️ WARNING: Variance preservation may need adjustment")

    return variance_preserved and distinct_samples

if __name__ == "__main__":
    success = test_audio_variance_preservation()
    exit(0 if success else 1)