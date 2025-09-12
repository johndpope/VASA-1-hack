#!/usr/bin/env python3
"""
Test script to verify audio processing improvements
"""
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def test_audio_processing():
    print("=== Testing Audio Processing ===")

    # Load a sample audio file
    audio_path = "junk/11.mp4"  # Assuming this exists
    if not Path(audio_path).exists():
        print(f"Audio file {audio_path} not found, creating synthetic audio")
        # Create synthetic audio with some variation
        sample_rate = 16000
        duration = 2.0  # 2 seconds
        num_samples = int(sample_rate * duration)

        # Create a signal with varying frequency to test feature extraction
        t = torch.linspace(0, duration, num_samples)
        frequency = 200 + 100 * torch.sin(2 * np.pi * 0.5 * t)  # Varying frequency
        audio = torch.sin(2 * np.pi * frequency * t)
        audio = audio.unsqueeze(0)  # Add channel dimension
    else:
        # Load real audio
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sample_rate}")

    # Initialize Wav2Vec
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').cuda().eval()

    # Process audio
    inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to('cuda'))
        features = outputs.last_hidden_state

    print(f"Raw Wav2Vec features shape: {features.shape}")
    print(f"Raw features variance: {features.var().item():.6f}")
    print(f"Raw features mean: {features.mean().item():.6f}")

    # Apply normalization (our improvement)
    features_normalized = (features - features.mean(dim=-1, keepdim=True)) / (features.std(dim=-1, keepdim=True) + 1e-8)
    features_normalized = features_normalized * 2.0  # Scale up

    print(f"Normalized features variance: {features_normalized.var().item():.6f}")
    print(f"Normalized features mean: {features_normalized.mean().item():.6f}")

    # Interpolate to window size
    window_size = 50
    features_interp = F.interpolate(
        features_normalized.transpose(1, 2),  # -> [1, D, L]
        size=window_size,
        mode='linear'
    ).transpose(1, 2)  # -> [1, T, D]

    print(f"Interpolated features shape: {features_interp.shape}")
    print(f"Final features variance: {features_interp.var().item():.6f}")
    print(f"Final features mean: {features_interp.mean().item():.6f}")

    # Test temporal variation
    temporal_var = features_interp.var(dim=-1).mean().item()
    print(f"Temporal variation (variance across time): {temporal_var:.6f}")

    if temporal_var < 0.1:
        print("WARNING: Low temporal variation detected!")
    else:
        print("GOOD: Sufficient temporal variation detected!")

if __name__ == "__main__":
    test_audio_processing()