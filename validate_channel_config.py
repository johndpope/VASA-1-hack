#!/usr/bin/env python3
"""
Validate channel_config.yaml dimensions and check for mismatches
"""

import yaml
from omegaconf import OmegaConf
import sys

def calculate_channel_dimensions(config):
    """Calculate total dimensions from channel layout"""
    
    # Manual calculation of landmark dimensions
    lips_size = config.landmarks.lips.points * config.landmarks.lips.coords  # 20*3 = 60
    right_eye_size = config.landmarks.right_eye.points * config.landmarks.right_eye.coords  # 8*3 = 24
    left_eye_size = config.landmarks.left_eye.points * config.landmarks.left_eye.coords  # 7*3 = 21
    jaw_size = config.landmarks.jaw.points * config.landmarks.jaw.coords  # 10*3 = 30
    nose_size = config.landmarks.nose.points * config.landmarks.nose.coords  # 4*3 = 12
    
    total_landmark_size = lips_size + right_eye_size + left_eye_size + jaw_size + nose_size
    
    # Calculate each channel dimension
    channels = {
        'prev_theta': 12,
        'prev_rotation': 3,
        'prev_translation': 3,
        'prev_expression': config.dimensions.prev_expression,  # 64
        'prev_audio': config.dimensions.prev_audio,  # 64
        'audio_features': config.dimensions.audio_features,  # 128
        'gaze': 2,
        'head_distance': 1,
        'emotion': 2,
        'speed_bucket': 1,
        'lips': lips_size,  # 60
        'right_eye': right_eye_size,  # 24
        'left_eye': left_eye_size,  # 21
        'jaw': jaw_size,  # 30
        'nose': nose_size,  # 12
        'blink_state': 32,
        'padding': 53  # Padding to match model_dim
    }
    
    return channels, total_landmark_size

def main():
    # Load config with OmegaConf to resolve substitutions
    config = OmegaConf.load('channel_config.yaml')
    
    print("=" * 70)
    print("CHANNEL CONFIGURATION VALIDATION")
    print("=" * 70)
    
    # Get model dimension
    model_dim = config.model.model_dim
    print(f"\nModel dimension: {model_dim}")
    
    # Calculate channel dimensions
    channels, total_landmark_size = calculate_channel_dimensions(config)
    
    print("\n" + "-" * 70)
    print("CHANNEL BREAKDOWN:")
    print("-" * 70)
    
    # Group channels by category
    prev_window_channels = ['prev_theta', 'prev_rotation', 'prev_translation', 'prev_expression', 'prev_audio']
    current_conditions = ['audio_features', 'gaze', 'head_distance', 'emotion', 'speed_bucket']
    landmark_channels = ['lips', 'right_eye', 'left_eye', 'jaw', 'nose']
    other_channels = ['blink_state', 'padding']
    
    print("\nPrevious Window Context:")
    prev_total = 0
    for ch in prev_window_channels:
        size = channels[ch]
        prev_total += size
        print(f"  {ch:20s}: {size:3d}")
    print(f"  {'Subtotal':20s}: {prev_total:3d}")
    
    print("\nCurrent Window Conditions:")
    cond_total = 0
    for ch in current_conditions:
        size = channels[ch]
        cond_total += size
        print(f"  {ch:20s}: {size:3d}")
    print(f"  {'Subtotal':20s}: {cond_total:3d}")
    
    print("\nLandmark Features:")
    landmark_total = 0
    for ch in landmark_channels:
        size = channels[ch]
        landmark_total += size
        print(f"  {ch:20s}: {size:3d}")
    print(f"  {'Subtotal':20s}: {landmark_total:3d}")
    print(f"  {'(All landmarks)':20s}: {total_landmark_size:3d}")
    
    print("\nOther Features:")
    other_total = 0
    for ch in other_channels:
        size = channels[ch]
        other_total += size
        print(f"  {ch:20s}: {size:3d}")
    print(f"  {'Subtotal':20s}: {other_total:3d}")
    
    # Calculate total
    total_channels = sum(channels.values())
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"Total channel dimensions: {total_channels}")
    print(f"Model dimension: {model_dim}")
    print(f"Difference: {model_dim - total_channels}")
    
    # Validation checks
    print("\n" + "-" * 70)
    print("VALIDATION:")
    print("-" * 70)
    
    if total_channels > model_dim:
        print(f"❌ ERROR: Total channels ({total_channels}) exceeds model_dim ({model_dim})")
        print(f"   Overflow: {total_channels - model_dim} channels")
    elif total_channels < model_dim:
        print(f"⚠️  WARNING: Total channels ({total_channels}) is less than model_dim ({model_dim})")
        print(f"   Unused capacity: {model_dim - total_channels} channels")
        print(f"   Consider adding padding or adjusting model_dim to {total_channels}")
    else:
        print(f"✅ Perfect match: Total channels == model_dim ({model_dim})")
    
    # Check projection dimensions
    print("\n" + "-" * 70)
    print("PROJECTION VALIDATION:")
    print("-" * 70)
    
    print(f"Audio projection output: {config.projections.audio.output_dim}")
    print(f"Audio features channel: {config.dimensions.audio_features}")
    
    if config.projections.audio.output_dim != config.dimensions.audio_features:
        print(f"❌ ERROR: Audio projection output ({config.projections.audio.output_dim}) != audio_features channel ({config.dimensions.audio_features})")
    else:
        print(f"✅ Audio dimensions match")
    
    # Note: landmark_norm_dim uses sum interpolation which isn't supported by OmegaConf
    # We'll calculate it manually
    print(f"\nCalculated landmark total: {total_landmark_size}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    if total_channels != model_dim:
        padding_needed = model_dim - total_channels
        if padding_needed > 0:
            print(f"1. Add a padding channel of size {padding_needed} to reach model_dim")
            print(f"2. OR adjust model_dim to {total_channels}")
        else:
            print(f"1. Reduce some channel dimensions by {-padding_needed} total")
            print(f"2. OR increase model_dim to {total_channels}")
    
    print("\n3. Consider adding this validation to your training script")
    print("4. Test config loading with OmegaConf to ensure all substitutions work")
    
    return total_channels == model_dim

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)