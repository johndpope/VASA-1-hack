#!/usr/bin/env python3
"""Quick test to verify the component contribution calculation fix."""

import torch
import numpy as np

# Simulate the old (incorrect) calculation
def old_calculation():
    # Create sample features with different variances
    audio_features = torch.randn(1, 10, 512) * 0.5  # Lower variance
    controls_features = torch.randn(1, 10, 256) * 1.5  # Higher variance
    blink_features = torch.randn(1, 10, 128) * 1.0  # Medium variance

    # Concatenate
    combined = torch.cat([audio_features, controls_features, blink_features], dim=-1)

    # Old (incorrect) calculation
    audio_contrib = audio_features.var().item() / (combined.var().item() + 1e-8)
    controls_contrib = controls_features.var().item() / (combined.var().item() + 1e-8)
    blink_contrib = blink_features.var().item() / (combined.var().item() + 1e-8)

    total = audio_contrib + controls_contrib + blink_contrib

    print("Old (incorrect) calculation:")
    print(f"  Audio: {audio_contrib:.2%}")
    print(f"  Controls: {controls_contrib:.2%}")
    print(f"  Blink: {blink_contrib:.2%}")
    print(f"  Total: {total:.2%} (should be 100%!)")

    return audio_contrib, controls_contrib, blink_contrib

# Simulate the new (correct) calculation
def new_calculation():
    # Create sample features with different variances
    audio_features = torch.randn(1, 10, 512) * 0.5  # Lower variance
    controls_features = torch.randn(1, 10, 256) * 1.5  # Higher variance
    blink_features = torch.randn(1, 10, 128) * 1.0  # Medium variance

    # New (correct) calculation
    audio_var = audio_features.var().item()
    controls_var = controls_features.var().item()
    blink_var = blink_features.var().item()
    total_var = audio_var + controls_var + blink_var + 1e-8

    audio_contrib = audio_var / total_var
    controls_contrib = controls_var / total_var
    blink_contrib = blink_var / total_var

    total = audio_contrib + controls_contrib + blink_contrib

    print("\nNew (correct) calculation:")
    print(f"  Audio: {audio_contrib:.2%}")
    print(f"  Controls: {controls_contrib:.2%}")
    print(f"  Blink: {blink_contrib:.2%}")
    print(f"  Total: {total:.2%} (should be exactly 100%)")

    return audio_contrib, controls_contrib, blink_contrib

if __name__ == "__main__":
    print("Testing component contribution calculations...")
    print("=" * 50)

    # Set seed for reproducibility
    torch.manual_seed(42)
    old_calculation()

    torch.manual_seed(42)
    new_calculation()

    print("\n" + "=" * 50)
    print("✓ The new calculation ensures contributions sum to 100%")