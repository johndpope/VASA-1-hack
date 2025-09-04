#!/usr/bin/env python3
"""
Test TDD model generation on training video
"""

import torch
import sys
sys.path.insert(0, 'nemo')
from vi import VASAInference
import time

print("\n" + "="*60)
print("Testing TDD Model Video Generation")
print("="*60)

# Test with different checkpoints
checkpoints = [
    ("TDD Model", "checkpoints/tdd_wandb/best_model.pth"),
    ("Overfitted Model", "checkpoints/overfitted/model.pth"),
]

for name, checkpoint_path in checkpoints:
    print(f"\n📹 Testing: {name}")
    print(f"   Checkpoint: {checkpoint_path}")
    print("-" * 40)
    
    try:
        # Check if checkpoint exists
        from pathlib import Path
        if not Path(checkpoint_path).exists():
            print(f"   ❌ Checkpoint not found, skipping")
            continue
            
        # Initialize inferencer
        inferencer = VASAInference(
            checkpoint_path=checkpoint_path,
            config_path='vasa_config_fixed.yaml'
        )
        
        # Generate output with unique filename
        timestamp = int(time.time())
        output_path = f"vasa-output-{name.lower().replace(' ', '_')}_{timestamp}.mp4"
        
        print(f"   Generating video: {output_path}")
        
        # Generate from training video
        inferencer.generate_from_video(
            input_video="./junk/10.mp4",
            output_path=output_path,
            fps=25.0,
            neutral_expression=False
        )
        
        print(f"   ✅ Generated: {output_path}")
        
        # Check file size
        import os
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024 / 1024  # MB
            print(f"   File size: {size:.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Comparison Complete")
print("="*60)
print("\nTo view the differences:")
print("  - TDD model: Should have learned motion patterns")
print("  - Overfitted model: Should perfectly recreate training video")
print("\nKey differences to look for:")
print("  - Motion quality and smoothness")
print("  - Expression variation")
print("  - Audio synchronization")
print("  - Overall fidelity to original")