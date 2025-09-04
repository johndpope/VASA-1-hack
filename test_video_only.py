#!/usr/bin/env python3
"""Test just the video processing"""

import sys
sys.path.insert(0, 'nemo')
from vi_complete import CompleteVASAInference
from pathlib import Path
import time

video_path = './junk/10.mp4'

if not Path(video_path).exists():
    print(f"⚠️  Test video not found: {video_path}")
    sys.exit(1)

print("\nTesting real video processing...")

# Test with minimal config
inferencer = CompleteVASAInference(
    checkpoint_path='checkpoints/tdd_wandb/best_model.pth',
    config_path='vasa_config_fixed.yaml',
    num_inference_steps=5,  # Very fast
    use_audio_tdd=False,
    test_lip_sync=False
)

timestamp = int(time.time())
output_path = f"vasa-test-quick-{timestamp}.mp4"

print(f"Input: {video_path}")
print(f"Output: {output_path}")
print("Starting generation...")

try:
    inferencer.generate_from_video_improved(
        input_video=video_path,
        output_path=output_path,
        fps=25.0,
        test_stages=False
    )
    print(f"✅ Success! Video saved as {output_path}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()