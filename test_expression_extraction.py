#!/usr/bin/env python3
"""Quick test to check expression extraction with debug output"""

import torch
import sys
import os
import importlib

# Set environment for debugging
os.environ['VASA_LOG_LEVEL'] = 'INFO'

sys.path.append('nemo')
from vasa_dataset import VASAIntegratedDataset
from omegaconf import OmegaConf

def test_expression_extraction():
    print("Loading volumetric avatar model...")

    # Load the volumetric avatar model
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Generator(OmegaConf.to_container(emo_config, resolve=True))

    # Load checkpoint
    model_path = emo_config.model.stage_1_checkpoint_path
    if os.path.exists(model_path):
        model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        print("Volumetric avatar loaded successfully")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}")
        return

    print("\nCreating dataset...")
    dataset = VASAIntegratedDataset(
        video_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/",
        emo_model=volumetric_avatar,
        max_videos=1,
        window_size=50,
        stride=50,
        context_size=10,
        random_seed=42,
        cache_dir='cache'
    )

    print(f"\nDataset has {len(dataset)} windows")
    print("\n" + "="*60)
    print("Extracting first window (this will trigger debug output)...")
    print("="*60 + "\n")

    # Get first window - this will trigger the expression extraction
    window_data = dataset[0]

    print("\n" + "="*60)
    print("Extraction complete. Check the [EXPRESSION DEBUG] output above.")
    print("="*60)

if __name__ == "__main__":
    test_expression_extraction()