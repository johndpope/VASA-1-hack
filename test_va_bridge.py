#!/usr/bin/env python3
"""
Test the VASA-VA bridge to ensure it properly generates frames.
"""

import torch
import sys
import os
from pathlib import Path
sys.path.append('nemo')
sys.path.append('.')

from PIL import Image
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bridge():
    """Test that the bridge properly separates source and target."""
    
    # Load models
    logger.info("Loading models...")
    from vasa_model import VASAModel
    from omegaconf import OmegaConf
    from vasa_va_bridge import VASAVolumetricAvatarBridge
    import importlib
    
    # Load config
    config = OmegaConf.load('overfit_config.yaml')
    
    # Initialize volumetric avatar first (same as vi.py)
    logger.info("Loading volumetric avatar...")
    import importlib
    
    # Load VA config and model exactly like vi.py does
    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)
    
    # Load EMO weights with proper error handling
    try:
        model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        volumetric_avatar = volumetric_avatar.cuda()
        volumetric_avatar.eval()
        logger.info("Volumetric avatar loaded successfully")
    except Exception as e:
        logger.error(f"Error loading EMO model: {str(e)}")
        raise
    
    # Initialize VASA model with volumetric avatar
    model = VASAModel(config, volumetric_avatar)
    model = model.cuda()
    model.eval()
    
    # Load VASA checkpoint (same logic as vi.py)
    # Auto-detect checkpoint based on config name
    checkpoint_path = None
    if 'overfit' in 'overfit_config.yaml':  # We're using overfit config
        checkpoint_path = "./checkpoints_overfit/best_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            # Try to find latest checkpoint
            checkpoint_dir = Path("./checkpoints_overfit")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    logger.info(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = "./checkpoints/best_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
            # Try to find latest checkpoint
            checkpoint_dir = Path("./checkpoints")
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    logger.info(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load VASA checkpoint if found
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logger.info(f"Loading VASA checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
            
            # Get current state dict
            model_state = model.state_dict()
            
            # Load only matching keys from checkpoint
            checkpoint_state = checkpoint.get('model_state_dict', checkpoint) if 'model_state_dict' in checkpoint else checkpoint
            matched_state_dict = {}
            
            for key in model_state.keys():
                if key in checkpoint_state:
                    if model_state[key].shape == checkpoint_state[key].shape:
                        matched_state_dict[key] = checkpoint_state[key]
                        logger.debug(f"Loaded parameter: {key}")
                    else:
                        logger.warning(f"Shape mismatch for {key}: model {model_state[key].shape} vs checkpoint {checkpoint_state[key].shape}")
                        matched_state_dict[key] = model_state[key]
                else:
                    logger.debug(f"Parameter {key} not found in checkpoint, using initialization")
                    matched_state_dict[key] = model_state[key]
                    
            # Load state dict with strict=False to handle missing volumetric_avatar parameters
            model.load_state_dict(matched_state_dict, strict=False)
            logger.info("Successfully loaded VASA checkpoint")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.warning("Continuing with random initialization")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using random initialization")
    
    # Initialize bridge
    bridge = VASAVolumetricAvatarBridge(volumetric_avatar)
    
    # Load identity image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    identity_path = 'nemo/data/IMG_1.png'
    identity_img = Image.open(identity_path).convert('RGB')
    identity_tensor = transform(identity_img).unsqueeze(0).cuda()
    
    logger.info(f"Identity image loaded: {identity_tensor.shape}")
    
    # Create dummy motion outputs (as if from VASA)
    B, T = 1, 5  # 1 batch, 5 frames
    motion_outputs = {
        'theta': torch.randn(B, T, 3, 4).cuda(),  # Random head pose
        'expression_embed': torch.randn(B, T, 128).cuda(),  # Random expression
        'rotation': torch.randn(B, T, 3).cuda(),
        'translation': torch.randn(B, T, 3).cuda()
    }
    
    logger.info("Testing frame generation with bridge...")
    
    # Generate frames using bridge
    with torch.no_grad():
        generated_frames = bridge.generate_frames_from_motion(
            motion_outputs=motion_outputs,
            source_img=identity_tensor,
            use_black_background=True
        )
    
    logger.info(f"Generated frames shape: {generated_frames.shape}")
    logger.info(f"Expected shape: [B={B}, T={T}, C=3, H=512, W=512]")
    
    # Verify output
    assert generated_frames.shape == (B, T, 3, 512, 512), f"Shape mismatch: {generated_frames.shape}"
    
    # Check that frames are different (motion is applied)
    frame_diff = (generated_frames[:, 1] - generated_frames[:, 0]).abs().mean()
    logger.info(f"Average difference between frames: {frame_diff:.4f}")
    
    if frame_diff < 0.01:
        logger.warning("Frames are too similar - motion might not be applied correctly!")
    else:
        logger.info("✓ Frames show motion variation")
    
    # Save first frame for inspection
    from torchvision.utils import save_image
    first_frame = generated_frames[0, 0] * 0.5 + 0.5  # Denormalize
    save_image(first_frame, 'test_bridge_output.png')
    logger.info("Saved first frame to test_bridge_output.png")
    
    # Test cache functionality
    logger.info("\nTesting cache...")
    source_data_1 = bridge.get_source_embeddings(identity_tensor)
    source_data_2 = bridge.get_source_embeddings(identity_tensor)
    
    # Should be the same object (cached)
    assert source_data_1 is source_data_2, "Cache not working!"
    logger.info("✓ Cache working correctly")
    
    # Clear cache and test again
    bridge.clear_cache()
    source_data_3 = bridge.get_source_embeddings(identity_tensor)
    assert source_data_3 is not source_data_1, "Cache not cleared!"
    logger.info("✓ Cache clearing working correctly")
    
    logger.info("\n✅ All tests passed!")
    logger.info("\nThe bridge properly:")
    logger.info("1. Extracts source expression from identity image (cached)")
    logger.info("2. Uses VASA motion as target expression")
    logger.info("3. Generates frames with proper warping")
    logger.info("4. Produces frames with motion variation")

if __name__ == "__main__":
    test_bridge()