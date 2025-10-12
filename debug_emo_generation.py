#!/usr/bin/env python3
"""
Debug EMO frame generation to see what values are produced
"""

import torch
import numpy as np
from pathlib import Path
import logging
from omegaconf import OmegaConf
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_emo_generation():
    """Test EMO frame generation directly"""

    # Load volumetric avatar model
    logger.info("Loading volumetric avatar model...")
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)

    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return

    model_dict = torch.load(model_path, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    logger.info("✅ Model loaded")

    # Load identity image
    from PIL import Image
    from torchvision import transforms

    identity_path = "nemo/data/IMG_1.png"
    if not Path(identity_path).exists():
        logger.error(f"Identity image not found: {identity_path}")
        return

    img = Image.open(identity_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    identity_image = transform(img).unsqueeze(0).cuda()
    logger.info(f"✅ Loaded identity image: {identity_image.shape}")
    logger.info(f"   Value range: [{identity_image.min():.3f}, {identity_image.max():.3f}]")

    # Create dummy source frames (random)
    source_frames = torch.rand(1, 5, 3, 512, 512).cuda()
    logger.info(f"✅ Created source frames: {source_frames.shape}")
    logger.info(f"   Value range: [{source_frames.min():.3f}, {source_frames.max():.3f}]")

    # Generate EMO frames using volumetric avatar
    logger.info("\n🎬 Generating EMO frames...")

    with torch.no_grad():
        try:
            # Use volumetric avatar bridge
            from vasa_va_bridge import VASAVolumetricAvatarBridge
            va_bridge = VASAVolumetricAvatarBridge(volumetric_avatar)

            # Generate frames
            emo_frames = va_bridge.generate_frames(
                identity_image=identity_image,
                source_frames=source_frames,
                num_frames=5
            )

            logger.info(f"✅ EMO frames generated: {emo_frames.shape}")
            logger.info(f"   Value range: [{emo_frames.min():.3f}, {emo_frames.max():.3f}]")
            logger.info(f"   Mean: {emo_frames.mean():.3f}")
            logger.info(f"   Std: {emo_frames.std():.3f}")

            # Check if frames are all zeros
            if emo_frames.max() == 0:
                logger.error("❌ EMO frames are all zeros!")
            elif emo_frames.max() < 0.01:
                logger.warning(f"⚠️ EMO frames are very dark (max={emo_frames.max():.6f})")
            else:
                logger.info("✅ EMO frames have valid values")

            # Test saving a frame
            test_frame = emo_frames[0, 0]  # First frame [C, H, W]
            test_frame_np = test_frame.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]

            logger.info(f"\n📸 Test frame for saving:")
            logger.info(f"   Shape: {test_frame_np.shape}")
            logger.info(f"   Dtype: {test_frame_np.dtype}")
            logger.info(f"   Range: [{test_frame_np.min():.3f}, {test_frame_np.max():.3f}]")

            # Simulate the normalization logic
            if test_frame_np.dtype == np.float32 or test_frame_np.dtype == np.float64:
                frame_min, frame_max = test_frame_np.min(), test_frame_np.max()
                logger.info(f"   Frame range: [{frame_min:.3f}, {frame_max:.3f}]")

                if frame_min >= 0.0 and frame_max <= 1.0:
                    logger.info("   → Would use [0, 1] normalization")
                    normalized = (test_frame_np * 255).astype(np.uint8)
                elif frame_min >= -1.0 and frame_max <= 1.0:
                    logger.info("   → Would use [-1, 1] normalization")
                    normalized = ((test_frame_np + 1.0) * 127.5).astype(np.uint8)
                else:
                    logger.info("   → Would use min-max normalization")
                    normalized = ((test_frame_np - frame_min) / (frame_max - frame_min + 1e-8) * 255).astype(np.uint8)

                logger.info(f"   Normalized range: [{normalized.min()}, {normalized.max()}]")

                if normalized.max() == 0:
                    logger.error("   ❌ Normalized frame is all zeros!")
                else:
                    logger.info("   ✅ Normalized frame has valid values")

        except Exception as e:
            logger.error(f"❌ Error generating EMO frames: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    test_emo_generation()
