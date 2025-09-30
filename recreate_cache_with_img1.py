#!/usr/bin/env python3
"""
Recreate the cache using IMG_1.png as the identity source.
This will fix the identity confusion issue.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import h5py
import logging
import importlib
from omegaconf import OmegaConf
import cv2

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recreate_cache_with_correct_identity():
    """Recreate cache with IMG_1.png as identity."""

    logger.info("Loading model...")
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    model = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        model.load_state_dict(model_dict, strict=False)

    model = model.cuda()
    model.eval()

    # Load IMG_1.png - the CORRECT identity source (the man)
    logger.info("Loading IMG_1.png as identity source...")
    img1 = Image.open("nemo/data/IMG_1.png").convert('RGB')
    img1_np = np.array(img1)

    # Resize to 512x512
    img1_resized = cv2.resize(img1_np, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor
    img1_tensor = torch.from_numpy(img1_resized).float() / 127.5 - 1.0
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    logger.info(f"IMG_1 tensor shape: {img1_tensor.shape}")

    # Extract identity from IMG_1
    with torch.no_grad():
        # Get face mask
        face_mask, _, _, _ = model.face_idt.forward(img1_tensor)
        face_mask = (face_mask > 0.6).float()

        # Mask image
        masked = img1_tensor * face_mask

        # Extract identity embedding
        idt_embed = model.idt_embedder_nw(masked)

        logger.info(f"Identity embedding shape: {idt_embed.shape}")
        logger.info(f"Face mask coverage: {face_mask.mean().item():.3f}")

    # Load existing cache to get the frames (but not the identity)
    old_cache_path = Path("proper_face_attributes.h5")
    new_cache_path = Path("proper_face_attributes_img1.h5")

    if old_cache_path.exists():
        logger.info("Loading frames from old cache...")

        # Read old cache
        with h5py.File(old_cache_path, 'r') as old_f:
            num_frames = old_f.attrs['num_frames']

            # Create new cache with IMG_1 identity
            with h5py.File(new_cache_path, 'w') as new_f:
                # Save the CORRECT identity (IMG_1)
                new_f.create_dataset('identity_frame', data=img1_tensor.cpu().numpy())
                new_f.create_dataset('identity_embed', data=idt_embed.cpu().numpy())
                new_f.create_dataset('identity_mask', data=face_mask.cpu().numpy())

                # Copy frame data from old cache
                for i in range(num_frames):
                    frame_key = f'frame_{i:04d}'
                    if frame_key in old_f:
                        logger.info(f"Copying {frame_key}")
                        old_group = old_f[frame_key]
                        new_group = new_f.create_group(frame_key)

                        # Copy all datasets in the frame group
                        for key in old_group.keys():
                            new_group.create_dataset(key, data=old_group[key][:])

                # Set attributes
                new_f.attrs['num_frames'] = num_frames
                new_f.attrs['identity_source'] = 'IMG_1.png'

        logger.info(f"\nCreated new cache: {new_cache_path}")
        logger.info(f"Identity: IMG_1.png (the man)")
        logger.info(f"Frames: {num_frames} frames from video")

        # Verify the new cache
        logger.info("\nVerifying new cache...")
        with h5py.File(new_cache_path, 'r') as f:
            cached_id = f['identity_frame'][:]
            logger.info(f"New cached identity shape: {cached_id.shape}")
            logger.info(f"New cached identity range: [{cached_id.min()}, {cached_id.max()}]")

            # Save a preview
            if len(cached_id.shape) == 4:
                cached_id = cached_id[0]
            if cached_id.shape[0] == 3:
                cached_id = np.transpose(cached_id, (1, 2, 0))
            if cached_id.min() < 0:
                cached_id = (cached_id + 1) / 2
            cached_id = np.clip(cached_id, 0, 1)

            preview = Image.fromarray((cached_id * 255).astype(np.uint8))
            preview.save("new_cache_identity_preview.png")
            logger.info("Saved new_cache_identity_preview.png")

    else:
        logger.error("Old cache not found! Cannot copy frames.")
        return

    logger.info("\n" + "="*60)
    logger.info("SUCCESS! New cache created with correct identity")
    logger.info(f"File: {new_cache_path}")
    logger.info("Identity: IMG_1.png (the man)")
    logger.info("Now use this cache for face swapping!")
    logger.info("="*60)


if __name__ == "__main__":
    recreate_cache_with_correct_identity()