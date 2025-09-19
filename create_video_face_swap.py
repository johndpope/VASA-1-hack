#!/usr/bin/env python3
"""
Create the face swap result that matches the video you showed.
Using IMG_1.png as identity with proper pipeline flow.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Tuple
import h5py

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_volumetric_model():
    """Load the volumetric avatar model."""
    logger.info("Loading volumetric avatar model...")

    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        logger.info("Model loaded successfully")

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def load_image_tensor(image_path: str) -> torch.Tensor:
    """Load image as tensor in [-1, 1] range."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    if img.shape[:2] != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    return img_tensor


def extract_identity_features(model, source_img: torch.Tensor) -> Dict:
    """Extract identity features from source image."""
    with torch.no_grad():
        # Get face mask with higher quality
        face_mask, _, _, _ = model.face_idt.forward(source_img)
        face_mask = (face_mask > 0.6).float()

        # Apply Gaussian smoothing to mask
        face_mask = F.avg_pool2d(face_mask, 3, stride=1, padding=1)

        # Mask source
        masked_source = source_img * face_mask

        # Extract identity embedding
        idt_embed = model.idt_embedder_nw(masked_source)

        # Get head pose
        source_theta, _, _, _ = model.head_pose_regressor.forward(source_img, True)

        # Prepare data dict
        data_dict = {
            'source_img': source_img,
            'source_mask': face_mask,
            'source_theta': source_theta,
            'target_img': source_img,
            'target_mask': face_mask,
            'target_theta': source_theta,
            'idt_embed': idt_embed
        }

        # Get expression embedding
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)

        # Get warp embeddings
        source_warp_embed, _, _, embed_dict = model.predict_embed(data_dict)

        # Generate XY warps for source
        source_xy_warp, _ = model.xy_generator_nw(source_warp_embed)

        # Extract source volume
        source_latents = model.local_encoder_nw(masked_source)

        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        source_volume = source_latents.view(1, c, d, s, s)

        # Process source volume
        if model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)

        # Apply source rotation and XY warp to get canonical
        grid = model.identity_grid_3d[:1]
        source_rot_warp = grid.bmm(source_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)

        rotated_source = model.grid_sample(source_volume, source_rot_warp)
        canonical_volume = model.grid_sample(rotated_source, source_xy_warp)

        # Process through volume network with identity
        processed_canonical = model.volume_process_nw(canonical_volume, embed_dict)

        return {
            'idt_embed': idt_embed,
            'embed_dict': embed_dict,
            'canonical_volume': processed_canonical,
            'source_theta': source_theta,
            'source_mask': face_mask
        }


def apply_target_to_identity(model, identity_info: Dict, target_img: torch.Tensor) -> torch.Tensor:
    """Apply target expression/pose to source identity."""
    with torch.no_grad():
        # Get target face mask
        target_mask, _, _, _ = model.face_idt.forward(target_img)
        target_mask = (target_mask > 0.6).float()
        target_mask = F.avg_pool2d(target_mask, 3, stride=1, padding=1)

        # Get target pose
        target_theta, _, _, _ = model.head_pose_regressor.forward(target_img, True)

        # Create target data dict WITH SOURCE IDENTITY
        data_dict = {
            'source_img': target_img,
            'source_mask': target_mask,
            'source_theta': target_theta,
            'target_img': target_img,
            'target_mask': target_mask,
            'target_theta': target_theta,
            'idt_embed': identity_info['idt_embed']  # SOURCE IDENTITY!
        }

        # Get target expression
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        target_pose_embed = data_dict['source_pose_embed']

        # Get target warps
        _, target_warp_embed, _, _ = model.predict_embed(data_dict)

        # Generate UV warps for target expression
        target_uv_warp, _ = model.uv_generator_nw(target_warp_embed)

        # Apply UV warp to canonical volume
        volume_with_expression = model.grid_sample(identity_info['canonical_volume'], target_uv_warp)

        # Apply target rotation
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        grid = model.identity_grid_3d[:1]
        target_rot_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        final_volume = model.grid_sample(volume_with_expression, target_rot_warp)

        # Decode
        target_latent_feats = final_volume.view(1, c * d, s, s)

        decode_dict = {
            'target_theta': target_theta,
            'target_pose_embed': target_pose_embed
        }

        generated_img, _, _, _ = model.decoder_nw(
            decode_dict,
            identity_info['embed_dict'],  # SOURCE IDENTITY!
            target_latent_feats,
            False,
            stage_two=True
        )

        # Fix range if needed (decoder outputs [0,1])
        if generated_img.min() >= 0 and generated_img.max() <= 1.1:
            generated_img = generated_img * 2 - 1

        # Get refined mask for compositing
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        gen_mask = (gen_mask > 0.65).float()

        # Smooth mask edges heavily for seamless blend
        for _ in range(3):
            gen_mask = F.avg_pool2d(gen_mask, 3, stride=1, padding=1)

        # Final composite
        background = target_img * (1 - gen_mask)
        final_img = generated_img * gen_mask + background

        # Ensure proper range
        final_img = torch.clamp(final_img, -1, 1)

        return final_img


def main():
    """Create the face swap to match the video."""

    # Load model
    model = load_volumetric_model()

    # Load IMG_1.png as identity source
    source_img = load_image_tensor("nemo/data/IMG_1.png")
    logger.info("Loaded IMG_1.png as identity source")

    # Extract identity features once
    logger.info("Extracting identity features...")
    identity_info = extract_identity_features(model, source_img)

    # Load target frames
    targets = []

    # First try video frames from cache - use the CORRECTED cache with IMG_1 identity
    cache_path = Path("proper_face_attributes_img1.h5")
    if cache_path.exists():
        with h5py.File(cache_path, 'r') as f:
            # Get frames around the middle (best expressions)
            for i in [3, 4, 5, 6, 7]:
                if f'frame_{i:04d}' in f:
                    frame_tensor = torch.from_numpy(f[f'frame_{i:04d}/frame'][:]).cuda()
                    targets.append((f"Frame_{i}", frame_tensor))
                    logger.info(f"Loaded cached frame {i}")

    # Also try static images
    for img_name in ["IMG_2.png", "IMG_3.png", "IMG_4.png"]:
        img_path = Path(f"nemo/data/{img_name}")
        if img_path.exists():
            img_tensor = load_image_tensor(str(img_path))
            targets.append((img_name, img_tensor))
            logger.info(f"Loaded {img_name}")

    if not targets:
        logger.error("No target images found!")
        return

    # Process each target
    results = []
    for name, target_img in targets:
        logger.info(f"Processing {name}...")

        # Apply target to identity
        final_img = apply_target_to_identity(model, identity_info, target_img)

        results.append({
            'name': name,
            'target': target_img,
            'result': final_img
        })

    # Visualize results
    n_cols = min(len(results), 6)
    fig, axes = plt.subplots(3, n_cols + 1, figsize=(3 * (n_cols + 1), 9))

    # Show identity source
    source_display = (source_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
    axes[0, 0].imshow(np.clip(source_display, 0, 1))
    axes[0, 0].set_title("Identity\n(IMG_1)", fontsize=10, weight='bold', color='blue')
    axes[0, 0].axis('off')

    axes[1, 0].text(0.5, 0.5, 'Source\nIdentity', ha='center', va='center',
                   fontsize=11, weight='bold', color='blue')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    # Show results
    for i in range(n_cols):
        col = i + 1
        result = results[i]

        # Target
        target_display = (result['target'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[0, col].imshow(np.clip(target_display, 0, 1))
        axes[0, col].set_title(f"{result['name']}", fontsize=9)
        axes[0, col].axis('off')

        # Result
        result_display = (result['result'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[1, col].imshow(np.clip(result_display, 0, 1))
        axes[1, col].set_title("Swapped", fontsize=9, weight='bold', color='green')
        axes[1, col].axis('off')

        # Save individual result
        result_img = (np.clip(result_display, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(result_img).save(f"video_swap_{result['name']}.png")

        # Show difference/quality
        diff = np.abs(result_display - target_display).mean(axis=2)
        axes[2, col].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title("Difference", fontsize=9)
        axes[2, col].axis('off')

    plt.suptitle("Face Swap Matching Video Result", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("video_matching_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\nSaved video_matching_results.png")

    # Save best single result
    if len(results) >= 3:
        best = results[2]  # Frame 5 usually has good expression
        best_img = (best['result'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        best_img = (np.clip(best_img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(best_img).save("video_best_result.png")
        logger.info("Saved video_best_result.png")

    logger.info(f"\nGenerated {len(results)} face swap results")
    logger.info("Results should match the video quality!")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Creating Video-Matching Face Swap")
    logger.info("Using IMG_1.png identity with proper pipeline")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Complete!")
    logger.info("=" * 60)