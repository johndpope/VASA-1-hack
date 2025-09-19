#!/usr/bin/env python3
"""
Test proper face swap following exact pipeline2.py logic.
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

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_volumetric_model():
    """Load the volumetric avatar model."""
    logger.info("Loading volumetric avatar model...")

    # Load config
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')

    # Initialize model
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Load weights
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        logger.info("Model weights loaded successfully")

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image to tensor."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)

    # Resize to 512x512
    if img.shape[:2] != (512, 512):
        img = cv2.resize(img, (512, 512))

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    return img_tensor


def process_source_identity(model, source_img: torch.Tensor) -> Tuple[Dict, Dict]:
    """
    Process source image to extract identity following pipeline2.py logic.
    Returns embed_dict and source volume info.
    """
    with torch.no_grad():
        # Get face mask
        face_mask, _, _, _ = model.face_idt.forward(source_img)
        face_mask = (face_mask > 0.6).float()

        # Mask source
        masked_source = source_img * face_mask

        # Get identity embedding
        idt_embed = model.idt_embedder_nw(masked_source)

        # Get head pose
        pred_theta, _, _, _ = model.head_pose_regressor.forward(source_img, True)

        # Create data dict for source
        data_dict = {
            'source_img': source_img,
            'source_mask': face_mask,
            'source_theta': pred_theta,
            'target_img': source_img,
            'target_mask': face_mask,
            'target_theta': pred_theta,
            'idt_embed': idt_embed
        }

        # Get expression embedding
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        source_pose_embed = data_dict['source_pose_embed']

        # Get warp embeddings
        source_warp_embed_dict, _, _, embed_dict = model.predict_embed(data_dict)

        # Generate XY warps
        source_xy_warp, _ = model.xy_generator_nw(source_warp_embed_dict)

        # Get source latents
        source_latents = model.local_encoder_nw(masked_source)

        # Create source volume
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size
        source_volume = source_latents.view(1, c, d, s, s)

        # Process source volume
        if model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)

        # Create rotation warp for source
        grid = model.identity_grid_3d[:1]
        source_rotation_warp = grid.bmm(pred_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)

        # Apply rotations and XY warp to get canonical
        rotated_source = model.grid_sample(source_volume, source_rotation_warp)
        canonical_volume = model.grid_sample(rotated_source, source_xy_warp)

        # Process through volume network with identity
        processed_canonical = model.volume_process_nw(canonical_volume, embed_dict)

        return {
            'embed_dict': embed_dict,
            'idt_embed': idt_embed,
            'source_volume': source_volume,
            'processed_canonical': processed_canonical,
            'source_theta': pred_theta,
            'source_xy_warp': source_xy_warp
        }


def apply_target_expression(model, source_info: Dict, target_img: torch.Tensor) -> torch.Tensor:
    """
    Apply target expression to source identity following pipeline2.py logic.
    """
    with torch.no_grad():
        # Get target mask
        face_mask, _, _, _ = model.face_idt.forward(target_img)
        face_mask = (face_mask > 0.6).float()

        # Get target head pose
        pred_target_theta, _, _, _ = model.head_pose_regressor.forward(target_img, True)

        # Create data dict for target
        data_dict = {
            'source_img': target_img,
            'source_mask': face_mask,
            'source_theta': pred_target_theta,
            'target_img': target_img,
            'target_mask': face_mask,
            'target_theta': pred_target_theta,
            'idt_embed': source_info['idt_embed']  # Use source identity!
        }

        # Get target expression
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        target_pose_embed = data_dict['source_pose_embed']  # Note: it's called source_pose_embed

        # Get target warp embeddings
        _, target_warp_embed_dict, _, _ = model.predict_embed(data_dict)

        # Generate UV warps for target expression
        target_uv_warp, _ = model.uv_generator_nw(target_warp_embed_dict)

        # Apply UV warp to processed canonical volume
        target_volume_with_expression = model.grid_sample(source_info['processed_canonical'], target_uv_warp)

        # Apply target rotation
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        grid = model.identity_grid_3d[:1]
        target_rotation_warp = grid.bmm(pred_target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        final_target_volume = model.grid_sample(target_volume_with_expression, target_rotation_warp)

        # Prepare for decoder
        target_latent_feats = final_target_volume.view(1, c * d, s, s)

        # Decode with source identity
        decode_data_dict = {
            'target_theta': pred_target_theta,
            'target_pose_embed': target_pose_embed
        }

        generated_img, _, _, _ = model.decoder_nw(
            decode_data_dict,
            source_info['embed_dict'],  # Use source embed_dict with identity!
            target_latent_feats,
            False,
            stage_two=True
        )

        # Apply face mask for final composite
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        gen_mask = (gen_mask > 0.6).float()

        # Composite
        background = target_img * (1 - gen_mask)
        final_img = generated_img * gen_mask + background

        return final_img, generated_img


def main():
    """Test proper face swap."""

    # Load model
    model = load_volumetric_model()

    # Load source (IMG_1.png)
    source_img = load_and_preprocess_image("nemo/data/IMG_1.png")
    logger.info("Loaded source image (IMG_1.png)")

    # Process source to extract identity
    logger.info("Extracting source identity...")
    source_info = process_source_identity(model, source_img)

    # Load target images
    target_paths = [
        "nemo/data/IMG_2.png",
        "nemo/data/IMG_3.png",
        "nemo/data/IMG_4.png"
    ]

    results = []
    for target_path in target_paths:
        if Path(target_path).exists():
            logger.info(f"Processing target: {target_path}")
            target_img = load_and_preprocess_image(target_path)

            # Apply target expression
            final_img, raw_img = apply_target_expression(model, source_info, target_img)

            results.append({
                'target_path': target_path,
                'target_img': target_img,
                'final': final_img,
                'raw': raw_img
            })

    # Visualize results
    if results:
        fig, axes = plt.subplots(3, len(results) + 1, figsize=(3 * (len(results) + 1), 9))

        # Show source
        source_display = source_img[0].cpu().permute(1, 2, 0).numpy()
        source_display = (source_display + 1) / 2

        axes[0, 0].imshow(source_display)
        axes[0, 0].set_title("Source\n(IMG_1.png)", fontsize=10, weight='bold', color='blue')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        for i, result in enumerate(results):
            col = i + 1

            # Target
            target_display = result['target_img'][0].cpu().permute(1, 2, 0).numpy()
            target_display = (target_display + 1) / 2
            axes[0, col].imshow(target_display)
            axes[0, col].set_title(f"Target\n({Path(result['target_path']).name})", fontsize=10)
            axes[0, col].axis('off')

            # Raw generated
            raw_display = result['raw'][0].cpu().permute(1, 2, 0).numpy()
            raw_display = (raw_display + 1) / 2
            axes[1, col].imshow(raw_display)
            axes[1, col].set_title("Generated\n(Raw)", fontsize=10)
            axes[1, col].axis('off')

            # Final
            final_display = result['final'][0].cpu().permute(1, 2, 0).numpy()
            final_display = (final_display + 1) / 2
            axes[2, col].imshow(final_display)
            axes[2, col].set_title("Final\n(Composite)", fontsize=10, color='green')
            axes[2, col].axis('off')

        plt.suptitle("Proper Face Swap Test (Following Pipeline2 Logic)", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("proper_face_swap_test.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved results to proper_face_swap_test.png")

        # Save best result separately
        if results:
            best = results[0]
            best_img = best['final'][0].cpu().permute(1, 2, 0).numpy()
            best_img = (best_img + 1) / 2
            best_img = (best_img * 255).astype(np.uint8)
            Image.fromarray(best_img).save("proper_face_swap_result.png")
            logger.info("Saved best result to proper_face_swap_result.png")

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Proper Face Swap (Pipeline2 Logic)")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Complete!")
    logger.info("=" * 60)