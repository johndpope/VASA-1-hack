#!/usr/bin/env python3
"""
Fixed face swap with proper range handling for compositing.
The decoder outputs [0,1] but inputs are [-1,1] - handle this correctly!
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


def load_and_preprocess_image(image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess image to tensor."""
    img = Image.open(image_path).convert('RGB')
    img_original = np.array(img)

    # Resize to 512x512
    if img_original.shape[:2] != (512, 512):
        img_original = cv2.resize(img_original, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor and normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_original).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    return img_tensor, img_original


def process_source_identity(model, source_img: torch.Tensor) -> Dict:
    """Process source image to extract identity."""
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

        # Create data dict
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

        # Create rotation warp
        grid = model.identity_grid_3d[:1]
        source_rotation_warp = grid.bmm(pred_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)

        # Apply to get canonical
        rotated_source = model.grid_sample(source_volume, source_rotation_warp)
        canonical_volume = model.grid_sample(rotated_source, source_xy_warp)

        # Process through volume network
        processed_canonical = model.volume_process_nw(canonical_volume, embed_dict)

        return {
            'embed_dict': embed_dict,
            'idt_embed': idt_embed,
            'processed_canonical': processed_canonical
        }


def apply_target_expression_fixed(model, source_info: Dict, target_img: torch.Tensor,
                                  target_original: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply target expression with FIXED range handling."""
    with torch.no_grad():
        # Get target face mask
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
            'idt_embed': source_info['idt_embed']  # Use source identity
        }

        # Get target expression
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        target_pose_embed = data_dict['source_pose_embed']

        # Get target warps
        _, target_warp_embed_dict, _, _ = model.predict_embed(data_dict)

        # Generate UV warps
        target_uv_warp, _ = model.uv_generator_nw(target_warp_embed_dict)

        # Apply UV warp to canonical
        target_with_expression = model.grid_sample(source_info['processed_canonical'], target_uv_warp)

        # Apply target rotation
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        grid = model.identity_grid_3d[:1]
        target_rotation_warp = grid.bmm(pred_target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        final_volume = model.grid_sample(target_with_expression, target_rotation_warp)

        # Decode
        target_latent_feats = final_volume.view(1, c * d, s, s)

        decode_dict = {
            'target_theta': pred_target_theta,
            'target_pose_embed': target_pose_embed
        }

        generated_img, _, _, _ = model.decoder_nw(
            decode_dict,
            source_info['embed_dict'],
            target_latent_feats,
            False,
            stage_two=True
        )

        # CRITICAL FIX: Check the range of generated_img
        gen_min = generated_img.min().item()
        gen_max = generated_img.max().item()
        logger.info(f"Generated image range: [{gen_min:.3f}, {gen_max:.3f}]")

        # The decoder outputs in [0, 1] range, convert to [-1, 1] for consistency
        if gen_min >= 0 and gen_max <= 1.1:  # Allow slight overshoot
            logger.info("Converting generated image from [0,1] to [-1,1]")
            generated_img = generated_img * 2 - 1

        # Now both are in [-1, 1] range, safe to composite

        # Get mask from generated image
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        gen_mask = (gen_mask > 0.6).float()

        # Smooth the mask edges
        gen_mask = F.avg_pool2d(gen_mask, 3, stride=1, padding=1)

        # Composite in [-1, 1] range
        background = target_img * (1 - gen_mask)
        final_img = generated_img * gen_mask + background

        # Ensure output is in [-1, 1]
        final_img = torch.clamp(final_img, -1, 1)

        return final_img, generated_img


def main():
    """Test face swap with fixed range handling."""

    # Load model
    model = load_volumetric_model()

    # Load images
    source_img, source_original = load_and_preprocess_image("nemo/data/IMG_1.png")
    logger.info("Loaded source image (IMG_1.png)")

    # Process source
    logger.info("Extracting source identity...")
    source_info = process_source_identity(model, source_img)

    # Load targets
    target_paths = []

    # Try to load from cache first
    video_frame_path = Path("proper_face_attributes.h5")
    if video_frame_path.exists():
        import h5py
        with h5py.File(video_frame_path, 'r') as f:
            # Get frames from cache
            for i in [0, 5, 9]:  # Beginning, middle, end
                if f'frame_{i:04d}' in f:
                    frame_data = torch.from_numpy(f[f'frame_{i:04d}/frame'][:]).cuda()
                    # Save temporarily
                    temp_path = f"temp_frame_{i}.png"
                    temp_frame = frame_data[0].cpu().permute(1, 2, 0).numpy()
                    temp_frame = ((temp_frame + 1) * 127.5).astype(np.uint8)
                    Image.fromarray(temp_frame).save(temp_path)
                    target_paths.append(temp_path)
                    logger.info(f"Added cached frame {i} as target")

    # Also add static images
    for path in ["nemo/data/IMG_2.png", "nemo/data/IMG_3.png", "nemo/data/IMG_4.png"]:
        if Path(path).exists():
            target_paths.append(path)

    results = []
    for target_path in target_paths[:6]:  # Limit to 6 for display
        if Path(target_path).exists():
            logger.info(f"Processing target: {target_path}")
            target_img, target_original = load_and_preprocess_image(target_path)

            # Apply with fixed range handling
            final_img, raw_img = apply_target_expression_fixed(
                model, source_info, target_img, target_original
            )

            results.append({
                'path': target_path,
                'target': target_img,
                'final': final_img,
                'raw': raw_img
            })

    # Visualize
    if results:
        n_results = len(results)
        fig, axes = plt.subplots(3, n_results + 1, figsize=(3 * (n_results + 1), 9))

        # Source
        source_display = source_img[0].cpu().permute(1, 2, 0).numpy()
        source_display = (source_display + 1) / 2  # Convert to [0, 1] for display
        source_display = np.clip(source_display, 0, 1)

        axes[0, 0].imshow(source_display)
        axes[0, 0].set_title("Source\n(IMG_1)", fontsize=10, weight='bold', color='blue')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        for i, result in enumerate(results):
            col = i + 1

            # Target
            target_display = result['target'][0].cpu().permute(1, 2, 0).numpy()
            target_display = (target_display + 1) / 2
            target_display = np.clip(target_display, 0, 1)
            axes[0, col].imshow(target_display)
            axes[0, col].set_title(f"Target {i+1}", fontsize=10)
            axes[0, col].axis('off')

            # Raw generated (now in [-1, 1])
            raw_display = result['raw'][0].cpu().permute(1, 2, 0).numpy()
            raw_display = (raw_display + 1) / 2
            raw_display = np.clip(raw_display, 0, 1)
            axes[1, col].imshow(raw_display)
            axes[1, col].set_title("Generated", fontsize=10)
            axes[1, col].axis('off')

            # Final composite
            final_display = result['final'][0].cpu().permute(1, 2, 0).numpy()
            final_display = (final_display + 1) / 2
            final_display = np.clip(final_display, 0, 1)
            axes[2, col].imshow(final_display)
            axes[2, col].set_title("Final", fontsize=10, weight='bold', color='green')
            axes[2, col].axis('off')

        plt.suptitle("Face Swap with Fixed Range Handling", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("fixed_range_face_swap.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved visualization to fixed_range_face_swap.png")

        # Save best result
        if results:
            # Pick the middle frame from cache if available
            best_idx = min(2, len(results) - 1)
            best = results[best_idx]
            best_img = best['final'][0].cpu().permute(1, 2, 0).numpy()
            best_img = (best_img + 1) / 2
            best_img = np.clip(best_img, 0, 1)
            best_img = (best_img * 255).astype(np.uint8)
            Image.fromarray(best_img).save("fixed_final_result.png")
            logger.info(f"Saved best result to fixed_final_result.png")

    # Clean up temp files
    for path in Path(".").glob("temp_frame_*.png"):
        path.unlink()
    if Path("temp_target.png").exists():
        Path("temp_target.png").unlink()

    logger.info("\n=== Complete ===")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Face Swap with Fixed Range Handling")
    logger.info("Decoder outputs [0,1], inputs are [-1,1] - handling correctly!")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)