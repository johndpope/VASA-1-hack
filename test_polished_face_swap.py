#!/usr/bin/env python3
"""
Polished face swap with proper compositing and finishing.
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
from PIL import Image, ImageFilter
from typing import Dict, Tuple
from scipy.ndimage import gaussian_filter

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
    """Load and preprocess image to tensor, also return original."""
    img = Image.open(image_path).convert('RGB')
    img_original = np.array(img)

    # Resize to 512x512
    if img_original.shape[:2] != (512, 512):
        img_original = cv2.resize(img_original, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_original).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    return img_tensor, img_original


def create_soft_mask(mask: torch.Tensor, erosion_size: int = 5, blur_size: int = 15) -> torch.Tensor:
    """Create a soft mask with smooth edges for better blending."""

    # Convert to numpy for processing
    mask_np = mask[0, 0].cpu().numpy()

    # Erode mask slightly to avoid edge artifacts
    if erosion_size > 0:
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)

    # Apply Gaussian blur for smooth edges
    mask_np = gaussian_filter(mask_np, sigma=blur_size)

    # Ensure mask is in [0, 1] range
    mask_np = np.clip(mask_np, 0, 1)

    # Convert back to tensor
    soft_mask = torch.from_numpy(mask_np).float().cuda()
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)

    return soft_mask


def color_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Transfer color statistics from target to source in masked region."""

    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Calculate statistics in masked region
    mask_bool = mask > 0.5

    if np.any(mask_bool):
        # Calculate mean and std for each channel
        for i in range(3):
            source_mean = np.mean(source_lab[:, :, i][mask_bool])
            source_std = np.std(source_lab[:, :, i][mask_bool])
            target_mean = np.mean(target_lab[:, :, i][mask_bool])
            target_std = np.std(target_lab[:, :, i][mask_bool])

            # Transfer statistics
            if source_std > 0:
                source_lab[:, :, i] = (source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean

    # Convert back to RGB
    source_lab = np.clip(source_lab, 0, 255)
    result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return result


def seamless_clone(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Use Poisson blending for seamless cloning."""

    # Ensure mask is binary
    mask_binary = (mask * 255).astype(np.uint8)

    # Find center of mask
    moments = cv2.moments(mask_binary)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cy, cx = mask_binary.shape[0] // 2, mask_binary.shape[1] // 2

    try:
        # Apply seamless cloning
        result = cv2.seamlessClone(
            source,
            target,
            mask_binary,
            (cx, cy),
            cv2.MIXED_CLONE
        )
        return result
    except:
        # Fallback to simple blending
        return source


def process_source_identity(model, source_img: torch.Tensor) -> Dict:
    """Process source image to extract identity."""
    with torch.no_grad():
        # Get face mask with higher threshold for cleaner extraction
        face_mask, _, _, _ = model.face_idt.forward(source_img)
        face_mask = (face_mask > 0.7).float()  # Higher threshold

        # Create soft mask for better blending
        soft_mask = create_soft_mask(face_mask, erosion_size=3, blur_size=10)

        # Mask source
        masked_source = source_img * soft_mask

        # Get identity embedding
        idt_embed = model.idt_embedder_nw(masked_source)

        # Get head pose
        pred_theta, _, _, _ = model.head_pose_regressor.forward(source_img, True)

        # Create data dict
        data_dict = {
            'source_img': source_img,
            'source_mask': soft_mask,
            'source_theta': pred_theta,
            'target_img': source_img,
            'target_mask': soft_mask,
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
            'processed_canonical': processed_canonical,
            'soft_mask': soft_mask
        }


def apply_target_with_polish(model, source_info: Dict, target_img: torch.Tensor,
                            target_original: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply target expression with polished compositing."""
    with torch.no_grad():
        # Get target face mask
        face_mask, _, _, _ = model.face_idt.forward(target_img)
        face_mask = (face_mask > 0.7).float()

        # Create soft mask
        soft_mask = create_soft_mask(face_mask, erosion_size=5, blur_size=20)

        # Get target head pose
        pred_target_theta, _, _, _ = model.head_pose_regressor.forward(target_img, True)

        # Create data dict for target expression
        data_dict = {
            'source_img': target_img,
            'source_mask': soft_mask,
            'source_theta': pred_target_theta,
            'target_img': target_img,
            'target_mask': soft_mask,
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

        # Get refined mask from generated image
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        gen_mask = (gen_mask > 0.65).float()

        # Create very soft mask for final blending
        final_mask = create_soft_mask(gen_mask, erosion_size=8, blur_size=25)

        # Convert to numpy for post-processing
        generated_np = generated_img[0].cpu().permute(1, 2, 0).numpy()
        generated_np = ((generated_np + 1) * 127.5).astype(np.uint8)

        target_np = target_img[0].cpu().permute(1, 2, 0).numpy()
        target_np = ((target_np + 1) * 127.5).astype(np.uint8)

        mask_np = final_mask[0, 0].cpu().numpy()

        # Apply color transfer for better matching
        generated_np = color_transfer(generated_np, target_np, mask_np)

        # Try seamless cloning for smooth integration
        try:
            final_np = seamless_clone(generated_np, target_original, mask_np)
        except:
            # Fallback to weighted blending
            mask_3ch = np.stack([mask_np] * 3, axis=-1)
            final_np = (generated_np * mask_3ch + target_original * (1 - mask_3ch)).astype(np.uint8)

        # Apply slight Gaussian blur at mask edges for smoothing
        edge_mask = cv2.Canny((mask_np * 255).astype(np.uint8), 50, 150)
        edge_mask = gaussian_filter(edge_mask.astype(float) / 255, sigma=3)
        edge_mask = np.stack([edge_mask] * 3, axis=-1)

        blurred = cv2.GaussianBlur(final_np, (5, 5), 1)
        final_np = (final_np * (1 - edge_mask) + blurred * edge_mask).astype(np.uint8)

        # Convert back to tensor
        final_tensor = torch.from_numpy(final_np).float() / 127.5 - 1.0
        final_tensor = final_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

        return final_tensor, generated_img


def main():
    """Test polished face swap."""

    # Load model
    model = load_volumetric_model()

    # Load source
    source_img, source_original = load_and_preprocess_image("nemo/data/IMG_1.png")
    logger.info("Loaded source image (IMG_1.png)")

    # Process source
    logger.info("Extracting source identity...")
    source_info = process_source_identity(model, source_img)

    # Load targets
    target_paths = [
        "nemo/data/IMG_2.png",
        "nemo/data/IMG_3.png",
        "nemo/data/IMG_4.png"
    ]

    # Also try with video frames if available
    video_frame_path = Path("proper_face_attributes.h5")
    if video_frame_path.exists():
        import h5py
        with h5py.File(video_frame_path, 'r') as f:
            if 'frame_0005' in f:
                # Extract a frame from cache for testing
                frame_data = torch.from_numpy(f['frame_0005/frame'][:]).cuda()
                # Save it temporarily
                temp_frame = frame_data[0].cpu().permute(1, 2, 0).numpy()
                temp_frame = ((temp_frame + 1) * 127.5).astype(np.uint8)
                Image.fromarray(temp_frame).save("temp_target.png")
                target_paths.insert(0, "temp_target.png")
                logger.info("Added cached video frame as target")

    results = []
    for target_path in target_paths:
        if Path(target_path).exists():
            logger.info(f"Processing target: {target_path}")
            target_img, target_original = load_and_preprocess_image(target_path)

            # Apply with polish
            final_img, raw_img = apply_target_with_polish(
                model, source_info, target_img, target_original
            )

            results.append({
                'path': target_path,
                'target': target_img,
                'final': final_img,
                'raw': raw_img,
                'original': target_original
            })

    # Visualize
    if results:
        fig, axes = plt.subplots(3, len(results) + 1, figsize=(3 * (len(results) + 1), 9))

        # Source
        source_display = source_img[0].cpu().permute(1, 2, 0).numpy()
        source_display = (source_display + 1) / 2

        axes[0, 0].imshow(source_display)
        axes[0, 0].set_title("Identity\n(IMG_1)", fontsize=10, weight='bold', color='blue')
        axes[0, 0].axis('off')

        axes[1, 0].text(0.5, 0.5, 'Source\nIdentity', ha='center', va='center',
                       fontsize=12, weight='bold', color='blue')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        for i, result in enumerate(results):
            col = i + 1

            # Target
            target_display = result['target'][0].cpu().permute(1, 2, 0).numpy()
            target_display = (target_display + 1) / 2
            axes[0, col].imshow(target_display)
            axes[0, col].set_title(f"Target {i+1}", fontsize=10)
            axes[0, col].axis('off')

            # Raw
            raw_display = result['raw'][0].cpu().permute(1, 2, 0).numpy()
            raw_display = (raw_display + 1) / 2
            axes[1, col].imshow(raw_display)
            axes[1, col].set_title("Raw", fontsize=10)
            axes[1, col].axis('off')

            # Polished
            final_display = result['final'][0].cpu().permute(1, 2, 0).numpy()
            final_display = (final_display + 1) / 2
            axes[2, col].imshow(final_display)
            axes[2, col].set_title("Polished", fontsize=10, weight='bold', color='green')
            axes[2, col].axis('off')

        plt.suptitle("Polished Face Swap with Seamless Compositing", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("polished_face_swap.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved visualization to polished_face_swap.png")

        # Save best result
        if results:
            best = results[0]
            best_img = best['final'][0].cpu().permute(1, 2, 0).numpy()
            best_img = (best_img + 1) / 2
            best_img = (best_img * 255).astype(np.uint8)
            Image.fromarray(best_img).save("polished_result.png")
            logger.info("Saved best result to polished_result.png")

    # Clean up temp file
    if Path("temp_target.png").exists():
        Path("temp_target.png").unlink()

    logger.info("\n=== Complete ===")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Polished Face Swap with Proper Compositing")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)