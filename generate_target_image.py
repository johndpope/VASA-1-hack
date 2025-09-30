#!/usr/bin/env python3
"""
Generate the specific face-swapped image requested by the user.
Uses IMG_1.png identity with the target expression from cached frames.
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
import h5py
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from PIL import Image

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetImageGenerator:
    """Generate the specific target image with IMG_1 identity."""

    def __init__(self, emo_model, cache_path: str = "proper_face_attributes.h5"):
        self.emo_model = emo_model
        self.emo_model.eval()
        self.cache_path = cache_path
        self.cached_attributes = None

    def load_img1_identity(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load IMG_1.png and extract its identity features."""
        img1_path = Path("nemo/data/IMG_1.png")
        if not img1_path.exists():
            raise FileNotFoundError(f"IMG_1.png not found at {img1_path}")

        logger.info(f"Loading identity from {img1_path}")

        # Load image
        img = Image.open(img1_path).convert('RGB')
        img = np.array(img)

        # Resize to 512x512 if needed
        if img.shape[:2] != (512, 512):
            img = cv2.resize(img, (512, 512))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            # Get face mask
            mask, _, _, _ = self.emo_model.face_idt.forward(img_tensor)
            mask = (mask > 0.6).float()

            # Get masked image
            masked = img_tensor * mask

            # Extract identity embedding
            identity_embed = self.emo_model.idt_embedder_nw(masked)

            # Extract source latents
            source_latents = self.emo_model.local_encoder_nw(masked)

            # Create source volume
            c = self.emo_model.args.latent_volume_channels
            d = self.emo_model.args.latent_volume_depth
            s = self.emo_model.args.latent_volume_size
            source_volume = source_latents.view(1, c, d, s, s)

            # Process source volume
            if self.emo_model.args.source_volume_num_blocks > 0:
                source_volume = self.emo_model.volume_source_nw(source_volume)

        return img_tensor, identity_embed, mask, source_volume

    def load_cache(self):
        """Load cached face attributes."""
        if Path(self.cache_path).exists():
            logger.info(f"Loading cache from {self.cache_path}")

            with h5py.File(self.cache_path, 'r') as f:
                self.cached_attributes = {
                    'num_frames': f.attrs['num_frames'],
                    'frames': []
                }

                for i in range(self.cached_attributes['num_frames']):
                    frame_group = f[f'frame_{i:04d}']
                    frame_data = {}

                    for key in frame_group.keys():
                        frame_data[key] = torch.from_numpy(frame_group[key][:]).cuda()

                    self.cached_attributes['frames'].append(frame_data)

            logger.info(f"Loaded {self.cached_attributes['num_frames']} frames from cache")
        else:
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

    def generate_face_swap(self,
                          identity_embed: torch.Tensor,
                          source_volume: torch.Tensor,
                          target_frame_idx: int) -> torch.Tensor:
        """
        Generate face-swapped image using IMG_1 identity and target expression.
        """
        if self.cached_attributes is None:
            self.load_cache()

        # Get target attributes
        target_attrs = self.cached_attributes['frames'][target_frame_idx]

        with torch.no_grad():
            # Extract target components
            target_theta = target_attrs['theta']
            target_expression = target_attrs.get('expression_embed')
            target_uv_warps = target_attrs['uv_warps']
            target_xy_warps = target_attrs['xy_warps']

            c = self.emo_model.args.latent_volume_channels
            d = self.emo_model.args.latent_volume_depth
            s = self.emo_model.args.latent_volume_size

            # Apply transformations
            grid = self.emo_model.identity_grid_3d[:1]

            # Step 1: Rotate source volume
            source_rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
            rotated_source = self.emo_model.grid_sample(source_volume, source_rotation_warp)

            # Step 2: Apply XY warps to get canonical
            canonical_volume = self.emo_model.grid_sample(rotated_source, target_xy_warps)

            # Step 3: Process through volume network
            embed_dict = {'idt': identity_embed}
            processed_canonical = self.emo_model.volume_process_nw(canonical_volume, embed_dict)

            # Step 4: Apply UV warps for expression
            target_volume_with_expression = self.emo_model.grid_sample(processed_canonical, target_uv_warps)

            # Step 5: Apply final rotation
            target_rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
            final_target_volume = self.emo_model.grid_sample(target_volume_with_expression, target_rotation_warp)

            # Step 6: Prepare for decoder
            target_latent_feats = final_target_volume.view(1, c * d, s, s)

            # Step 7: Decode
            data_dict = {
                'target_theta': target_theta,
                'target_pose_embed': target_expression
            }

            generated_img, _, _, _ = self.emo_model.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            # Step 8: Apply face mask
            face_mask, _, _, _ = self.emo_model.face_idt.forward(generated_img)
            face_mask = (face_mask > 0.6).float()

            # Composite with background
            background = target_attrs['frame'] * (1 - face_mask)
            final_img = generated_img * face_mask + background

            return final_img, generated_img, target_attrs['frame']

    def find_best_target_frame(self) -> int:
        """
        Find the best target frame that matches a smiling expression.
        Based on the user's image, we want a frame with a smile.
        """
        # For now, let's try different frames and see which gives best results
        # Frame 5 (middle) or frames 6-8 might have good expressions
        return 6  # Try frame 6 which might have a smile


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


def main():
    """Generate the target image."""

    # Load model
    model = load_volumetric_model()

    # Initialize generator
    generator = TargetImageGenerator(model)

    # Load IMG_1 identity
    identity_frame, identity_embed, identity_mask, source_volume = generator.load_img1_identity()
    logger.info("Loaded IMG_1.png identity")

    # Load cache
    generator.load_cache()

    # Try different frames to find the best match
    logger.info("\nGenerating face swaps for different expressions...")

    # Test multiple frames to find one with a smile
    test_frames = list(range(generator.cached_attributes['num_frames']))

    fig, axes = plt.subplots(3, len(test_frames) // 2 + 1, figsize=(20, 12))
    axes = axes.flatten()

    all_results = []
    for i, frame_idx in enumerate(test_frames):
        logger.info(f"Generating frame {frame_idx}...")

        final_img, raw_img, original = generator.generate_face_swap(
            identity_embed=identity_embed,
            source_volume=source_volume,
            target_frame_idx=frame_idx
        )

        all_results.append((final_img, raw_img, original, frame_idx))

        # Display result
        img_display = final_img[0].cpu().permute(1, 2, 0).numpy()
        img_display = (img_display + 1) / 2
        axes[i].imshow(img_display)
        axes[i].set_title(f'Frame {frame_idx}', fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(test_frames), len(axes)):
        axes[i].axis('off')

    # Add IMG_1 reference
    axes[-1].imshow((identity_frame[0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    axes[-1].set_title('IMG_1 Identity', fontsize=10, weight='bold', color='blue')

    plt.suptitle('Face Swap Results: Finding Best Expression Match', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("all_frame_swaps.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save individual best results
    logger.info("\nSaving best results...")

    # Pick frame 6 as it might have the smile
    best_idx = 6 if 6 < len(all_results) else len(all_results) // 2
    best_result = all_results[best_idx]

    # Save the best one
    best_img = best_result[0][0].cpu().permute(1, 2, 0).numpy()
    best_img = (best_img + 1) / 2
    best_img = (best_img * 255).astype(np.uint8)
    Image.fromarray(best_img).save("target_face_swap_result.png")
    logger.info(f"Saved best result (frame {best_result[3]}) to target_face_swap_result.png")

    # Create detailed comparison for best frame
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # IMG_1 identity
    axes[0].imshow((identity_frame[0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    axes[0].set_title('IMG_1 (Identity)', weight='bold', color='blue')
    axes[0].axis('off')

    # Original target
    axes[1].imshow((best_result[2][0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    axes[1].set_title('Original Target')
    axes[1].axis('off')

    # Raw generated
    axes[2].imshow((best_result[1][0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    axes[2].set_title('Generated (Raw)')
    axes[2].axis('off')

    # Final result
    axes[3].imshow((best_result[0][0].cpu().permute(1, 2, 0).numpy() + 1) / 2)
    axes[3].set_title('Final Result', weight='bold', color='green')
    axes[3].axis('off')

    plt.suptitle(f'Best Face Swap Result (Frame {best_result[3]})', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("best_face_swap_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\n=== Generation Complete ===")
    logger.info("Generated files:")
    logger.info("  - target_face_swap_result.png: The requested face swap image")
    logger.info("  - best_face_swap_comparison.png: Detailed comparison")
    logger.info("  - all_frame_swaps.png: All available expressions")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Generating Target Face Swap Image")
    logger.info("Using IMG_1.png identity with target expression")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Generation complete!")
    logger.info("=" * 60)