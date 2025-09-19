#!/usr/bin/env python3
"""
Recreate the middle frame from the video using nemo/data/IMG_1.png as identity.
Fixed version that properly uses the canonical volume from cache.
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
from typing import Dict, Optional
from PIL import Image

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedMiddleFrameRecreator:
    """Recreate target frames using IMG_1.png as identity - FIXED VERSION."""

    def __init__(self, emo_model, cache_path: str = "proper_face_attributes.h5"):
        self.emo_model = emo_model
        self.emo_model.eval()
        self.cache_path = cache_path
        self.cached_attributes = None

    def load_img1_identity(self):
        """Load IMG_1.png and extract its identity-specific features."""
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

        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

        with torch.no_grad():
            # Get face mask
            mask, _, _, _ = self.emo_model.face_idt.forward(img_tensor)
            mask = (mask > 0.6).float()

            # Get masked image
            masked = img_tensor * mask

            # Extract identity embedding for decoder
            identity_embed = self.emo_model.idt_embedder_nw(masked)

            # Extract source latents for creating source volume
            source_latents = self.emo_model.local_encoder_nw(masked)

            # Create source volume from IMG_1
            c = self.emo_model.args.latent_volume_channels
            d = self.emo_model.args.latent_volume_depth
            s = self.emo_model.args.latent_volume_size
            source_volume = source_latents.view(1, c, d, s, s)

            # Process source volume if needed
            if self.emo_model.args.source_volume_num_blocks > 0:
                source_volume = self.emo_model.volume_source_nw(source_volume)

        return img_tensor, identity_embed, mask, source_volume

    def load_cache(self):
        """Load cached face attributes including UV warps."""
        if Path(self.cache_path).exists():
            logger.info(f"Loading cache from {self.cache_path}")

            with h5py.File(self.cache_path, 'r') as f:
                self.cached_attributes = {
                    'num_frames': f.attrs['num_frames'],
                    'frames': []
                }

                # Also load the cached identity if available
                if 'identity_embed' in f:
                    self.cached_identity_embed = torch.from_numpy(f['identity_embed'][:]).cuda()
                else:
                    self.cached_identity_embed = None

                # Load all frame data
                for i in range(self.cached_attributes['num_frames']):
                    frame_group = f[f'frame_{i:04d}']
                    frame_data = {}

                    for key in frame_group.keys():
                        frame_data[key] = torch.from_numpy(frame_group[key][:]).cuda()

                    self.cached_attributes['frames'].append(frame_data)

            logger.info(f"Loaded {self.cached_attributes['num_frames']} frames from cache")
        else:
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

    def recreate_target_frame_properly(self,
                                      img1_identity_embed: torch.Tensor,
                                      img1_source_volume: torch.Tensor,
                                      target_frame_idx: int) -> Dict:
        """
        Properly recreate a target frame using IMG_1 identity.

        This follows the pipeline2.py approach:
        1. Use IMG_1's source volume
        2. Apply target's XY warps to get canonical
        3. Process through volume network with IMG_1's identity
        4. Apply target's UV warps and rotation
        5. Decode with IMG_1's identity embedding
        """

        if self.cached_attributes is None:
            self.load_cache()

        # Get target attributes from cache
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

            # Step 1: Apply target's rotation to IMG_1's source volume
            grid = self.emo_model.identity_grid_3d[:1]
            source_rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
            rotated_source = self.emo_model.grid_sample(img1_source_volume, source_rotation_warp)

            # Step 2: Apply target's XY warps to normalize to canonical
            # XY warps take the source expression to canonical space
            canonical_volume = self.emo_model.grid_sample(rotated_source, target_xy_warps)

            # Step 3: Process through volume network with IMG_1's identity
            embed_dict = {'idt': img1_identity_embed}
            processed_canonical = self.emo_model.volume_process_nw(canonical_volume, embed_dict)

            # Step 4: Apply target's UV warps to get target expression
            # UV warps apply the target expression from canonical space
            target_volume_with_expression = self.emo_model.grid_sample(processed_canonical, target_uv_warps)

            # Step 5: Apply target rotation
            target_rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
            final_target_volume = self.emo_model.grid_sample(target_volume_with_expression, target_rotation_warp)

            # Step 6: Reshape for decoder
            target_latent_feats = final_target_volume.view(1, c * d, s, s)

            # Step 7: Prepare data for decoder
            data_dict = {
                'target_theta': target_theta,
                'target_pose_embed': target_expression
            }

            # Step 8: Decode with IMG_1's identity
            generated_img, _, _, _ = self.emo_model.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            # Step 9: Apply face mask compositing
            face_mask, _, _, _ = self.emo_model.face_idt.forward(generated_img)
            face_mask = (face_mask > 0.6).float()

            # Composite with original background
            background = target_attrs['frame'] * (1 - face_mask)
            final_img = generated_img * face_mask + background

            return {
                'generated': final_img,
                'generated_raw': generated_img,
                'source_volume': img1_source_volume,
                'canonical_volume': processed_canonical,
                'target_volume': final_target_volume,
                'xy_warps': target_xy_warps,
                'uv_warps': target_uv_warps,
                'face_mask': face_mask,
                'original': target_attrs['frame'],
                'rotated_source': rotated_source,
                'target_with_expression': target_volume_with_expression
            }


def visualize_face_swap_fixed(results: Dict, identity_frame: torch.Tensor, save_path: str = "face_swap_fixed.png"):
    """Visualize the fixed face swap results."""

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Row 1: Main images
    # IMG_1.png identity
    identity = identity_frame[0].cpu().permute(1, 2, 0).numpy()
    identity = (identity + 1) / 2
    axes[0, 0].imshow(identity)
    axes[0, 0].set_title('IMG_1.png (Identity)', fontsize=10, weight='bold', color='blue')
    axes[0, 0].axis('off')

    # Original target
    orig = results['original'][0].cpu().permute(1, 2, 0).numpy()
    orig = (orig + 1) / 2
    axes[0, 1].imshow(orig)
    axes[0, 1].set_title('Original Target', fontsize=10)
    axes[0, 1].axis('off')

    # Generated raw
    gen_raw = results['generated_raw'][0].cpu().permute(1, 2, 0).numpy()
    gen_raw = (gen_raw + 1) / 2
    axes[0, 2].imshow(gen_raw)
    axes[0, 2].set_title('Generated (Raw)', fontsize=10)
    axes[0, 2].axis('off')

    # Generated final
    gen_final = results['generated'][0].cpu().permute(1, 2, 0).numpy()
    gen_final = (gen_final + 1) / 2
    axes[0, 3].imshow(gen_final)
    axes[0, 3].set_title('Generated (Final)', fontsize=10, weight='bold', color='green')
    axes[0, 3].axis('off')

    # Face mask
    mask = results['face_mask'][0, 0].cpu().numpy()
    axes[0, 4].imshow(mask, cmap='gray')
    axes[0, 4].set_title('Face Mask', fontsize=10)
    axes[0, 4].axis('off')

    # Row 2: Volume progression
    # Source volume (IMG_1)
    source_vol = results['source_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 0].imshow(source_vol, cmap='coolwarm')
    axes[1, 0].set_title('IMG_1 Source Volume', fontsize=10, color='blue')
    axes[1, 0].axis('off')

    # Rotated source
    rotated = results['rotated_source'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 1].imshow(rotated, cmap='coolwarm')
    axes[1, 1].set_title('After Source Rotation', fontsize=10)
    axes[1, 1].axis('off')

    # Canonical volume
    canonical = results['canonical_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 2].imshow(canonical, cmap='coolwarm')
    axes[1, 2].set_title('Canonical (XY warped)', fontsize=10)
    axes[1, 2].axis('off')

    # With expression
    with_expr = results['target_with_expression'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 3].imshow(with_expr, cmap='coolwarm')
    axes[1, 3].set_title('With Expression (UV)', fontsize=10)
    axes[1, 3].axis('off')

    # Final target volume
    target = results['target_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 4].imshow(target, cmap='coolwarm')
    axes[1, 4].set_title('Final Target Volume', fontsize=10)
    axes[1, 4].axis('off')

    # Row 3: Warps
    # XY warp magnitude
    xy_warp = results['xy_warps'][0, 8].cpu().numpy()
    xy_mag = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
    axes[2, 1].imshow(xy_mag, cmap='viridis')
    axes[2, 1].set_title('XY Warp (→Canonical)', fontsize=10)
    axes[2, 1].axis('off')

    # UV warp magnitude
    uv_warp = results['uv_warps'][0, 8].cpu().numpy()
    uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
    axes[2, 2].imshow(uv_mag, cmap='plasma')
    axes[2, 2].set_title('UV Warp (→Expression)', fontsize=10)
    axes[2, 2].axis('off')

    # Difference maps
    # Canonical vs Source difference
    canon_diff = np.abs(canonical - source_vol.mean())
    axes[2, 3].imshow(canon_diff, cmap='hot')
    axes[2, 3].set_title('Canonical Δ', fontsize=10)
    axes[2, 3].axis('off')

    # Final vs Canonical difference
    final_diff = np.abs(target - canonical)
    axes[2, 4].imshow(final_diff, cmap='hot')
    axes[2, 4].set_title('Expression Δ', fontsize=10)
    axes[2, 4].axis('off')

    # Hide unused subplot
    axes[2, 0].axis('off')

    plt.suptitle('Fixed Face Swap: IMG_1.png Identity → Target Expression', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved fixed face swap visualization to {save_path}")


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

    # Set optimizer mode
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def main():
    """Main function for fixed face swap."""

    # Load model
    model = load_volumetric_model()

    # Initialize recreator
    recreator = FixedMiddleFrameRecreator(model, cache_path="proper_face_attributes.h5")

    # Load IMG_1.png identity and create source volume
    identity_frame, identity_embed, identity_mask, img1_source_volume = recreator.load_img1_identity()
    logger.info(f"Loaded IMG_1.png: frame {identity_frame.shape}, embed {identity_embed.shape}, volume {img1_source_volume.shape}")

    # Load cache
    recreator.load_cache()
    logger.info(f"Cache loaded with {recreator.cached_attributes['num_frames']} frames")

    # Get middle frame index
    middle_idx = recreator.cached_attributes['num_frames'] // 2
    logger.info(f"Using frame {middle_idx} as target (middle frame)")

    # Recreate with fixed pipeline
    logger.info("Recreating middle frame with fixed pipeline...")
    results = recreator.recreate_target_frame_properly(
        img1_identity_embed=identity_embed,
        img1_source_volume=img1_source_volume,
        target_frame_idx=middle_idx
    )

    # Visualize results
    visualize_face_swap_fixed(results, identity_frame, "face_swap_fixed_middle.png")

    # Generate multiple frames for comparison
    logger.info("\nGenerating multiple frames for comparison...")

    num_frames = recreator.cached_attributes['num_frames']
    test_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, min(num_frames-1, num_frames)]

    fig, axes = plt.subplots(3, len(test_indices) + 1, figsize=((len(test_indices) + 1) * 3, 9))

    # Show IMG_1 in first column
    identity_img = identity_frame[0].cpu().permute(1, 2, 0).numpy()
    identity_img = (identity_img + 1) / 2
    axes[0, 0].imshow(identity_img)
    axes[0, 0].set_title('IMG_1.png', fontsize=9, weight='bold', color='blue')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Identity\nSource', ha='center', va='center', fontsize=10, weight='bold', color='blue')
    axes[2, 0].axis('off')

    for i, idx in enumerate(test_indices):
        col = i + 1  # Offset by 1 for IMG_1 column
        logger.info(f"Generating frame {idx}...")

        # Generate target frame
        results = recreator.recreate_target_frame_properly(
            img1_identity_embed=identity_embed,
            img1_source_volume=img1_source_volume,
            target_frame_idx=idx
        )

        # Original target
        orig = results['original'][0].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f'Target {idx}', fontsize=9)
        axes[0, col].axis('off')

        # Generated with IMG_1
        gen = results['generated'][0].cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1) / 2
        axes[1, col].imshow(gen)
        axes[1, col].set_title(f'Swapped {idx}', fontsize=9, color='green')
        axes[1, col].axis('off')

        # UV warp
        uv_warp = results['uv_warps'][0, 8].cpu().numpy()
        uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
        axes[2, col].imshow(uv_mag, cmap='plasma')
        axes[2, col].set_title(f'UV {idx}', fontsize=9)
        axes[2, col].axis('off')

    plt.suptitle('Fixed Face Swap: IMG_1.png → Multiple Target Frames', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("face_swap_fixed_multiple.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\n=== Fixed Face Swap Complete ===")
    logger.info("Generated files:")
    logger.info("  - face_swap_fixed_middle.png: Detailed middle frame with volume progression")
    logger.info("  - face_swap_fixed_multiple.png: Multiple frame comparisons")
    logger.info(f"  - Used IMG_1.png as identity source")
    logger.info(f"  - Properly applied warps following pipeline2.py approach")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Fixed Face Swap: IMG_1.png Identity")
    logger.info("Proper warp application following pipeline2.py")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)