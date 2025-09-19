#!/usr/bin/env python3
"""
Recreate the middle frame from the video using nemo/data/IMG_1.png as identity.
This uses cached warps to perform face swapping similar to pipeline2.py results.
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


class MiddleFrameRecreator:
    """Recreate target frames using IMG_1.png as identity."""

    def __init__(self, emo_model, cache_path: str = "proper_face_attributes.h5"):
        self.emo_model = emo_model
        self.emo_model.eval()
        self.cache_path = cache_path
        self.cached_attributes = None

    def load_img1_as_identity(self):
        """Load IMG_1.png and prepare it as identity frame."""
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

        # Extract identity embedding
        with torch.no_grad():
            # Get face mask
            mask, _, _, _ = self.emo_model.face_idt.forward(img_tensor)
            mask = (mask > 0.6).float()

            # Get masked image
            masked = img_tensor * mask

            # Extract identity embedding
            identity_embed = self.emo_model.idt_embedder_nw(masked)

        return img_tensor, identity_embed, mask

    def load_cache(self):
        """Load cached face attributes including UV warps."""
        if Path(self.cache_path).exists():
            logger.info(f"Loading cache from {self.cache_path}")

            with h5py.File(self.cache_path, 'r') as f:
                self.cached_attributes = {
                    'num_frames': f.attrs['num_frames'],
                    'frames': []
                }

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

    def recreate_target_frame(self,
                             identity_frame: torch.Tensor,
                             identity_embed: torch.Tensor,
                             target_frame_idx: int) -> Dict:
        """
        Recreate a target frame using IMG_1.png identity and cached warps.

        Args:
            identity_frame: IMG_1.png as identity source
            identity_embed: Identity embedding from IMG_1.png
            target_frame_idx: Index of target frame attributes to use

        Returns:
            Dictionary containing generated frame and intermediate results
        """

        if self.cached_attributes is None:
            self.load_cache()

        # Get target attributes from cache
        target_attrs = self.cached_attributes['frames'][target_frame_idx]

        with torch.no_grad():
            # Extract components
            target_theta = target_attrs['theta']
            target_mask = target_attrs['mask']
            target_expression = target_attrs.get('expression_embed')
            target_uv_warps = target_attrs['uv_warps']
            target_xy_warps = target_attrs['xy_warps']
            canonical_volume = target_attrs['canonical_volume']

            # Step 1: Get source volume from IMG_1.png identity
            masked_identity = identity_frame * target_mask
            source_latents = self.emo_model.local_encoder_nw(masked_identity)

            c = self.emo_model.args.latent_volume_channels
            d = self.emo_model.args.latent_volume_depth
            s = self.emo_model.args.latent_volume_size
            source_volume = source_latents.view(1, c, d, s, s)

            # Process source volume
            if self.emo_model.args.source_volume_num_blocks > 0:
                source_volume = self.emo_model.volume_source_nw(source_volume)

            # Step 2: Apply XY warps to normalize to canonical
            canonical_from_xy = self.emo_model.grid_sample(source_volume, target_xy_warps)

            # Step 3: Process through volume network
            processed_canonical = self.emo_model.volume_process_nw(canonical_from_xy)

            # Step 4: Apply UV warps to get target expression
            target_volume_uv = self.emo_model.grid_sample(processed_canonical, target_uv_warps)

            # Step 5: Apply rotation from target theta
            grid = self.emo_model.identity_grid_3d[:1]  # [1, d*s*s, 3]
            rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
            target_volume = self.emo_model.grid_sample(target_volume_uv, rotation_warp)

            # Step 6: Reshape for decoder
            target_latent_feats = target_volume.view(1, c * d, s, s)

            # Step 7: Prepare data dict for decoder
            data_dict = {
                'target_theta': target_theta,
                'target_pose_embed': target_expression
            }

            embed_dict = {'idt': identity_embed}

            # Step 8: Decode to generate final image
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
                'identity_frame': identity_frame,
                'source_volume': source_volume,
                'canonical_volume': processed_canonical,
                'target_volume': target_volume,
                'xy_warps': target_xy_warps,
                'uv_warps': target_uv_warps,
                'face_mask': face_mask,
                'original': target_attrs['frame']
            }


def visualize_face_swap(results: Dict, save_path: str = "face_swap_img1.png"):
    """Visualize the face swap results."""

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Row 1: Main images
    # IMG_1.png identity
    identity = results['identity_frame'][0].cpu().permute(1, 2, 0).numpy()
    identity = (identity + 1) / 2
    axes[0, 0].imshow(identity)
    axes[0, 0].set_title('IMG_1.png (Identity)', fontsize=10, weight='bold')
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

    # Row 2: Warps and volumes
    # Source volume
    source_vol = results['source_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 0].imshow(source_vol, cmap='coolwarm')
    axes[1, 0].set_title('Source Volume (IMG_1)', fontsize=10)
    axes[1, 0].axis('off')

    # XY warp
    xy_warp = results['xy_warps'][0, 8].cpu().numpy()
    xy_mag = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
    axes[1, 1].imshow(xy_mag, cmap='viridis')
    axes[1, 1].set_title('XY Warp (→Canonical)', fontsize=10)
    axes[1, 1].axis('off')

    # Canonical volume
    canonical = results['canonical_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 2].imshow(canonical, cmap='coolwarm')
    axes[1, 2].set_title('Canonical Volume', fontsize=10)
    axes[1, 2].axis('off')

    # UV warp
    uv_warp = results['uv_warps'][0, 8].cpu().numpy()
    uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
    axes[1, 3].imshow(uv_mag, cmap='plasma')
    axes[1, 3].set_title('UV Warp (→Target)', fontsize=10)
    axes[1, 3].axis('off')

    # Target volume
    target = results['target_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 4].imshow(target, cmap='coolwarm')
    axes[1, 4].set_title('Target Volume', fontsize=10)
    axes[1, 4].axis('off')

    plt.suptitle('Face Swap: IMG_1.png Identity → Target Expression', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved face swap visualization to {save_path}")


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
    """Main function to recreate middle frame with IMG_1.png."""

    # Load model
    model = load_volumetric_model()

    # Initialize recreator
    recreator = MiddleFrameRecreator(model, cache_path="proper_face_attributes.h5")

    # Load IMG_1.png as identity
    identity_frame, identity_embed, identity_mask = recreator.load_img1_as_identity()
    logger.info(f"Loaded IMG_1.png identity: {identity_frame.shape}, embed: {identity_embed.shape}")

    # Load cache
    recreator.load_cache()
    logger.info(f"Cache loaded with {recreator.cached_attributes['num_frames']} frames")

    # Get middle frame index (roughly half of available frames)
    middle_idx = recreator.cached_attributes['num_frames'] // 2
    logger.info(f"Using frame {middle_idx} as target (middle frame)")

    # Recreate the middle frame with IMG_1 identity
    logger.info("Recreating middle frame with IMG_1.png identity...")
    results = recreator.recreate_target_frame(
        identity_frame=identity_frame,
        identity_embed=identity_embed,
        target_frame_idx=middle_idx
    )

    # Visualize results
    visualize_face_swap(results, "face_swap_img1_middle.png")

    # Also generate a few more frames for comparison
    logger.info("\nGenerating multiple frames for comparison...")

    # Select frames: beginning, quarter, middle, three-quarters, end
    num_frames = recreator.cached_attributes['num_frames']
    test_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, min(num_frames-1, num_frames)]

    fig, axes = plt.subplots(3, len(test_indices), figsize=(len(test_indices) * 3, 9))

    for i, idx in enumerate(test_indices):
        logger.info(f"Generating frame {idx}...")

        # Generate target frame
        results = recreator.recreate_target_frame(
            identity_frame=identity_frame,
            identity_embed=identity_embed,
            target_frame_idx=idx
        )

        # Original target
        orig = results['original'][0].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Target {idx}', fontsize=9)
        axes[0, i].axis('off')

        # Generated with IMG_1
        gen = results['generated'][0].cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1) / 2
        axes[1, i].imshow(gen)
        axes[1, i].set_title(f'Generated {idx}', fontsize=9, color='green')
        axes[1, i].axis('off')

        # UV warp
        uv_warp = results['uv_warps'][0, 8].cpu().numpy()
        uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
        axes[2, i].imshow(uv_mag, cmap='plasma')
        axes[2, i].set_title(f'UV Warp {idx}', fontsize=9)
        axes[2, i].axis('off')

    # Add IMG_1 reference on the left
    fig.text(0.02, 0.75, 'Target\nFrames', fontsize=10, weight='bold', ha='center', va='center')
    fig.text(0.02, 0.5, 'IMG_1\nSwapped', fontsize=10, weight='bold', ha='center', va='center', color='green')
    fig.text(0.02, 0.25, 'UV\nWarps', fontsize=10, weight='bold', ha='center', va='center')

    plt.suptitle('Face Swap Results: IMG_1.png → Multiple Target Frames', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig("face_swap_img1_multiple.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\n=== Face Swap Complete ===")
    logger.info("Generated files:")
    logger.info("  - face_swap_img1_middle.png: Detailed middle frame recreation")
    logger.info("  - face_swap_img1_multiple.png: Multiple frame comparisons")
    logger.info(f"  - Used IMG_1.png as identity source")
    logger.info(f"  - Applied warps from cached {num_frames} frames")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Recreating Middle Frame with IMG_1.png Identity")
    logger.info("Face swapping using cached UV/XY warps")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)