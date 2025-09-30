#!/usr/bin/env python3
"""
Generate a single target frame using cached UV warps.
This replicates what pipeline2.py does but with clear warp extraction and application.
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

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleFrameGenerator:
    """Generate single frames using cached warps."""

    def __init__(self, emo_model, cache_path: str = "proper_face_attributes.h5"):
        self.emo_model = emo_model
        self.emo_model.eval()
        self.cache_path = cache_path
        self.cached_attributes = None

    def load_cache(self):
        """Load cached face attributes including UV warps."""
        if Path(self.cache_path).exists():
            logger.info(f"Loading cache from {self.cache_path}")

            with h5py.File(self.cache_path, 'r') as f:
                self.cached_attributes = {
                    'identity_frame': torch.from_numpy(f['identity_frame'][:]).cuda(),
                    'identity_embed': torch.from_numpy(f['identity_embed'][:]).cuda(),
                    'identity_mask': torch.from_numpy(f['identity_mask'][:]).cuda(),
                    'num_frames': f.attrs['num_frames'],
                    'frames': []
                }

                # Load all frame data
                for i in range(self.cached_attributes['num_frames']):
                    frame_group = f[f'frame_{i:04d}']
                    frame_data = {}

                    for key in frame_group.keys():
                        frame_data[key] = torch.from_numpy(frame_group[key][:]).cuda()

                    # Add identity embed to each frame
                    frame_data['idt_embed'] = self.cached_attributes['identity_embed']
                    self.cached_attributes['frames'].append(frame_data)

            logger.info(f"Loaded {self.cached_attributes['num_frames']} frames from cache")
        else:
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

    def generate_target_frame(self,
                            identity_frame: Optional[torch.Tensor] = None,
                            target_frame_idx: int = 0,
                            use_cached_identity: bool = True) -> Dict:
        """
        Generate a target frame using cached UV warps.

        Args:
            identity_frame: Optional custom identity frame, otherwise use cached
            target_frame_idx: Index of target frame attributes to use
            use_cached_identity: Whether to use cached identity or provided one

        Returns:
            Dictionary containing generated frame and intermediate results
        """

        if self.cached_attributes is None:
            self.load_cache()

        # Get identity frame and embedding
        if use_cached_identity or identity_frame is None:
            identity_frame = self.cached_attributes['identity_frame']
            identity_embed = self.cached_attributes['identity_embed']
        else:
            # Extract identity from provided frame
            with torch.no_grad():
                mask, _, _, _ = self.emo_model.face_idt.forward(identity_frame)
                mask = (mask > 0.6).float()
                masked = identity_frame * mask
                identity_embed = self.emo_model.idt_embedder_nw(masked)

        # Get target attributes
        target_attrs = self.cached_attributes['frames'][target_frame_idx]

        with torch.no_grad():
            # Extract components
            target_theta = target_attrs['theta']
            target_mask = target_attrs['mask']
            target_expression = target_attrs.get('expression_embed')
            target_uv_warps = target_attrs['uv_warps']
            target_xy_warps = target_attrs['xy_warps']
            canonical_volume = target_attrs['canonical_volume']

            # Step 1: Get source volume from identity
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
            # First apply UV warp
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
                'source_volume': source_volume,
                'canonical_volume': processed_canonical,
                'target_volume': target_volume,
                'xy_warps': target_xy_warps,
                'uv_warps': target_uv_warps,
                'face_mask': face_mask,
                'original': target_attrs['frame']
            }


def visualize_generation_pipeline(results: Dict, save_path: str = "generation_pipeline.png"):
    """Visualize the complete generation pipeline."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Images
    # Original
    orig = results['original'][0].cpu().permute(1, 2, 0).numpy()
    orig = (orig + 1) / 2
    axes[0, 0].imshow(orig)
    axes[0, 0].set_title('Original Target')
    axes[0, 0].axis('off')

    # Generated raw
    gen_raw = results['generated_raw'][0].cpu().permute(1, 2, 0).numpy()
    gen_raw = (gen_raw + 1) / 2
    axes[0, 1].imshow(gen_raw)
    axes[0, 1].set_title('Generated (Raw)')
    axes[0, 1].axis('off')

    # Generated final
    gen_final = results['generated'][0].cpu().permute(1, 2, 0).numpy()
    gen_final = (gen_final + 1) / 2
    axes[0, 2].imshow(gen_final)
    axes[0, 2].set_title('Generated (Final)')
    axes[0, 2].axis('off')

    # Face mask
    mask = results['face_mask'][0, 0].cpu().numpy()
    axes[0, 3].imshow(mask, cmap='gray')
    axes[0, 3].set_title('Face Mask')
    axes[0, 3].axis('off')

    # Row 2: Warps and volumes
    # XY warp
    xy_warp = results['xy_warps'][0, 8].cpu().numpy()
    xy_mag = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
    axes[1, 0].imshow(xy_mag, cmap='viridis')
    axes[1, 0].set_title('XY Warp (→Canonical)')
    axes[1, 0].axis('off')

    # UV warp
    uv_warp = results['uv_warps'][0, 8].cpu().numpy()
    uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
    axes[1, 1].imshow(uv_mag, cmap='plasma')
    axes[1, 1].set_title('UV Warp (→Target)')
    axes[1, 1].axis('off')

    # Canonical volume (slice)
    canonical = results['canonical_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 2].imshow(canonical, cmap='coolwarm')
    axes[1, 2].set_title('Canonical Volume')
    axes[1, 2].axis('off')

    # Target volume (slice)
    target = results['target_volume'][0, :, 8, :, :].mean(0).cpu().numpy()
    axes[1, 3].imshow(target, cmap='coolwarm')
    axes[1, 3].set_title('Target Volume')
    axes[1, 3].axis('off')

    plt.suptitle('Single Frame Generation Pipeline with UV Warps', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved pipeline visualization to {save_path}")


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
    """Main test function."""

    # Load model
    model = load_volumetric_model()

    # Initialize generator
    generator = SingleFrameGenerator(model, cache_path="proper_face_attributes.h5")

    # Load cache
    generator.load_cache()

    logger.info(f"Cache loaded with {generator.cached_attributes['num_frames']} frames")

    # Generate multiple frames for comparison
    test_indices = [0, 5, 10, 15, 20] if generator.cached_attributes['num_frames'] > 20 else range(min(5, generator.cached_attributes['num_frames']))

    fig, axes = plt.subplots(3, len(test_indices), figsize=(len(test_indices) * 3, 9))

    for i, idx in enumerate(test_indices):
        logger.info(f"\nGenerating frame {idx}...")

        # Generate target frame
        results = generator.generate_target_frame(target_frame_idx=idx)

        # Original
        orig = results['original'][0].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')

        # Generated
        gen = results['generated'][0].cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1) / 2
        axes[1, i].imshow(gen)
        axes[1, i].set_title(f'Generated {idx}')
        axes[1, i].axis('off')

        # UV warp
        uv_warp = results['uv_warps'][0, 8].cpu().numpy()
        uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
        axes[2, i].imshow(uv_mag, cmap='plasma')
        axes[2, i].set_title('UV Warp')
        axes[2, i].axis('off')

    plt.suptitle('Single Frame Generation Using Cached UV Warps', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("single_frame_generation_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Visualize pipeline for first frame
    logger.info("\nVisualizing generation pipeline...")
    results = generator.generate_target_frame(target_frame_idx=0)
    visualize_generation_pipeline(results, "generation_pipeline_detailed.png")

    logger.info("\n=== Generation Complete ===")
    logger.info("Generated files:")
    logger.info("  - single_frame_generation_results.png: Multiple frame results")
    logger.info("  - generation_pipeline_detailed.png: Detailed pipeline visualization")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Single Frame Generation with Cached UV Warps")
    logger.info("Clear extraction and application of warps")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)