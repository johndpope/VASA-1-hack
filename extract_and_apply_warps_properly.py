#!/usr/bin/env python3
"""
Extract face attributes from video and properly apply UV warps to generate target frames.
This implementation dissects the warping process to understand UV warp application.
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import imageio
from PIL import Image

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProperWarpExtractor:
    """Extract and apply warps correctly using EMOPortraits volumetric avatar model."""

    def __init__(self, emo_model):
        self.emo_model = emo_model
        self.emo_model.eval()

    def extract_face_attributes_from_video(self, video_path: str,
                                          cache_path: str = "face_attributes_cache.h5",
                                          num_frames: Optional[int] = None) -> Dict:
        """
        Extract complete face attributes from video including proper UV warps.

        Args:
            video_path: Path to input video
            cache_path: Path to save cached attributes
            num_frames: Number of frames to process (None = all frames)

        Returns:
            Dictionary containing all extracted attributes
        """

        logger.info(f"Extracting face attributes from {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames is None:
            num_frames = total_frames
        else:
            num_frames = min(num_frames, total_frames)

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Storage for attributes
        attributes = {
            'frames': [],
            'thetas': [],
            'masks': [],
            'idt_embeds': [],
            'expression_embeds': [],
            'source_pose_embeds': [],
            'target_pose_embeds': [],
            'xy_warps': [],
            'uv_warps': [],
            'source_volumes': [],
            'canonical_volumes': []
        }

        # Get identity frame (first frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, identity_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        # Preprocess identity frame
        identity_frame = cv2.cvtColor(identity_frame, cv2.COLOR_BGR2RGB)
        identity_frame = cv2.resize(identity_frame, (512, 512))
        identity_tensor = torch.from_numpy(identity_frame).float() / 255.0
        identity_tensor = (identity_tensor - 0.5) * 2.0  # Normalize to [-1, 1]
        identity_tensor = identity_tensor.permute(2, 0, 1).cuda().unsqueeze(0)

        # Extract identity embedding once
        with torch.no_grad():
            # Get identity mask and embedding
            idt_mask, _, _, _ = self.emo_model.face_idt.forward(identity_tensor)
            idt_mask = (idt_mask > 0.6).float()
            masked_identity = identity_tensor * idt_mask
            identity_embed = self.emo_model.idt_embedder_nw(masked_identity)

            # Store identity info
            attributes['identity_frame'] = identity_tensor
            attributes['identity_embed'] = identity_embed
            attributes['identity_mask'] = idt_mask

        # Process each frame
        for idx in tqdm(frame_indices, desc="Extracting attributes"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frame_tensor = (frame_tensor - 0.5) * 2.0
            frame_tensor = frame_tensor.permute(2, 0, 1).cuda().unsqueeze(0)

            # Extract attributes for this frame
            attrs = self._extract_frame_attributes(
                frame_tensor,
                identity_tensor,
                identity_embed
            )

            # Store attributes
            for key, value in attrs.items():
                if key in attributes and isinstance(attributes[key], list):
                    attributes[key].append(value)

            # Store complete frame attributes
            attrs['idt_embed'] = identity_embed
            attributes.setdefault('frame_attributes', []).append(attrs)

        cap.release()

        # Save to cache
        if cache_path:
            self._save_attributes_to_cache(attributes, cache_path)

        return attributes

    def _extract_frame_attributes(self, target_frame: torch.Tensor,
                                 identity_frame: torch.Tensor,
                                 identity_embed: torch.Tensor) -> Dict:
        """
        Extract complete attributes for a single frame including proper warps.

        This follows the EMOPortraits forward pass to extract all necessary components.
        """

        with torch.no_grad():
            # Get target frame mask and theta
            target_mask, _, _, _ = self.emo_model.face_idt.forward(target_frame)
            target_mask = (target_mask > 0.6).float()
            target_theta = self.emo_model.head_pose_regressor.forward(target_frame)

            # Get source (identity) theta
            source_theta = self.emo_model.head_pose_regressor.forward(identity_frame)

            # Prepare data dict for expression extraction
            data_dict = {
                'source_img': identity_frame,
                'target_img': target_frame,
                'source_theta': source_theta[:, :3, :] if source_theta.shape[-2] == 4 else source_theta,
                'target_theta': target_theta[:, :3, :] if target_theta.shape[-2] == 4 else target_theta,
                'source_mask': target_mask,  # Using target mask for consistency
                'target_mask': target_mask,
                'idt_embed': identity_embed
            }

            # Extract expression embeddings
            data_dict = self.emo_model.expression_embedder_nw(data_dict, True, False, True)

            # Get source and target pose embeddings
            source_pose_embed = data_dict.get('source_pose_embed')
            target_pose_embed = data_dict.get('target_pose_embed')

            # Generate warp embeddings through predict_embed
            source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = self.emo_model.predict_embed(data_dict)

            # Generate XY warps (source to canonical)
            xy_warps, xy_conf = self.emo_model.xy_generator_nw(source_warp_embed_dict)

            # Generate UV warps (canonical to target)
            uv_warps, uv_conf = self.emo_model.uv_generator_nw(target_warp_embed_dict)

            # Apply resizing if needed
            if self.emo_model.resize_warp:
                stride = self.emo_model.warp_resize_stride

                xy_warps = F.avg_pool3d(
                    xy_warps.permute(0, 4, 1, 2, 3),
                    kernel_size=stride,
                    stride=stride
                ).permute(0, 2, 3, 4, 1)

                uv_warps = F.avg_pool3d(
                    uv_warps.permute(0, 4, 1, 2, 3),
                    kernel_size=stride,
                    stride=stride
                ).permute(0, 2, 3, 4, 1)

            # Extract source volume
            masked_source = identity_frame * target_mask
            source_latents = self.emo_model.local_encoder_nw(masked_source)

            c = self.emo_model.args.latent_volume_channels
            d = self.emo_model.args.latent_volume_depth
            s = self.emo_model.args.latent_volume_size
            source_volume = source_latents.view(1, c, d, s, s)

            # Process to canonical volume
            if self.emo_model.args.source_volume_num_blocks > 0:
                source_volume = self.emo_model.volume_source_nw(source_volume)

            canonical_volume = self.emo_model.volume_process_nw(source_volume)

            return {
                'frame': target_frame,
                'theta': target_theta,
                'mask': target_mask,
                'idt_embed': identity_embed,
                'expression_embed': target_pose_embed if target_pose_embed is not None else source_pose_embed,
                'source_pose_embed': source_pose_embed,
                'target_pose_embed': target_pose_embed,
                'xy_warps': xy_warps,
                'uv_warps': uv_warps,
                'source_volume': source_volume,
                'canonical_volume': canonical_volume
            }

    def generate_frame_with_cached_warps(self,
                                        identity_frame: torch.Tensor,
                                        target_attributes: Dict,
                                        use_uv_warps: bool = True) -> torch.Tensor:
        """
        Generate a target frame using cached warps and attributes.

        Args:
            identity_frame: Source identity frame
            target_attributes: Dictionary containing target warps and attributes
            use_uv_warps: Whether to use UV warps for generation

        Returns:
            Generated target frame
        """

        with torch.no_grad():
            # Extract components from target attributes
            target_theta = target_attributes['theta']
            target_uv_warps = target_attributes['uv_warps']
            target_xy_warps = target_attributes['xy_warps']
            canonical_volume = target_attributes['canonical_volume']
            target_pose_embed = target_attributes.get('target_pose_embed')
            idt_embed = target_attributes['idt_embed']

            # Apply XY warps to get canonical space
            aligned_canonical_volume = self.emo_model.grid_sample(canonical_volume, target_xy_warps)

            if use_uv_warps:
                # Apply UV warps to transform from canonical to target
                # Create rotation warp from target theta
                d = canonical_volume.shape[2]
                s = canonical_volume.shape[3]

                grid = self.emo_model.identity_grid_3d.repeat(1, 1, 1, 1, 1)[:1]
                rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)

                # Combine UV warp with rotation
                combined_warp = self._combine_warps(target_uv_warps, rotation_warp)

                # Apply combined warp
                target_volume = self.emo_model.grid_sample(aligned_canonical_volume, combined_warp)
            else:
                # Just apply rotation without UV warps
                grid = self.emo_model.identity_grid_3d.repeat(1, 1, 1, 1, 1)[:1]
                rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
                target_volume = self.emo_model.grid_sample(aligned_canonical_volume, rotation_warp)

            # Reshape for decoder
            c = canonical_volume.shape[1]
            d = canonical_volume.shape[2]
            s = canonical_volume.shape[3]
            target_latent_feats = target_volume.view(1, c * d, s, s)

            # Prepare data dict for decoder
            data_dict = {
                'target_theta': target_theta,
                'target_pose_embed': target_pose_embed if target_pose_embed is not None else target_attributes['expression_embed']
            }

            embed_dict = {'idt': idt_embed}

            # Decode to generate image
            generated_img, _, _, _ = self.emo_model.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            return generated_img

    def _combine_warps(self, warp1: torch.Tensor, warp2: torch.Tensor) -> torch.Tensor:
        """Combine two warp fields."""
        # Simple addition for demonstration - may need more sophisticated combination
        return warp1 + warp2

    def _save_attributes_to_cache(self, attributes: Dict, cache_path: str):
        """Save extracted attributes to H5 cache."""

        logger.info(f"Saving attributes to {cache_path}")

        with h5py.File(cache_path, 'w') as f:
            # Save metadata
            num_frames = len(attributes.get('frame_attributes', []))
            f.attrs['num_frames'] = num_frames

            # Save identity data
            f.create_dataset('identity_frame', data=attributes['identity_frame'].cpu().numpy())
            f.create_dataset('identity_embed', data=attributes['identity_embed'].cpu().numpy())
            f.create_dataset('identity_mask', data=attributes['identity_mask'].cpu().numpy())

            # Save per-frame data from frame_attributes
            if 'frame_attributes' in attributes:
                for i, frame_attrs in enumerate(tqdm(attributes['frame_attributes'], desc="Saving to cache")):
                    frame_group = f.create_group(f'frame_{i:04d}')

                    for key in ['frame', 'theta', 'mask', 'expression_embed',
                               'xy_warps', 'uv_warps', 'canonical_volume']:
                        if key in frame_attrs:
                            data = frame_attrs[key]
                            if isinstance(data, torch.Tensor):
                                data = data.cpu().numpy()
                            frame_group.create_dataset(key, data=data)

                logger.info(f"Saved {len(attributes['frame_attributes'])} frames to cache")
            else:
                logger.warning("No frame_attributes found to save")

    def load_attributes_from_cache(self, cache_path: str) -> Dict:
        """Load attributes from H5 cache."""

        logger.info(f"Loading attributes from {cache_path}")

        attributes = {
            'frames': [],
            'thetas': [],
            'masks': [],
            'expression_embeds': [],
            'xy_warps': [],
            'uv_warps': [],
            'canonical_volumes': []
        }

        with h5py.File(cache_path, 'r') as f:
            # Load identity data
            attributes['identity_frame'] = torch.from_numpy(f['identity_frame'][:]).cuda()
            attributes['identity_embed'] = torch.from_numpy(f['identity_embed'][:]).cuda()
            attributes['identity_mask'] = torch.from_numpy(f['identity_mask'][:]).cuda()

            # Load per-frame data
            num_frames = f.attrs['num_frames']

            for i in range(num_frames):
                frame_group = f[f'frame_{i:04d}']

                frame_attrs = {}
                for key in frame_group.keys():
                    data = torch.from_numpy(frame_group[key][:]).cuda()
                    frame_attrs[key] = data

                    if key in attributes:
                        attributes[key].append(data)

                # Also store as complete frame attributes
                frame_attrs['idt_embed'] = attributes['identity_embed']
                attributes.setdefault('frame_attributes', []).append(frame_attrs)

        logger.info(f"Loaded {num_frames} frames from cache")
        return attributes


def visualize_warp_analysis(attributes: Dict, save_path: str = "warp_analysis.png"):
    """Analyze and visualize UV vs XY warps to understand differences."""

    # Check what data we have
    if 'frame_attributes' in attributes and len(attributes['frame_attributes']) > 0:
        frames_data = attributes['frame_attributes']
        n_frames = min(8, len(frames_data))
    else:
        logger.warning("No frame_attributes found in attributes dict")
        return

    fig, axes = plt.subplots(4, n_frames, figsize=(n_frames * 2, 8))

    for i in range(n_frames):
        idx = i * len(frames_data) // n_frames
        frame_attrs = frames_data[idx]

        # Original frame
        if 'frame' in frame_attrs:
            frame = frame_attrs['frame'][0].cpu().permute(1, 2, 0).numpy()
            frame = (frame + 1) / 2
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f'Frame {idx}', fontsize=8)
            axes[0, i].axis('off')

        # Take middle depth slice for warps
        if 'xy_warps' in frame_attrs:
            mid_depth = frame_attrs['xy_warps'].shape[1] // 2
        else:
            mid_depth = 8

        # XY warp magnitude
        if 'xy_warps' in frame_attrs:
            xy_warp = frame_attrs['xy_warps'][0, mid_depth].cpu().numpy()
            xy_mag = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
            axes[1, i].imshow(xy_mag, cmap='viridis', vmin=0, vmax=2)
            axes[1, i].set_title('XY Warp', fontsize=8)
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')

        # UV warp magnitude
        if 'uv_warps' in frame_attrs:
            uv_warp = frame_attrs['uv_warps'][0, mid_depth].cpu().numpy()
            uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
            axes[2, i].imshow(uv_mag, cmap='plasma', vmin=0, vmax=2)
            axes[2, i].set_title('UV Warp', fontsize=8)
            axes[2, i].axis('off')
        else:
            axes[2, i].axis('off')

        # Difference between UV and XY
        if 'xy_warps' in frame_attrs and 'uv_warps' in frame_attrs:
            diff = np.abs(uv_mag - xy_mag)
            axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[3, i].set_title('UV-XY Diff', fontsize=8)
            axes[3, i].axis('off')
        else:
            axes[3, i].axis('off')

    # Add row labels
    row_labels = ['Original', 'XY Warp', 'UV Warp', 'Difference']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.15, 0.5, label, transform=axes[i, 0].transAxes,
                       rotation=90, va='center', fontsize=10)

    plt.suptitle('UV vs XY Warp Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved warp analysis to {save_path}")


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

    # Initialize extractor
    extractor = ProperWarpExtractor(model)

    # Video path
    video_path = Path("temp_single_video/15.mp4")
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    # Extract face attributes with proper UV warps
    logger.info("\n=== Extracting Face Attributes ===")
    attributes = extractor.extract_face_attributes_from_video(
        str(video_path),
        cache_path="proper_face_attributes.h5",
        num_frames=30
    )

    # Analyze warps
    visualize_warp_analysis(attributes, "uv_xy_warp_analysis.png")

    # Check UV warp variations
    logger.info("\n=== UV Warp Analysis ===")
    for i in range(min(5, len(attributes['uv_warps']))):
        uv_warp = attributes['uv_warps'][i]
        uv_mean = uv_warp.mean().item()
        uv_std = uv_warp.std().item()
        uv_max = uv_warp.abs().max().item()

        logger.info(f"Frame {i} UV warp: mean={uv_mean:.4f}, std={uv_std:.4f}, max={uv_max:.4f}")

    # Check differences between consecutive frames
    logger.info("\n=== Inter-frame UV Warp Differences ===")
    for i in range(1, min(5, len(attributes['uv_warps']))):
        diff = torch.abs(attributes['uv_warps'][i] - attributes['uv_warps'][i-1]).mean().item()
        logger.info(f"Frame {i-1} to {i}: UV warp difference = {diff:.6f}")

    # Test single frame generation with cached warps
    logger.info("\n=== Testing Single Frame Generation ===")

    # Get identity frame
    identity_frame = attributes['identity_frame']

    # Generate frames using cached warps
    generated_frames = []

    for i in tqdm(range(0, len(attributes['frame_attributes']), 5), desc="Generating frames"):
        target_attrs = attributes['frame_attributes'][i]

        # Generate with UV warps
        generated = extractor.generate_frame_with_cached_warps(
            identity_frame,
            target_attrs,
            use_uv_warps=True
        )

        generated_frames.append(generated)

    # Create comparison
    fig, axes = plt.subplots(3, len(generated_frames), figsize=(len(generated_frames) * 2, 6))

    for i, gen_frame in enumerate(generated_frames):
        idx = i * 5

        # Original
        orig = attributes['frames'][idx][0].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {idx}', fontsize=8)
        axes[0, i].axis('off')

        # Generated with UV
        gen = gen_frame[0].cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1) / 2
        axes[1, i].imshow(gen)
        axes[1, i].set_title('With UV', fontsize=8)
        axes[1, i].axis('off')

        # UV warp visualization
        uv_warp = attributes['uv_warps'][idx][0, 8].cpu().numpy()
        uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)
        axes[2, i].imshow(uv_mag, cmap='plasma')
        axes[2, i].set_title('UV Warp', fontsize=8)
        axes[2, i].axis('off')

    plt.suptitle('Single Frame Generation with Cached UV Warps', fontsize=14)
    plt.tight_layout()
    plt.savefig("single_frame_generation.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\n=== Test Complete ===")
    logger.info("Generated files:")
    logger.info("  - proper_face_attributes.h5: Cached face attributes with UV warps")
    logger.info("  - uv_xy_warp_analysis.png: Analysis of UV vs XY warps")
    logger.info("  - single_frame_generation.png: Single frame generation results")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Proper UV Warp Extraction and Application")
    logger.info("Dissecting the warping pipeline for single frame generation")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)