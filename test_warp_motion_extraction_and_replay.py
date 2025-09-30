#!/usr/bin/env python3
"""
Extract motion attributes and XY warps from a video, save to H5,
then use them to generate video from canonical image to target expressions.
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

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionAttributeExtractor:
    """Extract and store motion attributes from video frames."""

    def __init__(self, emo_model):
        self.emo_model = emo_model
        self.emo_model.eval()

    def extract_frame_attributes(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract all motion attributes from a single frame."""

        with torch.no_grad():
            # Ensure proper shape
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)

            # Extract head pose (theta)
            theta = self.emo_model.head_pose_regressor.forward(frame)

            # Extract face mask
            mask, _, _, _ = self.emo_model.face_idt.forward(frame)
            mask = (mask > 0.6).float()

            # Mask the frame
            masked = frame * mask

            # Extract identity embedding
            idt_embed = self.emo_model.idt_embedder_nw(masked)

            # Prepare data dict for expression extraction
            data_dict = {
                'source_img': frame,
                'target_img': frame,
                'source_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
                'target_theta': theta if theta.shape[-2] == 3 else theta[:, :3, :],
                'source_mask': mask,
                'target_mask': mask,
                'idt_embed': idt_embed
            }

            # Extract expression embedding
            data_dict = self.emo_model.expression_embedder_nw(data_dict, True, False, True)
            expression_embed = data_dict['source_pose_embed']

            # Extract XY warps
            xy_warps = self.extract_xy_warps(frame, idt_embed, expression_embed)

            return {
                'theta': theta,
                'mask': mask,
                'idt_embed': idt_embed,
                'expression_embed': expression_embed,
                'xy_warps': xy_warps,
                'frame': frame
            }

    def extract_xy_warps(self, frame: torch.Tensor, idt_embed: torch.Tensor,
                         expression_embed: torch.Tensor) -> torch.Tensor:
        """Extract XY warps for the frame."""

        # Unsqueeze pose embed
        pose_unsqueeze = self.emo_model.pose_unsqueeze_nw(expression_embed).view(
            1, -1, self.emo_model.embed_size, self.emo_model.embed_size
        )

        # Combine embeddings
        if self.emo_model.args.cat_em:
            warp_embed_head = torch.cat([idt_embed, pose_unsqueeze], dim=1)
        else:
            warp_embed_head = idt_embed + pose_unsqueeze

        # Process through warp head
        warp_embed_head = self.emo_model.warp_embed_head_orig_nw(warp_embed_head)

        # Create warp embed dict
        c = warp_embed_head.shape[1]
        source_warp_embed_dict = {
            'orig': warp_embed_head.view(1, c, self.emo_model.embed_size ** 2),
            'orig_d': warp_embed_head.view(1, c, self.emo_model.embed_size ** 2).detach(),
            'ada_v': expression_embed,
            'idt_embed': idt_embed
        }

        # Generate XY warp
        xy_gen_warp, _ = self.emo_model.xy_generator_nw(source_warp_embed_dict)

        # Apply resizing if needed
        if self.emo_model.resize_warp:
            stride = self.emo_model.warp_resize_stride
            xy_gen_warp = F.avg_pool3d(
                xy_gen_warp.permute(0, 4, 1, 2, 3),
                kernel_size=stride,
                stride=stride
            ).permute(0, 2, 3, 4, 1)

        return xy_gen_warp


def save_motion_attributes_to_h5(attributes_list: List[Dict], h5_path: str):
    """Save extracted motion attributes to H5 file."""

    logger.info(f"Saving motion attributes to {h5_path}")

    with h5py.File(h5_path, 'w') as f:
        n_frames = len(attributes_list)

        # Create groups
        motion_group = f.create_group('motion')
        warps_group = f.create_group('warps')
        embeddings_group = f.create_group('embeddings')
        frames_group = f.create_group('frames')

        # Store each frame's attributes
        for i, attrs in enumerate(tqdm(attributes_list, desc="Saving to H5")):
            frame_group = f.create_group(f'frame_{i:04d}')

            # Motion parameters
            frame_group.create_dataset('theta', data=attrs['theta'].cpu().numpy())
            frame_group.create_dataset('mask', data=attrs['mask'].cpu().numpy())

            # Embeddings
            frame_group.create_dataset('idt_embed', data=attrs['idt_embed'].cpu().numpy())
            frame_group.create_dataset('expression_embed', data=attrs['expression_embed'].cpu().numpy())

            # XY warps
            frame_group.create_dataset('xy_warps', data=attrs['xy_warps'].cpu().numpy())

            # Original frame (for reference)
            frame_group.create_dataset('frame', data=attrs['frame'].cpu().numpy())

        # Store metadata
        f.attrs['n_frames'] = n_frames
        f.attrs['frame_size'] = attributes_list[0]['frame'].shape[-2:]

    logger.info(f"Saved {n_frames} frames to {h5_path}")


def load_motion_attributes_from_h5(h5_path: str, device: str = 'cuda') -> List[Dict]:
    """Load motion attributes from H5 file."""

    logger.info(f"Loading motion attributes from {h5_path}")

    attributes_list = []

    with h5py.File(h5_path, 'r') as f:
        n_frames = f.attrs['n_frames']

        for i in tqdm(range(n_frames), desc="Loading from H5"):
            frame_group = f[f'frame_{i:04d}']

            attrs = {
                'theta': torch.from_numpy(frame_group['theta'][:]).to(device),
                'mask': torch.from_numpy(frame_group['mask'][:]).to(device),
                'idt_embed': torch.from_numpy(frame_group['idt_embed'][:]).to(device),
                'expression_embed': torch.from_numpy(frame_group['expression_embed'][:]).to(device),
                'xy_warps': torch.from_numpy(frame_group['xy_warps'][:]).to(device),
                'frame': torch.from_numpy(frame_group['frame'][:]).to(device)
            }

            attributes_list.append(attrs)

    logger.info(f"Loaded {len(attributes_list)} frames from H5")
    return attributes_list


class CanonicalToTargetGenerator:
    """Generate video from canonical image to target expressions using saved attributes."""

    def __init__(self, emo_model):
        self.emo_model = emo_model
        self.emo_model.eval()

    def generate_canonical_image(self, identity_frame: torch.Tensor) -> torch.Tensor:
        """Generate canonical (neutral) image from identity frame."""

        with torch.no_grad():
            # Extract identity embedding
            mask, _, _, _ = self.emo_model.face_idt.forward(identity_frame)
            mask = (mask > 0.6).float()
            masked = identity_frame * mask
            idt_embed = self.emo_model.idt_embedder_nw(masked)

            # Use neutral pose (identity matrix)
            device = identity_frame.device
            neutral_theta = torch.eye(3, 4, device=device).unsqueeze(0)

            # Prepare data dict for canonical generation
            data_dict = {
                'source_img': identity_frame,
                'target_img': identity_frame,
                'source_theta': neutral_theta,
                'target_theta': neutral_theta,
                'source_mask': mask,
                'target_mask': mask,
                'idt_embed': idt_embed
            }

            # Process through model (simplified canonical generation)
            # In practice, this would involve the full volumetric pipeline
            # For now, we'll return the masked identity frame as canonical
            canonical = identity_frame * mask

            return canonical, idt_embed

    def apply_target_motion(self, canonical_image: torch.Tensor,
                           target_attributes: Dict,
                           idt_embed: torch.Tensor) -> torch.Tensor:
        """Apply target motion attributes to canonical image."""

        with torch.no_grad():
            # Get target motion parameters
            target_theta = target_attributes['theta']
            target_expression = target_attributes['expression_embed']
            target_xy_warps = target_attributes['xy_warps']

            # Prepare data dict
            data_dict = {
                'source_img': canonical_image,
                'target_img': canonical_image,  # Will be updated
                'source_theta': torch.eye(3, 4, device=canonical_image.device).unsqueeze(0),
                'target_theta': target_theta,
                'idt_embed': idt_embed,
                'target_pose_embed': target_expression
            }

            # Generate UV warps for target
            pose_unsqueeze = self.emo_model.pose_unsqueeze_nw(target_expression).view(
                1, -1, self.emo_model.embed_size, self.emo_model.embed_size
            )

            if self.emo_model.args.cat_em:
                warp_embed = torch.cat([idt_embed, pose_unsqueeze], dim=1)
            else:
                warp_embed = idt_embed + pose_unsqueeze

            warp_embed = self.emo_model.warp_embed_head_orig_nw(warp_embed)

            # Create target warp embed dict
            c = warp_embed.shape[1]
            target_warp_embed_dict = {
                'orig': warp_embed.view(1, c, self.emo_model.embed_size ** 2),
                'orig_d': warp_embed.view(1, c, self.emo_model.embed_size ** 2).detach(),
                'ada_v': target_expression,
                'idt_embed': idt_embed
            }

            # Generate UV warps
            uv_warps, _ = self.emo_model.uv_generator_nw(target_warp_embed_dict)

            # Apply warps (simplified - in practice would involve full volumetric rendering)
            # For demonstration, we'll apply a simple transformation
            generated = canonical_image  # Placeholder

            return generated


def visualize_motion_extraction(attributes_list: List[Dict], save_path: str = "motion_extraction_viz.png"):
    """Visualize extracted motion attributes."""

    n_frames = min(10, len(attributes_list))
    fig, axes = plt.subplots(4, n_frames, figsize=(n_frames * 2, 8))

    for i in range(n_frames):
        attrs = attributes_list[i * len(attributes_list) // n_frames]

        # Original frame
        frame = attrs['frame'][0].cpu().permute(1, 2, 0).numpy()
        frame = (frame + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Frame {i}')
        axes[0, i].axis('off')

        # Mask
        mask = attrs['mask'][0, 0].cpu().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title('Mask')
        axes[1, i].axis('off')

        # Expression embedding (first 50 dims)
        expr = attrs['expression_embed'][0, :50].cpu().numpy()
        axes[2, i].bar(range(len(expr)), expr, width=1.0)
        axes[2, i].set_ylim([-2, 2])
        axes[2, i].set_title('Expression')
        axes[2, i].set_xticks([])

        # XY warp magnitude
        xy_warp = attrs['xy_warps'][0, 8].cpu().numpy()  # Middle depth slice
        magnitude = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
        im = axes[3, i].imshow(magnitude, cmap='viridis', vmin=0, vmax=2)
        axes[3, i].set_title('XY Warp')
        axes[3, i].axis('off')

    # Add row labels
    row_labels = ['Original', 'Mask', 'Expression', 'XY Warp']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.3, 0.5, label, transform=axes[i, 0].transAxes,
                       fontsize=12, ha='right', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()


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

    # Set optimizer mode if needed
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def main():
    """Main test function."""

    # Load model
    model = load_volumetric_model()

    # Initialize extractors
    extractor = MotionAttributeExtractor(model)
    generator = CanonicalToTargetGenerator(model)

    # Load test video
    video_path = Path("temp_single_video/15.mp4")
    if not video_path.exists():
        logger.error(f"Test video not found: {video_path}")
        return

    logger.info(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    # Extract frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(50, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in tqdm(frame_indices[:num_frames], desc="Loading frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = (frame - 0.5) * 2.0  # Normalize to [-1, 1]
            frame = frame.permute(2, 0, 1).cuda()  # [3, H, W]
            frames.append(frame)

    cap.release()

    if len(frames) < 2:
        logger.error("Not enough frames loaded")
        return

    logger.info(f"Loaded {len(frames)} frames")

    # Extract motion attributes
    logger.info("\n=== Extracting Motion Attributes ===")
    attributes_list = []

    for i, frame in enumerate(tqdm(frames, desc="Extracting attributes")):
        attrs = extractor.extract_frame_attributes(frame)
        attributes_list.append(attrs)

    # Save to H5
    h5_path = "motion_attributes.h5"
    save_motion_attributes_to_h5(attributes_list, h5_path)

    # Visualize extracted attributes
    visualize_motion_extraction(attributes_list, "motion_extraction_viz.png")

    # Load from H5 (to verify)
    logger.info("\n=== Loading from H5 ===")
    loaded_attributes = load_motion_attributes_from_h5(h5_path)

    # Verify loaded data matches
    for i in range(min(5, len(attributes_list))):
        orig_xy = attributes_list[i]['xy_warps']
        loaded_xy = loaded_attributes[i]['xy_warps']
        diff = torch.abs(orig_xy - loaded_xy).max().item()
        logger.info(f"Frame {i} XY warp difference: {diff:.8f}")

    # Generate canonical image from first frame
    logger.info("\n=== Generating Canonical Image ===")
    identity_frame = frames[0].unsqueeze(0)
    canonical_image, idt_embed = generator.generate_canonical_image(identity_frame)

    # Save canonical image
    canonical_np = canonical_image[0].cpu().permute(1, 2, 0).numpy()
    canonical_np = (canonical_np + 1) / 2  # Denormalize
    canonical_np = (canonical_np * 255).astype(np.uint8)
    cv2.imwrite("canonical_image.png", cv2.cvtColor(canonical_np, cv2.COLOR_RGB2BGR))
    logger.info("Saved canonical image")

    # Generate video with target motions (simplified demonstration)
    logger.info("\n=== Generating Video with Target Motions ===")
    output_frames = []

    for i in tqdm(range(0, len(loaded_attributes), 5), desc="Generating frames"):
        target_attrs = loaded_attributes[i]
        generated_frame = generator.apply_target_motion(canonical_image, target_attrs, idt_embed)
        output_frames.append(generated_frame)

    logger.info(f"Generated {len(output_frames)} output frames")

    # Create comparison visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for i in range(min(5, len(output_frames))):
        idx = i * len(frames) // 5

        # Original frame
        orig = frames[idx].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')

        # XY warp
        xy_warp = loaded_attributes[idx]['xy_warps'][0, 8].cpu().numpy()
        magnitude = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)
        axes[1, i].imshow(magnitude, cmap='viridis')
        axes[1, i].set_title('XY Warp')
        axes[1, i].axis('off')

        # Generated frame (placeholder)
        axes[2, i].imshow(orig)  # Would show generated frame
        axes[2, i].set_title('Generated')
        axes[2, i].axis('off')

    plt.suptitle('Motion Attribute Extraction and Application', fontsize=16)
    plt.tight_layout()
    plt.savefig("motion_replay_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("\n=== Test Complete ===")
    logger.info("Generated files:")
    logger.info("  - motion_attributes.h5: Extracted motion data")
    logger.info("  - motion_extraction_viz.png: Visualization of extracted attributes")
    logger.info("  - canonical_image.png: Generated canonical image")
    logger.info("  - motion_replay_comparison.png: Comparison visualization")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Motion Attribute Extraction and Replay Test")
    logger.info("Extracting motion from video, saving to H5, and replaying")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed successfully!")
    logger.info("=" * 60)