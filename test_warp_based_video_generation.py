#!/usr/bin/env python3
"""
Complete implementation of warp-based video generation using EMOPortraits model.
This properly applies XY warps (source->canonical) and UV warps (canonical->target).
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

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarpBasedVideoGenerator:
    """Generate video using proper warp application through the EMOPortraits pipeline."""

    def __init__(self, emo_model):
        self.emo_model = emo_model
        self.emo_model.eval()

    def extract_motion_sequence(self, video_path: str, num_frames: int = 50) -> Dict:
        """Extract complete motion sequence from video."""

        logger.info(f"Extracting motion sequence from {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

        motion_data = {
            'frames': [],
            'thetas': [],
            'expressions': [],
            'xy_warps': [],
            'uv_warps': [],
            'masks': [],
            'idt_embeds': []
        }

        # Get first frame for identity
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, identity_frame = cap.read()
        if ret:
            identity_frame = cv2.cvtColor(identity_frame, cv2.COLOR_BGR2RGB)
            identity_frame = cv2.resize(identity_frame, (512, 512))
            identity_frame = torch.from_numpy(identity_frame).float() / 255.0
            identity_frame = (identity_frame - 0.5) * 2.0  # [-1, 1]
            identity_frame = identity_frame.permute(2, 0, 1).cuda()

        # Extract identity embedding once
        with torch.no_grad():
            identity_frame_batch = identity_frame.unsqueeze(0)
            mask, _, _, _ = self.emo_model.face_idt.forward(identity_frame_batch)
            mask = (mask > 0.6).float()
            masked = identity_frame_batch * mask
            identity_embed = self.emo_model.idt_embedder_nw(masked)

        # Process each frame
        for idx in tqdm(frame_indices, desc="Extracting motion"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Preprocess frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (512, 512))
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensor = (frame_tensor - 0.5) * 2.0
                frame_tensor = frame_tensor.permute(2, 0, 1).cuda().unsqueeze(0)

                # Extract motion attributes
                with torch.no_grad():
                    # Get theta
                    theta = self.emo_model.head_pose_regressor.forward(frame_tensor)

                    # Get mask
                    mask, _, _, _ = self.emo_model.face_idt.forward(frame_tensor)
                    mask = (mask > 0.6).float()

                    # Prepare data dict
                    data_dict = {
                        'source_img': frame_tensor,
                        'target_img': frame_tensor,
                        'source_theta': theta[:, :3, :] if theta.shape[-2] == 4 else theta,
                        'target_theta': theta[:, :3, :] if theta.shape[-2] == 4 else theta,
                        'source_mask': mask,
                        'target_mask': mask,
                        'idt_embed': identity_embed
                    }

                    # Get expression embedding
                    data_dict = self.emo_model.expression_embedder_nw(data_dict, True, False, True)
                    expression = data_dict['source_pose_embed']

                    # Get warps through predict_embed
                    source_warp_embed_dict, target_warp_embed_dict, _, _ = self.emo_model.predict_embed(data_dict)

                    # Generate XY and UV warps
                    xy_warp, _ = self.emo_model.xy_generator_nw(source_warp_embed_dict)
                    uv_warp, _ = self.emo_model.uv_generator_nw(target_warp_embed_dict)

                    # Apply resizing if needed
                    if self.emo_model.resize_warp:
                        stride = self.emo_model.warp_resize_stride
                        xy_warp = F.avg_pool3d(
                            xy_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)

                        uv_warp = F.avg_pool3d(
                            uv_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)

                # Store data
                motion_data['frames'].append(frame_tensor)
                motion_data['thetas'].append(theta)
                motion_data['expressions'].append(expression)
                motion_data['xy_warps'].append(xy_warp)
                motion_data['uv_warps'].append(uv_warp)
                motion_data['masks'].append(mask)
                motion_data['idt_embeds'].append(identity_embed)

        cap.release()
        motion_data['identity_frame'] = identity_frame_batch
        motion_data['identity_embed'] = identity_embed

        return motion_data

    def generate_video_with_motion(self, motion_data: Dict, output_path: str = "generated_video.mp4"):
        """Generate video using extracted motion data."""

        logger.info("Generating video with extracted motion")

        # Get identity frame and embedding
        identity_frame = motion_data['identity_frame']
        identity_embed = motion_data['identity_embed']

        generated_frames = []

        for i in tqdm(range(len(motion_data['frames'])), desc="Generating frames"):
            with torch.no_grad():
                # Get target motion
                target_theta = motion_data['thetas'][i]
                target_expression = motion_data['expressions'][i]
                target_uv_warp = motion_data['uv_warps'][i]

                # Prepare data dict for generation
                data_dict = {
                    'source_img': identity_frame,
                    'target_img': identity_frame,  # Will be replaced
                    'source_theta': torch.eye(3, 4).cuda().unsqueeze(0),
                    'target_theta': target_theta[:, :3, :] if target_theta.shape[-2] == 4 else target_theta,
                    'source_mask': motion_data['masks'][i],
                    'target_mask': motion_data['masks'][i],
                    'idt_embed': identity_embed,
                    'source_pose_embed': motion_data['expressions'][0],  # Neutral expression
                    'target_pose_embed': target_expression
                }

                # Generate through forward pass
                generated_dict, _, _, output_dict = self.emo_model.forward(
                    data_dict,
                    phase='test',
                    optimizer_idx=0,
                    visualize=False
                )

                # Extract generated frame
                if 'pred_target_img' in output_dict:
                    generated = output_dict['pred_target_img']
                elif 'fake' in generated_dict:
                    generated = generated_dict['fake']
                else:
                    # Fallback: use original frame
                    generated = motion_data['frames'][i]
                    logger.warning(f"Using original frame {i} as fallback")

                generated_frames.append(generated)

        # Save video
        self._save_video(generated_frames, output_path)

        return generated_frames

    def _save_video(self, frames: List[torch.Tensor], output_path: str, fps: int = 25):
        """Save generated frames as video."""

        writer = imageio.get_writer(output_path, fps=fps)

        for frame in frames:
            # Convert to numpy
            if frame.dim() == 4:
                frame = frame[0]
            frame_np = frame.cpu().permute(1, 2, 0).numpy()
            frame_np = (frame_np + 1) / 2  # [-1, 1] to [0, 1]
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(frame_np)

        writer.close()
        logger.info(f"Saved video to {output_path}")


def visualize_warp_differences(motion_data: Dict, save_path: str = "warp_differences.png"):
    """Visualize the differences between warps across frames."""

    n_frames = min(10, len(motion_data['xy_warps']))
    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 2, 6))

    # Take middle depth slice
    mid_depth = motion_data['xy_warps'][0].shape[1] // 2

    for i in range(n_frames):
        idx = i * len(motion_data['xy_warps']) // n_frames

        # XY warp magnitude
        xy_warp = motion_data['xy_warps'][idx][0, mid_depth].cpu().numpy()
        xy_mag = np.sqrt(xy_warp[..., 0]**2 + xy_warp[..., 1]**2)

        im0 = axes[0, i].imshow(xy_mag, cmap='viridis', vmin=0, vmax=2)
        axes[0, i].set_title(f'Frame {idx}', fontsize=8)
        axes[0, i].axis('off')

        # UV warp magnitude
        uv_warp = motion_data['uv_warps'][idx][0, mid_depth].cpu().numpy()
        uv_mag = np.sqrt(uv_warp[..., 0]**2 + uv_warp[..., 1]**2)

        im1 = axes[1, i].imshow(uv_mag, cmap='plasma', vmin=0, vmax=2)
        axes[1, i].axis('off')

        # Difference from first frame
        if i > 0:
            xy_warp_0 = motion_data['xy_warps'][0][0, mid_depth].cpu().numpy()
            xy_mag_0 = np.sqrt(xy_warp_0[..., 0]**2 + xy_warp_0[..., 1]**2)
            xy_diff_mag = np.abs(xy_mag - xy_mag_0)
        else:
            xy_diff_mag = np.zeros_like(xy_mag)

        im2 = axes[2, i].imshow(xy_diff_mag, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.15, 0.5, 'XY Warp', transform=axes[0, 0].transAxes, rotation=90, va='center')
    axes[1, 0].text(-0.15, 0.5, 'UV Warp', transform=axes[1, 0].transAxes, rotation=90, va='center')
    axes[2, 0].text(-0.15, 0.5, 'XY Diff', transform=axes[2, 0].transAxes, rotation=90, va='center')

    plt.suptitle('Warp Variations Across Frames', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved warp differences to {save_path}")


def compare_original_vs_generated(motion_data: Dict, generated_frames: List,
                                 save_path: str = "comparison.png"):
    """Compare original frames with generated ones."""

    n_compare = min(8, len(generated_frames))
    fig, axes = plt.subplots(2, n_compare, figsize=(n_compare * 2, 4))

    for i in range(n_compare):
        idx = i * len(generated_frames) // n_compare

        # Original frame
        orig = motion_data['frames'][idx][0].cpu().permute(1, 2, 0).numpy()
        orig = (orig + 1) / 2
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {idx}', fontsize=8)
        axes[0, i].axis('off')

        # Generated frame
        gen = generated_frames[idx]
        if gen.dim() == 4:
            gen = gen[0]
        gen = gen.cpu().permute(1, 2, 0).numpy()
        gen = (gen + 1) / 2
        axes[1, i].imshow(gen)
        axes[1, i].set_title(f'Generated {idx}', fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle('Original vs Generated Frames', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison to {save_path}")


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
    generator = WarpBasedVideoGenerator(model)

    # Extract motion from video
    video_path = Path("temp_single_video/15.mp4")
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    # Extract motion sequence
    motion_data = generator.extract_motion_sequence(str(video_path), num_frames=30)

    # Visualize warp differences
    visualize_warp_differences(motion_data, "warp_differences.png")

    # Check warp statistics
    logger.info("\n=== Warp Statistics ===")
    for i in range(min(5, len(motion_data['xy_warps']))):
        xy_warp = motion_data['xy_warps'][i]
        uv_warp = motion_data['uv_warps'][i]

        xy_mean = xy_warp.mean().item()
        xy_std = xy_warp.std().item()
        uv_mean = uv_warp.mean().item()
        uv_std = uv_warp.std().item()

        logger.info(f"Frame {i}: XY mean={xy_mean:.4f} std={xy_std:.4f}, UV mean={uv_mean:.4f} std={uv_std:.4f}")

    # Check inter-frame differences
    logger.info("\n=== Inter-frame Warp Differences ===")
    for i in range(1, min(5, len(motion_data['xy_warps']))):
        xy_diff = torch.abs(motion_data['xy_warps'][i] - motion_data['xy_warps'][0]).mean().item()
        uv_diff = torch.abs(motion_data['uv_warps'][i] - motion_data['uv_warps'][0]).mean().item()
        logger.info(f"Frame 0 vs {i}: XY diff={xy_diff:.4f}, UV diff={uv_diff:.4f}")

    # Generate video with motion
    generated_frames = generator.generate_video_with_motion(motion_data, "generated_video.mp4")

    # Create comparison
    compare_original_vs_generated(motion_data, generated_frames, "original_vs_generated.png")

    logger.info("\n=== Generation Complete ===")
    logger.info("Generated files:")
    logger.info("  - warp_differences.png: Visualizes XY/UV warp variations")
    logger.info("  - generated_video.mp4: Video with applied motion")
    logger.info("  - original_vs_generated.png: Frame comparison")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Warp-Based Video Generation Test")
    logger.info("Using proper EMOPortraits pipeline for motion transfer")
    logger.info("=" * 60)

    main()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)