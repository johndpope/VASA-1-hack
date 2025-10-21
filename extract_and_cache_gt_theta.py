#!/usr/bin/env python3
"""
Extract ground truth theta/SRT from video frames and cache to H5 file.
This avoids expensive recomputation and OOM issues during GT theta testing.
"""
import torch
import cv2
import h5py
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
sys.path.insert(0, 'nemo')
from logger import logger
from vi import VASAInference
import argparse

def extract_gt_theta_to_h5(
    video_path: str,
    h5_path: str,
    checkpoint_path: str,
    config_path: str,
    fps: float = 25.0,
    force: bool = False
):
    """Extract GT theta/SRT from video and save to H5 file."""

    h5_file = Path(h5_path)
    if h5_file.exists() and not force:
        logger.info(f"H5 cache already exists: {h5_path}")
        logger.info("Use --force to overwrite")
        return

    logger.info("="*80)
    logger.info("EXTRACTING GT THETA TO H5 CACHE")
    logger.info("="*80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {h5_path}")
    logger.info(f"Target FPS: {fps}")

    # Load model for head pose extraction
    inferencer = VASAInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Extract GT theta from each frame
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING THETA FROM VIDEO FRAMES")
    logger.info("="*80)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Video: {total_frames} frames @ {video_fps:.2f} fps")

    gt_theta_list = []
    gt_scale_list = []
    gt_rotation_list = []
    gt_translation_list = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
        frame_tensor = transform(frame_pil).unsqueeze(0).to(inferencer.device)

        # Extract theta/SRT using head_pose_regressor
        with torch.no_grad():
            theta_4x4, scale, rotation, translation = inferencer.volumetric_avatar.head_pose_regressor.forward(
                frame_tensor, return_srt=True
            )
            theta = theta_4x4[:, :3, :]  # Convert to 3x4

        gt_theta_list.append(theta.cpu())
        gt_scale_list.append(scale.cpu())
        gt_rotation_list.append(rotation.cpu())
        gt_translation_list.append(translation.cpu())

        frame_idx += 1
        if frame_idx % 25 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()

    # Stack into tensors
    gt_theta = torch.cat(gt_theta_list, dim=0)  # [T, 3, 4]
    gt_scale = torch.cat(gt_scale_list, dim=0)  # [T, 3]
    gt_rotation = torch.cat(gt_rotation_list, dim=0)  # [T, 3]
    gt_translation = torch.cat(gt_translation_list, dim=0)  # [T, 3]

    logger.info(f"\nExtracted GT theta: {gt_theta.shape}")
    logger.info(f"  Scale range: [{gt_scale.min().item():.2f}, {gt_scale.max().item():.2f}]")
    logger.info(f"  Rotation range: [{gt_rotation.min().item():.2f}, {gt_rotation.max().item():.2f}] rad")
    logger.info(f"  Translation range: [{gt_translation.min().item():.2f}, {gt_translation.max().item():.2f}]")

    # Save to H5 file
    logger.info(f"\nSaving to H5: {h5_path}")
    h5_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'w') as f:
        # Store tensors as numpy arrays
        f.create_dataset('theta', data=gt_theta.numpy())
        f.create_dataset('scale', data=gt_scale.numpy())
        f.create_dataset('rotation', data=gt_rotation.numpy())
        f.create_dataset('translation', data=gt_translation.numpy())

        # Store metadata
        f.attrs['video_path'] = video_path
        f.attrs['num_frames'] = total_frames
        f.attrs['video_fps'] = video_fps
        f.attrs['target_fps'] = fps
        f.attrs['checkpoint'] = checkpoint_path
        f.attrs['config'] = config_path

    logger.info("✅ GT theta cached to H5 file")
    logger.info("="*80)

def load_gt_theta_from_h5(h5_path: str, device='cuda'):
    """Load cached GT theta/SRT from H5 file."""
    logger.info(f"Loading GT theta from H5: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        gt_params = {
            'theta': torch.from_numpy(f['theta'][:]).to(device),
            'scale': torch.from_numpy(f['scale'][:]).to(device),
            'rotation': torch.from_numpy(f['rotation'][:]).to(device),
            'translation': torch.from_numpy(f['translation'][:]).to(device)
        }

        # Log metadata
        logger.info(f"  Video: {f.attrs['video_path']}")
        logger.info(f"  Frames: {f.attrs['num_frames']}")
        logger.info(f"  FPS: {f.attrs['video_fps']:.2f} -> {f.attrs['target_fps']:.2f}")
        logger.info(f"  Theta shape: {gt_params['theta'].shape}")

    return gt_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and cache GT theta from video')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='H5 output path (default: auto-generated)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_checkpoint.pt')
    parser.add_argument('--config', type=str, default='vasa_config.yaml')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--force', action='store_true', help='Overwrite existing cache')

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"cache/{video_name}_gt_theta.h5"

    extract_gt_theta_to_h5(
        video_path=args.video,
        h5_path=args.output,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        fps=args.fps,
        force=args.force
    )
