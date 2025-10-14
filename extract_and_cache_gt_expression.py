"""
Extract ground truth expression embeddings from video and cache to H5 file.

This allows us to use GT expressions during inference to isolate audio-expression coupling issues.
"""

import torch
import cv2
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import importlib
from logger import logger
import argparse
from tqdm import tqdm


def extract_gt_expressions_to_h5(
    video_path: str,
    h5_path: str,
    checkpoint_path: str,
    config_path: str,
    fps: float = 25.0,
    force: bool = False
):
    """
    Extract GT expression embeddings from video frames and save to H5.

    Args:
        video_path: Path to input video
        h5_path: Path to save H5 cache file
        checkpoint_path: Path to VASA checkpoint (for EMO model)
        config_path: Path to config file
        fps: Target FPS (default: 25.0)
        force: Force overwrite if H5 file exists
    """

    h5_path = Path(h5_path)
    if h5_path.exists() and not force:
        logger.info(f"H5 cache already exists: {h5_path}")
        logger.info("Use --force to overwrite")
        return

    # Ensure cache directory exists
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load EMO model
    logger.info("Loading EMO model...")
    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)

    model_dict = torch.load(model_path, map_location=device, weights_only=False)
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.to(device)
    volumetric_avatar.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Open video
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video FPS: {video_fps}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Target FPS: {fps}")

    # Calculate frame sampling
    frame_interval = video_fps / fps
    target_frame_count = int(total_frames / frame_interval)

    logger.info(f"Will extract {target_frame_count} frames at {fps} FPS")

    # Storage for GT expressions
    gt_expressions = []

    try:
        with torch.no_grad():
            frame_idx = 0
            extracted_count = 0

            pbar = tqdm(total=target_frame_count, desc="Extracting expressions")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames at target FPS
                if frame_idx % int(frame_interval) == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

                    # Extract expression embedding
                    # Get face mask
                    face_mask = volumetric_avatar.face_idt.forward(frame_tensor)[0]
                    face_mask = (face_mask > 0.6).float()
                    frame_masked = frame_tensor * face_mask

                    # Get identity embedding
                    idt_embed = volumetric_avatar.idt_embedder_nw(frame_masked)

                    # Get theta for expression embedder
                    theta, _, _, _ = volumetric_avatar.head_pose_regressor.forward(
                        frame_tensor, return_srt=True
                    )

                    # Prepare data dict for expression embedder
                    data_dict = {
                        'source_img': frame_tensor,
                        'source_mask': face_mask,
                        'source_theta': theta,
                        'target_img': frame_tensor,
                        'target_mask': face_mask,
                        'target_theta': theta,
                        'idt_embed': idt_embed
                    }

                    # Extract expression embedding
                    data_dict = volumetric_avatar.expression_embedder_nw(
                        data_dict, True, False, False
                    )
                    expression_embed = data_dict['source_pose_embed']  # [1, 128]

                    gt_expressions.append(expression_embed.cpu())

                    extracted_count += 1
                    pbar.update(1)

                    if extracted_count >= target_frame_count:
                        break

                frame_idx += 1

            pbar.close()

    finally:
        cap.release()

    # Stack all expressions
    gt_expressions = torch.cat(gt_expressions, dim=0)  # [N, 128]

    logger.info(f"\nExtracted {gt_expressions.shape[0]} expression embeddings")
    logger.info(f"Expression shape: {gt_expressions.shape}")
    logger.info(f"Expression range: [{gt_expressions.min():.3f}, {gt_expressions.max():.3f}]")

    # Save to H5
    logger.info(f"Saving to H5: {h5_path}")
    with h5py.File(h5_path, 'w') as f:
        # Save expression embeddings
        f.create_dataset('expression_embed', data=gt_expressions.numpy())

        # Save metadata
        f.attrs['video_path'] = str(video_path)
        f.attrs['num_frames'] = gt_expressions.shape[0]
        f.attrs['video_fps'] = video_fps
        f.attrs['target_fps'] = fps
        f.attrs['expression_dim'] = gt_expressions.shape[1]

    logger.info(f"✅ Successfully cached GT expressions to {h5_path}")
    logger.info(f"   Frames: {gt_expressions.shape[0]}")
    logger.info(f"   Expression dimension: {gt_expressions.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract GT expressions to H5 cache')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output H5 file (default: cache/<video>_gt_expression.h5)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints_overfit/best_checkpoint.pt',
                        help='Path to VASA checkpoint')
    parser.add_argument('--config', type=str, default='overfit_config.yaml',
                        help='Path to config file')
    parser.add_argument('--fps', type=float, default=25.0,
                        help='Target FPS for extraction')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite if H5 exists')

    args = parser.parse_args()

    # Generate default output path if not provided
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"cache/{video_name}_gt_expression.h5"

    extract_gt_expressions_to_h5(
        video_path=args.video,
        h5_path=args.output,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        fps=args.fps,
        force=args.force
    )
