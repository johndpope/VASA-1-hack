#!/usr/bin/env python3
"""
Run inference using cached GT theta from H5 file.
This is memory-efficient and avoids recomputing expensive head pose extraction.
"""
import torch
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, 'nemo')
from logger import logger
from vi import VASAInference
from extract_and_cache_gt_theta import load_gt_theta_from_h5
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VASA Inference with Cached GT Theta')
    parser.add_argument('--config', type=str, default='vasa_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_checkpoint.pt')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--gt-theta-h5', type=str, required=True, help='Cached GT theta H5 file')
    parser.add_argument('--output', type=str, default='output_cached_gt_theta.mp4')
    parser.add_argument('--target_image', type=str, default='./data/IMG_1.png')
    parser.add_argument('--fps', type=float, default=25.0)

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("VASA INFERENCE WITH CACHED GT THETA")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Input video: {args.input}")
    logger.info(f"GT theta H5: {args.gt_theta_h5}")
    logger.info(f"Target image: {args.target_image}")
    logger.info(f"Output: {args.output}")

    # Create standard inferencer
    inferencer = VASAInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )

    # Load cached GT theta
    gt_params = load_gt_theta_from_h5(args.gt_theta_h5, device=inferencer.device)

    logger.info("\n" + "="*80)
    logger.info("RUNNING INFERENCE WITH CACHED GT THETA")
    logger.info("="*80)

    # Extract audio from video
    _, audio_path = inferencer.extract_video_assets(args.input, inferencer.asset_dir)

    # Load target image
    target_img = Image.open(args.target_image).convert('RGB').resize((512, 512))
    target_tensor = inferencer.transform(target_img).unsqueeze(0).to(inferencer.device)

    # Extract source params from target image
    source_params = inferencer.extract_source_params(target_tensor)

    # Load and process audio
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    audio_windows = inferencer.process_audio(waveform, sr=16000, fps=args.fps)

    # Modified generation: use cached GT theta
    logger.info("\n=== Starting Cached GT Theta Generation ===")
    device = source_params['theta'].device

    motion_data = {
        'theta': source_params['theta'],
        'expression_embed': source_params['expression_embed']
    }

    idt_embed = source_params['idt_embed']
    logger.info(f"Identity embeddings: {idt_embed.shape}")

    generated_frames = []
    frame_counter = 0

    for window_idx, window_data in enumerate(audio_windows):
        logger.info(f"Processing window {window_idx}/{len(audio_windows)}")

        audio_features = window_data['audio_features']
        B = audio_features.shape[0] if audio_features.dim() >= 2 else 1
        T = audio_features.shape[1] if audio_features.dim() >= 2 else audio_features.shape[0]

        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)

        cond_signals = {
            'audio_features': audio_features.to(device),
            'gaze': torch.zeros(B, T, 2, device=device),
            'head_distance': torch.zeros(B, T, 1, device=device),
            'emotion': torch.zeros(B, T, 2, device=device),
            'speed_bucket': torch.ones(B, T, 1, device=device) * 4,
        }

        # Generate sequence to get expression and UV warps
        motion_sequence = inferencer.model.generate_sequence(
            initial_pose=motion_data,
            initial_dynamics=motion_data['expression_embed'],
            conditions=cond_signals,
            idt_embed=idt_embed,
            eta=0.8,
            num_steps=50,
            cfg_scales=None
        )

        # Use cached GT theta instead of predicted
        logger.info(f"[CACHED GT] Using pre-extracted GT theta from H5")
        start_idx = 0 if window_idx == 0 else inferencer.stride

        for t in range(start_idx, motion_sequence['expression_embed'].size(1)):
            # Use cached GT theta/SRT
            gt_frame_idx = min(frame_counter, gt_params['theta'].shape[0] - 1)

            curr_expression = motion_sequence['expression_embed'][:, t]
            curr_theta = gt_params['theta'][gt_frame_idx].unsqueeze(0)  # Cached GT
            curr_scale = gt_params['scale'][gt_frame_idx]  # Cached GT
            curr_rotation = gt_params['rotation'][gt_frame_idx]  # Cached GT
            curr_translation = gt_params['translation'][gt_frame_idx]  # Cached GT
            curr_uv_warps = motion_sequence['uv_warps'][:, t]

            if frame_counter % 25 == 0:
                logger.info(f"Frame {frame_counter}: Using cached GT theta")

            # Generate frame with cached GT theta
            frame = inferencer._generate_frame(
                source_params,
                curr_expression,
                curr_theta,
                curr_rotation,
                curr_scale,
                curr_translation,
                device,
                uv_warps=curr_uv_warps
            )

            generated_frames.append(frame)
            frame_counter += 1

        # Save expression for next window before freeing memory
        last_expression = motion_sequence['expression_embed'][:, -1]

        # Free memory between windows
        torch.cuda.empty_cache()
        del motion_sequence

        # Update motion data for next window
        motion_data = {
            'theta': gt_params['theta'][min(frame_counter-1, gt_params['theta'].shape[0]-1)].unsqueeze(0),
            'expression_embed': last_expression
        }

    logger.info(f"Total frames generated: {len(generated_frames)}")
    frames = torch.stack(generated_frames).squeeze(1)

    # Save video
    logger.info("Saving video...")
    inferencer._save_video({'frames': frames}, audio_path, args.output, args.fps)
    logger.info(f"✅ Saved cached GT theta video to: {args.output}")

    logger.info("\n" + "="*80)
    logger.info("CACHED GT THETA INFERENCE COMPLETE")
    logger.info("="*80)
