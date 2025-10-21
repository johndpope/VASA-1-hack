#!/usr/bin/env python3
"""
Diagnostic tool to check theta/SRT predictions and identify geometric distortions.
"""
import torch
import sys
sys.path.insert(0, 'nemo')
from pathlib import Path
from omegaconf import OmegaConf
import importlib
from PIL import Image
import numpy as np
from torchvision import transforms
from vasa_model import VASAModel
from motion_sequence_handler import MotionSequenceHandler
import torchaudio
from transformers import Wav2Vec2Processor
from wav2vec_module import AlignedWav2Vec2Model
from logger import logger
from utils.point_transforms import get_transform_matrix
import matplotlib.pyplot as plt

def analyze_theta_predictions(
    checkpoint_path: str,
    config_path: str,
    video_path: str,
    device: str = 'cuda'
):
    """Analyze theta/SRT predictions to find geometric distortion causes."""

    logger.info("="*80)
    logger.info("THETA/SRT DIAGNOSTIC TOOL")
    logger.info("="*80)

    # Load config
    config = OmegaConf.load(config_path)

    # Load models
    logger.info("\n1. Loading models...")
    emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )
    model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    model_dict = torch.load(model_path, map_location=device, weights_only=False)
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.to(device).eval()

    model = VASAModel(config=config, volumetric_avatar=volumetric_avatar, device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Extract first frame from video
    logger.info(f"\n2. Extracting identity from: {video_path}")
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to read video")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source_img = Image.fromarray(frame_rgb).resize((512, 512))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    source_tensor = transform(source_img).unsqueeze(0).to(device)

    # Extract GT theta/SRT from identity
    logger.info("\n3. Extracting GT theta/SRT from identity image...")
    with torch.no_grad():
        gt_theta_4x4, gt_scale, gt_rotation, gt_translation = volumetric_avatar.head_pose_regressor.forward(
            source_tensor, return_srt=True
        )
        gt_theta = gt_theta_4x4[:, :3, :]  # Convert to 3x4

    logger.info(f"\nGT Parameters:")
    logger.info(f"  Theta shape: {gt_theta.shape}")
    logger.info(f"  Scale: {gt_scale[0].cpu().numpy()}")
    logger.info(f"  Rotation (radians): {gt_rotation[0].cpu().numpy()}")
    logger.info(f"  Rotation (degrees): {(gt_rotation[0] * 180 / np.pi).cpu().numpy()}")
    logger.info(f"  Translation: {gt_translation[0].cpu().numpy()}")

    # Verify theta composition from SRT
    logger.info("\n4. Verifying theta = compose(S, R, T)...")
    recomposed_theta = get_transform_matrix(gt_scale, gt_rotation, gt_translation)
    recomposed_theta_3x4 = recomposed_theta[:, :3, :]

    theta_diff = (gt_theta - recomposed_theta_3x4).abs().max().item()
    logger.info(f"  Max difference between GT theta and recomposed theta: {theta_diff:.6f}")
    if theta_diff < 1e-4:
        logger.info("  ✅ Theta composition is correct!")
    else:
        logger.warning(f"  ⚠️  Theta composition has {theta_diff:.6f} error")

    # Load audio for prediction
    logger.info("\n5. Loading audio and preparing conditions...")
    _, audio_path = extract_audio(video_path)
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    audio_model = AlignedWav2Vec2Model('facebook/wav2vec2-base', freeze_feature_extractor=True).to(device).eval()

    # Process first window of audio
    window_size = config.motion.window_size
    window_duration = window_size / 25.0
    samples_per_window = int(window_duration * sr)
    window = waveform[:, :samples_per_window]

    inputs = audio_processor(window.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        audio_features = audio_model(
            inputs.input_values.to(device),
            output_fps=25,
            frame_num=window_size,
            use_back_resample=True
        )

    conditions = {
        'audio_features': audio_features,
        'gaze': torch.zeros(1, window_size, 2, device=device),
        'head_distance': torch.zeros(1, window_size, 1, device=device),
        'emotion': torch.zeros(1, window_size, 2, device=device),
    }

    # Get identity embeddings
    logger.info("\n6. Extracting identity embeddings...")
    with torch.no_grad():
        source_mask = volumetric_avatar.face_idt.forward(source_tensor)[0]
        source_mask = (source_mask > 0.6).float()
        source_masked = source_tensor * source_mask
        idt_embed = volumetric_avatar.idt_embedder_nw(source_masked)
        logger.info(f"  idt_embed shape: {idt_embed.shape}")

    # Run inference to get predicted theta/SRT
    logger.info("\n7. Running inference to get predictions...")
    motion_data = {
        'theta': gt_theta,
        'expression_embed': torch.zeros(1, 128, device=device)
    }

    with torch.no_grad():
        motion_sequence = model.generate_sequence(
            initial_pose=motion_data,
            initial_dynamics=motion_data['expression_embed'],
            conditions=conditions,
            idt_embed=idt_embed,
            eta=0.8,
            num_steps=50,
            cfg_scales=None
        )

    # Analyze predictions
    logger.info("\n8. Analyzing predictions...")
    pred_theta = motion_sequence['theta']  # [1, T, 3, 4]
    pred_scale = motion_sequence['scale']  # [1, T, 3]
    pred_rotation = motion_sequence['rotation']  # [1, T, 3]
    pred_translation = motion_sequence['translation']  # [1, T, 3]

    logger.info(f"\nPredicted Parameters (frame 0):")
    logger.info(f"  Theta shape: {pred_theta.shape}")
    logger.info(f"  Scale: {pred_scale[0, 0].cpu().numpy()}")
    logger.info(f"  Rotation (radians): {pred_rotation[0, 0].cpu().numpy()}")
    logger.info(f"  Rotation (degrees): {(pred_rotation[0, 0] * 180 / np.pi).cpu().numpy()}")
    logger.info(f"  Translation: {pred_translation[0, 0].cpu().numpy()}")

    # Check for extreme values
    logger.info("\n9. Checking for extreme values...")
    scale_range = (pred_scale.min().item(), pred_scale.max().item())
    rot_range = (pred_rotation.min().item(), pred_rotation.max().item())
    trans_range = (pred_translation.min().item(), pred_translation.max().item())

    logger.info(f"  Scale range: {scale_range}")
    logger.info(f"  Rotation range (rad): {rot_range}")
    logger.info(f"  Rotation range (deg): ({rot_range[0]*180/np.pi:.1f}, {rot_range[1]*180/np.pi:.1f})")
    logger.info(f"  Translation range: {trans_range}")

    warnings = []
    if scale_range[0] < 0.5 or scale_range[1] > 2.0:
        warnings.append(f"⚠️  EXTREME SCALE: {scale_range} (should be ~0.8-1.2)")
    if abs(rot_range[0]) > 1.5 or abs(rot_range[1]) > 1.5:
        warnings.append(f"⚠️  EXTREME ROTATION: {rot_range[1]*180/np.pi:.1f}° (should be <45°)")
    if abs(trans_range[0]) > 0.5 or abs(trans_range[1]) > 0.5:
        warnings.append(f"⚠️  EXTREME TRANSLATION: {trans_range} (should be <0.3)")

    if warnings:
        logger.error("\n🚨 GEOMETRIC DISTORTION DETECTED:")
        for w in warnings:
            logger.error(f"  {w}")
    else:
        logger.info("\n✅ All parameters in reasonable range")

    # Check theta composition in predictions
    logger.info("\n10. Verifying predicted theta = compose(pred_S, pred_R, pred_T)...")
    for frame_idx in [0, window_size//2, window_size-1]:
        pred_theta_frame = pred_theta[0, frame_idx]
        pred_s = pred_scale[0, frame_idx]
        pred_r = pred_rotation[0, frame_idx]
        pred_t = pred_translation[0, frame_idx]

        # Recompose
        recomposed = get_transform_matrix(pred_s.unsqueeze(0), pred_r.unsqueeze(0), pred_t.unsqueeze(0))
        recomposed_3x4 = recomposed[0, :3, :]

        diff = (pred_theta_frame - recomposed_3x4).abs().max().item()
        logger.info(f"  Frame {frame_idx}: theta vs compose(S,R,T) diff = {diff:.6f}")

    # Plot trajectories
    logger.info("\n11. Plotting trajectories...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Scale
    axes[0].plot(pred_scale[0, :, 0].cpu().numpy(), label='sx')
    axes[0].plot(pred_scale[0, :, 1].cpu().numpy(), label='sy')
    axes[0].plot(pred_scale[0, :, 2].cpu().numpy(), label='sz')
    axes[0].axhline(gt_scale[0, 0].item(), color='r', linestyle='--', alpha=0.5, label='GT sx')
    axes[0].axhline(0.5, color='orange', linestyle=':', alpha=0.3)
    axes[0].axhline(2.0, color='orange', linestyle=':', alpha=0.3)
    axes[0].set_ylabel('Scale')
    axes[0].legend()
    axes[0].set_title('Scale Trajectory (warning if outside 0.5-2.0)')
    axes[0].grid(True)

    # Rotation
    axes[1].plot((pred_rotation[0, :, 0] * 180 / np.pi).cpu().numpy(), label='yaw')
    axes[1].plot((pred_rotation[0, :, 1] * 180 / np.pi).cpu().numpy(), label='pitch')
    axes[1].plot((pred_rotation[0, :, 2] * 180 / np.pi).cpu().numpy(), label='roll')
    axes[1].axhline((gt_rotation[0, 0] * 180 / np.pi).item(), color='r', linestyle='--', alpha=0.5, label='GT yaw')
    axes[1].axhline(-45, color='orange', linestyle=':', alpha=0.3)
    axes[1].axhline(45, color='orange', linestyle=':', alpha=0.3)
    axes[1].set_ylabel('Rotation (degrees)')
    axes[1].legend()
    axes[1].set_title('Rotation Trajectory (warning if outside ±45°)')
    axes[1].grid(True)

    # Translation
    axes[2].plot(pred_translation[0, :, 0].cpu().numpy(), label='tx')
    axes[2].plot(pred_translation[0, :, 1].cpu().numpy(), label='ty')
    axes[2].plot(pred_translation[0, :, 2].cpu().numpy(), label='tz')
    axes[2].axhline(gt_translation[0, 0].item(), color='r', linestyle='--', alpha=0.5, label='GT tx')
    axes[2].axhline(-0.5, color='orange', linestyle=':', alpha=0.3)
    axes[2].axhline(0.5, color='orange', linestyle=':', alpha=0.3)
    axes[2].set_ylabel('Translation')
    axes[2].set_xlabel('Frame')
    axes[2].legend()
    axes[2].set_title('Translation Trajectory (warning if outside ±0.5)')
    axes[2].grid(True)

    plt.tight_layout()
    output_path = 'theta_diagnosis.png'
    plt.savefig(output_path)
    logger.info(f"\n✅ Saved trajectory plot to: {output_path}")

    return motion_sequence, gt_theta, gt_scale, gt_rotation, gt_translation

def extract_audio(video_path):
    """Extract audio from video."""
    import subprocess
    audio_path = "/tmp/diagnostic_audio.wav"
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        audio_path
    ]
    subprocess.run(command, check=True, capture_output=True)
    return video_path, audio_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='overfit_config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints_overfit/best_checkpoint.pt')
    parser.add_argument('--video', default='junk/videovideoeI2V8Bd5X9s-scene6_scene1.mp4')
    args = parser.parse_args()

    analyze_theta_predictions(args.checkpoint, args.config, args.video)
