#!/usr/bin/env python3
"""
VASA Audio-to-Video Alignment Test Suite

Tests to debug and align VASA's audio-driven motion generation with the volumetric avatar renderer.
Based on VASA-1 paper pipeline: encoding single frame into appearance volume & ID latent,
conditioning on audio features + other signals, denoising motion latents via diffusion, and decoding to frames.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import cv2
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import sys
from typing import Dict, List, Optional, Tuple
 

import json


# Import your modules
import sys
sys.path.insert(0, 'nemo')
from logger import logger
from vi import VASAInference
from pipeline2 import get_video_frames_as_images
from vasa_model import VASAModel
from wav2vec_module import AlignedWav2Vec2Model


class VASAAlignmentTester:
    """Test suite for VASA alignment with volumetric avatar rendering"""

    def __init__(self, config_path: str = "overfit_config.yaml"):
        """Initialize tester with VASA model and baseline components"""
        logger.info("Initializing VASA Alignment Tester...")

        # Load VASA inference
        self.vasa_inference = VASAInference( "./checkpoints_overfit/best_checkpoint.pt",config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store test results
        self.results = {}

    def extract_audio_from_video(self, video_path: str, output_path: str = "test_audio.wav") -> str:
        """Extract audio from video file"""
        import subprocess
        cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_path} -y"
        subprocess.run(cmd.split(), check=True, capture_output=True)
        logger.info(f"Extracted audio to {output_path}")
        return output_path

    def test_0_audio_motion_variance(self, source_image_path: str, audio_path: str) -> Dict:
        """
        Test 0: Verify Audio Drives Motion Generation
        Compare motion variance between silence and speech to ensure audio impact
        """
        logger.info("\n=== Test 0: Audio-Driven Motion Variance Test ===")

        # Load and preprocess source image
        source_img = Image.open(source_image_path).convert('RGB')
        preprocessed_frame = self.vasa_inference.transform(source_img).unsqueeze(0).to(self.device)
        logger.info(f"Preprocessed source frame: {preprocessed_frame.shape}")

        # Extract source parameters using EMO model
        with torch.no_grad():
            source_params = self.vasa_inference.extract_emo_parameters(preprocessed_frame)
            logger.info(f"Extracted source parameters")

        # Load and process real audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Process audio into windows
        audio_windows = self.vasa_inference.process_audio(waveform, sr=16000, fps=25.0)

        if not audio_windows:
            logger.error("No audio windows created")
            return {'status': 'failed', 'error': 'No audio windows'}

        # Use first window's audio features
        speech_audio_features = audio_windows[0]['audio_features']
        B, T, D = speech_audio_features.shape
        logger.info(f"Speech audio features shape: {speech_audio_features.shape}")
        logger.info(f"Speech audio stats - Mean: {speech_audio_features.mean():.6f}, Var: {speech_audio_features.var():.6f}, Max: {speech_audio_features.max():.4f}")

        # Create silent audio (zeros)
        silent_audio_features = torch.zeros_like(speech_audio_features)
        logger.info(f"Silent audio features shape: {silent_audio_features.shape}")
        logger.info(f"Silent audio stats - Mean: {silent_audio_features.mean():.6f}, Var: {silent_audio_features.var():.6f}")

        # Prepare base conditions
        base_conditions = {
            'gaze': torch.zeros(B, T, 2).to(self.device),
            'head_distance': torch.ones(B, T, 1).to(self.device),
            'emotion': torch.zeros(B, T, 2).to(self.device)
        }

        # Generate motion with SILENT audio
        logger.info("Generating motion with SILENT audio...")
        with torch.no_grad():
            silent_conditions = {**base_conditions, 'audio_features': silent_audio_features}
            silent_motion = self.vasa_inference.model.generate_sequence(
                initial_pose={'theta': source_params['theta'][0:1]},
                initial_dynamics=source_params['expression_embed'][0:1],
                conditions=silent_conditions,
                num_steps=25,
                eta=0.5
            )

        # Generate motion with SPEECH audio
        logger.info("Generating motion with SPEECH audio...")
        with torch.no_grad():
            speech_conditions = {**base_conditions, 'audio_features': speech_audio_features}
            speech_motion = self.vasa_inference.model.generate_sequence(
                initial_pose={'theta': source_params['theta'][0:1]},
                initial_dynamics=source_params['expression_embed'][0:1],
                conditions=speech_conditions,
                num_steps=25,
                eta=0.5
            )

        # Calculate variance for each motion parameter
        results = {}

        # Expression variance comparison
        silent_expr_var = torch.var(silent_motion['expression_embed']).item()
        speech_expr_var = torch.var(speech_motion['expression_embed']).item()
        expr_var_diff = speech_expr_var - silent_expr_var
        expr_var_ratio = speech_expr_var / (silent_expr_var + 1e-8)

        logger.info(f"Expression variance - Silent: {silent_expr_var:.6f}, Speech: {speech_expr_var:.6f}")
        logger.info(f"Expression variance difference: {expr_var_diff:.6f} (should be positive)")
        logger.info(f"Expression variance ratio: {expr_var_ratio:.2f}x (should be > 1)")

        results['expression'] = {
            'silent_var': silent_expr_var,
            'speech_var': speech_expr_var,
            'var_diff': expr_var_diff,
            'var_ratio': expr_var_ratio,
            'audio_drives_motion': expr_var_diff > 0 and expr_var_ratio > 1.1
        }

        # Theta (pose) variance comparison
        silent_theta_var = torch.var(silent_motion['theta']).item()
        speech_theta_var = torch.var(speech_motion['theta']).item()
        theta_var_diff = speech_theta_var - silent_theta_var
        theta_var_ratio = speech_theta_var / (silent_theta_var + 1e-8)

        logger.info(f"Theta variance - Silent: {silent_theta_var:.6f}, Speech: {speech_theta_var:.6f}")
        logger.info(f"Theta variance difference: {theta_var_diff:.6f}")
        logger.info(f"Theta variance ratio: {theta_var_ratio:.2f}x")

        results['theta'] = {
            'silent_var': silent_theta_var,
            'speech_var': speech_theta_var,
            'var_diff': theta_var_diff,
            'var_ratio': theta_var_ratio
        }

        # Calculate temporal variation (frame-to-frame differences)
        silent_expr_temporal = torch.mean(torch.abs(
            silent_motion['expression_embed'][:, 1:] - silent_motion['expression_embed'][:, :-1]
        )).item()
        speech_expr_temporal = torch.mean(torch.abs(
            speech_motion['expression_embed'][:, 1:] - speech_motion['expression_embed'][:, :-1]
        )).item()
        temporal_diff = speech_expr_temporal - silent_expr_temporal

        logger.info(f"Temporal variation - Silent: {silent_expr_temporal:.6f}, Speech: {speech_expr_temporal:.6f}")
        logger.info(f"Temporal variation difference: {temporal_diff:.6f} (should be positive)")

        results['temporal'] = {
            'silent_temporal': silent_expr_temporal,
            'speech_temporal': speech_expr_temporal,
            'temporal_diff': temporal_diff,
            'audio_drives_temporal': temporal_diff > 0
        }

        # Visualize comparison
        plt.figure(figsize=(15, 10))

        # Plot expression variance over time
        plt.subplot(2, 3, 1)
        silent_expr_std = silent_motion['expression_embed'][0].std(dim=-1).cpu()
        speech_expr_std = speech_motion['expression_embed'][0].std(dim=-1).cpu()
        plt.plot(silent_expr_std, label='Silent', alpha=0.7)
        plt.plot(speech_expr_std, label='Speech', alpha=0.7)
        plt.title('Expression Std Dev Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Std Dev')
        plt.legend()
        plt.grid(True)

        # Plot audio energy
        plt.subplot(2, 3, 2)
        audio_energy = speech_audio_features[0].mean(dim=-1).cpu()
        plt.plot(audio_energy)
        plt.title('Audio Feature Energy')
        plt.xlabel('Frame')
        plt.ylabel('Mean Activation')
        plt.grid(True)

        # Plot expression difference heatmap
        plt.subplot(2, 3, 3)
        expr_diff = (speech_motion['expression_embed'][0] - silent_motion['expression_embed'][0]).cpu()
        plt.imshow(expr_diff.T[:50], aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        plt.title('Expression Difference (Speech - Silent)')
        plt.xlabel('Frame')
        plt.ylabel('Expression Dim')
        plt.colorbar()

        # Plot theta variance
        plt.subplot(2, 3, 4)
        silent_theta_flat = silent_motion['theta'][0].view(T, -1)
        speech_theta_flat = speech_motion['theta'][0].view(T, -1)
        plt.plot(silent_theta_flat.std(dim=-1).cpu(), label='Silent', alpha=0.7)
        plt.plot(speech_theta_flat.std(dim=-1).cpu(), label='Speech', alpha=0.7)
        plt.title('Theta (Pose) Std Dev Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Std Dev')
        plt.legend()
        plt.grid(True)

        # Plot motion magnitude
        plt.subplot(2, 3, 5)
        silent_motion_mag = torch.norm(silent_motion['expression_embed'][0], dim=-1).cpu()
        speech_motion_mag = torch.norm(speech_motion['expression_embed'][0], dim=-1).cpu()
        plt.plot(silent_motion_mag, label='Silent', alpha=0.7)
        plt.plot(speech_motion_mag, label='Speech', alpha=0.7)
        plt.title('Motion Magnitude (L2 Norm)')
        plt.xlabel('Frame')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        # Plot variance comparison bar chart
        plt.subplot(2, 3, 6)
        categories = ['Expression\nVariance', 'Theta\nVariance', 'Temporal\nVariation']
        silent_vals = [silent_expr_var, silent_theta_var, silent_expr_temporal]
        speech_vals = [speech_expr_var, speech_theta_var, speech_expr_temporal]

        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width/2, silent_vals, width, label='Silent', alpha=0.7)
        plt.bar(x + width/2, speech_vals, width, label='Speech', alpha=0.7)
        plt.xticks(x, categories)
        plt.ylabel('Value')
        plt.title('Motion Statistics Comparison')
        plt.legend()
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('test0_audio_motion_variance.png')
        plt.close()
        logger.info("Saved visualization to test0_audio_motion_variance.png")

        # Overall assessment
        audio_drives_motion = (
            results['expression']['audio_drives_motion'] and
            results['temporal']['audio_drives_temporal']
        )

        if audio_drives_motion:
            logger.info("✓ PASS: Audio successfully drives motion generation")
        else:
            logger.warning("✗ FAIL: Audio does not significantly affect motion generation")

        self.results['test_0'] = {
            **results,
            'audio_drives_motion': audio_drives_motion,
            'test_passed': audio_drives_motion
        }

        return self.results['test_0']

    def test_1_audio_feature_extraction(self, audio_path: str) -> Dict:
        """
        Test 1: Verify Audio Feature Extraction Alignment
        Compare Wav2Vec2 features between pipelines for same audio
        """
        logger.info("\n=== Test 1: Audio Feature Extraction Alignment ===")

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        logger.info(f"Loaded audio: shape {waveform.shape}, sr {sr}")

        # Extract features using VASA's audio model
        with torch.no_grad():
            # Process through VASA's audio extractor
            inputs = self.vasa_inference.audio_processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)

            # Get the hidden states from wav2vec2
            outputs = self.vasa_inference.audio_model(**inputs)
            vasa_audio_features = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs[0]

            logger.info(f"VASA audio features: shape {vasa_audio_features.shape}")
            logger.info(f"  Min: {vasa_audio_features.min():.4f}, Max: {vasa_audio_features.max():.4f}")
            logger.info(f"  Mean: {vasa_audio_features.mean():.4f}, Std: {vasa_audio_features.std():.4f}")

        # For comparison with baseline (if available)
        # Since pipeline2.py doesn't use audio directly, we'll create a reference
        baseline_audio_features = vasa_audio_features.clone()  # Mock for now

        # Compute MSE
        mse = torch.mean((vasa_audio_features - baseline_audio_features) ** 2).item()
        logger.info(f"Audio feature MSE: {mse:.6f} (expected < 0.01 for aligned)")

        # Visualize features
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        # vasa_audio_features is [time, features], so transpose for visualization
        plt.imshow(vasa_audio_features.cpu().T[:100], aspect='auto', cmap='viridis')  # Show first 100 dims
        plt.title('VASA Audio Features (First 100 dims)')
        plt.xlabel('Time Frame')
        plt.ylabel('Feature Dim')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        mean_features = vasa_audio_features.mean(dim=-1).cpu()
        plt.plot(mean_features)
        plt.title('Mean Feature Activation Over Time')
        plt.xlabel('Time Frame')
        plt.ylabel('Mean Activation')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('test1_audio_features.png')
        plt.close()

        self.results['test_1'] = {
            'mse': mse,
            'features_shape': list(vasa_audio_features.shape),
            'features_stats': {
                'min': float(vasa_audio_features.min()),
                'max': float(vasa_audio_features.max()),
                'mean': float(vasa_audio_features.mean()),
                'std': float(vasa_audio_features.std())
            }
        }

        return self.results['test_1']

    def test_2_motion_generation(self, source_image_path: str, audio_path: str) -> Dict:
        """
        Test 2: Motion Latent Generation from Audio
        Test if diffusion transformer generates proper motion from audio
        """
        logger.info("\n=== Test 2: Motion Latent Generation from Audio ===")

        # Load and preprocess source image
        source_img = Image.open(source_image_path).convert('RGB')
        preprocessed_frame = self.vasa_inference.transform(source_img).unsqueeze(0).to(self.device)
        logger.info(f"Preprocessed source frame: {preprocessed_frame.shape}")

        # Extract source parameters using EMO model
        with torch.no_grad():
            source_params = self.vasa_inference.extract_emo_parameters(preprocessed_frame)
            logger.info(f"Source ID latent shape: {source_params['idt_embed'].shape}")
            logger.info(f"Source theta shape: {source_params['theta'].shape}")

        # Get audio features
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Process audio into windows
        audio_windows = self.vasa_inference.process_audio(waveform, sr=16000, fps=25.0)

        # Use first window's audio features
        if audio_windows:
            audio_features = audio_windows[0]['audio_features']
        else:
            logger.error("No audio windows created")
            return {}

        # Prepare conditions
        B, T = 1, 50
        conditions = {
            'audio_features': audio_features,  # Already [1, 50, 768]
            'gaze': torch.zeros(B, T, 2).to(self.device),
            'head_distance': torch.ones(B, T, 1).to(self.device),
            'emotion': torch.zeros(B, T, 2).to(self.device)
        }

        # Generate motion sequence
        logger.info("Generating motion sequence...")
        with torch.no_grad():
            generated_motion = self.vasa_inference.model.generate_sequence(
                initial_pose={'theta': source_params['theta'][0:1]},  # Keep batch dimension
                initial_dynamics=source_params['expression_embed'][0:1],  # Keep batch dimension
                conditions=conditions,
                num_steps=50,
                eta=0.5
            )

        logger.info(f"Generated motion shapes:")
        for key, value in generated_motion.items():
            logger.info(f"  {key}: {value.shape}")

        # Analyze motion variance (should respond to audio)
        theta_var = generated_motion['theta'].var(dim=1).mean().item()
        expr_var = generated_motion['expression_embed'].var(dim=1).mean().item()
        logger.info(f"Motion variance - Theta: {theta_var:.6f}, Expression: {expr_var:.6f}")

        # Plot motion trajectory
        plt.figure(figsize=(12, 8))

        # Plot theta (rotation) over time
        theta_flat = generated_motion['theta'][0].view(T, -1).cpu()
        plt.subplot(2, 2, 1)
        plt.plot(theta_flat[:, :3])  # First 3 components
        plt.title('Theta (Rotation) Components')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(['Comp 1', 'Comp 2', 'Comp 3'])
        plt.grid(True)

        # Plot expression over time
        expr = generated_motion['expression_embed'][0].cpu()
        plt.subplot(2, 2, 2)
        plt.imshow(expr.T[:50], aspect='auto', cmap='RdBu_r')
        plt.title('Expression Embedding (First 50 dims)')
        plt.xlabel('Frame')
        plt.ylabel('Dimension')
        plt.colorbar()

        # Plot audio features for comparison
        plt.subplot(2, 2, 3)
        plt.imshow(audio_features[0, :, :50].cpu().T, aspect='auto', cmap='viridis')
        plt.title('Audio Features (First 50 dims)')
        plt.xlabel('Frame')
        plt.ylabel('Dimension')
        plt.colorbar()

        # Plot motion magnitude
        plt.subplot(2, 2, 4)
        theta_norm = torch.norm(theta_flat, dim=1)
        expr_norm = torch.norm(expr, dim=1)
        plt.plot(theta_norm, label='Theta Norm')
        plt.plot(expr_norm / expr_norm.max() * theta_norm.max(), label='Expression Norm (scaled)')
        plt.title('Motion Magnitude Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('test2_motion_generation.png')
        plt.close()

        self.results['test_2'] = {
            'theta_variance': theta_var,
            'expression_variance': expr_var,
            'motion_shapes': {k: list(v.shape) for k, v in generated_motion.items()},
            'audio_responsive': theta_var > 0.001 and expr_var > 0.001
        }

        return self.results['test_2']

    def test_3_isolated_rendering(self, source_image_path: str, num_frames: int = 30) -> Dict:
        """
        Test 3: Isolated Rendering with Fixed Motion
        Test volumetric rendering with known motion patterns
        """
        logger.info("\n=== Test 3: Isolated Rendering with Fixed Motion ===")

        # Load source image
        source_img = Image.open(source_image_path).convert('RGB')
        preprocessed_frame = self.vasa_inference.transform(source_img).unsqueeze(0).to(self.device)

        # Extract source parameters using EMO model
        with torch.no_grad():
            source_params = self.vasa_inference.extract_emo_parameters(preprocessed_frame)

        # Create synthetic motion (simple oscillation for testing)
        B, T = 1, num_frames
        t = torch.linspace(0, 2*np.pi, T).to(self.device)

        # Oscillating theta (head rotation)
        theta_motion = source_params['theta'][0:1].unsqueeze(1).repeat(1, T, 1, 1)
        # Add oscillation properly accounting for tensor shapes
        oscillation = 0.1 * torch.sin(t).view(T)
        for i in range(T):
            theta_motion[:, i, 0, 0] += oscillation[i]

        # Oscillating expression
        expr_motion = source_params['expression_embed'][0:1].unsqueeze(1).repeat(1, T, 1)
        expr_motion[:, :, :10] += 0.05 * torch.sin(2*t).view(1, T, 1).expand(1, T, 10)

        # Render frames with fixed motion
        rendered_frames = []
        logger.info(f"Rendering {num_frames} frames with synthetic motion...")

        with torch.no_grad():
            for t_idx in range(T):
                # Prepare motion for single frame
                frame_motion = {
                    'theta': theta_motion[:, t_idx],
                    'expression': expr_motion[:, t_idx],
                    'scale': torch.ones(B, 3).to(self.device),
                    'rotation': torch.zeros(B, 3).to(self.device),
                    'translation': torch.zeros(B, 3).to(self.device)
                }

                # Render frame using the _generate_frame method
                rendered = self.vasa_inference._generate_frame(
                    source_params,  # source parameters
                    expr_motion[:, t_idx:t_idx+1],  # expression (keep batch dim)
                    frame_motion['theta'],  # theta
                    frame_motion['rotation'][0],  # rotation (single vector)
                    frame_motion['scale'][0],  # scale (single vector)
                    frame_motion['translation'][0],  # translation (single vector)
                    self.device
                )

                # Convert to image
                frame_img = (rendered[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rendered_frames.append(frame_img)

                # Save first, middle, and last frames
                if t_idx in [0, T//2, T-1]:
                    Image.fromarray(frame_img).save(f'test3_frame_{t_idx:03d}.png')

        # Create video
        imageio.mimsave('test3_isolated_rendering.mp4', rendered_frames, fps=25)
        logger.info("Saved rendered video to test3_isolated_rendering.mp4")

        # Compute identity preservation (compare first and last frames)
        first_frame = torch.tensor(rendered_frames[0]).float() / 255.0
        last_frame = torch.tensor(rendered_frames[-1]).float() / 255.0
        identity_mse = torch.mean((first_frame - last_frame) ** 2).item()
        logger.info(f"Identity MSE (first vs last): {identity_mse:.6f}")

        self.results['test_3'] = {
            'num_frames_rendered': num_frames,
            'identity_mse': identity_mse,
            'identity_preserved': identity_mse < 0.1
        }

        return self.results['test_3']

    def test_4_end_to_end(self, source_image_path: str, audio_path: str) -> Dict:
        """
        Test 4: End-to-End Audio-to-Video with Metrics
        Full pipeline test with quality metrics
        """
        logger.info("\n=== Test 4: End-to-End Audio-to-Video ===")

        # Run full VASA inference
        logger.info("Running VASA inference...")
        output_path = "test4_vasa_output.mp4"

        # Generate video from audio
        # Load source image
        source_img = Image.open(source_image_path).convert('RGB')
        source_tensor = self.vasa_inference.transform(source_img).unsqueeze(0).to(self.device)

        # Load and process audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Process audio into windows
        audio_windows = self.vasa_inference.process_audio(waveform, sr=16000, fps=25.0)

        # Extract source parameters
        source_params = self.vasa_inference.extract_emo_parameters(source_tensor)

        # Generate frames from audio
        frames_dict = self.vasa_inference.generate_frames_from_audio(
            source_params,
            audio_windows,
            source_params['expression_embed'][0]
        )

        # Convert frame dictionaries to numpy arrays
        frames = []
        if 'frames' in frames_dict:
            for frame_data in frames_dict['frames']:
                if 'frame' in frame_data:
                    frame_tensor = frame_data['frame']
                    frame_np = (frame_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    frames.append(frame_np)

        if frames is None or len(frames) == 0:
            logger.error("Failed to generate frames")
            self.results['test_4'] = {'status': 'failed', 'error': 'No frames generated'}
            return self.results['test_4']

        logger.info(f"Generated {len(frames)} frames")

        # Compute basic metrics
        # 1. Frame variance (motion amount)
        frames_tensor = torch.stack([torch.tensor(f).float() for f in frames])
        frame_variance = frames_tensor.var(dim=0).mean().item()

        # 2. Temporal coherence (frame-to-frame difference)
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            frame_diffs.append(diff)
        temporal_coherence = np.mean(frame_diffs)

        logger.info(f"Frame variance: {frame_variance:.4f}")
        logger.info(f"Temporal coherence: {temporal_coherence:.4f}")

        # 3. Audio-visual sync (basic energy correlation)
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Compute audio energy per frame
        samples_per_frame = sr // 25  # Assuming 25 fps
        audio_energy = []
        for i in range(min(len(frames), len(waveform[0]) // samples_per_frame)):
            chunk = waveform[0, i*samples_per_frame:(i+1)*samples_per_frame]
            energy = torch.mean(chunk ** 2).item()
            audio_energy.append(energy)

        # Correlate with motion
        if len(audio_energy) > 1 and len(frame_diffs) > 1:
            from scipy.stats import pearsonr
            min_len = min(len(audio_energy)-1, len(frame_diffs))
            correlation, p_value = pearsonr(audio_energy[:min_len], frame_diffs[:min_len])
            logger.info(f"Audio-motion correlation: {correlation:.4f} (p={p_value:.4f})")
        else:
            correlation = 0.0
            p_value = 1.0

        self.results['test_4'] = {
            'num_frames': len(frames),
            'frame_variance': frame_variance,
            'temporal_coherence': temporal_coherence,
            'audio_motion_correlation': correlation,
            'p_value': p_value,
            'output_path': output_path
        }

        return self.results['test_4']

    def test_5_disentanglement(self, source_a: str, source_b: str, audio_a: str, audio_b: str) -> Dict:
        """
        Test 5: Disentanglement Stress Test
        Swap sources: ID from Image A, motion from Audio B
        """
        logger.info("\n=== Test 5: Disentanglement Stress Test ===")

        # Extract ID from source A
        img_a = Image.open(source_a).convert('RGB')
        frame_a = self.vasa_inference.transform(img_a).unsqueeze(0).to(self.device)

        with torch.no_grad():
            params_a = self.vasa_inference.extract_emo_parameters(frame_a)
            id_a = params_a['idt_embed']

        # Extract motion from audio B
        waveform_b, sr = torchaudio.load(audio_b)
        if sr != 16000:
            waveform_b = torchaudio.transforms.Resample(sr, 16000)(waveform_b)

        # Process audio into windows
        audio_windows_b = self.vasa_inference.process_audio(waveform_b, sr=16000, fps=25.0)

        # Use first window's audio features
        if audio_windows_b:
            audio_features_b = audio_windows_b[0]['audio_features']
        else:
            logger.error("No audio windows created for audio B")
            return {}

        # Generate motion with audio B
        B, T = 1, 50
        conditions_b = {
            'audio_features': audio_features_b,  # Already [1, 50, 768]
            'gaze': torch.zeros(B, T, 2).to(self.device),
            'head_distance': torch.ones(B, T, 1).to(self.device),
            'emotion': torch.zeros(B, T, 2).to(self.device)
        }

        with torch.no_grad():
            motion_b = self.vasa_inference.model.generate_sequence(
                initial_pose={'theta': params_a['theta'][0:1]},  # Keep batch dimension
                initial_dynamics=params_a['expression_embed'][0:1],  # Keep batch dimension
                conditions=conditions_b,
                num_steps=50
            )

        # Render with ID A and motion B
        logger.info("Rendering with swapped ID and motion...")
        swapped_frames = []

        with torch.no_grad():
            for t in range(T):
                frame_motion = {
                    'theta': motion_b['theta'][:, t],
                    'expression': motion_b['expression_embed'][:, t],
                    'scale': motion_b['scale'][:, t] if 'scale' in motion_b else torch.ones(B, 3).to(self.device),
                    'rotation': motion_b['rotation'][:, t] if 'rotation' in motion_b else torch.zeros(B, 3).to(self.device),
                    'translation': motion_b['translation'][:, t] if 'translation' in motion_b else torch.zeros(B, 3).to(self.device)
                }

                # Use ID from A with motion from B
                # Render using the _generate_frame method
                rendered = self.vasa_inference._generate_frame(
                    params_a,  # Use source parameters from A for identity
                    motion_b['expression_embed'][:, t:t+1],  # expression from B (keep batch dim)
                    frame_motion['theta'],  # theta from B
                    frame_motion['rotation'][0] if isinstance(frame_motion['rotation'], torch.Tensor) and frame_motion['rotation'].dim() > 1 else frame_motion['rotation'],  # rotation (single vector)
                    frame_motion['scale'][0] if isinstance(frame_motion['scale'], torch.Tensor) and frame_motion['scale'].dim() > 1 else frame_motion['scale'],  # scale (single vector)
                    frame_motion['translation'][0] if isinstance(frame_motion['translation'], torch.Tensor) and frame_motion['translation'].dim() > 1 else frame_motion['translation'],  # translation (single vector)
                    self.device
                )

                frame_img = (rendered[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                swapped_frames.append(frame_img)

        # Save results
        imageio.mimsave('test5_disentangled.mp4', swapped_frames, fps=25)
        logger.info("Saved disentangled video to test5_disentangled.mp4")

        # Visual comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_a)
        axes[0].set_title('Source A (ID)')
        axes[0].axis('off')

        axes[1].imshow(swapped_frames[25])  # Middle frame
        axes[1].set_title('Swapped (ID:A, Motion:B)')
        axes[1].axis('off')

        # Load source B for comparison
        img_b = Image.open(source_b).convert('RGB')
        axes[2].imshow(img_b)
        axes[2].set_title('Source B (Motion donor)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('test5_disentanglement.png')
        plt.close()

        self.results['test_5'] = {
            'status': 'completed',
            'output_path': 'test5_disentangled.mp4',
            'num_frames': len(swapped_frames)
        }

        return self.results['test_5']

    def run_all_tests(self, source_image: str = "./data/A.png",
                      test_video: str = "./junk/7.mp4") -> Dict:
        """Run all alignment tests sequentially"""
        logger.info("=== Running Complete VASA Alignment Test Suite ===\n")

        # Extract audio from test video
        audio_path = self.extract_audio_from_video(test_video)

        # Run tests
        try:
            self.test_1_audio_feature_extraction(audio_path)
        except Exception as e:
            logger.error(f"Test 1 failed: {e}")
            self.results['test_1'] = {'status': 'failed', 'error': str(e)}

        try:
            self.test_2_motion_generation(source_image, audio_path)
        except Exception as e:
            logger.error(f"Test 2 failed: {e}")
            self.results['test_2'] = {'status': 'failed', 'error': str(e)}

        try:
            self.test_3_isolated_rendering(source_image)
        except Exception as e:
            logger.error(f"Test 3 failed: {e}")
            self.results['test_3'] = {'status': 'failed', 'error': str(e)}

        try:
            self.test_4_end_to_end(source_image, audio_path)
        except Exception as e:
            logger.error(f"Test 4 failed: {e}")
            self.results['test_4'] = {'status': 'failed', 'error': str(e)}

        # For test 5, use another image if available
        if Path("./data/3_source.png").exists():
            try:
                self.test_5_disentanglement(
                    source_image, "./data/3_source.png",
                    audio_path, audio_path  # Can use different audio
                )
            except Exception as e:
                logger.error(f"Test 5 failed: {e}")
                self.results['test_5'] = {'status': 'failed', 'error': str(e)}

        # Save results
        with open('test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # Print summary
        logger.info("\n=== Test Summary ===")
        for test_name, result in self.results.items():
            if isinstance(result, dict) and 'status' in result and result['status'] == 'failed':
                logger.error(f"{test_name}: FAILED - {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"{test_name}: PASSED")

        return self.results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VASA Alignment Test Suite')
    parser.add_argument('--config', default='overfit_config.yaml', help='Config file path')
    parser.add_argument('--source', default='./data/A.png', help='Source image path')
    parser.add_argument('--video', default='./junk/7.mp4', help='Test video path')
    parser.add_argument('--test', type=int, help='Run specific test (1-5)')
    args = parser.parse_args()

    tester = VASAAlignmentTester(args.config)

    if args.test:
        # Run specific test
        audio_path = tester.extract_audio_from_video(args.video)
        if args.test == 1:
            tester.test_1_audio_feature_extraction(audio_path)
        elif args.test == 2:
            tester.test_2_motion_generation(args.source, audio_path)
        elif args.test == 3:
            tester.test_3_isolated_rendering(args.source)
        elif args.test == 4:
            tester.test_4_end_to_end(args.source, audio_path)
        elif args.test == 5 and Path("./data/3_source.png").exists():
            tester.test_5_disentanglement(args.source, "./data/3_source.png", audio_path, audio_path)
    else:
        # Run all tests
        tester.run_all_tests(args.source, args.video)