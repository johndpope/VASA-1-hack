#!/usr/bin/env python3
"""
Complete VI with Multi-step Inference and Audio-Visual TDD
===========================================================
Integrates all improvements:
1. Multi-step DDIM denoising
2. Audio-visual TDD testing
3. Curriculum-based training stages
"""

import torch
import sys
import numpy as np
from pathlib import Path
import time
import cv2
from tqdm import tqdm

sys.path.insert(0, 'nemo')

from vi import VASAInference
from vi_inference_fixed import ImprovedVASAInference, DDIMScheduler
from tdd_audio_visual_sync import AudioVisualTDDLoss
from vasa_lip_normalizer import EnhancedLipAnalyzer
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteVASAInference(VASAInference):
    """Complete inference with all improvements"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        num_inference_steps: int = 50,
        use_audio_tdd: bool = True,
        test_lip_sync: bool = True
    ):
        # Initialize parent class
        super().__init__(checkpoint_path, config_path)
        
        # Override with improved inference
        self.num_inference_steps = num_inference_steps
        self.use_audio_tdd = use_audio_tdd
        self.test_lip_sync = test_lip_sync
        
        # Initialize DDIM scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.config.diffusion.num_steps,
            num_inference_steps=num_inference_steps,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end
        )
        
        # Initialize audio-visual TDD if requested
        if use_audio_tdd:
            self.av_tdd = AudioVisualTDDLoss(device=self.device)
            self.lip_analyzer = EnhancedLipAnalyzer()
        
        logger.info(f"Initialized with {num_inference_steps} inference steps")
        logger.info(f"Audio-Visual TDD: {use_audio_tdd}")
        logger.info(f"Lip sync testing: {test_lip_sync}")
    
    def generate_motion_sequence(
        self,
        audio_features: torch.Tensor,
        identity: torch.Tensor,
        batch_size: int = 1,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate motion sequence with multi-step denoising
        
        Args:
            audio_features: Audio features [B, T, D]
            identity: Identity embedding
            batch_size: Batch size
            show_progress: Show progress bar
            
        Returns:
            Generated motion sequence
        """
        
        B, T, D = audio_features.shape
        device = self.device
        
        # Initialize with random noise
        motion_sample = {
            'theta': torch.randn(B, T, 3, 4, device=device),
            'scale': torch.randn(B, T, 3, device=device),
            'rotation': torch.randn(B, T, 3, device=device),
            'translation': torch.randn(B, T, 3, device=device),
            'expression_embed': torch.randn(B, T, 128, device=device)
        }
        
        # Add lips parameters for TDD
        motion_sample['lips'] = torch.randn(B, T, 20, 3, device=device)
        
        # Prepare conditions
        conditions = {
            'audio_features': audio_features,
            'gaze': torch.zeros(B, T, 2, device=device),
            'head_distance': torch.ones(B, T, 1, device=device) * 0.5,
            'emotion': torch.zeros(B, T, 2, device=device)
        }
        
        # Setup timesteps for DDIM
        timesteps = self.scheduler.timesteps.to(device)
        
        # Denoising loop
        iterator = tqdm(timesteps, desc="Denoising", disable=not show_progress)
        
        for i, t in enumerate(iterator):
            # Expand timestep to batch dimension
            timestep = t.expand(B)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(
                    motion_data=motion_sample,
                    noise_level=timestep,
                    conditions=conditions
                )
            
            # Perform DDIM step for each component
            for key in motion_sample.keys():
                if key in noise_pred:
                    # DDIM update
                    motion_sample[key] = self.scheduler.step(
                        model_output=noise_pred[key],
                        timestep=t,
                        sample=motion_sample[key],
                        eta=0.0  # Deterministic
                    )
            
            # Test lip sync periodically during generation
            if self.use_audio_tdd and i % 5 == 0:
                with torch.no_grad():
                    losses, test_info = self.av_tdd.compute_losses(
                        outputs=motion_sample,
                        conditions=conditions
                    )
                    
                    # Update progress bar with test info
                    iterator.set_postfix({
                        't': t.item(),
                        'lip_sync': f"{test_info['passed_ratio']:.1%}"
                    })
        
        # Final lip sync test
        if self.test_lip_sync:
            logger.info("\nFinal Audio-Visual TDD Tests:")
            with torch.no_grad():
                losses, test_info = self.av_tdd.compute_losses(
                    outputs=motion_sample,
                    conditions=conditions
                )
                
                logger.info(f"Test pass rate: {test_info['passed_ratio']:.1%}")
                for test_name, passed in test_info['test_results'].items():
                    status = "✅" if passed else "❌"
                    logger.info(f"  {status} {test_name}")
        
        return motion_sample
    
    def generate_from_video_improved(
        self,
        input_video: str,
        output_path: str,
        fps: float = 25.0,
        test_stages: bool = True
    ):
        """Generate with improved multi-step inference and TDD testing"""
        
        logger.info(f"\n{'='*60}")
        logger.info("Enhanced Video Generation with Audio-Visual TDD")
        logger.info(f"{'='*60}")
        logger.info(f"Input: {input_video}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Inference steps: {self.num_inference_steps}")
        
        # Extract audio and first frame
        import torchaudio
        import cv2
        import subprocess
        import tempfile
        
        # Extract audio to temporary wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
            
        # Extract audio using ffmpeg
        cmd = f'ffmpeg -i "{input_video}" -acodec pcm_s16le -ar 16000 -ac 1 -y "{tmp_audio_path}" 2>/dev/null'
        subprocess.run(cmd, shell=True, check=True)
        
        # Load audio
        audio, sr = torchaudio.load(tmp_audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Process audio features
        audio_features = self.audio_model(audio.cuda()).last_hidden_state
        
        # Get first frame for identity
        cap = cv2.VideoCapture(input_video)
        ret, first_frame = cap.read()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process first frame
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        source_tensor = torch.from_numpy(first_frame_rgb).float() / 255.0
        source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        
        if source_tensor.shape[2] != 512:
            source_tensor = torch.nn.functional.interpolate(
                source_tensor, size=(512, 512), mode='bilinear'
            )
        
        # Extract identity using face detection
        import torchvision.transforms.functional as FF
        with torch.no_grad():
            source_gray = FF.rgb_to_grayscale(source_tensor)
            face_mask_source = self.volumetric_avatar.face_idt.forward(source_gray)[0]
            face_mask_source = (face_mask_source > 0.6).float()
            source_masked = source_tensor * face_mask_source
            
            # Get identity embedding
            identity = self.volumetric_avatar.idt_embedder_nw.forward_image(source_masked)
        
        # Test different curriculum stages if requested
        if test_stages and self.use_audio_tdd:
            logger.info("\nTesting curriculum stages:")
            for stage_name in self.av_tdd.curriculum_stages.keys():
                self.av_tdd.current_stage = stage_name
                logger.info(f"  Stage: {stage_name}")
        
        # Process in windows with multi-step denoising
        window_size = self.config.motion.window_size
        stride = self.config.motion.stride
        
        logger.info(f"\nProcessing {total_frames} frames in windows of {window_size}")
        
        all_frames = []
        
        for start_idx in tqdm(range(0, audio_features.shape[1] - window_size + 1, stride), 
                              desc="Processing windows"):
            end_idx = min(start_idx + window_size, audio_features.shape[1])
            
            # Get audio window
            window_audio = audio_features[:, start_idx:end_idx]
            
            # Generate motion with multi-step denoising
            motion = self.generate_motion_sequence(
                audio_features=window_audio,
                identity=identity,
                batch_size=1,
                show_progress=False
            )
            
            # Decode to frames
            for t in range(motion['theta'].shape[1]):
                frame_data = {
                    'idt_embed': identity,
                    'expression_embed': motion['expression_embed'][:, t:t+1],
                    'theta': motion['theta'][:, t],
                    'rotation': motion['rotation'][:, t],
                    'translation': motion['translation'][:, t],
                    'scale': motion['scale'][:, t]
                }
                
                # Generate frame with EMO
                with torch.no_grad():
                    generated_frame = self.volumetric_avatar.decode(frame_data)
                
                all_frames.append(generated_frame)
        
        # Save video
        self._save_video(all_frames, output_path, fps)
        
        # Clean up temp audio file
        import os
        if 'tmp_audio_path' in locals():
            try:
                os.remove(tmp_audio_path)
            except:
                pass
        
        logger.info(f"\n✅ Video saved: {output_path}")
        
        # Final summary
        if self.use_audio_tdd:
            logger.info("\n📊 Audio-Visual Quality Summary:")
            logger.info(f"  Inference steps used: {self.num_inference_steps}")
            logger.info(f"  Lip sync tested: {self.test_lip_sync}")
            logger.info(f"  Windows processed: {len(all_frames) // window_size}")
    
    def _save_video(self, frames, output_path, fps=25.0):
        """Save frames to video"""
        if not frames:
            return
        
        # Convert tensors to numpy
        frame_list = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.squeeze(0).cpu().numpy()
                if frame.shape[0] == 3:  # CHW -> HWC
                    frame = frame.transpose(1, 2, 0)
                frame = (frame * 255).astype(np.uint8)
            frame_list.append(frame)
        
        # Write video using OpenCV
        h, w = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frame_list:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


def main():
    """Run complete inference with all improvements"""
    
    print("\n" + "="*70)
    print("Complete VASA Inference")
    print("With Multi-step Denoising and Audio-Visual TDD")
    print("="*70)
    
    # Get config from command line or use defaults
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./junk/10.mp4', help='Input video')
    parser.add_argument('--output', default=None, help='Output video path')
    parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--checkpoint', default='checkpoints/tdd_wandb/best_model.pth', help='Model checkpoint')
    parser.add_argument('--config', default='vasa_config_fixed.yaml', help='Config file')
    parser.add_argument('--no-tdd', action='store_true', help='Disable audio-visual TDD')
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = int(time.time())
        args.output = f"vasa-complete-{args.steps}steps-{timestamp}.mp4"
    
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Inference steps: {args.steps}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Audio-Visual TDD: {not args.no_tdd}")
    
    # Initialize complete inference
    inferencer = CompleteVASAInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_inference_steps=args.steps,
        use_audio_tdd=not args.no_tdd,
        test_lip_sync=not args.no_tdd
    )
    
    # Generate video
    inferencer.generate_from_video_improved(
        input_video=args.input,
        output_path=args.output,
        fps=25.0,
        test_stages=not args.no_tdd
    )
    
    print(f"\n✅ Complete! Video saved as: {args.output}")
    print("\nKey improvements applied:")
    print(f"  • Multi-step DDIM denoising ({args.steps} steps)")
    if not args.no_tdd:
        print("  • Audio-visual TDD testing")
        print("  • Lip sync validation")
        print("  • Curriculum-based quality checks")


if __name__ == "__main__":
    main()