#!/usr/bin/env python3
"""
Sanity Check VI - Uses ground truth expression embeddings
==========================================================
This should perfectly reproduce the original expressions since
we're using the exact expression codes from the source video.
"""

import torch
import sys
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

sys.path.insert(0, 'nemo')

from vi import VASAInference
from vasa_model import VASAModel
import importlib
from omegaconf import OmegaConf
import torchaudio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanityCheckInference(VASAInference):
    """Extended inference that can use ground truth expressions"""
    
    def extract_ground_truth_expressions(self, video_path: str):
        """Extract expression embeddings from the original video"""
        logger.info(f"Extracting ground truth expressions from {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video has {total_frames} frames at {fps} fps")
        
        all_expressions = []
        all_theta = []
        all_rotation = []
        all_translation = []
        all_scale = []
        
        # Process each frame through EMO to get expression codes
        with torch.no_grad():
            for frame_idx in tqdm(range(total_frames), desc="Extracting expressions"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to tensor
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
                
                # Resize to 512x512 if needed
                if frame_tensor.shape[2] != 512 or frame_tensor.shape[3] != 512:
                    frame_tensor = torch.nn.functional.interpolate(
                        frame_tensor, size=(512, 512), mode='bilinear'
                    )
                
                # Extract expression using EMO encoder
                idt_embed = self.emo_model.encode_image(frame_tensor)
                
                # Get motion parameters
                motion_params = self.emo_model.get_motion_parameters(frame_tensor)
                
                all_expressions.append(idt_embed)
                
                if 'theta' in motion_params:
                    all_theta.append(motion_params['theta'])
                if 'rotation' in motion_params:
                    all_rotation.append(motion_params['rotation'])
                if 'translation' in motion_params:
                    all_translation.append(motion_params['translation'])
                if 'scale' in motion_params:
                    all_scale.append(motion_params['scale'])
        
        cap.release()
        
        # Stack all expressions
        ground_truth = {
            'expression_embed': torch.stack(all_expressions).squeeze(1),  # [T, 128]
            'num_frames': total_frames,
            'fps': fps
        }
        
        if all_theta:
            ground_truth['theta'] = torch.stack(all_theta).squeeze(1)
        if all_rotation:
            ground_truth['rotation'] = torch.stack(all_rotation).squeeze(1)
        if all_translation:
            ground_truth['translation'] = torch.stack(all_translation).squeeze(1)
        if all_scale:
            ground_truth['scale'] = torch.stack(all_scale).squeeze(1)
        
        logger.info(f"Extracted expressions shape: {ground_truth['expression_embed'].shape}")
        
        return ground_truth
    
    def generate_with_ground_truth(
        self,
        input_video: str,
        output_path: str,
        use_ground_truth_expressions: bool = True,
        use_ground_truth_motion: bool = False
    ):
        """Generate video using ground truth expressions/motion"""
        
        logger.info(f"\n{'='*60}")
        logger.info("SANITY CHECK: Using ground truth embeddings")
        logger.info(f"{'='*60}")
        
        # Extract ground truth from video
        ground_truth = self.extract_ground_truth_expressions(input_video)
        
        # Extract audio
        audio, sr = torchaudio.load(input_video)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Process audio features
        audio_features = self.wav2vec(audio.cuda()).last_hidden_state
        
        # Get first frame as identity
        cap = cv2.VideoCapture(input_video)
        ret, first_frame = cap.read()
        cap.release()
        
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        source_tensor = torch.from_numpy(first_frame_rgb).float() / 255.0
        source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        
        if source_tensor.shape[2] != 512:
            source_tensor = torch.nn.functional.interpolate(
                source_tensor, size=(512, 512), mode='bilinear'
            )
        
        # Extract identity
        identity = self.emo_model.encode_image(source_tensor)
        
        logger.info("\nGenerating with ground truth expressions...")
        
        # Process in windows
        window_size = self.config.motion.window_size
        stride = self.config.motion.stride
        num_frames = ground_truth['num_frames']
        
        all_frames = []
        
        for start_idx in range(0, num_frames - window_size + 1, stride):
            end_idx = min(start_idx + window_size, num_frames)
            
            logger.info(f"Processing frames {start_idx}-{end_idx}")
            
            # Get window of expressions
            window_expressions = ground_truth['expression_embed'][start_idx:end_idx]
            
            # Get window of audio
            audio_start = int(start_idx * audio_features.shape[1] / num_frames)
            audio_end = int(end_idx * audio_features.shape[1] / num_frames)
            window_audio = audio_features[:, audio_start:audio_end]
            
            # Prepare motion data
            B = 1
            T = window_expressions.shape[0]
            
            motion_data = {}
            
            if use_ground_truth_expressions:
                # USE EXACT EXPRESSION EMBEDDINGS
                motion_data['expression_embed'] = window_expressions.unsqueeze(0)  # [1, T, 128]
                logger.info(f"  Using GT expressions: {motion_data['expression_embed'].shape}")
            
            if use_ground_truth_motion and 'theta' in ground_truth:
                # USE EXACT MOTION PARAMETERS
                motion_data['theta'] = ground_truth['theta'][start_idx:end_idx].unsqueeze(0)
                motion_data['rotation'] = ground_truth['rotation'][start_idx:end_idx].unsqueeze(0)
                motion_data['translation'] = ground_truth['translation'][start_idx:end_idx].unsqueeze(0)
                motion_data['scale'] = ground_truth['scale'][start_idx:end_idx].unsqueeze(0)
                logger.info("  Using GT motion parameters")
            else:
                # Generate motion from model
                motion_data['theta'] = torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda()
                motion_data['scale'] = torch.ones(B, T, 3).cuda()
                motion_data['rotation'] = torch.zeros(B, T, 3).cuda()
                motion_data['translation'] = torch.zeros(B, T, 3).cuda()
            
            # Generate with VASA model
            with torch.no_grad():
                conditions = {
                    'audio_features': window_audio,
                    'gaze': torch.zeros(B, T, 2).cuda(),
                    'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
                    'emotion': torch.zeros(B, T, 2).cuda()
                }
                
                # Forward through VASA - should preserve expressions
                outputs = self.model.forward(
                    motion_data=motion_data,
                    noise_level=torch.zeros(B, device='cuda'),
                    conditions=conditions
                )
                
                # Check if expressions were preserved
                if use_ground_truth_expressions:
                    expr_diff = torch.norm(
                        outputs['expression_embed'] - motion_data['expression_embed']
                    ).item()
                    logger.info(f"  Expression preservation error: {expr_diff:.6f}")
                    
                    if expr_diff > 0.1:
                        logger.warning(f"  ⚠️ Expressions drifted! Error: {expr_diff}")
                    else:
                        logger.info(f"  ✅ Expressions preserved!")
                
                # Generate frames using EMO decoder
                for t in range(T):
                    frame_data = {
                        'idt_embed': identity,
                        'expression_embed': outputs['expression_embed'][:, t:t+1],
                        'theta': outputs['theta'][:, t],
                        'rotation': outputs['rotation'][:, t],
                        'translation': outputs['translation'][:, t],
                        'scale': outputs['scale'][:, t]
                    }
                    
                    # Decode to image
                    generated_frame = self.emo_model.decode(frame_data)
                    all_frames.append(generated_frame)
        
        # Save video
        logger.info(f"\nSaving video to {output_path}")
        self._save_video(all_frames, output_path, fps=ground_truth['fps'])
        
        logger.info(f"\n{'='*60}")
        logger.info("SANITY CHECK COMPLETE")
        logger.info("If expressions match perfectly, the model can reproduce them")
        logger.info("If they drift, there's an issue with expression preservation")
        logger.info(f"{'='*60}")
    
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
        
        # Write video
        h, w = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in frame_list:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Video saved: {output_path}")


def main():
    """Run sanity check with ground truth expressions"""
    
    print("\n" + "="*70)
    print("EXPRESSION SANITY CHECK")
    print("Using ground truth expression embeddings from original video")
    print("="*70)
    
    # Initialize with TDD model
    inferencer = SanityCheckInference(
        checkpoint_path="checkpoints/tdd_wandb/best_model.pth",
        config_path='vasa_config_fixed.yaml'
    )
    
    # Test with ground truth expressions
    inferencer.generate_with_ground_truth(
        input_video="./junk/10.mp4",
        output_path="vasa-sanity-check-expressions.mp4",
        use_ground_truth_expressions=True,  # Use exact expressions
        use_ground_truth_motion=False       # Generate motion
    )
    
    print("\nAlso generating with ground truth motion for comparison...")
    
    # Test with both ground truth expressions AND motion
    inferencer.generate_with_ground_truth(
        input_video="./junk/10.mp4",
        output_path="vasa-sanity-check-full.mp4",
        use_ground_truth_expressions=True,  # Use exact expressions
        use_ground_truth_motion=True        # Use exact motion too
    )
    
    print("\n" + "="*70)
    print("Sanity check complete!")
    print("\nOutputs:")
    print("  1. vasa-sanity-check-expressions.mp4 - GT expressions, generated motion")
    print("  2. vasa-sanity-check-full.mp4 - GT expressions AND motion")
    print("\nIf the expressions match perfectly, the model preserves them correctly.")
    print("If they drift, there's a problem with the expression codebook.")
    print("="*70)


if __name__ == "__main__":
    main()