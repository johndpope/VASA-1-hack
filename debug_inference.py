#!/usr/bin/env python3
"""Debug script to understand inference quality issues."""

import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_outputs():
    """Compare VASA output with nemo output."""
    
    # Load a frame from VASA output
    vasa_video = 'vasa-output-vasa_config-11.mp4'
    nemo_video = '3_identity.mp4'
    
    # Extract first frame from each
    def extract_frame(video_path, frame_num=10):
        """Extract a specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    vasa_frame = extract_frame(vasa_video, 10)
    nemo_frame = extract_frame(nemo_video, 10)
    
    if vasa_frame is not None and nemo_frame is not None:
        # Save comparison
        comparison = np.hstack([vasa_frame, nemo_frame])
        Image.fromarray(comparison).save('quality_comparison.png')
        logger.info("Saved comparison to quality_comparison.png")
        
        # Compute statistics
        vasa_std = np.std(vasa_frame)
        nemo_std = np.std(nemo_frame)
        
        logger.info(f"VASA frame std: {vasa_std:.2f}")
        logger.info(f"Nemo frame std: {nemo_std:.2f}")
        
        # Check sharpness (using Laplacian)
        vasa_gray = cv2.cvtColor(vasa_frame, cv2.COLOR_RGB2GRAY)
        nemo_gray = cv2.cvtColor(nemo_frame, cv2.COLOR_RGB2GRAY)
        
        vasa_sharpness = cv2.Laplacian(vasa_gray, cv2.CV_64F).var()
        nemo_sharpness = cv2.Laplacian(nemo_gray, cv2.CV_64F).var()
        
        logger.info(f"VASA sharpness: {vasa_sharpness:.2f}")
        logger.info(f"Nemo sharpness: {nemo_sharpness:.2f}")
        
        # Check if motion is being applied
        # Extract multiple frames from VASA
        frames = []
        for i in [0, 10, 20, 30]:
            frame = extract_frame(vasa_video, i)
            if frame is not None:
                frames.append(frame)
        
        if len(frames) > 1:
            # Check frame-to-frame differences
            diffs = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                diffs.append(diff)
            
            avg_motion = np.mean(diffs)
            logger.info(f"Average frame-to-frame difference: {avg_motion:.2f}")
            
            if avg_motion < 1.0:
                logger.warning("Very low motion detected - frames might be too similar")
            elif avg_motion > 50:
                logger.warning("Very high motion detected - might be unstable")

def check_model_outputs():
    """Check if the model is producing reasonable outputs."""
    import sys
    import yaml
    from omegaconf import OmegaConf
    sys.path.insert(0, '.')
    
    from vasa_model import VASAModel
    
    # Load config directly
    with open('vasa_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)
    model = VASAModel(config)
    
    # Load checkpoint
    checkpoint = torch.load('./checkpoints_overfit/best_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Create test inputs
    B, T = 1, 50
    audio_features = torch.randn(B, T, 768)
    
    with torch.no_grad():
        # Test motion generation
        motion = model.motion_transformer.generate_sequence(
            audio_features=audio_features,
            gaze_direction=torch.randn(B, T, 2),
            emotion_offset=torch.tensor([[0.5]]),
            speed_bucket=torch.ones(B, T, 1) * 4,
            num_frames=T
        )
        
        logger.info("Motion statistics:")
        for key, tensor in motion.items():
            if tensor is not None:
                logger.info(f"  {key}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

if __name__ == "__main__":
    logger.info("=== Comparing Output Quality ===")
    compare_outputs()