#!/usr/bin/env python3
"""
Analyze VA's expected motion space from real driving videos.
This will collect statistics on target_pose_embed and target_theta distributions
to understand the expected ranges for proper VASA normalization.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import cv2
import importlib

sys.path.append('nemo')
sys.path.append('.')

from torchvision import transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAMotionAnalyzer:
    """Analyze motion distributions from VA's perspective."""
    
    def __init__(self):
        """Initialize the analyzer with VA model."""
        logger.info("Initializing VA Motion Analyzer...")
        
        # Load VA model (same as pipeline2.py)
        model_path = './logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
        from omegaconf import OmegaConf
        emo_config = OmegaConf.load('./models/stage_1/volumetric_avatar/va.yaml')
        
        self.va = importlib.import_module(
            'models.stage_1.volumetric_avatar.va'
        ).Model(emo_config, training=False)
        
        # Load weights
        model_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        self.va.load_state_dict(model_dict, strict=False)
        self.va = self.va.cuda()
        self.va.eval()
        
        logger.info("VA model loaded successfully")
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Storage for statistics
        self.stats = {
            'target_pose_embed': [],
            'target_theta': [],
            'source_pose_embed': [],
            'source_theta': []
        }
        
    def process_video_pair(self, source_img_path, driving_video_path):
        """Process a source image and driving video to extract motion stats."""
        logger.info(f"Processing: {source_img_path} with {driving_video_path}")
        
        # Load source image
        source_img = Image.open(source_img_path).convert('RGB')
        source_tensor = self.transform(source_img).unsqueeze(0).cuda()
        
        # Get source embeddings
        with torch.no_grad():
            # Get face mask
            face_mask, _, _, _ = self.va.face_idt.forward(source_tensor)
            face_mask = (face_mask > 0.6).float()
            source_masked = source_tensor * face_mask
            
            # Get identity embedding
            idt_embed = self.va.idt_embedder_nw(source_masked)
            
            # Get source pose/expression
            data_dict = {
                'source_img': source_tensor,
                'target_img': source_tensor,
                'source_mask': face_mask,
                'target_mask': face_mask,
                'idt_embed': idt_embed
            }
            
            # Get head pose
            if hasattr(self.va, 'head_pose_regressor'):
                source_theta = self.va.head_pose_regressor.forward(source_tensor)
                if source_theta.shape[-2] == 4:
                    source_theta = source_theta[:, :3, :]
                data_dict['source_theta'] = source_theta
                data_dict['target_theta'] = source_theta
            
            # Get expression embedding
            data_dict = self.va.expression_embedder_nw(data_dict, True, False)
            source_pose_embed = data_dict['source_pose_embed']
            
            # Store source stats
            self.stats['source_pose_embed'].append(source_pose_embed.cpu().numpy())
            self.stats['source_theta'].append(source_theta.cpu().numpy())
        
        # Process driving video frames
        cap = cv2.VideoCapture(driving_video_path)
        frame_count = 0
        max_frames = 100  # Limit to first 100 frames for speed
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize and transform
            frame_pil = frame_pil.resize((512, 512), Image.LANCZOS)
            frame_tensor = self.transform(frame_pil).unsqueeze(0).cuda()
            
            with torch.no_grad():
                # Get target embeddings
                target_mask = self.va.face_idt.forward(frame_tensor)[0]
                target_mask = (target_mask > 0.6).float()
                
                target_dict = {
                    'source_img': source_tensor,
                    'target_img': frame_tensor,
                    'source_mask': face_mask,
                    'target_mask': target_mask,
                    'idt_embed': idt_embed,
                    'source_pose_embed': source_pose_embed,
                    'source_theta': source_theta
                }
                
                # Get target pose
                if hasattr(self.va, 'head_pose_regressor'):
                    target_theta = self.va.head_pose_regressor.forward(frame_tensor)
                    if target_theta.shape[-2] == 4:
                        target_theta = target_theta[:, :3, :]
                    target_dict['target_theta'] = target_theta
                
                # Get target expression
                target_dict = self.va.expression_embedder_nw(target_dict, True, False)
                target_pose_embed = target_dict['target_pose_embed']
                
                # Store target stats
                self.stats['target_pose_embed'].append(target_pose_embed.cpu().numpy())
                self.stats['target_theta'].append(target_theta.cpu().numpy())
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                logger.info(f"  Processed {frame_count} frames...")
        
        cap.release()
        logger.info(f"  Total frames processed: {frame_count}")
        
    def compute_statistics(self):
        """Compute mean and std for all collected data."""
        results = {}
        
        for key in self.stats:
            if len(self.stats[key]) > 0:
                data = np.concatenate(self.stats[key], axis=0)
                
                # Compute statistics
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                min_val = np.min(data, axis=0)
                max_val = np.max(data, axis=0)
                
                results[key] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'shape': data.shape,
                    'num_samples': len(self.stats[key])
                }
                
                logger.info(f"\n{key} statistics:")
                logger.info(f"  Shape: {data.shape}")
                logger.info(f"  Mean range: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
                logger.info(f"  Std range: [{np.min(std):.4f}, {np.max(std):.4f}]")
                logger.info(f"  Value range: [{np.min(data):.4f}, {np.max(data):.4f}]")
        
        return results
    
    def save_statistics(self, output_path='va_motion_statistics.pkl'):
        """Save computed statistics to file."""
        results = self.compute_statistics()
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nStatistics saved to {output_path}")
        
        # Also save as JSON (for readable version)
        json_path = output_path.replace('.pkl', '.json')
        json_results = {}
        for key, stats in results.items():
            json_results[key] = {
                'mean_range': [float(np.min(stats['mean'])), float(np.max(stats['mean']))],
                'std_range': [float(np.min(stats['std'])), float(np.max(stats['std']))],
                'value_range': [float(np.min(stats['min'])), float(np.max(stats['max']))],
                'shape': list(stats['shape']),
                'num_samples': stats['num_samples']
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Readable statistics saved to {json_path}")
        
        return results

def main():
    """Main analysis function."""
    analyzer = VAMotionAnalyzer()
    
    # Process test videos
    # You can add more video pairs here
    test_pairs = [
        ('nemo/data/IMG_1.png', 'junk/3.mp4'),
        ('nemo/data/IMG_1.png', 'junk/7.mp4'),
        ('nemo/data/IMG_1.png', 'junk/9.mp4'),
    ]
    
    for source, driving in test_pairs:
        if os.path.exists(source) and os.path.exists(driving):
            analyzer.process_video_pair(source, driving)
        else:
            logger.warning(f"Skipping missing files: {source} or {driving}")
    
    # Compute and save statistics
    results = analyzer.save_statistics()
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info("\nKey findings for VASA normalization:")
    
    if 'target_pose_embed' in results:
        embed_stats = results['target_pose_embed']
        logger.info(f"\nTarget Expression Embedding:")
        logger.info(f"  - Typical range: [{np.percentile(embed_stats['min'], 5):.3f}, {np.percentile(embed_stats['max'], 95):.3f}]")
        logger.info(f"  - Mean std: {np.mean(embed_stats['std']):.3f}")
        
    if 'target_theta' in results:
        theta_stats = results['target_theta']
        logger.info(f"\nTarget Pose (Theta):")
        logger.info(f"  - Typical range: [{np.percentile(theta_stats['min'], 5):.3f}, {np.percentile(theta_stats['max'], 95):.3f}]")
        logger.info(f"  - Mean std: {np.mean(theta_stats['std']):.3f}")
    
    logger.info("\nUse these statistics to normalize VASA outputs in vasa_va_bridge.py")

if __name__ == "__main__":
    main()