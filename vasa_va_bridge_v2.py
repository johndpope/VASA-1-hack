#!/usr/bin/env python3
"""
Improved VASA-VA Bridge with normalization, temporal smoothing, and 3-avatar visualization.
Based on analysis of VA's expected motion space distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np
import pickle
import os

logger = logging.getLogger(__name__)

class VASAVolumetricAvatarBridgeV2:
    """
    Enhanced bridge between VASA's motion generation and volumetric avatar's frame generation.
    Includes normalization, temporal smoothing, and 3-avatar visualization.
    """
    
    def __init__(self, volumetric_avatar, stats_path='va_motion_statistics.pkl'):
        """
        Args:
            volumetric_avatar: The volumetric avatar model
            stats_path: Path to VA motion statistics file
        """
        self.va = volumetric_avatar
        self._source_cache = {}
        self._neutral_cache = {}
        
        # Temporal smoothing state
        self.prev_theta = None
        self.prev_expression = None
        
        # Load VA motion statistics for normalization
        self.va_stats = None
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                self.va_stats = pickle.load(f)
            logger.info(f"Loaded VA motion statistics from {stats_path}")
            
            # Extract key statistics for normalization
            if 'target_pose_embed' in self.va_stats:
                embed_stats = self.va_stats['target_pose_embed']
                self.embed_mean = torch.tensor(embed_stats['mean'], dtype=torch.float32).cuda()
                self.embed_std = torch.tensor(embed_stats['std'], dtype=torch.float32).cuda()
                # Clamp std to avoid division by zero
                self.embed_std = torch.clamp(self.embed_std, min=0.01)
                logger.info(f"Expression embed normalization ready - mean shape: {self.embed_mean.shape}")
                
            if 'target_theta' in self.va_stats:
                theta_stats = self.va_stats['target_theta']
                self.theta_mean = torch.tensor(theta_stats['mean'], dtype=torch.float32).cuda()
                self.theta_std = torch.tensor(theta_stats['std'], dtype=torch.float32).cuda()
                # Clamp std to avoid division by zero
                self.theta_std = torch.clamp(self.theta_std, min=0.01)
                logger.info(f"Theta normalization ready - mean shape: {self.theta_mean.shape}")
        else:
            logger.warning(f"No VA statistics found at {stats_path} - normalization disabled")
            
        # Improved masking threshold
        self.mask_threshold = 0.8  # Increased from 0.6 to prevent green glow
        
    def clear_cache(self):
        """Clear all caches and temporal state."""
        self._source_cache = {}
        self._neutral_cache = {}
        self.prev_theta = None
        self.prev_expression = None
        
    def normalize_expression(self, vasa_expression: torch.Tensor) -> torch.Tensor:
        """
        Normalize VASA expression to match VA's expected distribution.
        
        Args:
            vasa_expression: VASA-generated expression [B, T, 128] or [B, 128]
            
        Returns:
            Normalized expression matching VA space
        """
        if self.va_stats is None or not hasattr(self, 'embed_mean'):
            return vasa_expression
            
        # Handle both [B, T, D] and [B, D] shapes
        original_shape = vasa_expression.shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            vasa_expression = vasa_expression.view(B * T, D)
        
        # First, compute VASA's current statistics
        vasa_mean = vasa_expression.mean(dim=0, keepdim=True)
        vasa_std = vasa_expression.std(dim=0, keepdim=True)
        vasa_std = torch.clamp(vasa_std, min=0.01)
        
        # Standardize VASA output
        normalized = (vasa_expression - vasa_mean) / vasa_std
        
        # Re-scale to VA's distribution
        normalized = normalized * self.embed_std.unsqueeze(0) + self.embed_mean.unsqueeze(0)
        
        # Clip to reasonable range (based on VA analysis)
        normalized = torch.clamp(normalized, min=-3.0, max=3.0)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            normalized = normalized.view(B, T, D)
            
        return normalized
        
    def normalize_theta(self, vasa_theta: torch.Tensor) -> torch.Tensor:
        """
        Normalize VASA theta (pose) to match VA's expected distribution.
        
        Args:
            vasa_theta: VASA-generated theta [B, T, 3, 4] or [B, 3, 4]
            
        Returns:
            Normalized theta matching VA space
        """
        if self.va_stats is None or not hasattr(self, 'theta_mean'):
            return vasa_theta
            
        # Handle both [B, T, 3, 4] and [B, 3, 4] shapes
        original_shape = vasa_theta.shape
        if len(original_shape) == 4:
            B, T, H, W = original_shape
            vasa_theta = vasa_theta.view(B * T, H, W)
        
        # Normalize rotation components (first 3x3)
        rotation = vasa_theta[:, :3, :3]
        translation = vasa_theta[:, :3, 3:4]
        
        # Apply normalization to translation (which affects head position)
        # Keep rotation matrix normalized (orthogonal)
        translation_flat = translation.view(-1, 3)
        trans_mean = self.theta_mean[:, 3].unsqueeze(0)
        trans_std = self.theta_std[:, 3].unsqueeze(0)
        
        # Normalize translation
        normalized_trans = (translation_flat - translation_flat.mean(dim=0, keepdim=True)) / translation_flat.std(dim=0, keepdim=True).clamp(min=0.01)
        normalized_trans = normalized_trans * trans_std + trans_mean
        normalized_trans = torch.clamp(normalized_trans, min=-0.5, max=0.5)  # Limit head movement
        
        # Reconstruct theta
        normalized_theta = torch.cat([rotation, normalized_trans.view(-1, 3, 1)], dim=2)
        
        # Reshape back if needed
        if len(original_shape) == 4:
            normalized_theta = normalized_theta.view(B, T, 3, 4)
            
        return normalized_theta
        
    def apply_temporal_smoothing(self, expression: torch.Tensor, theta: torch.Tensor, 
                                alpha_expr: float = 0.9, alpha_pose: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply exponential moving average smoothing to reduce jitter.
        
        Args:
            expression: Current expression [B, 128]
            theta: Current theta [B, 3, 4]
            alpha_expr: Smoothing factor for expression (0-1, higher = more smoothing)
            alpha_pose: Smoothing factor for pose (0-1, higher = more smoothing)
            
        Returns:
            Smoothed expression and theta
        """
        # Smooth expression
        if self.prev_expression is not None:
            expression = alpha_expr * self.prev_expression + (1 - alpha_expr) * expression
        self.prev_expression = expression.clone()
        
        # Smooth theta (pose)
        if self.prev_theta is not None:
            # Smooth translation component more aggressively
            theta[:, :, 3] = alpha_pose * self.prev_theta[:, :, 3] + (1 - alpha_pose) * theta[:, :, 3]
            # Lighter smoothing on rotation to preserve expressiveness
            theta[:, :, :3] = 0.3 * self.prev_theta[:, :, :3] + 0.7 * theta[:, :, :3]
        self.prev_theta = theta.clone()
        
        return expression, theta
        
    def get_source_embeddings(self, source_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract and cache source embeddings from identity image.
        
        Args:
            source_img: Identity image [B, C, H, W]
            
        Returns:
            Dictionary with source embeddings
        """
        # Create a stable cache key based on tensor shape and device
        cache_key = f"{source_img.shape}_{source_img.device}_{source_img.dtype}"
        
        if cache_key in self._source_cache:
            logger.info(f"Using cached source embeddings for key: {cache_key}")
            return self._source_cache[cache_key]
        
        logger.info("Computing source embeddings from identity image (will be cached)")
        
        with torch.no_grad():
            # Get face mask with higher threshold
            face_mask, _, _, _ = self.va.face_idt.forward(source_img)
            face_mask = (face_mask > self.mask_threshold).float()
            source_masked = source_img * face_mask
            
            # Get identity embedding
            idt_embed = self.va.idt_embedder_nw(source_masked)
            
            # Get source pose/expression
            data_dict = {
                'source_img': source_img,
                'target_img': source_img,
                'source_mask': face_mask,
                'target_mask': face_mask,
                'idt_embed': idt_embed
            }
            
            # Get head pose
            if hasattr(self.va, 'head_pose_regressor'):
                source_theta = self.va.head_pose_regressor.forward(source_img)
                if source_theta.shape[-2] == 4:
                    source_theta = source_theta[:, :3, :]
                data_dict['source_theta'] = source_theta
                data_dict['target_theta'] = source_theta
            
            # Get expression embedding
            data_dict = self.va.expression_embedder_nw(data_dict, True, False)
            source_pose_embed = data_dict['source_pose_embed']
            
            # Encode source to latent volume
            source_latents = self.va.local_encoder_nw(source_masked)
            c = self.va.args.latent_volume_channels
            d = self.va.args.latent_volume_depth
            s = self.va.args.latent_volume_size
            source_volume = source_latents.view(-1, c, d, s, s)
            
            if self.va.args.source_volume_num_blocks > 0:
                source_volume = self.va.volume_source_nw(source_volume)
            
            # Cache results
            result = {
                'source_pose_embed': source_pose_embed,
                'source_theta': data_dict.get('source_theta'),
                'source_volume': source_volume,
                'source_mask': face_mask,
                'idt_embed': idt_embed,
                'source_masked': source_masked
            }
            
            self._source_cache[cache_key] = result
            logger.info(f"Cached source embeddings with key: {cache_key}")
            
        return result
        
    def get_neutral_volume(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get or create a neutral canonical volume for driving proxy visualization.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Neutral canonical volume
        """
        cache_key = 'neutral'
        
        if cache_key in self._neutral_cache:
            return self._neutral_cache[cache_key].repeat(batch_size, 1, 1, 1, 1)
            
        # Create a neutral volume (could load a specific neutral face if available)
        # For now, use a zero-expression volume
        c = self.va.args.latent_volume_channels
        d = self.va.args.latent_volume_depth
        s = self.va.args.latent_volume_size
        
        # Create neutral volume (initialized with small random values)
        neutral_volume = torch.randn(1, c, d, s, s, device='cuda') * 0.01
        
        if self.va.args.source_volume_num_blocks > 0:
            neutral_volume = self.va.volume_process_nw(neutral_volume)
            
        self._neutral_cache[cache_key] = neutral_volume
        return neutral_volume.repeat(batch_size, 1, 1, 1, 1)
        
    def generate_frames_with_viz(
        self,
        motion_outputs: Dict[str, torch.Tensor],
        source_img: torch.Tensor,
        use_black_background: bool = True,
        enable_3avatar: bool = True,
        enable_smoothing: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate frames with optional 3-avatar visualization.
        
        Args:
            motion_outputs: VASA-generated motion
            source_img: Identity image [B, C, H, W]
            use_black_background: Whether to use black background
            enable_3avatar: Generate 3-avatar visualization
            enable_smoothing: Apply temporal smoothing
            
        Returns:
            Dictionary with:
                - 'frames': Generated frames [B, T, C, H, W]
                - 'viz_frames': 3-avatar visualization [B, T, C, H, 3*W] (optional)
                - 'driving_proxy': Driving proxy frames [B, T, C, H, W] (optional)
        """
        B, T = motion_outputs['theta'].shape[:2]
        device = motion_outputs['theta'].device
        
        # Get source embeddings
        source_data = self.get_source_embeddings(source_img)
        
        # Process canonical volume
        source_volume = source_data['source_volume']
        canonical_volume = self.va.volume_process_nw(source_volume)
        
        # Get neutral volume if needed for driving proxy
        neutral_volume = None
        if enable_3avatar:
            neutral_volume = self.get_neutral_volume(B)
            
        c = self.va.args.latent_volume_channels
        d = self.va.args.latent_volume_depth
        s = self.va.args.latent_volume_size
        
        generated_frames = []
        driving_proxy_frames = [] if enable_3avatar else None
        viz_frames = [] if enable_3avatar else None
        
        for b in range(B):
            batch_frames = []
            batch_driving = [] if enable_3avatar else None
            batch_viz = [] if enable_3avatar else None
            
            canonical_volume_b = canonical_volume[b:b+1]
            neutral_volume_b = neutral_volume[b:b+1] if neutral_volume is not None else None
            
            for t in range(T):
                # Get VASA motion for this frame
                target_expression = motion_outputs['expression_embed'][b:b+1, t]
                target_theta = motion_outputs['theta'][b:b+1, t]
                
                # Convert theta format if needed
                if target_theta.shape[-2] == 4:
                    target_theta = target_theta[:, :3, :]
                
                # Apply normalization
                target_expression = self.normalize_expression(target_expression)
                target_theta = self.normalize_theta(target_theta)
                
                # Apply temporal smoothing
                if enable_smoothing:
                    target_expression, target_theta = self.apply_temporal_smoothing(
                        target_expression, target_theta
                    )
                
                # Create data dict
                data_dict = {
                    'source_img': source_img[b:b+1],
                    'target_img': source_img[b:b+1],
                    'source_mask': source_data['source_mask'][b:b+1],
                    'target_mask': source_data['source_mask'][b:b+1],
                    'source_theta': source_data['source_theta'][b:b+1] if source_data['source_theta'] is not None else target_theta,
                    'target_theta': target_theta,
                    'idt_embed': source_data['idt_embed'][b:b+1],
                    'source_pose_embed': source_data['source_pose_embed'][b:b+1],
                    'target_pose_embed': target_expression
                }
                
                # Generate embeddings for warping
                _, target_warp_embed_dict, _, embed_dict = self.va.predict_embed(data_dict)
                
                # Generate UV warp
                target_uv_warp, _ = self.va.uv_generator_nw(target_warp_embed_dict)
                
                # Handle resizing
                if self.va.resize_warp:
                    stride = self.va.warp_resize_stride
                    target_uv_warp = F.avg_pool3d(
                        target_uv_warp.permute(0, 4, 1, 2, 3),
                        kernel_size=stride,
                        stride=stride
                    ).permute(0, 2, 3, 4, 1)
                
                # Create rotation warp
                grid = self.va.identity_grid_3d.repeat_interleave(1, dim=0)
                target_rotation_warp = grid.bmm(target_theta.transpose(1, 2)).view(-1, d, s, s, 3)
                
                # 1. Generate target frame (main output)
                aligned_volume = self.va.grid_sample(
                    self.va.grid_sample(canonical_volume_b, target_uv_warp),
                    target_rotation_warp
                )
                target_latent_feats = aligned_volume.view(1, c * d, s, s)
                
                frame, _, _, _ = self.va.decoder_nw(
                    data_dict,
                    embed_dict,
                    target_latent_feats,
                    False,
                    stage_two=True
                )
                
                # Apply improved masking
                with torch.no_grad():
                    face_mask = self.va.face_idt.forward(frame)[0]
                    face_mask = (face_mask > self.mask_threshold).float()
                    
                    if face_mask.dim() == 3:
                        face_mask = face_mask.unsqueeze(0)
                    
                    # Apply Gaussian blur for softer edges (skip if not available)
                    try:
                        # Convert to PIL, blur, and back
                        from torchvision import transforms
                        to_pil = transforms.ToPILImage()
                        to_tensor = transforms.ToTensor()
                        blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
                        
                        # Process each channel
                        face_mask_pil = to_pil(face_mask[0])
                        face_mask_blurred = blur(face_mask_pil)
                        face_mask = to_tensor(face_mask_blurred).unsqueeze(0).to(face_mask.device)
                    except:
                        # Skip blurring if it fails
                        pass
                    
                    if use_black_background:
                        black_bg = torch.zeros_like(frame)
                        frame = frame * face_mask + black_bg * (1 - face_mask)
                    else:
                        bg = source_img[b:b+1]
                        frame = frame * face_mask + bg * (1 - face_mask)
                
                batch_frames.append(frame)
                
                # 2. Generate driving proxy (if enabled)
                if enable_3avatar and neutral_volume_b is not None:
                    # Apply motion to neutral volume
                    driving_aligned = self.va.grid_sample(
                        self.va.grid_sample(neutral_volume_b, target_uv_warp),
                        target_rotation_warp
                    )
                    driving_latent = driving_aligned.view(1, c * d, s, s)
                    
                    driving_frame, _, _, _ = self.va.decoder_nw(
                        data_dict,
                        embed_dict,
                        driving_latent,
                        False,
                        stage_two=True
                    )
                    
                    # Apply masking to driving proxy
                    driving_mask = self.va.face_idt.forward(driving_frame)[0]
                    driving_mask = (driving_mask > self.mask_threshold).float()
                    if driving_mask.dim() == 3:
                        driving_mask = driving_mask.unsqueeze(0)
                    # Apply blur if possible
                    try:
                        from torchvision import transforms
                        blur = transforms.GaussianBlur(kernel_size=5, sigma=1.0)
                        driving_mask = blur(driving_mask)
                    except:
                        pass
                    
                    if use_black_background:
                        driving_frame = driving_frame * driving_mask
                    
                    batch_driving.append(driving_frame)
                    
                    # 3. Create 3-avatar visualization
                    # Source | Driving | Target
                    viz_frame = torch.cat([
                        source_img[b:b+1],  # Static source
                        driving_frame,       # Motion on neutral
                        frame               # Final result
                    ], dim=3)  # Concatenate along width
                    batch_viz.append(viz_frame)
            
            # Stack frames for this batch
            batch_frames = torch.cat(batch_frames, dim=0)
            generated_frames.append(batch_frames)
            
            if enable_3avatar:
                batch_driving = torch.cat(batch_driving, dim=0)
                driving_proxy_frames.append(batch_driving)
                batch_viz = torch.cat(batch_viz, dim=0)
                viz_frames.append(batch_viz)
        
        # Stack all batches
        generated_frames = torch.stack(generated_frames, dim=0)
        
        result = {'frames': generated_frames}
        
        if enable_3avatar:
            driving_proxy_frames = torch.stack(driving_proxy_frames, dim=0)
            viz_frames = torch.stack(viz_frames, dim=0)
            result['driving_proxy'] = driving_proxy_frames
            result['viz_frames'] = viz_frames
            
        return result