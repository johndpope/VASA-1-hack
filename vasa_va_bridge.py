#!/usr/bin/env python3
"""
Bridge module to properly connect VASA's motion generation with volumetric avatar's frame generation.

The key insight: Pipeline2.py works because it:
1. Extracts source expression/pose from the identity image
2. Uses DRIVER expression/pose for the target
3. Warps the source volume according to the difference

VASA should do the same:
1. Extract source expression from identity image ONCE
2. Use VASA-generated expression as target
3. Let volumetric avatar handle the warping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VASAVolumetricAvatarBridge:
    """
    Bridge between VASA's motion generation and volumetric avatar's frame generation.
    Ensures we follow the same pipeline as pipeline2.py for consistent quality.
    """
    
    def __init__(self, volumetric_avatar):
        """
        Args:
            volumetric_avatar: The volumetric avatar model
        """
        self.va = volumetric_avatar
        self._source_cache = {}
        
    def clear_cache(self):
        """Clear the source embedding cache."""
        self._source_cache = {}
        
    def get_source_embeddings(self, source_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract and cache source embeddings from identity image.
        This should be done ONCE per identity, not per frame.
        
        Args:
            source_img: Identity image [B, C, H, W]
            
        Returns:
            Dictionary with source embeddings
        """
        # Create a cache key based on tensor id
        cache_key = id(source_img)
        
        if cache_key in self._source_cache:
            logger.debug("Using cached source embeddings")
            return self._source_cache[cache_key]
        
        logger.info("Computing source embeddings from identity image")
        
        with torch.no_grad():
            # Get face mask
            face_mask, _, _, _ = self.va.face_idt.forward(source_img)
            face_mask = (face_mask > 0.6).float()
            source_masked = source_img * face_mask
            
            # Get identity embedding
            idt_embed = self.va.idt_embedder_nw(source_masked)
            
            # Get source pose/expression from the identity image itself
            # This is crucial - the source should have its OWN expression, not the target's
            data_dict = {
                'source_img': source_img,
                'target_img': source_img,  # Use source as target for extraction
                'source_mask': face_mask,
                'target_mask': face_mask,
                'idt_embed': idt_embed
            }
            
            # Get head pose from source
            if hasattr(self.va, 'head_pose_regressor'):
                source_theta = self.va.head_pose_regressor.forward(source_img)
                if source_theta.shape[-2] == 4:
                    source_theta = source_theta[:, :3, :]
                data_dict['source_theta'] = source_theta
                data_dict['target_theta'] = source_theta
            
            # Get expression embedding from source
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
            
            # Cache the results
            result = {
                'source_pose_embed': source_pose_embed,
                'source_theta': data_dict.get('source_theta'),
                'source_volume': source_volume,
                'source_mask': face_mask,
                'idt_embed': idt_embed,
                'source_masked': source_masked
            }
            
            self._source_cache[cache_key] = result
            logger.info(f"Cached source embeddings - pose_embed shape: {source_pose_embed.shape}")
            
        return result
    
    def generate_frames_from_motion(
        self,
        motion_outputs: Dict[str, torch.Tensor],
        source_img: torch.Tensor,
        use_black_background: bool = True,
        background_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate frames using VASA motion and volumetric avatar.
        
        This properly bridges VASA's motion generation with volumetric avatar's expectations:
        1. Source expression comes from identity image (computed once)
        2. Target expression comes from VASA generation
        3. Volumetric avatar handles the warping between them
        
        Args:
            motion_outputs: VASA-generated motion (expression_embed, theta, etc.)
            source_img: Identity image [B, C, H, W]
            use_black_background: Whether to use black background
            background_image: Optional background image
            
        Returns:
            Generated frames [B, T, C, H, W]
        """
        B, T = motion_outputs['theta'].shape[:2]
        device = motion_outputs['theta'].device
        
        # Get source embeddings (cached after first call)
        source_data = self.get_source_embeddings(source_img)
        
        # Process canonical volume
        source_volume = source_data['source_volume']
        canonical_volume = self.va.volume_process_nw(source_volume)
        
        c = self.va.args.latent_volume_channels
        d = self.va.args.latent_volume_depth
        s = self.va.args.latent_volume_size
        
        generated_frames = []
        
        for b in range(B):
            batch_frames = []
            canonical_volume_b = canonical_volume[b:b+1]
            
            for t in range(T):
                # Get VASA-generated motion for this frame
                target_expression = motion_outputs['expression_embed'][b:b+1, t]
                target_theta = motion_outputs['theta'][b:b+1, t]

                # Convert theta format if needed
                if target_theta.shape[-2] == 4:
                    target_theta = target_theta[:, :3, :]

                # Check if we have VASA-predicted warping fields
                use_predicted_warps = (
                    'xy_warps' in motion_outputs and
                    'rigid_warps' in motion_outputs and
                    'uv_warps' in motion_outputs
                )

                if use_predicted_warps:
                    # USE VASA-PREDICTED WARPING FIELDS (as in nemo/pipeline_face_attr.py)
                    logger.debug(f"Using VASA-predicted warps for frame {t}")

                    # Extract warps for this frame and batch
                    source_xy_warp = motion_outputs['xy_warps'][b:b+1, t]  # [1, D, H, W, 3]
                    target_rotation_warp = motion_outputs['rigid_warps'][b:b+1, t]  # [1, D, H, W, 3]
                    target_uv_warp = motion_outputs['uv_warps'][b:b+1, t]  # [1, D, H, W, 3]

                    # Handle resizing if needed for xy warps
                    if self.va.resize_warp and source_xy_warp.shape[2] != s:
                        stride = self.va.warp_resize_stride
                        source_xy_warp = F.avg_pool3d(
                            source_xy_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)

                    # Handle resizing for uv warps
                    if self.va.resize_warp and target_uv_warp.shape[2] != s:
                        stride = self.va.warp_resize_stride
                        target_uv_warp = F.avg_pool3d(
                            target_uv_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)

                    # Create data dict for decoder (still needed for some parameters)
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

                    # Get embed_dict for decoder (minimal computation)
                    _, _, _, embed_dict = self.va.predict_embed(data_dict)

                else:
                    # FALLBACK: Generate warps using volumetric avatar (original behavior)
                    logger.debug(f"Generating warps using volumetric avatar for frame {t}")

                    # Create data dict with proper source/target separation
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
                    source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = self.va.predict_embed(data_dict)

                    # Generate XY warp from source
                    source_xy_warp, _ = self.va.xy_generator_nw(source_warp_embed_dict)

                    # Generate UV warp from target expression
                    target_uv_warp, _ = self.va.uv_generator_nw(target_warp_embed_dict)

                    # Handle resizing if needed
                    if self.va.resize_warp:
                        stride = self.va.warp_resize_stride
                        source_xy_warp = F.avg_pool3d(
                            source_xy_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)
                        target_uv_warp = F.avg_pool3d(
                            target_uv_warp.permute(0, 4, 1, 2, 3),
                            kernel_size=stride,
                            stride=stride
                        ).permute(0, 2, 3, 4, 1)

                    # Create rotation warp from target pose
                    grid = self.va.identity_grid_3d.repeat_interleave(1, dim=0)
                    target_rotation_warp = grid.bmm(target_theta.transpose(1, 2)).view(-1, d, s, s, 3)
                
                # Apply warps to canonical volume (following nemo/pipeline_face_attr.py order)
                # 1. Apply source XY warp (non-rigid deformation in source space) if using predicted warps
                # 2. Apply rotation warp (rigid transformation)
                # 3. Apply target UV warp (expression-specific deformation)
                if use_predicted_warps:
                    # When using VASA-predicted warps, apply source_xy_warp first
                    aligned_volume = self.va.grid_sample(
                        self.va.grid_sample(
                            self.va.grid_sample(canonical_volume_b, source_xy_warp),
                            target_rotation_warp
                        ),
                        target_uv_warp
                    )
                else:
                    # Original order when generating warps (no source_xy_warp applied here)
                    aligned_volume = self.va.grid_sample(
                        self.va.grid_sample(canonical_volume_b, target_uv_warp),
                        target_rotation_warp
                    )
                
                target_latent_feats = aligned_volume.view(1, c * d, s, s)
                
                # Generate frame through decoder
                frame, _, _, _ = self.va.decoder_nw(
                    data_dict,
                    embed_dict,
                    target_latent_feats,
                    False,
                    stage_two=True
                )
                
                # Apply background compositing
                with torch.no_grad():
                    face_mask = self.va.face_idt.forward(frame)[0]
                    face_mask = (face_mask > 0.6).float()
                    
                    if face_mask.dim() == 3:
                        face_mask = face_mask.unsqueeze(0)
                    
                    if use_black_background:
                        black_bg = torch.zeros_like(frame)
                        frame = frame * face_mask + black_bg * (1 - face_mask)
                    elif background_image is not None:
                        bg = background_image[b:b+1] if background_image.shape[0] > 1 else background_image
                        frame = frame * face_mask + bg * (1 - face_mask)
                    else:
                        bg = source_img[b:b+1]
                        frame = frame * face_mask + bg * (1 - face_mask)
                
                batch_frames.append(frame)
            
            # Stack frames for this batch item
            batch_frames = torch.cat(batch_frames, dim=0)
            generated_frames.append(batch_frames)
        
        # Stack all batch items [B, T, C, H, W]
        generated_frames = torch.stack(generated_frames, dim=0)
        
        return generated_frames