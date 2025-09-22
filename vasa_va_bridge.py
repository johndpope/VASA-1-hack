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

            # Get proper embed_dict from predict_embed (matching create_video_face_swap.py)
            # This is crucial for identity preservation!
            source_warp_embed, _, _, embed_dict = self.va.predict_embed(data_dict)

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
                'source_masked': source_masked,
                'embed_dict': embed_dict  # The proper embed_dict from predict_embed
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
        Generate frames exactly like create_video_face_swap.py using VASA-predicted warps.

        Args:
            motion_outputs: VASA-generated motion with UV warps, expression_embed, theta
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

        # Process canonical volume once for all frames
        source_volume = source_data['source_volume']
        canonical_volume = self.va.volume_process_nw(source_volume)

        c = self.va.args.latent_volume_channels
        d = self.va.args.latent_volume_depth
        s = self.va.args.latent_volume_size

        # Use the proper embed_dict from predict_embed (as in create_video_face_swap.py)
        # This is CRUCIAL for correct identity preservation!
        identity_embed_dict = source_data['embed_dict']

        generated_frames = []

        for b in range(B):
            batch_frames = []
            canonical_volume_b = canonical_volume[b:b+1]

            for t in range(T):
                # Get VASA predictions for this frame (matching H5 cache structure)
                target_uv_warp = motion_outputs['uv_warps'][b:b+1, t]  # [1, 16, 64, 64, 3]
                target_theta = motion_outputs['theta'][b:b+1, t]  # [1, 3, 4] or [1, 4, 4]
                target_pose_embed = motion_outputs['expression_embed'][b:b+1, t]  # [1, 128]

                # Convert theta to 3x4 if needed
                if target_theta.shape[-2] == 4:
                    target_theta = target_theta[:, :3, :]

                # Generate 3D grid and rotation warp (exactly as in create_video_face_swap.py)
                grid = self.va.identity_grid_3d.repeat_interleave(1, dim=0)
                target_rotation_warp = grid.bmm(target_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

                # Apply warps exactly like create_video_face_swap.py - nested grid_sample calls
                aligned_target_volume = self.va.grid_sample(
                    self.va.grid_sample(canonical_volume_b, target_uv_warp),
                    target_rotation_warp
                )

                # Prepare for decoder
                target_latent_feats = aligned_target_volume.view(1, c * d, s, s)

                # Create decode_dict exactly as in create_video_face_swap.py
                decode_dict = {
                    'target_theta': target_theta,
                    'target_pose_embed': target_pose_embed
                }

                # Decode exactly as in create_video_face_swap.py
                generated_img, _, _, _ = self.va.decoder_nw(
                    decode_dict,
                    identity_embed_dict,  # Source identity
                    target_latent_feats,
                    False,
                    stage_two=True
                )

                # Adjust range if needed (as in create_video_face_swap.py)
                if generated_img.min() >= 0 and generated_img.max() <= 1.1:
                    generated_img = generated_img * 2 - 1

                # Get refined mask for compositing (as in create_video_face_swap.py)
                gen_mask, _, _, _ = self.va.face_idt.forward(generated_img)
                gen_mask = (gen_mask > 0.65).float()
                for _ in range(3):  # Smooth mask edges
                    gen_mask = F.avg_pool2d(gen_mask, 3, stride=1, padding=1)

                # Composite with background
                if use_black_background:
                    background = torch.zeros_like(generated_img)
                elif background_image is not None:
                    background = background_image[b:b+1] if background_image.shape[0] > 1 else background_image
                else:
                    background = source_img[b:b+1]

                frame = generated_img * gen_mask + background * (1 - gen_mask)
                frame = torch.clamp(frame, -1, 1)

                batch_frames.append(frame)

            # Stack frames for this batch item
            batch_frames = torch.cat(batch_frames, dim=0)
            generated_frames.append(batch_frames)

        # Stack all batch items [B, T, C, H, W]
        generated_frames = torch.stack(generated_frames, dim=0)

        return generated_frames