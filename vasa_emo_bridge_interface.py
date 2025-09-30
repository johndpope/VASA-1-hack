#!/usr/bin/env python3
"""
Clean interface for bridging VASA with any volumetric avatar implementation.
This abstracts all VA-specific details from the dataset.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WarpExtractionConfig:
    """Configuration for warp extraction."""
    compute_xy_warps: bool = True
    compute_rigid_warps: bool = True
    compute_uv_warps: bool = True
    compute_source_theta: bool = True
    resize_warps: bool = True
    warp_size: Tuple[int, int, int] = (16, 64, 64)  # (depth, height, width)


@dataclass
class FrameWarpData:
    """Data for a single frame's warps."""
    xy_warp: Optional[torch.Tensor] = None  # [1, D, H, W, 3]
    rigid_warp: Optional[torch.Tensor] = None  # [1, D, H, W, 3]
    uv_warp: Optional[torch.Tensor] = None  # [1, D, H, W, 3]
    source_theta: Optional[torch.Tensor] = None  # [1, 3, 4] or [1, 4, 4]
    expression_embed: Optional[torch.Tensor] = None  # Expression embedding
    pose_embed: Optional[torch.Tensor] = None  # Pose embedding


@dataclass
class WindowWarpData:
    """Warp data for an entire window of frames."""
    xy_warps: Optional[torch.Tensor] = None  # [T, D, H, W, 3]
    rigid_warps: Optional[torch.Tensor] = None  # [T, D, H, W, 3]
    uv_warps: Optional[torch.Tensor] = None  # [T, D, H, W, 3]
    source_thetas: Optional[torch.Tensor] = None  # [T, 3, 4]
    expression_embeds: Optional[torch.Tensor] = None  # [T, expr_dim]
    pose_embeds: Optional[torch.Tensor] = None  # [T, pose_dim]
    identity_embed: Optional[torch.Tensor] = None  # [1, id_dim] - shared across window


class VolumetricAvatarBridgeInterface:
    """
    Abstract interface for volumetric avatar models.
    Implement this to support different VA backends.
    """

    def extract_warps_for_window(
        self,
        frames: torch.Tensor,
        identity_frame_idx: int = 0,
        config: Optional[WarpExtractionConfig] = None
    ) -> WindowWarpData:
        """
        Extract warps for an entire window of frames at once.

        Args:
            frames: Tensor of frames [T, C, H, W]
            identity_frame_idx: Which frame to use as identity reference (default: 0)
            config: Configuration for what to extract

        Returns:
            WindowWarpData containing all warps for the window
        """
        raise NotImplementedError

    def extract_warps_for_frame(
        self,
        identity_frame: torch.Tensor,
        target_frame: torch.Tensor,
        config: Optional[WarpExtractionConfig] = None
    ) -> FrameWarpData:
        """
        Extract warps for a single frame pair.

        Args:
            identity_frame: Identity/canonical frame [1, C, H, W]
            target_frame: Target frame with expression [1, C, H, W]
            config: Configuration for what to extract

        Returns:
            FrameWarpData containing warps for this frame
        """
        raise NotImplementedError

    def get_identity_embedding(
        self,
        identity_frame: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract identity embedding from a frame.

        Args:
            identity_frame: Identity frame [1, C, H, W]

        Returns:
            Identity embedding tensor
        """
        raise NotImplementedError

    def generate_canonical_view(
        self,
        identity_frame: torch.Tensor,
        use_identity_warps: bool = True
    ) -> torch.Tensor:
        """
        Generate canonical (neutral, front-facing) view of a person.

        Args:
            identity_frame: Source identity frame [B, C, H, W]
            use_identity_warps: If True, use identity warps (no deformation)

        Returns:
            Canonical view image [B, C, H, W]
        """
        raise NotImplementedError

    def clear_cache(self):
        """Clear any internal caches."""
        pass


class EMOPortraitsBridge(VolumetricAvatarBridgeInterface):
    """
    Bridge implementation for EMOPortraits/MegaPortraits volumetric avatar.
    """

    def __init__(self, emo_model):
        """
        Args:
            emo_model: The EMOPortraits volumetric avatar model
        """
        self.model = emo_model
        self.model.eval()

        # Set optimizer mode if needed
        if not hasattr(self.model, 'optimizer_idx_to_mode'):
            self.model.optimizer_idx_to_mode = {0: 'gen'}

        self._cache = {}

    def clear_cache(self):
        """Clear internal caches."""
        self._cache = {}

    def extract_warps_for_window(
        self,
        frames: torch.Tensor,
        identity_frame_idx: int = 0,
        config: Optional[WarpExtractionConfig] = None
    ) -> WindowWarpData:
        """
        Extract warps for an entire window of frames.
        This is optimized to reuse the identity frame processing.
        """
        if config is None:
            config = WarpExtractionConfig()

        T = frames.shape[0]
        device = frames.device

        # Get identity frame
        identity_frame = frames[identity_frame_idx:identity_frame_idx+1]

        # Process identity frame once
        identity_embed = self.get_identity_embedding(identity_frame)

        # Collect warps for all frames
        all_xy_warps = []
        all_rigid_warps = []
        all_uv_warps = []
        all_source_thetas = []
        all_expression_embeds = []
        all_pose_embeds = []

        logger.info(f"Extracting warps for {T} frames with identity frame {identity_frame_idx}")

        for t in range(T):
            target_frame = frames[t:t+1]

            # Extract warps for this frame
            frame_data = self._extract_single_frame_warps(
                identity_frame=identity_frame,
                target_frame=target_frame,
                identity_embed=identity_embed,
                config=config
            )

            # Collect results
            if config.compute_xy_warps and frame_data.xy_warp is not None:
                all_xy_warps.append(frame_data.xy_warp)
            if config.compute_rigid_warps and frame_data.rigid_warp is not None:
                all_rigid_warps.append(frame_data.rigid_warp)
            if config.compute_uv_warps and frame_data.uv_warp is not None:
                all_uv_warps.append(frame_data.uv_warp)
            if config.compute_source_theta and frame_data.source_theta is not None:
                all_source_thetas.append(frame_data.source_theta)
            if frame_data.expression_embed is not None:
                all_expression_embeds.append(frame_data.expression_embed)
            if frame_data.pose_embed is not None:
                all_pose_embeds.append(frame_data.pose_embed)

        # Stack all warps
        result = WindowWarpData(
            identity_embed=identity_embed
        )

        if all_xy_warps:
            result.xy_warps = torch.cat(all_xy_warps, dim=0)  # [T, D, H, W, 3]
        if all_rigid_warps:
            result.rigid_warps = torch.cat(all_rigid_warps, dim=0)
        if all_uv_warps:
            result.uv_warps = torch.cat(all_uv_warps, dim=0)
        if all_source_thetas:
            result.source_thetas = torch.cat(all_source_thetas, dim=0)
        if all_expression_embeds:
            result.expression_embeds = torch.cat(all_expression_embeds, dim=0)
        if all_pose_embeds:
            result.pose_embeds = torch.cat(all_pose_embeds, dim=0)

        logger.info(f"Extracted warps - XY: {result.xy_warps.shape if result.xy_warps is not None else None}")

        return result

    def extract_warps_for_frame(
        self,
        identity_frame: torch.Tensor,
        target_frame: torch.Tensor,
        config: Optional[WarpExtractionConfig] = None
    ) -> FrameWarpData:
        """Extract warps for a single frame pair."""
        if config is None:
            config = WarpExtractionConfig()

        identity_embed = self.get_identity_embedding(identity_frame)
        return self._extract_single_frame_warps(
            identity_frame=identity_frame,
            target_frame=target_frame,
            identity_embed=identity_embed,
            config=config
        )

    def _extract_single_frame_warps(
        self,
        identity_frame: torch.Tensor,
        target_frame: torch.Tensor,
        identity_embed: torch.Tensor,
        config: WarpExtractionConfig
    ) -> FrameWarpData:
        """Internal method to extract warps for a single frame."""

        # Prepare data dict
        data_dict = {
            'source_img': identity_frame,
            'target_img': target_frame,
            'idt_embed': identity_embed
        }

        with torch.no_grad():
            # Forward pass through model
            _, _, _, output_dict = self.model.forward(
                data_dict,
                phase='test',
                optimizer_idx=0,
                visualize=False
            )

        # Extract warps from output
        result = FrameWarpData()

        # XY warp (non-rigid expression normalization)
        if config.compute_xy_warps:
            for key in ['source_xy_warp', 'xy_warp', 'source_xy_warp_resize']:
                if key in output_dict:
                    result.xy_warp = output_dict[key].cpu()
                    break

        # Rigid warp (head pose)
        if config.compute_rigid_warps:
            for key in ['source_rotation_warp', 'rigid_warp']:
                if key in output_dict:
                    result.rigid_warp = output_dict[key].cpu()
                    break

        # UV warp (target deformation)
        if config.compute_uv_warps:
            for key in ['target_uv_warp', 'uv_warp', 'driver_uv_warp']:
                if key in output_dict:
                    result.uv_warp = output_dict[key].cpu()
                    break

        # Source theta (pose matrix)
        if config.compute_source_theta:
            for key in ['source_theta', 'theta']:
                if key in output_dict:
                    theta = output_dict[key]
                    # Convert 4x4 to 3x4 if needed
                    if theta.shape[-2:] == (4, 4):
                        theta = theta[..., :3, :]
                    result.source_theta = theta.cpu()
                    break

        # Expression and pose embeddings
        if 'source_pose_embed' in output_dict:
            result.pose_embed = output_dict['source_pose_embed'].cpu()
        if 'target_expression_embed' in output_dict:
            result.expression_embed = output_dict['target_expression_embed'].cpu()

        return result

    def get_identity_embedding(
        self,
        identity_frame: torch.Tensor
    ) -> torch.Tensor:
        """Extract identity embedding from a frame."""

        # Check cache
        cache_key = id(identity_frame)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with torch.no_grad():
            # Get face mask
            face_mask, _, _, _ = self.model.face_idt.forward(identity_frame)
            face_mask = (face_mask > 0.6).float()
            source_masked = identity_frame * face_mask

            # Get identity embedding
            idt_embed = self.model.idt_embedder_nw(source_masked)

            # Cache it
            self._cache[cache_key] = idt_embed

        return idt_embed

    def generate_canonical_view(
        self,
        identity_frame: torch.Tensor,
        use_identity_warps: bool = True
    ) -> torch.Tensor:
        """
        Generate canonical (neutral, front-facing) view of a person.

        Args:
            identity_frame: Source identity frame [B, C, H, W]
            use_identity_warps: If True, use identity warps (no deformation)

        Returns:
            Canonical view image [B, C, H, W]
        """
        B = identity_frame.shape[0]
        device = identity_frame.device

        with torch.no_grad():
            # Get identity embedding
            idt_embed = self.get_identity_embedding(identity_frame)

            # Get face mask
            face_mask, _, _, _ = self.model.face_idt.forward(identity_frame)
            face_mask = (face_mask > 0.6).float()
            source_masked = identity_frame * face_mask

            # Create canonical pose (identity matrix = no rotation)
            canonical_theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1)

            # Create data dict for canonical pose
            data_dict = {
                'source_img': identity_frame,
                'target_img': identity_frame,
                'source_mask': face_mask,
                'target_mask': face_mask,
                'idt_embed': idt_embed,
                'source_theta': canonical_theta,
                'target_theta': canonical_theta
            }

            # Get canonical expression (neutral)
            data_dict = self.model.expression_embedder_nw(data_dict, True, False)

            # Process source volume
            source_latents = self.model.local_encoder_nw(source_masked)
            c = self.model.args.latent_volume_channels
            d = self.model.args.latent_volume_depth
            s = self.model.args.latent_volume_size
            source_volume = source_latents.view(B, c, d, s, s)

            if self.model.args.source_volume_num_blocks > 0:
                source_volume = self.model.volume_source_nw(source_volume)

            # Process to canonical volume
            canonical_volume = self.model.volume_process_nw(source_volume)

            if use_identity_warps:
                # Use identity warps (no deformation) for true canonical
                aligned_volume = canonical_volume
            else:
                # Generate minimal warps for canonical pose
                source_warp_embed_dict, target_warp_embed_dict, _, _ = self.model.predict_embed(data_dict)

                source_xy_warp, _ = self.model.xy_generator_nw(source_warp_embed_dict)
                target_uv_warp, _ = self.model.uv_generator_nw(target_warp_embed_dict)

                # Resize if needed
                if self.model.resize_warp:
                    import torch.nn.functional as F
                    stride = self.model.warp_resize_stride
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

                # Apply warps
                aligned_volume = self.model.grid_sample(
                    self.model.grid_sample(canonical_volume, source_xy_warp),
                    target_uv_warp
                )

            # Decode to image
            target_latent_feats = aligned_volume.view(B, c * d, s, s)

            # Get embeddings for decoder
            _, _, _, embed_dict = self.model.predict_embed(data_dict)

            canonical_img, _, _, _ = self.model.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                False,
                stage_two=True
            )

            # Apply black background
            canonical_mask, _, _, _ = self.model.face_idt.forward(canonical_img)
            canonical_mask = (canonical_mask > 0.6).float()
            black_bg = torch.zeros_like(canonical_img)
            canonical_img = canonical_img * canonical_mask + black_bg * (1 - canonical_mask)

        return canonical_img


def create_bridge(model_type: str = "emoportraits", model: Any = None) -> VolumetricAvatarBridgeInterface:
    """
    Factory function to create the appropriate bridge.

    Args:
        model_type: Type of model ("emoportraits", "megaportraits", etc.)
        model: The actual model instance

    Returns:
        Bridge instance implementing VolumetricAvatarBridgeInterface
    """
    if model_type.lower() in ["emoportraits", "megaportraits", "emo", "va"]:
        return EMOPortraitsBridge(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")