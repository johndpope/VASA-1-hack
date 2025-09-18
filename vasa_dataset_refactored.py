#!/usr/bin/env python3
"""
Refactored section of vasa_dataset.py using the clean bridge interface.
This shows how to replace the _extract_emo_features method.
"""

import torch
import logging
from typing import Dict, Optional
from vasa_emo_bridge_interface import (
    create_bridge,
    WarpExtractionConfig,
    VolumetricAvatarBridgeInterface
)

logger = logging.getLogger(__name__)


class VASAIntegratedDatasetRefactored:
    """
    Example of refactored dataset using the bridge interface.
    Only showing the relevant methods that interact with the volumetric model.
    """

    def __init__(
        self,
        video_folder: str,
        emo_model,  # The volumetric avatar model
        window_size: int = 4,
        **kwargs
    ):
        """Initialize dataset with bridge interface."""

        # Create the bridge instead of storing model directly
        self.emo_bridge: VolumetricAvatarBridgeInterface = create_bridge(
            model_type="emoportraits",
            model=emo_model
        )

        # Configuration for warp extraction
        self.warp_config = WarpExtractionConfig(
            compute_xy_warps=True,
            compute_rigid_warps=True,
            compute_uv_warps=True,
            compute_source_theta=True,
            resize_warps=True,
            warp_size=(16, 64, 64)
        )

        self.window_size = window_size
        # ... other initialization ...

    def _extract_emo_features(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract EMO features using the clean bridge interface.

        This replaces the original complex method with a clean abstraction.

        Args:
            frames: Tensor of frames [T, C, H, W] or [1, T, C, H, W]

        Returns:
            Dictionary containing all extracted warps and embeddings
        """

        # Handle batch dimension if present
        if frames.dim() == 5:
            # [1, T, C, H, W] -> [T, C, H, W]
            frames = frames.squeeze(0)

        T = frames.shape[0]

        logger.debug(f"=== EMO Feature Extraction via Bridge ===")
        logger.debug(f"Input frames shape: {frames.shape}")
        logger.debug(f"Extracting warps for {T} frames")

        try:
            # Use bridge to extract all warps for the window at once
            # This is much cleaner than the original implementation
            window_warps = self.emo_bridge.extract_warps_for_window(
                frames=frames,
                identity_frame_idx=0,  # Use first frame as identity
                config=self.warp_config
            )

            # Convert to the format expected by the rest of the dataset
            outputs = {}

            # XY warps (non-rigid expression normalization)
            if window_warps.xy_warps is not None:
                outputs['xy_warps'] = window_warps.xy_warps
                logger.debug(f"  XY warps: {window_warps.xy_warps.shape}")

            # Rigid warps (head pose)
            if window_warps.rigid_warps is not None:
                outputs['rigid_warps'] = window_warps.rigid_warps
                logger.debug(f"  Rigid warps: {window_warps.rigid_warps.shape}")

            # UV warps (target deformation)
            if window_warps.uv_warps is not None:
                outputs['uv_warps'] = window_warps.uv_warps
                logger.debug(f"  UV warps: {window_warps.uv_warps.shape}")

            # Source theta (pose matrices)
            if window_warps.source_thetas is not None:
                outputs['source_theta'] = window_warps.source_thetas
                logger.debug(f"  Source theta: {window_warps.source_thetas.shape}")

            # Identity embedding (shared across window)
            if window_warps.identity_embed is not None:
                # Expand to match window size for compatibility
                outputs['idt_embed'] = window_warps.identity_embed.expand(T, -1)
                logger.debug(f"  Identity embed: {outputs['idt_embed'].shape}")

            # Expression embeddings
            if window_warps.expression_embeds is not None:
                outputs['expression_embed'] = window_warps.expression_embeds
                logger.debug(f"  Expression embeds: {window_warps.expression_embeds.shape}")

            # Pose embeddings
            if window_warps.pose_embeds is not None:
                outputs['pose_embed'] = window_warps.pose_embeds
                logger.debug(f"  Pose embeds: {window_warps.pose_embeds.shape}")

            logger.debug(f"=== EMO Feature Extraction Complete ===\n")

            return outputs

        except Exception as e:
            logger.error(f"Error in EMO feature extraction via bridge: {str(e)}")
            logger.error(f"Input frames shape: {frames.shape}")
            raise

    def extract_warps_for_specific_frames(
        self,
        video_path: str,
        frame_indices: list
    ) -> Dict[str, torch.Tensor]:
        """
        Helper method to extract warps for specific frames from a video.
        This shows the flexibility of the bridge interface.
        """

        # Load frames from video
        frames = self._load_frames_from_video(video_path, frame_indices)

        # Extract warps using bridge
        window_warps = self.emo_bridge.extract_warps_for_window(
            frames=frames,
            identity_frame_idx=0,
            config=self.warp_config
        )

        # Return as dictionary
        return {
            'xy_warps': window_warps.xy_warps,
            'rigid_warps': window_warps.rigid_warps,
            'uv_warps': window_warps.uv_warps,
            'source_thetas': window_warps.source_thetas,
            'identity_embed': window_warps.identity_embed
        }

    def get_warps_for_training(
        self,
        frames: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get warps formatted specifically for VASA training.
        This abstracts all the EMO model details.
        """

        # Extract warps via bridge
        window_warps = self.emo_bridge.extract_warps_for_window(
            frames=frames,
            identity_frame_idx=0,
            config=self.warp_config
        )

        # Format for VASA training
        motion_data = {
            'xy_warps': window_warps.xy_warps,
            'rigid_warps': window_warps.rigid_warps,
            'uv_warps': window_warps.uv_warps,
            'source_theta': window_warps.source_thetas
        }

        # Add audio if provided
        if audio_features is not None:
            motion_data['audio_features'] = audio_features

        return motion_data

    def clear_emo_cache(self):
        """Clear any caches in the EMO bridge."""
        self.emo_bridge.clear_cache()

    def _load_frames_from_video(self, video_path: str, frame_indices: list) -> torch.Tensor:
        """Placeholder for frame loading logic."""
        # Implementation would load specific frames from video
        pass


# Example usage showing how clean the interface is
def example_usage():
    """Example of how to use the refactored dataset."""

    # Load volumetric model
    import importlib
    from omegaconf import OmegaConf
    from pathlib import Path

    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Create dataset with bridge
    dataset = VASAIntegratedDatasetRefactored(
        video_folder='vasa_train_data/',
        emo_model=volumetric_avatar,
        window_size=4
    )

    # Load some frames (example)
    frames = torch.randn(4, 3, 512, 512).cuda()  # [T, C, H, W]

    # Extract warps - all complexity hidden behind clean interface
    motion_data = dataset.get_warps_for_training(frames)

    print("Extracted motion data:")
    for key, value in motion_data.items():
        if value is not None:
            print(f"  {key}: {value.shape}")

    # Can also extract warps for specific frames
    warps = dataset.extract_warps_for_specific_frames(
        video_path="path/to/video.mp4",
        frame_indices=[0, 10, 20, 30]
    )

    print("\nExtracted warps for specific frames:")
    for key, value in warps.items():
        if value is not None:
            print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    # This would run the example
    # example_usage()
    print("Refactored dataset with clean bridge interface")