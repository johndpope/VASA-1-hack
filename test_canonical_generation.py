#!/usr/bin/env python3
"""
Test generating canonical view of a person from their identity.
The canonical view is the neutral, front-facing representation.
"""

import torch
import sys
import logging
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np

# Add nemo to path
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_volumetric_model():
    """Load the volumetric avatar model."""
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    # Set optimizer mode
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def generate_canonical_view(model, source_img: torch.Tensor) -> torch.Tensor:
    """
    Generate canonical (neutral, front-facing) view of a person.

    Args:
        model: Volumetric avatar model
        source_img: Source identity image [B, C, H, W]

    Returns:
        Canonical view image [B, C, H, W]
    """
    logger.info("Generating canonical view...")

    with torch.no_grad():
        # Get face mask
        face_mask, _, _, _ = model.face_idt.forward(source_img)
        face_mask = (face_mask > 0.6).float()
        source_masked = source_img * face_mask

        # Get identity embedding
        idt_embed = model.idt_embedder_nw(source_masked)

        # Create neutral/canonical pose
        B = source_img.shape[0]
        device = source_img.device

        # Canonical theta is identity matrix (no rotation)
        canonical_theta = torch.eye(3, 4, device=device).unsqueeze(0).expand(B, -1, -1)

        # Create data dict for canonical pose
        data_dict = {
            'source_img': source_img,
            'target_img': source_img,  # Using source as target for canonical
            'source_mask': face_mask,
            'target_mask': face_mask,
            'idt_embed': idt_embed,
            'source_theta': canonical_theta,
            'target_theta': canonical_theta
        }

        # Get canonical expression (neutral)
        # The model should map to canonical/neutral expression
        data_dict = model.expression_embedder_nw(data_dict, True, False)
        source_pose_embed = data_dict['source_pose_embed']

        # Process source volume
        source_latents = model.local_encoder_nw(source_masked)
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size
        source_volume = source_latents.view(B, c, d, s, s)

        if model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)

        # Process to canonical volume
        canonical_volume = model.volume_process_nw(source_volume)

        # Generate canonical warps (should be minimal/identity)
        source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = model.predict_embed(data_dict)

        # For canonical view, we want minimal warping
        # The warps should be close to identity since we're already in canonical space

        # Option 1: Use identity warps (no deformation)
        use_identity_warps = True

        if use_identity_warps:
            # For canonical view with identity warps, no warping needed
            # The canonical volume is already in canonical space
            aligned_volume = canonical_volume

        else:
            # Option 2: Use model-generated warps for canonical pose
            source_xy_warp, _ = model.xy_generator_nw(source_warp_embed_dict)
            target_uv_warp, _ = model.uv_generator_nw(target_warp_embed_dict)

            # Resize if needed
            if model.resize_warp:
                stride = model.warp_resize_stride
                source_xy_warp = torch.nn.functional.avg_pool3d(
                    source_xy_warp.permute(0, 4, 1, 2, 3),
                    kernel_size=stride,
                    stride=stride
                ).permute(0, 2, 3, 4, 1)

                target_uv_warp = torch.nn.functional.avg_pool3d(
                    target_uv_warp.permute(0, 4, 1, 2, 3),
                    kernel_size=stride,
                    stride=stride
                ).permute(0, 2, 3, 4, 1)

            # Apply warps
            aligned_volume = model.grid_sample(
                model.grid_sample(canonical_volume, source_xy_warp),
                target_uv_warp
            )

        # Decode to image
        target_latent_feats = aligned_volume.view(B, c * d, s, s)

        canonical_img, _, _, _ = model.decoder_nw(
            data_dict,
            embed_dict,
            target_latent_feats,
            False,
            stage_two=True
        )

        # Apply background
        canonical_mask, _, _, _ = model.face_idt.forward(canonical_img)
        canonical_mask = (canonical_mask > 0.6).float()

        # Black background for canonical view
        black_bg = torch.zeros_like(canonical_img)
        canonical_img = canonical_img * canonical_mask + black_bg * (1 - canonical_mask)

        logger.info(f"Generated canonical view with shape: {canonical_img.shape}")

    return canonical_img


def test_canonical_generation():
    """Test generating canonical views from different source poses."""

    logger.info("Loading model...")
    model = load_volumetric_model()

    # Load test frames from video
    import cv2
    video_path = Path("temp_single_video/15.mp4")

    if not video_path.exists():
        logger.error("Test video not found!")
        # Create synthetic test image
        logger.info("Creating synthetic test image...")
        test_img = torch.randn(1, 3, 512, 512).cuda()
    else:
        logger.info(f"Loading frames from {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        # Get frames at different points (different expressions/poses)
        frame_indices = [0, 50, 100, 150]
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (512, 512))
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC -> CHW
                frames.append(frame)

        cap.release()

        if not frames:
            logger.error("Could not load frames from video")
            return

        # Stack frames
        test_imgs = torch.stack(frames).cuda()
        logger.info(f"Loaded {len(frames)} test frames")

    # Generate canonical views
    logger.info("\n=== Generating Canonical Views ===")

    if 'test_imgs' in locals():
        # Process each frame
        canonical_views = []

        for i, test_img in enumerate(test_imgs):
            logger.info(f"\nProcessing frame {i+1}/{len(test_imgs)}...")
            canonical_img = generate_canonical_view(model, test_img.unsqueeze(0))
            canonical_views.append(canonical_img)

        # Stack results
        canonical_views = torch.cat(canonical_views, dim=0)

        # Visualize results
        visualize_canonical_views(test_imgs, canonical_views)
    else:
        # Single synthetic image
        canonical_img = generate_canonical_view(model, test_img)
        logger.info("Generated canonical view from synthetic image")

    logger.info("\n=== Canonical Generation Complete ===")


def visualize_canonical_views(source_imgs, canonical_imgs):
    """Visualize source images and their canonical views."""

    n = source_imgs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

    fig.suptitle("Source Images → Canonical Views", fontsize=14)

    for i in range(n):
        # Source image
        img_src = source_imgs[i].cpu().permute(1, 2, 0).numpy()
        img_src = np.clip(img_src, 0, 1)
        axes[0, i].imshow(img_src)
        axes[0, i].set_title(f"Source {i+1}")
        axes[0, i].axis('off')

        # Canonical view
        img_can = canonical_imgs[i].cpu().permute(1, 2, 0).numpy()
        img_can = np.clip(img_can, 0, 1)
        axes[1, i].imshow(img_can)
        axes[1, i].set_title(f"Canonical {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = "canonical_views.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()

    # Check consistency
    logger.info("\n=== Checking Canonical View Consistency ===")

    # Canonical views from the same person should be similar
    for i in range(n - 1):
        for j in range(i + 1, n):
            can_i = canonical_imgs[i]
            can_j = canonical_imgs[j]

            # Compute similarity
            diff = torch.abs(can_i - can_j).mean().item()
            logger.info(f"Canonical {i+1} vs {j+1}: diff = {diff:.4f}")

            if diff < 0.1:
                logger.info(f"  ✓ Similar canonical views (good consistency)")
            else:
                logger.warning(f"  ⚠️ Different canonical views (may need tuning)")


if __name__ == "__main__":
    test_canonical_generation()