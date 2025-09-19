#!/usr/bin/env python3
"""
Debug identity extraction to see what's going wrong.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_identity():
    """Debug what identity is being extracted."""

    # First, let's verify what IMG_1.png actually looks like
    img1_path = Path("nemo/data/IMG_1.png")
    if not img1_path.exists():
        logger.error(f"IMG_1.png not found at {img1_path}")
        return

    # Load and display IMG_1.png
    img1 = Image.open(img1_path)
    img1_np = np.array(img1)

    logger.info(f"IMG_1.png shape: {img1_np.shape}")
    logger.info(f"IMG_1.png dtype: {img1_np.dtype}")
    logger.info(f"IMG_1.png range: [{img1_np.min()}, {img1_np.max()}]")

    # Also load other images to compare
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("IMG_1.png (Source Identity)", fontsize=11, weight='bold', color='red')
    axes[0, 0].axis('off')

    # Check other images in the directory
    img_files = sorted(Path("nemo/data").glob("IMG_*.png"))[:4]
    for idx, img_path in enumerate(img_files):
        if idx < 4:
            img = Image.open(img_path)
            axes[0, idx].imshow(img)
            axes[0, idx].set_title(f"{img_path.name}", fontsize=10)
            axes[0, idx].axis('off')

    # Row 2: Let's also check what's in the cache
    import h5py
    cache_path = Path("proper_face_attributes.h5")
    if cache_path.exists():
        with h5py.File(cache_path, 'r') as f:
            # Check identity frame in cache
            if 'identity_frame' in f:
                cached_identity = f['identity_frame'][:]
                logger.info(f"Cached identity shape: {cached_identity.shape}")
                logger.info(f"Cached identity range: [{cached_identity.min()}, {cached_identity.max()}]")

                # Convert to displayable format
                if len(cached_identity.shape) == 4:  # NCHW format
                    cached_identity = cached_identity[0]  # Remove batch dim
                if cached_identity.shape[0] == 3:  # CHW format
                    cached_identity = np.transpose(cached_identity, (1, 2, 0))

                # Normalize to [0, 1] for display
                if cached_identity.min() < 0:
                    cached_identity = (cached_identity + 1) / 2
                else:
                    cached_identity = cached_identity / 255.0 if cached_identity.max() > 1 else cached_identity

                axes[1, 0].imshow(cached_identity)
                axes[1, 0].set_title("Cached Identity Frame", fontsize=10, color='blue')
                axes[1, 0].axis('off')

            # Check some cached frames
            for i in range(3):
                if f'frame_{i:04d}' in f:
                    frame = f[f'frame_{i:04d}/frame'][:]
                    # Remove batch dimension if present
                    if len(frame.shape) == 4:
                        frame = frame[0]
                    if frame.shape[0] == 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.min() < 0:
                        frame = (frame + 1) / 2
                    else:
                        frame = frame / 255.0 if frame.max() > 1 else frame

                    axes[1, i+1].imshow(frame)
                    axes[1, i+1].set_title(f"Cached Frame {i}", fontsize=10)
                    axes[1, i+1].axis('off')

    plt.suptitle("Identity Debug: What are we actually using?", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("debug_identity.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Saved debug_identity.png")

    # Now let's trace through the model to see what happens
    logger.info("\n" + "="*60)
    logger.info("Testing identity extraction from IMG_1.png")
    logger.info("="*60)

    # Load model
    import importlib
    from omegaconf import OmegaConf
    import cv2

    logger.info("Loading model...")
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    model = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        model.load_state_dict(model_dict, strict=False)

    model = model.cuda()
    model.eval()

    # Load IMG_1 as tensor - handle RGBA by converting to RGB
    img1_rgb = img1_np[:, :, :3] if img1_np.shape[2] == 4 else img1_np
    img1_resized = cv2.resize(img1_rgb, (512, 512))
    img1_tensor = torch.from_numpy(img1_resized).float() / 127.5 - 1.0
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    logger.info(f"IMG_1 tensor shape: {img1_tensor.shape}")
    logger.info(f"IMG_1 tensor range: [{img1_tensor.min().item():.3f}, {img1_tensor.max().item():.3f}]")

    with torch.no_grad():
        # Extract face mask
        face_mask, _, _, _ = model.face_idt.forward(img1_tensor)
        face_mask = (face_mask > 0.6).float()

        logger.info(f"Face mask shape: {face_mask.shape}")
        logger.info(f"Face mask coverage: {face_mask.mean().item():.3f}")

        # Masked image
        masked = img1_tensor * face_mask

        # Identity embedding
        idt_embed = model.idt_embedder_nw(masked)
        logger.info(f"Identity embedding shape: {idt_embed.shape}")
        logger.info(f"Identity embedding mean: {idt_embed.mean().item():.3f}")
        logger.info(f"Identity embedding std: {idt_embed.std().item():.3f}")

        # Let's also check what the decoder produces with just this identity
        # Create dummy inputs
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        # Get source latents
        source_latents = model.local_encoder_nw(masked)
        source_volume = source_latents.view(1, c, d, s, s)

        if model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)

        # Simple decode test
        dummy_theta, _, _, _ = model.head_pose_regressor.forward(img1_tensor, True)

        dummy_dict = {
            'target_theta': dummy_theta,
            'target_pose_embed': None
        }

        embed_dict = {'idt': idt_embed}

        # Process volume
        processed = model.volume_process_nw(source_volume, embed_dict)

        # Reshape for decoder
        latent_feats = processed.view(1, c * d, s, s)

        # Decode with identity only
        reconstructed, _, _, _ = model.decoder_nw(
            dummy_dict,
            embed_dict,
            latent_feats,
            False,
            stage_two=True
        )

        logger.info(f"Reconstructed shape: {reconstructed.shape}")
        logger.info(f"Reconstructed range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")

        # Convert and save reconstruction
        if reconstructed.min() >= 0:
            recon_np = reconstructed[0].cpu().permute(1, 2, 0).numpy()
        else:
            recon_np = (reconstructed[0].cpu().permute(1, 2, 0).numpy() + 1) / 2

        recon_np = np.clip(recon_np, 0, 1)

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img1)
        axes[0].set_title("Original IMG_1.png", fontsize=11, weight='bold')
        axes[0].axis('off')

        mask_display = face_mask[0, 0].cpu().numpy()
        axes[1].imshow(mask_display, cmap='gray')
        axes[1].set_title("Extracted Face Mask", fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(recon_np)
        axes[2].set_title("Reconstructed from Identity", fontsize=11, color='red')
        axes[2].axis('off')

        plt.suptitle("Identity Extraction Test", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("identity_reconstruction_test.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved identity_reconstruction_test.png")

        # Save the reconstruction separately
        recon_img = (recon_np * 255).astype(np.uint8)
        Image.fromarray(recon_img).save("identity_only_decode.png")
        logger.info("Saved identity_only_decode.png")

    logger.info("\n" + "="*60)
    logger.info("Debug complete! Check the generated images:")
    logger.info("  - debug_identity.png: Shows source images and cached data")
    logger.info("  - identity_reconstruction_test.png: Shows identity extraction")
    logger.info("  - identity_only_decode.png: Decoded using only IMG_1 identity")
    logger.info("="*60)


if __name__ == "__main__":
    debug_identity()