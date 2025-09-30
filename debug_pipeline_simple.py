#!/usr/bin/env python3
"""
Simplified debug pipeline to trace the face swap issue.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import json
import cv2
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import importlib
from omegaconf import OmegaConf

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDebugTracer:
    """Simple debug tracer."""

    def __init__(self, output_dir="debug_simple"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.step = 0

    def save_image(self, img, name, convert_from_tensor=True):
        """Save image with step number."""
        self.step += 1

        if convert_from_tensor and isinstance(img, torch.Tensor):
            if len(img.shape) == 4:
                img = img[0]
            if img.shape[0] in [1, 3]:
                img = img.cpu().permute(1, 2, 0).numpy()
            else:
                img = img.cpu().numpy()

            # Normalize
            if img.min() < -0.1:
                img = (img + 1) / 2
            elif img.max() > 1.1:
                img = img / 255.0
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            # Squeeze single channel images
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img.squeeze(2)

        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                Image.fromarray(img, mode='L').save(
                    self.output_dir / f"{self.step:03d}_{name}.png"
                )
            else:
                Image.fromarray(img).save(
                    self.output_dir / f"{self.step:03d}_{name}.png"
                )

        logger.info(f"  Step {self.step:03d}: Saved {name}")

    def log(self, message):
        """Log a message."""
        logger.info(f"[{self.step:03d}] {message}")


def debug_face_swap():
    """Debug the face swap pipeline step by step."""

    tracer = SimpleDebugTracer()

    logger.info("="*60)
    logger.info("DEBUG FACE SWAP PIPELINE")
    logger.info("="*60)

    # Load model
    logger.info("Loading model...")
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    model = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    model_dict = torch.load(model_path, map_location='cuda')
    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    model.eval()

    # Load images
    logger.info("\nLoading images...")
    source_img = Image.open("nemo/data/IMG_1.png").convert('RGB')
    source_img = np.array(source_img.resize((512, 512)))
    tracer.save_image(source_img, "01_source_original", convert_from_tensor=False)

    target_img = Image.open("nemo/data/IMG_2.png").convert('RGB')
    target_img = np.array(target_img.resize((512, 512)))
    tracer.save_image(target_img, "02_target_original", convert_from_tensor=False)

    # Convert to tensors
    source_tensor = torch.from_numpy(source_img).float() / 127.5 - 1.0
    source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    target_tensor = torch.from_numpy(target_img).float() / 127.5 - 1.0
    target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        # ========== SOURCE IDENTITY EXTRACTION ==========
        tracer.log("PHASE 1: SOURCE IDENTITY EXTRACTION")

        # Face mask
        face_mask_s, _, _, _ = model.face_idt.forward(source_tensor)
        tracer.save_image(face_mask_s, "03_source_mask_raw")

        face_mask_s = (face_mask_s > 0.6).float()
        tracer.save_image(face_mask_s, "04_source_mask_thresh")

        # Masked source
        masked_source = source_tensor * face_mask_s
        tracer.save_image(masked_source, "05_masked_source")

        # Identity embedding
        idt_embed = model.idt_embedder_nw(masked_source)
        tracer.log(f"Identity embed shape: {idt_embed.shape}")
        tracer.log(f"Identity embed range: [{idt_embed.min().item():.3f}, {idt_embed.max().item():.3f}]")

        # Head pose
        source_theta, _, _, _ = model.head_pose_regressor.forward(source_tensor, True)
        tracer.log(f"Source theta shape: {source_theta.shape}")

        # Data dict
        data_dict_s = {
            'source_img': source_tensor,
            'source_mask': face_mask_s,
            'source_theta': source_theta,
            'target_img': source_tensor,
            'target_mask': face_mask_s,
            'target_theta': source_theta,
            'idt_embed': idt_embed
        }

        # Expression
        data_dict_s = model.expression_embedder_nw(data_dict_s, True, False, False)
        source_pose_embed = data_dict_s.get('source_pose_embed')
        tracer.log(f"Source pose embed: {source_pose_embed.shape if source_pose_embed is not None else None}")

        # Warp embeddings
        source_warp_embed, _, _, embed_dict = model.predict_embed(data_dict_s)
        tracer.log(f"Embed dict keys: {list(embed_dict.keys())}")

        # XY warps
        source_xy_warp, _ = model.xy_generator_nw(source_warp_embed)
        tracer.log(f"Source XY warp shape: {source_xy_warp.shape}")

        # Visualize XY warp
        if source_xy_warp.shape[1] > 8:
            xy_slice = source_xy_warp[0, 8].cpu().numpy()
            xy_mag = np.sqrt(xy_slice[..., 0]**2 + xy_slice[..., 1]**2)
            xy_mag = (xy_mag - xy_mag.min()) / (xy_mag.max() - xy_mag.min())
            tracer.save_image((xy_mag * 255).astype(np.uint8), "06_source_xy_warp", False)

        # Source volume
        source_latents = model.local_encoder_nw(masked_source)
        tracer.log(f"Source latents shape: {source_latents.shape}")

        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size
        tracer.log(f"Volume dims: c={c}, d={d}, s={s}")

        source_volume = source_latents.view(1, c, d, s, s)

        if model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)

        # Rotation warp
        grid = model.identity_grid_3d[:1]
        source_rot_warp = grid.bmm(source_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)

        # Apply warps
        rotated_source = model.grid_sample(source_volume, source_rot_warp)
        canonical_volume = model.grid_sample(rotated_source, source_xy_warp)

        # Process canonical
        processed_canonical = model.volume_process_nw(canonical_volume, embed_dict)
        tracer.log(f"Processed canonical shape: {processed_canonical.shape}")

        # ========== TARGET EXPRESSION APPLICATION ==========
        tracer.log("\nPHASE 2: TARGET EXPRESSION APPLICATION")

        # Target mask
        face_mask_t, _, _, _ = model.face_idt.forward(target_tensor)
        tracer.save_image(face_mask_t, "07_target_mask_raw")

        face_mask_t = (face_mask_t > 0.6).float()
        tracer.save_image(face_mask_t, "08_target_mask_thresh")

        # Target pose
        target_theta, _, _, _ = model.head_pose_regressor.forward(target_tensor, True)
        tracer.log(f"Target theta shape: {target_theta.shape}")

        # Target data dict WITH SOURCE IDENTITY
        data_dict_t = {
            'source_img': target_tensor,
            'source_mask': face_mask_t,
            'source_theta': target_theta,
            'target_img': target_tensor,
            'target_mask': face_mask_t,
            'target_theta': target_theta,
            'idt_embed': idt_embed  # SOURCE IDENTITY!
        }

        tracer.log("Using SOURCE identity embedding in target dict!")

        # Target expression
        data_dict_t = model.expression_embedder_nw(data_dict_t, True, False, False)
        target_pose_embed = data_dict_t.get('source_pose_embed')
        tracer.log(f"Target pose embed: {target_pose_embed.shape if target_pose_embed is not None else None}")

        # Target warps
        _, target_warp_embed, _, _ = model.predict_embed(data_dict_t)

        # UV warps
        target_uv_warp, _ = model.uv_generator_nw(target_warp_embed)
        tracer.log(f"Target UV warp shape: {target_uv_warp.shape}")

        # Visualize UV warp
        if target_uv_warp.shape[1] > 8:
            uv_slice = target_uv_warp[0, 8].cpu().numpy()
            uv_mag = np.sqrt(uv_slice[..., 0]**2 + uv_slice[..., 1]**2)
            uv_mag = (uv_mag - uv_mag.min()) / (uv_mag.max() - uv_mag.min())
            tracer.save_image((uv_mag * 255).astype(np.uint8), "09_target_uv_warp", False)

        # Apply UV warp
        volume_with_expression = model.grid_sample(processed_canonical, target_uv_warp)

        # Target rotation
        target_rot_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        final_volume = model.grid_sample(volume_with_expression, target_rot_warp)

        # Decode
        target_latent_feats = final_volume.view(1, c * d, s, s)

        decode_dict = {
            'target_theta': target_theta,
            'target_pose_embed': target_pose_embed
        }

        tracer.log("Decoding with SOURCE embed_dict!")

        generated_img, _, _, _ = model.decoder_nw(
            decode_dict,
            embed_dict,  # SOURCE IDENTITY!
            target_latent_feats,
            False,
            stage_two=True
        )

        tracer.log(f"Generated image range: [{generated_img.min().item():.3f}, {generated_img.max().item():.3f}]")
        tracer.save_image(generated_img, "10_generated_raw")

        # Check range
        if generated_img.min() >= 0 and generated_img.max() <= 1.1:
            tracer.log("Converting from [0,1] to [-1,1]")
            generated_img = generated_img * 2 - 1
            tracer.save_image(generated_img, "11_generated_converted")

        # Final mask
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        tracer.save_image(gen_mask, "12_gen_mask_raw")

        gen_mask = (gen_mask > 0.65).float()
        tracer.save_image(gen_mask, "13_gen_mask_thresh")

        # Smooth mask
        for _ in range(3):
            gen_mask = torch.nn.functional.avg_pool2d(gen_mask, 3, stride=1, padding=1)
        tracer.save_image(gen_mask, "14_gen_mask_smooth")

        # Composite
        background = target_tensor * (1 - gen_mask)
        tracer.save_image(background, "15_background")

        foreground = generated_img * gen_mask
        tracer.save_image(foreground, "16_foreground")

        final = foreground + background
        final = torch.clamp(final, -1, 1)
        tracer.save_image(final, "17_final_result")

        tracer.log(f"Final range: [{final.min().item():.3f}, {final.max().item():.3f}]")

        # Save comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(source_img)
        axes[0].set_title("Source (IMG_1)")
        axes[0].axis('off')

        axes[1].imshow(target_img)
        axes[1].set_title("Target (IMG_2)")
        axes[1].axis('off')

        gen_display = (generated_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[2].imshow(np.clip(gen_display, 0, 1))
        axes[2].set_title("Generated")
        axes[2].axis('off')

        final_display = (final[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[3].imshow(np.clip(final_display, 0, 1))
        axes[3].set_title("Final")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(tracer.output_dir / "comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"\nDebug images saved to: {tracer.output_dir}")
    logger.info("Check the sequence to identify where the issue occurs!")


if __name__ == "__main__":
    debug_face_swap()