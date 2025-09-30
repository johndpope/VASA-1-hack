#!/usr/bin/env python3
"""
Debug version with comprehensive logging to understand the exact pipeline flow.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import cv2
from PIL import Image
from typing import Dict, Tuple, Any
import json

# Add paths
sys.path.insert(0, 'nemo')

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to enable/disable detailed tensor logging
DETAILED_TENSOR_LOG = True

def log_tensor(name: str, tensor: Any, detailed: bool = True):
    """Log tensor information with shape and statistics."""
    if tensor is None:
        logger.debug(f"  {name}: None")
        return

    if not isinstance(tensor, torch.Tensor):
        logger.debug(f"  {name}: type={type(tensor)}")
        return

    info = f"  {name}: shape={list(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}"

    if detailed and DETAILED_TENSOR_LOG and tensor.numel() > 0:
        with torch.no_grad():
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                info += f", range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
                info += f", mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"

                # Check for NaN or Inf
                if torch.isnan(tensor).any():
                    info += ", WARNING: Contains NaN!"
                if torch.isinf(tensor).any():
                    info += ", WARNING: Contains Inf!"

    logger.debug(info)


def log_dict(name: str, data_dict: Dict, detailed: bool = True):
    """Log dictionary contents."""
    logger.debug(f"\n=== {name} ===")
    if data_dict is None:
        logger.debug("  Dictionary is None")
        return

    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            log_tensor(key, value, detailed)
        elif isinstance(value, dict):
            logger.debug(f"  {key}: nested dict with {len(value)} keys")
        else:
            logger.debug(f"  {key}: {type(value).__name__}")


class DebugVolumetricAvatar:
    """Wrapper around volumetric avatar model with debug logging."""

    def __init__(self, model):
        self.model = model
        self.step_counter = 0

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        return getattr(self.model, name)

    def log_step(self, step_name: str):
        """Log a processing step."""
        self.step_counter += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {self.step_counter}: {step_name}")
        logger.info(f"{'='*60}")

    def face_idt_forward(self, img: torch.Tensor):
        """Wrapped face_idt forward with logging."""
        self.log_step("Face IDT Forward")
        log_tensor("input", img)

        result = self.model.face_idt.forward(img)

        log_tensor("face_mask", result[0])
        log_tensor("cloth_mask", result[3] if len(result) > 3 else None)

        return result

    def idt_embedder_forward(self, masked_img: torch.Tensor):
        """Wrapped identity embedder with logging."""
        self.log_step("Identity Embedder")
        log_tensor("masked_input", masked_img)

        idt_embed = self.model.idt_embedder_nw(masked_img)

        log_tensor("idt_embed", idt_embed)

        return idt_embed

    def local_encoder_forward(self, masked_img: torch.Tensor):
        """Wrapped local encoder with logging."""
        self.log_step("Local Encoder")
        log_tensor("masked_input", masked_img)

        latents = self.model.local_encoder_nw(masked_img)

        log_tensor("source_latents", latents)

        return latents

    def head_pose_forward(self, img: torch.Tensor):
        """Wrapped head pose regressor with logging."""
        self.log_step("Head Pose Regressor")
        log_tensor("input", img)

        theta, scale, rotation, translation = self.model.head_pose_regressor.forward(img, True)

        log_tensor("theta", theta)
        logger.debug(f"  scale={scale}, rotation={rotation}, translation={translation}")

        return theta, scale, rotation, translation

    def expression_embedder_forward(self, data_dict: Dict):
        """Wrapped expression embedder with logging."""
        self.log_step("Expression Embedder")
        log_dict("input_data_dict", data_dict, detailed=False)

        result_dict = self.model.expression_embedder_nw(data_dict, True, False, False)

        log_dict("output_data_dict", result_dict, detailed=False)
        log_tensor("source_pose_embed", result_dict.get('source_pose_embed'))
        log_tensor("target_pose_embed", result_dict.get('target_pose_embed'))

        return result_dict

    def predict_embed_forward(self, data_dict: Dict):
        """Wrapped predict_embed with logging."""
        self.log_step("Predict Embed")
        log_dict("input_data_dict", data_dict, detailed=False)

        source_warp, target_warp, _, embed_dict = self.model.predict_embed(data_dict)

        logger.debug("Source warp embed dict:")
        if isinstance(source_warp, dict):
            for k, v in source_warp.items():
                log_tensor(f"  source_warp[{k}]", v, detailed=False)

        logger.debug("Target warp embed dict:")
        if isinstance(target_warp, dict):
            for k, v in target_warp.items():
                log_tensor(f"  target_warp[{k}]", v, detailed=False)

        log_dict("embed_dict", embed_dict, detailed=False)

        return source_warp, target_warp, _, embed_dict

    def xy_generator_forward(self, warp_embed_dict: Dict):
        """Wrapped XY generator with logging."""
        self.log_step("XY Warp Generator")

        xy_warp, xy_conf = self.model.xy_generator_nw(warp_embed_dict)

        log_tensor("xy_warp", xy_warp)
        log_tensor("xy_conf", xy_conf if xy_conf is not None else None)

        return xy_warp, xy_conf

    def uv_generator_forward(self, warp_embed_dict: Dict):
        """Wrapped UV generator with logging."""
        self.log_step("UV Warp Generator")

        uv_warp, uv_conf = self.model.uv_generator_nw(warp_embed_dict)

        log_tensor("uv_warp", uv_warp)
        log_tensor("uv_conf", uv_conf if uv_conf is not None else None)

        return uv_warp, uv_conf

    def volume_source_forward(self, volume: torch.Tensor):
        """Wrapped volume source network with logging."""
        self.log_step("Volume Source Network")
        log_tensor("input_volume", volume)

        if self.model.args.source_volume_num_blocks > 0:
            processed = self.model.volume_source_nw(volume)
            log_tensor("processed_volume", processed)
            return processed
        else:
            logger.debug("  Skipping (num_blocks=0)")
            return volume

    def volume_process_forward(self, volume: torch.Tensor, embed_dict: Dict):
        """Wrapped volume process network with logging."""
        self.log_step("Volume Process Network")
        log_tensor("input_volume", volume)
        log_dict("embed_dict", embed_dict, detailed=False)

        processed = self.model.volume_process_nw(volume, embed_dict)

        log_tensor("processed_volume", processed)

        return processed

    def grid_sample_forward(self, volume: torch.Tensor, warp: torch.Tensor, name: str = ""):
        """Wrapped grid sample with logging."""
        self.log_step(f"Grid Sample {name}")
        log_tensor("input_volume", volume)
        log_tensor("warp_field", warp)

        warped = self.model.grid_sample(volume, warp)

        log_tensor("warped_volume", warped)

        return warped

    def decoder_forward(self, data_dict: Dict, embed_dict: Dict,
                       latent_feats: torch.Tensor, stage_two: bool = True):
        """Wrapped decoder with logging."""
        self.log_step("Decoder Network")
        log_dict("data_dict", data_dict, detailed=False)
        log_dict("embed_dict", embed_dict, detailed=False)
        log_tensor("latent_feats", latent_feats)
        logger.debug(f"  stage_two={stage_two}")

        img, _, deep_f, img_f = self.model.decoder_nw(
            data_dict, embed_dict, latent_feats, False, stage_two=stage_two
        )

        log_tensor("generated_img", img)
        log_tensor("deep_features", deep_f)
        log_tensor("img_features", img_f)

        return img, _, deep_f, img_f


def debug_face_swap_pipeline():
    """Run face swap with detailed debug logging."""

    logger.info("\n" + "="*80)
    logger.info("STARTING DEBUG FACE SWAP PIPELINE")
    logger.info("="*80)

    # Load model
    logger.info("\nLoading volumetric avatar model...")
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')

    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        logger.info("Model weights loaded successfully")

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    # Wrap model with debug logging
    model = DebugVolumetricAvatar(volumetric_avatar)

    # Load images
    logger.info("\nLoading images...")

    def load_image(path):
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        if img.shape[:2] != (512, 512):
            img = cv2.resize(img, (512, 512))
        img_tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        return img_tensor

    source_img = load_image("nemo/data/IMG_1.png")
    target_img = load_image("nemo/data/IMG_2.png")

    logger.info(f"Source image loaded: {source_img.shape}")
    logger.info(f"Target image loaded: {target_img.shape}")

    # ============ PROCESS SOURCE (IDENTITY) ============
    logger.info("\n" + "#"*80)
    logger.info("PHASE 1: PROCESSING SOURCE IMAGE (IDENTITY EXTRACTION)")
    logger.info("#"*80)

    with torch.no_grad():
        # Face detection and masking
        face_mask, _, _, cloth = model.face_idt_forward(source_img)
        face_mask = (face_mask > 0.6).float()

        # Mask source
        model.log_step("Masking Source Image")
        masked_source = source_img * face_mask
        log_tensor("masked_source", masked_source)

        # Identity embedding
        idt_embed = model.idt_embedder_forward(masked_source)

        # Head pose
        source_theta, _, _, _ = model.head_pose_forward(source_img)

        # Data dict for source
        model.log_step("Creating Source Data Dict")
        source_data_dict = {
            'source_img': source_img,
            'source_mask': face_mask,
            'source_theta': source_theta,
            'target_img': source_img,
            'target_mask': face_mask,
            'target_theta': source_theta,
            'idt_embed': idt_embed
        }
        log_dict("source_data_dict", source_data_dict, detailed=False)

        # Expression embedding
        source_data_dict = model.expression_embedder_forward(source_data_dict)
        source_pose_embed = source_data_dict['source_pose_embed']

        # Warp embeddings
        source_warp_embed, _, _, embed_dict = model.predict_embed_forward(source_data_dict)

        # XY warps
        source_xy_warp, _ = model.xy_generator_forward(source_warp_embed)

        # Source volume
        source_latents = model.local_encoder_forward(masked_source)

        model.log_step("Creating Source Volume")
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size
        logger.debug(f"  Volume dimensions: c={c}, d={d}, s={s}")

        source_volume = source_latents.view(1, c, d, s, s)
        log_tensor("source_volume", source_volume)

        # Process source volume
        source_volume = model.volume_source_forward(source_volume)

        # Create rotation warp
        model.log_step("Creating Source Rotation Warp")
        grid = model.identity_grid_3d[:1]
        log_tensor("identity_grid", grid)

        source_rotation_warp = grid.bmm(source_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        log_tensor("source_rotation_warp", source_rotation_warp)

        # Apply warps to get canonical
        rotated_source = model.grid_sample_forward(source_volume, source_rotation_warp, "(source rotation)")
        canonical_volume = model.grid_sample_forward(rotated_source, source_xy_warp, "(source XY)")

        # Process canonical
        processed_canonical = model.volume_process_forward(canonical_volume, embed_dict)

        logger.info("\n>>> SOURCE PROCESSING COMPLETE <<<")
        logger.info(f"  Identity embedding shape: {idt_embed.shape}")
        logger.info(f"  Processed canonical shape: {processed_canonical.shape}")

    # ============ PROCESS TARGET (EXPRESSION) ============
    logger.info("\n" + "#"*80)
    logger.info("PHASE 2: PROCESSING TARGET IMAGE (EXPRESSION APPLICATION)")
    logger.info("#"*80)

    with torch.no_grad():
        # Face detection for target
        target_face_mask, _, _, _ = model.face_idt_forward(target_img)
        target_face_mask = (target_face_mask > 0.6).float()

        # Target head pose
        target_theta, _, _, _ = model.head_pose_forward(target_img)

        # Data dict for target (BUT WITH SOURCE IDENTITY!)
        model.log_step("Creating Target Data Dict (with Source Identity)")
        target_data_dict = {
            'source_img': target_img,
            'source_mask': target_face_mask,
            'source_theta': target_theta,
            'target_img': target_img,
            'target_mask': target_face_mask,
            'target_theta': target_theta,
            'idt_embed': idt_embed  # SOURCE IDENTITY!
        }
        log_dict("target_data_dict", target_data_dict, detailed=False)

        # Target expression
        target_data_dict = model.expression_embedder_forward(target_data_dict)
        target_pose_embed = target_data_dict['source_pose_embed']

        # Target warps
        _, target_warp_embed, _, _ = model.predict_embed_forward(target_data_dict)

        # UV warps
        target_uv_warp, _ = model.uv_generator_forward(target_warp_embed)

        # Apply UV warp to canonical
        target_with_expression = model.grid_sample_forward(
            processed_canonical, target_uv_warp, "(target UV)"
        )

        # Target rotation
        model.log_step("Creating Target Rotation Warp")
        target_rotation_warp = grid.bmm(target_theta[:, :3, :].transpose(1, 2)).view(1, d, s, s, 3)
        log_tensor("target_rotation_warp", target_rotation_warp)

        final_volume = model.grid_sample_forward(
            target_with_expression, target_rotation_warp, "(target rotation)"
        )

        # Prepare for decoder
        model.log_step("Preparing Decoder Input")
        target_latent_feats = final_volume.view(1, c * d, s, s)
        log_tensor("target_latent_feats", target_latent_feats)

        decode_dict = {
            'target_theta': target_theta,
            'target_pose_embed': target_pose_embed
        }
        log_dict("decode_dict", decode_dict, detailed=False)

        # Decode
        generated_img, _, _, _ = model.decoder_forward(
            decode_dict, embed_dict, target_latent_feats, stage_two=True
        )

        # Final masking
        gen_mask, _, _, _ = model.face_idt_forward(generated_img)
        gen_mask = (gen_mask > 0.6).float()

        # Composite
        model.log_step("Final Compositing")
        background = target_img * (1 - gen_mask)
        final_img = generated_img * gen_mask + background
        log_tensor("final_img", final_img)

    # Save results
    logger.info("\n" + "#"*80)
    logger.info("PHASE 3: SAVING RESULTS")
    logger.info("#"*80)

    # Convert and save
    def tensor_to_image(tensor):
        img = tensor[0].cpu().permute(1, 2, 0).numpy()
        img = ((img + 1) * 127.5).astype(np.uint8)
        return img

    source_np = tensor_to_image(source_img)
    target_np = tensor_to_image(target_img)
    generated_np = tensor_to_image(generated_img)
    final_np = tensor_to_image(final_img)

    # Create comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(source_np)
    axes[0].set_title("Source (Identity)")
    axes[0].axis('off')

    axes[1].imshow(target_np)
    axes[1].set_title("Target (Expression)")
    axes[1].axis('off')

    axes[2].imshow(generated_np)
    axes[2].set_title("Generated (Raw)")
    axes[2].axis('off')

    axes[3].imshow(final_np)
    axes[3].set_title("Final (Composite)")
    axes[3].axis('off')

    plt.suptitle("Debug Face Swap Pipeline", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("debug_face_swap_result.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Saved result to debug_face_swap_result.png")

    # Also save individual
    Image.fromarray(final_np).save("debug_final_output.png")
    logger.info("Saved final output to debug_final_output.png")

    logger.info("\n" + "="*80)
    logger.info("DEBUG PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info("\nSummary:")
    logger.info(f"  Total steps logged: {model.step_counter}")
    logger.info(f"  Source identity shape: {idt_embed.shape}")
    logger.info(f"  Final output shape: {final_img.shape}")
    logger.info(f"  Output range: [{final_img.min().item():.3f}, {final_img.max().item():.3f}]")


if __name__ == "__main__":
    try:
        debug_face_swap_pipeline()
    except Exception as e:
        logger.error(f"Error during pipeline: {e}")
        import traceback
        traceback.print_exc()