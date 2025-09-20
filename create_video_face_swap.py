#!/usr/bin/env python3
"""
Create face swap results matching a video using IMG_1.png as identity.
Follows the pipeline flow from pipeline5.py with H5 warp caching.
Fixed pose estimation and robust error handling.
"""
from ibug.face_detection import RetinaFacePredictor

from ibug.face_detection.utils.head_pose_estimator import HeadPoseEstimator
import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
import math
from pathlib import Path
import importlib
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Tuple, List, Optional
import h5py
from torchvision.transforms import ToTensor, ToPILImage

# Add paths for model imports
sys.path.insert(0, 'nemo')

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize transforms
to_tensor = ToTensor()
to_pil = ToPILImage()


def load_volumetric_model(device: str = 'cuda') -> torch.nn.Module:
    """Load the volumetric avatar model and prepare it for inference."""
    logger.info("Loading volumetric avatar model...")

    # Load configuration
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')
    
    # Import and instantiate model
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Load checkpoint
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    try:
        if Path(model_path).exists():
            model_dict = torch.load(model_path, map_location=device)
            volumetric_avatar.load_state_dict(model_dict, strict=False)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.error(f"Model checkpoint not found at {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    # Move to device and set to evaluation mode
    volumetric_avatar = volumetric_avatar.to(device)
    volumetric_avatar.eval()

    # Ensure optimizer_idx_to_mode is set (for compatibility with pipeline)
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar


def load_image_tensor(image_path: str, device: str = 'cuda') -> torch.Tensor:
    """Load image as tensor in [0, 1] range, compatible with model expectations."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        raise

    # Resize to 512x512 if needed
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.LANCZOS)

    # Convert to tensor [1, 3, 512, 512] in [0, 1]
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    logger.debug(f"Loaded image {image_path} with shape {img_tensor.shape}")

    return img_tensor


def get_pose_matrix(model, img_tensor: torch.Tensor, face_detector, pose_estimator) -> torch.Tensor:
    """Estimate 4x4 pose matrix using RetinaFace and HeadPoseEstimator."""
    logger.debug("Estimating pose for image...")

    # Convert tensor to numpy BGR image for face detection
    img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Detect faces
    try:
        detections = face_detector(img_np)
    except Exception as e:
        logger.warning(f"Face detection failed: {str(e)}")
        return torch.eye(4, dtype=torch.float32, device=img_tensor.device).unsqueeze(0)

    if len(detections) == 0:
        logger.debug("No face detected, using identity matrix for pose")
        return torch.eye(4, dtype=torch.float32, device=img_tensor.device).unsqueeze(0)

    # Take the first (most confident) face
    face = detections[0]
    landmarks = face[5:].reshape(-1, 2)

    # Estimate pose angles
    try:
        pitch, yaw, roll = pose_estimator(landmarks)
        logger.debug(f"Pose angles: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f} degrees")
    except Exception as e:
        logger.warning(f"Pose estimation failed: {str(e)}")
        return torch.eye(4, dtype=torch.float32, device=img_tensor.device).unsqueeze(0)

    # Convert to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    # Rotation matrices (Rz @ Ry @ Rx for head pose)
    cos = torch.cos
    sin = torch.sin
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos(torch.tensor(pitch_rad, device=img_tensor.device)), -sin(torch.tensor(pitch_rad, device=img_tensor.device))],
        [0, sin(torch.tensor(pitch_rad, device=img_tensor.device)), cos(torch.tensor(pitch_rad, device=img_tensor.device))]
    ], dtype=torch.float32, device=img_tensor.device)

    R_y = torch.tensor([
        [cos(torch.tensor(yaw_rad, device=img_tensor.device)), 0, sin(torch.tensor(yaw_rad, device=img_tensor.device))],
        [0, 1, 0],
        [-sin(torch.tensor(yaw_rad, device=img_tensor.device)), 0, cos(torch.tensor(yaw_rad, device=img_tensor.device))]
    ], dtype=torch.float32, device=img_tensor.device)

    R_z = torch.tensor([
        [cos(torch.tensor(roll_rad, device=img_tensor.device)), -sin(torch.tensor(roll_rad, device=img_tensor.device)), 0],
        [sin(torch.tensor(roll_rad, device=img_tensor.device)), cos(torch.tensor(roll_rad, device=img_tensor.device)), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=img_tensor.device)

    R = R_z @ R_y @ R_x

    # Translation from bbox center (normalized to [-1, 1])
    l, t, r, b = face[0:4]
    h_img, w_img = img_tensor.shape[2:]
    center_x_norm = 2 * ((l + r) / 2 / w_img) - 1
    center_y_norm = 2 * ((t + b) / 2 / h_img) - 1
    depth = -5.0  # Fixed depth; can be tuned based on face size

    t_vec = torch.tensor([center_x_norm, center_y_norm, depth], dtype=torch.float32, device=img_tensor.device).unsqueeze(0)

    # Construct 4x4 matrix
    theta = torch.eye(4, dtype=torch.float32, device=img_tensor.device).unsqueeze(0)
    theta[0, :3, :3] = R
    theta[0, :3, 3] = t_vec[0]

    logger.debug(f"Pose matrix shape: {theta.shape}")
    return theta


def extract_identity_features(model, source_img: torch.Tensor, face_detector, pose_estimator) -> Dict:
    """Extract identity features from source image to create canonical volume."""
    logger.info("Extracting identity features from source image...")

    with torch.no_grad():
        # Get face mask with higher quality
        face_mask, _, _, _ = model.face_idt.forward(source_img)
        face_mask = (face_mask > 0.6).float()
        # Smooth mask edges
        face_mask = F.avg_pool2d(face_mask, 3, stride=1, padding=1)
        logger.debug(f"Face mask shape: {face_mask.shape}")

        # Mask source image
        masked_source = source_img * face_mask

        # Extract identity embedding
        idt_embed = model.idt_embedder_nw(masked_source)
        logger.debug(f"Identity embedding shape: {idt_embed.shape}")

        # Get head pose using model's head_pose_regressor (matching pipeline)
        source_theta = model.head_pose_regressor.forward(source_img)

        # Prepare data dict
        data_dict = {
            'source_img': source_img,
            'source_mask': face_mask,
            'source_theta': source_theta,
            'target_img': source_img,  # Same as source for canonical
            'target_mask': face_mask,
            'target_theta': source_theta,
            'idt_embed': idt_embed
        }

        # Get expression embedding
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        logger.debug(f"Data dict keys after expression embedder: {list(data_dict.keys())}")

        # Get warp embeddings
        source_warp_embed, _, _, embed_dict = model.predict_embed(data_dict)
        logger.debug(f"Source warp embed keys: {list(source_warp_embed.keys())}")

        # Generate XY warps for source
        source_xy_warp, _ = model.xy_generator_nw(source_warp_embed)
        logger.debug(f"Source XY warp shape: {source_xy_warp.shape}")

        # Extract source volume
        source_latents = model.local_encoder_nw(masked_source)
        logger.debug(f"Source latents shape: {source_latents.shape}")

        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        source_volume = source_latents.view(1, c, d, s, s)
        logger.debug(f"Source volume shape: {source_volume.shape}")

        # Process source volume if needed
        if hasattr(model.args, 'source_volume_num_blocks') and model.args.source_volume_num_blocks > 0:
            source_volume = model.volume_source_nw(source_volume)
            logger.debug(f"Processed source volume shape: {source_volume.shape}")

        # Apply INVERSE source rotation and XY warp to get canonical volume (matching pipeline4.py)
        grid = model.identity_grid_3d.repeat_interleave(1, dim=0)
        inv_source_theta = source_theta.float().inverse().type(source_theta.type())
        source_rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

        # Apply warps in correct order: rotation first, then XY warp
        rotated_source = model.grid_sample(source_volume, source_rotation_warp)
        canonical_volume = model.grid_sample(rotated_source, source_xy_warp)
        logger.debug(f"Canonical volume shape: {canonical_volume.shape}")

        # Process canonical volume
        processed_canonical = model.volume_process_nw(canonical_volume, embed_dict)
        logger.debug(f"Processed canonical volume shape: {processed_canonical.shape}")

        return {
            'idt_embed': idt_embed,
            'embed_dict': embed_dict,
            'canonical_volume': processed_canonical,
            'source_theta': source_theta,
            'source_mask': face_mask
        }


def calculate_target_warps(model, identity_info: Dict, target_img: torch.Tensor,
                          cache_h5_path: Optional[str] = None, frame_idx: int = 0) -> Dict:
    """Calculate warps for target image and optionally cache them."""
    logger.info(f"Calculating warps for frame {frame_idx}...")

    with torch.no_grad():
        # Get target face mask
        target_mask, _, _, _ = model.face_idt.forward(target_img)
        target_mask = (target_mask > 0.6).float()
        target_mask = F.avg_pool2d(target_mask, 3, stride=1, padding=1)
        logger.debug(f"Target mask shape: {target_mask.shape}")

        # Get target pose using model's head_pose_regressor (matching pipeline)
        target_theta = model.head_pose_regressor.forward(target_img)

        # Create target data dict with source identity
        data_dict = {
            'source_img': target_img,
            'source_mask': target_mask,
            'source_theta': target_theta,
            'target_img': target_img,
            'target_mask': target_mask,
            'target_theta': target_theta,
            'idt_embed': identity_info['idt_embed']  # Source identity
        }

        # Get target expression
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        target_pose_embed = data_dict['source_pose_embed']
        logger.debug(f"Target pose embed shape: {target_pose_embed.shape}")

        # Get target warps
        _, target_warp_embed, _, _ = model.predict_embed(data_dict)
        logger.debug(f"Target warp embed keys: {list(target_warp_embed.keys())}")

        # Generate UV warps for target expression
        target_uv_warp, target_delta_uv = model.uv_generator_nw(target_warp_embed)
        logger.debug(f"Target UV warp shape: {target_uv_warp.shape}")

        # Match pipeline5.py logic - no resizing, just use the UV warp as-is
        target_uv_warp_resize = target_uv_warp
        logger.debug(f"Using UV warp without resizing, shape: {target_uv_warp_resize.shape}")

        # Cache warps to H5 if requested
        if cache_h5_path:
            logger.info(f"Caching warps for frame {frame_idx} to {cache_h5_path}")
            try:
                with h5py.File(cache_h5_path, 'a') as f:
                    if f'frame_{frame_idx}' in f:
                        del f[f'frame_{frame_idx}']  # Overwrite if exists
                    grp = f.create_group(f'frame_{frame_idx}')
                    # Save with compression to reduce file size
                    grp.create_dataset('uv_warp', data=target_uv_warp_resize.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
                    grp.create_dataset('theta', data=target_theta.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
                    grp.create_dataset('target_pose_embed', data=target_pose_embed.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
            except Exception as e:
                logger.error(f"Failed to cache warps: {str(e)}")
                raise

        # Return warp data
        return {
            'uv_warp': target_uv_warp_resize,
            'theta': target_theta,
            'target_pose_embed': target_pose_embed,
            'target_mask': target_mask
        }


def decode_with_warps(model, identity_info: Dict, warp_data: Dict,
                     target_img: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode using pre-calculated warps to generate the final image."""
    logger.info("Decoding with warps...")

    with torch.no_grad():
        # Extract warp data
        target_uv_warp_resize = warp_data['uv_warp']
        target_theta = warp_data['theta']
        target_pose_embed = warp_data['target_pose_embed']

        # Get volume dimensions (matching pipeline5.py)
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        # Generate 3D grid and rotation warp for target (matching pipeline5.py)
        grid = model.identity_grid_3d.repeat_interleave(1, dim=0)
        target_rotation_warp = grid.bmm(target_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

        # Apply warps exactly like pipeline5.py - nested grid_sample calls
        aligned_target_volume = model.grid_sample(
            model.grid_sample(identity_info['canonical_volume'], target_uv_warp_resize),
            target_rotation_warp
        )
        logger.debug(f"Aligned target volume shape: {aligned_target_volume.shape}")

        # Decode
        target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
        decode_dict = {
            'target_theta': target_theta,
            'target_pose_embed': target_pose_embed
        }

        generated_img, _, _, _ = model.decoder_nw(
            decode_dict,
            identity_info['embed_dict'],  # Source identity
            target_latent_feats,
            False,
            stage_two=True
        )
        logger.debug(f"Generated image shape: {generated_img.shape}")

        # Adjust range if needed (assume model outputs [0, 1])
        if generated_img.min() >= 0 and generated_img.max() <= 1.1:
            generated_img = generated_img * 2 - 1

        # Get refined mask for compositing
        gen_mask, _, _, _ = model.face_idt.forward(generated_img)
        gen_mask = (gen_mask > 0.65).float()
        for _ in range(3):  # Smooth mask edges
            gen_mask = F.avg_pool2d(gen_mask, 3, stride=1, padding=1)
        logger.debug(f"Generated mask shape: {gen_mask.shape}")

        # Composite with target background if target image provided
        if target_img is not None:
            background = target_img * (1 - gen_mask)
            final_img = generated_img * gen_mask + background
        else:
            final_img = generated_img

        final_img = torch.clamp(final_img, -1, 1)

        return final_img


def load_cached_warps(cache_h5_path: str, frame_idx: int, device: str = 'cuda') -> Optional[Dict]:
    """Load cached warps from H5 file."""
    try:
        with h5py.File(cache_h5_path, 'r') as f:
            frame_key = f'frame_{frame_idx}'
            if frame_key not in f:
                logger.warning(f"Frame {frame_idx} not found in cache")
                return None

            grp = f[frame_key]
            warp_data = {
                'uv_warp': torch.from_numpy(grp['uv_warp'][:]).to(device),
                'theta': torch.from_numpy(grp['theta'][:]).to(device),
                'target_pose_embed': torch.from_numpy(grp['target_pose_embed'][:]).to(device)
            }
            logger.info(f"Loaded cached warps for frame {frame_idx}")
            return warp_data
    except Exception as e:
        logger.error(f"Failed to load cached warps: {str(e)}")
        return None


def apply_target_to_identity(model, identity_info: Dict, target_img: torch.Tensor,
                            cache_h5_path: Optional[str] = None, frame_idx: int = 0,
                            use_cached: bool = True) -> Tuple[torch.Tensor, Dict]:
    """Apply target expression/pose to source identity using refactored workflow.

    This is now a convenience wrapper that combines warp calculation and decoding.
    Set use_cached=True to try loading cached warps first.
    """
    logger.info(f"Applying target to identity for frame {frame_idx}...")

    # Try to load cached warps first if requested
    warp_data = None
    if use_cached and cache_h5_path:
        warp_data = load_cached_warps(cache_h5_path, frame_idx, device=target_img.device)

    # Calculate warps if not cached
    if warp_data is None:
        warp_data = calculate_target_warps(model, identity_info, target_img,
                                          cache_h5_path, frame_idx)

    # Decode with warps
    final_img = decode_with_warps(model, identity_info, warp_data, target_img)

    return final_img, warp_data


def main(cache_h5_path: Optional[str] = None):
    """Create face swap results matching a video using IMG_1.png as identity."""
    logger.info("=" * 60)
    logger.info("Creating Video-Matching Face Swap")
    logger.info("Using IMG_1.png identity with proper pipeline")
    logger.info("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize face detector and pose estimator
    face_detector = RetinaFacePredictor(threshold=0.8, device=device,
                                       model=RetinaFacePredictor.get_model('mobilenet0.25'))
    pose_estimator = HeadPoseEstimator()

    # Load model
    model = load_volumetric_model(device)

    # Load source image (IMG_1.png)
    source_path = "nemo/data/IMG_1.png"
    try:
        source_img = load_image_tensor(source_path, device)
        logger.info(f"Loaded source image: {source_path}")
    except Exception as e:
        logger.error(f"Failed to load source image: {str(e)}")
        return

    # Extract identity features
    try:
        identity_info = extract_identity_features(model, source_img, face_detector, pose_estimator)
        logger.info("Successfully extracted identity features")
    except Exception as e:
        logger.error(f"Failed to extract identity features: {str(e)}")
        return

    # Load target frames
    targets: List[Tuple[str, torch.Tensor]] = []
    cache_path = Path("proper_face_attributes_img1.h5")
    if cache_path.exists():
        try:
            with h5py.File(cache_path, 'r') as f:
                # Load frames around the middle (best expressions)
                for i in [3, 4, 5, 6, 7]:
                    if f'frame_{i:04d}' in f:
                        frame_tensor = torch.from_numpy(f[f'frame_{i:04d}/frame'][:]).to(device)
                        targets.append((f"Frame_{i:04d}", frame_tensor))
                        logger.info(f"Loaded cached frame {i:04d}")
        except Exception as e:
            logger.warning(f"Failed to load cached frames: {str(e)}")

    # Load static images as fallback
    for img_name in ["IMG_2.png", "IMG_3.png", "IMG_4.png"]:
        img_path = Path(f"nemo/data/{img_name}")
        if img_path.exists():
            try:
                img_tensor = load_image_tensor(str(img_path), device)
                targets.append((img_name, img_tensor))
                logger.info(f"Loaded static image {img_name}")
            except Exception as e:
                logger.warning(f"Failed to load {img_name}: {str(e)}")

    if not targets:
        logger.error("No target images or frames found!")
        return

    # Process each target
    results = []
    for frame_idx, (name, target_img) in enumerate(targets):
        try:
            final_img, cache_data = apply_target_to_identity(
                model, identity_info, target_img,
                cache_h5_path=cache_h5_path, frame_idx=frame_idx
            )
            results.append({
                'name': name,
                'target': target_img,
                'result': final_img
            })
            logger.info(f"Processed {name} successfully")
        except Exception as e:
            logger.error(f"Failed to process {name}: {str(e)}")
            continue

    if not results:
        logger.error("No results generated!")
        return

    # Visualize results
    n_cols = min(len(results), 6)
    fig, axes = plt.subplots(3, n_cols + 1, figsize=(3 * (n_cols + 1), 9))

    # Show identity source
    source_display = (source_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
    axes[0, 0].imshow(np.clip(source_display, 0, 1))
    axes[0, 0].set_title("Identity\n(IMG_1)", fontsize=10, weight='bold', color='blue')
    axes[0, 0].axis('off')

    axes[1, 0].text(0.5, 0.5, 'Source\nIdentity', ha='center', va='center',
                    fontsize=11, weight='bold', color='blue')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    # Show results
    for i in range(min(n_cols, len(results))):
        col = i + 1
        result = results[i]

        # Target
        target_display = (result['target'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[0, col].imshow(np.clip(target_display, 0, 1))
        axes[0, col].set_title(f"{result['name']}", fontsize=9)
        axes[0, col].axis('off')

        # Result
        result_display = (result['result'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[1, col].imshow(np.clip(result_display, 0, 1))
        axes[1, col].set_title("Swapped", fontsize=9, weight='bold', color='green')
        axes[1, col].axis('off')

        # Save individual result
        try:
            result_img = (np.clip(result_display, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(result_img).save(f"video_swap_{result['name']}.png")
            logger.info(f"Saved individual result: video_swap_{result['name']}.png")
        except Exception as e:
            logger.error(f"Failed to save video_swap_{result['name']}.png: {str(e)}")

        # Show difference
        diff = np.abs(result_display - target_display).mean(axis=2)
        axes[2, col].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, col].set_title("Difference", fontsize=9)
        axes[2, col].axis('off')

    # Save visualization
    try:
        plt.suptitle("Face Swap Matching Video Result", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("video_matching_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved video_matching_results.png")
    except Exception as e:
        logger.error(f"Failed to save visualization: {str(e)}")

    # Save best single result (if available)
    if len(results) >= 3:
        try:
            best = results[2]  # Frame 5 usually has good expression
            best_img = (best['result'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            best_img = (np.clip(best_img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(best_img).save("video_best_result.png")
            logger.info("Saved video_best_result.png")
        except Exception as e:
            logger.error(f"Failed to save best result: {str(e)}")

    logger.info(f"Generated {len(results)} face swap results")
    logger.info("Results should match the video quality!")

    logger.info("\n" + "=" * 60)
    logger.info("Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Use cleaned cache file if it exists (11MB vs 330MB)
    import os
    cache_file = "proper_face_attributes_img1_cleaned.h5" if os.path.exists("proper_face_attributes_img1_cleaned.h5") else "proper_face_attributes_img1.h5"
    main(cache_h5_path=cache_file)