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

        # Get expression embedding (with face alignment for pose normalization)
        # This properly separates head pose from facial expression
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        source_pose_embed = data_dict['source_pose_embed']  # Aligned expression embedding
        logger.debug(f"Data dict keys after expression embedder: {list(data_dict.keys())}")
        logger.debug(f"Source expression embedding shape: {source_pose_embed.shape}")

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
            'source_pose_embed': source_pose_embed,  # Aligned expression embedding
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
        # Also extract scale, rotation, translation components
        target_theta, scale, rotation, translation = model.head_pose_regressor.forward(
            target_img, return_srt=True
        )
        logger.debug(f"Target SRT - Scale: {scale.shape}, Rotation: {rotation.shape}, Translation: {translation.shape}")

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

        # Get target expression (with face alignment for pose normalization)
        # This ensures we extract pure expression, not mixed with head pose
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        target_pose_embed = data_dict['source_pose_embed']  # Note: 'source_pose_embed' because target is in source position
        logger.debug(f"Target pose embed shape: {target_pose_embed.shape}")
        logger.debug(f"Target pose embed norm: {torch.norm(target_pose_embed).item():.3f}")

        # EXPERIMENT: Also extract unaligned expression embedding
        # This is what happens when we directly feed the raw image without pose normalization
        target_pose_embed_unaligned = model.expression_embedder_nw.net_face(target_img)[0]
        logger.debug(f"Unaligned pose embed shape: {target_pose_embed_unaligned.shape}")
        logger.debug(f"Unaligned pose embed norm: {torch.norm(target_pose_embed_unaligned).item():.3f}")
        logger.debug(f"Aligned vs Unaligned cosine similarity: {torch.cosine_similarity(target_pose_embed, target_pose_embed_unaligned, dim=1).item():.3f}")

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
                    # Cache unaligned embedding for experiments
                    grp.create_dataset('target_pose_embed_unaligned', data=target_pose_embed_unaligned.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
                    # Cache SRT components
                    grp.create_dataset('scale', data=scale.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
                    grp.create_dataset('rotation', data=rotation.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
                    grp.create_dataset('translation', data=translation.cpu().numpy(),
                                      compression='gzip', compression_opts=4)
            except Exception as e:
                logger.error(f"Failed to cache warps: {str(e)}")
                raise

        # Return warp data
        return {
            'uv_warp': target_uv_warp_resize,
            'theta': target_theta,
            'scale': scale,
            'rotation': rotation,
            'translation': translation,
            'target_pose_embed': target_pose_embed,
            'target_pose_embed_unaligned': target_pose_embed_unaligned,  # EXPERIMENT: unaligned embedding
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
            # Load unaligned embedding if available (for experiments)
            if 'target_pose_embed_unaligned' in grp:
                warp_data['target_pose_embed_unaligned'] = torch.from_numpy(grp['target_pose_embed_unaligned'][:]).to(device)

            # Load SRT components if available
            if 'scale' in grp:
                warp_data['scale'] = torch.from_numpy(grp['scale'][:]).to(device)
                warp_data['rotation'] = torch.from_numpy(grp['rotation'][:]).to(device)
                warp_data['translation'] = torch.from_numpy(grp['translation'][:]).to(device)
                logger.info(f"Loaded cached warps with SRT components for frame {frame_idx}")
            else:
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


def visualize_srt_components(warp_data_list: List[Dict], names: List[str], output_path: str = "srt_components_analysis.png"):
    """Visualize Scale, Rotation, Translation components across frames."""
    if not warp_data_list or not all('scale' in w for w in warp_data_list):
        logger.warning("No SRT data available for visualization")
        return

    n_frames = len(warp_data_list)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Extract SRT values
    scales = [w['scale'].cpu().numpy().flatten() for w in warp_data_list]
    rotations = [w['rotation'].cpu().numpy().flatten()[:3] for w in warp_data_list]  # First 3 rotation params
    translations = [w['translation'].cpu().numpy().flatten() for w in warp_data_list]

    # Plot Scale
    scale_data = np.array(scales)
    axes[0].plot(scale_data, marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Scale Components', fontsize=14, weight='bold')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Scale')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Scale X', 'Scale Y', 'Scale Z'][:scale_data.shape[1]], loc='best')

    # Plot Rotation
    rotation_data = np.array(rotations)
    axes[1].plot(rotation_data, marker='s', linewidth=2, markersize=8)
    axes[1].set_title('Rotation Components (First 3 params)', fontsize=14, weight='bold')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Rotation')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Rot 1', 'Rot 2', 'Rot 3'], loc='best')

    # Plot Translation
    translation_data = np.array(translations)
    axes[2].plot(translation_data, marker='^', linewidth=2, markersize=8)
    axes[2].set_title('Translation Components', fontsize=14, weight='bold')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Translation')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(['Trans X', 'Trans Y', 'Trans Z'][:translation_data.shape[1]], loc='best')

    # Set x-axis labels
    for ax in axes:
        ax.set_xticks(range(n_frames))
        ax.set_xticklabels([n.replace('.png', '').replace('Frame_', 'F') for n in names], rotation=45)

    plt.suptitle('SRT (Scale, Rotation, Translation) Components Analysis', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")

    # Log statistics
    logger.info("\n=== SRT Component Statistics ===")
    for idx, name in enumerate(names):
        logger.info(f"{name}:")
        logger.info(f"  Scale: {scales[idx]}")
        logger.info(f"  Rotation (first 3): {rotations[idx]}")
        logger.info(f"  Translation: {translations[idx]}")


def experiment_scale_modification(model, identity_info: Dict, warp_data: Dict,
                                 scale_factor: float = 1.5, scale_axis: str = 'all') -> torch.Tensor:
    """Experiment: Generate frame with modified scale values.

    Args:
        model: The volumetric avatar model
        identity_info: Source identity information
        warp_data: Target warp data including scale, rotation, translation
        scale_factor: Factor to multiply scale by (e.g., 1.5 for 50% bigger)
        scale_axis: Which axis to scale - 'x', 'y', 'z', or 'all'

    Returns:
        Generated image with modified scale
    """
    logger.info(f"SCALE EXPERIMENT: Modifying scale by {scale_factor}x on axis: {scale_axis}")

    with torch.no_grad():
        # Get original scale values
        original_scale = warp_data['scale'].clone()
        modified_scale = original_scale.clone()

        # Modify scale based on axis
        if scale_axis == 'all':
            modified_scale = modified_scale * scale_factor
        elif scale_axis == 'x':
            modified_scale[:, 0] *= scale_factor
        elif scale_axis == 'y':
            modified_scale[:, 1] *= scale_factor
        elif scale_axis == 'z':
            modified_scale[:, 2] *= scale_factor

        logger.info(f"Original scale: {original_scale.cpu().numpy().flatten()}")
        logger.info(f"Modified scale: {modified_scale.cpu().numpy().flatten()}")

        # Reconstruct theta matrix with modified scale
        # The theta matrix combines rotation, scale, and translation
        # We need to reconstruct it with the modified scale
        rotation = warp_data['rotation']
        translation = warp_data['translation']

        # Build new transformation matrix
        # Note: This is a simplified reconstruction - the exact format depends on how
        # the head_pose_regressor builds the theta matrix
        device = modified_scale.device
        batch_size = modified_scale.shape[0]

        # Create identity matrix
        theta_modified = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Apply scale (diagonal elements)
        theta_modified[:, 0, 0] = modified_scale[:, 0]
        theta_modified[:, 1, 1] = modified_scale[:, 1]
        theta_modified[:, 2, 2] = modified_scale[:, 2]

        # Apply rotation (simplified - may need adjustment based on actual rotation format)
        # This assumes rotation contains rotation angles or quaternion parameters
        # You may need to convert rotation to a rotation matrix first

        # Apply translation
        theta_modified[:, :3, 3] = translation

        # Create modified decode dict
        decode_dict = {
            'target_theta': theta_modified,
            'target_pose_embed': warp_data['target_pose_embed']
        }

        # Get volume dimensions
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        # Generate 3D grid and rotation warp with modified theta
        grid = model.identity_grid_3d.repeat_interleave(1, dim=0)
        target_rotation_warp = grid.bmm(theta_modified[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

        # Apply warps with modified scale
        aligned_target_volume = model.grid_sample(
            model.grid_sample(identity_info['canonical_volume'], warp_data['uv_warp']),
            target_rotation_warp
        )

        # Decode
        target_latent_feats = aligned_target_volume.view(1, c * d, s, s)

        generated_img, _, _, _ = model.decoder_nw(
            decode_dict,
            identity_info['embed_dict'],
            target_latent_feats,
            False,
            stage_two=True
        )

        # Adjust range
        if generated_img.min() >= 0 and generated_img.max() <= 1.1:
            generated_img = generated_img * 2 - 1

        # Apply face mask if available
        if 'target_mask' in warp_data:
            target_mask = warp_data['target_mask']
            target_mask_small = F.interpolate(target_mask, size=256)
            generated_img = generated_img * target_mask_small - (1 - target_mask_small)

        generated_img = torch.clamp(generated_img, -1, 1)

        return generated_img


def experiment_unaligned_reconstruction(model, identity_info: Dict, warp_data: Dict) -> torch.Tensor:
    """Experiment: Reconstruct face using UNALIGNED expression embedding.

    This tests what happens when we use the raw, non-pose-normalized expression.
    Expected: The reconstruction will mix expression with head pose, causing artifacts.
    """
    logger.info("EXPERIMENT: Reconstructing with unaligned expression embedding...")

    with torch.no_grad():
        # Use the same warps but swap the expression embedding
        decode_dict = {
            'target_theta': warp_data['theta'],
            'target_pose_embed': warp_data['target_pose_embed_unaligned']  # Use UNALIGNED embedding
        }

        # Get volume dimensions
        c = model.args.latent_volume_channels
        d = model.args.latent_volume_depth
        s = model.args.latent_volume_size

        # Generate 3D grid and rotation warp for target
        grid = model.identity_grid_3d.repeat_interleave(1, dim=0)
        target_rotation_warp = grid.bmm(warp_data['theta'][:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

        # Apply warps to canonical volume
        aligned_target_volume = model.grid_sample(
            model.grid_sample(identity_info['canonical_volume'], warp_data['uv_warp']),
            target_rotation_warp
        )

        # Decode
        target_latent_feats = aligned_target_volume.view(1, c * d, s, s)
        generated_img, gen_mask, _, _ = model.decoder_nw(
            decode_dict,
            identity_info['embed_dict'],
            target_latent_feats,
            False,  # with_bones
            0  # iteration
        )

        generated_img = torch.clamp(generated_img, -1, 1)

        logger.info(f"Unaligned reconstruction complete, shape: {generated_img.shape}")

    return generated_img


def generate_custom_scale_frame(scale_x: float = 1.0, scale_y: float = 1.0, scale_z: float = 1.0,
                               frame_idx: int = 0, cache_h5_path: str = "proper_face_attributes_img1.h5"):
    """Generate a single frame with custom scale values.

    Args:
        scale_x: Scale factor for X axis (width)
        scale_y: Scale factor for Y axis (height)
        scale_z: Scale factor for Z axis (depth)
        frame_idx: Which frame to use as base (default: 0)
        cache_h5_path: Path to cache file with warps

    Returns:
        Generated image tensor
    """
    logger.info(f"Generating custom scaled frame: X={scale_x}, Y={scale_y}, Z={scale_z}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load model using existing function
    model = load_volumetric_model(device)

    # Initialize face detector and pose estimator for identity extraction
    face_detector = RetinaFacePredictor(threshold=0.8, device=device,
                                       model=RetinaFacePredictor.get_model('mobilenet0.25'))
    pose_estimator = HeadPoseEstimator()

    # Load source identity
    source_img = load_image_tensor("nemo/data/IMG_1.png", device)
    identity_info = extract_identity_features(model, source_img, face_detector, pose_estimator)

    # Load cached warps for the frame
    warp_data = load_cached_warps(cache_h5_path, frame_idx, device)
    if warp_data is None:
        logger.error(f"No cached warps found for frame {frame_idx}")
        return None

    # Apply custom scale
    if 'scale' in warp_data:
        original_scale = warp_data['scale'].clone()
        warp_data['scale'][:, 0] = original_scale[:, 0] * scale_x
        warp_data['scale'][:, 1] = original_scale[:, 1] * scale_y
        warp_data['scale'][:, 2] = original_scale[:, 2] * scale_z

        logger.info(f"Original scale: {original_scale.cpu().numpy().flatten()}")
        logger.info(f"Modified scale: {warp_data['scale'].cpu().numpy().flatten()}")

        # Rebuild theta matrix with custom scale
        device = warp_data['scale'].device
        batch_size = warp_data['scale'].shape[0]

        theta_modified = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        theta_modified[:, 0, 0] = warp_data['scale'][:, 0]
        theta_modified[:, 1, 1] = warp_data['scale'][:, 1]
        theta_modified[:, 2, 2] = warp_data['scale'][:, 2]
        theta_modified[:, :3, 3] = warp_data['translation']

        warp_data['theta'] = theta_modified

    # Generate image
    generated_img = decode_with_warps(model, identity_info, warp_data)

    # Save result
    img_np = (generated_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    filename = f"custom_scale_x{scale_x}_y{scale_y}_z{scale_z}.png"
    Image.fromarray(img_np).save(filename)
    logger.info(f"Saved {filename}")

    return generated_img


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
    expression_embeddings = []  # Collect expression embeddings for visualization
    unaligned_experiments = []  # EXPERIMENT: collect unaligned reconstructions
    srt_data = []  # Collect SRT components for visualization

    for frame_idx, (name, target_img) in enumerate(targets):
        try:
            final_img, cache_data = apply_target_to_identity(
                model, identity_info, target_img,
                cache_h5_path=cache_h5_path, frame_idx=frame_idx
            )
            results.append({
                'name': name,
                'target': target_img,
                'result': final_img,
                'expression_embed': cache_data.get('target_pose_embed')  # Store expression embedding
            })

            # Collect expression embedding for analysis
            if 'target_pose_embed' in cache_data:
                expression_embeddings.append({
                    'frame': name,
                    'embed': cache_data['target_pose_embed']
                })

            # Collect SRT data for visualization
            if all(k in cache_data for k in ['scale', 'rotation', 'translation']):
                srt_data.append(cache_data)

            # EXPERIMENT: Generate unaligned reconstruction for comparison
            if 'target_pose_embed_unaligned' in cache_data:
                try:
                    unaligned_img = experiment_unaligned_reconstruction(model, identity_info, cache_data)
                    unaligned_experiments.append({
                        'name': name,
                        'aligned': final_img,
                        'unaligned': unaligned_img,
                        'target': target_img
                    })
                except Exception as e:
                    logger.warning(f"Unaligned experiment failed for {name}: {str(e)}")

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

    # Create expression embedding comparison plots
    if expression_embeddings:
        try:
            # Create figure for expression embedding analysis
            n_frames = len(expression_embeddings)
            fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 8))

            # If only one frame, make axes 2D
            if n_frames == 1:
                axes = axes.reshape(2, 1)

            for idx, expr_data in enumerate(expression_embeddings):
                frame_name = expr_data['frame']
                embed = expr_data['embed']

                # Convert to numpy if tensor
                if isinstance(embed, torch.Tensor):
                    embed = embed.detach().cpu().numpy()

                # Ensure correct shape
                if len(embed.shape) > 1:
                    embed = embed.squeeze()  # Remove batch dimension if present

                # Plot first 50 dimensions as bars
                dims_to_plot = min(50, len(embed))

                # Top row: Bar plot of expression embedding
                axes[0, idx].bar(range(dims_to_plot), embed[:dims_to_plot], color='steelblue')
                axes[0, idx].set_title(f'{frame_name}\nExpression Embedding', fontsize=10)
                axes[0, idx].set_xlabel('Dimension')
                axes[0, idx].set_ylabel('Value')
                axes[0, idx].set_ylim([-2, 2])  # Standard range for embeddings
                axes[0, idx].grid(True, alpha=0.3)

                # Bottom row: Heatmap visualization of full embedding
                # Reshape to 2D for visualization (e.g., 8x16 for 128 dims)
                embed_2d = embed.reshape(8, 16)
                im = axes[1, idx].imshow(embed_2d, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
                axes[1, idx].set_title(f'Full 128-dim (8x16)', fontsize=10)
                axes[1, idx].set_xlabel('Dimension')
                axes[1, idx].set_ylabel('Group')

                # Add statistics
                norm = np.linalg.norm(embed)
                mean = np.mean(embed)
                std = np.std(embed)
                axes[1, idx].text(0.02, 0.98, f'Norm: {norm:.2f}\nMean: {mean:.3f}\nStd: {std:.3f}',
                                transform=axes[1, idx].transAxes,
                                fontsize=8, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add colorbar for heatmaps
            plt.colorbar(im, ax=axes[1, :], orientation='horizontal', pad=0.1, fraction=0.05)

            plt.suptitle('Expression Embeddings Analysis (128 dims)', fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig("expression_embeddings_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Saved expression_embeddings_analysis.png")

            # Log embedding statistics
            logger.info("\n=== Expression Embedding Statistics ===")
            for expr_data in expression_embeddings:
                embed = expr_data['embed']
                if isinstance(embed, torch.Tensor):
                    embed = embed.detach().cpu().numpy()
                if len(embed.shape) > 1:
                    embed = embed.squeeze()

                norm = np.linalg.norm(embed)
                mean = np.mean(embed)
                std = np.std(embed)
                min_val = np.min(embed)
                max_val = np.max(embed)

                logger.info(f"{expr_data['frame']}: norm={norm:.2f}, mean={mean:.3f}, std={std:.3f}, range=[{min_val:.2f}, {max_val:.2f}]")

        except Exception as e:
            logger.error(f"Failed to create expression embedding visualization: {str(e)}")

        # Create frame-to-frame comparison
        if len(expression_embeddings) > 1:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Compute similarity matrix
                n = len(expression_embeddings)
                similarity_matrix = np.zeros((n, n))
                embeddings_array = []

                for i, expr_data in enumerate(expression_embeddings):
                    embed = expr_data['embed']
                    if isinstance(embed, torch.Tensor):
                        embed = embed.detach().cpu().numpy()
                    if len(embed.shape) > 1:
                        embed = embed.squeeze()
                    embeddings_array.append(embed)

                # Calculate cosine similarity between all pairs
                for i in range(n):
                    for j in range(n):
                        norm_i = np.linalg.norm(embeddings_array[i])
                        norm_j = np.linalg.norm(embeddings_array[j])
                        if norm_i > 0 and norm_j > 0:
                            similarity_matrix[i, j] = np.dot(embeddings_array[i], embeddings_array[j]) / (norm_i * norm_j)
                        else:
                            similarity_matrix[i, j] = 0

                # Plot similarity matrix
                im1 = axes[0].imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[0].set_title('Cosine Similarity Matrix', fontsize=12, weight='bold')
                axes[0].set_xlabel('Frame')
                axes[0].set_ylabel('Frame')

                # Add frame labels
                frame_labels = [e['frame'] for e in expression_embeddings]
                axes[0].set_xticks(range(n))
                axes[0].set_yticks(range(n))
                axes[0].set_xticklabels(frame_labels, rotation=45, ha='right')
                axes[0].set_yticklabels(frame_labels)

                # Add values to heatmap
                for i in range(n):
                    for j in range(n):
                        text = axes[0].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white",
                                          fontsize=8)

                plt.colorbar(im1, ax=axes[0])

                # Plot embedding trajectory (PCA projection)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(np.array(embeddings_array))

                axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(n), cmap='viridis', s=100)
                for i, txt in enumerate(frame_labels):
                    axes[1].annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=9)

                axes[1].plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k--', alpha=0.3)
                axes[1].set_title('Expression Embedding Trajectory (PCA)', fontsize=12, weight='bold')
                axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
                axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
                axes[1].grid(True, alpha=0.3)

                plt.suptitle('Expression Embedding Frame Comparison', fontsize=14, weight='bold')
                plt.tight_layout()
                plt.savefig("expression_embeddings_comparison.png", dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("Saved expression_embeddings_comparison.png")

            except Exception as e:
                logger.error(f"Failed to create expression comparison: {str(e)}")

    # Visualize SRT components if available
    if srt_data:
        try:
            names = [r['name'] for r in results[:len(srt_data)]]
            visualize_srt_components(srt_data, names)
        except Exception as e:
            logger.error(f"Failed to create SRT visualization: {str(e)}")

    # Run scale modification experiments
    scale_experiments = []
    if len(results) > 0 and srt_data:
        try:
            # Use first frame for experiments
            first_result = results[0]
            first_srt = srt_data[0] if srt_data else None

            if first_srt and all(k in first_srt for k in ['scale', 'rotation', 'translation']):
                logger.info("\n=== SCALE MODIFICATION EXPERIMENTS ===")

                # Test different scale factors
                scale_configs = [
                    (1.0, 'all', 'Original'),
                    (0.7, 'all', 'Scaled 0.7x (smaller)'),
                    (1.3, 'all', 'Scaled 1.3x (larger)'),
                    (1.5, 'x', 'X-axis 1.5x'),
                    (1.5, 'y', 'Y-axis 1.5x'),
                    (1.5, 'z', 'Z-axis 1.5x'),
                ]

                for scale_factor, axis, label in scale_configs:
                    if scale_factor == 1.0:
                        # Use original
                        img = first_result['result']
                    else:
                        img = experiment_scale_modification(
                            model, identity_info, first_srt,
                            scale_factor=scale_factor, scale_axis=axis
                        )

                    scale_experiments.append({
                        'label': label,
                        'image': img,
                        'scale_factor': scale_factor,
                        'axis': axis
                    })

                # Create visualization
                n_exp = len(scale_experiments)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for idx, exp in enumerate(scale_experiments[:6]):
                    img = exp['image'][0].cpu().permute(1, 2, 0).numpy()
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)

                    axes[idx].imshow(img)
                    axes[idx].set_title(exp['label'], fontsize=12, weight='bold')
                    axes[idx].axis('off')

                plt.suptitle('Scale Modification Experiments', fontsize=16, weight='bold')
                plt.tight_layout()
                plt.savefig('scale_modification_experiments.png', dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("Saved scale_modification_experiments.png")

        except Exception as e:
            logger.error(f"Failed to run scale experiments: {str(e)}")

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

    # EXPERIMENT: Visualize aligned vs unaligned reconstructions
    if unaligned_experiments:
        try:
            n_exp = min(len(unaligned_experiments), 4)
            fig, axes = plt.subplots(4, n_exp, figsize=(4 * n_exp, 16))

            # If only one experiment, make axes 2D
            if n_exp == 1:
                axes = axes.reshape(4, 1)

            for idx in range(n_exp):
                exp = unaligned_experiments[idx]

                # Row 0: Target image
                target_display = (exp['target'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                axes[0, idx].imshow(np.clip(target_display, 0, 1))
                axes[0, idx].set_title(f'Target\n{exp["name"]}', fontsize=10)
                axes[0, idx].axis('off')

                # Row 1: Aligned reconstruction (correct)
                aligned_display = (exp['aligned'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                axes[1, idx].imshow(np.clip(aligned_display, 0, 1))
                axes[1, idx].set_title('Aligned Embedding\n(Correct)', fontsize=10, color='green')
                axes[1, idx].axis('off')

                # Row 2: Unaligned reconstruction (experiment)
                unaligned_display = (exp['unaligned'][0].cpu().permute(1, 2, 0).numpy() + 1) / 2
                axes[2, idx].imshow(np.clip(unaligned_display, 0, 1))
                axes[2, idx].set_title('Unaligned Embedding\n(Experiment)', fontsize=10, color='red')
                axes[2, idx].axis('off')

                # Row 3: Difference between aligned and unaligned
                diff = np.abs(aligned_display - unaligned_display).mean(axis=2)
                im = axes[3, idx].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                axes[3, idx].set_title('Difference\n(Aligned vs Unaligned)', fontsize=10)
                axes[3, idx].axis('off')

                # Add metrics
                mse = np.mean((aligned_display - unaligned_display) ** 2)
                axes[3, idx].text(0.5, -0.05, f'MSE: {mse:.4f}',
                                transform=axes[3, idx].transAxes,
                                ha='center', fontsize=9)

            plt.suptitle('EXPERIMENT: Aligned vs Unaligned Expression Embedding Reconstruction',
                        fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig('unaligned_embedding_experiment.png', dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Saved unaligned_embedding_experiment.png")

            logger.info("\n=== EXPERIMENT RESULTS ===")
            logger.info("Unaligned embeddings mix expression with head pose!")
            logger.info("This causes artifacts and incorrect expression transfer.")
            logger.info("Aligned embeddings properly separate pose from expression.")
            logger.info("=" * 30)

        except Exception as e:
            logger.error(f"Failed to create unaligned experiment visualization: {str(e)}")

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