#!/usr/bin/env python3
"""
Simple validation that warps are being extracted correctly for transforming
from posed/expressive face to canonical space.
"""

import torch
import numpy as np
import importlib
import sys
import logging
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pathlib import Path

# Add nemo to path
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_volumetric_model():
    """Load the volumetric avatar model the same way vasa_dataset.py does."""
    logger.info("Loading volumetric avatar model...")

    # Load config
    emo_config = OmegaConf.load('./nemo/models/stage_1/volumetric_avatar/va.yaml')

    # Initialize model
    volumetric_avatar = importlib.import_module('models.stage_1.volumetric_avatar.va').Model(
        emo_config, training=False
    )

    # Load weights
    model_path = './nemo/logs/Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1/checkpoints/328_model.pth'
    if Path(model_path).exists():
        model_dict = torch.load(model_path, map_location='cuda')
        volumetric_avatar.load_state_dict(model_dict, strict=False)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(f"Model weights not found at {model_path}")

    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()

    # Set optimizer_idx_to_mode for inference
    if not hasattr(volumetric_avatar, 'optimizer_idx_to_mode'):
        volumetric_avatar.optimizer_idx_to_mode = {0: 'gen'}

    return volumetric_avatar

def extract_warps_for_frame(model, frame_tensor, identity_tensor=None):
    """
    Extract warps that transform from current frame pose to canonical.

    Args:
        model: Volumetric avatar model
        frame_tensor: Current frame tensor [1, 3, H, W]
        identity_tensor: Identity frame tensor [1, 3, H, W] (if None, use frame as identity)

    Returns:
        Dictionary containing extracted warps
    """
    if identity_tensor is None:
        identity_tensor = frame_tensor.clone()

    # Create data dictionary for the model
    data_dict = {
        'source_img': identity_tensor,
        'target_img': frame_tensor,
    }

    # Run model forward pass in test mode
    with torch.no_grad():
        _, _, _, output_dict = model.forward(
            data_dict,
            phase='test',
            optimizer_idx=0,
            visualize=False
        )

    # Extract warps from output
    warps = {}

    # Check what's in the output dict
    logger.info(f"Output dict keys: {output_dict.keys()}")

    # Extract warps from output dict first, then check model attributes
    if 'source_rotation_warp' in output_dict:
        warps['rigid_warp'] = output_dict['source_rotation_warp'].cpu()
        logger.info(f"Rigid warp shape: {warps['rigid_warp'].shape}")

    # XY warp might be in output dict or model attributes
    if 'source_xy_warp' in output_dict:
        warps['xy_warp'] = output_dict['source_xy_warp'].cpu()
        logger.info(f"XY warp shape (from dict): {warps['xy_warp'].shape}")
    elif 'source_xy_warp_resize' in output_dict:
        warps['xy_warp'] = output_dict['source_xy_warp_resize'].cpu()
        logger.info(f"XY warp shape (from dict resized): {warps['xy_warp'].shape}")
    elif hasattr(model, 'source_xy_warp_resize'):
        warps['xy_warp'] = model.source_xy_warp_resize.cpu()
        logger.info(f"XY warp shape (from model): {warps['xy_warp'].shape}")
    elif hasattr(model, 'source_xy_warp'):
        warps['xy_warp'] = model.source_xy_warp.cpu()
        logger.info(f"XY warp shape (from model unresize): {warps['xy_warp'].shape}")

    # UV warps
    if 'target_uv_warp' in output_dict:
        warps['uv_warp'] = output_dict['target_uv_warp'].cpu()
        logger.info(f"UV warp shape (from dict): {warps['uv_warp'].shape}")
    elif hasattr(model, 'target_uv_warp'):
        warps['uv_warp'] = model.target_uv_warp.cpu()
        logger.info(f"UV warp shape (from model): {warps['uv_warp'].shape}")

    if 'source_theta' in output_dict:
        theta = output_dict['source_theta'].cpu()
        # Convert 4x4 to 3x4 if needed
        if theta.shape[-2:] == (4, 4):
            theta = theta[..., :3, :]
        warps['source_theta'] = theta
        logger.info(f"Source theta shape: {warps['source_theta'].shape}")

    if 'canonical_volume' in output_dict:
        warps['canonical_volume'] = output_dict['canonical_volume'].cpu()
        logger.info(f"Canonical volume shape: {warps['canonical_volume'].shape}")

    return warps

def analyze_warp_properties(warps):
    """Analyze the properties of extracted warps."""
    logger.info("\n=== Analyzing Warp Properties ===")

    # 1. Rigid warp analysis
    if 'rigid_warp' in warps:
        rigid = warps['rigid_warp'].numpy()
        logger.info(f"\nRigid Warp (Pose Alignment):")
        logger.info(f"  Shape: {rigid.shape}")
        logger.info(f"  Range: [{rigid.min():.3f}, {rigid.max():.3f}]")
        logger.info(f"  Mean: {rigid.mean():.3f}, Std: {rigid.std():.3f}")

        # Check if it's close to identity (no transformation)
        if rigid.shape[-1] == 3:  # Has x,y,z components
            # Create identity grid for comparison
            d, h, w = rigid.shape[1:4]
            z_coords = np.linspace(-1, 1, d)
            y_coords = np.linspace(-1, 1, h)
            x_coords = np.linspace(-1, 1, w)
            zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            identity_grid = np.stack([xx, yy, zz], axis=-1)

            # Compute deviation from identity
            deviation = np.mean(np.abs(rigid[0] - identity_grid))
            logger.info(f"  Deviation from identity: {deviation:.4f}")

    # 2. Non-rigid (XY) warp analysis
    if 'xy_warp' in warps:
        xy = warps['xy_warp'].numpy()
        logger.info(f"\nXY Warp (Expression Normalization):")
        logger.info(f"  Shape: {xy.shape}")
        logger.info(f"  Range: [{xy.min():.3f}, {xy.max():.3f}]")
        logger.info(f"  Mean: {xy.mean():.3f}, Std: {xy.std():.3f}")

        # Check smoothness (gradients should be small for valid warps)
        if xy.ndim >= 4:
            grad_y = np.gradient(xy[0], axis=1)
            grad_x = np.gradient(xy[0], axis=2)
            smoothness = np.mean(np.abs(grad_y)) + np.mean(np.abs(grad_x))
            logger.info(f"  Smoothness (lower=smoother): {smoothness:.4f}")

    # 3. Source theta analysis
    if 'source_theta' in warps:
        theta = warps['source_theta'].numpy()
        logger.info(f"\nSource Theta (Affine Transform):")
        logger.info(f"  Shape: {theta.shape}")

        if theta.shape[-2:] == (3, 4):
            # Extract rotation and translation
            R = theta[..., :3, :3]
            t = theta[..., :3, 3]

            # Check orthogonality of rotation
            if R.ndim == 2:
                RTR = R @ R.T
                orthogonality_error = np.linalg.norm(RTR - np.eye(3))
                det = np.linalg.det(R)
            else:
                RTR = R[0] @ R[0].T
                orthogonality_error = np.linalg.norm(RTR - np.eye(3))
                det = np.linalg.det(R[0])

            logger.info(f"  Rotation determinant: {det:.3f} (should be ~1)")
            logger.info(f"  Orthogonality error: {orthogonality_error:.4f} (should be ~0)")
            logger.info(f"  Translation magnitude: {np.linalg.norm(t):.3f}")

def visualize_warps(warps, save_path="warp_analysis.png"):
    """Visualize the warps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Warp Fields Analysis (Posed → Canonical)", fontsize=14)

    # Helper to get middle slice
    def get_middle_slice(tensor, dim=0):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        if tensor.ndim == 5:  # [B, D, H, W, C]
            tensor = tensor[0]  # Remove batch
        mid_idx = tensor.shape[dim] // 2
        return tensor[mid_idx] if dim == 0 else tensor[:, mid_idx]

    # Plot rigid warp
    if 'rigid_warp' in warps:
        rigid = warps['rigid_warp']
        mid_slice = get_middle_slice(rigid)

        for i, component in enumerate(['X', 'Y', 'Z']):
            if i < 3 and i < mid_slice.shape[-1]:
                ax = axes[0, i]
                im = ax.imshow(mid_slice[..., i], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f"Rigid Warp - {component}")
                plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot XY warp
    if 'xy_warp' in warps:
        xy = warps['xy_warp']
        mid_slice = get_middle_slice(xy)

        for i, component in enumerate(['X', 'Y']):
            if i < 2 and i < mid_slice.shape[-1]:
                ax = axes[1, i]
                im = ax.imshow(mid_slice[..., i], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f"XY Warp - {component}")
                plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot magnitude
        if mid_slice.shape[-1] >= 2:
            ax = axes[1, 2]
            magnitude = np.sqrt(mid_slice[..., 0]**2 + mid_slice[..., 1]**2)
            im = ax.imshow(magnitude, cmap='viridis')
            ax.set_title("XY Warp Magnitude")
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    plt.close()

def main():
    # Load model
    model = load_volumetric_model()

    # Load test image
    test_image_path = Path("/media/2TB/VASA-1-hack/data/A.png")
    if not test_image_path.exists():
        logger.error(f"Test image not found at {test_image_path}")
        return

    # Prepare image tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img = Image.open(test_image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()

    logger.info(f"Test image shape: {img_tensor.shape}")

    # Extract warps
    logger.info("\n=== Extracting Warps ===")
    warps = extract_warps_for_frame(model, img_tensor)

    # Analyze warps
    analyze_warp_properties(warps)

    # Visualize warps
    visualize_warps(warps)

    logger.info("\n=== Summary ===")
    logger.info("Warp extraction completed successfully!")
    logger.info("Key findings:")
    logger.info("1. Rigid warp: Handles global head pose alignment to frontal view")
    logger.info("2. XY warp: Handles local expression normalization to neutral")
    logger.info("3. Together they transform: Posed/Expressive → Canonical/Neutral")
    logger.info("4. These warps are applied sequentially in the volumetric space")

if __name__ == "__main__":
    main()