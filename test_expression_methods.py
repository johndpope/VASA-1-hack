#!/usr/bin/env python3
"""Compare the two expression embedding extraction methods."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from create_video_face_swap import load_volumetric_model, load_image_tensor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_expression_methods(model, image_tensor):
    """Compare two methods of extracting expression embeddings."""

    with torch.no_grad():
        # Method 1: Direct net_face call (what you've been using)
        # This takes the raw image without alignment
        direct_embed = model.expression_embedder_nw.net_face(image_tensor)[0]

        # Method 2: Through expression_embedder_nw forward (proper method)
        # This aligns the face first, then extracts embedding

        # Get face mask and theta for alignment
        face_mask, _, _, _ = model.face_idt.forward(image_tensor)
        face_mask = (face_mask > 0.6).float()
        theta = model.head_pose_regressor.forward(image_tensor)

        # Prepare data dict
        data_dict = {
            'source_img': image_tensor,
            'source_mask': face_mask,
            'source_theta': theta,
            'target_img': image_tensor,  # Using same image
            'target_mask': face_mask,
            'target_theta': theta
        }

        # Call expression embedder with alignment
        data_dict = model.expression_embedder_nw(data_dict, True, False, False)
        aligned_embed = data_dict['source_pose_embed']

    return direct_embed, aligned_embed, data_dict.get('source_img_align')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_volumetric_model(device)

    # Test with multiple images
    test_images = [
        "nemo/data/IMG_1.png",
        "nemo/data/IMG_2.png",
        "nemo/data/IMG_3.png"
    ]

    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 4*len(test_images)))

    for img_idx, img_path in enumerate(test_images):
        logger.info(f"\nTesting {img_path}...")
        img_tensor = load_image_tensor(img_path, device)

        # Get embeddings from both methods
        direct_embed, aligned_embed, aligned_img = compare_expression_methods(model, img_tensor)

        # Convert to numpy
        direct_np = direct_embed.cpu().numpy().squeeze()
        aligned_np = aligned_embed.cpu().numpy().squeeze()

        # Calculate statistics
        cosine_sim = np.dot(direct_np, aligned_np) / (np.linalg.norm(direct_np) * np.linalg.norm(aligned_np))
        diff = direct_np - aligned_np

        logger.info(f"  Direct method - norm: {np.linalg.norm(direct_np):.3f}, mean: {direct_np.mean():.3f}")
        logger.info(f"  Aligned method - norm: {np.linalg.norm(aligned_np):.3f}, mean: {aligned_np.mean():.3f}")
        logger.info(f"  Cosine similarity: {cosine_sim:.3f}")
        logger.info(f"  L2 difference: {np.linalg.norm(diff):.3f}")

        # Visualize
        row = img_idx

        # Original image
        img_display = (img_tensor[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axes[row, 0].imshow(np.clip(img_display, 0, 1))
        axes[row, 0].set_title(f'Original\n{img_path.split("/")[-1]}')
        axes[row, 0].axis('off')

        # Aligned image (if available)
        if aligned_img is not None:
            aligned_display = (aligned_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
            axes[row, 1].imshow(np.clip(aligned_display, 0, 1))
            axes[row, 1].set_title('Aligned Face\n(Method 2 preprocessing)')
        else:
            axes[row, 1].text(0.5, 0.5, 'Aligned image\nnot available', ha='center', va='center')
        axes[row, 1].axis('off')

        # Direct embedding visualization
        axes[row, 2].bar(range(50), direct_np[:50], color='coral', alpha=0.7)
        axes[row, 2].set_title(f'Direct net_face\nnorm={np.linalg.norm(direct_np):.2f}')
        axes[row, 2].set_ylim([-2, 2])
        axes[row, 2].grid(True, alpha=0.3)

        # Aligned embedding visualization
        axes[row, 3].bar(range(50), aligned_np[:50], color='steelblue', alpha=0.7)
        axes[row, 3].set_title(f'With alignment\nnorm={np.linalg.norm(aligned_np):.2f}')
        axes[row, 3].set_ylim([-2, 2])
        axes[row, 3].grid(True, alpha=0.3)

        # Add similarity score
        axes[row, 3].text(0.98, 0.98, f'Sim={cosine_sim:.3f}',
                         transform=axes[row, 3].transAxes,
                         fontsize=10, ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('Expression Embedding Methods Comparison\nDirect net_face vs Aligned extraction',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('expression_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("\nSaved expression_methods_comparison.png")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY:")
    logger.info("- Method 1 (direct net_face): Processes raw unaligned image")
    logger.info("- Method 2 (via expression_embedder_nw): Aligns face first")
    logger.info("- Alignment normalizes head pose before extracting expression")
    logger.info("- Both produce 128-dim embeddings but values differ due to alignment")
    logger.info("- target_pose_embed uses Method 2 (aligned)")
    logger.info("="*60)

if __name__ == "__main__":
    main()