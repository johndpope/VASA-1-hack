#!/usr/bin/env python3
"""
Use the exact pipeline2.py InferenceWrapper to match the video result.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import logging
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline2 InferenceWrapper
from pipeline2 import InferenceWrapper


def create_exact_face_swap():
    """Use the exact pipeline2.py workflow."""

    logger.info("Initializing InferenceWrapper...")

    # Initialize wrapper exactly as pipeline2 does
    wrapper = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        which_epoch='328',
        model_file_name='328_model.pth',
        project_dir='./nemo/',
        folder='logs',
        model_='va',
        use_gpu=True,
        args_overwrite={'image_size': 512}
    )

    logger.info("Loading images...")

    # Load IMG_1.png as source identity
    source_path = Path("nemo/data/IMG_1.png")
    source_image = Image.open(source_path).convert('RGB')
    logger.info(f"Loaded source: {source_path}")

    # Load target frames from cache if available
    target_images = []
    target_names = []

    # Try to load cached frames
    import h5py
    cache_path = Path("proper_face_attributes.h5")
    if cache_path.exists():
        with h5py.File(cache_path, 'r') as f:
            # Get middle frames for best expression
            for i in [4, 5, 6]:
                if f'frame_{i:04d}' in f:
                    frame_data = torch.from_numpy(f[f'frame_{i:04d}/frame'][:]).cuda()
                    # Convert to PIL
                    frame_np = frame_data[0].cpu().permute(1, 2, 0).numpy()
                    frame_np = ((frame_np + 1) * 127.5).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                    target_images.append(frame_pil)
                    target_names.append(f"Frame_{i}")
                    logger.info(f"Added cached frame {i}")

    # Also add static test images
    for img_path in ["nemo/data/IMG_2.png", "nemo/data/IMG_3.png", "nemo/data/IMG_4.png"]:
        if Path(img_path).exists():
            img = Image.open(img_path).convert('RGB')
            target_images.append(img)
            target_names.append(Path(img_path).stem)

    if not target_images:
        logger.error("No target images found!")
        return

    logger.info(f"Processing {len(target_images)} targets...")

    # Process face swap for each target
    results = []

    for idx, (target_img, target_name) in enumerate(zip(target_images, target_names)):
        logger.info(f"\nProcessing {target_name}...")

        # Reset and process source
        logger.info("  Setting source identity...")
        with torch.no_grad():
            _ = wrapper.forward(
                source_image=source_image,
                driver_image=None,
                crop=True,
                reset_tracking=True,
                modnet_mask=False
            )

        # Apply target expression
        logger.info("  Applying target expression...")
        with torch.no_grad():
            pred_imgs, raw_img = wrapper.forward(
                source_image=None,  # Use cached source
                driver_image=target_img,
                crop=True,
                reset_tracking=False,
                smooth_pose=False,
                modnet_mask=False
            )

        if pred_imgs:
            results.append({
                'name': target_name,
                'source': source_image,
                'target': target_img,
                'result': pred_imgs[0],
                'raw': raw_img
            })
            logger.info(f"  ✓ Generated result for {target_name}")
        else:
            logger.warning(f"  ✗ No result for {target_name}")

    # Visualize results
    if results:
        n_results = min(len(results), 6)
        fig, axes = plt.subplots(3, n_results + 1, figsize=(3 * (n_results + 1), 9))

        # Show source
        axes[0, 0].imshow(source_image)
        axes[0, 0].set_title("Identity\n(IMG_1)", fontsize=10, weight='bold', color='blue')
        axes[0, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Source\nIdentity', ha='center', va='center',
                       fontsize=12, weight='bold', color='blue')
        axes[1, 0].axis('off')
        axes[2, 0].axis('off')

        for i in range(n_results):
            col = i + 1
            result = results[i]

            # Target
            axes[0, col].imshow(result['target'])
            axes[0, col].set_title(f"Target\n{result['name']}", fontsize=9)
            axes[0, col].axis('off')

            # Raw tensor output (if available)
            if result['raw'] is not None:
                raw_np = result['raw'][0].cpu().detach().numpy()
                # Check if needs conversion from [0,1] to display range
                if raw_np.min() >= 0 and raw_np.max() <= 1.1:
                    raw_np = raw_np.transpose(1, 2, 0)
                else:
                    raw_np = ((raw_np.transpose(1, 2, 0) + 1) / 2)
                raw_np = np.clip(raw_np, 0, 1)
                axes[1, col].imshow(raw_np)
                axes[1, col].set_title("Raw Output", fontsize=9)
            else:
                axes[1, col].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[1, col].axis('off')

            # Final result
            axes[2, col].imshow(result['result'])
            axes[2, col].set_title("Final Result", fontsize=9, weight='bold', color='green')
            axes[2, col].axis('off')

        # Hide unused subplots
        for i in range(n_results + 1, axes.shape[1]):
            for j in range(3):
                axes[j, i].axis('off')

        plt.suptitle("Face Swap Using Exact Pipeline2 Workflow", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("exact_pipeline2_results.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"\nSaved comparison to exact_pipeline2_results.png")

        # Save individual best results
        for i, result in enumerate(results[:3]):
            result['result'].save(f"exact_result_{i}_{result['name']}.png")
            logger.info(f"Saved exact_result_{i}_{result['name']}.png")

        # Create a single best result
        if len(results) > 0:
            best_idx = min(1, len(results) - 1)  # Try to get frame 5 or second result
            best = results[best_idx]
            best['result'].save("exact_best_result.png")
            logger.info(f"\nSaved best result to exact_best_result.png ({best['name']})")

            # Also create side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(source_image)
            axes[0].set_title("Source Identity\n(IMG_1.png)", fontsize=11, weight='bold')
            axes[0].axis('off')

            axes[1].imshow(best['target'])
            axes[1].set_title(f"Target Expression\n({best['name']})", fontsize=11)
            axes[1].axis('off')

            axes[2].imshow(best['result'])
            axes[2].set_title("Face Swap Result", fontsize=11, weight='bold', color='green')
            axes[2].axis('off')

            plt.suptitle("Best Face Swap Result (Pipeline2)", fontsize=14, weight='bold')
            plt.tight_layout()
            plt.savefig("exact_best_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()

            logger.info("Saved best comparison to exact_best_comparison.png")

    logger.info("\n=== Complete ===")
    logger.info(f"Generated {len(results)} face swap results")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Exact Pipeline2 Face Swap")
    logger.info("Using InferenceWrapper directly")
    logger.info("=" * 60)

    try:
        create_exact_face_swap()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)