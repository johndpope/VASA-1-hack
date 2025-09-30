#!/usr/bin/env python3
"""
Test pipeline2.py directly to understand the correct face swap flow.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import logging

# Add paths
sys.path.insert(0, 'nemo')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline2
from pipeline2 import InferenceWrapper


def test_pipeline2_face_swap():
    """Test face swap using pipeline2.py directly."""

    logger.info("Initializing InferenceWrapper...")

    # Initialize the wrapper
    wrapper = InferenceWrapper(
        experiment_name='Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1',
        which_epoch='328',
        model_file_name='328_model.pth',
        project_dir='./nemo/',
        folder='logs',
        model_='va',
        args_overwrite={'image_size': 512}
    )

    logger.info("Loading source and target images...")

    # Load IMG_1.png as source (identity)
    source_path = Path("nemo/data/IMG_1.png")
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    source_image = Image.open(source_path).convert('RGB')

    # Load a target video frame
    # For testing, let's use one of the video frames
    # We need to find a good target frame - let's check what's available
    video_frames_path = Path("nemo/data/")

    # List available images
    available_images = list(video_frames_path.glob("*.png"))
    logger.info(f"Available images: {[img.name for img in available_images]}")

    # Use IMG_0845.png as target if it exists, otherwise use IMG_1.png
    target_path = video_frames_path / "IMG_0845.png"
    if not target_path.exists():
        # Try to find another image
        other_imgs = [img for img in available_images if img.name != "IMG_1.png"]
        if other_imgs:
            target_path = other_imgs[0]
        else:
            target_path = source_path  # Use same as source for testing

    logger.info(f"Using target image: {target_path}")
    target_image = Image.open(target_path).convert('RGB')

    # Test 1: Source only (should extract identity)
    logger.info("\n=== Test 1: Processing source image ===")
    with torch.no_grad():
        result = wrapper.forward(
            source_image=source_image,
            driver_image=None,
            crop=True,
            reset_tracking=True
        )
    logger.info("Source processing complete - identity extracted")

    # Test 2: Apply target expression to source identity
    logger.info("\n=== Test 2: Applying target to source identity ===")
    with torch.no_grad():
        pred_imgs, raw_img = wrapper.forward(
            source_image=None,  # Use cached source
            driver_image=target_image,
            crop=True,
            reset_tracking=False
        )

    if pred_imgs:
        # Save result
        result_img = pred_imgs[0]
        result_img.save("pipeline2_face_swap_test.png")
        logger.info(f"Saved face swap result to pipeline2_face_swap_test.png")

        # Also save comparison
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(source_image)
        axes[0].set_title("Source (IMG_1.png)")
        axes[0].axis('off')

        axes[1].imshow(target_image)
        axes[1].set_title(f"Target ({target_path.name})")
        axes[1].axis('off')

        axes[2].imshow(result_img)
        axes[2].set_title("Face Swap Result")
        axes[2].axis('off')

        plt.suptitle("Pipeline2 Direct Face Swap Test", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("pipeline2_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved comparison to pipeline2_comparison.png")
    else:
        logger.warning("No result returned from pipeline2")

    # Test 3: Multiple target frames
    logger.info("\n=== Test 3: Testing multiple targets ===")

    # Reset and set source again
    with torch.no_grad():
        wrapper.forward(
            source_image=source_image,
            driver_image=None,
            crop=True,
            reset_tracking=True
        )

    # Try multiple targets
    test_targets = available_images[:5] if len(available_images) > 1 else [source_path]
    results = []

    for target_path in test_targets:
        if target_path.name == "IMG_1.png":
            continue  # Skip source

        logger.info(f"Processing target: {target_path.name}")
        target = Image.open(target_path).convert('RGB')

        with torch.no_grad():
            pred_imgs, _ = wrapper.forward(
                source_image=None,
                driver_image=target,
                crop=True,
                reset_tracking=False
            )

        if pred_imgs:
            results.append((target, pred_imgs[0]))

    if results:
        # Create grid of results
        fig, axes = plt.subplots(2, len(results) + 1, figsize=(3 * (len(results) + 1), 6))

        # Show source in first column
        axes[0, 0].imshow(source_image)
        axes[0, 0].set_title("Source\n(IMG_1.png)", fontsize=9, weight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')

        for i, (target, result) in enumerate(results):
            col = i + 1

            axes[0, col].imshow(target)
            axes[0, col].set_title(f"Target {i+1}", fontsize=9)
            axes[0, col].axis('off')

            axes[1, col].imshow(result)
            axes[1, col].set_title(f"Result {i+1}", fontsize=9, color='green')
            axes[1, col].axis('off')

        plt.suptitle("Pipeline2 Multiple Targets Test", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("pipeline2_multiple_test.png", dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {len(results)} results to pipeline2_multiple_test.png")

    logger.info("\n=== Pipeline2 Test Complete ===")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Pipeline2 Face Swap Directly")
    logger.info("=" * 60)

    try:
        test_pipeline2_face_swap()
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)