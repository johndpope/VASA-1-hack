#!/usr/bin/env python3
"""
Visualize and explain XY warps vs UV warps.
Understanding the different coordinate systems and their purposes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patches
import matplotlib.patches as mpatches

def create_warp_explanation():
    """Create comprehensive visualization explaining XY vs UV warps."""

    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle("Understanding XY Warps vs UV Warps in Volumetric Avatar",
                 fontsize=16, fontweight='bold')

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2],
                          width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # ============= XY WARPS (Source Space) =============
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    xy_text = """XY WARPS (SOURCE SPACE / CANONICAL WARPING)

• Coordinate System: XY refers to spatial coordinates in the SOURCE/CANONICAL volume
• Direction: Warps FROM current expression TO canonical (neutral) space
• Purpose: Removes expression-specific deformations to create neutral appearance
• Applied to: The canonical volume before any target expression is added
• Effect: "Undoes" the current expression to get back to neutral

Key insight: XY warps normalize the expression by mapping expressive regions back to their
canonical positions. For example, if someone is smiling, XY warps move the mouth corners
back to their neutral positions."""

    ax1.text(0.05, 0.5, xy_text, fontsize=11, va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

    # ============= UV WARPS (Target Space) =============
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')

    uv_text = """UV WARPS (TARGET SPACE / EXPRESSION WARPING)

• Coordinate System: UV refers to texture/surface coordinates in the TARGET space
• Direction: Warps FROM canonical TO target expression
• Purpose: Applies the desired target expression to the normalized volume
• Applied to: The volume after XY warping (or canonical volume)
• Effect: Adds the target expression (smile, frown, etc.) to the neutral face

Key insight: UV warps apply new expressions by deforming the canonical volume according to
the target expression parameters. They work in "texture space" to ensure smooth deformation
of facial features."""

    ax2.text(0.05, 0.5, uv_text, fontsize=11, va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))

    # ============= VISUAL PIPELINE =============
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 3)
    ax3.axis('off')
    ax3.set_title("Warping Pipeline: Expression Transfer", fontsize=12, weight='bold', pad=20)

    # Draw boxes for each stage
    stages = [
        (1, 1.5, "Source\nExpressive\nFace", 'lightyellow'),
        (3, 1.5, "XY Warp\n↓\nRemove\nExpression", 'lightblue'),
        (5, 1.5, "Canonical\nNeutral\nVolume", 'white'),
        (7, 1.5, "UV Warp\n↓\nApply Target\nExpression", 'lightgreen'),
        (9, 1.5, "Target\nExpressive\nFace", 'lightyellow')
    ]

    for x, y, text, color in stages:
        if "Warp" in text:
            # Warp stages (diamonds)
            diamond = mpatches.FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor='black',
                                             linewidth=2)
            ax3.add_patch(diamond)
        else:
            # Data stages (rectangles)
            rect = mpatches.FancyBboxPatch((x-0.5, y-0.5), 1, 1,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='black',
                                         linewidth=2)
            ax3.add_patch(rect)

        ax3.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')

    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax3.annotate('', xy=(2.5, 1.5), xytext=(1.5, 1.5), arrowprops=arrow_props)
    ax3.annotate('', xy=(4.5, 1.5), xytext=(3.5, 1.5), arrowprops=arrow_props)
    ax3.annotate('', xy=(6.5, 1.5), xytext=(5.5, 1.5), arrowprops=arrow_props)
    ax3.annotate('', xy=(8.5, 1.5), xytext=(7.5, 1.5), arrowprops=arrow_props)

    # Add coordinate system labels
    ax3.text(3, 0.7, "XY Space\n(Source coords)", ha='center', fontsize=8, style='italic', color='blue')
    ax3.text(7, 0.7, "UV Space\n(Target coords)", ha='center', fontsize=8, style='italic', color='green')

    # Add examples below
    ax3.text(1, 0.3, "😊", ha='center', fontsize=20)
    ax3.text(3, 0.3, "⬇", ha='center', fontsize=15)
    ax3.text(5, 0.3, "😐", ha='center', fontsize=20)
    ax3.text(7, 0.3, "⬇", ha='center', fontsize=15)
    ax3.text(9, 0.3, "😮", ha='center', fontsize=20)

    plt.savefig("xy_uv_warps_explained.png", dpi=150, bbox_inches='tight')
    print("Saved explanation to xy_uv_warps_explained.png")
    plt.close()


def create_coordinate_system_viz():
    """Visualize the coordinate systems for XY and UV warps."""

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle("Coordinate Systems in 3D Volumetric Warping",
                 fontsize=14, fontweight='bold')

    # XY Coordinate System (3D Volume)
    ax1.set_title("XY Warps: 3D Volume Space", fontsize=12, weight='bold')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')

    # Draw 3D volume representation
    volume = plt.Rectangle((-1.5, -1.5), 3, 3, fill=True,
                          facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.3)
    ax1.add_patch(volume)

    # Draw grid
    for i in range(-1, 2):
        ax1.axhline(y=i*0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=i*0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Add arrows for axes
    ax1.arrow(-1.5, 0, 2.7, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax1.arrow(0, -1.5, 0, 2.7, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax1.text(1.3, -0.2, 'X (Width)', fontsize=10, color='red', weight='bold')
    ax1.text(-0.3, 1.3, 'Y (Height)', fontsize=10, color='red', weight='bold')

    # Add depth indication
    ax1.text(0, -1.8, 'D (Depth) ⊗', fontsize=10, color='blue', ha='center')

    # Add warp vectors
    np.random.seed(42)
    for _ in range(5):
        x, y = np.random.uniform(-1, 1, 2)
        dx, dy = np.random.uniform(-0.3, 0.3, 2)
        ax1.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05,
                 fc='darkblue', ec='darkblue', alpha=0.7)

    ax1.set_xlabel("Spatial warping in 3D volume\n(16×64×64 grid)", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # UV Coordinate System (Surface/Texture)
    ax2.set_title("UV Warps: Surface/Texture Space", fontsize=12, weight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')

    # Draw UV square
    uv_square = plt.Rectangle((0, 0), 1, 1, fill=True,
                             facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.3)
    ax2.add_patch(uv_square)

    # Draw UV grid
    for i in range(5):
        ax2.axhline(y=i*0.2, color='gray', linewidth=0.5, alpha=0.5)
        ax2.axvline(x=i*0.2, color='gray', linewidth=0.5, alpha=0.5)

    # Add arrows for axes
    ax2.arrow(0, 0.5, 0.9, 0, head_width=0.03, head_length=0.03, fc='red', ec='red')
    ax2.arrow(0.5, 0, 0, 0.9, head_width=0.03, head_length=0.03, fc='red', ec='red')
    ax2.text(0.95, 0.48, 'U', fontsize=10, color='red', weight='bold')
    ax2.text(0.48, 0.95, 'V', fontsize=10, color='red', weight='bold')

    # Add warp vectors
    for _ in range(5):
        u, v = np.random.uniform(0.2, 0.8, 2)
        du, dv = np.random.uniform(-0.1, 0.1, 2)
        ax2.arrow(u, v, du, dv, head_width=0.02, head_length=0.02,
                 fc='darkgreen', ec='darkgreen', alpha=0.7)

    ax2.set_xlabel("Texture/surface warping\n(0-1 normalized coordinates)", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Combined Effect
    ax3.set_title("Combined XY + UV Warping", fontsize=12, weight='bold')
    ax3.axis('off')

    combined_text = """Sequential Application:

1. XY Warp (Blue):
   • Operates in 3D volume space
   • Removes source expression
   • Creates canonical volume

2. UV Warp (Green):
   • Operates in surface space
   • Applies target expression
   • Deforms canonical to target

Result:
Source Expression → Canonical → Target Expression

The two-stage process ensures:
• Clean expression transfer
• Identity preservation
• Smooth deformations
"""

    ax3.text(0.1, 0.5, combined_text, fontsize=10, va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig("coordinate_systems_visualization.png", dpi=150, bbox_inches='tight')
    print("Saved coordinate systems to coordinate_systems_visualization.png")
    plt.close()


def create_warp_magnitude_comparison():
    """Create a comparison of typical warp magnitudes."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Typical Warp Magnitudes for Different Expressions",
                 fontsize=14, fontweight='bold')

    # Create synthetic warp magnitude maps
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)

    # XY Warp - Smile to Neutral
    ax = axes[0, 0]
    # Higher magnitude around mouth area
    mouth_mask = np.exp(-((X)**2 + (Y+0.3)**2)/0.1)
    xy_magnitude = mouth_mask * 1.5
    im = ax.imshow(xy_magnitude, cmap='viridis', vmin=0, vmax=2)
    ax.set_title("XY Warp: Smile → Neutral", fontsize=10, weight='bold')
    ax.set_xlabel("Removes smile deformation")
    plt.colorbar(im, ax=ax, label='Magnitude')

    # XY Warp - Open mouth to Neutral
    ax = axes[0, 1]
    mouth_mask = np.exp(-((X)**2 + (Y+0.3)**2)/0.15)
    xy_magnitude = mouth_mask * 2.0
    im = ax.imshow(xy_magnitude, cmap='viridis', vmin=0, vmax=2)
    ax.set_title("XY Warp: Open → Closed", fontsize=10, weight='bold')
    ax.set_xlabel("Closes mouth")
    plt.colorbar(im, ax=ax, label='Magnitude')

    # UV Warp - Neutral to Surprise
    ax = axes[1, 0]
    # High magnitude for eyebrows and mouth
    eyebrow_mask = np.exp(-((X)**2 + (Y-0.3)**2)/0.1)
    mouth_mask = np.exp(-((X)**2 + (Y+0.3)**2)/0.1)
    uv_magnitude = eyebrow_mask * 0.8 + mouth_mask * 1.2
    im = ax.imshow(uv_magnitude, cmap='plasma', vmin=0, vmax=2)
    ax.set_title("UV Warp: Neutral → Surprise", fontsize=10, weight='bold')
    ax.set_xlabel("Raises eyebrows, opens mouth")
    plt.colorbar(im, ax=ax, label='Magnitude')

    # UV Warp - Neutral to Frown
    ax = axes[1, 1]
    # Downward motion around mouth
    mouth_mask = np.exp(-((X)**2 + (Y+0.3)**2)/0.12)
    forehead_mask = np.exp(-((X)**2 + (Y-0.4)**2)/0.15)
    uv_magnitude = mouth_mask * 1.0 + forehead_mask * 0.5
    im = ax.imshow(uv_magnitude, cmap='plasma', vmin=0, vmax=2)
    ax.set_title("UV Warp: Neutral → Frown", fontsize=10, weight='bold')
    ax.set_xlabel("Lowers mouth corners")
    plt.colorbar(im, ax=ax, label='Magnitude')

    # Add text explanation
    fig.text(0.5, 0.02,
            "XY warps (top) normalize expressions back to canonical. "
            "UV warps (bottom) apply target expressions to canonical volume.",
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig("warp_magnitude_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved magnitude comparison to warp_magnitude_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Creating XY vs UV warp explanations...")
    create_warp_explanation()
    create_coordinate_system_viz()
    create_warp_magnitude_comparison()
    print("\nGenerated visualizations:")
    print("  - xy_uv_warps_explained.png")
    print("  - coordinate_systems_visualization.png")
    print("  - warp_magnitude_comparison.png")