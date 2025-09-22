#!/usr/bin/env python3
"""Generate single frame thumbnails with debug overlay for wandb logging"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Optional, Tuple
import io

def create_three_panel_thumbnail(
    identity_frame: Optional[torch.Tensor] = None,
    target_frame: Optional[torch.Tensor] = None,
    predicted_frame: Optional[torch.Tensor] = None,
    motion_params: Optional[Dict[str, torch.Tensor]] = None,
    size: Tuple[int, int] = (768, 256),  # Wider for 3 panels
    add_overlay: bool = True
) -> np.ndarray:
    """
    Create a three-panel thumbnail showing Identity | Target | Predicted.

    Args:
        identity_frame: Identity/source frame [C, H, W] or [H, W, C]
        target_frame: Target/ground truth frame
        predicted_frame: Generated/predicted frame
        motion_params: Dict with motion parameters for overlay
        size: Output thumbnail size (width, height)
        add_overlay: Whether to add debug visualization overlay

    Returns:
        numpy array of thumbnail image [H, W, C] in uint8 format
    """

    # Helper function to process frame
    def process_frame(frame):
        if frame is None:
            return np.zeros((256, 256, 3), dtype=np.float32)

        if isinstance(frame, torch.Tensor):
            if frame.dim() == 4:  # [B, C, H, W]
                frame = frame[0]
            if frame.dim() == 3 and frame.shape[0] == 3:  # [C, H, W]
                frame = frame.permute(1, 2, 0)
            frame = frame.detach().cpu().numpy()

        # Normalize to [0, 1] if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
        if frame.min() < 0:
            frame = (frame + 1) / 2  # Convert from [-1, 1] to [0, 1]

        return frame

    # Process all frames
    identity_np = process_frame(identity_frame)
    target_np = process_frame(target_frame)
    predicted_np = process_frame(predicted_frame)

    # Create figure with 3 subplots
    fig_scale = size[0] / 384  # Base scale on target width
    fig, axes = plt.subplots(1, 3, figsize=(12 * fig_scale, 4 * fig_scale))

    # Show Identity
    axes[0].imshow(identity_np)
    axes[0].set_title("Identity (Source)", fontsize=10 * fig_scale, weight='bold', color='blue')
    axes[0].axis('off')

    # Show Target
    axes[1].imshow(target_np)
    frame_idx = motion_params.get('_frame_idx', -1) if motion_params else -1
    title = f"Target (t={frame_idx})" if frame_idx >= 0 else "Target Frame"
    axes[1].set_title(title, fontsize=10 * fig_scale, weight='bold', color='green')
    axes[1].axis('off')

    # Show Predicted
    axes[2].imshow(predicted_np)
    axes[2].set_title("Predicted", fontsize=10 * fig_scale, weight='bold', color='red')
    axes[2].axis('off')

    # Add motion overlay on predicted frame
    if add_overlay and motion_params is not None:
        ax = axes[2]

        # Calculate motion indicators
        indicators = []

        # Add frame index if available
        if '_frame_idx' in motion_params:
            indicators.append(f"Frame: {motion_params['_frame_idx']}")

        # Add motion stats
        if 'theta' in motion_params:
            theta = motion_params['theta']
            if isinstance(theta, torch.Tensor):
                if theta.dim() > 1 and theta.shape[0] > 1:
                    theta_diff = (theta[1:] - theta[:-1]).abs().mean().item()
                    indicators.append(f"Motion: {theta_diff:.4f}")

        # Add expression stats
        if 'expression_embed' in motion_params:
            expr = motion_params['expression_embed']
            if isinstance(expr, torch.Tensor):
                expr_std = expr.std().item()
                indicators.append(f"Expr σ: {expr_std:.3f}")

        # Add text overlay
        font_scale = max(1.0, size[0] / 512)
        text_y = 0.98
        for indicator in indicators[:3]:  # Limit to 3 indicators to avoid clutter
            ax.text(0.02, text_y, indicator, transform=ax.transAxes,
                   fontsize=7 * font_scale, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
                   verticalalignment='top')
            text_y -= 0.06

        # Add theta warp matrix visualization
        if 'theta' in motion_params:
            theta = motion_params['theta']
            if isinstance(theta, torch.Tensor):
                # Get the theta matrix for current frame
                if theta.dim() == 4:  # [B, T, 3, 4]
                    frame_idx = motion_params.get('_frame_idx', 0)
                    if frame_idx < theta.shape[1]:
                        theta_matrix = theta[0, frame_idx].detach().cpu().numpy()  # [3, 4]
                    else:
                        theta_matrix = theta[0, -1].detach().cpu().numpy()
                elif theta.dim() == 3:  # [B, 3, 4]
                    theta_matrix = theta[0].detach().cpu().numpy()
                elif theta.dim() == 2:  # [3, 4]
                    theta_matrix = theta.detach().cpu().numpy()
                else:
                    theta_matrix = None

                if theta_matrix is not None and theta_matrix.shape == (3, 4):
                    # Create inset axes for theta matrix in bottom-right corner
                    inset_ax = ax.inset_axes([0.55, 0.02, 0.43, 0.25])

                    # Visualize theta matrix as heatmap
                    im = inset_ax.imshow(theta_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

                    # Add values as text annotations
                    for i in range(3):
                        for j in range(4):
                            val = theta_matrix[i, j]
                            color = 'white' if abs(val) > 0.5 else 'black'
                            inset_ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                        fontsize=5 * font_scale, color=color, weight='bold')

                    # Style the inset
                    inset_ax.set_xticks([0, 1, 2, 3])
                    inset_ax.set_yticks([0, 1, 2])
                    inset_ax.set_xticklabels(['R1', 'R2', 'R3', 'T'], fontsize=5 * font_scale)
                    inset_ax.set_yticklabels(['X', 'Y', 'Z'], fontsize=5 * font_scale)
                    inset_ax.set_title('Theta Warp (3x4)', fontsize=6 * font_scale, color='white', pad=2)

                    # Add subtle border
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor('white')
                        spine.set_linewidth(0.5)

    # Convert figure to numpy array
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Get numpy array from figure
    buf = io.BytesIO()
    target_dpi = max(100, int(size[0] / 6))  # Scale DPI based on target size
    fig.savefig(buf, format='png', dpi=target_dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)

    # Resize to exact target size with high quality resampling
    img = img.resize(size, Image.Resampling.LANCZOS)
    thumbnail = np.array(img)

    plt.close(fig)
    buf.close()

    return thumbnail


def create_debug_thumbnail(
    generated_frame: torch.Tensor,
    source_frame: Optional[torch.Tensor] = None,
    motion_params: Optional[Dict[str, torch.Tensor]] = None,
    size: Tuple[int, int] = (512, 512),
    add_overlay: bool = True
) -> np.ndarray:
    """
    Create a thumbnail with debug overlay showing motion parameters.
    
    Args:
        generated_frame: Generated frame tensor [C, H, W] or [H, W, C]
        source_frame: Optional source/identity frame for comparison
        motion_params: Dict with 'theta', 'rotation', 'translation', 'expression', etc.
        size: Output thumbnail size (width, height)
        add_overlay: Whether to add debug visualization overlay
        
    Returns:
        numpy array of thumbnail image [H, W, C] in uint8 format
    """
    
    # Handle tensor format
    if isinstance(generated_frame, torch.Tensor):
        if generated_frame.dim() == 4:  # [B, C, H, W]
            generated_frame = generated_frame[0]
        if generated_frame.dim() == 3 and generated_frame.shape[0] == 3:  # [C, H, W]
            generated_frame = generated_frame.permute(1, 2, 0)
        generated_frame = generated_frame.detach().cpu().numpy()
    
    # Normalize to [0, 1] if needed
    if generated_frame.max() > 1.0:
        generated_frame = generated_frame / 255.0
    if generated_frame.min() < 0:
        generated_frame = (generated_frame - generated_frame.min()) / (generated_frame.max() - generated_frame.min())
    
    # Create figure with subplots - adjust figure size based on target size
    fig_scale = size[0] / 128  # Scale figure size based on target resolution
    if source_frame is not None and add_overlay:
        fig, axes = plt.subplots(1, 2, figsize=(8 * fig_scale, 4 * fig_scale))
        
        # Process source frame
        if isinstance(source_frame, torch.Tensor):
            if source_frame.dim() == 4:
                source_frame = source_frame[0]
            if source_frame.dim() == 3 and source_frame.shape[0] == 3:
                source_frame = source_frame.permute(1, 2, 0)
            source_frame = source_frame.detach().cpu().numpy()
        
        if source_frame.max() > 1.0:
            source_frame = source_frame / 255.0
        if source_frame.min() < 0:
            source_frame = (source_frame - source_frame.min()) / (source_frame.max() - source_frame.min())
        
        # Show target/ground truth
        axes[0].imshow(source_frame)
        axes[0].set_title("Target Frame (Ground Truth)", fontsize=10 * fig_scale)
        axes[0].axis('off')
        
        # Show generated with frame index if available
        frame_idx = motion_params.get('_frame_idx', -1) if motion_params else -1
        title = f"Generated Frame (t={frame_idx})" if frame_idx >= 0 else "Generated Frame"
        axes[1].imshow(generated_frame)
        axes[1].set_title(title, fontsize=10 * fig_scale)
        axes[1].axis('off')
        
        ax = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4 * fig_scale, 4 * fig_scale))
        ax.imshow(generated_frame)
        ax.axis('off')
    
    # Add debug overlay if requested
    if add_overlay and motion_params is not None:
        # Calculate motion indicators
        indicators = []
        
        # Add frame index if available
        if '_frame_idx' in motion_params:
            indicators.append(f"Frame: {motion_params['_frame_idx']}")
        
        if 'rotation' in motion_params:
            rot = motion_params['rotation']
            if isinstance(rot, torch.Tensor):
                if rot.dim() > 1:
                    rot = rot[-1] if rot.shape[0] > 1 else rot[0]  # Get last frame
                rot = rot.detach().cpu().numpy()
            # Convert rotation to degrees for visualization
            if isinstance(rot, np.ndarray):
                rot_deg = np.rad2deg(rot) if rot.max() < np.pi else rot
                if rot_deg.shape == (3,):
                    indicators.append(f"R: [{float(rot_deg[0]):.1f}, {float(rot_deg[1]):.1f}, {float(rot_deg[2]):.1f}]°")
                else:
                    indicators.append(f"R: {float(rot_deg.mean()):.1f}°")
        
        if 'translation' in motion_params:
            trans = motion_params['translation']
            if isinstance(trans, torch.Tensor):
                if trans.dim() > 1:
                    trans = trans[-1] if trans.shape[0] > 1 else trans[0]
                trans = trans.detach().cpu().numpy()
            if isinstance(trans, np.ndarray):
                if trans.shape == (3,):
                    indicators.append(f"T: [{float(trans[0]):.3f}, {float(trans[1]):.3f}, {float(trans[2]):.3f}]")
                else:
                    indicators.append(f"T: {float(trans.mean()):.3f}")
        
        if 'expression' in motion_params:
            expr = motion_params['expression']
            if isinstance(expr, torch.Tensor):
                if expr.dim() > 1:
                    expr = expr[-1] if expr.shape[0] > 1 else expr[0]
                expr_std = expr.std().item()
                expr_mean = expr.mean().item()
            else:
                expr_std = 0
                expr_mean = 0
            indicators.append(f"Expr: μ={expr_mean:.3f}, σ={expr_std:.3f}")
        
        if 'gaze' in motion_params:
            gaze = motion_params['gaze']
            if isinstance(gaze, torch.Tensor):
                if gaze.dim() > 1:
                    gaze = gaze[-1] if gaze.shape[0] > 1 else gaze[0]
                gaze = gaze.detach().cpu().numpy()
            if isinstance(gaze, np.ndarray):
                if gaze.shape == (2,):
                    indicators.append(f"Gaze: [{float(gaze[0]):.2f}, {float(gaze[1]):.2f}]")

                    # Draw gaze arrows on the image
                    # Assuming gaze[0] is pitch (vertical) and gaze[1] is yaw (horizontal)
                    # Convert from radians to pixel coordinates
                    img_height, img_width = generated_frame.shape[:2]

                    # Estimate eye positions (typical face proportions)
                    # Left eye at ~0.35 width, right eye at ~0.65 width, both at ~0.4 height
                    left_eye_x = int(0.35 * img_width)
                    right_eye_x = int(0.65 * img_width)
                    eye_y = int(0.4 * img_height)

                    # Convert gaze angles to arrow endpoints
                    # Scale factor for arrow length
                    arrow_length = img_width * 0.15

                    # Calculate arrow direction from gaze angles
                    # gaze[1] is yaw (horizontal), gaze[0] is pitch (vertical)
                    dx = np.sin(gaze[1]) * arrow_length
                    dy = -np.sin(gaze[0]) * arrow_length  # Negative because image y-axis is inverted

                    # Draw arrows for both eyes
                    for eye_x in [left_eye_x, right_eye_x]:
                        # Starting point (eye position)
                        start_x = eye_x / img_width
                        start_y = eye_y / img_height

                        # End point based on gaze direction
                        end_x = (eye_x + dx) / img_width
                        end_y = (eye_y + dy) / img_height

                        # Draw arrow using matplotlib annotation
                        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                  xycoords='axes fraction', textcoords='axes fraction',
                                  arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                                shrinkA=0, shrinkB=0))
                else:
                    indicators.append(f"Gaze: {float(gaze.mean()):.2f}")
        
        # Add motion variance indicator
        if 'theta' in motion_params:
            theta = motion_params['theta']
            if isinstance(theta, torch.Tensor) and theta.dim() > 1 and theta.shape[0] > 1:
                # Calculate frame-to-frame motion
                theta_diff = (theta[1:] - theta[:-1]).abs().mean().item()
                motion_color = 'green' if theta_diff > 1e-3 else 'red'
                indicators.append(f"Motion: {theta_diff:.4f}")
                
                # Add motion indicator bar
                rect = patches.Rectangle((0.02, 0.02), 0.04, 0.2, 
                                        linewidth=1, edgecolor='white',
                                        facecolor=motion_color, alpha=0.7,
                                        transform=ax.transAxes)
                ax.add_patch(rect)
        
        # Add text overlay - scale font based on image size
        font_scale = max(1.0, size[0] / 256)
        text_y = 0.98
        for indicator in indicators:
            ax.text(0.02, text_y, indicator, transform=ax.transAxes,
                   fontsize=8 * font_scale, color='white', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5),
                   verticalalignment='top')
            text_y -= 0.08
        
        # Add variance heatmap in corner
        if 'expression' in motion_params:
            expr = motion_params['expression']
            if isinstance(expr, torch.Tensor) and expr.dim() > 1:
                # Create small heatmap of expression variance over time
                expr_var = expr.var(dim=-1).detach().cpu().numpy()
                if len(expr_var) > 1:
                    # Normalize variance for visualization
                    expr_var_norm = (expr_var - expr_var.min()) / (expr_var.max() - expr_var.min() + 1e-8)
                    
                    # Create small inset axes for variance plot
                    inset_ax = ax.inset_axes([0.7, 0.02, 0.28, 0.15])
                    inset_ax.plot(expr_var_norm, color='lime', linewidth=1)
                    inset_ax.set_xlim(0, len(expr_var_norm))
                    inset_ax.set_ylim(0, 1)
                    inset_ax.set_xticks([])
                    inset_ax.set_yticks([])
                    inset_ax.set_title("Expression Var", fontsize=6, color='white')
                    inset_ax.patch.set_facecolor('black')
                    inset_ax.patch.set_alpha(0.5)
    
    # Convert figure to numpy array
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    
    # Get numpy array from figure
    buf = io.BytesIO()
    # Use higher DPI for 512x512 images to ensure clarity
    target_dpi = max(100, int(size[0] / 4))  # Scale DPI based on target size
    fig.savefig(buf, format='png', dpi=target_dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    
    # Resize to exact target size with high quality resampling
    img = img.resize(size, Image.Resampling.LANCZOS)
    thumbnail = np.array(img)
    
    plt.close(fig)
    buf.close()
    
    return thumbnail


def generate_simple_thumbnail(
    predicted_motion: Dict[str, torch.Tensor],
    ground_truth: Optional[torch.Tensor] = None,
    size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Generate a simple thumbnail without volumetric rendering, just showing motion stats.
    
    Args:
        predicted_motion: Predicted motion parameters from model
        ground_truth: Optional ground truth frame for comparison
        size: Output thumbnail size
        
    Returns:
        Thumbnail as numpy array
    """
    # Create a figure showing motion statistics - scale based on target size
    fig_scale = size[0] / 128
    fig, ax = plt.subplots(1, 1, figsize=(4 * fig_scale, 4 * fig_scale))
    
    # Create background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#0d0d0d')
    
    # Calculate motion statistics
    stats_text = []
    
    # Motion variance
    if 'theta' in predicted_motion:
        theta = predicted_motion['theta']
        if isinstance(theta, torch.Tensor) and theta.dim() > 1 and theta.shape[1] > 1:
            theta_var = theta.var(dim=1).mean().item()
            theta_diff = (theta[:, 1:] - theta[:, :-1]).abs().mean().item()
            color = '#00ff00' if theta_diff > 1e-3 else '#ff0000'
            stats_text.append(('Motion Variance:', f'{theta_var:.6f}', color))
            stats_text.append(('Frame Diff:', f'{theta_diff:.6f}', color))
    
    # Expression stats
    if 'expression' in predicted_motion:
        expr = predicted_motion['expression']
        if isinstance(expr, torch.Tensor):
            expr_std = expr.std().item()
            expr_mean = expr.abs().mean().item()
            stats_text.append(('Expression σ:', f'{expr_std:.4f}', '#00ffff'))
            stats_text.append(('Expression |μ|:', f'{expr_mean:.4f}', '#00ffff'))
    
    # Rotation stats
    if 'rotation' in predicted_motion:
        rot = predicted_motion['rotation']
        if isinstance(rot, torch.Tensor):
            rot_range = (rot.max() - rot.min()).item()
            stats_text.append(('Rotation Range:', f'{np.rad2deg(rot_range):.1f}°', '#ffff00'))
    
    # Display stats - scale font size based on resolution
    font_scale = max(1.0, size[0] / 256)  # Scale fonts for larger images
    y_pos = 0.9
    for label, value, color in stats_text:
        ax.text(0.1, y_pos, label, transform=ax.transAxes,
                fontsize=10 * font_scale, color='#808080', fontweight='normal')
        ax.text(0.6, y_pos, value, transform=ax.transAxes,
                fontsize=10 * font_scale, color=color, fontweight='bold')
        y_pos -= 0.15
    
    # Add motion indicator bar
    if 'theta' in predicted_motion:
        theta = predicted_motion['theta']
        if isinstance(theta, torch.Tensor) and theta.dim() > 1 and theta.shape[1] > 1:
            # Create motion intensity bar
            motion_intensity = (theta[:, 1:] - theta[:, :-1]).abs().mean().item()
            bar_height = min(motion_intensity * 1000, 1.0)  # Scale for visibility
            bar_color = plt.cm.RdYlGn(bar_height)  # Red to green colormap
            
            rect = patches.Rectangle((0.85, 0.1), 0.1, bar_height * 0.8,
                                    linewidth=2, edgecolor='white',
                                    facecolor=bar_color, alpha=0.8,
                                    transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(0.9, 0.05, 'Motion', transform=ax.transAxes,
                   fontsize=8 * font_scale, color='white', ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = io.BytesIO()
    # Use higher DPI for 512x512 images
    target_dpi = max(100, int(size[0] / 4))
    fig.savefig(buf, format='png', dpi=target_dpi, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize(size, Image.Resampling.LANCZOS)
    thumbnail = np.array(img)
    
    plt.close(fig)
    buf.close()
    
    return thumbnail


def generate_training_thumbnail(
    model,
    volumetric_avatar,
    source_params: Dict[str, torch.Tensor],
    predicted_motion: Dict[str, torch.Tensor],
    source_frames: Optional[torch.Tensor] = None,
    window_idx: int = -1
) -> np.ndarray:
    """
    Generate a single frame thumbnail from training using the last frame of predicted motion.
    
    Args:
        model: VASA model
        volumetric_avatar: Volumetric avatar model for rendering
        source_params: Source identity parameters (canonical_volume, idt_embed, etc.)
        predicted_motion: Predicted motion parameters from model
        source_frames: Actual source frames from dataset [B, T, C, H, W] or [B, T, H, W, C]
        window_idx: Which frame to use (-1 for last frame)
        
    Returns:
        Thumbnail as numpy array
    """
    
    with torch.no_grad():
        # Get the last frame's parameters
        if window_idx == -1:
            window_idx = predicted_motion['theta'].shape[1] - 1
        
        frame_params = {}
        for key in ['theta', 'rotation', 'translation', 'scale', 'expression']:
            if key in predicted_motion:
                if predicted_motion[key].dim() > 2:
                    frame_params[key] = predicted_motion[key][:, window_idx:window_idx+1]
                else:
                    frame_params[key] = predicted_motion[key][:, window_idx:window_idx+1]
        
        # Use volumetric avatar to render the frame
        try:
            # Get dimensions
            c = volumetric_avatar.args.latent_volume_channels
            d = volumetric_avatar.args.latent_volume_depth
            s = volumetric_avatar.args.latent_volume_size
            
            # Use actual source image if provided
            if source_frames is not None and source_frames.numel() > 0:
                # Extract first frame as source
                if source_frames.dim() == 5:  # [B, T, C, H, W]
                    source_img = source_frames[0, 0]  # First batch, first frame
                    target_img = source_frames[0, window_idx] if window_idx < source_frames.shape[1] else source_frames[0, -1]
                elif source_frames.dim() == 4:  # [B, C, H, W]
                    source_img = source_frames[0]
                    target_img = source_frames[0]
                else:
                    source_img = torch.zeros(3, 512, 512).cuda()
                    target_img = torch.zeros(3, 512, 512).cuda()
                    
                # Ensure correct shape and device
                if source_img.shape[-1] != 512:
                    import torch.nn.functional as F
                    source_img = F.interpolate(source_img.unsqueeze(0), size=(512, 512), mode='bilinear')[0]
                    target_img = F.interpolate(target_img.unsqueeze(0), size=(512, 512), mode='bilinear')[0]
            else:
                source_img = torch.zeros(3, 512, 512).cuda()
                target_img = torch.zeros(3, 512, 512).cuda()
            
            # Create data dict for decoder
            data_dict = {
                'source_img': source_img.unsqueeze(0),  # Add batch dimension
                'target_img': target_img.unsqueeze(0),  # Use actual target frame
                'source_theta': frame_params.get('theta', torch.eye(3, 4).unsqueeze(0).cuda()),
                'target_theta': frame_params.get('theta', torch.eye(3, 4).unsqueeze(0).cuda()),
            }
            
            # Generate warp embeddings
            source_warp_embed_dict, target_warp_embed_dict, _, embed_dict = \
                volumetric_avatar.predict_embed(data_dict)
            
            # Apply motion to canonical volume using warping fields
            if 'canonical_volume' in source_params:
                canonical_volume = source_params['canonical_volume']

                # Check if we have warping fields from the motion prediction
                if 'xy_warps' in predicted_motion and 'rigid_warps' in predicted_motion and 'uv_warps' in predicted_motion:
                    # Extract warping fields for the selected frame
                    xy_warp = predicted_motion['xy_warps'][:, window_idx]  # [B, 16, 64, 64, 3]
                    rigid_warp = predicted_motion['rigid_warps'][:, window_idx]  # [B, 16, 64, 64, 3]
                    uv_warp = predicted_motion['uv_warps'][:, window_idx]  # [B, 16, 64, 64, 3]

                    # Apply warps in sequence like pipeline_face_attr.py
                    # 1. Apply source warps to canonical volume (neutralize identity)
                    source_warped = volumetric_avatar.grid_sample(
                        volumetric_avatar.grid_sample(canonical_volume, rigid_warp),
                        xy_warp
                    )

                    # 2. Apply target warp to get final volume
                    grid = volumetric_avatar.identity_grid_3d.repeat_interleave(1, dim=0)
                    if 'theta' in frame_params:
                        target_rotation_warp = grid.bmm(frame_params['theta'][:, 0, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                    else:
                        target_rotation_warp = grid.view(-1, d, s, s, 3)

                    # Apply UV warp and then rotation
                    warped_volume = volumetric_avatar.grid_sample(
                        volumetric_avatar.grid_sample(source_warped, uv_warp),
                        target_rotation_warp
                    )
                    target_latent_feats = warped_volume.view(1, c * d, s, s)
                else:
                    # Fallback: simple rotation warp without xy/uv warps
                    grid = volumetric_avatar.identity_grid_3d.repeat_interleave(1, dim=0)
                    if 'theta' in frame_params:
                        target_rotation_warp = grid.bmm(frame_params['theta'][:, 0, :3].transpose(1, 2)).view(-1, d, s, s, 3)
                    else:
                        target_rotation_warp = grid.view(-1, d, s, s, 3)

                    # Apply warping
                    warped_volume = volumetric_avatar.grid_sample(canonical_volume, target_rotation_warp)
                    target_latent_feats = warped_volume.view(1, c * d, s, s)
            else:
                # Fallback: use zero volume
                target_latent_feats = torch.zeros(1, c * d, s, s).cuda()
            
            # Generate frame through decoder
            generated_frame, _, _, _ = volumetric_avatar.decoder_nw(
                data_dict,
                embed_dict,
                target_latent_feats,
                None  # No separate warping, already applied
            )
            
            # Create thumbnail with overlay
            thumbnail = create_debug_thumbnail(
                generated_frame,
                source_frame=source_params.get('source_img', None),
                motion_params=predicted_motion,
                size=(256, 512),  # Wider to show both source and generated
                add_overlay=True
            )
            
        except Exception as e:
            # Fallback: create simple thumbnail without volumetric rendering
            import traceback
            traceback.print_exc()
            print(f"Warning: Could not render with volumetric avatar: {e}")
            
            # Create a placeholder thumbnail
            placeholder = torch.randn(3, 512, 512) * 0.1 + 0.5
            thumbnail = create_debug_thumbnail(
                placeholder,
                motion_params=predicted_motion,
                size=(512, 512),
                add_overlay=True
            )
    
    return thumbnail

def generate_window_thumbnail(
    generated_frames: Optional[torch.Tensor] = None,
    target_frames: Optional[torch.Tensor] = None,
    identity_frame: Optional[torch.Tensor] = None,
    motion_outputs: Optional[Dict] = None,
    window: Optional[Dict] = None,
    motion_data: Optional[Dict] = None,
    outputs: Optional[Dict] = None,
    size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Generate thumbnail from generated and target frames with identity reference.

    Args:
        generated_frames: Generated frames from volumetric avatar [B, T, C, H, W]
        target_frames: Target/ground truth frames [B, T, C, H, W]
        identity_frame: Identity/source frame used for generation [C, H, W] or [B, C, H, W]
        motion_outputs: Motion outputs for overlay stats
        window: (Optional, for backward compatibility) Window data dictionary
        motion_data: (Optional, for backward compatibility) Motion data from prepare_motion_data
        outputs: (Optional, for backward compatibility) Model outputs
        size: Target thumbnail size

    Returns:
        Thumbnail as numpy array showing Identity | Target | Predicted
    """
    import random
    
    # Handle backward compatibility
    if generated_frames is None and outputs is not None:
        outputs = outputs
    elif motion_outputs is not None:
        outputs = motion_outputs
    
    # Try to extract frames from various sources
    selected_generated = None
    selected_target = None
    selected_idx = 0
    
    # Use generated frames if provided (from disentanglement loss)
    if generated_frames is not None and isinstance(generated_frames, torch.Tensor):
        if generated_frames.numel() > 0:
            # Handle shape [B, T, C, H, W]
            if generated_frames.dim() == 5:
                B, T, C, H, W = generated_frames.shape
                # Select a random frame, biased towards later frames
                if T > 5:
                    min_idx = T // 3
                    selected_idx = random.randint(min_idx, T - 1)
                else:
                    selected_idx = random.randint(0, max(0, T - 1))
                selected_generated = generated_frames[0, selected_idx]  # [C, H, W]
            elif generated_frames.dim() == 4:  # [B, C, H, W]
                selected_idx = 0
                selected_generated = generated_frames[0]
            elif generated_frames.dim() == 3:  # [C, H, W]
                selected_generated = generated_frames
    
    # Use target frames for comparison
    if target_frames is not None and isinstance(target_frames, torch.Tensor):
        if target_frames.numel() > 0:
            if target_frames.dim() == 5:  # [B, T, C, H, W]
                selected_target = target_frames[0, selected_idx] if selected_idx < target_frames.shape[1] else target_frames[0, 0]
            elif target_frames.dim() == 4:  # [B, C, H, W]
                selected_target = target_frames[0]
            elif target_frames.dim() == 3:  # [C, H, W]
                selected_target = target_frames
    
    # Fallback to window frames if no generated frames
    if selected_generated is None and window is not None and 'frames' in window:
        frames = window['frames']
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            num_frames = frames.shape[0] if frames.dim() >= 3 else 1
            if num_frames > 0:
                # Select a random frame, biased towards later frames to see more motion
                if num_frames > 5:
                    # For longer sequences, prefer middle to end frames
                    min_idx = num_frames // 3
                    selected_idx = random.randint(min_idx, num_frames - 1)
                else:
                    selected_idx = random.randint(0, max(0, num_frames - 1))
                selected_generated = frames[selected_idx] if frames.dim() >= 3 else frames
                selected_target = frames[0] if frames.dim() >= 3 else frames
    
    # Fallback to motion_data
    if selected_generated is None and motion_data is not None and 'frames' in motion_data:
        frames = motion_data['frames']
        if isinstance(frames, torch.Tensor) and frames.numel() > 0:
            if frames.dim() > 3:  # [B, T, C, H, W]
                num_frames = frames.shape[1]
                random_idx = random.randint(0, max(0, num_frames - 1))
                selected_generated = frames[0, random_idx]
                selected_target = frames[0, 0]
            elif frames.dim() == 3:  # [C, H, W]
                selected_generated = frames
                selected_target = frames
    
    # Extract random motion parameters (use same index as frame if possible)
    random_motion_params = {}
    motion_frame_idx = selected_idx
    if outputs and 'theta' in outputs:
        if isinstance(outputs['theta'], torch.Tensor) and outputs['theta'].shape[1] > 1:
            # Use the same index as the selected frame, or random if out of bounds
            if motion_frame_idx >= outputs['theta'].shape[1]:
                motion_frame_idx = random.randint(0, outputs['theta'].shape[1] - 1)
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor) and outputs[key].dim() > 1:
                    random_motion_params[key] = outputs[key][:, motion_frame_idx:motion_frame_idx+1]
                else:
                    random_motion_params[key] = outputs[key]
        else:
            random_motion_params = outputs
    
    # Add frame index to motion params for display
    random_motion_params['_frame_idx'] = motion_frame_idx

    # Process identity frame if provided
    selected_identity = None
    if identity_frame is not None and isinstance(identity_frame, torch.Tensor):
        if identity_frame.dim() == 4:  # [B, C, H, W]
            selected_identity = identity_frame[0]
        elif identity_frame.dim() == 3:  # [C, H, W]
            selected_identity = identity_frame
        else:
            selected_identity = identity_frame

    # Create 3-panel thumbnail: Identity | Target | Predicted
    if selected_generated is not None or selected_target is not None or selected_identity is not None:
        return create_three_panel_thumbnail(
            identity_frame=selected_identity,
            target_frame=selected_target,
            predicted_frame=selected_generated,
            motion_params=random_motion_params,
            size=size
        )
    else:
        # Fallback to simple stats thumbnail
        return generate_simple_thumbnail(
            random_motion_params,
            ground_truth=selected_target,
            size=size
        )
