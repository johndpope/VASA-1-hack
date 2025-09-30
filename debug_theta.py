"""
Debug script to analyze theta (head pose) predictions during training
"""
import torch
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def analyze_theta_collapse(
    predicted_motion: Dict[str, torch.Tensor],
    target_motion: Dict[str, torch.Tensor],
    step: int = 0,
    log_to_wandb: bool = True
) -> Dict[str, float]:
    """
    Analyze if theta predictions are collapsing or not learning properly.

    Returns dict of metrics for logging.
    """
    metrics = {}

    if 'theta' not in predicted_motion or 'theta' not in target_motion:
        logger.warning("Theta not found in motion data")
        return metrics

    pred_theta = predicted_motion['theta']  # [B, T, 3, 4]
    target_theta = target_motion['theta']    # [B, T, 3, 4]

    B, T = pred_theta.shape[:2]
    device = pred_theta.device

    # 1. Check variance across time (should be > 0 if head is moving)
    pred_theta_std = torch.std(pred_theta, dim=1).mean().item()  # Std across time
    target_theta_std = torch.std(target_theta, dim=1).mean().item()

    metrics['theta_std_pred'] = pred_theta_std
    metrics['theta_std_target'] = target_theta_std
    metrics['theta_std_ratio'] = pred_theta_std / (target_theta_std + 1e-8)

    # 2. Decompose into rotation and translation
    pred_R = pred_theta[..., :3, :3]     # [B, T, 3, 3] rotation
    pred_t = pred_theta[..., :3, 3]      # [B, T, 3] translation
    target_R = target_theta[..., :3, :3]
    target_t = target_theta[..., :3, 3]

    # 3. Check rotation deviation from identity (head rotation amount)
    I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    pred_rot_deviation = torch.norm(pred_R - I, dim=(-2, -1)).mean().item()
    target_rot_deviation = torch.norm(target_R - I, dim=(-2, -1)).mean().item()

    metrics['rotation_deviation_pred'] = pred_rot_deviation
    metrics['rotation_deviation_target'] = target_rot_deviation

    # 4. Check translation magnitude
    pred_trans_mag = torch.norm(pred_t, dim=-1).mean().item()
    target_trans_mag = torch.norm(target_t, dim=-1).mean().item()

    metrics['translation_mag_pred'] = pred_trans_mag
    metrics['translation_mag_target'] = target_trans_mag

    # 5. Angular error in rotation (more meaningful than MSE)
    # Compute trace to get rotation angle
    R_diff = torch.matmul(pred_R.transpose(-2, -1), target_R)  # [B, T, 3, 3]
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B, T]
    # angle = arccos((trace - 1) / 2)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -0.9999, 0.9999)  # Numerical stability
    angle_error = torch.acos(cos_angle).mean().item()  # In radians
    angle_error_deg = np.degrees(angle_error)

    metrics['rotation_angle_error_rad'] = angle_error
    metrics['rotation_angle_error_deg'] = angle_error_deg

    # 6. Check if theta is constant (collapsed)
    # Compute frame-to-frame differences
    theta_diff = torch.diff(pred_theta, dim=1)  # [B, T-1, 3, 4]
    theta_variation = torch.norm(theta_diff, dim=(-2, -1)).mean().item()

    target_diff = torch.diff(target_theta, dim=1)
    target_variation = torch.norm(target_diff, dim=(-2, -1)).mean().item()

    metrics['theta_temporal_variation_pred'] = theta_variation
    metrics['theta_temporal_variation_target'] = target_variation

    # 7. Diagnosis
    is_collapsed = pred_theta_std < 0.01 and theta_variation < 0.01
    is_under_rotating = pred_rot_deviation < target_rot_deviation * 0.5
    is_static = theta_variation < target_variation * 0.1

    if is_collapsed:
        logger.warning(f"[Step {step}] THETA COLLAPSED! Std: {pred_theta_std:.6f}, Variation: {theta_variation:.6f}")
    elif is_under_rotating:
        logger.warning(f"[Step {step}] UNDER-ROTATING! Pred deviation: {pred_rot_deviation:.4f} vs Target: {target_rot_deviation:.4f}")
    elif is_static:
        logger.warning(f"[Step {step}] STATIC PREDICTIONS! Temporal variation: {theta_variation:.4f} vs Target: {target_variation:.4f}")

    # Log detailed info periodically
    if step % 100 == 0:
        logger.info(f"[Step {step}] Theta Analysis:")
        logger.info(f"  Std - Pred: {pred_theta_std:.4f}, Target: {target_theta_std:.4f}")
        logger.info(f"  Rotation Deviation - Pred: {pred_rot_deviation:.4f}, Target: {target_rot_deviation:.4f}")
        logger.info(f"  Translation Mag - Pred: {pred_trans_mag:.4f}, Target: {target_trans_mag:.4f}")
        logger.info(f"  Angular Error: {angle_error_deg:.2f} degrees")
        logger.info(f"  Temporal Variation - Pred: {theta_variation:.4f}, Target: {target_variation:.4f}")

    # Log to wandb
    if log_to_wandb:
        try:
            import wandb
            wandb_metrics = {f"theta_debug/{k}": v for k, v in metrics.items()}
            wandb_metrics['theta_debug/is_collapsed'] = float(is_collapsed)
            wandb_metrics['theta_debug/is_under_rotating'] = float(is_under_rotating)
            wandb_metrics['theta_debug/is_static'] = float(is_static)
            wandb.log(wandb_metrics, step=step)
        except:
            pass

    return metrics


def check_theta_gradients(model, step: int = 0):
    """
    Check if theta_head is getting gradients
    """
    theta_head_params = []
    for name, param in model.named_parameters():
        if 'theta_head' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            theta_head_params.append((name, grad_norm))

    if theta_head_params:
        avg_grad = np.mean([g for _, g in theta_head_params])
        logger.info(f"[Step {step}] Theta head gradient norms: avg={avg_grad:.6f}")

        if avg_grad < 1e-6:
            logger.warning("THETA HEAD GRADIENTS TOO SMALL! Consider increasing lambda_pose")

        # Log to wandb
        try:
            import wandb
            wandb.log({"theta_debug/theta_head_grad_norm": avg_grad}, step=step)
        except:
            pass
    else:
        logger.warning("No gradients found for theta_head!")


# Usage in training loop:
# from debug_theta import analyze_theta_collapse, check_theta_gradients
#
# # After forward pass:
# metrics = analyze_theta_collapse(predicted_motion, target_motion, step=global_step)
#
# # After backward pass:
# check_theta_gradients(model, step=global_step)