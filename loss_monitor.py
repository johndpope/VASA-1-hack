"""
Loss Range Monitoring for VASA Training
Warns when losses are outside healthy ranges based on empirical observations
"""

from typing import Dict, Optional
import torch
from logger import logger


class LossRangeMonitor:
    """Monitor loss values and warn when outside healthy ranges."""

    # Healthy ranges for well-trained model [min, warning, critical]
    LOSS_RANGES = {
        # Core reconstruction losses
        'theta_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Theta (3DMM shape parameters) reconstruction'
        },
        'rotation_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Head rotation (Euler angles) reconstruction'
        },
        'translation_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Head translation reconstruction'
        },
        'scale_loss': {
            'healthy': (0.0001, 0.01),
            'warning': 0.05,
            'critical': 0.1,
            'description': '3D scale reconstruction'
        },
        'expression_loss': {
            'healthy': (0.01, 0.2),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Expression embedding reconstruction'
        },
        'expression_mse': {
            'healthy': (0.01, 0.2),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Expression MSE loss'
        },
        'expression_variance_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Expression variance matching (prevents collapse)'
        },
        'expression_temporal_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Expression temporal variation'
        },

        # Warp losses
        'uv_warp_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 5.0,
            'description': '3D UV warp field reconstruction'
        },
        'uv_warp_magnitude': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 1.0,
            'description': 'UV warp magnitude matching (collapse prevention)'
        },
        'xy_warp_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 5.0,
            'description': '2D XY warp reconstruction'
        },
        'rigid_warp_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 5.0,
            'description': 'Rigid warp reconstruction'
        },

        # Lip sync losses
        'lips_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Lip landmark matching'
        },
        'audio_lip_correlation': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Audio-lip motion correlation'
        },
        'audio_expr_coupling': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Audio-expression magnitude coupling'
        },

        # Control losses
        'control_gaze': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 1.0,
            'description': 'Gaze direction control'
        },
        'control_distance': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 1.0,
            'description': 'Head distance control'
        },
        'control_emotion': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 1.0,
            'description': 'Emotion control (valence/arousal)'
        },
        'control_blink': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Blink pattern control (total loss)'
        },

        # Blink sub-losses
        'blink_openness_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Eye openness matching (left/right eye 0-1)'
        },
        'blink_phase_loss': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Blink phase matching (open/closing/closed/opening)'
        },
        'blink_phase_accuracy': {
            'healthy': (0.7, 1.0),
            'warning': 0.5,
            'critical': 0.3,
            'description': 'Blink phase classification accuracy (should be high)'
        },
        'blink_openness_mae': {
            'healthy': (0.01, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Mean absolute error on eye openness'
        },

        # Mouth openness
        'mouth_openness_direct': {
            'healthy': (0.01, 0.2),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Direct mouth openness to audio supervision'
        },

        # Aggregated losses
        'reconstruction': {
            'healthy': (0.1, 1.0),
            'warning': 2.0,
            'critical': 5.0,
            'description': 'Total reconstruction loss'
        },
        'pose_loss': {
            'healthy': (0.01, 0.2),
            'warning': 0.5,
            'critical': 2.0,
            'description': 'Combined pose losses'
        },
        'dynamics_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Combined dynamics losses'
        },
        'total': {
            'healthy': (0.5, 3.0),
            'warning': 5.0,
            'critical': 10.0,
            'description': 'Total weighted loss'
        }
    }

    def __init__(self, enable_warnings: bool = True, enable_critical: bool = True):
        """
        Initialize loss monitor.

        Args:
            enable_warnings: Log warning when loss exceeds warning threshold
            enable_critical: Log critical error when loss exceeds critical threshold
        """
        self.enable_warnings = enable_warnings
        self.enable_critical = enable_critical
        self.warning_counts = {}
        self.critical_counts = {}

    def check_loss(
        self,
        loss_name: str,
        loss_value: float,
        step: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Check if loss is within healthy range.

        Args:
            loss_name: Name of the loss to check
            loss_value: Current loss value
            step: Optional training step for logging

        Returns:
            Dict with status info: {
                'status': 'healthy' | 'warning' | 'critical' | 'unknown',
                'message': str,
                'value': float,
                'range': tuple
            }
        """
        if loss_name not in self.LOSS_RANGES:
            return {
                'status': 'unknown',
                'message': f'No monitoring range defined for {loss_name}',
                'value': loss_value,
                'range': None
            }

        config = self.LOSS_RANGES[loss_name]
        healthy_min, healthy_max = config['healthy']
        warning_thresh = config['warning']
        critical_thresh = config['critical']
        description = config['description']

        # Determine status
        if loss_value < healthy_min:
            status = 'too_low'
            message = (
                f"⚠️ {loss_name} too low: {loss_value:.6f} "
                f"(healthy range: {healthy_min:.3f} - {healthy_max:.3f})\n"
                f"   Description: {description}\n"
                f"   Possible issue: Loss weight may be too low or feature not learning"
            )
        elif loss_value <= healthy_max:
            status = 'healthy'
            message = f"✅ {loss_name}: {loss_value:.6f} (healthy)"
        elif loss_value <= warning_thresh:
            status = 'warning'
            message = (
                f"⚠️ {loss_name} elevated: {loss_value:.6f} "
                f"(healthy max: {healthy_max:.3f}, warning: {warning_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"   Monitor: May need more training or hyperparameter adjustment"
            )
            self.warning_counts[loss_name] = self.warning_counts.get(loss_name, 0) + 1
        elif loss_value <= critical_thresh:
            status = 'high_warning'
            message = (
                f"🔶 {loss_name} HIGH: {loss_value:.6f} "
                f"(warning: {warning_thresh:.3f}, critical: {critical_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"   Action needed: Check loss weight, learning rate, or training stability"
            )
            self.warning_counts[loss_name] = self.warning_counts.get(loss_name, 0) + 1
        else:
            status = 'critical'
            message = (
                f"🔴 CRITICAL: {loss_name} = {loss_value:.6f} "
                f"(critical threshold: {critical_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"   URGENT: Loss not converging! Check:\n"
                f"     - Loss weight (may be too high)\n"
                f"     - Learning rate (may be too high/low)\n"
                f"     - Data quality (check for corrupted samples)\n"
                f"     - Gradient flow (check for vanishing/exploding gradients)"
            )
            self.critical_counts[loss_name] = self.critical_counts.get(loss_name, 0) + 1

        # Log warnings
        if status == 'too_low':
            if self.enable_warnings:
                logger.debug(message)
        elif status == 'healthy':
            # Only log healthy on first occurrence or every 1000 steps
            if step is None or step % 1000 == 0:
                logger.debug(message)
        elif status == 'warning':
            if self.enable_warnings:
                logger.warning(message)
        elif status == 'high_warning':
            if self.enable_warnings:
                logger.warning(message)
        elif status == 'critical':
            if self.enable_critical:
                logger.error(message)

        return {
            'status': status,
            'message': message,
            'value': loss_value,
            'range': config['healthy'],
            'warning': warning_thresh,
            'critical': critical_thresh
        }

    def check_losses(
        self,
        losses: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        log_summary: bool = False
    ) -> Dict[str, Dict]:
        """
        Check multiple losses at once.

        Args:
            losses: Dict of loss_name -> loss_value (can be tensors or floats)
            step: Optional training step for logging
            log_summary: Log summary of all warnings/criticals

        Returns:
            Dict of loss_name -> status_dict
        """
        results = {}
        warnings = []
        criticals = []

        for loss_name, loss_value in losses.items():
            # Convert tensor to float
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()

            # Skip non-numeric values
            if not isinstance(loss_value, (int, float)):
                continue

            result = self.check_loss(loss_name, loss_value, step)
            results[loss_name] = result

            if result['status'] in ['warning', 'high_warning']:
                warnings.append(loss_name)
            elif result['status'] == 'critical':
                criticals.append(loss_name)

        # Log summary if requested
        if log_summary and (warnings or criticals):
            summary_msg = f"\n{'='*80}\n"
            summary_msg += f"LOSS MONITORING SUMMARY (Step {step})\n"
            summary_msg += f"{'='*80}\n"

            if criticals:
                summary_msg += f"\n🔴 CRITICAL LOSSES ({len(criticals)}):\n"
                for name in criticals:
                    summary_msg += f"  - {name}: {results[name]['value']:.6f} "
                    summary_msg += f"(critical > {results[name]['critical']:.3f})\n"

            if warnings:
                summary_msg += f"\n⚠️ WARNING LOSSES ({len(warnings)}):\n"
                for name in warnings:
                    summary_msg += f"  - {name}: {results[name]['value']:.6f} "
                    summary_msg += f"(warning > {results[name]['warning']:.3f})\n"

            summary_msg += f"\n{'='*80}\n"

            if criticals:
                logger.error(summary_msg)
            else:
                logger.warning(summary_msg)

        return results

    def get_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about warnings/criticals seen.

        Returns:
            Dict with warning/critical counts
        """
        return {
            'warning_counts': self.warning_counts.copy(),
            'critical_counts': self.critical_counts.copy(),
            'total_warnings': sum(self.warning_counts.values()),
            'total_criticals': sum(self.critical_counts.values())
        }

    def reset_statistics(self):
        """Reset warning/critical counts."""
        self.warning_counts = {}
        self.critical_counts = {}
