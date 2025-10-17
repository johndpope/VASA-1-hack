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
        'expression_cosine': {
            'healthy': (0.0, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Direct GT expression cosine similarity loss (1.0 - cosine_sim). Low loss = high similarity = good GT matching'
        },

        # Lip sync losses
        'lips_total': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Total lip loss (position + velocity)'
        },
        'lips_pos_loss': {
            'healthy': (0.01, 0.3),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Lip position/landmark loss'
        },
        'lips_vel_loss': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Lip velocity/motion smoothness loss'
        },
        'nonlip_total': {
            'healthy': (0.01, 0.3),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Non-lip facial motion total (eyes, nose, jaw)'
        },
        'facial_motion_total': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Total facial motion loss (lips + nonlip)'
        },
        'audio_lip_correlation': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Audio-lip motion correlation'
        },
        'audio_expression_coupling': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Audio-expression magnitude coupling'
        },
        'sync_loss': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Audio-visual synchronization (SyncNet/Synchformer)'
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
        'control_speed': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Motion speed control'
        },
        'control_total': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Total control loss (all control signals)'
        },
        'speed_loss': {
            'healthy': (0.001, 0.1),
            'warning': 0.2,
            'critical': 0.5,
            'description': 'Speed prediction loss'
        },
        'speed_accuracy': {
            'healthy': (0.7, 1.0),
            'warning': 0.5,
            'critical': 0.3,
            'description': 'Speed bucket classification accuracy (should be high)'
        },

        # Blink sub-losses (from _compute_blink_loss)
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

        # Disentanglement losses
        'disentangle_total': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Total disentanglement loss (consist + cross_id)'
        },
        'l_consist': {
            'healthy': (0.01, 0.3),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Consistency loss for disentanglement'
        },
        'l_cross_id': {
            'healthy': (0.01, 0.3),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Cross-identity similarity loss'
        },

        # Flow-DPO losses (VideoReward framework - Liu et al., 2025)
        'flow_dpo': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Flow-DPO preference-based alignment loss'
        },
        'regret_w': {
            'healthy': (-0.5, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Regret for preferred (ground-truth) samples - should be near 0'
        },
        'regret_l': {
            'healthy': (0.0, 1.0),
            'warning': 2.0,
            'critical': 4.0,
            'description': 'Regret for dispreferred (noisy) samples - should be positive'
        },
        'regret_diff': {
            'healthy': (-0.5, 1.0),
            'warning': 2.0,
            'critical': 4.0,
            'description': 'Regret difference (advantage) - preferred over dispreferred'
        },

        # Regularization losses
        'velocity_smoothness': {
            'healthy': (0.001, 0.05),
            'warning': 0.1,
            'critical': 0.5,
            'description': 'Motion velocity smoothness'
        },
        'perceptual': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Perceptual loss (VGG features)'
        },
        'mouth_perceptual': {
            'healthy': (0.1, 5.0),
            'warning': 10.0,
            'critical': 20.0,
            'description': 'Mouth-focused perceptual loss (LPIPS weighted 100x on mouth region, TalkVid style)'
        },
        'verification': {
            'healthy': (0.01, 0.5),
            'warning': 1.0,
            'critical': 2.0,
            'description': 'Face verification/identity loss'
        },

        # Deprecated/monitoring only (commented losses in code)
        'expression_std': {
            'healthy': (0.01, 0.2),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Expression standard deviation (monitoring only)'
        },
        'motion_diversity': {
            'healthy': (0.01, 0.3),
            'warning': 0.5,
            'critical': 1.0,
            'description': 'Motion diversity loss (currently disabled)'
        },

        # Aggregated losses
        'reconstruction': {
            'healthy': (0.1, 1.0),
            'warning': 2.0,
            'critical': 5.0,
            'description': 'Total reconstruction loss'
        },
        'total': {
            'healthy': (0.5, 3.0),
            'warning': 5.0,
            'critical': 10.0,
            'description': 'Total weighted loss'
        }
    }

    # Mapping from loss names to their config lambda parameter names
    LOSS_TO_LAMBDA = {
        'reconstruction': 'lambda_reconstruction',
        'dynamics_loss': 'lambda_dynamics',
        'expression_cosine': 'lambda_expression_cosine',
        'theta_loss': 'lambda_pose',
        'scale_loss': 'lambda_scale',
        'rotation_loss': 'lambda_rotation',
        'translation_loss': 'lambda_translation',
        'expression_loss': 'lambda_dynamics',  # Part of dynamics
        'expression_variance_loss': 'lambda_expression_variance',
        'expression_temporal_loss': 'lambda_expression_temporal',
        'lips_total': 'lambda_lips',
        'lips_pos_loss': 'lambda_lips',
        'lips_vel_loss': 'lambda_lips',
        'nonlip_total': 'lambda_nonlip',
        'facial_motion_total': 'lambda_control',
        'audio_lip_correlation': 'lambda_audio_lip',
        'audio_expression_coupling': 'lambda_audio_expr_coupling',
        'sync_loss': 'lambda_sync',
        'control_gaze': 'lambda_gaze_direction',
        'control_distance': 'lambda_head_distance',
        'control_emotion': 'lambda_emotion',
        'control_blink': 'lambda_blink',
        'control_speed': 'lambda_speed',
        'control_total': 'lambda_control',
        'blink_openness_loss': 'lambda_blink',
        'blink_phase_loss': 'lambda_blink',
        'mouth_openness_direct': 'lambda_mouth_openness',
        'disentangle_total': 'lambda_consistency',
        'l_consist': 'lambda_consist',
        'l_cross_id': 'lambda_cross_id',
        'flow_dpo': 'lambda_flow_dpo',
        'velocity_smoothness': 'lambda_velocity',
        'perceptual': 'lambda_perceptual',
        'mouth_perceptual': 'lambda_mouth_perceptual',
        'verification': 'lambda_verification',
    }

    def __init__(self, enable_warnings: bool = True, enable_critical: bool = True, history_size: int = 50, config=None):
        """
        Initialize loss monitor.

        Args:
            enable_warnings: Log warning when loss exceeds warning threshold
            enable_critical: Log critical error when loss exceeds critical threshold
            history_size: Number of recent loss values to track for trend visualization
            config: Optional config object (OmegaConf) to extract lambda weights from
        """
        self.enable_warnings = enable_warnings
        self.enable_critical = enable_critical
        self.warning_counts = {}
        self.critical_counts = {}
        self.history_size = history_size
        self.loss_history = {}  # Dict[loss_name, List[float]]
        self.config = config  # Store config for lambda weight lookups

    def _get_lambda_weight(self, loss_name: str) -> Optional[float]:
        """Get the lambda weight for a loss from config."""
        if self.config is None or loss_name not in self.LOSS_TO_LAMBDA:
            return None

        lambda_name = self.LOSS_TO_LAMBDA[loss_name]
        try:
            # Config is OmegaConf, access via loss.lambda_X
            if hasattr(self.config, 'loss') and hasattr(self.config.loss, lambda_name.replace('lambda_', '')):
                return getattr(self.config.loss, lambda_name.replace('lambda_', ''))
        except:
            pass

        return None

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
            # Get lambda weight if available
            lambda_weight = self._get_lambda_weight(loss_name)
            lambda_info = ""
            if lambda_weight is not None:
                lambda_param = self.LOSS_TO_LAMBDA.get(loss_name, "")
                lambda_info = f"   Config weight: {lambda_param} = {lambda_weight}\n"

            message = (
                f"⚠️ {loss_name} elevated: {loss_value:.6f} "
                f"(healthy max: {healthy_max:.3f}, warning: {warning_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"{lambda_info}"
                f"   Monitor: May need more training or hyperparameter adjustment"
            )
            self.warning_counts[loss_name] = self.warning_counts.get(loss_name, 0) + 1
        elif loss_value <= critical_thresh:
            status = 'high_warning'
            # Get lambda weight if available
            lambda_weight = self._get_lambda_weight(loss_name)
            lambda_info = ""
            if lambda_weight is not None:
                lambda_param = self.LOSS_TO_LAMBDA.get(loss_name, "")
                lambda_info = f"   Config weight: {lambda_param} = {lambda_weight}\n"

            message = (
                f"🔶 {loss_name} HIGH: {loss_value:.6f} "
                f"(warning: {warning_thresh:.3f}, critical: {critical_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"{lambda_info}"
                f"   Action needed: Check loss weight, learning rate, or training stability"
            )
            self.warning_counts[loss_name] = self.warning_counts.get(loss_name, 0) + 1
        else:
            status = 'critical'
            # Get lambda weight if available
            lambda_weight = self._get_lambda_weight(loss_name)
            lambda_info = ""
            if lambda_weight is not None:
                lambda_param = self.LOSS_TO_LAMBDA.get(loss_name, "")
                lambda_info = f"   Config weight: {lambda_param} = {lambda_weight}\n"

            message = (
                f"🔴 CRITICAL: {loss_name} = {loss_value:.6f} "
                f"(critical threshold: {critical_thresh:.3f})\n"
                f"   Description: {description}\n"
                f"{lambda_info}"
                f"   URGENT: Loss not converging! Check:\n"
                f"     - Loss weight (may be too high)\n"
                f"     - Learning rate (may be too high/low)\n"
                f"     - Data quality (check for corrupted samples)\n"
                f"     - Gradient flow (check for vanishing/exploding gradients)"
            )
            self.critical_counts[loss_name] = self.critical_counts.get(loss_name, 0) + 1

        # Track loss history for visualization
        self._track_loss_history(loss_name, loss_value)

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

    def _track_loss_history(self, loss_name: str, loss_value: float):
        """Track loss value in history for trend visualization."""
        if loss_name not in self.loss_history:
            self.loss_history[loss_name] = []

        self.loss_history[loss_name].append(loss_value)

        # Keep only recent history
        if len(self.loss_history[loss_name]) > self.history_size:
            self.loss_history[loss_name] = self.loss_history[loss_name][-self.history_size:]

    def _generate_ascii_graph(self, loss_name: str, width: int = 60, height: int = 8) -> str:
        """
        Generate ASCII line graph showing loss trend toward target.

        Args:
            loss_name: Name of loss to visualize
            width: Graph width in characters
            height: Graph height in lines

        Returns:
            ASCII art string showing the graph
        """
        if loss_name not in self.loss_history or len(self.loss_history[loss_name]) < 2:
            return f"No history for {loss_name}"

        if loss_name not in self.LOSS_RANGES:
            return f"No range defined for {loss_name}"

        history = self.loss_history[loss_name]
        config = self.LOSS_RANGES[loss_name]
        healthy_min, healthy_max = config['healthy']
        warning_thresh = config['warning']
        critical_thresh = config['critical']

        # Determine value range for y-axis
        min_val = min(history + [healthy_min])
        max_val = max(history + [critical_thresh])

        # Add 10% padding
        range_padding = (max_val - min_val) * 0.1
        y_min = max(0, min_val - range_padding)
        y_max = max_val + range_padding

        # Initialize graph grid
        graph = [[' ' for _ in range(width)] for _ in range(height)]

        # Draw target zones as background
        def value_to_y(val):
            if y_max == y_min:
                return height // 2
            normalized = (val - y_min) / (y_max - y_min)
            return int((1 - normalized) * (height - 1))

        # Mark healthy zone
        healthy_min_y = value_to_y(healthy_min)
        healthy_max_y = value_to_y(healthy_max)
        for y in range(height):
            if healthy_max_y <= y <= healthy_min_y:
                for x in range(width):
                    if graph[y][x] == ' ':
                        graph[y][x] = '░'

        # Draw loss trend line
        points_per_x = max(1, len(history) / width)
        for x in range(width):
            idx = int(x * points_per_x)
            if idx < len(history):
                val = history[idx]
                y = value_to_y(val)
                if 0 <= y < height:
                    # Use different symbols based on status
                    if val <= healthy_max:
                        graph[y][x] = '●'  # Healthy
                    elif val <= warning_thresh:
                        graph[y][x] = '◆'  # Warning
                    else:
                        graph[y][x] = '■'  # Critical

        # Build graph string
        lines = []
        lines.append(f"\n{loss_name} Trend (last {len(history)} samples)")
        lines.append(f"{'─' * (width + 10)}")

        # Y-axis labels and graph
        for y in range(height):
            # Calculate y-axis value
            if height > 1:
                y_val = y_max - (y / (height - 1)) * (y_max - y_min)
            else:
                y_val = (y_max + y_min) / 2

            # Format y-axis label
            label = f"{y_val:6.3f}"

            # Mark threshold lines
            threshold_marker = ""
            if abs(y_val - critical_thresh) < (y_max - y_min) * 0.05:
                threshold_marker = " 🔴 CRITICAL"
            elif abs(y_val - warning_thresh) < (y_max - y_min) * 0.05:
                threshold_marker = " ⚠️  WARNING"
            elif abs(y_val - healthy_max) < (y_max - y_min) * 0.05:
                threshold_marker = " ✅ TARGET"

            line = label + " │" + ''.join(graph[y]) + threshold_marker
            lines.append(line)

        # X-axis
        lines.append("       └" + "─" * width)
        lines.append(f"        {'oldest':<{width//2}}{'latest':>{width//2}}")

        # Current value and status
        current = history[-1]
        if current <= healthy_max:
            status = "✅ HEALTHY"
        elif current <= warning_thresh:
            status = "⚠️  WARNING"
        elif current <= critical_thresh:
            status = "🔶 HIGH"
        else:
            status = "🔴 CRITICAL"

        lines.append(f"\nCurrent: {current:.6f} {status}")
        lines.append(f"Target range: {healthy_min:.3f} - {healthy_max:.3f}")
        lines.append(f"Legend: ● healthy  ◆ warning  ■ critical  ░ target zone")

        return '\n'.join(lines)

    def visualize_loss(self, loss_name: str) -> str:
        """
        Generate visualization for a specific loss.

        Args:
            loss_name: Name of loss to visualize

        Returns:
            ASCII art visualization
        """
        return self._generate_ascii_graph(loss_name)

    def visualize_summary(
        self,
        loss_names: Optional[list] = None,
        show_graphs: bool = True,
        graph_width: int = 50,
        graph_height: int = 6
    ) -> str:
        """
        Generate summary visualization for multiple losses.

        Args:
            loss_names: List of loss names to visualize (None = all with warnings/criticals)
            show_graphs: Whether to include ASCII graphs
            graph_width: Width of each graph
            graph_height: Height of each graph

        Returns:
            Formatted summary string
        """
        # Determine which losses to visualize
        if loss_names is None:
            # Show losses that have warnings or criticals
            loss_names = []
            for name in self.loss_history.keys():
                if name in self.warning_counts or name in self.critical_counts:
                    loss_names.append(name)

            # If none have warnings, show all tracked losses
            if not loss_names:
                loss_names = list(self.loss_history.keys())

        if not loss_names:
            return "No loss history to visualize"

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("LOSS VISUALIZATION SUMMARY")
        lines.append("=" * 80)

        for loss_name in loss_names:
            if loss_name not in self.loss_history:
                continue

            history = self.loss_history[loss_name]
            if len(history) < 2:
                continue

            # Get current status
            current = history[-1]
            if loss_name in self.LOSS_RANGES:
                config = self.LOSS_RANGES[loss_name]
                healthy_min, healthy_max = config['healthy']
                warning_thresh = config['warning']

                if current <= healthy_max:
                    status = "✅ HEALTHY"
                elif current <= warning_thresh:
                    status = "⚠️  WARNING"
                else:
                    status = "🔴 CRITICAL"

                # Compute trend
                if len(history) >= 10:
                    recent_avg = sum(history[-10:]) / 10
                    older_avg = sum(history[-20:-10]) / 10 if len(history) >= 20 else recent_avg
                    if recent_avg < older_avg * 0.95:
                        trend = "📉 IMPROVING"
                    elif recent_avg > older_avg * 1.05:
                        trend = "📈 WORSENING"
                    else:
                        trend = "➡️  STABLE"
                else:
                    trend = "➡️  TRACKING"

                lines.append(f"\n{loss_name}")
                lines.append(f"  Status: {status}  |  Trend: {trend}")
                lines.append(f"  Current: {current:.6f}  |  Target: {healthy_min:.3f} - {healthy_max:.3f}")

                # Show ASCII graph if requested
                if show_graphs:
                    graph = self._generate_ascii_graph(loss_name, width=graph_width, height=graph_height)
                    # Indent graph
                    for line in graph.split('\n'):
                        lines.append("  " + line)

        lines.append("\n" + "=" * 80)

        return '\n'.join(lines)
