"""Motion sequence handler for VASA model."""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import traceback
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger


class MotionSequenceHandler:
    """Handles motion sequence processing for VASA."""
    def __init__(
        self,
        window_size: int = 50,       # Main sequence length (T)
        stride: int = 25,            # Window stride
        context_size: int = 10,      # Context length (K)
    ):
        self.window_size = window_size
        self.stride = stride
        self.context_size = context_size

        self.overlap_size = window_size - stride
        
        logger.info(f"Initialized MotionSequenceHandler:")
        logger.info(f"  window_size: {window_size}")
        logger.info(f"  stride: {stride}")
        logger.info(f"  context_size: {context_size}")

        logger.info(f"  overlap_size: {self.overlap_size}")


    def process_batch(self, batch: Dict[str, torch.Tensor], current_window_size: Optional[int] = None) -> List[Dict]:
        """Process batch into overlapping windows with adjusted validation."""
        try:
            # Use current window size or default
            window_size = current_window_size or self.window_size
            
            B = batch['frames'].shape[0]  # Get batch size
            T = batch['frames'].shape[1]  # Get sequence length
            logger.debug(f"Processing batch: B={B}, T={T}, window_size={window_size}")

            # Adjusted minimum frames - make context optional
            min_frames = window_size  # Just require window size
            
            if T < min_frames:
                logger.warning(
                    f"Sequence too short: {T} frames, need minimum {min_frames}\n"
                    f"window_size={window_size}, self.window_size={self.window_size}\n"
                    f"Config check: {hasattr(self, 'window_size')}"
                )
                return []

            windows = []
            stride = self.stride

            # Process each batch item
            for b in range(B):
                # Adjust window count calculation
                n_windows = 1  # Just one window per sequence if T == window_size
                if T > window_size:
                    n_windows = max(1, (T - window_size) // stride + 1)
                    
                logger.debug(f"Batch {b}: Creating {n_windows} windows")

                for window_idx in range(n_windows):
                    start_frame = window_idx * stride
                    end_frame = start_frame + window_size

                    if end_frame > T:
                        logger.debug(f"Window {window_idx} would exceed sequence length - breaking")
                        break

                    # Create window data preserving batch dimension
                    window_data = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            # Handle different tensor shapes
                            if key == 'audio_features' and len(value.shape) == 4:  # [B, 1, T, D]
                                window_data[key] = value[b:b+1, :, start_frame:end_frame]
                            else:
                                # Remove extra dims if present
                                tensor = value
                                if len(tensor.shape) > 3 and tensor.shape[1] == 1:
                                    tensor = tensor.squeeze(1)
                                window_data[key] = tensor[b:b+1, start_frame:end_frame]
                        elif key == 'lip_metrics' and isinstance(value, dict):
                            # Handle lip_metrics dictionary
                            window_data['lip_metrics'] = {}
                            for metric_key, metric_tensor in value.items():
                                if isinstance(metric_tensor, torch.Tensor):
                                    # Extract the window for this metric
                                    window_data['lip_metrics'][metric_key] = metric_tensor[b:b+1, start_frame:end_frame]
                            logger.debug(f"Added lip_metrics to window with keys: {list(window_data['lip_metrics'].keys())}")

                    # Add metadata
                    window_data['metadata'] = {
                        'batch_idx': b,
                        'window_idx': window_idx,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'has_context': window_idx > 0,
                        'total_windows': n_windows,
                        'window_size': window_size
                    }

                    windows.append(window_data)
                    logger.debug(f"  Window {window_idx}: {start_frame} -> {end_frame}")

                    # Log tensor shapes for debugging
                    logger.debug(f"\nWindow {window_idx} tensor shapes:")
                    for k, v in window_data.items():
                        if isinstance(v, torch.Tensor):
                            logger.debug(f"  {k}: {v.shape}")
                        elif k == 'lip_metrics' and isinstance(v, dict):
                            for metric_key, metric_tensor in v.items():
                                if isinstance(metric_tensor, torch.Tensor):
                                    logger.debug(f"  lip_metrics[{metric_key}]: {metric_tensor.shape}")

            if windows:
                logger.info(
                    f"Created {len(windows)} windows\n"
                    f"  Window size: {window_size}\n"
                    f"  First window: 0 -> {window_size}"
                )

            return windows

        except Exception as e:
            logger.error(f"Error in process_batch: {str(e)}")
            logger.error(traceback.format_exc())
            return []
            
    def prepare_motion_data(self, window: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare motion data from window matching H5 cache structure.

        NOTE: SRT (scale/rotation/translation) restored - model predicts theta + expression + SRT.
        SRT values are used for direct supervision against ground truth from EMO.
        """
        # UV warps are optional - if not cached, use zeros (will be derived from Nemo networks)
        if 'uv_warps' not in window:
            logger.debug("UV warps not in cache, using zeros (will be derived from Nemo)")
            B, T = window['theta'].shape[0], window['theta'].shape[1]
            device = window['theta'].device
            uv_warps = torch.zeros((B, T, 16, 64, 64, 3), device=device)
        else:
            uv_warps = window['uv_warps']

        motion_data = {
            'theta': window['theta'],            # [B, T, 3, 4] - pose matrix
            'expression_embed': window['expression_embed'],  # [B, T, 128] - aligned expression
            'uv_warps': uv_warps,      # [B, T, 16, 64, 64, 3] - UV warps (zeros if not cached, derived during forward)
        }

        # Add SRT ground truth values if available (for direct supervision)
        if 'scale' in window:
            motion_data['scale'] = window['scale']  # [B, T, 3] - GT scale from EMO
        if 'rotation' in window:
            motion_data['rotation'] = window['rotation']  # [B, T, 3] - GT rotation from EMO
        if 'translation' in window:
            motion_data['translation'] = window['translation']  # [B, T, 3] - GT translation from EMO

        # Include audio features for sync loss and other audio-related losses
        if 'audio_features' in window:
            motion_data['audio_features'] = window['audio_features']  # wav2vec features [B, T, 768]
        if 'audio_mel_spec' in window:
            motion_data['audio_mel_spec'] = window['audio_mel_spec']  # Mel spectrogram [B, T, 128] - for Synchformer
        if 'audio_mfcc' in window:
            motion_data['audio_mfcc'] = window['audio_mfcc']  # MFCC features [B, T, 13]
        # Legacy key support
        if 'mfcc' in window:
            motion_data['mfcc'] = window['mfcc']

      
        return motion_data  

    def merge_windows(self, windows, total_frames, device):
        """Merge overlapping motion sequence windows (SRT removed)."""
        # Initialize output tensors (only theta + expression)
        merged_sequence = {
            'theta': torch.zeros((1, total_frames, 3, 4), device=device),
            'expression_embed': torch.zeros((1, total_frames, 128), device=device)  # Assuming embed dim is 128
        }
        
        # Initialize weight buffer for blending
        weights = torch.zeros(total_frames, device=device)
        
        # Process each window
        for window_data in windows:
            motion_data = window_data['frames']
            overlap_info = window_data['overlap_info']
            
            start_idx = overlap_info['overlap_start']
            end_idx = overlap_info['overlap_end']
            is_first = overlap_info['is_first']
            is_last = overlap_info['is_last']
            
            # Calculate blend weights
            window_length = end_idx - start_idx
            if is_first:
                window_weights = torch.ones(window_length, device=device)
            elif is_last:
                window_weights = torch.linspace(0, 1, window_length, device=device)
            else:
                window_weights = torch.linspace(0, 1, window_length, device=device)
            
            # Update weights buffer
            weights[start_idx:end_idx] += window_weights
            
            # Add motion sequences with weights
            window_slice = slice(start_idx, end_idx)
            merged_sequence['theta'][:, window_slice] += motion_data['theta'][:, :window_length] * window_weights.view(1, -1, 1, 1)
            merged_sequence['scale'][:, window_slice] += motion_data['scale'][:, :window_length] * window_weights.view(1, -1, 1)

            merged_sequence['rotation'][:, window_slice] += motion_data['rotation'][:, :window_length] * window_weights.view(1, -1, 1)
            merged_sequence['translation'][:, window_slice] += motion_data['translation'][:, :window_length] * window_weights.view(1, -1, 1)
            merged_sequence['expression_embed'][:, window_slice] += motion_data['expression_embed'][:, :window_length] * window_weights.view(1, -1, 1)
        
        # Normalize by weights
        weights = weights.clamp(min=1e-8)  # Avoid division by zero
        merged_sequence['theta'] /= weights.view(1, -1, 1, 1)
        merged_sequence['rotation'] /= weights.view(1, -1, 1)
        merged_sequence['scale'] /= weights.view(1, -1, 1)
        merged_sequence['translation'] /= weights.view(1, -1, 1)
        merged_sequence['expression_embed'] /= weights.view(1, -1, 1)
        
        return merged_sequence

        

    def _merge_theta_matrices(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        weights: torch.Tensor,
        weight_accumulator: torch.Tensor
    ):
        """Specially handle merging of theta (transformation) matrices."""
        # Split rotation and translation
        R1 = target[..., :3, :3]
        t1 = target[..., :3, 3]
        R2 = source[..., :3, :3]
        t2 = source[..., :3, 3]

        # Convert rotations to quaternions for proper interpolation
        q1 = self._matrix_to_quaternion(R1)
        q2 = self._matrix_to_quaternion(R2)

        # Perform SLERP
        dot_product = (q1 * q2).sum(-1)
        q2_adj = torch.where(dot_product < 0, -q2, q2)  # Ensure shortest path
        omega = torch.acos((q1 * q2_adj).sum(-1).clamp(-1, 1))
        sin_omega = torch.sin(omega)

        # Handle small angle case
        mask = sin_omega > 1e-6
        q_interp = torch.where(
            mask.unsqueeze(-1),
            (torch.sin((1 - weights) * omega) / sin_omega).unsqueeze(-1) * q1 +
            (torch.sin(weights * omega) / sin_omega).unsqueeze(-1) * q2_adj,
            q1 + weights.unsqueeze(-1) * (q2_adj - q1)
        )

        # Convert back to rotation matrices
        R_interp = self._quaternion_to_matrix(q_interp)

        # Linear interpolation for translation
        t_interp = t1 + weights.unsqueeze(-1) * (t2 - t1)

        # Update target
        target[..., :3, :3] = R_interp
        target[..., :3, 3] = t_interp
        weight_accumulator += weights

    def _normalize_theta_matrices(
        self,
        theta_matrices: torch.Tensor,
        weights: torch.Tensor
    ):
        """Normalize merged theta matrices ensuring valid rotations."""
        # Handle rotation part
        R = theta_matrices[..., :3, :3]
        U, _, V = torch.svd(R)
        R_normalized = torch.matmul(U, V.transpose(-2, -1))

        # Normalize translation part
        t = theta_matrices[..., :3, 3] / (weights.unsqueeze(-1) + 1e-8)

        # Reconstruct normalized matrices
        theta_matrices[..., :3, :3] = R_normalized
        theta_matrices[..., :3, 3] = t
        theta_matrices[..., 3, 3] = 1.0

    def _get_blend_weights(
        self, 
        window_idx: int, 
        total_windows: int,
        start_idx: int,
        window_size: int
    ) -> torch.Tensor:
        """Calculate blending weights for window transitions."""
        if window_idx == 0:  # First window
            weights = torch.cat([
                torch.ones(start_idx + self.stride),
                torch.linspace(1, 0, self.overlap_size)
            ])
        elif window_idx == total_windows - 1:  # Last window
            weights = torch.cat([
                torch.linspace(0, 1, self.overlap_size),
                torch.ones(window_size - self.overlap_size)
            ])
        else:  # Middle windows
            weights = torch.cat([
                torch.linspace(0, 1, self.overlap_size),
                torch.ones(self.stride),
                torch.linspace(1, 0, self.overlap_size)
            ])
            
        return weights

    def _interpolate_rotations(
        self,
        rot1: torch.Tensor,
        rot2: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate rotation matrices using SLERP."""
        # Convert to quaternions
        quat1 = self._matrix_to_quaternion(rot1)
        quat2 = self._matrix_to_quaternion(rot2)
        
        # Compute dot product
        dot = torch.sum(quat1 * quat2, dim=-1, keepdim=True)
        
        # If dot < 0, negate one of the inputs to take shorter interpolation path
        flip_mask = (dot < 0).float()
        quat2 = quat2 * (1 - 2 * flip_mask)
        
        # SLERP interpolation
        theta = torch.acos(torch.clamp(dot, -1, 1))
        sin_theta = torch.sin(theta)
        
        # Handle small angle case
        mask = (sin_theta > 1e-6).float()
        t = weights.unsqueeze(-1)
        
        interpolated = torch.zeros_like(quat1)
        interpolated = mask * (
            torch.sin((1-t) * theta) / sin_theta * quat1 +
            torch.sin(t * theta) / sin_theta * quat2
        ) + (1 - mask) * (quat1 + t * (quat2 - quat1))
        
        # Convert back to rotation matrix
        return self._quaternion_to_matrix(interpolated)

    def _matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert batch of 3x3 rotation matrices to quaternions."""
        m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
        m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
        m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
        
        trace = m00 + m11 + m22
        
        def when_trace_positive():
            r = torch.sqrt(1 + trace)
            s = 0.5 / r
            return torch.stack([
                0.5 * r,
                (m21 - m12) * s,
                (m02 - m20) * s,
                (m10 - m01) * s
            ], dim=-1)
            
        def when_m00_largest():
            r = torch.sqrt(1 + m00 - m11 - m22)
            s = 0.5 / r
            return torch.stack([
                (m21 - m12) * s,
                0.5 * r,
                (m01 + m10) * s,
                (m02 + m20) * s
            ], dim=-1)
            
        def when_m11_largest():
            r = torch.sqrt(1 - m00 + m11 - m22)
            s = 0.5 / r
            return torch.stack([
                (m02 - m20) * s,
                (m01 + m10) * s,
                0.5 * r,
                (m12 + m21) * s
            ], dim=-1)
            
        def when_m22_largest():
            r = torch.sqrt(1 - m00 - m11 + m22)
            s = 0.5 / r
            return torch.stack([
                (m10 - m01) * s,
                (m02 + m20) * s,
                (m12 + m21) * s,
                0.5 * r
            ], dim=-1)
        
        # Choose appropriate conversion based on largest diagonal element
        where_trace_positive = trace > 0
        where_m00_largest = (m00 > m11) & (m00 > m22) & ~where_trace_positive
        where_m11_largest = (m11 > m00) & (m11 > m22) & ~where_trace_positive
        where_m22_largest = (m22 >= m00) & (m22 >= m11) & ~where_trace_positive
        
        quaternion = torch.zeros(matrix.shape[:-2] + (4,), device=matrix.device)
        quaternion = torch.where(where_trace_positive.unsqueeze(-1), when_trace_positive(), quaternion)
        quaternion = torch.where(where_m00_largest.unsqueeze(-1), when_m00_largest(), quaternion)
        quaternion = torch.where(where_m11_largest.unsqueeze(-1), when_m11_largest(), quaternion)
        quaternion = torch.where(where_m22_largest.unsqueeze(-1), when_m22_largest(), quaternion)
        
        return quaternion

    def _quaternion_to_matrix(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert batch of quaternions to 3x3 rotation matrices."""
        qx, qy, qz, qw = torch.unbind(quaternion, dim=-1)
        
        # Compute matrix elements
        m00 = 1 - 2 * (qy**2 + qz**2)
        m01 = 2 * (qx*qy - qz*qw)
        m02 = 2 * (qx*qz + qy*qw)
        
        m10 = 2 * (qx*qy + qz*qw)
        m11 = 1 - 2 * (qx**2 + qz**2)
        m12 = 2 * (qy*qz - qx*qw)
        
        m20 = 2 * (qx*qz - qy*qw)
        m21 = 2 * (qy*qz + qx*qw)
        m22 = 1 - 2 * (qx**2 + qy**2)
        
        # Stack into rotation matrix
        matrix = torch.stack([
            m00, m01, m02,
            m10, m11, m12,
            m20, m21, m22
        ], dim=-1).view(quaternion.shape[:-1] + (3, 3))
        
        return matrix