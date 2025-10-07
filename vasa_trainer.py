import h5py
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
import itertools
from vasa_sampler import WindowSequenceSampler, create_window_sequence_collate_fn
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import logging
import os
from rich.logging import RichHandler
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, List
from collections import defaultdict
from omegaconf import OmegaConf
from datetime import datetime
from vasa_model import VASAModel
from motion_sequence_handler import MotionSequenceHandler
from vasa_losses import VASALossModule
from tdd_progressive_loss import TDDProgressiveLoss
from loss_monitor import LossRangeMonitor
import importlib
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

def debug_warps(
    predicted_motion: dict,
    target_motion: dict = None,
    num_frames: int = 50,
    log_to_wandb: bool = False,
    step: int = None
):
    """
    Debug warps over a sequence of frames by analyzing norms of rigid (theta rotation/translation)
    and non-rigid (UV displacements) warps. Compares to target if provided to check variation matching.

    Args:
        predicted_motion: Dict with keys like 'theta' [B, T, 3, 4], 'uv_warps' [B, T, D, S, S, 3]
        target_motion: Optional target/ground truth motion dict for comparison
        num_frames: Number of frames to analyze (up to T)
        log_to_wandb: If True, log metrics to wandb
        step: Training step for wandb logging
    """
    # Extract dimensions
    T = min(num_frames, next(iter(predicted_motion.values())).shape[1] if predicted_motion else 0)
    if T == 0:
        logger.warning("No frames in predicted_motion")
        return

    metrics = {
        'rot_norm': [],
        'trans_norm': [],
        'uv_disp_norm': []
    }
    if target_motion:
        target_metrics = {k: [] for k in metrics}
        diff_metrics = {k: [] for k in metrics}

    for t in range(T):
        frame_metrics = {}

        # Rigid: theta [B, T, 3, 4] -> rotation [3,3], translation [3]
        if 'theta' in predicted_motion:
            theta = predicted_motion['theta'][0, t]  # [3,4]
            R = theta[:3, :3]  # Rotation matrix
            trans = theta[:3, 3]   # Translation vector

            # Rotation deviation norm: ||R - I||_F
            I = torch.eye(3, device=theta.device)
            rot_norm = torch.norm(R - I, p='fro').item()
            trans_norm = torch.norm(trans, p=2).item()

            frame_metrics['rot_norm'] = rot_norm
            frame_metrics['trans_norm'] = trans_norm
        else:
            frame_metrics['rot_norm'] = 0.0
            frame_metrics['trans_norm'] = 0.0

        # Non-rigid: uv_warps [B, T, D, S, S, 3] displacements
        if 'uv_warps' in predicted_motion:
            uv = predicted_motion['uv_warps'][0, t]  # [D, S, S, 3]
            # Mean L2 norm per voxel
            uv_disp_norm = torch.mean(torch.norm(uv.view(-1, 3), p=2, dim=1)).item()
            frame_metrics['uv_disp_norm'] = uv_disp_norm
        else:
            frame_metrics['uv_disp_norm'] = 0.0

        # Append to lists
        for k, v in frame_metrics.items():
            metrics[k].append(v)

        # Handle target if provided
        if target_motion:
            target_frame = {}
            if 'theta' in target_motion:
                theta_tgt = target_motion['theta'][0, t]
                R_tgt = theta_tgt[:3, :3]
                trans_tgt = theta_tgt[:3, 3]
                rot_norm_tgt = torch.norm(R_tgt - I, p='fro').item()
                trans_norm_tgt = torch.norm(trans_tgt, p=2).item()
                target_frame['rot_norm'] = rot_norm_tgt
                target_frame['trans_norm'] = trans_norm_tgt

                # Differences
                diff_metrics['rot_norm'].append(abs(rot_norm - rot_norm_tgt))
                diff_metrics['trans_norm'].append(abs(trans_norm - trans_norm_tgt))

            if 'uv_warps' in target_motion:
                uv_tgt = target_motion['uv_warps'][0, t]
                uv_disp_norm_tgt = torch.mean(torch.norm(uv_tgt.view(-1, 3), p=2, dim=1)).item()
                target_frame['uv_disp_norm'] = uv_disp_norm_tgt
                diff_metrics['uv_disp_norm'].append(abs(uv_disp_norm - uv_disp_norm_tgt))

            for k, v in target_frame.items():
                target_metrics[k].append(v)

    # Compute averages
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    if target_motion:
        avg_target = {k: np.mean(v) if v else 0.0 for k, v in target_metrics.items()}
        avg_diff = {k: np.mean(v) if v else 0.0 for k, v in diff_metrics.items()}

    if log_to_wandb and wandb.run is not None:
        log_dict = {f"debug/warp_{k}_avg": v for k, v in avg_metrics.items()}
        if target_motion:
            log_dict.update({f"debug/warp_target_{k}_avg": v for k, v in avg_target.items()})
            log_dict.update({f"debug/warp_diff_{k}_avg": v for k, v in avg_diff.items()})

        # Also log variance to detect collapse
        log_dict[f"debug/warp_uv_variance"] = np.var(metrics['uv_disp_norm']) if metrics['uv_disp_norm'] else 0.0

        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
    else:
        logger.info("\n=== Warp Debug Averages ===")
        for k, v in avg_metrics.items():
            logger.info(f"  Predicted {k}: {v:.4f}")
        if target_motion:
            for k, v in avg_target.items():
                logger.info(f"  Target {k}: {v:.4f}")
            for k, v in avg_diff.items():
                logger.info(f"  Diff {k}: {v:.4f}")
from logger import logger,TorchDebugger
import traceback
from vasa_dataset import WorkerState, VASAIntegratedDataset
from vasa_va_bridge import VASAVolumetricAvatarBridge
from torch.utils.data import random_split
import torch.multiprocessing as mp
import random
from torch.profiler import profile, record_function, ProfilerActivity
from mem import memory_stats,clean_memory,TensorMemoryManager
import gc
from torchvision.utils import save_image
import math
from typing import *
from torch.optim import AdamW


class CFGScheduleHandler:
    """Handles CFG scale scheduling during training."""
    
    def __init__(self, config):
        self.config = config
        self.start_epoch = config.diffusion.cfg_start_epoch
        self.ramp_epochs = 10
        self.max_scales = config.train.cfg_scales
        
    def get_scales(self, current_epoch: int) -> Optional[Dict[str, float]]:
        """Get current CFG scales based on training progress."""
        if current_epoch < self.start_epoch:
            return None
            
        # Calculate ramp progress
        progress = min(1.0, (current_epoch - self.start_epoch) / self.ramp_epochs)
        
        return {
            'audio': self.max_scales.audio * progress,
            'gaze': self.max_scales.gaze * progress,
            'head_distance': self.max_scales.head_distance * progress,
            'emotion': self.max_scales.emotion * progress
        }

class GradientMonitor:
    """Monitors and logs gradient statistics during training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.reset()
        
    def reset(self):
        """Reset gradient statistics."""
        self.grad_norms = defaultdict(list)
        self.grad_means = defaultdict(list)
        self.grad_vars = defaultdict(list)
        
    def update(self):
        """Update gradient statistics."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    self.grad_norms[name].append(grad.norm().item())
                    self.grad_means[name].append(grad.mean().item())
                    self.grad_vars[name].append(grad.var().item())
                    
    def get_stats(self) -> Dict[str, float]:
        """Get current gradient statistics."""
        stats = {}
        
        # Overall statistics
        all_norms = [n for norms in self.grad_norms.values() for n in norms]
        if all_norms:
            stats.update({
                'grad_norm_min': min(all_norms),
                'grad_norm_max': max(all_norms),
                'grad_norm_mean': np.mean(all_norms),
                'grad_norm_std': np.std(all_norms)
            })
            
        # Per-layer statistics
        for name, norms in self.grad_norms.items():
            if norms:
                stats.update({
                    f'grad_norm_{name}_mean': np.mean(norms),
                    f'grad_norm_{name}_std': np.std(norms)
                })
                
        # Add mean/variance statistics
        for name, means in self.grad_means.items():
            if means:
                stats[f'grad_mean_{name}'] = np.mean(means)
        for name, vars in self.grad_vars.items():
            if vars:
                stats[f'grad_var_{name}'] = np.mean(vars)
                
        return stats

class LearningRateMonitor:
    """Monitors learning rates for all parameter groups."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        
    def get_lrs(self) -> Dict[str, float]:
        """Get current learning rates for all parameter groups."""
        lrs = {}
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            lrs[f'lr_{group_name}'] = group['lr']
        return lrs
    
class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear learning rate scheduler with warmup.
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get updated learning rates with specialized handling for expression parameters.
        Implements stage-based warmup and decay with expression-specific scaling.
        
        Returns:
            List of learning rates for each parameter group
        """
        try:
            if self.last_epoch < 0:
                return [group['lr'] for group in self.optimizer.param_groups]
                
            # Get parameter group info
            lrs = []
            for group in self.optimizer.param_groups:
                group_name = group.get('name', '')
                initial_lr = group['initial_lr']
                
                # Expression parameters get special treatment
                is_expression = 'expression' in group_name
                
                # Custom warmup schedule for expressions
                if is_expression:
                    warmup_steps = self.num_warmup_steps // 2  # Faster warmup
                    peak_lr = initial_lr * 2.0  # Higher peak learning rate
                else:
                    warmup_steps = self.num_warmup_steps
                    peak_lr = initial_lr

                # Handle warmup phase
                if self.last_epoch < warmup_steps:
                    # Linear warmup with expression scaling
                    warmup_progress = float(self.last_epoch) / float(max(1, warmup_steps))
                    lr = peak_lr * warmup_progress
                    
                    # Add minimum lr during warmup
                    lr = max(self.min_lr, lr)
                    
                    lrs.append(lr)
                    continue

                # Post-warmup decay phase
                decay_steps = self.num_training_steps - warmup_steps
                current_decay_step = self.last_epoch - warmup_steps
                
                # Compute decay factor
                if decay_steps <= 0:
                    decay_factor = 1.0
                else:
                    decay_progress = float(current_decay_step) / float(max(1, decay_steps))
                    
                    # Expression parameters get slower decay
                    if is_expression:
                        decay_factor = max(0.2, 1.0 - (0.8 * decay_progress))  # Slower decay
                    else:
                        decay_factor = max(0.0, 1.0 - decay_progress)  # Linear decay

                # Apply decay to peak learning rate
                lr = max(self.min_lr, peak_lr * decay_factor)
                
                # Add noise to break plateaus for expression parameters
                if is_expression and self.last_epoch % 50 == 0:  # Every 50 steps
                    noise_scale = 0.1 * lr  # 10% noise
                    lr = lr * (1.0 + random.uniform(-noise_scale, noise_scale))
                
                lrs.append(lr)

            return lrs

        except Exception as e:
            logger.error(f"Error in learning rate calculation: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to minimum learning rate
            return [self.min_lr for _ in self.optimizer.param_groups]


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-7,
    last_epoch: int = -1
) -> LinearWarmupScheduler:
    """
    Creates a scheduler with a linear warmup and decay schedule.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        num_warmup_steps: Number of warmup steps at start
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    
    Returns:
        Configured learning rate scheduler
    """
    return LinearWarmupScheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
        last_epoch=last_epoch
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 1e-7,
    last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Creates a scheduler with a linear warmup and cosine annealing schedule.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        num_warmup_steps: Number of warmup steps at start
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        min_lr: Minimum learning rate after decay
        last_epoch: Last epoch (-1 default)
    
    Returns:
        Configured learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        # Cosine decay with minimum learning rate
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        decay = max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return decay * 0.8  

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def worker_init_fn(worker_id: int):
    """Initialize worker process with proper error handling"""
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(worker_id)
        np.random.seed(worker_id)
        random.seed(worker_id)
        
        # Initialize worker state
        WorkerState.initialize_worker(worker_id)
        
        logger.info(f"Successfully initialized worker {worker_id}")
        
    except Exception as e:
        logger.error(f"Failed to initialize worker {worker_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


class TrainingState:
    """Manages training state and scheduling"""
    def __init__(self, config: dict):
        self.config = config
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize schedules
        self.schedules = {
            'cfg': self._init_cfg_schedule(),
            'window': self._init_window_schedule(),
            'dropout': self._init_dropout_schedule()
        }
        
    def _init_cfg_schedule(self):
        """Initialize CFG scale scheduling"""
        return {
            'audio': lambda e: min(3.0, 0.5 + e * 0.1),
            'gaze': lambda e: 1.0,
            'head_distance': lambda e: 0.8,
            'emotion': lambda e: 0.5
        }
        
    def _init_window_schedule(self):
        """Initialize window size scheduling"""
        base_size = self.config.motion.window_size
        return lambda e: min(50, base_size + e * 2)
        
    def _init_dropout_schedule(self):
        """Initialize condition dropout scheduling"""
        return {
            'audio': lambda e: max(0.1, 0.3 - e * 0.02),
            'gaze': lambda e: 0.1,
            'head_distance': lambda e: 0.1,
            'emotion': lambda e: 0.1
        }
        
    def get_current_scales(self) -> Dict[str, float]:
        """Get current CFG scales based on epoch"""
        return {
            k: schedule(self.epoch) 
            for k, schedule in self.schedules['cfg'].items()
        }
        
    def get_current_dropouts(self) -> Dict[str, float]:
        """Get current dropout probabilities"""
        return {
            k: schedule(self.epoch)
            for k, schedule in self.schedules['dropout'].items()
        }
        
    def get_window_size(self) -> int:
        """Get current window size"""
        return int(self.schedules['window'](self.epoch))



class MetricsTracker:
    """Tracks and logs training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset metric accumulation"""
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for k, v in metrics.items():
            self.metrics[k] += v
            self.counts[k] += 1
            
    def get_averages(self) -> Dict[str, float]:
        """Get average metrics"""
        return {
            k: v / self.counts[k]
            for k, v in self.metrics.items()
        }

def save_video_frames(
    frames: torch.Tensor,
    output_path: Path,
    fps: int = 25
):
    """Save tensor of frames as video using same approach as vi.py"""
    # Convert to numpy and correct format
    frames = frames.cpu().numpy()
    if frames.ndim == 3:  # Single frame CHW -> HWC
        frames = frames.transpose(1, 2, 0)
        frames = np.expand_dims(frames, 0)  # Add batch dimension
    elif frames.ndim == 4 and frames.shape[1] == 3:  # NCHW -> NHWC
        frames = frames.transpose(0, 2, 3, 1)
    
    # Scale to uint8 range
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
        
    # Setup video writer with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frames.shape[2], frames.shape[1])
    )
    
    # Write frames
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()



class VisualizationHandler:
    """Handles generation of visualizations"""
    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.vis_dir = output_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_batch(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        batch_idx: int
    ):
        """Save visualization of generated sequences"""
        epoch_dir = self.vis_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)
        
        # Save N samples
        for i in range(min(
            outputs['pred_target_img'].shape[0],
            self.config.vis.num_vis_samples
        )):
            # Save ground truth
            save_video_frames(
                batch['frames'][i],
                epoch_dir / f'batch_{batch_idx}_sample_{i}_true.mp4'
            )
            
            # Save prediction
            save_video_frames(
                outputs['pred_target_img'][i],
                epoch_dir / f'batch_{batch_idx}_sample_{i}_pred.mp4'
            )



def collate_vasa_batch(batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """Custom collate function that preserves required metadata for windowing."""
    try:
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        # Get all windows from batch items
        all_windows = []
        for item in batch:
            if 'windows' in item:
                # Old path: item has 'windows' key with list of windows
                windows = item['windows']
                # Add video path to window metadata
                for window in windows:
                    if 'metadata' not in window:
                        window['metadata'] = {}
                    window['metadata']['video_path'] = item.get('video_path', '')
                all_windows.extend(windows)
            elif 'theta' in item:
                # New path: item IS a window (from VASAIntegratedDataset.__getitem__)
                # Treat the item itself as a window
                if 'metadata' not in item:
                    item['metadata'] = {}
                all_windows.append(item)

        if not all_windows:
            logger.error("No valid windows in batch - neither 'windows' key nor direct window data found")
            return None

        # Get tensor keys from first window
        first_window = all_windows[0]
        tensor_keys = [k for k, v in first_window.items() if isinstance(v, torch.Tensor)]

        # Stack tensors
        collated = {}
        for key in tensor_keys:
            try:
                tensors = []
                for window in all_windows:
                    if key in window:
                        tensors.append(window[key])
                if tensors:
                    collated[key] = torch.stack(tensors)
            except Exception as e:
                logger.error(f"Error stacking {key}: {str(e)}")
                continue

        # Add metadata list
        collated['metadata'] = [w.get('metadata', {}) for w in all_windows]

        # Log collated shapes
        logger.info("Collated batch shapes:")
        for key, value in collated.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: {value.shape}")

        return collated

    except Exception as e:
        logger.error(f"Error in collate function: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    

class VASATrainer:
    """VASA trainer with integrated loss computation."""          
    def __init__(
        self,
        model: VASAModel,
        config: dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.config_path = config_path or 'vasa_config.yaml'
        self.output_dir = Path(output_dir) if output_dir else None
        self.train_loader = train_loader

        # Initialize accelerator - respect config for mixed precision
        # Check if mixed precision should be enabled from config
        use_amp = getattr(config.motion, 'amp', False)  # Default to False if not specified
        mixed_precision = "fp16" if use_amp else "no"
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision
        )
        
        logger.info(f"Mixed precision training: {'ENABLED (fp16)' if use_amp else 'DISABLED'}")
        logger.info(f"Gradient accumulation steps: {config.train.gradient_accumulation_steps}")
        
        # Only enable gradient anomaly detection in debug mode (it's slow)
        if getattr(config, 'debug', False):
            torch.autograd.set_detect_anomaly(True)
            logger.info("Gradient anomaly detection: ENABLED (debug mode)")
        else:
            torch.autograd.set_detect_anomaly(False)
            logger.info("Gradient anomaly detection: DISABLED (normal mode)")

     
            
        # Initialize loss module
        self.loss_module = VASALossModule(
            volumetric_avatar=model.volumetric_avatar,
            config=config,
            device=self.accelerator.device
        )
        
        # Initialize volumetric avatar bridge for proper frame generation
        self.va_bridge = VASAVolumetricAvatarBridge(model.volumetric_avatar)
        logger.info("Initialized VASAVolumetricAvatarBridge for proper frame generation")
        
        # Initialize TDD progressive loss if enabled
        self.tdd_loss_module = None
        if config.loss.get('use_tdd_progressive', False):
            self.tdd_loss_module = TDDProgressiveLoss(
                config=config,
                device=self.accelerator.device
            )
            logger.info("🎯 TDD Progressive Loss System Enabled")
            logger.info("   Losses will unlock progressively based on training milestones")
        
        # Load high-quality identity image if specified
        self.identity_image = None
        if config.dataset.get('use_identity_image', False):
            identity_path = config.dataset.get('identity_image_path', None)
            if identity_path and os.path.exists(identity_path):
                from PIL import Image
                import torchvision.transforms as transforms
                
                logger.info(f"Loading high-quality identity image from: {identity_path}")
                
                # Load and preprocess the identity image
                # IMPORTANT: Volumetric avatar expects [0, 1] range, not [-1, 1]
                img = Image.open(identity_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),  # Converts to [0, 1] range
                    # DO NOT normalize to [-1, 1] - volumetric avatar expects [0, 1]
                ])
                self.identity_image = transform(img).unsqueeze(0)  # [1, C, H, W] in [0, 1] range
                logger.info(f"Identity image loaded with shape: {self.identity_image.shape}, range: [0, 1]")
            else:
                logger.warning(f"Identity image path not found: {identity_path}")
                logger.warning("Falling back to using video frames for identity")

        # Get use_derived_warps flag from model config
        self.use_derived_warps = getattr(config.model, 'use_derived_warps', False)
        if self.use_derived_warps:
            logger.info("✅ Using derived warps: will compute identity-conditioned warps on-the-fly")
        else:
            logger.info("Using pre-computed warps from dataset")

        # Pre-compute identity embedding once for derived warps (MEMORY OPTIMIZATION)
        self.idt_embed = None
        if self.use_derived_warps and self.identity_image is not None:
            with torch.no_grad():
                identity_img = self.identity_image.to(self.accelerator.device)

                # Get face mask
                identity_mask, _, _, _ = self.model.volumetric_avatar.face_idt.forward(identity_img)
                identity_mask = (identity_mask > 0.6).float()
                identity_mask = torch.nn.functional.avg_pool2d(identity_mask, 3, stride=1, padding=1)

                # Mask identity image
                masked_identity = identity_img * identity_mask

                # Extract identity embedding (spatial feature map)
                idt_embed_spatial = self.model.volumetric_avatar.idt_embedder_nw(masked_identity)

                # Convert to vector via global average pooling
                self.idt_embed = torch.nn.functional.adaptive_avg_pool2d(
                    idt_embed_spatial, (1, 1)
                ).squeeze(-1).squeeze(-1)  # [1, 512]

                logger.info(f"✅ Pre-computed identity embedding: {self.idt_embed.shape}")

                # Clean up
                del identity_img, identity_mask, masked_identity, idt_embed_spatial

        # Initialize MotionSequenceHandler
        self.motion_handler = MotionSequenceHandler(
            window_size=config.motion.window_size,
            stride=config.motion.stride,
            context_size=config.motion.context_size
            
        )

             # Initialize optimizer
        self.init_optimizers()
        self.cfg_scheduler = CFGScheduleHandler(config)
        self.grad_monitor = GradientMonitor(model)
        self.lr_monitor = LearningRateMonitor(self.optimizer)
        self.loss_monitor = LossRangeMonitor(enable_warnings=True, enable_critical=True)
        logger.info("✅ Loss range monitoring enabled - will warn on unhealthy loss values")

        # Prepare training components
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.loss_module
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            train_loader,
            self.loss_module
        )

        if val_loader is not None:
            self.val_loader = self.accelerator.prepare(val_loader)
        else:
            self.val_loader = None

        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Set up metrics tracker
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.training_state = TrainingState(config)
        self.worker_state = WorkerState.get_instance()

      


    
    def get_layer_wise_learning_rates(self, model: VASAModel) -> List[Dict[str, Any]]:
        """
        Get learning rate parameters with separate learning rates for each motion component.

        Args:
            model: VASAModel instance to extract parameters from

        Returns:
            List of parameter groups with different learning rates
        """
        # Initialize parameter groups
        param_groups = []
        motion_proj_names = ['theta', 'rotation', 'translation', 'expression']

        # Keep track of parameters to avoid duplicates
        used_params = set()

        # Get expression learning rate multiplier from config
        expression_lr_mult = getattr(self.config.train, 'expression_lr_mult', 2.0)

        # Create separate parameter groups for each motion projection type
        for proj_name in motion_proj_names:
            proj_params = []

            # Find parameters for this specific projection
            for name, param in model.named_parameters():
                if f'motion_projections.{proj_name}' in name and param not in used_params:
                    proj_params.append(param)
                    used_params.add(param)

            # Learning rate multipliers (configurable from config)
            lr_multipliers = {
                'theta': 1.0,                      # Base rate for pose matrix
                'rotation': 1.0,                   # Base rate for rotational parameters
                'translation': 1.0,                # Base rate for translation
                'expression': expression_lr_mult   # Configurable rate for expression (default: 2.0x)
            }

            # Warn if no parameters found
            if not proj_params:
                logger.warning(f"No parameters found for motion projection: {proj_name}")
                continue

            # Create parameter group
            param_groups.append({
                'params': proj_params,
                'lr': self.config.train.lr * lr_multipliers.get(proj_name, 1.0),
                'name': f'motion_proj_{proj_name}',
                'weight_decay': self.config.train.weight_decay * 0.5
            })

        # Add expression head parameters (expr_head, expression_emb) with same high LR
        expr_head_params = []
        for name, param in model.named_parameters():
            if ('expr_head' in name or 'expression_emb' in name or 'expr_emb' in name) and param not in used_params:
                expr_head_params.append(param)
                used_params.add(param)

        if expr_head_params:
            param_groups.append({
                'params': expr_head_params,
                'lr': self.config.train.lr * expression_lr_mult,
                'name': 'expression_head',
                'weight_decay': self.config.train.weight_decay * 0.5
            })
        
        # Collect remaining parameters that haven't been added to any group
        other_params = [
            param for name, param in model.named_parameters() 
            if param not in used_params
        ]
        
        # Add other model parameters as the final group if any exist
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.train.lr,
                'name': 'other',
                'weight_decay': self.config.train.weight_decay
            })
        
        # Log parameter group details
        logger.info("Learning Rate Parameter Groups:")
        for group in param_groups:
            params_count = len(group['params'])
            group_name = group.get('name', 'unnamed')
            group_lr = group.get('lr', 'default')
            logger.info(f"  {group_name}:")
            logger.info(f"    Number of parameters: {params_count}")
            logger.info(f"    Learning rate: {group_lr}")
        
        return param_groups
        

    # Update the trainer initialization to use layer-wise learning rates
    def init_optimizers(self):
        """Initialize optimizers with layer-wise learning rates"""
        # Get layer-wise parameters with custom learning rates
        layerwise_params = self.get_layer_wise_learning_rates(self.model)
        
        # # Initialize optimizer with parameter groups
        self.optimizer = AdamW(
            layerwise_params,
            betas=(self.config.train.beta1, self.config.train.beta2),
            weight_decay=self.config.train.weight_decay,
            eps=1e-8
        )

        # self.optimizer = torch.optim.AdamW(
        #     self.model.motion_transformer.parameters(),
        #     lr=config.train.lr,
        #     betas=(config.train.beta1, config.train.beta2),
        #     weight_decay=config.train.weight_decay,
        #     eps=1e-8
        # )
        
        # Initialize scheduler
        num_training_steps = len(self.train_loader) * self.config.train.num_epochs
        warmup_steps = self.config.motion.warmup_steps
        
        if self.config.motion.scheduler.type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps, 
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                min_lr=self.config.motion.scheduler.min_lr
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # Initialize gradients to zero to ensure clean state
        self.optimizer.zero_grad()
        
        # Check for any NaN/Inf in model parameters at initialization
        self._check_model_parameters()

    def _check_model_parameters(self):
        """Check model parameters for NaN/Inf values."""
        for name, param in self.model.named_parameters():
            if param is not None:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"NaN/Inf detected in parameter {name} at initialization!")
                    logger.error(f"  Shape: {param.shape}, dtype: {param.dtype}")
                    logger.error(f"  Has NaN: {torch.isnan(param).any()}, Has Inf: {torch.isinf(param).any()}")
                    # Try to fix by reinitializing
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                        logger.info(f"  Reinitialized {name} with xavier_uniform")
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        logger.info(f"  Reinitialized {name} with zeros")

    def train(self):
        """Main training loop."""
        logger.info("Starting training")
        num_epochs = self.config.train.num_epochs

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_stats = self.train_epoch()
            
            # Validation phase - check if we should validate this epoch
            val_stats = None
            # Check both logging.validation.enabled and top-level validation.enabled for backward compatibility
            val_enabled = self.config.get('logging', {}).get('validation', {}).get('enabled',
                          self.config.get('validation', {}).get('enabled', True))
            if self.val_loader and val_enabled:
                val_frequency = self.config.get('logging', {}).get('validation', {}).get('frequency',
                               self.config.get('validation', {}).get('frequency', 5))
                # Use total_epochs or num_epochs, with fallback to 100
                total_epochs = self.config.get('total_epochs', self.config.get('num_epochs', 100))
                if epoch % val_frequency == 0 or epoch == total_epochs - 1:
                    logger.info(f"Running validation at epoch {epoch} (frequency: every {val_frequency} epochs)")
                    val_stats = self.validate()
                else:
                    logger.info(f"Skipping validation at epoch {epoch} (next validation at epoch {((epoch // val_frequency) + 1) * val_frequency})")
            
            # Check if this is the best model based on training or validation loss
            if val_stats and isinstance(val_stats, dict):
                current_loss = val_stats.get('total', val_stats.get('loss', float('inf')))
            else:
                current_loss = train_stats.get('total', float('inf'))
            
            # Save checkpoint only if it's the best model
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.save_checkpoint(is_best=True)
                logger.info(f"New best model saved with loss: {current_loss:.4f}")
            
            # ALWAYS save a checkpoint every epoch (in addition to best model)
            # This prevents losing hours of training if memory crashes
            epoch_checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            self.save_epoch_checkpoint(epoch_checkpoint_path)
            
            # Keep only last 3 epoch checkpoints to save disk space
            self.cleanup_old_epoch_checkpoints(keep_last=3)


    def get_current_stage(self, epoch: int = None) -> tuple:
        """Get current training stage based on epoch.

        Returns:
            (stage_number, stage_name, active_losses)
        """
        if epoch is None:
            epoch = self.current_epoch

        if epoch >= 45:
            return 6, "Lip Sync Refinement", ["lips: 2.0", "audio_lip: 10.0", "temporal: 0.01"]
        elif epoch >= 35:
            return 5, "Emotion", ["emotion: 0.0→0.8", "perceptual: 0.5"]
        elif epoch >= 25:
            return 4, "Head Pose", ["head_distance: 0.1→1.0", "identity: 0.3"]
        elif epoch >= 15:
            return 3, "Eye Gaze", ["gaze_direction: 0.1→1.5", "nonlip: 0.1"]
        elif epoch >= 5:
            return 2, "Blinking", ["blink: 0.1→2.0", "expression_l1: 1.5"]
        else:
            return 1, "Foundation", ["reconstruction: 2.0", "dynamics: 1.0", "pose: 5.0", "audio_lip: 10.0"]

    def train_epoch(self) -> Dict[str, float]:
        """Training loop with proper noise level sampling."""
        self.model.train()  # Put model in training mode
        self.train_metrics.reset()

        # Log stage transition if needed
        stage_num, stage_name, active_losses = self.get_current_stage()
        prev_stage_num, _, _ = self.get_current_stage(self.current_epoch - 1) if self.current_epoch > 0 else (0, "", [])

        if stage_num != prev_stage_num and self.config.wandb.enabled and self.accelerator.is_local_main_process:
            logger.info(f"🎯 STAGE TRANSITION: Entering Stage {stage_num} - {stage_name}")
            logger.info(f"   Active losses: {', '.join(active_losses)}")

            if wandb.run:
                wandb.log({
                    "training/stage_transition": stage_num,
                    "training/stage_transition_name": stage_name,
                    "training/stage_transition_epoch": self.current_epoch
                }, step=self.global_step)
        num_batches = len(self.train_loader)
        
        # Clear VA bridge cache at start of epoch
        if hasattr(self, 'va_bridge'):
            self.va_bridge.clear_cache()
        
        logger.info(f"\n=== Starting Epoch {self.current_epoch} ===")
        logger.info(f"Batch size: {self.config.train.batch_size}")
        logger.info(f"Total batches: {num_batches}")
        
        # Update TDD progressive loss module if enabled
        if self.tdd_loss_module is not None:
            # Get metrics from last epoch (if available)
            metrics = {}
            if hasattr(self, 'last_epoch_metrics'):
                metrics = self.last_epoch_metrics
            
            self.tdd_loss_module.update_epoch(self.current_epoch, metrics)
            
            # Log TDD status
            tdd_status = self.tdd_loss_module.get_status()
            logger.info(f"🎯 TDD Loss Status:")
            logger.info(f"   Stage: {tdd_status['current_stage']}")
            logger.info(f"   Active losses: {', '.join(tdd_status['active_losses'])}")
            if tdd_status['pending_losses']:
                next_loss = tdd_status['pending_losses'][0]
                logger.info(f"   Next unlock: {next_loss['name']} at epoch {next_loss['min_epoch']}")
        
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=num_batches,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Epoch {self.current_epoch}"
        )
        
        # Zero gradients at the start of epoch to ensure clean state
        self.optimizer.zero_grad()
        logger.info("Zeroed gradients at start of epoch")
        
        # Get current CFG scales based on epoch
        cfg_scales = self._get_cfg_scales()
        
        # Initialize epoch metrics
        epoch_metrics = defaultdict(list)
        grad_norms = defaultdict(list)
        
        # Initialize WandB metrics table for this epoch
        if self.config.wandb.enabled and self.accelerator.is_local_main_process:
            # TRAINING STAGE PROGRESSION:
            # Stage 1 (0-4):   Foundation - reconstruction, dynamics, pose, audio_lip (10.0)
            # Stage 2 (5-14):  Blinking - blink (0.1→2.0), expression_l1 (1.5)
            # Stage 3 (15-24): Eye Gaze - gaze_direction (0.1→1.5), nonlip (0.1)
            # Stage 4 (25-34): Head Pose - head_distance (0.1→1.0), identity (0.3)
            # Stage 5 (35-44): Emotion - emotion (0.0→0.8), perceptual (0.5)
            # Stage 6 (45+):   Lip Sync - lips (2.0), audio_lip (10.0), temporal (0.01)

            self.epoch_table = wandb.Table(columns=[
                "batch_idx", "window_idx", "total_loss",
                # Core reconstruction losses
                "reconstruction", "perceptual", "temporal",
                # Motion prediction losses (matching H5 cache structure)
                "uv_warp_loss", "theta_loss", "expression_loss",
                "scale_loss", "rotation_loss", "translation_loss",
                # Lip motion losses (critical for mouth movement)
                "audio_lip_loss", "lips_loss", "expression_l1",
                # Progressive stage losses
                "blink_loss", "gaze_loss", "emotion_loss",
                # Consistency losses
                "l_consist", "l_cross_id", "velocity_smoothness",
                # Training metrics
                "grad_norm", "learning_rate", "stage"
            ])

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                logger.info(f"Processing batch {batch_idx}")
                if batch is None:
                    logger.warning(f"Batch {batch_idx} is None")
                    continue
                    
                logger.info(f"Batch {batch_idx} has data, processing windows...")
                
                # Store current batch data for video generation
                self.current_batch_data = batch
                
                # Process batch into windows
                windows = self.motion_handler.process_batch(
                    batch, 
                    current_window_size=self.training_state.get_window_size()
                )
                if not windows:
                    logger.warning(f"No valid windows in batch {batch_idx}")
                    continue
                    
                logger.info(f"Processing {len(windows)} windows for batch {batch_idx}")

                # Track total loss for entire batch
                batch_total_loss = 0
                batch_metrics = defaultdict(list)
                windows_processed = 0  # Track how many windows actually did backward pass

                # Process each window
                for window_idx, window in enumerate(windows):
                    logger.debug(f"  Processing window {window_idx}/{len(windows)}")
                    if not window:
                        logger.warning(f"  Window {window_idx} is None")
                        continue
                    # Use accumulate per window instead of per batch
                    with self.accelerator.accumulate(self.model):
                        try:
                            logger.debug(f"  Preparing motion data for window {window_idx}")
                            # Extract target motion parameters and move to GPU immediately
                            motion_data = self.motion_handler.prepare_motion_data(window)

                            # Move all motion_data tensors to GPU if not already there
                            device = self.accelerator.device
                            motion_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                         for k, v in motion_data.items()}

                            B = motion_data['theta'].shape[0]

                            # In train_epoch:
                            if self.config.train.turn_off_noise:
                                t = torch.zeros((B,), dtype=torch.long, device=device)  # OVERFIT t = 0 no noise/minimal perturbation of the input
                            else:
                                # Sample timestep uniformly
                                t = torch.randint(0, self.config.diffusion.num_steps, (B,), device=device)

                            # Rest stays the same
                            noise = {k: torch.randn_like(v) for k, v in motion_data.items()}
                            noised_motion = self.model._add_noise_to_motion(
                                motion_data=motion_data,
                                noise=noise,
                                noise_level=t
                            )
                            logger.debug("Added scheduled noise to motion")

                            # Extract control signals with dropout during training
                            # Debug: Check if audio_features exists in window
                            if 'audio_features' not in window:
                                logger.error(f"audio_features not in window! Available keys: {list(window.keys())}")
                                raise ValueError("audio_features missing from window data")

                            # Extract control signals and move to GPU immediately
                            control_signals = {}
                            control_keys = ['gaze', 'head_distance', 'emotion', 'speed_bucket',
                                          'lips', 'right_eye', 'left_eye', 'jaw', 'nose',
                                          'blink_state', 'audio_features']
                            for key in control_keys:
                                value = window.get(key)
                                if value is not None:
                                    if isinstance(value, torch.Tensor):
                                        control_signals[key] = value.to(device, non_blocking=True)
                                    else:
                                        control_signals[key] = value
                                else:
                                    control_signals[key] = None

                            # Debug: Verify audio_features is included
                            if 'audio_features' not in control_signals or control_signals['audio_features'] is None:
                                logger.error(f"audio_features is None or missing in control_signals!")
                                logger.error(f"Window keys: {list(window.keys())}")
                                logger.error(f"Control signal keys: {list(control_signals.keys())}")

                            # Apply control signal dropout
                            if not self.config.train.turn_off_noise:
                                dropout_probs = self.config.train.dropout_probs
                                control_signals = self._apply_condition_dropout(
                                    control_signals,
                                    dropout_probs
                                )

                                # Debug: Check if audio_features survived dropout
                                if 'audio_features' not in control_signals:
                                    logger.error(f"audio_features missing after dropout! Keys: {list(control_signals.keys())}")
                                    raise ValueError("audio_features removed by dropout - this should never happen!")

                            # Use pre-computed identity embedding for derived warps (MEMORY OPTIMIZATION)
                            idt_embed = self.idt_embed if self.use_derived_warps else None
                            if self.use_derived_warps and idt_embed is None:
                                logger.warning("[IDT_EMBED] Pre-computed idt_embed not available, falling back to runtime computation")
                                # Fallback: compute from first frame if needed
                                target_frames_temp = window.get('frames', None)
                                if target_frames_temp is not None:
                                    identity_frame = target_frames_temp[0:1, 0]  # [1, C, H, W]
                                    with torch.no_grad():
                                        identity_mask, _, _, _ = self.model.volumetric_avatar.face_idt.forward(identity_frame)
                                        identity_mask = (identity_mask > 0.6).float()
                                        identity_mask = torch.nn.functional.avg_pool2d(identity_mask, 3, stride=1, padding=1)
                                        masked_identity = identity_frame * identity_mask
                                        idt_embed_spatial = self.model.volumetric_avatar.idt_embedder_nw(masked_identity)
                                        idt_embed = torch.nn.functional.adaptive_avg_pool2d(
                                            idt_embed_spatial, (1, 1)
                                        ).squeeze(-1).squeeze(-1)
                                        logger.debug(f"[IDT_EMBED] Computed fallback idt_embed: {idt_embed.shape}")

                            # Get actual frames from the window data for disentanglement loss
                            target_frames = window.get('frames', None)  # Get frames from dataset

                            # Generate frames if we have disentanglement losses enabled
                            # OPTIMIZATION: Only generate the 2 frames needed for disentanglement loss
                            # Note: VASA paper doesn't specify needing all frames for these losses,
                            # and mathematically only 2 frames are used (first and last)
                            generated_frames = None
                            use_sparse_frames = getattr(self.config.loss, 'use_sparse_frames', False)  # Read from config, default to False
                            enable_frame_generation = getattr(self.config.loss, 'enable_frame_generation', False)  # Read from config

                            # Determine if this is a visualization step
                            # MEMORY OPTIMIZATION: Skip warp generation during normal training
                            is_vis_step = (self.global_step % self.config.vis.vis_freq == 0)

                            # Generate frames if:
                            # 1. Disentanglement losses are enabled (lambda_consist > 0 or lambda_cross_id > 0)
                            # 2. OR frame generation is explicitly enabled for visualization
                            # 3. OR this is a visualization step (for control losses and logging)
                            needs_frames = (self.config.loss.lambda_consist > 0 or
                                          self.config.loss.lambda_cross_id > 0 or
                                          enable_frame_generation or
                                          is_vis_step)  # Enable frames on vis steps for control losses

                            # Only generate warps when frames are needed
                            needs_warps = needs_frames

                            # Log frame generation decision on vis steps
                            if is_vis_step:
                                logger.info(f"[VIS STEP {self.global_step}] Generating frames for control losses and visualization (vis_freq={self.config.vis.vis_freq})")

                            # Forward pass with CFG during inference
                            outputs = self.model(
                                motion_data=noised_motion,  # Use noised motion
                                noise_level=t,              # Pass timestep
                                conditions=control_signals,
                                noise=noise,                # Pass noise for loss computation
                                idt_embed=idt_embed,        # Pass identity for derived warps
                                generate_warps=needs_warps  # Only generate warps when needed (saves VRAM)
                            )

                            if needs_frames and target_frames is not None:
                                try:
                                    # Get source images - use high-quality identity image if available
                                    if self.identity_image is not None:
                                        # Use the same high-quality identity image for all samples in batch
                                        source_img = self.identity_image.repeat(B, 1, 1, 1).to(self.accelerator.device)
                                        logger.info(f"Using high-quality identity image for source, shape: {source_img.shape}, range: [{source_img.min():.2f}, {source_img.max():.2f}]")
                                        
                                        # Save debug image on first batch
                                        if self.global_step == 0:
                                            from torchvision.utils import save_image
                                            denorm_img = source_img[0] * 0.5 + 0.5  # Denormalize
                                            save_image(denorm_img, f"debug_source_img_step{self.global_step}.png")
                                            logger.info(f"Saved debug source image to debug_source_img_step{self.global_step}.png")
                                    else:
                                        # Fall back to first frame of each video in the batch
                                        source_img = target_frames[:, 0]  # [B, C, H, W]
                                    
                                    # Prepare source params for frame generation
                                    # Pass target frames so volumetric avatar knows what to generate
                                    source_params = {
                                        'source_img': source_img,
                                        'target_img': target_frames  # Pass the actual target frames
                                    }
                                    
                                    # Generate frames using volumetric avatar
                                    with torch.no_grad():
                                        T = outputs['theta'].shape[1]
                                        
                                        if use_sparse_frames:
                                            # OPTIMIZATION: Only generate first and last frame (90% memory savings)
                                            # This is sufficient for disentanglement losses which only use these 2 frames
                                            sparse_outputs = {}
                                            for key, value in outputs.items():
                                                if isinstance(value, torch.Tensor) and value.dim() > 2 and value.shape[1] == T:
                                                    # Extract only first and last frame [B, 2, ...]
                                                    sparse_outputs[key] = torch.stack([value[:, 0], value[:, -1]], dim=1)
                                                else:
                                                    sparse_outputs[key] = value
                                            
                                            # Generate only 2 frames instead of T frames using bridge
                                            # Check if using improved bridge
                                            if hasattr(self.va_bridge, 'generate_frames_with_viz'):
                                                result = self.va_bridge.generate_frames_with_viz(
                                                    motion_outputs=sparse_outputs,
                                                    source_img=source_img,
                                                    use_black_background=True,
                                                    enable_3avatar=False,  # Disable for training to save memory
                                                    enable_smoothing=True  # Enable smoothing
                                                )
                                                generated_frames = result['frames'].detach()
                                            else:
                                                generated_frames, gen_stats = self.va_bridge.generate_frames_from_motion(
                                                    motion_outputs=sparse_outputs,
                                                    source_img=source_img,
                                                    use_black_background=True
                                                )
                                                generated_frames = generated_frames.detach()
                                                # Store generation stats for metrics
                                                if not hasattr(self, '_gen_stats'):
                                                    self._gen_stats = []
                                                self._gen_stats.append(gen_stats)
                                            
                                            # Also make target frames sparse to match
                                            sparse_target_frames = torch.stack([target_frames[:, 0], target_frames[:, -1]], dim=1)
                                            target_frames = sparse_target_frames
                                        else:
                                            # Generate all frames (more faithful to paper but memory intensive) using bridge
                                            # Check if using improved bridge
                                            if hasattr(self.va_bridge, 'generate_frames_with_viz'):
                                                result = self.va_bridge.generate_frames_with_viz(
                                                    motion_outputs=outputs,
                                                    source_img=source_img,
                                                    use_black_background=True,
                                                    enable_3avatar=False,  # Disable for training to save memory
                                                    enable_smoothing=True  # Enable smoothing
                                                )
                                                generated_frames = result['frames'].detach()
                                            else:
                                                generated_frames, gen_stats = self.va_bridge.generate_frames_from_motion(
                                                    motion_outputs=outputs,
                                                    source_img=source_img,
                                                    use_black_background=True
                                                )
                                                # Store generation stats and check quality BEFORE loss computation
                                                if not hasattr(self, '_gen_stats'):
                                                    self._gen_stats = []
                                                self._gen_stats.append(gen_stats)

                                                # EARLY EXIT: Skip window if frame quality is too poor
                                                min_valid_ratio = getattr(self.config.loss, 'emo_min_valid_frames', 0.5)
                                                if gen_stats['valid_masks_ratio'] < min_valid_ratio:
                                                    logger.warning(f"⚠️ SKIPPING WINDOW {window_idx}: Frame generation quality too poor ({gen_stats['valid_masks_ratio']:.1%} valid frames < {min_valid_ratio:.0%} threshold)")
                                                    logger.warning(f"  Skipping BEFORE loss computation to avoid wasted computation and gradient issues")
                                                    self._gen_stats = []
                                                    del generated_frames
                                                    torch.cuda.empty_cache()
                                                    continue  # Skip to next window

                                                generated_frames = generated_frames.detach()

                                    frame_type = "sparse (2 frames)" if use_sparse_frames else "full"
                                    logger.debug(f"Generated frames shape ({frame_type}): {generated_frames.shape}")
                                    logger.debug(f"Target frames shape ({frame_type}): {target_frames.shape}")
                                except Exception as e:
                                    logger.error(f"Failed to generate frames for disentanglement loss: {str(e)}")
                                    generated_frames = None

                            # Prepare source identity for loss computation
                            source_identity_for_loss = None
                            if self.identity_image is not None:
                                source_identity_for_loss = self.identity_image.repeat(B, 1, 1, 1).to(self.accelerator.device)
                            
                            # Add lip_metrics to targets if available
                            targets_with_lip = motion_data.copy()
                            if 'lip_metrics' in window:
                                targets_with_lip['lip_metrics'] = window['lip_metrics']
                                logger.debug(f"Added lip_metrics to targets - keys: {list(window['lip_metrics'].keys())}")
                                # Log the shape of openness metric for verification
                                if 'openness' in window['lip_metrics']:
                                    logger.debug(f"  lip_metrics['openness'] shape: {window['lip_metrics']['openness'].shape}")
                            else:
                                logger.warning(f"lip_metrics not found in window! Available keys: {list(window.keys())}")

                            # Compute losses including perceptual loss
                            losses, metrics = self.loss_module.compute_losses(
                                outputs=outputs,
                                targets=targets_with_lip,  # Motion data with lip_metrics
                                conditions=control_signals,
                                noise=outputs['noise'],
                                return_metrics=True,
                                current_epoch=self.current_epoch,
                                step=self.global_step,
                                generated_frames=generated_frames,
                                target_frames=target_frames,
                                source_identity=source_identity_for_loss,  # Pass high-quality identity
                                dataset=self.train_loader.dataset  # Pass dataset for feature extraction
                            )

                            # Monitor loss ranges (every 100 steps, log summary)
                            if self.global_step % 100 == 0 and window_idx == 0:
                                loss_status = self.loss_monitor.check_losses(
                                    losses=losses,
                                    step=self.global_step,
                                    log_summary=True
                                )
                            else:
                                # Check without logging summary (warnings/criticals still logged)
                                loss_status = self.loss_monitor.check_losses(
                                    losses=losses,
                                    step=self.global_step,
                                    log_summary=False
                                )

                            # Clean up generated frames immediately after loss computation
                            if generated_frames is not None:
                                del generated_frames
                                torch.cuda.empty_cache()

                            # Update batch metrics - store per-window
                            for k, v in losses.items():
                                if isinstance(v, torch.Tensor):
                                    batch_metrics[k].append(v.detach())
                            if metrics:
                                for k, v in metrics.items():
                                    batch_metrics[f"metric_{k}"].append(v)

                            # Add generation stats if available (quality check already done earlier)
                            if hasattr(self, '_gen_stats') and self._gen_stats:
                                # Average across all generations in this batch
                                avg_valid_ratio = sum(s['valid_masks_ratio'] for s in self._gen_stats) / len(self._gen_stats)
                                batch_metrics['frame_generation/valid_masks_ratio'].append(avg_valid_ratio)

                                # Clear after logging
                                self._gen_stats = []

                            # Debug warps periodically to detect collapse
                            if self.global_step % 100 == 0 and window_idx == 0:  # Log every 100 steps for first window
                                try:
                                    debug_warps(
                                        predicted_motion=outputs,
                                        target_motion=targets_with_lip,
                                        num_frames=min(10, outputs['theta'].shape[1]),  # Analyze first 10 frames
                                        log_to_wandb=True,
                                        step=self.global_step
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to debug warps: {e}")

                            # Check for NaN/Inf in loss before backward pass
                            if not torch.isfinite(losses['total']):
                                logger.error(f"NaN/Inf detected in loss at epoch {self.current_epoch}, batch {batch_idx}, window {window_idx}")
                                logger.error(f"Loss components: {losses}")
                                
                                # Log detailed info about the NaN
                                for key, value in losses.items():
                                    if isinstance(value, torch.Tensor) and not torch.isfinite(value):
                                        logger.error(f"  NaN/Inf in {key}: {value.item()}")
                                
                                # Check motion outputs for NaN
                                if 'outputs' in locals():
                                    for key, value in outputs.items():
                                        if isinstance(value, torch.Tensor):
                                            if torch.isnan(value).any():
                                                logger.error(f"  NaN in output {key}, shape {value.shape}")
                                            if torch.isinf(value).any():
                                                logger.error(f"  Inf in output {key}, shape {value.shape}")
                                
                                # Log to wandb for tracking
                                if wandb.run is not None:
                                    wandb.log({
                                        'gradient_explosion/nan_loss_detected': 1.0,
                                        'gradient_explosion/step': self.global_step,
                                        'gradient_explosion/epoch': self.current_epoch,
                                        'gradient_explosion/batch': batch_idx,
                                        'gradient_explosion/window': window_idx,
                                    }, step=self.global_step)
                                
                                # Skip this window
                                continue
                            
                            # Check loss before backward to catch issues early
                            if not torch.isfinite(losses['total']):
                                logger.error(f"Non-finite loss detected before backward: {losses['total'].item()}")
                                logger.error(f"Loss components: {[(k, v.item() if torch.is_tensor(v) else v) for k, v in losses.items()]}")
                                # Skip this window
                                self.optimizer.zero_grad()
                                continue
                            
                            # Backward pass for this window (accumulate gradients)
                            try:
                                self.accelerator.backward(losses['total'])
                            except RuntimeError as e:
                                if "Function" in str(e) and "returned nan values" in str(e):
                                    logger.error(f"NaN in backward pass: {str(e)}")
                                    logger.error(f"This typically indicates numerical instability in operations like asin, log, or sqrt")
                                    logger.error(f"Window {window_idx}, Batch {batch_idx}, Step {self.global_step}")
                                    
                                    # Clear gradients and skip
                                    self.optimizer.zero_grad()
                                    continue
                                else:
                                    raise
                            
                            # Store loss value before cleanup
                            window_loss = losses['total'].item()
                            batch_total_loss += window_loss
                            windows_processed += 1  # Count this window as processed
                            logger.info(f"  Window {window_idx} completed - loss: {window_loss:.4f}")
                            
                            # Log visualizations before cleanup (every 5 batches)
                            if batch_idx % 5 == 0 and window_idx == 0:  # Only log first window
                                if 'outputs' in locals():
                                    self._log_visualizations(outputs, motion_data, self.global_step)
                                    # Gradient stats disabled for performance
                                    # self._log_gradient_stats(self.global_step)
                            
                            # Store outputs needed for thumbnail before cleanup (every batch for first window)
                            stored_outputs = None
                            if 'outputs' in locals() and window_idx == 0:  # Generate thumbnail for every batch's first window
                                stored_outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                                for k, v in outputs.items()}
                                # Add gaze from control signals for visualization
                                if 'control_signals' in locals() and control_signals.get('gaze') is not None:
                                    stored_outputs['gaze'] = control_signals['gaze'].detach().cpu() if isinstance(control_signals['gaze'], torch.Tensor) else control_signals['gaze']
                            
                            # Clear intermediate tensors to prevent memory buildup
                            del losses
                            if 'outputs' in locals():
                                del outputs
                            if 'noise' in locals():
                                del noise
                            if 'noised_motion' in locals():
                                del noised_motion
                            
                            # Add row to metrics table
                            if self.config.wandb.enabled and self.accelerator.is_local_main_process:
                                # Convert grad_norm to scalar if it's a tensor
                                grad_norm_value = 0.0
                                if 'grad_norm' in locals() and grad_norm is not None:
                                    if isinstance(grad_norm, torch.Tensor):
                                        grad_norm_value = grad_norm.item()
                                    elif grad_norm is not None:
                                        grad_norm_value = float(grad_norm)
                                else:
                                    grad_norm_value = 0.0  # Default if grad_norm not yet computed
                                
                                # Debug: Log what's in metrics
                                if batch_idx == 0 and window_idx == 0:
                                    logger.info(f"DEBUG: Metrics keys: {list(metrics.keys())}")
                                    logger.info(f"DEBUG: l_consist value: {metrics.get('l_consist', 'NOT FOUND')}")
                                    logger.info(f"DEBUG: l_cross_id value: {metrics.get('l_cross_id', 'NOT FOUND')}")
                                
                                # Get current learning rate
                                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0.0001

                                # Determine current training stage based on epoch
                                current_stage = 1
                                if self.current_epoch >= 45:
                                    current_stage = 6  # Lip Sync refinement
                                elif self.current_epoch >= 35:
                                    current_stage = 5  # Emotion
                                elif self.current_epoch >= 25:
                                    current_stage = 4  # Head Pose
                                elif self.current_epoch >= 15:
                                    current_stage = 3  # Eye Gaze
                                elif self.current_epoch >= 5:
                                    current_stage = 2  # Blinking

                                self.epoch_table.add_data(
                                    batch_idx,
                                    window_idx,
                                    window_loss,  # total_loss
                                    # Core reconstruction losses
                                    metrics.get('reconstruction', 0.0),
                                    metrics.get('perceptual', 0.0),
                                    metrics.get('temporal', 0.0),
                                    # Motion prediction losses (matching H5 cache structure)
                                    metrics.get('uv_warp_loss', 0.0),
                                    metrics.get('theta_loss', metrics.get('pose_loss', 0.0)),  # theta is pose
                                    metrics.get('expression_loss', 0.0),
                                    metrics.get('scale_loss', 0.0),
                                    metrics.get('rotation_loss', 0.0),
                                    metrics.get('translation_loss', 0.0),
                                    # Lip motion losses (critical for mouth movement)
                                    metrics.get('audio_lip_correlation', metrics.get('audio_lip', 0.0)),  # Audio-lip sync
                                    metrics.get('mouth_openness_direct', metrics.get('lips', 0.0)),  # Mouth openness
                                    metrics.get('expression_loss', metrics.get('expression_l1', 0.0)),  # Expression L1/L2
                                    # Progressive stage losses
                                    metrics.get('blink_loss', metrics.get('blink', 0.0)),
                                    metrics.get('gaze_loss', metrics.get('gaze_direction', 0.0)),
                                    metrics.get('emotion_loss', metrics.get('emotion', 0.0)),
                                    # Consistency losses
                                    metrics.get('l_consist', 0.0),
                                    metrics.get('l_cross_id', 0.0),
                                    metrics.get('velocity_smoothness', 0.0),  # Combined velocity/smoothness
                                    # Training metrics
                                    grad_norm_value,
                                    current_lr,
                                    current_stage  # Add stage indicator
                                )
                            
                            # Generate thumbnail with single frame only to save memory
                            # Generate for every batch to ensure we get thumbnails every epoch
                            if window_idx == 0 and self.config.wandb.enabled and stored_outputs is not None:
                                try:
                                    from thumbnail_generator import generate_window_thumbnail
                                    import random
                                    
                                    # Generate just ONE frame for thumbnail to save memory
                                    single_frame_generated = None
                                    single_frame_target = None
                                    frame_idx = 0  # Default frame index
                                    
                                    if target_frames is not None:
                                        # Pick a random frame index
                                        T = target_frames.shape[1] if target_frames.dim() > 4 else 1
                                        frame_idx = random.randint(T//3, T-1) if T > 3 else 0
                                        
                                        # Extract single target frame
                                        single_frame_target = target_frames[:, frame_idx:frame_idx+1] if T > 1 else target_frames[:, 0:1]
                                        
                                        # Generate just this one frame
                                        with torch.no_grad():
                                            try:
                                                # Use high-quality identity image if available
                                                if self.identity_image is not None:
                                                    source_img = self.identity_image.repeat(B, 1, 1, 1).to(self.accelerator.device)
                                                else:
                                                    source_img = target_frames[:, 0]  # [B, C, H, W]
                                                
                                                # Extract single frame motion params
                                                single_motion = {}
                                                if stored_outputs is not None:
                                                    device = self.accelerator.device
                                                    for key in stored_outputs:
                                                        if isinstance(stored_outputs[key], torch.Tensor) and stored_outputs[key].dim() > 2:
                                                            single_motion[key] = stored_outputs[key][:, frame_idx:frame_idx+1].to(device)
                                                        else:
                                                            single_motion[key] = stored_outputs[key] if not isinstance(stored_outputs[key], torch.Tensor) else stored_outputs[key].to(device)

                                                # Check if we have uv_warps (required for frame generation)
                                                if 'uv_warps' not in single_motion:
                                                    logger.debug("Skipping thumbnail generation - no warps available (non-vis step)")
                                                    single_frame_generated = None
                                                else:
                                                    # Generate single frame only using bridge
                                                    single_frame_generated, _ = self.va_bridge.generate_frames_from_motion(
                                                        motion_outputs=single_motion,
                                                        source_img=source_img,
                                                        use_black_background=True  # Use black background for thumbnails
                                                    )
                                                    single_frame_generated = single_frame_generated.detach()
                                                
                                                # Clear memory immediately
                                                del single_motion
                                                torch.cuda.empty_cache()
                                                
                                            except Exception as e:
                                                logger.warning(f"Could not generate single frame for thumbnail: {e}")
                                                single_frame_generated = None
                                    
                                    # Pass single frames to thumbnail generator
                                    # Include identity frame for 3-panel view
                                    identity_for_thumbnail = None
                                    if self.identity_image is not None:
                                        identity_for_thumbnail = self.identity_image[0]  # Remove batch dimension
                                    elif source_img is not None and source_img.numel() > 0:
                                        identity_for_thumbnail = source_img[0] if source_img.dim() > 3 else source_img

                                    # Get EMO frame from dataset if available
                                    single_frame_emo = None
                                    try:
                                        # Check if window has pre-generated EMO frames
                                        if 'emo_frames' in window and 'emo_keyframe_indices' in window:
                                            emo_frames = window['emo_frames']  # Could be [B, num_keyframes, C, H, W] or [num_keyframes, C, H, W]
                                            emo_indices = window['emo_keyframe_indices']
                                            logger.info(f"✅ Found EMO frames in window, shape: {emo_frames.shape}, indices shape: {emo_indices.shape}")

                                            # Handle batched EMO frames - extract for current window
                                            if emo_frames.dim() == 5:  # [B, num_keyframes, C, H, W]
                                                emo_frames = emo_frames[window_idx]  # [num_keyframes, C, H, W]
                                            if emo_indices.dim() == 2:  # [B, num_keyframes]
                                                emo_indices = emo_indices[window_idx]  # [num_keyframes]

                                            # Find the closest EMO keyframe to our selected frame
                                            closest_idx = 0
                                            min_diff = abs(emo_indices[0].item() - frame_idx)
                                            for i, emo_idx in enumerate(emo_indices):
                                                diff = abs(emo_idx.item() - frame_idx)
                                                if diff < min_diff:
                                                    min_diff = diff
                                                    closest_idx = i

                                            single_frame_emo = emo_frames[closest_idx].detach().cpu()
                                            logger.info(f"Using pre-generated EMO frame {closest_idx} (closest to frame {frame_idx})")
                                        else:
                                            logger.debug(f"No pre-generated EMO frames in window (has emo_frames: {'emo_frames' in window}, has indices: {'emo_keyframe_indices' in window})")

                                        if single_frame_emo is None and hasattr(self, 'va_bridge') and self.va_bridge is not None and stored_outputs is not None:
                                            # Fallback: generate EMO frame on-the-fly (slower)
                                            logger.debug("No pre-generated EMO frames, generating on-the-fly...")
                                            with torch.no_grad():
                                                # Get motion for this specific frame
                                                frame_motion = {}
                                                if 'theta' in stored_outputs and stored_outputs['theta'] is not None:
                                                    theta_tensor = stored_outputs['theta']
                                                    if theta_tensor.dim() >= 2 and frame_idx < theta_tensor.shape[1]:
                                                        frame_motion['theta'] = theta_tensor[:, frame_idx:frame_idx+1].to(self.device)

                                                if 'expression_embed' in stored_outputs and stored_outputs['expression_embed'] is not None:
                                                    expr_tensor = stored_outputs['expression_embed']
                                                    if expr_tensor.dim() >= 2 and frame_idx < expr_tensor.shape[1]:
                                                        frame_motion['expression_embed'] = expr_tensor[:, frame_idx:frame_idx+1].to(self.device)

                                                # Only proceed if we have valid motion data
                                                if frame_motion:
                                                    # Prepare identity image
                                                    emo_identity = identity_for_thumbnail.unsqueeze(0) if identity_for_thumbnail.dim() == 3 else identity_for_thumbnail
                                                    emo_identity = emo_identity.to(self.device)

                                                    # Generate EMO frame using volumetric avatar
                                                    emo_output = self.va_bridge.generate_from_motion(
                                                        identity_image=emo_identity,
                                                        motion_params=frame_motion
                                                    )

                                                    if emo_output is not None and 'generated_frames' in emo_output:
                                                        emo_frames = emo_output['generated_frames']
                                                        if emo_frames.dim() >= 2:
                                                            single_frame_emo = emo_frames[0, 0].detach().cpu() if emo_frames.shape[1] > 0 else emo_frames[0].detach().cpu()
                                                            logger.debug(f"Generated EMO frame on-the-fly, shape: {single_frame_emo.shape}")
                                    except Exception as e:
                                        logger.debug(f"Could not get EMO frame for thumbnail: {e}")
                                        # Continue without EMO frame

                                    # Prepare EMO frames tensor if available
                                    emo_frames_for_thumbnail = None
                                    if single_frame_emo is not None:
                                        # Add batch and time dimensions: [B=1, T=1, C, H, W]
                                        emo_frames_for_thumbnail = single_frame_emo.unsqueeze(0).unsqueeze(0)
                                        logger.info(f"✅ Prepared EMO frame for 4-panel thumbnail, shape: {emo_frames_for_thumbnail.shape}")
                                    else:
                                        logger.debug(f"No EMO frame available - will use 3-panel thumbnail")

                                    thumbnail = generate_window_thumbnail(
                                        generated_frames=single_frame_generated,  # VASA generated frame
                                        target_frames=single_frame_target,        # Ground truth frame
                                        identity_frame=identity_for_thumbnail,    # Identity/source frame
                                        emo_generated_frames=emo_frames_for_thumbnail,  # EMO generated frame (4-panel if available)
                                        motion_outputs=stored_outputs,            # For motion stats overlay
                                        size=(1024, 256) if emo_frames_for_thumbnail is not None else (768, 256)  # Wider if 4-panel
                                    )

                                    # Log to wandb with more descriptive caption
                                    if thumbnail is not None:
                                        if single_frame_generated is not None:
                                            frame_info = f"Identity | Target | Predicted (frame {frame_idx} of T={stored_outputs['theta'].shape[1] if 'theta' in stored_outputs else 'unknown'})"
                                        else:
                                            frame_info = f"Identity | Target | (generation failed)"

                                        wandb.log({
                                            "visuals/training_thumbnail": wandb.Image(
                                                thumbnail,
                                                caption=f"Epoch {self.current_epoch}, Batch {batch_idx}, {frame_info}"
                                            )
                                        }, step=self.global_step)
                                        logger.info(f"📸 Generated and logged training thumbnail for epoch {self.current_epoch}")

                                        # Log warping field visualizations with target/predicted comparison
                                        try:
                                            if 'xy_warps' in motion_data and 'rigid_warps' in motion_data:
                                                import matplotlib.pyplot as plt
                                                import numpy as np

                                                # Get first batch, middle frame, middle depth slice
                                                b_idx = 0
                                                t_idx = motion_data['xy_warps'].shape[1] // 2  # Middle frame
                                                d_idx = 8  # Middle depth slice (16/2)

                                                # Check if we have predictions to compare (stored_outputs contains model predictions)
                                                has_predictions = stored_outputs and 'xy_warps' in stored_outputs

                                                if has_predictions:
                                                    # Create larger figure for target vs predicted comparison
                                                    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                                                else:
                                                    # Original layout for target only
                                                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                                                if has_predictions:
                                                    # === TARGET WARPS (Row 0) ===
                                                    # XY Warps (source non-rigid) - TARGET
                                                    xy_warp_target = motion_data['xy_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    xy_magnitude_target = np.linalg.norm(xy_warp_target, axis=-1)
                                                    im1 = axes[0, 0].imshow(xy_magnitude_target, cmap='viridis')
                                                    axes[0, 0].set_title(f'TARGET XY Warp (frame {t_idx}, depth {d_idx})')
                                                    axes[0, 0].axis('off')
                                                    plt.colorbar(im1, ax=axes[0, 0])

                                                    # Rigid Warps - TARGET
                                                    rigid_warp_target = motion_data['rigid_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    rigid_magnitude_target = np.linalg.norm(rigid_warp_target, axis=-1)
                                                    im2 = axes[0, 1].imshow(rigid_magnitude_target, cmap='plasma')
                                                    axes[0, 1].set_title(f'TARGET Rigid Warp')
                                                    axes[0, 1].axis('off')
                                                    plt.colorbar(im2, ax=axes[0, 1])

                                                    # UV Warps - TARGET
                                                    uv_warp_target = motion_data['uv_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    uv_magnitude_target = np.linalg.norm(uv_warp_target, axis=-1)
                                                    im3 = axes[0, 2].imshow(uv_magnitude_target, cmap='coolwarm')
                                                    axes[0, 2].set_title(f'TARGET UV Warp')
                                                    axes[0, 2].axis('off')
                                                    plt.colorbar(im3, ax=axes[0, 2])

                                                    # === PREDICTED WARPS (Row 1) ===
                                                    # XY Warps - PREDICTED
                                                    xy_warp_pred = stored_outputs['xy_warps'][b_idx, t_idx, d_idx].numpy()
                                                    xy_magnitude_pred = np.linalg.norm(xy_warp_pred, axis=-1)
                                                    im4 = axes[1, 0].imshow(xy_magnitude_pred, cmap='viridis')
                                                    axes[1, 0].set_title(f'PREDICTED XY Warp')
                                                    axes[1, 0].axis('off')
                                                    plt.colorbar(im4, ax=axes[1, 0])

                                                    # Rigid Warps - PREDICTED
                                                    rigid_warp_pred = stored_outputs['rigid_warps'][b_idx, t_idx, d_idx].numpy()
                                                    rigid_magnitude_pred = np.linalg.norm(rigid_warp_pred, axis=-1)
                                                    im5 = axes[1, 1].imshow(rigid_magnitude_pred, cmap='plasma')
                                                    axes[1, 1].set_title(f'PREDICTED Rigid Warp')
                                                    axes[1, 1].axis('off')
                                                    plt.colorbar(im5, ax=axes[1, 1])

                                                    # UV Warps - PREDICTED
                                                    uv_warp_pred = stored_outputs['uv_warps'][b_idx, t_idx, d_idx].numpy()
                                                    uv_magnitude_pred = np.linalg.norm(uv_warp_pred, axis=-1)
                                                    im6 = axes[1, 2].imshow(uv_magnitude_pred, cmap='coolwarm')
                                                    axes[1, 2].set_title(f'PREDICTED UV Warp')
                                                    axes[1, 2].axis('off')
                                                    plt.colorbar(im6, ax=axes[1, 2])

                                                    # === DIFFERENCE MAPS (Row 2) ===
                                                    # XY Warp Difference
                                                    xy_diff = np.abs(xy_magnitude_target - xy_magnitude_pred)
                                                    im7 = axes[2, 0].imshow(xy_diff, cmap='hot')
                                                    axes[2, 0].set_title(f'XY Warp Error (MAE: {xy_diff.mean():.4f})')
                                                    axes[2, 0].axis('off')
                                                    plt.colorbar(im7, ax=axes[2, 0])

                                                    # Rigid Warp Difference
                                                    rigid_diff = np.abs(rigid_magnitude_target - rigid_magnitude_pred)
                                                    im8 = axes[2, 1].imshow(rigid_diff, cmap='hot')
                                                    axes[2, 1].set_title(f'Rigid Warp Error (MAE: {rigid_diff.mean():.4f})')
                                                    axes[2, 1].axis('off')
                                                    plt.colorbar(im8, ax=axes[2, 1])

                                                    # UV Warp Difference
                                                    uv_diff = np.abs(uv_magnitude_target - uv_magnitude_pred)
                                                    im9 = axes[2, 2].imshow(uv_diff, cmap='hot')
                                                    axes[2, 2].set_title(f'UV Warp Error (MAE: {uv_diff.mean():.4f})')
                                                    axes[2, 2].axis('off')
                                                    plt.colorbar(im9, ax=axes[2, 2])

                                                    # === FLOW VISUALIZATIONS (Row 3) ===
                                                    flow_row = 3
                                                else:
                                                    # Original single visualization for targets only
                                                    xy_warp_target = motion_data['xy_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    xy_magnitude_target = np.linalg.norm(xy_warp_target, axis=-1)
                                                    im1 = axes[0, 0].imshow(xy_magnitude_target, cmap='viridis')
                                                    axes[0, 0].set_title(f'XY Warp Magnitude (frame {t_idx}, depth {d_idx})')
                                                    axes[0, 0].axis('off')
                                                    plt.colorbar(im1, ax=axes[0, 0])

                                                    rigid_warp_target = motion_data['rigid_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    rigid_magnitude_target = np.linalg.norm(rigid_warp_target, axis=-1)
                                                    im2 = axes[0, 1].imshow(rigid_magnitude_target, cmap='plasma')
                                                    axes[0, 1].set_title(f'Rigid Warp Magnitude')
                                                    axes[0, 1].axis('off')
                                                    plt.colorbar(im2, ax=axes[0, 1])

                                                    uv_warp_target = motion_data['uv_warps'][b_idx, t_idx, d_idx].cpu().numpy()
                                                    uv_magnitude_target = np.linalg.norm(uv_warp_target, axis=-1)
                                                    im3 = axes[0, 2].imshow(uv_magnitude_target, cmap='coolwarm')
                                                    axes[0, 2].set_title(f'UV Warp Magnitude')
                                                    axes[0, 2].axis('off')
                                                    plt.colorbar(im3, ax=axes[0, 2])

                                                    flow_row = 1

                                                # Warp flow visualization (X and Y components)
                                                axes[flow_row, 0].quiver(
                                                    np.arange(0, 64, 4), np.arange(0, 64, 4),
                                                    xy_warp_target[::4, ::4, 0], xy_warp_target[::4, ::4, 1],
                                                    angles='xy', scale_units='xy', scale=0.5, color='blue'
                                                )
                                                axes[flow_row, 0].set_title('XY Warp Flow (Target)')
                                                axes[flow_row, 0].set_xlim(0, 64)
                                                axes[flow_row, 0].set_ylim(64, 0)
                                                axes[flow_row, 0].set_aspect('equal')

                                                # Source theta warp visualization
                                                source_theta = motion_data['source_theta_warp'][b_idx, t_idx].cpu().numpy()  # [3, 4]
                                                axes[flow_row, 1].imshow(source_theta, cmap='RdBu', aspect='auto')
                                                axes[flow_row, 1].set_title(f'Source Theta Warp (frame {t_idx})')
                                                axes[flow_row, 1].set_xlabel('Coefficients')
                                                axes[flow_row, 1].set_ylabel('Dimensions')
                                                for i in range(3):
                                                    for j in range(4):
                                                        axes[flow_row, 1].text(j, i, f'{source_theta[i, j]:.2f}',
                                                                       ha='center', va='center', color='black')

                                                # Temporal warp variation (std across time) or predicted flow if available
                                                if has_predictions:
                                                    # Show predicted flow visualization
                                                    axes[flow_row, 2].quiver(
                                                        np.arange(0, 64, 4), np.arange(0, 64, 4),
                                                        xy_warp_pred[::4, ::4, 0], xy_warp_pred[::4, ::4, 1],
                                                        angles='xy', scale_units='xy', scale=0.5, color='red'
                                                    )
                                                    axes[flow_row, 2].set_title('XY Warp Flow (Predicted)')
                                                    axes[flow_row, 2].set_xlim(0, 64)
                                                    axes[flow_row, 2].set_ylim(64, 0)
                                                    axes[flow_row, 2].set_aspect('equal')
                                                else:
                                                    # Original temporal variation visualization
                                                    xy_temporal_std = torch.std(motion_data['xy_warps'][b_idx], dim=0).mean(dim=0).mean(dim=-1).cpu().numpy()
                                                    axes[flow_row, 2].imshow(xy_temporal_std, cmap='hot')
                                                    axes[flow_row, 2].set_title('XY Warp Temporal Variation (std)')
                                                    axes[flow_row, 2].axis('off')

                                                # Add overall title
                                                if has_predictions:
                                                    fig.suptitle('Warping Fields: Target vs Predicted Comparison', fontsize=16, y=1.02)
                                                else:
                                                    fig.suptitle('Warping Fields: Target Only', fontsize=16, y=1.02)

                                                plt.tight_layout()
                                                wandb.log({"visuals/warping_fields": wandb.Image(fig)}, step=self.global_step)
                                                plt.close(fig)

                                                if has_predictions:
                                                    logger.info("📊 Logged warping field comparison (target vs predicted)")
                                                else:
                                                    logger.info("📊 Logged warping field visualizations (target only)")

                                        except Exception as e:
                                            logger.warning(f"Could not visualize warping fields: {e}")
                                    else:
                                        logger.warning("Thumbnail generation returned None")

                                except Exception as e:
                                    logger.warning(f"Could not generate thumbnail: {e}")

                        except Exception as e:
                            logger.error(f"Error processing window {window_idx}: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue

                # After processing all windows in the batch, do gradient clipping and optimizer step
                # BUT ONLY if we actually processed at least one window (did at least one backward)
                if windows_processed == 0:
                    logger.warning(f"⚠️ Batch {batch_idx}: All {len(windows)} windows were skipped (poor quality). Skipping optimizer step.")
                    continue  # Skip to next batch

                try:
                    # Update gradient monitor if enabled (before clipping to see raw gradients)
                    if self.config.train.gradient_monitoring.enabled and self.accelerator.sync_gradients:
                        self.grad_monitor.update()
                    
                    # Compute gradient norm and clip gradients
                    grad_norm = None
                    if self.accelerator.sync_gradients:
                        # Check if mixed precision is enabled
                        use_amp = getattr(self.config.motion, 'amp', False)
                        
                        if use_amp:
                            # Use accelerator's clipping for mixed precision
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.max_grad_norm
                            )
                        else:
                            # Use direct gradient clipping when mixed precision is off
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.train.max_grad_norm
                            )

                        # ADDITIONAL: Clip gradient values to prevent extreme spikes
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)

                        # Log gradient norm for monitoring
                        if grad_norm is not None and wandb.run is not None:
                            wandb.log({
                                'gradients/norm': grad_norm,
                            }, step=self.global_step)
                        
                        # Check for gradient explosion
                        if grad_norm is not None:
                            if grad_norm > 1e6 or not torch.isfinite(torch.tensor(grad_norm)):
                                logger.warning(f"Large/infinite gradient norm detected: {grad_norm:.2e}")
                                
                                # Log explosion event to wandb
                                if wandb.run is not None:
                                    wandb.log({
                                        'gradient_explosion/large_grad_norm': grad_norm,
                                        'gradient_explosion/event': 1.0,
                                    }, step=self.global_step)
                                
                                # Skip optimizer step if gradient is infinite/nan
                                if not torch.isfinite(torch.tensor(grad_norm)):
                                    logger.error(f"Skipping optimizer step due to infinite gradient at step {self.global_step}")
                                    self.optimizer.zero_grad()
                                    
                                    # Reset scaler state if using mixed precision
                                    if use_amp and hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None:
                                        self.accelerator.scaler.update()  # Reset scaler state
                                        logger.info("Reset gradient scaler state after infinite gradient")
                                    
                                    continue  # Skip to next batch
                        else:
                            # If grad_norm is None but we have gradients, compute it manually
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    param_norm = p.grad.data.norm(2).item()
                                    total_norm += param_norm ** 2
                            grad_norm = total_norm ** 0.5
                            
                            if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm > 1e6:
                                logger.error(f"Computed gradient norm is invalid: {grad_norm}. Skipping optimizer step.")
                                self.optimizer.zero_grad()
                                
                                # Reset scaler state if using mixed precision
                                use_amp = getattr(self.config.motion, 'amp', False)
                                if use_amp and hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None:
                                    self.accelerator.scaler.update()  # Reset scaler state
                                    logger.info("Reset gradient scaler state after invalid computed gradient norm")
                                
                                continue
                    
                    # Optimizer step after ALL windows processed
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                except Exception as e:
                    logger.error(f"Error during optimizer step: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Clear gradients and continue training
                    self.optimizer.zero_grad()
                    
                    # Reset scaler state if using mixed precision and an error occurred
                    use_amp = getattr(self.config.motion, 'amp', False)
                    if use_amp and hasattr(self.accelerator, 'scaler') and self.accelerator.scaler is not None:
                        try:
                            self.accelerator.scaler.update()  # Reset scaler state
                            logger.info("Reset gradient scaler state after optimizer error")
                        except Exception as scaler_error:
                            logger.warning(f"Could not reset scaler: {scaler_error}")
                    
                    continue
                
                # Average metrics across windows and immediately convert to scalars
                avg_metrics = {}
                for k, v in batch_metrics.items():
                    if isinstance(v[0], torch.Tensor):
                        # Stack, compute mean, and immediately convert to scalar
                        stacked = torch.stack(v)
                        avg_metrics[k] = stacked.mean().item()
                        del stacked  # Free memory immediately
                    else:
                        avg_metrics[k] = sum(v) / len(v)
                avg_metrics['total'] = batch_total_loss / len(windows)
                
                # Clear batch_metrics to free memory
                del batch_metrics
                
                # Update metrics
                self.train_metrics.update(avg_metrics)
                
                # Update epoch metrics
                for k, v in avg_metrics.items():
                    epoch_metrics[k].append(v)
                        
                # Log metrics and gradients
                if batch_idx % self.config.train.gradient_monitoring.log_freq == 0:
                    # Get gradient statistics from monitor
                    grad_stats = {}
                    if self.config.train.gradient_monitoring.enabled:
                        grad_stats = self.grad_monitor.get_stats()
                        # Reset monitor for next logging period
                        self.grad_monitor.reset()
                    
                    self._log_training_stats(
                        batch_idx, 
                        num_batches,
                        avg_metrics,
                        grad_norms,
                        grad_stats
                    )
                
                # Periodic CUDA cache clearing to prevent memory fragmentation
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    if batch_idx % 50 == 0:
                        # Log memory stats every 50 batches
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Update progress bar
                self.progress_bar.update(1)
                self.global_step += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
                    
        # Compute epoch averages
        epoch_averages = {
            k: sum(v) / len(v) for k, v in epoch_metrics.items()
        }
        
        # Store for TDD progressive loss
        self.last_epoch_metrics = epoch_averages
        
        # Log epoch metrics with stage information
        if self.config.wandb.enabled and self.accelerator.is_local_main_process:
            # Determine current training stage
            current_stage = 1
            stage_name = "Foundation"
            if self.current_epoch >= 45:
                current_stage = 6
                stage_name = "Lip Sync Refinement"
            elif self.current_epoch >= 35:
                current_stage = 5
                stage_name = "Emotion"
            elif self.current_epoch >= 25:
                current_stage = 4
                stage_name = "Head Pose"
            elif self.current_epoch >= 15:
                current_stage = 3
                stage_name = "Eye Gaze"
            elif self.current_epoch >= 5:
                current_stage = 2
                stage_name = "Blinking"

            # Log metrics with stage info
            log_dict = {f"epoch/{k}": v for k, v in epoch_averages.items()}
            log_dict["training/stage"] = current_stage
            log_dict["training/stage_name"] = stage_name

            # Log specific lip motion metrics with correct keys
            log_dict["lip_motion/audio_lip_correlation"] = epoch_averages.get('audio_lip_correlation', 0.0)
            log_dict["lip_motion/mouth_openness"] = epoch_averages.get('mouth_openness_direct', 0.0)
            log_dict["lip_motion/expression_loss"] = epoch_averages.get('expression_loss', 0.0)

            # Log loss monitoring statistics
            loss_stats = self.loss_monitor.get_statistics()
            log_dict["loss_monitoring/total_warnings"] = loss_stats['total_warnings']
            log_dict["loss_monitoring/total_criticals"] = loss_stats['total_criticals']
            if loss_stats['warning_counts']:
                for loss_name, count in loss_stats['warning_counts'].items():
                    log_dict[f"loss_monitoring/warnings_{loss_name}"] = count
            if loss_stats['critical_counts']:
                for loss_name, count in loss_stats['critical_counts'].items():
                    log_dict[f"loss_monitoring/criticals_{loss_name}"] = count

            wandb.log(log_dict, step=self.global_step)
            
            # Log the metrics table for this epoch
            wandb.log({"epoch_metrics_table": self.epoch_table}, step=self.global_step)

            # Also add to wandb summary so it persists
            wandb.run.summary["epoch_metrics_table"] = self.epoch_table
            
            # Add alert if loss is too high (for overfit detection)
            if epoch_averages.get('total', 0) > 50.0:
                wandb.alert(
                    title="High Training Loss",
                    text=f"Total loss is {epoch_averages.get('total', 0):.2f} at epoch {self.current_epoch}",
                    level=wandb.AlertLevel.WARN
                )

        # Log learning rates
        self._log_learning_rates(self.global_step)
        
        self.progress_bar.close()
        return epoch_averages
    def _log_learning_rates(self, step: Optional[int] = None):
        """
        Internal method to log learning rates for each parameter group.
        Args:
            step: Optional current training step for wandb logging
        """
        try:
            if not self.accelerator.is_local_main_process:
                return
                
            logger.info("\n=== Learning Rates ===")
            lr_dict = {}

            # Log each parameter group's learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                current_lr = param_group['lr']
                
                # Log to console
                logger.info(f"{group_name:30} LR: {current_lr:.2e}")
                
                # Add to dict for wandb
                lr_dict[f'lr/{group_name}'] = current_lr

            # Log min and max learning rates
            all_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            lr_dict.update({
                'lr/min': min(all_lrs),
                'lr/max': max(all_lrs),
                'lr/mean': sum(all_lrs) / len(all_lrs)
            })

            # Log parameter statistics for each group
            param_stats = {}
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                param_norms = []
                grad_norms = []
                
                for param in param_group['params']:
                    if param.requires_grad:
                        # Parameter norm
                        param_norm = param.data.norm(2).item()
                        param_norms.append(param_norm)
                        
                        # Gradient norm (if gradient exists)
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            grad_norms.append(grad_norm)
                
                if param_norms:
                    param_stats.update({
                        f'params/{group_name}/norm_mean': np.mean(param_norms),
                        f'params/{group_name}/norm_std': np.std(param_norms)
                    })
                
                if grad_norms:
                    param_stats.update({
                        f'grads/{group_name}/norm_mean': np.mean(grad_norms),
                        f'grads/{group_name}/norm_std': np.std(grad_norms)
                    })

            # Log everything to wandb if enabled
            if self.config.wandb.enabled and step is not None:
                combined_metrics = {
                    'step': step,
                    **lr_dict,
                    **param_stats
                }
                wandb.log(combined_metrics, step=step)

            return lr_dict, param_stats

        except Exception as e:
            logger.error(f"Error logging learning rates: {str(e)}")
            logger.error(traceback.format_exc())
            return {}, {}

    def log_learning_rates(self, step: int):
        """
        Public method for logging learning rates.
        Just calls internal _log_learning_rates method.
        """
        return self._log_learning_rates(step)
        
    def _get_cfg_scales(self) -> Optional[Dict[str, float]]:
        """Get current CFG scales based on training progress."""
        if self.current_epoch < self.config.diffusion.cfg_start_epoch:
            return None
            
        # Linearly increase scales from 0 to max over 10 epochs
        ramp_epochs = 10
        progress = min(1.0, (self.current_epoch - self.config.diffusion.cfg_start_epoch) / ramp_epochs)
        
        return {
            'audio': self.config.train.cfg_scales.audio * progress,
            'gaze': self.config.train.cfg_scales.gaze * progress,
            'head_distance': self.config.train.cfg_scales.head_distance * progress,
            'emotion': self.config.train.cfg_scales.emotion * progress
        }


    def log_ddim_plots(self, motion_data, step):
        """Create DDIM visualizations for wandb"""
        
        # 1. Noise Schedule Plot
        fig1 = plt.figure(figsize=(10,5))
        plt.plot(self.scheduler.alphas_cumprod.cpu().numpy(), label='Signal')
        plt.plot(1 - self.scheduler.alphas_cumprod.cpu().numpy(), label='Noise')
        plt.title('DDIM Noise Schedule')
        plt.legend()
        wandb.log({"ddim/noise_schedule": wandb.Image(fig1)}, step=step)

        # 2. Motion Parameter Distributions
        for key in motion_data:
            fig2 = plt.figure(figsize=(8,4))
            values = motion_data[key].detach().cpu().numpy()
            plt.hist(values.flatten(), bins=50, alpha=0.7)
            plt.title(f'{key} Distribution')
            wandb.log({f"ddim/dist_{key}": wandb.Image(fig2)}, step=step)

        # 3. Denoising Progress Table
        denoising_table = wandb.Table(columns=["timestep", "snr", "loss"])
        for t, snr, loss in zip(timesteps, signal_noise_ratios, losses):
            denoising_table.add_data(t, snr, loss)
        wandb.log({"ddim/denoising_progress": denoising_table})
        


    def log_ddim_metrics(self, timesteps, noised_motion, prediction, denoised, step):
        """Log DDIM-specific metrics to wandb"""
        
        # 1. Noise Schedule Progress
        wandb.log({
            'ddim/alpha_cumulative': self.scheduler.alphas_cumprod[timesteps].mean(),
            'ddim/noise_level': timesteps.float().mean(),
        }, step=step)

        # 2. Signal-to-Noise Ratio for each motion parameter
        for key in ['theta', 'rotation', 'scale', 'translation', 'expression_embed']:
            if key in noised_motion:
                signal = noised_motion[key].abs().mean()
                noise = (noised_motion[key] - denoised[key]).abs().mean()
                snr = signal / (noise + 1e-8)
                wandb.log({
                    f'ddim/snr_{key}': snr,
                    f'ddim/signal_{key}': signal,
                    f'ddim/noise_{key}': noise
                }, step=step)

        # 3. Important Stats per timestep range
        ranges = [(0,200), (200,400), (400,600), (600,800), (800,1000)]
        for start, end in ranges:
            mask = (timesteps >= start) & (timesteps < end)
            if mask.any():
                wandb.log({
                    f'ddim/loss_t{start}-{end}': loss[mask].mean(),
                    f'ddim/pred_norm_t{start}-{end}': prediction[mask].norm(2).mean()
                }, step=step)

        # 4. Denoising Quality
        wandb.log({
            'ddim/prediction_mse': F.mse_loss(prediction, target),
            'ddim/denoised_motion_diff': F.l1_loss(denoised, original_motion)
        }, step=step)




    def _log_training_stats(
        self,
        batch_idx: int,
        num_batches: int,
        metrics: Dict[str, float],
        grad_norms: Dict[str, List[float]],
        grad_stats: Optional[Dict[str, float]] = None
    ):
        """Log detailed training statistics."""
        if not self.config.wandb.enabled or not self.accelerator.is_local_main_process:
            return
            
        # Format metrics for logging
        step_metrics = {
            # Basic progress metrics
            'train/epoch': self.current_epoch,
            'train/step': self.global_step,
            'train/progress': (batch_idx + 1) / num_batches,
            'train/learning_rate': self.scheduler.get_last_lr()[0],
            
            # Loss components
            'train/total_loss': metrics.get('total', 0.0),
            'train/reconstruction_loss': metrics.get('reconstruction', 0.0),
            'train/control_loss': metrics.get('control_total', 0.0)
        }
        
        # Add component losses
        for k, v in metrics.items():
            if k.startswith('control_') or k.startswith('metric_') or 'warp' in k:
                step_metrics[f'train/{k}'] = v
                
        # Add gradient statistics from manual tracking
        if grad_norms:
            grad_norm_summary = {
                'min': min(min(norms) for norms in grad_norms.values()),
                'max': max(max(norms) for norms in grad_norms.values()),
                'mean': np.mean([np.mean(norms) for norms in grad_norms.values()]),
                'std': np.std([np.std(norms) for norms in grad_norms.values()])
            }
            
            for k, v in grad_norm_summary.items():
                step_metrics[f'train/grad_{k}'] = v
        
        # Add detailed gradient statistics from GradientMonitor
        if grad_stats:
            # Log overall gradient statistics
            for k, v in grad_stats.items():
                if not k.startswith('grad_norm_') and not k.startswith('grad_mean_') and not k.startswith('grad_var_'):
                    step_metrics[f'gradients/{k}'] = v
            
            # Log per-layer statistics (only for key layers to avoid too many metrics)
            key_layers = ['motion_transformer', 'condition_embedding', 'output_projection']
            for layer in key_layers:
                for k, v in grad_stats.items():
                    if layer in k:
                        step_metrics[f'gradients/layers/{k}'] = v
                
        # Log to wandb
        wandb.log(step_metrics, step=self.global_step)
        
        # Update progress bar
        self.progress_bar.set_postfix(
            **{k.split('/')[-1]: f"{v:.4f}" for k, v in step_metrics.items() 
            if isinstance(v, (float, int))}
        )

    def _extract_features_and_embeddings(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract source identity features and canonical volume."""
        try:
            logger.debug("\n=== Feature Extraction Start ===")
            logger.debug(f"Input source_img shape: {data_dict['source_img'].shape}")

            # 1. Generate masks using MODNet
            identity_image = data_dict['source_img']
            _, _, source_mask_modnet = self.worker_state.modnet(identity_image, True)
            source_mask_modnet = source_mask_modnet.to(data_dict['source_img'].device)
            logger.debug(f"MODNet mask shape: {source_mask_modnet.shape}")
            
            # Get face parsing mask as fallback
            face_mask_source, _, _, _ = self.model.volumetric_avatar.face_idt.forward(data_dict['source_img'])
            face_mask_source = (face_mask_source > 0.6).float()
            logger.debug(f"Face parsing mask shape: {face_mask_source.shape}")
            
            # Combine masks
            source_mask = source_mask_modnet if self.model.volumetric_avatar.args.use_modnet_mask else face_mask_source
            logger.debug(f"Combined mask shape: {source_mask.shape}")
            
            # Apply mask to source image
            source_masked = data_dict['source_img'] * source_mask
            logger.debug(f"Masked source image shape: {source_masked.shape}")
            
            # 2. Extract identity embeddings
            data_dict['idt_embed'] = self.model.volumetric_avatar.idt_embedder_nw(source_masked)
            logger.debug(f"Identity embedding shape: {data_dict['idt_embed'].shape}")
            
            # 3. Extract base latent features
            source_latents = self.model.volumetric_avatar.local_encoder_nw(source_masked)
            logger.debug(f"Source latents shape: {source_latents.shape}")
            
            # 4. Process source volume
            B = source_latents.shape[0]
            source_latent_volume = source_latents.view(
                B,
                self.model.volumetric_avatar.args.latent_volume_channels,
                self.model.volumetric_avatar.args.latent_volume_depth,
                self.model.volumetric_avatar.args.latent_volume_size,
                self.model.volumetric_avatar.args.latent_volume_size
            )
            logger.debug(f"Reshaped source volume shape: {source_latent_volume.shape}")
            
            if self.model.volumetric_avatar.args.source_volume_num_blocks > 0:
                source_latent_volume = self.model.volumetric_avatar.volume_source_nw(source_latent_volume)
                logger.debug(f"Processed source volume shape: {source_latent_volume.shape}")
                
            # 5. Process canonical volume - skip expression embedding
            if self.model.volumetric_avatar.args.unet_first:
                logger.debug("Processing with UNet first...")
                canonical_volume = self.model.volumetric_avatar.volume_process_nw(source_latent_volume)
            else:
                logger.debug("Using source volume as canonical...")
                canonical_volume = source_latent_volume

            logger.debug(f"Final canonical volume shape: {canonical_volume.shape}")
            logger.debug("=== Feature Extraction Complete ===\n")
            return canonical_volume

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _generate_synced_frames(self, identity_image: torch.Tensor, motion_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate synchronized frames using extracted features and motion parameters.
        Uses proper non-rigid warps from compute_warps_from_zdyn.

        Args:
            identity_image: Source identity image tensor [B, C, H, W] or [C, H, W]
            motion_outputs: Dictionary containing predicted motion parameters

        Returns:
            Generated frames tensor [B, T, C, H, W]
        """
        try:
            # Ensure identity_image has batch dimension
            if len(identity_image.shape) == 3:
                identity_image = identity_image.unsqueeze(0)

            # Get dimensions
            motion_batch_size = motion_outputs['theta'].size(0)
            seq_len = motion_outputs['theta'].size(1)
            device = identity_image.device
            logger.debug(f"Motion batch size: {motion_batch_size}, sequence length: {seq_len}")

            # Use pre-computed identity embedding (MEMORY OPTIMIZATION)
            # We always have this from initialization if using derived warps
            if self.idt_embed is None:
                raise ValueError("idt_embed not pre-computed! This should be set during trainer initialization.")

            idt_embed = self.idt_embed
            logger.debug(f"Using pre-computed idt_embed: {idt_embed.shape}")

            # Compute canonical volume once
            try:
                # Extract features for first image only
                canonical_volume = self._extract_features_and_embeddings({
                    'source_img': identity_image[0:1]
                })

                # Expand to match batch size
                canonical_volume = canonical_volume.expand(motion_batch_size, -1, -1, -1, -1)
                logger.debug(f"Expanded canonical volume shape: {canonical_volume.shape}")

                # Initialize list for generated frames
                generated_frames = []

                # Process each timestep
                for t in range(seq_len):
                    # Get motion parameters for current timestep
                    curr_theta = motion_outputs['theta'][:, t:t+1]  # [B, 1, 3, 4] - keep T dim
                    curr_expression = motion_outputs['expression_embed'][:, t:t+1]  # [B, 1, 128]

                    # Generate proper non-rigid warps using nemo's pipeline
                    uv_warp = self.model.compute_warps_from_zdyn(
                        zdyn=curr_expression,  # [B, 1, 128]
                        idt_embed=idt_embed,   # [1, 512]
                        theta=curr_theta       # [B, 1, 3, 4]
                    )  # Returns [B, 1, 16, 64, 64, 3]

                    # Remove T dimension and get single timestep warp
                    uv_warp = uv_warp[:, 0]  # [B, 16, 64, 64, 3]

                    # Apply non-rigid UV warping to canonical volume
                    warped_volume = self.model.volumetric_avatar.grid_sample(canonical_volume, uv_warp)

                    # Reshape for decoder
                    c = self.model.volumetric_avatar.args.latent_volume_channels
                    d = self.model.volumetric_avatar.args.latent_volume_depth
                    s = self.model.volumetric_avatar.args.latent_volume_size
                    latent_feats = warped_volume.view(motion_batch_size, c * d, s, s)

                    # Generate frame using decoder
                    frame, _, _, _ = self.model.volumetric_avatar.decoder_nw(
                        {'expression_embed': curr_expression[:, 0]},  # Remove T dim for decoder
                        None,
                        latent_feats,
                        False
                    )

                    # Save frame (detach to avoid memory leak)
                    generated_frames.append(frame.detach())

                    # Clear intermediate tensors to free memory
                    del uv_warp, warped_volume, latent_feats

                # Stack frames along time dimension [B, T, C, H, W]
                generated_frames = torch.stack(generated_frames, dim=1)
                logger.debug(f"Final generated frames shape: {generated_frames.shape}")

                return generated_frames

            finally:
                # Clean up canonical volume
                del canonical_volume
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating synced frames: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        


    
    def validate(self) -> Dict[str, float]:
        """Run validation with proper model mode handling."""
        if not self.val_loader:
            return {}

        # Check if validation is enabled
        val_config = self.config.get('validation', {})
        if not val_config.get('enabled', True):
            logger.info("Validation is disabled in config")
            return {}

        # Put model in eval mode
        self.model.eval()
        self.val_metrics.reset()

        # Get validation settings
        max_batches = val_config.get('max_batches', None)
        skip_expensive = val_config.get('skip_expensive_metrics', False)
        allowed_metrics = val_config.get('metrics', ['reconstruction_loss'])

        logger.info(f"Validation settings: max_batches={max_batches}, skip_expensive={skip_expensive}, metrics={allowed_metrics}")
        
        cfg_scales = self._get_cfg_scales()  # Get current CFG scales

        with torch.no_grad():
            # Limit number of batches if configured
            val_iterator = enumerate(self.val_loader)
            if max_batches:
                val_iterator = itertools.islice(val_iterator, max_batches)
                total_batches = min(max_batches, len(self.val_loader))
            else:
                total_batches = len(self.val_loader)

            for batch_idx, batch in tqdm(
                val_iterator,
                total=total_batches,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Validation (max {max_batches} batches)" if max_batches else "Validation"
            ):
                try:
                    if batch is None:
                        continue

                    # Debug: Log batch keys to understand structure
                    if batch_idx == 0:  # Only log once
                        logger.info(f"Validation batch keys: {list(batch.keys())}")
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                logger.info(f"  {key}: shape {value.shape}")
                            elif isinstance(value, list):
                                logger.info(f"  {key}: list of length {len(value)}")

                    # For validation, the batch already contains windows from collate_vasa_batch
                    # We need to restructure it for motion_handler or bypass it
                    if 'theta' in batch and isinstance(batch['theta'], torch.Tensor):
                        # Batch already has motion data stacked, create window-like structure
                        B = batch['theta'].shape[0] if 'theta' in batch else 1
                        windows = []
                        for b in range(B):
                            window = {}
                            for key, value in batch.items():
                                if isinstance(value, torch.Tensor) and value.shape[0] >= b + 1:
                                    # Remove the extra dimension if present from stacking
                                    if value.ndim > 2 and value.shape[1] == 1:
                                        window[key] = value[b, 0]  # Remove batch and squeeze singleton
                                    else:
                                        window[key] = value[b:b+1]  # Keep batch dimension
                                elif key == 'metadata' and isinstance(value, list) and len(value) > b:
                                    window['metadata'] = value[b]
                            windows.append(window)

                        if not windows:
                            logger.warning("No valid windows extracted from validation batch")
                            continue
                    else:
                        # Try to process normally if batch has expected structure
                        if 'frames' not in batch:
                            # Skip if no frames to process
                            logger.warning("No 'frames' key in validation batch, skipping")
                            continue

                        windows = self.motion_handler.process_batch(
                            batch,
                            current_window_size=self.config.motion.window_size
                        )
                        if not windows:
                            logger.warning("No valid windows in validation batch")
                            continue

                    # Process each window independently
                    for window in windows:
                        try:
                            # Generate sequence with CFG
                            generated_sequence = self.model.forward(
                                motion_data={
                                    'theta': window['theta'],
                                    'scale': window['scale'],
                                    'rotation': window['rotation'],
                                    'translation': window['translation'],
                                    'expression_embed': window['expression_embed']
                                },
                                noise_level=torch.zeros(1, device=self.accelerator.device),
                                conditions={
                                    'audio_features': window['audio_features'],
                                    'gaze': window.get('gaze'),
                                    'head_distance': window.get('head_distance'),
                                    'emotion': window.get('emotion'),
                                    'speed_bucket': window.get('speed_bucket'),
                                    'lips': window.get('lips'),
                                    'right_eye': window.get('right_eye'),
                                    'left_eye': window.get('left_eye'),
                                    'jaw': window.get('jaw'),
                                    'nose': window.get('nose'),
                                    'blink_state': window.get('blink_state')
                                },
                    
                            )

                            # Compute metrics for this window
                            metrics = {}

                            # Always compute reconstruction metrics (they're fast)
                            if 'reconstruction_loss' in allowed_metrics or not allowed_metrics:
                                metrics.update(self.loss_module._compute_reconstruction_losses(
                                    generated_sequence, window, None
                                ))

                            # Control metrics if applicable and not skipping expensive
                            if not skip_expensive and self.current_epoch >= self.config.train.control_start_epoch:
                                if 'control_loss' in allowed_metrics or 'motion_naturalness' in allowed_metrics:
                                    control_metrics = self.loss_module._compute_control_losses(
                                        generated_sequence, window, self.current_epoch
                                    )
                                    metrics.update(control_metrics)

                            # Generate frames for sync evaluation if needed (expensive!)
                            if not skip_expensive and self.config.loss.use_sync_loss:
                                if 'lip_sync' in allowed_metrics or 'id_similarity' in allowed_metrics:
                                    try:
                                        generated_frames = self._generate_synced_frames(
                                            window['frames'][:, 0],  # Use first frame as identity
                                            generated_sequence
                                        )
                                        # Evaluate sync quality
                                        sync_metrics = self.loss_module.evaluate_sync_quality(
                                            generated_frames=generated_frames,
                                            audio_features=window['audio_features'],
                                            audio_mfcc=window.get('audio_mfcc')
                                        )
                                        metrics.update(sync_metrics)

                                    except Exception as e:
                                        logger.error(f"Error generating frames: {str(e)}")
                                        logger.error(traceback.format_exc())

                            # Update validation metrics
                            self.val_metrics.update(metrics)

                        except Exception as e:
                            logger.error(f"Error processing validation window: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue

                except Exception as e:
                    logger.error(f"Error processing validation batch: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Log validation metrics
            if self.config.wandb.enabled and self.accelerator.is_local_main_process:
                val_metrics = self.val_metrics.get_averages()
                wandb.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=self.global_step
                )

            return self.val_metrics.get_averages()
            
    def _apply_condition_dropout(
        self,
        conditions: Dict[str, torch.Tensor],
        dropout_probs: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Apply random dropout to conditions based on config probabilities.

        IMPORTANT: audio_features is NEVER dropped as it's required for the model.
        """
        dropped_conditions = {}
        for k, v in conditions.items():
            # NEVER drop audio_features - it's required!
            if k == 'audio_features':
                dropped_conditions[k] = v
            elif k in dropout_probs and v is not None:
                if random.random() < dropout_probs[k]:
                    dropped_conditions[k] = None
                else:
                    dropped_conditions[k] = v
            else:
                dropped_conditions[k] = v
        return dropped_conditions



    def log_learning_rates(self, step: int):
        """Log learning rates for each parameter group to wandb and console."""
        try:
            if not self.accelerator.is_local_main_process:
                return
                
            logger.info("\n=== Learning Rates ===")
            lr_dict = {}

            # Log each parameter group's learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                current_lr = param_group['lr']
                
                # Log to console
                logger.info(f"{group_name:30} LR: {current_lr:.2e}")
                
                # Add to dict for wandb
                lr_dict[f'lr/{group_name}'] = current_lr

            # Log min and max learning rates
            all_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            lr_dict.update({
                'lr/min': min(all_lrs),
                'lr/max': max(all_lrs),
                'lr/mean': sum(all_lrs) / len(all_lrs)
            })

            # Log parameter statistics for each group
            param_stats = {}
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', f'group_{i}')
                param_norms = []
                grad_norms = []
                
                for param in param_group['params']:
                    if param.requires_grad:
                        # Parameter norm
                        param_norm = param.data.norm(2).item()
                        param_norms.append(param_norm)
                        
                        # Gradient norm (if gradient exists)
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            grad_norms.append(grad_norm)
                
                if param_norms:
                    param_stats.update({
                        f'params/{group_name}/norm_mean': np.mean(param_norms),
                        f'params/{group_name}/norm_std': np.std(param_norms)
                    })
                
                if grad_norms:
                    param_stats.update({
                        f'grads/{group_name}/norm_mean': np.mean(grad_norms),
                        f'grads/{group_name}/norm_std': np.std(grad_norms)
                    })

            # Log everything to wandb
            if self.config.wandb.enabled:
                combined_metrics = {
                    'step': step,
                    **lr_dict,
                    **param_stats
                }
                self.accelerator.log(combined_metrics)

        except Exception as e:
            logger.error(f"Error logging learning rates: {str(e)}")
            logger.error(traceback.format_exc())
            
 
    def _log_step_metrics(self, batch_idx: int, num_batches: int):
        """Log metrics for current training step with wandb integration."""
        if not self.config.wandb.enabled or not self.accelerator.is_local_main_process:
            return
                
        metrics = self.train_metrics.get_averages()
        
        # Format metrics for logging
        step_metrics = {
            # Basic progress metrics
            'train/epoch': self.current_epoch,
            'train/step': self.global_step,
            'train/progress': (batch_idx + 1) / num_batches,
            'train/learning_rate': self.scheduler.get_last_lr()[0],
            
            # Control losses - ensure values are tensors
            'train/control_distance': metrics.get('control_distance', 0.0),
            'train/control_emotion': metrics.get('control_emotion', 0.0), 
            'train/control_gaze': metrics.get('control_gaze', 0.0),
            'train/control_speed': metrics.get('control_speed', 0.0),
            'train/control_total': metrics.get('control_total', 0.0),
            
            # Reconstruction losses
            'train/reconstruction': metrics.get('reconstruction', 0.0),
            'train/pose_loss': metrics.get('pose_loss', 0.0),
            'train/dynamics_loss': metrics.get('dynamics_loss', 0.0),
            'train/motion_loss': metrics.get('motion_loss', 0.0),
            
            # Component losses
            'train/theta_loss': metrics.get('theta_loss', 0.0),
            'train/rotation_loss': metrics.get('rotation_loss', 0.0),
            'train/translation_loss': metrics.get('translation_loss', 0.0),
            'train/expression_loss': metrics.get('expression_loss', 0.0)
        }

        # Add speed metrics if available
        if 'speed_loss' in metrics:
            step_metrics.update({
                'train/speed_loss': metrics['speed_loss'],
                'train/speed_accuracy': metrics.get('speed_accuracy', 0.0)
            })

        # Log to wandb using accelerator
        logger.debug("Logging metrics to wandb:")
        for k, v in step_metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.debug(f"  {k}: {v:.6f}")
        
        wandb.log(step_metrics, step=self.global_step)


        # Update progress bar
        self.progress_bar.set_postfix(
            **{k.split('/')[-1]: f"{v:.4f}" for k, v in step_metrics.items() 
            if isinstance(v, (float, int))}
        )


    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint without volumetric avatar.
        
        Args:
            is_best: If True, saves as best model checkpoint
        """
        if self.output_dir is None:
            return

        try:
            # Get unwrapped model state dict
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()

            # Filter out volumetric_avatar parameters (it's frozen and pre-trained)
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('volumetric_avatar.')
            }

            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': filtered_state_dict,  # Exclude volumetric_avatar
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'loss': self.best_val_loss,
                'config': self.config
            }

            # Always save as best_checkpoint.pt
            save_path = self.output_dir / 'best_checkpoint.pt'

            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            # Log saved parameters (count from filtered state dict)
            total_params = sum(p.numel() for p in unwrapped_model.parameters())
            saved_params = sum(v.numel() for v in filtered_state_dict.values())
            excluded_params = sum(v.numel() for k, v in state_dict.items() if k.startswith('volumetric_avatar.'))

            logger.info(f"Total model parameters: {total_params:,}")
            logger.info(f"Saved parameters: {saved_params:,}")
            logger.info(f"Excluded volumetric_avatar parameters: {excluded_params:,}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
    
    def save_epoch_checkpoint(self, save_path: Path):
        """Save checkpoint for current epoch (called every epoch).

        Args:
            save_path: Path to save the checkpoint
        """
        if self.output_dir is None:
            return

        try:
            # Get unwrapped model state dict
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = unwrapped_model.state_dict()

            # Filter out volumetric_avatar parameters (consistent with save_checkpoint)
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('volumetric_avatar.')
            }

            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': filtered_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'loss': self.best_val_loss,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f"💾 Saved epoch checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving epoch checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
    
    def cleanup_old_epoch_checkpoints(self, keep_last: int = 3):
        """Remove old epoch checkpoints to save disk space.
        
        Args:
            keep_last: Number of recent epoch checkpoints to keep
        """
        if self.output_dir is None:
            return
            
        try:
            # Find all epoch checkpoint files
            epoch_checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
            
            if len(epoch_checkpoints) > keep_last:
                # Remove old checkpoints
                for checkpoint_path in epoch_checkpoints[:-keep_last]:
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {str(e)}")
            

    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint with diffusion schedule handling"""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            # PyTorch 2.6 requires weights_only=False for checkpoints with configs
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device, weights_only=False)

            # Get original diffusion schedule configuration
            old_steps = checkpoint['config'].diffusion.num_steps
            new_steps = self.config.diffusion.num_steps

            if old_steps != new_steps:
                logger.info(f"Adjusting diffusion schedule from {old_steps} to {new_steps} steps")

                # Remove diffusion scheduler buffers - they'll be reinitialized
                skip_keys = [
                    'scheduler.alpha_cumprod',
                    'scheduler.betas',
                    'scheduler.alphas',
                    'scheduler.alphas_cumprod_prev',
                    'scheduler.sqrt_alphas_cumprod',
                    'scheduler.sqrt_one_minus_alphas_cumprod',
                    'scheduler.log_one_minus_alphas_cumprod',
                    'scheduler.sqrt_recip_alphas_cumprod',
                    'scheduler.sqrt_recipm1_alphas_cumprod',
                    'scheduler.posterior_variance'
                ]

                filtered_state_dict = {
                    k: v for k, v in checkpoint['model_state_dict'].items()
                    if not any(skip_key in k for skip_key in skip_keys)
                }

                # Load filtered state dict
                self.model.load_state_dict(filtered_state_dict, strict=False)

                # Reinitialize scheduler with new steps
                from diffusers import DDIMScheduler
                self.model.scheduler = DDIMScheduler(
                    num_train_timesteps=new_steps,
                    beta_start=self.config.diffusion.beta_start,
                    beta_end=self.config.diffusion.beta_end,
                    clip_sample=True
                )
                self.model.scheduler.set_timesteps(50)  # Default inference steps

            else:
                # Load model state dict directly if steps match
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Restore volumetric_avatar from original pre-trained model
            # (since we exclude it from checkpoints to save space)
            if hasattr(self.model, 'volumetric_avatar'):
                logger.info("Restoring volumetric_avatar from pre-trained model...")
                va_model_path = self.config.paths.volumetric_model
                va_state_dict = torch.load(va_model_path, map_location=self.accelerator.device)

                # Load volumetric avatar state
                missing_keys, unexpected_keys = self.model.volumetric_avatar.load_state_dict(va_state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Missing {len(missing_keys)} keys when loading volumetric_avatar")
                if unexpected_keys:
                    logger.debug(f"Found {len(unexpected_keys)} unexpected keys (likely discriminator weights)")
                logger.info("Volumetric avatar restored successfully")

            # Load optimizer and scheduler state with error handling
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully")
            except ValueError as e:
                if "doesn't match the size of optimizer's group" in str(e):
                    logger.warning("⚠️ Optimizer state mismatch - likely due to model architecture changes")
                    logger.warning("  Starting with fresh optimizer state (learning will continue from scratch)")
                    logger.info("  Model weights are still loaded, only optimizer momentum/history is reset")
                    # Don't load optimizer state, start fresh
                else:
                    raise e

            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Scheduler state loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load scheduler state: {e}")
                    logger.warning("Starting with fresh scheduler")

            self.current_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('loss', float('inf'))

            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', -1)}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _log_visualizations(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], step: int):
        """Log visualizations to WandB for training inspection."""
        if not self.config.wandb.enabled or not self.accelerator.is_local_main_process:
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create expression embedding comparison plot
            if 'expression_embed' in outputs and 'expression_embed' in targets:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot predicted expression embedding (first 50 dims)
                pred_expr = outputs['expression_embed'][0, 0, :50]
                if isinstance(pred_expr, torch.Tensor):
                    pred_expr = pred_expr.detach().cpu().numpy()
                elif not isinstance(pred_expr, np.ndarray):
                    pred_expr = np.array(pred_expr)
                axes[0].bar(range(len(pred_expr)), pred_expr)
                axes[0].set_title(f'Predicted Expression (Step {step})')
                axes[0].set_xlabel('Dimension')
                axes[0].set_ylabel('Value')
                
                # Plot target expression embedding
                target_expr = targets['expression_embed'][0, 0, :50]
                if isinstance(target_expr, torch.Tensor):
                    target_expr = target_expr.detach().cpu().numpy()
                elif not isinstance(target_expr, np.ndarray):
                    target_expr = np.array(target_expr)
                axes[1].bar(range(len(target_expr)), target_expr)
                axes[1].set_title('Target Expression')
                axes[1].set_xlabel('Dimension')
                axes[1].set_ylabel('Value')
                
                plt.tight_layout()
                wandb.log({"visuals/expression_comparison": wandb.Image(fig)}, step=step)
                plt.close(fig)
                
                # Add new candle-like expression visualization
                from visualize_expression import create_expression_candles, create_expression_difference_map
                from visualize_audio_expression import create_audio_expression_visualization
                from warp_visualization import visualize_uv_warps, log_warp_statistics

                # Create candle visualization for entire window
                if outputs['expression_embed'].shape[1] > 1:  # If we have temporal dimension
                    fig_candles = create_expression_candles(
                        target_expression=targets['expression_embed'][0],  # First batch item, all frames
                        predicted_expression=outputs['expression_embed'][0],
                        window_idx=step // 100,  # Use step to create window index
                        reduce_to=32
                    )
                    wandb.log({"visuals/expression_candles": wandb.Image(fig_candles)}, step=step)
                    plt.close(fig_candles)

                    # Add UV warp candles visualization
                    if 'uv_warps' in outputs and 'uv_warps' in targets:
                        try:
                            fig_warp_candles = visualize_uv_warps(
                                uv_warps=outputs['uv_warps'][0:1],  # First batch
                                target_uv_warps=targets['uv_warps'][0:1],
                                step=step,
                                wandb_logger=wandb
                            )
                            plt.close(fig_warp_candles)
                        except Exception as e:
                            logger.warning(f"Failed to create warp visualization: {e}")
                    
                    # Create difference map
                    fig_diff = create_expression_difference_map(
                        target_expression=targets['expression_embed'][0],
                        predicted_expression=outputs['expression_embed'][0],
                        window_idx=step // 100,
                        reduce_to=32
                    )
                    wandb.log({"visuals/expression_diff_map": wandb.Image(fig_diff)}, step=step)
                    plt.close(fig_diff)
                    
                    # Add audio-to-expression visualization if audio features are available
                    if 'audio_features' in targets:
                        fig_audio_expr = create_audio_expression_visualization(
                            audio_features=targets['audio_features'][0],  # First batch item
                            target_expression=targets['expression_embed'][0],
                            predicted_expression=outputs['expression_embed'][0],
                            window_idx=step // 100,
                            audio_reduce_to=32,
                            expr_reduce_to=32
                        )
                        wandb.log({"visuals/audio_to_expression": wandb.Image(fig_audio_expr)}, step=step)
                        plt.close(fig_audio_expr)
            
            # Log motion parameter comparison
            motion_params = ['theta', 'rotation', 'translation', 'scale']
            for param in motion_params:
                if param in outputs and param in targets:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Get first frame of first batch item
                    pred_val = outputs[param][0, 0]
                    if isinstance(pred_val, torch.Tensor):
                        pred_val = pred_val.detach().cpu().numpy().flatten()[:10]
                    else:
                        pred_val = np.array(pred_val).flatten()[:10]
                    
                    target_val = targets[param][0, 0]
                    if isinstance(target_val, torch.Tensor):
                        target_val = target_val.detach().cpu().numpy().flatten()[:10]
                    else:
                        target_val = np.array(target_val).flatten()[:10]
                    
                    axes[0].plot(pred_val, 'b-', label='Predicted')
                    axes[0].plot(target_val, 'r--', label='Target')
                    axes[0].set_title(f'{param.capitalize()} Comparison')
                    axes[0].legend()
                    axes[0].grid(True)
                    
                    # Plot difference
                    diff = pred_val - target_val
                    axes[1].bar(range(len(diff)), diff)
                    axes[1].set_title(f'{param.capitalize()} Difference')
                    axes[1].grid(True)
                    
                    plt.tight_layout()
                    wandb.log({f"visuals/{param}_comparison": wandb.Image(fig)}, step=step)
                    plt.close(fig)
                    
        except Exception as e:
            logger.warning(f"Error in visualization logging: {str(e)}")

    # def _generate_sample_video(self, outputs: Dict[str, torch.Tensor], max_frames: int = 50):
    #     """Generate a sample video for visualization using VASAInference."""
    #     try:
    #         # Try to use VASAInference to generate actual avatar video
    #         from vi import VASAInference
    #         import random
    #         import os
            
    #         # Get list of available videos
    #         video_folder = self.config.paths.video_folder
    #         video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
            
    #         if video_files:
    #             # Select a random video for generation
    #             sample_video = os.path.join(video_folder, random.choice(video_files))
                
    #             # Determine checkpoint path based on config
    #             if 'overfit' in str(self.config_path):
    #                 checkpoint_dir = "./checkpoints_overfit"
    #                 config_file = 'overfit_config.yaml'
    #             else:
    #                 checkpoint_dir = "./checkpoints"
    #                 config_file = 'vasa_config.yaml'
                
    #             # Check if we have a saved checkpoint
    #             checkpoint_path = None
    #             if os.path.exists(f"{checkpoint_dir}/best_checkpoint.pt"):
    #                 checkpoint_path = f"{checkpoint_dir}/best_checkpoint.pt"
    #             elif self.current_epoch > 0:
    #                 # Try to find latest checkpoint
    #                 epoch_checkpoint = f"{checkpoint_dir}/checkpoint_epoch_{self.current_epoch}.pt"
    #                 if os.path.exists(epoch_checkpoint):
    #                     checkpoint_path = epoch_checkpoint
                
    #             if checkpoint_path and os.path.exists(checkpoint_path):
    #                 try:
    #                     logger.info(f"Generating sample video using checkpoint: {checkpoint_path}")
                        
    #                     # Initialize inference model
    #                     inferencer = VASAInference(
    #                         checkpoint_path=checkpoint_path,
    #                         config_path=config_file
    #                     )
                        
    #                     # Generate output video
    #                     video_path = self.output_dir / f"vasa_epoch_{self.current_epoch}_step_{self.global_step}.mp4"
    #                     video_path.parent.mkdir(parents=True, exist_ok=True)
                        
    #                     inferencer.generate_from_video(
    #                         input_video=sample_video,
    #                         output_path=str(video_path),
    #                         fps=25.0,
    #                         neutral_expression=False
    #                     )
                        
    #                     logger.info(f"Generated VASA video: {video_path}")
    #                     return video_path
                        
    #                 except Exception as e:
    #                     logger.warning(f"Failed to generate VASA video: {str(e)}")
    #                     # Fall back to gradient visualization
    #                     pass
            
    #         # Fallback: gradient visualization if VASA generation fails
    #         with torch.no_grad():
    #             # Limit to max_frames
    #             actual_frames = min(outputs['theta'].shape[1], max_frames) if 'theta' in outputs else 20
                
    #             frames = []
                
    #             # Get motion parameters for visualization
    #             if 'expression' in outputs:
    #                 expression = outputs['expression'][0, :actual_frames]  # [T, 256]
    #                 # Normalize expression to [0, 1] for visualization
    #                 expr_min = expression.min()
    #                 expr_max = expression.max()
    #                 if expr_max > expr_min:
    #                     expression = (expression - expr_min) / (expr_max - expr_min)
    #             else:
    #                 expression = None
                
    #             for i in range(actual_frames):
    #                 # Create visualization frame showing motion parameters
    #                 frame = torch.zeros(3, 512, 512)
                    
    #                 # Red channel: time progress
    #                 frame[0] = i / actual_frames
                    
    #                 # Green channel: expression magnitude if available
    #                 if expression is not None:
    #                     # Average expression values for this frame
    #                     expr_mag = expression[i].mean().item()
    #                     frame[1] = expr_mag
    #                 else:
    #                     frame[1] = 0.5
                    
    #                 # Blue channel: inverse time
    #                 frame[2] = 1.0 - (i / actual_frames)
                    
    #                 frames.append(frame)
                
    #             frames = torch.stack(frames)  # [T, C, H, W]
                
    #             # Save as MP4 video
    #             import cv2
                
    #             video_path = self.output_dir / f"gradient_epoch_{self.current_epoch}_step_{self.global_step}.mp4"
    #             video_path.parent.mkdir(parents=True, exist_ok=True)
                
    #             # Convert to numpy and ensure proper format
    #             frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
                
    #             if frames_np.max() <= 1.0:
    #                 frames_np = (frames_np * 255).astype(np.uint8)
                
    #             # Use mp4v codec for better compatibility
    #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #             out = cv2.VideoWriter(
    #                 str(video_path),
    #                 fourcc,
    #                 25.0,
    #                 (frames_np.shape[2], frames_np.shape[1])
    #             )
                
    #             for frame in frames_np:
    #                 out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #             out.release()
                
    #             return video_path
                
    #     except Exception as e:
    #         logger.error(f"Error generating sample video: {str(e)}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #         return None

    def _log_gradient_stats(self, step: int):
        """Log gradient statistics to WandB - DISABLED for performance."""
        return  # Disabled for performance
        
        # Original code kept for reference:
        # if not self.config.wandb.enabled or not self.accelerator.is_local_main_process:
        #     return
        #     
        # try:
        #     grad_stats = {}
        #     for name, param in self.model.named_parameters():
        #         if param.grad is not None:
        #             grad_norm = param.grad.norm().item()
        #             grad_stats[f"gradients/{name}_norm"] = grad_norm
        #             
        #             # Log histogram for important layers - DISABLED
        #             # if any(key in name for key in ['motion_proj', 'transformer', 'output']):
        #             #     wandb.log({f"grad_hist/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=step)
        #     
        #     # Log aggregated stats
        #     if grad_stats:
        #         wandb.log(grad_stats, step=step)
        # except Exception as e:
        #     logger.warning(f"Error logging gradient stats: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VASA Training')
    parser.add_argument('--config', type=str, default='vasa_config.yaml',
                        help='Path to config file (default: vasa_config.yaml)')
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    # Load volumetric model 
    config = OmegaConf.load(args.config)
    print(f"Loaded config from: {args.config}")
    model_path = config.paths.volumetric_model
    emo_config = OmegaConf.load(config.paths.volumetric_config)
    
    # CRITICAL: Set proper paths and flags for the config
    emo_config.project_dir = './nemo'  # Set correct project directory
    emo_config.model_checkpoint = True  # Enable checkpoint loading
    
    # Add nemo to path if needed
    import sys
    sys.path.insert(0, 'nemo')
    volumetric_avatar = importlib.import_module(f'models.stage_1.volumetric_avatar.va').Model(emo_config, training=False)

    # Load model weights
    model_dict = torch.load(model_path, map_location='cuda')
    missing_keys, unexpected_keys = volumetric_avatar.load_state_dict(model_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing {len(missing_keys)} keys when loading volumetric model")
    if unexpected_keys:
        print(f"Info: {len(unexpected_keys)} unexpected keys (likely discriminator weights)")
    
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    
    

    # Config already loaded above

    # Initialize wandb with enhanced configuration
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            config=OmegaConf.to_container(config, resolve=True),
            group="overfit-experiments" if "overfit" in args.config else "main",
            notes=f"Config: {args.config}, Videos: {config.dataset.max_videos}, Window: {config.motion.window_size}",
            save_code=True,  # Save code for reproducibility
            tags=["overfit", "fast-convergence"] if "overfit" in args.config else ["training"]
        )

    model = VASAModel(
        config=config,
        volumetric_avatar=volumetric_avatar,
        device=config.device
    )
    model = model.cuda()

    import os
    # Add these near the start of your script
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    # Set environment variables for better CUDA operation
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'

    gc.collect()
    torch.cuda.empty_cache()

    # Create VA bridge for EMO generation if enabled
    va_bridge_for_dataset = None
    if config.dataset.get('generate_emo_frames', False):
        logger.info("Creating VA bridge for EMO frame generation in dataset...")
        # VASAVolumetricAvatarBridge just needs the loaded volumetric_avatar model
        va_bridge_for_dataset = VASAVolumetricAvatarBridge(volumetric_avatar)

    # Create dataset with volumetric model
    use_single_bucket = config.dataset.get('use_single_bucket', False)  # Get from config
    full_dataset = VASAIntegratedDataset(
        # video_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/", #ovs-GiY_848_1
        video_folder=config.paths.video_folder,
        emo_model=volumetric_avatar,
        window_size=config.motion.window_size,  # Use config window_size (20)
        stride=config.motion.stride,  # Use config stride (10)
        context_size=config.motion.context_size,  # Use config context_size (10)
        max_videos=config.dataset.get('max_videos', None),  # Use config value or all videos if not specified
        frame_size=(512, 512),
        sequence_length=config.motion.window_size,  # Match window_size
        cache_audio=True,
        preextract_audio=True,
        random_seed=42,
        cache_dir=config.paths.get('cache_dir', 'cache'),  # Use config cache dir
        use_single_bucket=use_single_bucket,  # Pass single-bucket flag
        # EMO generation parameters
        generate_emo_frames=config.dataset.get('generate_emo_frames', False),
        emo_identity_path=config.dataset.get('emo_identity_path', 'nemo/data/IMG_1.png'),
        emo_keyframes_per_window=config.dataset.get('emo_keyframes_per_window', 50),
        va_bridge=va_bridge_for_dataset
    )

    # Check if single-bucket cache exists
    if use_single_bucket and hasattr(full_dataset.cache, 'has_cache'):
        if not full_dataset.cache.has_cache():
            logger.info("Single-bucket cache not found. Consider running preprocess_single_bucket.py first.")
        else:
            cache_info = full_dataset.cache.get_cache_info()
            logger.info(f"Using single-bucket cache: {cache_info} windows, {cache_info['file_size_mb']:.1f} MB")

    # Print dataset stats
    logger.info(f"Dataset created:")
    logger.info(f"  Total videos: {len(full_dataset.video_paths)}")
    logger.info(f"  Total windows: {len(full_dataset.windows)}")


    # Calculate appropriate batch size based on dataset size
    dataset_size = len(full_dataset)
    batch_size = min(1, dataset_size)  # Use batch size 1 for testing

    # Create custom sampler for maintaining window sequences
    # Use full_dataset for sampler since it needs the windows attribute
    windows_per_sequence = config.train.get('windows_per_batch', 4) if hasattr(config, 'train') else 4
    train_sampler = WindowSequenceSampler(
        full_dataset,
        batch_size=batch_size,
        windows_per_sequence=windows_per_sequence,  # Number of consecutive windows from config
        shuffle=True
    )
    logger.info(f"Using {windows_per_sequence} consecutive windows per sequence")
    
    # For validation, we'll use the full dataset but not sample all windows
    # This is a simplified approach - in production you'd want a proper val split
    val_dataset = full_dataset
    train_dataset = full_dataset
    
    # Create custom collate function for adding prev_context
    collate_fn = create_window_sequence_collate_fn(
        context_size=config.motion.context_size if hasattr(config, 'motion') else 10
    )
    
    # Create data loaders with custom sampler
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues (like in train_overfit.py)
        pin_memory=False  # Disabled because tensors are already on GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid CUDA multiprocessing issues
        pin_memory=False,  # Disabled because tensors are already on GPU
        collate_fn=collate_vasa_batch
    )



    # Create output directory
    output_dir = Path(config.paths.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer and utilities
    trainer = VASATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        config_path=args.config
    )


    # validation_handler = ValidationHandler(config)

    # Load checkpoint if continuing training (check env var first, then config)
    import os
    env_resume = os.environ.get('VASA_RESUME_FROM')
    resume_path = env_resume if env_resume else (config.train.resume_from if hasattr(config.train, 'resume_from') else None)

    if resume_path:  # Checks if not None and not empty string
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    if config.train.turn_off_noise:  # Checks if empty string            
        logger.info(f"👹 Config turn_off_noise: {config.train.turn_off_noise}")


    if config.train.control_start_epoch > 1:  # Checks if empty string            
        logger.info(f"👹 Delaying control losses / frame verification until : {config.train.control_start_epoch}")

    # Debug dataset and dataloader
    logger.info(f"Dataset size: {len(full_dataset)}")
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")

    # Test a single item from dataset
    test_item = full_dataset[0]
    if test_item is not None:
        logger.info(f"Sample item keys: {list(test_item.keys())}")
        for k, v in test_item.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"Shape of {k}: {v.shape}")
    else:
        logger.error("Got None test item")


    # Train
    trainer.train()
 
