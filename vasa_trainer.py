import h5py
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import logging
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
from vasa_model import VASAModel,VASALossModule,MotionSequenceHandler
import importlib
import sys
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')
from logger import logger,TorchDebugger
import traceback
from vasa_dataset import WorkerState, VASAIntegratedDataset
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
        self.expression_warmup_steps = 100  # Shorter warmup for expressions

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
    """Save tensor of frames as video"""
    # Convert to numpy and correct format
    frames = frames.cpu().numpy()
    if frames.shape[0] == 3:  # CHW -> HWC
        frames = frames.transpose(1, 2, 0)
    
    # Scale to uint8 range
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
        
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frames.shape[1], frames.shape[0])
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
                windows = item['windows']
                # Add video path to window metadata
                for window in windows:
                    if 'metadata' not in window:
                        window['metadata'] = {}
                    window['metadata']['video_path'] = item.get('video_path', '')
                all_windows.extend(windows)

        if not all_windows:
            logger.error("No valid windows in batch")
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
        output_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        self.train_loader = train_loader

        # Initialize accelerator with mixed precision enabled for faster training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            mixed_precision="fp16"  # Force FP16 for faster convergence
        )
        
        logger.info(f"Mixed precision training: ENABLED (fp16)")
        logger.info(f"Gradient accumulation steps: {config.train.gradient_accumulation_steps}")

     
            
        # Initialize loss module
        self.loss_module = VASALossModule(
            volumetric_avatar=model.volumetric_avatar,
            config=config,
            device=self.accelerator.device
        )
        

        # Initialize MotionSequenceHandler
        self.motion_handler = MotionSequenceHandler(
            window_size=config.motion.window_size,
            stride=config.motion.stride,
            context_size=config.motion.context_size,
            min_window_size=config.motion.min_window_size,
            max_window_size=config.motion.max_window_size,
            
        )

             # Initialize optimizer
        self.init_optimizers()
        self.cfg_scheduler = CFGScheduleHandler(config)
        self.grad_monitor = GradientMonitor(model)
        self.lr_monitor = LearningRateMonitor(self.optimizer)

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
        
        # Create separate parameter groups for each motion projection type
        for proj_name in motion_proj_names:
            proj_params = []
            
            # Find parameters for this specific projection
            for name, param in model.named_parameters():
                if f'motion_projections.{proj_name}' in name and param not in used_params:
                    proj_params.append(param)
                    used_params.add(param)
            
            # Default learning rate multipliers (can be adjusted)
            lr_multipliers = {
                'theta': 1.0,        # Slight boost for pose matrix
                'rotation': 1.0,     # Moderate boost for rotational parameters
                'translation': 1.0,  # Base learning rate
                'expression': 1.0    # Slight higher rate for expression
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

    def train(self):
        """Main training loop."""
        logger.info("Starting training")
        num_epochs = self.config.train.num_epochs

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_stats = self.train_epoch()
            
            # Validation phase
            val_stats = self.validate() if self.val_loader else None
            
            # # Save checkpoint if best model
            # if val_stats and val_stats['total'] < self.best_val_loss:
            #     self.best_val_loss = val_stats['total']
            #     self.save_checkpoint(is_best=True)

            # Regular checkpoint saving
            if epoch % self.config.train.save_freq == 0:
                self.save_checkpoint(is_best=False)


    def train_epoch(self) -> Dict[str, float]:
        """Training loop with proper noise level sampling."""
        self.model.train()  # Put model in training mode
        self.train_metrics.reset()
        num_batches = len(self.train_loader)
        
        logger.info(f"\n=== Starting Epoch {self.current_epoch} ===")
        logger.info(f"Batch size: {self.config.train.batch_size}")
        logger.info(f"Total batches: {num_batches}")
        
        # Initialize progress bar
        self.progress_bar = tqdm(
            total=num_batches,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Epoch {self.current_epoch}"
        )
        
        # Get current CFG scales based on epoch
        cfg_scales = self._get_cfg_scales()
        
        # Initialize epoch metrics
        epoch_metrics = defaultdict(list)
        grad_norms = defaultdict(list)

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                logger.info(f"Processing batch {batch_idx}")
                if batch is None:
                    logger.warning(f"Batch {batch_idx} is None")
                    continue
                    
                logger.info(f"Batch {batch_idx} has data, processing windows...")
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

                # Process each window
                for window_idx, window in enumerate(windows):
                    logger.info(f"  Processing window {window_idx}/{len(windows)}")
                    if not window:
                        logger.warning(f"  Window {window_idx} is None")
                        continue
                    # Use accumulate per window instead of per batch
                    with self.accelerator.accumulate(self.model):
                        try:
                            logger.info(f"  Preparing motion data for window {window_idx}")
                            # Extract target motion parameters
                            motion_data = self.motion_handler.prepare_motion_data(window)
                            B = motion_data['theta'].shape[0]
                            device = motion_data['theta'].device

                            # In train_epoch:
                            if self.config.train.turn_off_noise:
                                t = torch.zeros((B,), dtype=torch.long, device=device)  # OVERFIT t = 0 no noise/minimal perturbation of the input
                            else:
                                # Sample timestep uniformly
                                t = torch.randint(0, self.model.num_steps, (B,), device=device)

                            # Rest stays the same
                            noise = {k: torch.randn_like(v) for k, v in motion_data.items()}
                            noised_motion = self.model._add_noise_to_motion(
                                motion_data=motion_data,
                                noise=noise,
                                noise_level=t
                            )
                            logger.debug("Added scheduled noise to motion")

                            # Extract control signals with dropout during training
                            control_signals = {
                                'gaze': window.get('gaze'),
                                'head_distance': window.get('head_distance'),
                                'emotion': window.get('emotion'),
                                'speed_bucket': window.get('speed_bucket'),
                                'lips': window.get('lips'),
                                'right_eye': window.get('right_eye'),
                                'left_eye': window.get('left_eye'),
                                'jaw': window.get('jaw'),
                                'nose': window.get('nose'),
                                'blink_state': window.get('blink_state'),
                                'audio_features': window.get('audio_features')
                            }

                            # Apply control signal dropout
                            if not self.config.train.turn_off_noise:
                                dropout_probs = self.config.train.dropout_probs
                                control_signals = self._apply_condition_dropout(
                                    control_signals, 
                                    dropout_probs
                                )

                            # Forward pass with CFG during inference
                            outputs = self.model(
                                motion_data=noised_motion,  # Use noised motion
                                noise_level=t,              # Pass timestep
                                conditions=control_signals,
                                noise=noise                 # Pass noise for loss computation
                            )
                            
                            # Tell loss module which window we're on for visualization
                            self.loss_module._window_count_this_batch = window_idx
                            
                            # Compute losses using noise prediction
                            losses, metrics = self.loss_module.compute_losses(
                                outputs=outputs,
                                targets=motion_data,  # Original motion data
                                conditions=control_signals,
                                noise=outputs['noise'],    
                                return_metrics=True,
                                current_epoch=self.current_epoch,
                                step=self.global_step
                            )

                            # Update batch metrics - store per-window
                            for k, v in losses.items():
                                if isinstance(v, torch.Tensor):
                                    batch_metrics[k].append(v.detach())
                            if metrics:
                                for k, v in metrics.items():
                                    batch_metrics[f"metric_{k}"].append(v)

                            # Check for NaN/Inf in loss before backward pass
                            if not torch.isfinite(losses['total']):
                                logger.error(f"NaN/Inf detected in loss at epoch {self.current_epoch}, batch {batch_idx}, window {window_idx}")
                                logger.error(f"Loss components: {losses}")
                                # Skip this window
                                continue
                            
                            # Backward pass for this window
                            self.accelerator.backward(losses['total'])
                            
                            if self.accelerator.sync_gradients:
                                # Clip gradients to prevent explosions
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.train.max_grad_norm
                                )
                                
                                # Log gradient norm for monitoring
                                if grad_norm > 1e6:
                                    logger.warning(f"Large gradient norm detected: {grad_norm:.2e}")
                                
                            # Optimizer step after each window
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            
                            batch_total_loss += losses['total'].item()
                            logger.info(f"  Window {window_idx} completed - loss: {losses['total'].item():.4f}")

                        except Exception as e:
                            logger.error(f"Error processing window {window_idx}: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue

                # Average metrics across windows
                avg_metrics = {
                    k: torch.stack(v).mean().item() if isinstance(v[0], torch.Tensor)
                    else sum(v) / len(v)
                    for k, v in batch_metrics.items()
                }
                avg_metrics['total'] = batch_total_loss / len(windows)
                
                # Update metrics
                self.train_metrics.update(avg_metrics)
                
                # Update epoch metrics
                for k, v in avg_metrics.items():
                    epoch_metrics[k].append(v)
                        
                # Log metrics and gradients
                if batch_idx % self.config.train.gradient_monitoring.log_freq == 0:
                    self._log_training_stats(
                        batch_idx, 
                        num_batches,
                        avg_metrics,
                        grad_norms
                    )
                
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
        
        # Log epoch metrics
        if self.config.wandb.enabled and self.accelerator.is_local_main_process:
            wandb.log(
                {f"epoch/{k}": v for k, v in epoch_averages.items()},
                step=self.global_step
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
        grad_norms: Dict[str, List[float]]
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
            if k.startswith('control_') or k.startswith('metric_'):
                step_metrics[f'train/{k}'] = v
                
        # Add gradient statistics
        if grad_norms:
            grad_stats = {
                'min': min(min(norms) for norms in grad_norms.values()),
                'max': max(max(norms) for norms in grad_norms.values()),
                'mean': np.mean([np.mean(norms) for norms in grad_norms.values()]),
                'std': np.std([np.std(norms) for norms in grad_norms.values()])
            }
            
            for k, v in grad_stats.items():
                step_metrics[f'train/grad_{k}'] = v
                
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
        Computes canonical volume once and reuses it for the entire sequence.
        
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
            logger.debug(f"Motion batch size: {motion_batch_size}, sequence length: {seq_len}")
            
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
                
                # Create sampling grid once
                device = canonical_volume.device
                grid_s = torch.linspace(-1, 1, self.model.volumetric_avatar.args.latent_volume_size)
                grid_z = torch.linspace(-1, 1, self.model.volumetric_avatar.args.latent_volume_depth)
                w, v, u = torch.meshgrid(grid_z, grid_s, grid_s, indexing='ij')
                e = torch.ones_like(u)
                identity_grid_3d = torch.stack([u, v, w, e], dim=3).view(1, -1, 4).to(device)
                grid = identity_grid_3d.expand(motion_batch_size, -1, -1)
                
                # Process each timestep
                for t in range(seq_len):
                    # Get motion parameters for current timestep
                    curr_theta = motion_outputs['theta'][:, t]  # [B, 3, 4]
                    curr_expression = motion_outputs['expression_embed'][:, t]
                    
                    # Apply rotation transform
                    target_rotation_warp = grid.bmm(curr_theta[:, :3].transpose(1, 2)).view(
                        motion_batch_size,
                        self.model.volumetric_avatar.args.latent_volume_depth,
                        self.model.volumetric_avatar.args.latent_volume_size,
                        self.model.volumetric_avatar.args.latent_volume_size,
                        3
                    )
                    
                    # Apply warping to canonical volume
                    warped_volume = self.model.volumetric_avatar.grid_sample(canonical_volume, target_rotation_warp)
                    
                    # Reshape for decoder
                    c = self.model.volumetric_avatar.args.latent_volume_channels
                    d = self.model.volumetric_avatar.args.latent_volume_depth
                    s = self.model.volumetric_avatar.args.latent_volume_size
                    latent_feats = warped_volume.view(motion_batch_size, c * d, s, s)
                    
                    # Generate frame using decoder
                    frame, _, _, _ = self.model.volumetric_avatar.decoder_nw(
                        {'expression_embed': curr_expression},
                        None,
                        latent_feats,
                        False
                    )
                    
                    save_image(frame[0], f"frame_{t}.png")
                    # Save frame (detach to avoid memory leak)
                    generated_frames.append(frame.detach())
                    
                    # Clear warped volume to free memory
                    del warped_volume
                    del latent_feats
                    torch.cuda.empty_cache()
                
                # Stack frames along time dimension [B, T, C, H, W]
                generated_frames = torch.stack(generated_frames, dim=1)
                logger.debug(f"Final generated frames shape: {generated_frames.shape}")
                
                return generated_frames
                
            finally:
                # Clean up canonical volume
                del canonical_volume
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error generating synced frames: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        


    
    def validate(self) -> Dict[str, float]:
        """Run validation with proper model mode handling."""
        if not self.val_loader:
            return {}
            
        # Put model in eval mode
        self.model.eval()
        self.val_metrics.reset()
        
        cfg_scales = self._get_cfg_scales()  # Get current CFG scales

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                disable=not self.accelerator.is_local_main_process,
                desc="Validation"
            ):
                try:
                    if batch is None:
                        continue

                    # Process batch into windows
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
                            
                            # Reconstruction metrics
                            metrics.update(self.loss_module._compute_reconstruction_losses(
                                generated_sequence, window, None
                            ))
                            
                            # Control metrics if applicable
                            if self.current_epoch >= self.config.train.control_start_epoch:
                                control_metrics = self.loss_module._compute_control_losses(
                                    generated_sequence, window, self.current_epoch
                                )
                                metrics.update(control_metrics)

                            # Generate frames for sync evaluation if needed
                            if self.config.loss.use_sync_loss:
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
        """Apply random dropout to conditions based on config probabilities."""
        dropped_conditions = {}
        for k, v in conditions.items():
            if k in dropout_probs and v is not None:
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

            # Filter out volumetric_avatar parameters
            # filtered_state_dict = {
            #     k: v for k, v in state_dict.items()
            #     if not k.startswith('volumetric_avatar.')
            # }

            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': state_dict, # filtered_state_dict to exclude volumetric_avatar,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'loss': self.best_val_loss,
                'config': self.config
            }

            if is_best:
                save_path = self.output_dir / 'model_best.pt'
            else:
                save_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'

            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            # Log saved parameters
            total_params = sum(p.numel() for p in unwrapped_model.parameters())
            saved_params = sum(v.numel() for v in state_dict.values())
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Saved parameters: {saved_params:,}")
            logger.info(f"Excluded parameters: {total_params - saved_params:,}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            

    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint with diffusion schedule handling"""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
            
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

            # Load optimizer and scheduler state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.current_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('loss', float('inf'))

            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', -1)}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)

    # Load volumetric model 
    config = OmegaConf.load('vasa_config.yaml')
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

    # Initialize wandb
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            # entity=config.wandb.entity,
            # name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True)
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

    # Create dataset with volumetric model
    full_dataset = VASAIntegratedDataset(
        # video_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/", #ovs-GiY_848_1
        video_folder=config.paths.video_folder,
        emo_model=volumetric_avatar,
        window_size=config.motion.window_size,  # Use config window_size (20)
        stride=config.motion.stride,  # Use config stride (10)
        context_size=config.motion.context_size,  # Use config context_size (10)
        max_videos=1,  # Reduced from 10 to 1 for testing
        frame_size=(512, 512),
        sequence_length=config.motion.window_size,  # Match window_size
        cache_audio=True,
        preextract_audio=True,
        random_seed=42,
        cache_dir=config.paths.get('cache_dir', 'cache')  # Use config cache dir
    )

    # Print dataset stats
    logger.info(f"Dataset created:")
    logger.info(f"  Total videos: {len(full_dataset.video_paths)}")
    logger.info(f"  Total windows: {len(full_dataset.windows)}")


    # Split dataset into train/val
    val_size = int(0.1 * len(full_dataset))  # 10% for validation
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Calculate appropriate batch size based on dataset size
    dataset_size = len(full_dataset)
    batch_size = min(1, dataset_size)  # Use batch size 1 for testing



    # Worker initialization function for consistency
    def worker_init_fn(worker_id):
        import numpy as np
        np.random.seed(worker_id)
        
    # Create data loaders with optimized settings for faster training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Enable parallel data loading
        pin_memory=True,  # Pin memory for faster GPU transfer
        drop_last=True,
        collate_fn=collate_vasa_batch,
        # Add these safety settings
        persistent_workers=True if batch_size > 1 else False,  # Keep workers alive
        prefetch_factor=2,  # Prefetch 2 batches per worker
        multiprocessing_context='spawn',  # Use spawn context for safety
        worker_init_fn=worker_init_fn  # Ensure worker consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for testing
        shuffle=False,
        num_workers=1,  # Single worker for validation
        pin_memory=True,  # Pin memory for faster GPU transfer
        collate_fn=collate_vasa_batch,
        multiprocessing_context='spawn',
        persistent_workers=False,
        worker_init_fn=worker_init_fn  # Ensure worker consistency
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
        output_dir=output_dir
    )


    # validation_handler = ValidationHandler(config)

    # Load checkpoint if continuing training
    if hasattr(config.train, 'resume_from') and config.train.resume_from is not None:
        if config.train.resume_from:  # Checks if empty string            
            logger.info(f"Resuming training from checkpoint: {config.train.resume_from}")
            trainer.load_checkpoint(config.train.resume_from)

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
 
