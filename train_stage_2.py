import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from pathlib import Path

from dataset import VASADataset
from VASA import VASAFaceEncoder, VASADiffusionTransformer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from omegaconf import OmegaConf
import wandb
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, Any, Optional
from accelerate.logging import get_logger


logger = get_logger(__name__)


class VASAStage2Trainer:
    """
    Complete implementation of Stage 2 VASA training - Diffusion Transformer for motion generation
    with distributed training, logging, and checkpoint management
    """
    def __init__(
        self, 
        face_encoder,
        diffusion_model,
        config,
    ):
        self.config = config
        self.face_encoder = face_encoder
        self.diffusion_model = diffusion_model
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with="wandb",
            mixed_precision=None
        )

        # Initialize trackers
        self.accelerator.init_trackers(
            project_name=config.project_name,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={"wandb": {
                "name": config.experiment_name,
                "dir": config.training.log_dir
            }}
        )

        # Freeze face encoder
        for param in self.face_encoder.parameters():
            param.requires_grad = False
            
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.diffusion_model.parameters(),
            lr=config.training.lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.min_lr
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.metric_history = []
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Extract data
        images = batch['frames']
        audio_features = batch['audio_features']
        gaze = batch['gaze']
        emotion = batch['emotion']

        # Extract facial representations using frozen encoder
        with torch.no_grad():
            facial_reps = self.face_encoder.encode_holistic(
                images[:, 0],  # Source frame
                gaze=gaze[:, 0],
                emotion=emotion[:, 0]
            )

        # Get facial dynamics sequence
        facial_dynamics = facial_reps['facial_dynamics']
        
        # Sample timestep
        batch_size = facial_dynamics.shape[0]
        t = torch.randint(0, self.config.num_steps, (batch_size,), 
                         device=self.accelerator.device)

        # Add noise
        noise = torch.randn_like(facial_dynamics)
        noisy_dynamics = self.diffusion_model.add_noise(facial_dynamics, t, noise)

        # Prepare conditions
        conditions = {
            'audio': audio_features,
            'gaze': gaze,
            'emotion': emotion
        }

        # Forward pass with CFG
        with self.accelerator.autocast():
            model_output = self.diffusion_model(
                noisy_dynamics,
                t,
                conditions=conditions,
                use_cfg=self.training
            )

            # Compute losses
            losses = {}
            
            # Base diffusion loss
            losses['diffusion'] = F.mse_loss(
                model_output['predicted_noise'], 
                noise
            )

            # CFG losses
            if self.training and self.config.cfg_scales is not None:
                for cond_type, scale in self.config.cfg_scales.items():
                    if scale > 0 and f'masked_{cond_type}' in model_output:
                        losses[f'cfg_{cond_type}'] = F.mse_loss(
                            model_output[f'masked_{cond_type}'],
                            model_output['uncond']
                        )

            # Total loss
            total_loss = losses['diffusion'] + sum(
                self.config.lambda_cfg * losses[f'cfg_{k}']
                for k in ['audio', 'gaze']
                if f'cfg_{k}' in losses
            )

        # Backward pass
        self.accelerator.backward(total_loss)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.diffusion_model.parameters(),
                self.config.max_grad_norm
            )

        return {
            'total_loss': total_loss.item(),
            **{k: v.item() for k, v in losses.items()}
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.diffusion_model.train()
        metrics = defaultdict(float)
        
        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(train_loader):
            with self.accelerator.accumulate(self.diffusion_model):
                step_metrics = self.train_step(batch)
                
                # Update metrics
                for k, v in step_metrics.items():
                    metrics[k] += v
                
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log samples periodically
            if batch_idx % self.config.sample_interval == 0:
                self.log_samples(batch, self.global_step)
                
            progress_bar.update(1)
            self.global_step += 1

        # Average metrics
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        
        progress_bar.close()
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.diffusion_model.eval()
        metrics = defaultdict(float)
        
        for batch in tqdm(val_loader, desc="Validation"):
            step_metrics = self.train_step(batch)
            for k, v in step_metrics.items():
                metrics[k] += v
                
        metrics = {k: v / len(val_loader) for k, v in metrics.items()}
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop"""
        # Prepare for training
        (
            self.face_encoder,
            self.diffusion_model,
            self.optimizer,
            self.scheduler,
            train_loader,
            val_loader
        ) = self.accelerator.prepare(
            self.face_encoder,
            self.diffusion_model,
            self.optimizer,
            self.scheduler,
            train_loader,
            val_loader
        )

        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                # Training
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.validate(val_loader)
                
                # Update scheduler
                self.scheduler.step()
                
                # Log metrics
                metrics = {
                    "train/" + k: v for k, v in train_metrics.items()
                }
                metrics.update({
                    "val/" + k: v for k, v in val_metrics.items()
                })
                metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                
                self.accelerator.log(metrics, step=self.global_step)
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(
                        epoch=epoch,
                        metrics=metrics,
                        is_best=val_metrics['total_loss'] < self.best_metric
                    )
                    
                # Update best metric
                if val_metrics['total_loss'] < self.best_metric:
                    self.best_metric = val_metrics['total_loss']
                
                # Early stopping
                if not self.check_improvement(val_metrics['total_loss']):
                    print("Early stopping triggered!")
                    break
                    
                self.current_epoch = epoch + 1
                
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
            
        finally:
            self.cleanup()

    def log_samples(self, batch: Dict[str, torch.Tensor], step: int):
        """Log sample generations"""
        if self.accelerator.is_local_main_process:
            with torch.no_grad():
                # Generate motion sequence
                generated_motion = self.generate_motion(
                    batch['audio_features'][:1],
                    {
                        'gaze': batch['gaze'][:1],
                        'emotion': batch['emotion'][:1]
                    }
                )
                
                # Log samples
                self.accelerator.log({
                    "samples/motion": wandb.Histogram(
                        generated_motion.cpu().numpy()
                    ),
                    "samples/audio": wandb.Histogram(
                        batch['audio_features'][:1].cpu().numpy()
                    )
                }, step=step)

    @torch.no_grad()
    def generate_motion(self, audio_features, conditions, cfg_scales=None):
        """Generate motion sequence"""
        self.diffusion_model.eval()
        
        # Initialize from noise
        motion = torch.randn(
            (1, self.config.sequence_length, self.config.motion_dim),
            device=self.accelerator.device
        )
        
        # Iterative denoising
        for t in reversed(range(self.config.num_steps)):
            timesteps = torch.full((1,), t, device=self.accelerator.device)
            model_output = self.diffusion_model(
                motion,
                timesteps,
                conditions=conditions,
                cfg_scales=cfg_scales
            )
            motion = self.diffusion_model.update_sample(
                motion,
                model_output['predicted_noise'],
                timesteps
            )
            
        return motion

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        unwrapped_model = self.accelerator.unwrap_model(self.diffusion_model)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'metric_history': self.metric_history,
            'config': self.config
        }
        
        save_dir = Path(self.config.training.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = save_dir / 'best_model.pt'
        else:
            path = save_dir / f'checkpoint_epoch_{epoch}.pt'
            
        self.accelerator.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.metric_history = checkpoint['metric_history']

    def check_improvement(self, current_metric: float) -> bool:
        """Check for improvement (for early stopping)"""
        self.metric_history.append(current_metric)
        if len(self.metric_history) > self.config.patience:
            best_recent = min(self.metric_history[-self.config.patience:])
            if best_recent >= min(self.metric_history[:-self.config.patience]):
                return False
        return True

    def cleanup(self):
        """Cleanup resources"""
        self.accelerator.end_training()

