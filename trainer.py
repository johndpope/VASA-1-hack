import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import yaml

from typing import List, Optional, Tuple
from VASA import *
from loss import *
from vasa_config import VASAConfig
from generator import VideoGenerator,MotionGenerator,VideoPostProcessor
import wandb
from dataset import VASADataset
from omegaconf import OmegaConf
from utils import get_vasa_exp_name


class TrainingLogger:
    """Logging utility for training progress"""
    def __init__(self, 
                 exp_name: str,
                 log_dir: str,
                 use_wandb: bool = True):
        self.exp_name = exp_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(project="vasa", name=exp_name)
        

        self.use_wandb = use_wandb
        
        # Setup file logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to both file and wandb"""
        # Log to file
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Step {step} | {metric_str}')
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_video(self, video: torch.Tensor, audio: torch.Tensor, 
                  step: int, tag: str = 'generated'):
        """Log video samples to wandb"""
        if self.use_wandb:
            wandb.log({
                f'{tag}_video': wandb.Video(
                    video.cpu().numpy(),
                    fps=25,
                    format="mp4"
                )
            }, step=step)


class VASATrainer:
    """
    Complete VASA training orchestrator that integrates all components:
    - Model management
    - Training loops
    - Video generation
    - Metrics tracking
    - Distributed training
    """
    def __init__(
        self,
        config: VASAConfig,
        logger: TrainingLogger,
        local_rank: int = -1,
        resume_path: Optional[str] = None
    ):
        self.config = config
        self.logger = logger
        self.local_rank = local_rank
        self.global_step = 0
        
        # Setup distributed training
        self.setup_distributed()
        
        # Initialize components
        self.setup_models()
        self.setup_optimizers()
        self.setup_data()
        self.setup_video_generator()
        self.setup_losses()
        self.setup_evaluator()
        
        self.loss_module = VASALossModule(self.config, self.device)

        # Resume if needed
        if resume_path:
            self.resume_from_checkpoint(resume_path)

      # Track training stages
        self.current_stage = None
        self.stages_completed = set()

    def train_all_stages(self):
        """Execute all training stages in order"""
        # Stage 1: Face Latent Space Learning
        if not self.is_stage_completed('latent_space'):
            self.train_latent_space()
            self.mark_stage_completed('latent_space')
        
        # Stage 2: Holistic Facial Dynamics Generation
        if not self.is_stage_completed('dynamics'):
            self.train_dynamics()
            self.mark_stage_completed('dynamics')

    def train_latent_space(self):
        """Stage 1: Face Latent Space Learning"""
        self.logger.info("Starting Stage 1: Face Latent Space Learning")
        self.current_stage = 'latent_space'
        
        # Initialize models for stage 1 # gbase from megaportraits - https://github.com/johndpope/MegaPortrait-hack
        face_encoder = VASAFaceEncoder().to(self.device)
        
        # Initialize loss components
        identity_loss = IdentityLoss().to(self.device)
        dpe_loss = DPELoss().to(self.device)
        
        # Optimizer for stage 1
        optimizer = torch.optim.Adam(
            face_encoder.parameters(),
            lr=self.config.latent_space_lr
        )
        
        # Training loop for stage 1
        for epoch in range(self.config.latent_space_epochs):
            face_encoder.train()
            epoch_losses = defaultdict(float)
            
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Extract face components
                face_components = face_encoder(batch['frames'])
                
                # Compute disentanglement losses
                losses = {}
                
                # Identity preservation loss
                losses['identity'] = identity_loss(
                    face_components['identity'],
                    batch['frames']
                )
                
                # DPE loss for disentanglement
                dpe_losses = dpe_loss(
                    batch['frames'][:, 0],  # Source frame
                    batch['frames'][:, 1:],  # Target frames
                    face_components['pose_transfer'],
                    face_components['expression_transfer']
                )
                losses.update(dpe_losses)
                
                # Total loss for stage 1
                total_loss = (
                    self.config.lambda_identity * losses['identity'] +
                    self.config.lambda_dpe * losses['dpe']
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v.item()
            
            # Log epoch metrics
            self.logger.log_metrics({
                'stage': 1,
                'epoch': epoch,
                **{f'stage1_{k}': v/len(self.train_loader) 
                   for k, v in epoch_losses.items()}
            })
        
        # Save stage 1 checkpoint
        self.save_stage_checkpoint(face_encoder, 'latent_space')

    def train_dynamics(self):
        """Stage 2: Holistic Facial Dynamics Generation"""
        self.logger.info("Starting Stage 2: Facial Dynamics Generation")
        self.current_stage = 'dynamics'
        
        # Load pre-trained face encoder
        face_encoder = self.load_stage_checkpoint('latent_space')
        face_encoder.eval()  # Freeze face encoder weights
        
        # Initialize diffusion model
        diffusion_model = VASADiffusionTransformer(
            d_model=self.config.model.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers
        ).to(self.device)
        
        # Initialize diffusion process
        diffusion = VASADiffusion(
            num_steps=self.config.num_steps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end
        )
        
        # Optimizer for stage 2
        optimizer = torch.optim.Adam(
            diffusion_model.parameters(),
            lr=self.config.dynamics_lr
        )
        
        # Training loop for stage 2
        for epoch in range(self.config.dynamics_epochs):
            diffusion_model.train()
            epoch_losses = defaultdict(float)
            
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Extract face dynamics with frozen encoder
                with torch.no_grad():
                    face_components = face_encoder(batch['frames'])
                
                # Sample timestep and add noise
                t = torch.randint(
                    0,
                    self.config.num_steps,
                    (batch['frames'].shape[0],),
                    device=self.device
                )
                
                noise = torch.randn_like(face_components['dynamics'])
                noisy_dynamics = diffusion.q_sample(
                    face_components['dynamics'],
                    t,
                    noise
                )
                
                # Prepare conditions
                conditions = {
                    'gaze': batch['gaze'],
                    'distance': batch['distance'],
                    'emotion': batch['emotion']
                }
                
                # Forward pass with CFG
                diffusion_output = diffusion_model(
                    noisy_dynamics,
                    t,
                    batch['audio_features'],
                    conditions,
                    {
                        'audio': self.config.cfg_audio_scale,
                        'gaze': self.config.cfg_gaze_scale
                    }
                )
                
                # Compute stage 2 losses
                losses = {}
                
                # Diffusion loss
                losses['diffusion'] = F.mse_loss(
                    diffusion_output['full'],
                    face_components['dynamics']
                )
                
                # CFG losses
                for cond_type in ['audio', 'gaze']:
                    if f'masked_{cond_type}' in diffusion_output:
                        losses[f'cfg_{cond_type}'] = F.mse_loss(
                            diffusion_output[f'masked_{cond_type}'],
                            diffusion_output['uncond']
                        )
                
                # Total loss for stage 2
                total_loss = (
                    losses['diffusion'] +
                    sum(self.config.lambda_cfg * losses[f'cfg_{k}']
                        for k in ['audio', 'gaze'])
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v.item()
            
            # Log epoch metrics
            self.logger.log_metrics({
                'stage': 2,
                'epoch': epoch,
                **{f'stage2_{k}': v/len(self.train_loader) 
                   for k, v in epoch_losses.items()}
            })
        
        # Save stage 2 checkpoint
        self.save_stage_checkpoint(diffusion_model, 'dynamics')

    def save_stage_checkpoint(self, model: nn.Module, stage: str):
        """Save stage-specific checkpoint"""
        checkpoint = {
            'model_state': model.state_dict(),
            'stage': stage,
            'config': self.config
        }
        path = self.logger.log_dir / f'checkpoint_stage_{stage}.pt'
        torch.save(checkpoint, path)

    def load_stage_checkpoint(self, stage: str) -> nn.Module:
        """Load stage-specific checkpoint"""
        path = self.logger.log_dir / f'checkpoint_stage_{stage}.pt'
        checkpoint = torch.load(path, map_location=self.device)
        
        if stage == 'latent_space':
            model = VASAFaceEncoder(
                feature_dim=self.config.model.d_model
            ).to(self.device)
        elif stage == 'dynamics':
            model = VASADiffusionTransformer(
                d_model=self.config.model.d_model,
                nhead=self.config.nhead,
                num_layers=self.config.num_layers
            ).to(self.device)
            
        model.load_state_dict(checkpoint['model_state'])
        return model

    def is_stage_completed(self, stage: str) -> bool:
        """Check if a training stage has been completed"""
        return stage in self.stages_completed

    def mark_stage_completed(self, stage: str):
        """Mark a training stage as completed"""
        self.stages_completed.add(stage)
        self.logger.info(f"Completed training stage: {stage}")

        
    def setup_distributed(self):
        """Initialize distributed training if needed"""
        self.distributed = self.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            
        self.device = torch.device(
            f'cuda:{self.local_rank}' if self.local_rank != -1 else 'cuda'
        )

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """Custom collate function to handle variable length sequences"""
            # Filter out any None or invalid samples
            batch = [b for b in batch if b is not None and all(v is not None for v in b.values())]
            if not batch:
                return self._get_empty_batch()
            
            # Stack tensors
            return {
                'frames': torch.stack([b['frames'] for b in batch]),
                'audio_features': torch.stack([b['audio_features'] for b in batch]),
                'gaze': torch.stack([b['gaze'] for b in batch]),
                'distance': torch.stack([b['distance'] for b in batch]),
                'emotion': torch.stack([b['emotion'] for b in batch]),
                'metadata': [b['metadata'] for b in batch]
            }

    def _get_empty_batch(self) -> Dict[str, torch.Tensor]:
        """Return an empty batch with correct dimensions"""
        return {
            'frames': torch.zeros((0, self.config.sequence_length, 3, *self.config.frame_size)),
            'audio_features': torch.zeros((0, self.config.sequence_length, 768)),
            'gaze': torch.zeros((0, self.config.sequence_length, 2)),
            'distance': torch.zeros((0, self.config.sequence_length, 1)),
            'emotion': torch.zeros((0, self.config.sequence_length, 8)),
            'metadata': []
        }

    def setup_models(self):
        """Initialize all model components"""
        # Create models
        self.face_encoder = VASAFaceEncoder(
            feature_size=self.config.model.d_model
        ).to(self.device)
        
        self.diffusion_model = VASADiffusionTransformer(
            d_model=self.config.model.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            seq_length=self.config.sequence_length,
            motion_dim=self.config.motion_dim,
            audio_dim=self.config.audio_dim
        ).to(self.device)
        
        self.face_decoder = VASAFaceDecoder().to(self.device)
        
        self.diffusion = VASADiffusion(
            num_steps=self.config.num_steps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end
        )
        
        # Wrap for distributed training
        if self.distributed:
            self.face_encoder = DDP(
                self.face_encoder,
                device_ids=[self.local_rank]
            )
            self.diffusion_model = DDP(
                self.diffusion_model,
                device_ids=[self.local_rank]
            )
            self.face_decoder = DDP(
                self.face_decoder,
                device_ids=[self.local_rank]
            )

    def setup_optimizers(self):
        """Initialize optimizers and schedulers"""
        # Optimizers
        self.encoder_optimizer = Adam(
            self.face_encoder.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        self.diffusion_optimizer = Adam(
            self.diffusion_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        self.decoder_optimizer = Adam(
            self.face_decoder.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Schedulers
        self.encoder_scheduler = CosineAnnealingLR(
            self.encoder_optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.min_lr
        )
        
        self.diffusion_scheduler = CosineAnnealingLR(
            self.diffusion_optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.min_lr
        )
        
        self.decoder_scheduler = CosineAnnealingLR(
            self.decoder_optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.min_lr
        )

    def setup_data(self):
        """Initialize data loaders with updated dataset configuration"""
        # Create datasets with new parameters
        train_dataset = VASADataset(
            video_folder=self.config.data_config['train_videos'],
            frame_size=self.config.frame_size,
            sequence_length=self.config.sequence_length,
            hop_length=self.config.hop_length,
            cache_audio=True,
            preextract_audio=self.config.preextract_audio,
            max_videos=self.config.max_videos,
            random_seed=self.config.random_seed
        )
        
        val_dataset = VASADataset(
            video_folder=self.config.data_config['val_videos'],
            frame_size=self.config.frame_size,
            sequence_length=self.config.sequence_length,
            hop_length=self.config.hop_length,
            cache_audio=True,
            preextract_audio=self.config.preextract_audio,
            max_videos=self.config.max_val_videos,
            random_seed=self.config.random_seed
        )
        
        # Setup samplers for distributed training
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )

    def setup_video_generator(self):
        """Initialize video generation pipeline"""
        self.video_generator = VideoGenerator(
            face_encoder=self.face_encoder,
            motion_generator=MotionGenerator(
                model=self.diffusion_model,
                sampler=self.diffusion,
                window_size=self.config.sequence_length,
                stride=self.config.sequence_length - self.config.overlap
            ),
            face_decoder=self.face_decoder,
            device=self.device
        )
        
        self.post_processor = VideoPostProcessor()

    def setup_losses(self):
        """Initialize loss functions"""
        self.identity_loss = IdentityLoss().to(self.device)
        self.dpe_loss = DPELoss().to(self.device)
        # self.vasa_loss = VASALoss().to(self.device)

    def setup_evaluator(self):
        """Initialize evaluation metrics"""
        self.evaluator = Evaluator()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with updated data handling"""
        # Skip empty batches
        if batch['frames'].size(0) == 0:
            return {'total': 0.0}
        
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.diffusion_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Extract face components
        face_components = self.face_encoder(batch['frames'])
        
        # Sample timestep and add noise
        t = torch.randint(
            0,
            self.config.num_steps,
            (batch['frames'].shape[0],),
            device=self.device
        )
        
        noise = torch.randn_like(face_components['dynamics'])
        noisy_dynamics = self.diffusion.q_sample(
            face_components['dynamics'],
            t,
            noise
        )
        
        # Prepare conditions with updated structure
        conditions = {
            'gaze': batch['gaze'],
            'distance': batch['distance'],
            'emotion': batch['emotion']
        }
        
        # Forward pass through diffusion model
        diffusion_output = self.diffusion_model(
            noisy_dynamics,
            t,
            batch['audio_features'],
            conditions,
            {
                'audio': self.config.cfg_audio_scale,
                'gaze': self.config.cfg_gaze_scale
            }
        )
        
        # Generate frames
        generated_frames = self.face_decoder(
            face_components['appearance_volume'],
            face_components['identity'],
            diffusion_output['full']
        )
        
        # Compute losses
        losses = self.loss_module.compute_losses(
            generated_frames=generated_frames,
            batch=batch,
            face_components=face_components,
            diffusion_output=diffusion_output
        )
        
        # Backward pass
        losses['total'].backward()
        
        # Clip gradients
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.face_encoder.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.diffusion_model.parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.face_decoder.parameters(),
                self.config.max_grad_norm
            )
        
        # Update model parameters
        self.encoder_optimizer.step()
        self.diffusion_optimizer.step()
        self.decoder_optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation"""
        self.face_encoder.eval()
        self.diffusion_model.eval()
        self.face_decoder.eval()
        
        val_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Generate video
            generated_video = self.video_generator.generate_video(
                batch['frames'][:1],
                batch['audio_features'][:1],
                {
                    'gaze': batch['gaze'][:1],
                    'distance': batch['distance'][:1],
                    'emotion': batch['emotion'][:1]
                },
                {
                    'audio': self.config.audio_scale,
                    'gaze': self.config.gaze_scale
                }
            )
            
            # Compute metrics
            metrics = self.evaluator.compute_metrics(
                generated_video,
                batch['audio_features'][:1],
                batch['frames'][:1]
            )
            
            # Accumulate metrics
            for k, v in metrics.items():
                val_metrics[k] += v
            num_batches += 1
        
        # Average metrics
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        # Log validation metrics
        self.logger.log_metrics(
            {f'val_{k}': v for k, v in val_metrics.items()},
            epoch
        )
        
        return val_metrics

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.face_encoder.train()
        self.diffusion_model.train()
        self.face_decoder.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
            # Training step
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1
            
            # Log step metrics
            self.global_step = epoch * len(self.train_loader) + batch_idx
            self.logger.log_metrics(metrics, self.global_step)
            
            # Generate samples periodically
            if batch_idx % self.config.generation_interval == 0:
                self.generate_samples(batch, self.global_step)
        
        # Average epoch metrics
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Log epoch metrics
        self.logger.log_metrics(
            {f'epoch_{k}': v for k, v in epoch_metrics.items()},
            epoch
        )
        
        # Update schedulers
        self.encoder_scheduler.step()
        self.diffusion_scheduler.step()
        self.decoder_scheduler.step()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'face_encoder_state': self.face_encoder.state_dict(),
            'diffusion_model_state': self.diffusion_model.state_dict(),
            'face_decoder_state': self.face_decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'encoder_scheduler': self.encoder_scheduler.state_dict(),
            'diffusion_scheduler': self.diffusion_scheduler.state_dict(),
            'decoder_scheduler': self.decoder_scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        save_path = self.logger.log_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.logger.log_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.face_encoder.load_state_dict(checkpoint['face_encoder_state'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state'])
        self.face_decoder.load_state_dict(checkpoint['face_decoder_state'])
        
        # Load optimizer states
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        
        # Load scheduler states
        self.encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler'])
        self.diffusion_scheduler.load_state_dict(checkpoint['diffusion_scheduler'])
        self.decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler'])
        
        # Restore training state
        self.global_step = checkpoint['global_step']
        
        return checkpoint['epoch']

    def train(self, start_epoch: int = 0):
        """Full training loop"""
        best_metric = float('inf')  # For sync_distance, lower is better
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                # Training epoch
                self.logger.logger.info(f"Starting epoch {epoch}")
                self.train_epoch(epoch)
                
                # Validation
                self.logger.logger.info("Running validation...")
                val_metrics = self.validate(epoch)
                
                # Check for best model
                current_metric = val_metrics['sync_distance']
                is_best = current_metric < best_metric
                if is_best:
                    best_metric = current_metric
                    self.logger.logger.info(f"New best model! Sync distance: {best_metric:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Log epoch summary
                self.logger.logger.info(
                    f"Epoch {epoch} completed | "
                    f"Train Loss: {self.last_train_loss:.4f} | "
                    f"Val Sync Distance: {current_metric:.4f} | "
                    f"Best Sync Distance: {best_metric:.4f}"
                )
                
                # Early stopping check
                if self.config.early_stopping and epoch > self.config.warmup_epochs:
                    if not self.check_improvement(current_metric):
                        self.logger.logger.info("Early stopping triggered!")
                        break
                
        except Exception as e:
            self.logger.logger.error(f"Training error: {str(e)}")
            raise
        
        finally:
            # Final cleanup and logging
            self.cleanup()

    def check_improvement(self, current_metric: float) -> bool:
        """Check if there's been improvement for early stopping"""
        self.metric_history.append(current_metric)
        if len(self.metric_history) > self.config.patience:
            # Check if there's been improvement in the last patience epochs
            best_recent = min(self.metric_history[-self.config.patience:])
            if best_recent >= min(self.metric_history[:-self.config.patience]):
                return False
        return True

    def cleanup(self):
        """Cleanup resources and finalize logging"""
        if self.distributed:
            dist.destroy_process_group()
        self.logger.logger.info("Training completed!")

    
    
    @torch.no_grad()
    def generate_samples(self, batch: Dict[str, torch.Tensor], step: int):
        """Generate and log video samples with updated data structure"""
        self.face_encoder.eval()
        self.diffusion_model.eval()
        self.face_decoder.eval()
        
        try:
            # Skip empty batches
            if batch['frames'].size(0) == 0:
                return
            
            # Generate video
            generated_video = self.video_generator.generate_video(
                source_image=batch['frames'][:1],
                audio_features=batch['audio_features'][:1],
                conditions={
                    'gaze': batch['gaze'][:1],
                    'distance': batch['distance'][:1],
                    'emotion': batch['emotion'][:1]
                },
                cfg_scales={
                    'audio': self.config.cfg_audio_scale,
                    'gaze': self.config.cfg_gaze_scale
                }
            )
            
            # Post-process video if enabled
            if self.config.apply_post_processing:
                generated_video = self.post_processor.apply_temporal_smoothing(
                    generated_video,
                    window_size=self.config.smoothing_window
                )
            
            # Save video sample
            if self.is_main_process():
                metadata = batch['metadata'][0]  # Get metadata for first sample
                self.save_video_sample(
                    generated_video,
                    step,
                    batch['audio_features'][:1],
                    metadata
                )
            
            # Store for later reference
            self.last_generated_video = generated_video
            
        except Exception as e:
            self.logger.logger.error(f"Error in sample generation: {str(e)}")
            
        finally:
            self.face_encoder.train()
            self.diffusion_model.train()
            self.face_decoder.train()

    def save_video_sample(
        self,
        video: torch.Tensor,
        step: int,
        audio: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        """Save generated video sample with metadata"""
        save_path = self.logger.log_dir / 'samples' / f'sample_step_{step}.mp4'
        save_path.parent.mkdir(exist_ok=True)
        
        try:
            self.logger.log_video(
                video=video.cpu().numpy(),
                audio=audio.cpu().numpy(),
                step=step,
                tag=f"generated_{metadata['video_name']}"
            )
            
            # Log additional metadata
            self.logger.log_metrics({
                'sample_metadata': {
                    'source_video': metadata['video_path'],
                    'start_frame': metadata['start_frame'],
                    'fps': metadata['fps']
                }
            }, step)
            
        except Exception as e:
            self.logger.logger.error(f"Error saving video sample: {str(e)}")

    def compute_losses(self,
                      generated: torch.Tensor,
                      target: torch.Tensor,
                      face_components: Dict[str, torch.Tensor],
                      diffusion_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all training losses"""
        losses = {}
        
        # Reconstruction loss
        losses['recon'] = F.l1_loss(generated, target)
        
        # Identity preservation loss
        losses['identity'] = self.identity_loss(generated, target)
        
        # DPE loss for disentanglement
        losses['dpe'] = self.dpe_loss(
            face_components['dynamics'],
            diffusion_output['full']
        )
        
        # CFG losses
        for cond_type in ['audio', 'gaze']:
            if f'masked_{cond_type}' in diffusion_output:
                losses[f'cfg_{cond_type}'] = F.mse_loss(
                    diffusion_output[f'masked_{cond_type}'],
                    diffusion_output['uncond']
                )
        
        # Weight and combine losses
        losses['total_loss'] = (
            self.config.lambda_recon * losses['recon'] +
            self.config.lambda_identity * losses['identity'] +
            self.config.lambda_dpe * losses['dpe'] +
            sum(self.config.lambda_cfg * loss 
                for name, loss in losses.items() 
                if name.startswith('cfg_'))
        )
        
        return losses

    def is_main_process(self) -> bool:
        """Check if this is the main process in distributed training"""
        return not self.distributed or self.local_rank == 0

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {
            'encoder': self.encoder_scheduler.get_last_lr()[0],
            'diffusion': self.diffusion_scheduler.get_last_lr()[0],
            'decoder': self.decoder_scheduler.get_last_lr()[0]
        }

    @property
    def last_train_loss(self) -> float:
        """Get the last training loss"""
        return self._last_train_loss if hasattr(self, '_last_train_loss') else float('nan')

    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for saving"""
        return {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'metric_history': self.metric_history,
            'models': {
                'face_encoder': self.face_encoder.state_dict(),
                'diffusion_model': self.diffusion_model.state_dict(),
                'face_decoder': self.face_decoder.state_dict()
            },
            'optimizers': {
                'encoder': self.encoder_optimizer.state_dict(),
                'diffusion': self.diffusion_optimizer.state_dict(),
                'decoder': self.decoder_optimizer.state_dict()
            },
            'schedulers': {
                'encoder': self.encoder_scheduler.state_dict(),
                'diffusion': self.diffusion_scheduler.state_dict(),
                'decoder': self.decoder_scheduler.state_dict()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state"""
        self.current_epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_metric = state_dict['best_metric']
        self.metric_history = state_dict['metric_history']
        
        # Load model states
        self.face_encoder.load_state_dict(state_dict['models']['face_encoder'])
        self.diffusion_model.load_state_dict(state_dict['models']['diffusion_model'])
        self.face_decoder.load_state_dict(state_dict['models']['face_decoder'])
        
        # Load optimizer states
        self.encoder_optimizer.load_state_dict(state_dict['optimizers']['encoder'])
        self.diffusion_optimizer.load_state_dict(state_dict['optimizers']['diffusion'])
        self.decoder_optimizer.load_state_dict(state_dict['optimizers']['decoder'])
        
        # Load scheduler states
        self.encoder_scheduler.load_state_dict(state_dict['schedulers']['encoder'])
        self.diffusion_scheduler.load_state_dict(state_dict['schedulers']['diffusion'])
        self.decoder_scheduler.load_state_dict(state_dict['schedulers']['decoder'])






'''
Stage 1: Latent Space Learning
- Focus on disentanglement
- Identity preservation
- DPE losses

Stage 2: Dynamics Generation
- Frozen face encoder
- Diffusion training
- CFG losses

'''

def main():
    # Parse arguments
   
    try:

        # Load config
        config = OmegaConf.load('./configs/vasa.yaml')
        # Create structured config
        schema = OmegaConf.structured(VASAConfig)
        # Merge configs
        config = OmegaConf.merge(schema, config)
        # Setup logger

            # Generate example name
        experiment_name =  get_vasa_exp_name(config)
        print(f"Generated experiment name: {experiment_name}")
        logger = TrainingLogger(
            experiment_name,
            Path(config.output_dir),
            use_wandb=config.use_wandb
        )
        
        # Initialize trainer
        trainer = VASATrainer(config, logger)
        trainer.train_all_stages()

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise

if __name__ == "__main__":
    main()

