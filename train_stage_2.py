import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from pathlib import Path

from dataset import VASADataset
from VASA import VASAFaceEncoder, VASADiffusionTransformer
# from trainers import VASAStage2Trainer
import torch
from torch.utils.data import DataLoader
from einops import rearrange

class VASAStage2Trainer:
    """
    Implements Stage 2 of VASA training - the Diffusion Transformer for motion generation
    """
    def __init__(
        self, 
        face_encoder,
        diffusion_model,
        config,
        device='cuda'
    ):
        self.face_encoder = face_encoder
        self.diffusion_model = diffusion_model
        self.config = config
        self.device = device

        # Freeze the face encoder weights
        for param in self.face_encoder.parameters():
            param.requires_grad = False

    def train_step(self, batch):
        """Single training step for the diffusion model"""
        # Unpack batch
        images = batch['frames'].to(self.device)
        audio_features = batch['audio_features'].to(self.device)
        gaze = batch['gaze'].to(self.device)
        emotion = batch['emotion'].to(self.device)

        # Extract facial representations using frozen encoder
        with torch.no_grad():
            facial_reps = self.face_encoder.encode_holistic(
                images[:, 0],  # Source frame
                gaze=gaze[:, 0],
                emotion=emotion[:, 0]
            )

        # Get facial dynamics sequence for training
        facial_dynamics = facial_reps['facial_dynamics']
        
        # Sample timestep
        batch_size = facial_dynamics.shape[0]
        t = torch.randint(0, self.config.num_steps, (batch_size,), device=self.device)

        # Add noise to dynamics
        noise = torch.randn_like(facial_dynamics)
        noisy_dynamics = self.diffusion_model.add_noise(facial_dynamics, t, noise)

        # Prepare conditions
        conditions = {
            'audio': audio_features,
            'gaze': gaze,
            'emotion': emotion
        }

        # Forward pass with classifier-free guidance
        model_output = self.diffusion_model(
            noisy_dynamics,
            t,
            conditions=conditions,
            use_cfg=self.training
        )

        # Compute loss
        loss = F.mse_loss(model_output['predicted_noise'], noise)

        # Add CFG losses if training
        if self.training and self.config.cfg_scales is not None:
            for cond_type, scale in self.config.cfg_scales.items():
                if scale > 0 and f'masked_{cond_type}' in model_output:
                    cfg_loss = F.mse_loss(
                        model_output[f'masked_{cond_type}'],
                        model_output['uncond']
                    )
                    loss = loss + self.config.lambda_cfg * cfg_loss

        return loss

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.diffusion_model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            loss = self.train_step(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    @torch.no_grad()
    def generate_motion(self, audio_features, conditions, cfg_scales=None):
        """Generate motion sequence using trained diffusion model"""
        self.diffusion_model.eval()
        
        # Initialize from noise
        motion = torch.randn(
            (1, self.config.sequence_length, self.config.motion_dim),
            device=self.device
        )
        
        # Iterative denoising
        for t in reversed(range(self.config.num_steps)):
            timesteps = torch.full((1,), t, device=self.device)
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


def train_vasa_stage2(config_path: str):
    """
    Main training function for VASA Stage 2
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project="vasa",
            name=f"stage2_{config.experiment_name}",
            config=OmegaConf.to_container(config, resolve=True)
        )
    
    # Initialize datasets
    train_dataset = VASADataset(
        video_folder=config.data.train_path,
        frame_size=tuple(config.data.frame_size),
        sequence_length=config.model.sequence_length,
        hop_length=config.data.hop_length,
        max_videos=config.data.max_videos,
        cache_audio=True,
        preextract_audio=True
    )
    
    val_dataset = VASADataset(
        video_folder=config.data.val_path,
        frame_size=tuple(config.data.frame_size),
        sequence_length=config.model.sequence_length,
        hop_length=config.data.hop_length,
        max_videos=config.data.max_val_videos,
        cache_audio=True,
        preextract_audio=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Load pretrained face encoder
    face_encoder = VASAFaceEncoder(
        feature_dim=config.model.feature_dim
    ).to(device)
    
    # Load encoder checkpoint
    encoder_ckpt = torch.load(config.model.encoder_checkpoint)
    face_encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    face_encoder.eval()  # Set to eval mode since we're freezing it
    
    # Initialize diffusion model
    diffusion_model = VASADiffusionTransformer(
        seq_length=config.model.sequence_length,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        motion_dim=config.model.motion_dim,
        audio_dim=config.model.audio_dim
    ).to(device)
    
    # Initialize trainer
    trainer = VASAStage2Trainer(
        face_encoder=face_encoder,
        diffusion_model=diffusion_model,
        config=config,
        device=device
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        diffusion_model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=config.training.min_lr
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        # Train epoch
        train_loss = trainer.train_epoch(train_loader, optimizer)
        
        # Validation
        val_loss = validate(trainer, val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        if config.use_wandb:
            wandb.log(metrics)
            
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                trainer,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config.training.checkpoint_dir,
                is_best=True
            )
            
        # Regular checkpoint saving
        if epoch % config.training.save_frequency == 0:
            save_checkpoint(
                trainer,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config.training.checkpoint_dir
            )

def validate(trainer, val_loader):
    """Run validation"""
    trainer.diffusion_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = trainer.train_step(batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def save_checkpoint(trainer, optimizer, scheduler, epoch, metrics, checkpoint_dir, is_best=False):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainer.diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    if is_best:
        path = checkpoint_dir / 'best_model.pt'
    else:
        path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
    torch.save(checkpoint, path)

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage2.yaml")
    train_vasa_stage2(config)