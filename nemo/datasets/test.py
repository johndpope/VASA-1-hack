import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import sys
from datetime import datetime

from celebvhq_pairs import DataModule, add_dataset_args

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = save_dir / 'training.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def save_checkpoint(model, optimizer, epoch, save_path, metrics=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', None)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup save directory
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.save_dir)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')
        
        # Initialize dataset
        self.data_module = DataModule(args)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        
        # Initialize model (placeholder - replace with your actual model)
        self.model = self.init_model()
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )
        
        # Initialize criterion
        self.criterion = nn.MSELoss()
        
        # Load checkpoint if specified
        self.start_epoch = 0
        if args.resume and args.checkpoint_path:
            self.start_epoch, metrics = load_checkpoint(
                self.model, 
                self.optimizer,
                args.checkpoint_path
            )
            self.logger.info(f"Resumed from checkpoint at epoch {self.start_epoch}")
            if metrics:
                self.logger.info(f"Previous metrics: {metrics}")

    def init_model(self):
        """Initialize model (placeholder - replace with your model)"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', total=num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                source_images = batch['source_img'].to(self.device)
                target_images = batch['target_img'].to(self.device)
                source_masks = batch['source_mask'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(source_images)
                
                # Calculate loss
                loss = self.criterion(outputs, target_images)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                current_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                # Log to tensorboard
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                
                # Save sample images periodically
                if batch_idx % self.args.save_image_interval == 0:
                    self.save_samples(
                        source_images, target_images, outputs,
                        epoch, batch_idx
                    )
        
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with tqdm(self.val_loader, desc='Validation', total=num_batches) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                source_images = batch['source_img'].to(self.device)
                target_images = batch['target_img'].to(self.device)
                source_masks = batch['source_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(source_images)
                
                # Calculate loss
                loss = self.criterion(outputs, target_images)
                
                # Update statistics
                total_loss += loss.item()
                current_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})
        
        val_loss = total_loss / num_batches
        self.writer.add_scalar('val/loss', val_loss, epoch)
        
        return val_loss

    def save_samples(self, source_images, target_images, outputs, epoch, batch_idx):
        """Save sample images"""
        # Create grid of images
        num_samples = min(8, source_images.size(0))
        samples = []
        
        for i in range(num_samples):
            samples.extend([
                source_images[i], 
                target_images[i], 
                outputs[i]
            ])
        
        # Convert to grid
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=3, normalize=True, pad_value=1)
        
        # Save grid
        save_path = self.save_dir / 'samples' / f'epoch_{epoch}_batch_{batch_idx}.png'
        save_path.parent.mkdir(exist_ok=True)
        from torchvision.utils import save_image
        save_image(grid, save_path)
        
        # Log to tensorboard
        self.writer.add_image('samples', grid, epoch)

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Log metrics
            self.logger.info(
                f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}'
            )
            
            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.save_dir / f'checkpoint_epoch_{epoch}.pt',
                metrics
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.save_dir / 'best_model.pt',
                    metrics
                )
                self.logger.info(f'New best model saved with val_loss={val_loss:.4f}')
            
            # Learning rate scheduling could be added here
            
        self.writer.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='CelebVHQ Training Script')
    
    # Add dataset arguments
    parser = add_dataset_args(parser)
    
    # Add training arguments
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                      help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                      help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                      help='Beta2 for Adam optimizer')
    parser.add_argument('--save_image_interval', type=int, default=100,
                      help='How often to save sample images')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()