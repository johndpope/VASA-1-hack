#!/usr/bin/env python3
"""
Run VASA Training with TDD Testing and Automatic Fixes
=======================================================
This script runs training with automatic adjustments based on test failures.
"""

import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
import logging
from omegaconf import OmegaConf, DictConfig
import importlib
import wandb
from torch.utils.data import DataLoader, random_split
import traceback
import gc
from typing import Dict, Any
import json
from datetime import datetime

# Add nemo to path
if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from vasa_dataset import VASAIntegratedDataset
from vasa_trainer_tdd import VASATrainerWithTDD
from vasa_trainer import collate_vasa_batch, worker_init_fn
from vasa_tdd_tests import TestResult
from logger import logger

# Configure rich logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)


class AutoFixTrainer:
    """Trainer that automatically adjusts based on test results."""
    
    def __init__(self, base_config: DictConfig):
        self.base_config = base_config
        self.current_config = OmegaConf.create(OmegaConf.to_container(base_config))
        self.iteration = 0
        self.fix_history = []
        self.test_history = []
        
    def analyze_failures(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test failures and determine fixes."""
        fixes = {
            'learning_rate': None,
            'window_size': None,
            'loss_weights': {},
            'noise': None,
            'control_epoch': None,
            'message': []
        }
        
        # Check static reconstruction
        if 'Static Reconstruction' in test_results:
            result = test_results['Static Reconstruction']
            if not result.get('passed', False):
                loss = result.get('score', 1.0)
                if loss > 0.5:
                    # Major reconstruction issue
                    fixes['learning_rate'] = self.current_config.train.lr * 2.0
                    fixes['noise'] = True  # Turn off noise
                    fixes['loss_weights']['reconstruction'] = 20.0
                    fixes['message'].append(f"📈 Doubling LR to {fixes['learning_rate']:.4f} due to high recon loss ({loss:.4f})")
                elif loss > 0.2:
                    # Minor reconstruction issue
                    fixes['loss_weights']['reconstruction'] = 10.0
                    fixes['message'].append(f"⚖️ Increasing reconstruction weight to 10.0")
        
        # Check motion quality
        if 'Optical Flow Consistency' in test_results:
            result = test_results['Optical Flow Consistency']
            if not result.get('passed', False):
                consistency = result.get('score', 0.0)
                if consistency < 0.1:
                    # No motion detected
                    if self.current_config.motion.window_size > 1:
                        fixes['window_size'] = max(1, self.current_config.motion.window_size // 2)
                        fixes['message'].append(f"📏 Reducing window size to {fixes['window_size']}")
                    fixes['loss_weights']['dynamics'] = 0.0
                    fixes['message'].append("🚫 Disabling dynamics loss until reconstruction works")
        
        # Check temporal coherence
        if 'Temporal Coherence' in test_results:
            result = test_results['Temporal Coherence']
            if not result.get('passed', False):
                coherence = result.get('score', 0.0)
                if coherence < 0.3 and self.current_config.motion.window_size > 5:
                    fixes['loss_weights']['temporal'] = 5.0
                    fixes['message'].append("🔗 Increasing temporal loss weight to 5.0")
        
        return fixes
    
    def apply_fixes(self, fixes: Dict[str, Any]) -> DictConfig:
        """Apply fixes to configuration."""
        new_config = OmegaConf.create(OmegaConf.to_container(self.current_config))
        
        if fixes['learning_rate'] is not None:
            new_config.train.lr = fixes['learning_rate']
            # Also adjust other learning rates proportionally
            for key in new_config.train.learning_rates:
                new_config.train.learning_rates[key] *= 2.0
        
        if fixes['window_size'] is not None:
            new_config.motion.window_size = fixes['window_size']
            new_config.dataset.sequence_length = fixes['window_size']
        
        if fixes['noise'] is not None:
            new_config.train.turn_off_noise = fixes['noise']
        
        if fixes['control_epoch'] is not None:
            new_config.train.control_start_epoch = fixes['control_epoch']
        
        for loss_name, weight in fixes['loss_weights'].items():
            if hasattr(new_config.loss, f'lambda_{loss_name}'):
                setattr(new_config.loss, f'lambda_{loss_name}', weight)
        
        # Log fixes
        if fixes['message']:
            console.print("\n[bold yellow]📐 Applying Automatic Fixes:[/bold yellow]")
            for msg in fixes['message']:
                console.print(f"  {msg}")
        
        self.fix_history.append({
            'iteration': self.iteration,
            'fixes': fixes,
            'timestamp': datetime.now().isoformat()
        })
        
        return new_config
    
    def should_restart_training(self, test_results: Dict[str, Any], epoch: int) -> bool:
        """Determine if training should restart with new config."""
        critical_tests = ['Static Reconstruction']
        
        for test_name in critical_tests:
            if test_name in test_results:
                result = test_results[test_name]
                if not result.get('passed', False):
                    # Check if we're making progress
                    if epoch > 10 and result.get('score', 1.0) > 0.5:
                        return True  # Not improving, restart with fixes
        
        return False
    
    def update_progressive_schedule(self, epoch: int):
        """Update configuration based on progressive schedule."""
        schedule_updates = []
        
        # Window size schedule
        if epoch == 10 and self.current_config.motion.window_size == 1:
            self.current_config.motion.window_size = 3
            self.current_config.dataset.sequence_length = 3
            schedule_updates.append("📏 Window size: 1 → 3")
        elif epoch == 20 and self.current_config.motion.window_size == 3:
            self.current_config.motion.window_size = 10
            self.current_config.dataset.sequence_length = 10
            schedule_updates.append("📏 Window size: 3 → 10")
        elif epoch == 30 and self.current_config.motion.window_size == 10:
            self.current_config.motion.window_size = 25
            self.current_config.dataset.sequence_length = 25
            schedule_updates.append("📏 Window size: 10 → 25")
        
        # Loss weight schedule
        if epoch == 10:
            self.current_config.loss.lambda_dynamics = 2.0
            self.current_config.loss.lambda_temporal = 1.0
            schedule_updates.append("⚖️ Enabled dynamics and temporal losses")
        elif epoch == 20:
            self.current_config.loss.lambda_pose = 1.0
            self.current_config.loss.lambda_reconstruction = 5.0
            schedule_updates.append("⚖️ Enabled pose loss, reduced reconstruction weight")
        elif epoch == 30:
            self.current_config.loss.lambda_control = 1.0
            self.current_config.train.turn_off_noise = False
            schedule_updates.append("🎮 Enabled control losses and noise")
        
        if schedule_updates:
            console.print(f"\n[bold green]📅 Progressive Schedule Updates (Epoch {epoch}):[/bold green]")
            for update in schedule_updates:
                console.print(f"  {update}")
        
        return self.current_config


def print_test_summary(test_results: Dict[str, Any]):
    """Print a nice summary of test results."""
    table = Table(title="Test Results Summary", show_lines=True)
    table.add_column("Test Name", style="cyan", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Score", justify="right", width=12)
    table.add_column("Target", justify="right", width=12)
    table.add_column("Message", width=40)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            status = "✅" if result.get('passed', False) else "❌"
            score = f"{result.get('score', 0.0):.4f}"
            target = f"{result.get('target', 0.0):.4f}"
            message = result.get('message', '')[:40]
            
            # Color code based on performance
            if result.get('passed', False):
                score = f"[green]{score}[/green]"
            else:
                score = f"[red]{score}[/red]"
            
            table.add_row(test_name, status, score, target, message)
    
    console.print(table)


def setup_environment():
    """Set up environment for optimal CUDA operation."""
    import os
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    
    gc.collect()
    torch.cuda.empty_cache()


def load_volumetric_model(config):
    """Load the volumetric avatar model."""
    logger.info("Loading volumetric avatar model...")
    
    model_path = config.paths.volumetric_model
    emo_config = OmegaConf.load(config.paths.volumetric_config)
    
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(emo_config, training=False)
    
    model_dict = torch.load(model_path, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda()
    volumetric_avatar.eval()
    
    logger.info("✅ Volumetric avatar model loaded successfully")
    return volumetric_avatar


def main():
    """Main training function with automatic fixes."""
    try:
        mp.set_start_method('spawn', force=True)
        
        # Load base configuration
        console.print("[bold blue]🔧 Loading Fixed Configuration...[/bold blue]")
        config = OmegaConf.load('vasa_config_fixed.yaml')
        
        # Initialize auto-fix trainer
        auto_trainer = AutoFixTrainer(config)
        
        # Set up environment
        setup_environment()
        
        # Initialize wandb
        if config.wandb.enabled:
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.get('name', 'vasa')}_auto_fix",
                config=OmegaConf.to_container(config, resolve=True),
                tags=['tdd', 'auto-fix', 'progressive']
            )
        
        # Load models and data
        volumetric_avatar = load_volumetric_model(config)
        
        # Training loop with automatic restarts
        max_iterations = 5
        for iteration in range(max_iterations):
            auto_trainer.iteration = iteration
            
            console.print(f"\n[bold magenta]🔄 Training Iteration {iteration + 1}/{max_iterations}[/bold magenta]")
            
            # Create VASA model with current config
            current_config = auto_trainer.current_config
            
            model = VASAModel(
                config=current_config,
                volumetric_avatar=volumetric_avatar,
                device=current_config.device
            )
            model = model.cuda()
            
            # Create dataset with current config
            full_dataset = VASAIntegratedDataset(
                video_folder=current_config.paths.video_folder,
                emo_model=volumetric_avatar,
                max_videos=current_config.dataset.max_videos,
                frame_size=(512, 512),
                sequence_length=current_config.dataset.sequence_length,
                cache_audio=True,
                preextract_audio=True,
                random_seed=42 + iteration  # Different seed each iteration
            )
            
            # Split dataset
            val_size = int(current_config.dataset.val_split * len(full_dataset))
            train_size = len(full_dataset) - val_size
            
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=current_config.train.batch_size,
                shuffle=True,
                num_workers=current_config.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_vasa_batch,
                persistent_workers=False,
                worker_init_fn=worker_init_fn if current_config.num_workers > 0 else None
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_vasa_batch
            )
            
            # Create output directory
            output_dir = Path(current_config.paths.checkpoint_dir) / f"tdd_auto_fix_iter_{iteration}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create TDD trainer with hooks for auto-fixing
            class AutoFixTDDTrainer(VASATrainerWithTDD):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.auto_trainer = auto_trainer
                    self.last_test_results = {}
                
                def train_epoch(self):
                    # Update progressive schedule
                    self.config = self.auto_trainer.update_progressive_schedule(self.current_epoch)
                    
                    # Run normal training epoch
                    epoch_stats = super().train_epoch()
                    
                    # Store test results
                    if hasattr(self, 'test_runner') and self.test_runner:
                        if self.test_runner.results_history:
                            self.last_test_results = self.test_runner.results_history[-1].get('results', {})
                            
                            # Print test summary
                            print_test_summary(self.last_test_results)
                            
                            # Check if we need to apply fixes
                            if self.current_epoch % 5 == 0:  # Check every 5 epochs
                                fixes = self.auto_trainer.analyze_failures(self.last_test_results)
                                if any(v for v in fixes.values() if v and v != {}):
                                    new_config = self.auto_trainer.apply_fixes(fixes)
                                    self.config = new_config
                                    self.auto_trainer.current_config = new_config
                    
                    return epoch_stats
                
                def should_early_stop(self):
                    # Check if we should restart with fixes
                    if self.auto_trainer.should_restart_training(self.last_test_results, self.current_epoch):
                        console.print("[bold red]🔄 Restarting training with fixes...[/bold red]")
                        return True
                    
                    return super().should_early_stop()
            
            # Create trainer
            trainer = AutoFixTDDTrainer(
                model=model,
                config=current_config,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
                tdd_config=current_config.tdd
            )
            
            # Run training
            console.print("[bold green]🚀 Starting Training...[/bold green]")
            trainer.train()
            
            # Check if we achieved good results
            if trainer.test_runner and trainer.test_runner.results_history:
                last_results = trainer.test_runner.results_history[-1]
                pass_rate = last_results['passed_count'] / last_results['total_count']
                
                if pass_rate > 0.8:  # 80% tests passing
                    console.print(f"[bold green]✅ Training successful! Pass rate: {pass_rate:.1%}[/bold green]")
                    break
                else:
                    console.print(f"[bold yellow]⚠️ Pass rate: {pass_rate:.1%}. Applying fixes...[/bold yellow]")
                    
                    # Analyze and prepare fixes for next iteration
                    fixes = auto_trainer.analyze_failures(last_results.get('results', {}))
                    auto_trainer.current_config = auto_trainer.apply_fixes(fixes)
            
            # Clean up for next iteration
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save final report
        report_path = Path(current_config.paths.checkpoint_dir) / "auto_fix_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'iterations': auto_trainer.iteration + 1,
                'fix_history': auto_trainer.fix_history,
                'final_config': OmegaConf.to_container(auto_trainer.current_config),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        console.print(f"\n[bold green]📊 Auto-fix report saved to: {report_path}[/bold green]")
        
        if config.wandb.enabled:
            wandb.finish()
        
        console.print("\n[bold green]✅ Training with Auto-Fix completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Training failed: {str(e)}[/bold red]")
        logger.error(traceback.format_exc())
        
        if 'wandb' in locals() and wandb.run is not None:
            wandb.finish(exit_code=1)
        
        raise


if __name__ == "__main__":
    main()