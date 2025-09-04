#!/usr/bin/env python3
"""
TDD Training with Comprehensive WandB Logging
==============================================
Tracks all test results, metrics, and improvements in wandb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import logging

if 'nemo' not in sys.path:
    sys.path.insert(0, 'nemo')

from vasa_model import VASAModel
from tdd_loss_balanced import BalancedTDDLoss
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TDDWandBLogger:
    """Enhanced WandB logging for TDD training"""
    
    def __init__(self, project_name="vasa-tdd", run_name=None):
        """Initialize wandb with TDD-specific configuration"""
        
        if run_name is None:
            run_name = f"tdd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize wandb with custom config
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                "training_type": "test_driven_development",
                "loss_module": "DifferentiableTDDLoss",
                "test_criteria": {
                    "motion_variance": {"min": 0.01, "max": 0.5, "target": 0.1},
                    "rotation_variance": {"min": 0.001, "max": 0.3, "target": 0.05},
                    "expression_variation": {"min": 1.0, "max": 10.0, "target": 3.0},
                    "temporal_consistency": {"min": 0.7, "target": 0.9},
                    "reconstruction": {"max": 0.1, "target": 0.02},
                    "audio_sync": {"min": 0.3, "target": 0.7},
                },
                "mode": "offline"  # Set to "online" for cloud logging
            },
            mode="offline",
            tags=["tdd", "test-driven", "motion-quality"]
        )
        
        # Define custom charts
        self._define_custom_charts()
        
        # Track test history
        self.test_history = []
        self.improvement_history = []
        
    def _define_custom_charts(self):
        """Define custom wandb charts for TDD metrics"""
        
        # Test pass rate over time
        wandb.define_metric("tdd/test_pass_rate", summary="max")
        wandb.define_metric("tdd/tests_passed", summary="max")
        wandb.define_metric("tdd/tests_total", summary="last")
        
        # Individual test results
        test_names = [
            "motion_variance", "rotation_variance", "expression_variation",
            "temporal_consistency", "reconstruction", "audio_sync", "not_static"
        ]
        for test in test_names:
            wandb.define_metric(f"tdd/test_{test}", summary="last")
            wandb.define_metric(f"tdd/metric_{test}", summary="mean")
        
        # Loss components
        wandb.define_metric("loss/total", summary="min")
        wandb.define_metric("loss/motion_variance", summary="mean")
        wandb.define_metric("loss/static_penalty", summary="mean")
        wandb.define_metric("loss/reconstruction", summary="mean")
        
        # Motion quality metrics
        wandb.define_metric("motion/magnitude", summary="mean")
        wandb.define_metric("motion/theta_variance", summary="mean")
        wandb.define_metric("motion/rotation_variance", summary="mean")
        wandb.define_metric("motion/expression_variance", summary="mean")
        
        # Training metrics
        wandb.define_metric("train/learning_rate", summary="last")
        wandb.define_metric("train/gradient_norm", summary="mean")
        
    def log_tdd_step(
        self,
        step: int,
        losses: Dict[str, torch.Tensor],
        test_info: Dict,
        learning_rate: float = None,
        grad_norm: float = None
    ):
        """Log comprehensive TDD metrics for a training step"""
        
        # Prepare metrics dictionary
        metrics = {
            "step": step,
            "loss/total": losses['total'].item(),
        }
        
        # Log individual losses
        for key, value in losses.items():
            if key != 'total' and isinstance(value, torch.Tensor):
                metrics[f"loss/{key}"] = value.item()
        
        # Log test results (pass/fail as 1/0)
        test_results = test_info.get('test_results', {})
        tests_passed = sum(test_results.values())
        tests_total = len(test_results)
        
        metrics.update({
            "tdd/test_pass_rate": test_info.get('passed_ratio', 0) * 100,
            "tdd/tests_passed": tests_passed,
            "tdd/tests_total": tests_total,
        })
        
        # Log individual test results
        for test_name, passed in test_results.items():
            metrics[f"tdd/test_{test_name}"] = 1 if passed else 0
        
        # Log motion metrics
        motion_metrics = test_info.get('metrics', {})
        for metric_name, value in motion_metrics.items():
            if isinstance(value, (int, float)):
                if 'var' in metric_name:
                    metrics[f"motion/{metric_name}"] = value
                else:
                    metrics[f"tdd/metric_{metric_name}"] = value
        
        # Log training metrics
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate
        if grad_norm is not None:
            metrics["train/gradient_norm"] = grad_norm
        
        # Track test improvements
        self._track_improvements(step, test_results)
        
        # Log to wandb
        wandb.log(metrics, step=step)
        
        # Log test summary table periodically
        if step % 100 == 0:
            self._log_test_summary_table(step, test_results, motion_metrics)
    
    def _track_improvements(self, step: int, test_results: Dict[str, bool]):
        """Track which tests improved over time"""
        
        if not self.test_history:
            # First step - all tests are "new"
            improvements = {name: "new" for name in test_results}
        else:
            # Compare with previous results
            prev_results = self.test_history[-1]['results']
            improvements = {}
            for name, passed in test_results.items():
                if name not in prev_results:
                    improvements[name] = "new"
                elif not prev_results[name] and passed:
                    improvements[name] = "improved"
                elif prev_results[name] and not passed:
                    improvements[name] = "regressed"
                else:
                    improvements[name] = "unchanged"
        
        self.test_history.append({
            'step': step,
            'results': test_results.copy(),
            'improvements': improvements
        })
        
        # Log improvements
        improved_count = sum(1 for v in improvements.values() if v == "improved")
        regressed_count = sum(1 for v in improvements.values() if v == "regressed")
        
        if improved_count > 0 or regressed_count > 0:
            wandb.log({
                "tdd/tests_improved": improved_count,
                "tdd/tests_regressed": regressed_count,
            }, step=step)
    
    def _log_test_summary_table(self, step: int, test_results: Dict, metrics: Dict):
        """Log a summary table of all test results"""
        
        # Create table data
        table_data = []
        for test_name, passed in test_results.items():
            row = [
                test_name,
                "✅ PASS" if passed else "❌ FAIL",
                metrics.get(test_name.replace('_', '_'), "N/A"),
                self._get_test_criteria(test_name)
            ]
            table_data.append(row)
        
        # Create and log table
        table = wandb.Table(
            columns=["Test Name", "Status", "Current Value", "Criteria"],
            data=table_data
        )
        wandb.log({"tdd/test_summary": table}, step=step)
    
    def _get_test_criteria(self, test_name: str) -> str:
        """Get criteria string for a test"""
        criteria_map = {
            "motion_variance": "[0.01, 0.5]",
            "rotation_variance": "[0.001, 0.3]",
            "expression_variation": "[1.0, 10.0]",
            "temporal_consistency": ">= 0.7",
            "reconstruction": "<= 0.1",
            "audio_sync": ">= 0.3",
            "not_static": ">= 0.001"
        }
        return criteria_map.get(test_name, "N/A")
    
    def log_validation_results(
        self,
        epoch: int,
        val_loss: float,
        test_pass_rate: float,
        detailed_results: Dict
    ):
        """Log validation results with detailed breakdown"""
        
        metrics = {
            "epoch": epoch,
            "val/loss": val_loss,
            "val/test_pass_rate": test_pass_rate * 100,
        }
        
        # Add detailed test results
        for test_name, result in detailed_results.items():
            metrics[f"val/test_{test_name}"] = 1 if result else 0
        
        wandb.log(metrics, step=epoch)
        
    def log_checkpoint(
        self,
        checkpoint_path: str,
        test_score: float,
        step: int,
        is_best: bool = False
    ):
        """Log checkpoint information"""
        
        artifact_name = "best_model" if is_best else f"checkpoint_step_{step}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "test_score": test_score,
                "step": step,
                "is_best": is_best
            }
        )
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
        
        # Also log as metric
        wandb.log({
            "checkpoint/saved": 1,
            "checkpoint/test_score": test_score,
            "checkpoint/is_best": 1 if is_best else 0
        }, step=step)
    
    def create_test_evolution_plot(self):
        """Create a plot showing how tests evolved over training"""
        
        if not self.test_history:
            return
        
        import matplotlib.pyplot as plt
        
        # Extract test names
        test_names = list(self.test_history[0]['results'].keys())
        
        # Create matrix of test results
        steps = [h['step'] for h in self.test_history]
        matrix = np.zeros((len(test_names), len(steps)))
        
        for i, history in enumerate(self.test_history):
            for j, test_name in enumerate(test_names):
                matrix[j, i] = 1 if history['results'].get(test_name, False) else 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_xticks(range(0, len(steps), max(1, len(steps)//10)))
        ax.set_xticklabels([steps[i] for i in range(0, len(steps), max(1, len(steps)//10))])
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Test Name')
        ax.set_title('Test Results Evolution During Training')
        
        plt.colorbar(im, ax=ax, label='Pass (1) / Fail (0)')
        plt.tight_layout()
        
        wandb.log({"tdd/test_evolution": wandb.Image(fig)})
        plt.close()
    
    def finish(self):
        """Finalize logging and create summary plots"""
        
        # Create evolution plot
        self.create_test_evolution_plot()
        
        # Log final summary
        if self.test_history:
            final_results = self.test_history[-1]['results']
            final_pass_rate = sum(final_results.values()) / len(final_results)
            
            wandb.summary['final_test_pass_rate'] = final_pass_rate * 100
            wandb.summary['total_steps'] = len(self.test_history)
            
            # Count total improvements
            total_improvements = sum(
                1 for h in self.test_history 
                for v in h['improvements'].values() 
                if v == 'improved'
            )
            wandb.summary['total_test_improvements'] = total_improvements
        
        wandb.finish()


def train_with_wandb():
    """Main training function with comprehensive wandb logging"""
    
    print("\n" + "="*60)
    print("TDD Training with WandB Logging")
    print("="*60)
    
    # Initialize wandb logger
    wandb_logger = TDDWandBLogger(
        project_name="vasa-tdd",
        run_name="tdd_full_tracking"
    )
    
    # Load config and models
    config = OmegaConf.load('vasa_config_fixed.yaml')
    
    print("\n1. Loading volumetric avatar...")
    volumetric_config = OmegaConf.load(config.paths.volumetric_config)
    volumetric_avatar = importlib.import_module(
        'models.stage_1.volumetric_avatar.va'
    ).Model(volumetric_config, training=False)
    
    model_dict = torch.load(config.paths.volumetric_model, map_location='cuda')
    volumetric_avatar.load_state_dict(model_dict, strict=False)
    volumetric_avatar = volumetric_avatar.cuda().eval()
    
    print("2. Creating VASA model...")
    model = VASAModel(config, volumetric_avatar).cuda().train()
    
    print("3. Initializing Balanced TDD loss module...")
    tdd_loss = BalancedTDDLoss(config, device='cuda', stage='motion_first')
    
    # Optimizer with higher learning rate for motion
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5
    )
    
    # Training parameters
    B, T = 2, 30
    num_steps = 2000  # More steps for curriculum learning
    checkpoint_dir = Path("checkpoints/tdd_wandb")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_test_score = 0.0
    
    print(f"\n4. Starting TDD training: B={B}, T={T}, Steps={num_steps}")
    print("-" * 60)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Create realistic targets
        t = torch.linspace(0, 4*3.14159, T).cuda()
        targets = {
            'theta': torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).cuda(),
            'scale': torch.ones(B, T, 3).cuda(),
            'rotation': torch.zeros(B, T, 3).cuda(),
            'translation': torch.zeros(B, T, 3).cuda(),
            'expression_embed': torch.randn(B, T, 128).cuda() * 0.3
        }
        
        # Add realistic motion patterns
        targets['rotation'][:, :, 0] = torch.sin(t * 0.5) * 0.15  # Pitch
        targets['rotation'][:, :, 1] = torch.cos(t * 0.7) * 0.2   # Yaw
        targets['rotation'][:, :, 2] = torch.sin(t * 0.3) * 0.05  # Roll
        
        # Update theta with rotation
        for b in range(B):
            for ti in range(T):
                # Create rotation matrix from Euler angles
                pitch = targets['rotation'][b, ti, 0]
                yaw = targets['rotation'][b, ti, 1]
                roll = targets['rotation'][b, ti, 2]
                
                # Apply yaw rotation to theta
                targets['theta'][b, ti, 0, 0] = torch.cos(yaw)
                targets['theta'][b, ti, 0, 2] = torch.sin(yaw)
                targets['theta'][b, ti, 2, 0] = -torch.sin(yaw)
                targets['theta'][b, ti, 2, 2] = torch.cos(yaw)
        
        # Audio features with energy variations
        audio_features = torch.randn(B, T, 768).cuda()
        audio_energy = 1 + torch.sin(t * 2) * 0.5
        audio_features *= audio_energy.unsqueeze(0).unsqueeze(-1)
        
        conditions = {
            'audio_features': audio_features,
            'gaze': torch.randn(B, T, 2).cuda() * 0.1,
            'head_distance': torch.ones(B, T, 1).cuda() * 0.5,
            'emotion': torch.randn(B, T, 2).cuda() * 0.2
        }
        
        # Add controlled noise
        noise_level = torch.ones(B, device='cuda') * (0.3 * (1 - step / num_steps))  # Decay noise
        noise = {key: torch.randn_like(val) * 0.05 for key, val in targets.items()}
        
        noisy_inputs = {}
        for key in targets.keys():
            if len(targets[key].shape) == 4:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1, 1)
            elif len(targets[key].shape) == 3:
                noisy_inputs[key] = targets[key] + noise[key] * noise_level.view(B, 1, 1)
        
        # Forward pass
        outputs = model.forward(
            motion_data=noisy_inputs,
            noise_level=noise_level,
            conditions=conditions
        )
        
        # Update curriculum stage
        progress = step / num_steps
        if progress < 0.3:
            tdd_loss.stage = 'motion_first'
        elif progress < 0.7:
            tdd_loss.stage = 'balanced'
        else:
            tdd_loss.stage = 'quality_focus'
        tdd_loss.weights = tdd_loss.stage_weights[tdd_loss.stage]
        
        # Compute TDD losses
        losses, test_info = tdd_loss.compute_losses(
            outputs=outputs,
            targets=targets,
            conditions=conditions,
            step=step
        )
        
        # Backward pass
        losses['total'].backward()
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        wandb_logger.log_tdd_step(
            step=step,
            losses=losses,
            test_info=test_info,
            learning_rate=current_lr,
            grad_norm=grad_norm.item()
        )
        
        # Track best model
        test_score = test_info['passed_ratio']
        if test_score > best_test_score:
            best_test_score = test_score
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_score': best_test_score,
                'step': step,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Log checkpoint to wandb
            wandb_logger.log_checkpoint(
                checkpoint_path=str(checkpoint_path),
                test_score=best_test_score,
                step=step,
                is_best=True
            )
            
            print(f"Step {step:4d}: Loss={losses['total'].item():7.2f}, "
                  f"Tests={test_score:5.1%} ⬆️ NEW BEST")
        
        # Regular logging
        elif step % 50 == 0:
            print(f"Step {step:4d}: Loss={losses['total'].item():7.2f}, "
                  f"Tests={test_score:5.1%}, LR={current_lr:.6f}")
            
            # Show test breakdown
            if step % 200 == 0:
                print("\n  Test Results:")
                for name, passed in test_info.get('test_results', {}).items():
                    status = "✅" if passed else "❌"
                    metric_val = test_info.get('metrics', {}).get(
                        name.replace('not_static', 'motion_magnitude'), 'N/A'
                    )
                    print(f"    {status} {name}: {metric_val}")
                print()
        
        # Save periodic checkpoints
        if step % 250 == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pth'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_score': test_score,
                'step': step,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            
            wandb_logger.log_checkpoint(
                checkpoint_path=str(checkpoint_path),
                test_score=test_score,
                step=step,
                is_best=False
            )
        
        # Early stopping if all tests pass
        if test_score >= 0.95:
            print(f"\n✅ 95% tests passing at step {step}! Training successful.")
            break
    
    # Final summary
    print("\n" + "="*60)
    print("TDD Training Complete!")
    print(f"Best test score: {best_test_score:.1%}")
    print(f"Final step: {step}")
    print("="*60)
    
    # Finalize wandb
    wandb_logger.finish()
    
    print("\n📊 Check WandB dashboard for detailed metrics and visualizations")
    print("   Run: wandb ui")


if __name__ == "__main__":
    train_with_wandb()