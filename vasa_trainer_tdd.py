"""
Enhanced VASA Trainer with TDD Integration
==========================================
This module extends the existing trainer with comprehensive TDD testing.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging
from vasa_trainer import VASATrainer
from vasa_tdd_tests import VASATestRunner, TestResult
import wandb

logger = logging.getLogger(__name__)


class VASATrainerWithTDD(VASATrainer):
    """Enhanced VASA trainer with integrated TDD testing."""
    
    def __init__(self, *args, **kwargs):
        # Extract TDD config if provided
        self.tdd_config = kwargs.pop('tdd_config', None)
        
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        
        # Initialize test runner if TDD enabled
        if self.tdd_config and self.tdd_config.get('enabled', False):
            self.test_runner = VASATestRunner(self.tdd_config)
            self.test_failures = []
            self.critical_test_failures = 0
            logger.info("TDD Testing enabled for training")
        else:
            self.test_runner = None
            logger.info("TDD Testing disabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """Training epoch with integrated TDD testing."""
        # Run base training epoch
        epoch_stats = super().train_epoch()
        
        # Run TDD tests if enabled
        if self.test_runner:
            test_results = self._run_tdd_tests(epoch_stats)
            self._handle_test_results(test_results)
            
            # Add test metrics to epoch stats
            epoch_stats.update(self._extract_test_metrics(test_results))
        
        return epoch_stats
    
    def _run_tdd_tests(self, epoch_stats: Dict[str, float]) -> Dict[str, TestResult]:
        """Run TDD tests based on current training state."""
        # Prepare test data from training statistics
        test_data = self._prepare_test_data(epoch_stats)
        
        # Run tests appropriate for current epoch
        test_results = self.test_runner.run_epoch_tests(
            epoch=self.current_epoch,
            model=self.model,
            data=test_data
        )
        
        return test_results
    
    def _prepare_test_data(self, epoch_stats: Dict[str, float]) -> Dict:
        """Prepare test data from training statistics and model state."""
        test_data = {
            # Loss values for progressive tests
            'reconstruction_loss': epoch_stats.get('reconstruction', 1.0),
            'dynamics_loss': epoch_stats.get('dynamics_loss', 1.0),
            'control_losses': {
                'gaze': epoch_stats.get('control_gaze', 0.0),
                'emotion': epoch_stats.get('control_emotion', 0.0),
                'head_distance': epoch_stats.get('control_distance', 0.0),
                'speed': epoch_stats.get('control_speed', 0.0)
            },
            
            # Window and training parameters
            'window_size': self.training_state.get_window_size(),
            'current_epoch': self.current_epoch,
            
            # Get last batch data if available
            'predictions': getattr(self, '_last_predictions', None),
            'targets': getattr(self, '_last_targets', None),
            'motion_params': getattr(self, '_last_motion_params', {}),
            
            # Feature data for specific tests
            'visual_features': getattr(self, '_last_visual_features', None),
            'audio_features': getattr(self, '_last_audio_features', None),
            'lip_features': getattr(self, '_last_lip_features', None),
            
            # Control signal predictions
            'predicted_gaze': getattr(self, '_last_predicted_gaze', None),
            'target_gaze': getattr(self, '_last_target_gaze', None),
            'predicted_pose': getattr(self, '_last_predicted_pose', None),
            'target_pose': getattr(self, '_last_target_pose', None),
            'predicted_expression': getattr(self, '_last_predicted_expression', None),
            'target_expression': getattr(self, '_last_target_expression', None)
        }
        
        return test_data
    
    def _handle_test_results(self, test_results: Dict[str, TestResult]):
        """Handle test results and take appropriate actions."""
        # Count failures
        failures = [r for r in test_results.values() if not r.passed]
        self.test_failures = failures
        
        # Check critical tests based on epoch
        critical_tests = self._get_critical_tests()
        critical_failures = [
            r for r in failures 
            if r.name in critical_tests
        ]
        
        if critical_failures:
            self.critical_test_failures += 1
            logger.warning(f"⚠️ {len(critical_failures)} critical test(s) failed!")
            
            # Take corrective actions
            self._handle_critical_failures(critical_failures)
        else:
            # Reset counter if all critical tests pass
            self.critical_test_failures = 0
        
        # Log test summary
        passed = len(test_results) - len(failures)
        total = len(test_results)
        logger.info(f"Test Summary: {passed}/{total} tests passed")
        
        # Log to wandb if enabled
        if self.config.wandb.enabled:
            self._log_test_results_to_wandb(test_results)
    
    def _get_critical_tests(self) -> List[str]:
        """Get list of critical tests based on training stage."""
        if self.current_epoch < 10:
            return ['Static Reconstruction']
        elif self.current_epoch < 20:
            return ['Static Reconstruction', 'Short Sequence Generation']
        elif self.current_epoch < 30:
            return ['Static Reconstruction', 'Temporal Coherence', 'Control Response']
        else:
            return [
                'Static Reconstruction',
                'Temporal Coherence', 
                'Lip Sync Test',
                'Control Response'
            ]
    
    def _handle_critical_failures(self, critical_failures: List[TestResult]):
        """Take corrective actions for critical test failures."""
        for failure in critical_failures:
            logger.warning(f"Handling critical failure: {failure.name}")
            
            if failure.name == 'Static Reconstruction':
                # Reconstruction failing - reduce learning rate
                if self.critical_test_failures > 2:
                    logger.info("Reducing learning rate due to persistent reconstruction issues")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        
            elif failure.name == 'Temporal Coherence':
                # Motion issues - reduce window size temporarily
                if self.critical_test_failures > 3:
                    logger.info("Reducing window size due to temporal coherence issues")
                    self.motion_handler.window_size = max(3, self.motion_handler.window_size - 2)
                    
            elif failure.name == 'Lip Sync Test':
                # Sync issues - increase sync loss weight
                if hasattr(self.loss_module, 'loss_weights'):
                    logger.info("Increasing sync loss weight")
                    self.loss_module.loss_weights['sync'] *= 1.5
                    
            elif failure.name == 'Control Response':
                # Control issues - adjust dropout
                if self.critical_test_failures > 2:
                    logger.info("Reducing control dropout due to response issues")
                    for k in self.config.train.dropout_probs:
                        self.config.train.dropout_probs[k] *= 0.8
    
    def _extract_test_metrics(self, test_results: Dict[str, TestResult]) -> Dict[str, float]:
        """Extract metrics from test results for logging."""
        metrics = {}
        
        # Overall test pass rate
        passed = sum(1 for r in test_results.values() if r.passed)
        total = len(test_results)
        metrics['test/pass_rate'] = passed / total if total > 0 else 0.0
        
        # Individual test scores
        for name, result in test_results.items():
            clean_name = name.lower().replace(' ', '_').replace(':', '')
            metrics[f'test/{clean_name}_score'] = result.score
            metrics[f'test/{clean_name}_passed'] = float(result.passed)
        
        # Suite-level metrics
        suite_metrics = {
            'image_fidelity': ['PSNR Test', 'SSIM Test', 'LPIPS Test'],
            'motion_quality': ['Optical Flow Consistency', 'Temporal Coherence', 'Motion Smoothness'],
            'synchronization': ['Lip Sync Test', 'Audio-Visual Correlation'],
            'control_signals': ['Gaze Accuracy', 'Head Pose Accuracy', 'Expression Transfer']
        }
        
        for suite_name, test_names in suite_metrics.items():
            suite_results = [test_results.get(name) for name in test_names if name in test_results]
            if suite_results:
                suite_passed = sum(1 for r in suite_results if r.passed)
                suite_total = len(suite_results)
                metrics[f'test/{suite_name}_pass_rate'] = suite_passed / suite_total
        
        return metrics
    
    def _log_test_results_to_wandb(self, test_results: Dict[str, TestResult]):
        """Log test results to Weights & Biases."""
        if not self.accelerator.is_local_main_process:
            return
        
        # Create test summary table
        test_table = wandb.Table(
            columns=['Test Name', 'Passed', 'Score', 'Target', 'Message']
        )
        
        for name, result in test_results.items():
            test_table.add_data(
                name,
                '✅' if result.passed else '❌',
                f"{result.score:.4f}",
                f"{result.target:.4f}",
                result.message
            )
        
        wandb.log({
            'test/results_table': test_table,
            'test/epoch': self.current_epoch,
            'test/critical_failures': self.critical_test_failures
        }, step=self.global_step)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint with test results."""
        # Save base checkpoint
        super().save_checkpoint(is_best)
        
        # Save test history if available
        if self.test_runner:
            test_history_path = self.output_dir / 'test_history.json'
            import json
            
            history = {
                'epochs': [r['epoch'] for r in self.test_runner.results_history],
                'pass_rates': [r['passed_count'] / r['total_count'] 
                              for r in self.test_runner.results_history],
                'results': self.test_runner.results_history
            }
            
            with open(test_history_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logger.info(f"Saved test history to {test_history_path}")
    
    def should_early_stop(self) -> bool:
        """Determine if training should stop early based on test results."""
        if not self.test_runner:
            return False
        
        # Stop if critical tests fail consistently
        if self.critical_test_failures > self.tdd_config.get('max_critical_failures', 10):
            logger.error("Stopping training due to persistent critical test failures")
            return True
        
        # Stop if regression detected
        if hasattr(self.test_runner, 'regression_suite'):
            if self.test_runner.regression_suite.check_regression():
                regression_threshold = self.tdd_config.get('regression_tolerance', 5)
                if self.critical_test_failures > regression_threshold:
                    logger.error("Stopping training due to regression")
                    return True
        
        return False
    
    def train(self):
        """Main training loop with early stopping based on tests."""
        logger.info("Starting TDD-enhanced training")
        num_epochs = self.config.train.num_epochs
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_stats = self.train_epoch()
            
            # Validation phase
            val_stats = self.validate() if self.val_loader else None
            
            # Check early stopping
            if self.should_early_stop():
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint if best model
            if val_stats and val_stats.get('total', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_stats['total']
                self.save_checkpoint(is_best=True)
            
            # Regular checkpoint saving
            if epoch % self.config.train.save_freq == 0:
                self.save_checkpoint(is_best=False)
        
        # Generate final test report
        if self.test_runner:
            self.test_runner.plot_progress()
            logger.info("Training completed. Test report saved.")


# Helper function to create TDD-enhanced trainer
def create_tdd_trainer(model, config, train_loader, val_loader=None, output_dir=None):
    """
    Create a TDD-enhanced trainer instance.
    
    Args:
        model: VASA model
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        output_dir: Output directory for checkpoints
    
    Returns:
        VASATrainerWithTDD instance
    """
    # TDD configuration
    tdd_config = {
        'enabled': config.get('tdd', {}).get('enabled', True),
        'test_output_dir': str(Path(output_dir) / 'test_results'),
        'max_critical_failures': 10,
        'regression_tolerance': 5,
        'targets': {
            # Image fidelity targets
            'psnr': config.get('tdd', {}).get('targets', {}).get('psnr', 28.0),
            'ssim': config.get('tdd', {}).get('targets', {}).get('ssim', 0.85),
            'lpips': config.get('tdd', {}).get('targets', {}).get('lpips', 0.15),
            
            # Motion quality targets
            'flow_consistency': 0.9,
            'temporal_coherence': 0.8,
            'motion_smoothness': 0.85,
            
            # Synchronization targets
            'lip_sync_error': 40.0,  # ms
            'av_correlation': 0.7,
            
            # Control signal targets
            'gaze_error': 5.0,  # degrees
            'pose_error': 5.0,  # degrees
            'expression_accuracy': 0.8,
            
            # Progressive training targets
            'static_loss': 0.1,
            'dynamics_loss': 0.05,
            'control_response': 0.7
        }
    }
    
    # Create trainer with TDD
    trainer = VASATrainerWithTDD(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        tdd_config=tdd_config
    )
    
    return trainer


if __name__ == "__main__":
    logger.info("TDD-enhanced VASA trainer module loaded")