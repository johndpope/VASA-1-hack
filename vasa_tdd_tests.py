"""
VASA Model Test-Driven Development (TDD) Test Suite
=====================================================
This module implements comprehensive tests for VASA model quality assurance.
Tests are designed to run during training to ensure model convergence and quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import cv2
from scipy import signal
from torchvision import transforms
import logging
from abc import ABC, abstractmethod
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for test results with detailed metrics."""
    name: str
    passed: bool
    score: float
    target: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'target': self.target,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


class BaseQualityTest(ABC):
    """Abstract base class for all quality tests."""
    
    def __init__(self, name: str, target_score: float):
        self.name = name
        self.target_score = target_score
        self.history: List[TestResult] = []
        
    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 **kwargs) -> TestResult:
        """Evaluate the quality metric."""
        pass
    
    def log_result(self, result: TestResult):
        """Log test result to history."""
        self.history.append(result)
        if result.passed:
            logger.info(f"✅ {result.name}: PASSED (score: {result.score:.4f} >= {result.target:.4f})")
        else:
            logger.warning(f"❌ {result.name}: FAILED (score: {result.score:.4f} < {result.target:.4f})")
            logger.warning(f"   {result.message}")


# ============================================================================
# IMAGE FIDELITY TESTS
# ============================================================================

class PSNRTest(BaseQualityTest):
    """Peak Signal-to-Noise Ratio test for reconstruction quality."""
    
    def __init__(self, target_psnr: float = 28.0):
        super().__init__("PSNR Test", target_psnr)
        
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 **kwargs) -> TestResult:
        """Calculate PSNR between predictions and targets."""
        mse = F.mse_loss(predictions, targets)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        passed = psnr.item() >= self.target_score
        message = f"PSNR: {psnr.item():.2f}dB"
        if not passed:
            message += f" - Below target of {self.target_score}dB"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=psnr.item(),
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


class SSIMTest(BaseQualityTest):
    """Structural Similarity Index test for perceptual quality."""
    
    def __init__(self, target_ssim: float = 0.85):
        super().__init__("SSIM Test", target_ssim)
        
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor,
                 **kwargs) -> TestResult:
        """Calculate SSIM between predictions and targets."""
        ssim_val = self._compute_ssim(predictions, targets)
        
        passed = ssim_val >= self.target_score
        message = f"SSIM: {ssim_val:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=ssim_val,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()


class LPIPSTest(BaseQualityTest):
    """Learned Perceptual Image Patch Similarity test."""
    
    def __init__(self, target_lpips: float = 0.15):
        super().__init__("LPIPS Test", target_lpips)
        self.lpips_fn = None  # Would be initialized with actual LPIPS model
        
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor,
                 **kwargs) -> TestResult:
        """Calculate LPIPS between predictions and targets."""
        # Simplified LPIPS calculation (would use actual LPIPS model)
        lpips_val = F.l1_loss(predictions, targets).item() * 0.5  # Placeholder
        
        passed = lpips_val <= self.target_score
        message = f"LPIPS: {lpips_val:.4f}"
        if not passed:
            message += f" - Above target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=lpips_val,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


# ============================================================================
# MOTION QUALITY TESTS
# ============================================================================

class OpticalFlowConsistencyTest(BaseQualityTest):
    """Test for temporal consistency using optical flow."""
    
    def __init__(self, target_consistency: float = 0.9):
        super().__init__("Optical Flow Consistency", target_consistency)
        
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate optical flow consistency between consecutive frames."""
        if predictions.dim() == 5:  # [B, T, C, H, W]
            flow_errors = []
            for b in range(predictions.size(0)):
                for t in range(predictions.size(1) - 1):
                    flow_error = self._compute_flow_error(
                        predictions[b, t], predictions[b, t+1],
                        targets[b, t], targets[b, t+1]
                    )
                    flow_errors.append(flow_error)
            consistency = 1.0 - np.mean(flow_errors)
        else:
            consistency = 1.0
            
        passed = consistency >= self.target_score
        message = f"Flow consistency: {consistency:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=consistency,
            target=self.target_score,
            message=message,
            timestamp=datetime.now(),
            metadata={'flow_errors': flow_errors if predictions.dim() == 5 else []}
        )
        
        self.log_result(result)
        return result
    
    def _compute_flow_error(self, frame1: torch.Tensor, frame2: torch.Tensor,
                           target1: torch.Tensor, target2: torch.Tensor) -> float:
        """Compute optical flow error between predicted and target sequences."""
        # Simplified optical flow calculation
        pred_diff = (frame2 - frame1).abs().mean().item()
        target_diff = (target2 - target1).abs().mean().item()
        return abs(pred_diff - target_diff) / (target_diff + 1e-8)


class TemporalCoherenceTest(BaseQualityTest):
    """Test for temporal coherence in video sequences."""
    
    def __init__(self, target_coherence: float = 0.8):
        super().__init__("Temporal Coherence", target_coherence)
        
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate temporal coherence of predictions."""
        if predictions.dim() == 5:  # [B, T, C, H, W]
            coherence_scores = []
            for b in range(predictions.size(0)):
                # Compute frame-to-frame differences
                pred_diffs = []
                for t in range(predictions.size(1) - 1):
                    diff = F.l1_loss(predictions[b, t], predictions[b, t+1])
                    pred_diffs.append(diff.item())
                
                # Coherence is inverse of variance in differences
                if len(pred_diffs) > 1:
                    coherence = 1.0 / (1.0 + np.std(pred_diffs))
                else:
                    coherence = 1.0
                coherence_scores.append(coherence)
            
            avg_coherence = np.mean(coherence_scores)
        else:
            avg_coherence = 1.0
            
        passed = avg_coherence >= self.target_score
        message = f"Temporal coherence: {avg_coherence:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=avg_coherence,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


class MotionSmoothnessTest(BaseQualityTest):
    """Test for smooth motion transitions."""
    
    def __init__(self, target_smoothness: float = 0.85):
        super().__init__("Motion Smoothness", target_smoothness)
        
    def evaluate(self, motion_params: Dict[str, torch.Tensor], 
                 **kwargs) -> TestResult:
        """Evaluate smoothness of motion parameters."""
        smoothness_scores = []
        
        for param_name, param_values in motion_params.items():
            if param_values.dim() >= 2:  # Has temporal dimension
                # Compute second-order differences (acceleration)
                if param_values.size(1) > 2:
                    first_diff = param_values[:, 1:] - param_values[:, :-1]
                    second_diff = first_diff[:, 1:] - first_diff[:, :-1]
                    
                    # Smoothness is inverse of acceleration magnitude
                    accel_mag = second_diff.norm(dim=-1).mean().item()
                    smoothness = 1.0 / (1.0 + accel_mag)
                    smoothness_scores.append(smoothness)
        
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 1.0
        
        passed = avg_smoothness >= self.target_score
        message = f"Motion smoothness: {avg_smoothness:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=avg_smoothness,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


# ============================================================================
# SYNCHRONIZATION TESTS
# ============================================================================

class LipSyncTest(BaseQualityTest):
    """Test for audio-visual lip synchronization."""
    
    def __init__(self, target_sync_error: float = 40.0):  # milliseconds
        super().__init__("Lip Sync Test", target_sync_error)
        
    def evaluate(self, lip_features: torch.Tensor, audio_features: torch.Tensor,
                 fps: float = 25.0, **kwargs) -> TestResult:
        """Evaluate lip sync accuracy."""
        # Cross-correlation to find optimal alignment
        if lip_features.dim() >= 2 and audio_features.dim() >= 2:
            # Flatten features for correlation
            lip_flat = lip_features.mean(dim=-1).cpu().numpy()
            audio_flat = audio_features.mean(dim=-1).cpu().numpy()
            
            # Compute cross-correlation
            correlation = signal.correlate(lip_flat, audio_flat, mode='same')
            lag = np.argmax(np.abs(correlation)) - len(correlation) // 2
            
            # Convert lag to milliseconds
            sync_error = abs(lag) * (1000.0 / fps)
        else:
            sync_error = 0.0
            
        passed = sync_error <= self.target_score
        message = f"Sync error: {sync_error:.1f}ms"
        if not passed:
            message += f" - Above target of {self.target_score}ms"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=sync_error,
            target=self.target_score,
            message=message,
            timestamp=datetime.now(),
            metadata={'lag_frames': lag if lip_features.dim() >= 2 else 0}
        )
        
        self.log_result(result)
        return result


class AudioVisualCorrelationTest(BaseQualityTest):
    """Test for overall audio-visual correlation."""
    
    def __init__(self, target_correlation: float = 0.7):
        super().__init__("Audio-Visual Correlation", target_correlation)
        
    def evaluate(self, visual_features: torch.Tensor, audio_features: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate correlation between audio and visual features."""
        if visual_features.numel() > 0 and audio_features.numel() > 0:
            # Normalize features
            visual_norm = F.normalize(visual_features.flatten(), dim=0)
            audio_norm = F.normalize(audio_features.flatten()[:visual_norm.size(0)], dim=0)
            
            # Compute correlation
            correlation = torch.dot(visual_norm, audio_norm).item()
        else:
            correlation = 0.0
            
        passed = correlation >= self.target_score
        message = f"A/V correlation: {correlation:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=correlation,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


# ============================================================================
# CONTROL SIGNAL TESTS
# ============================================================================

class GazeAccuracyTest(BaseQualityTest):
    """Test for gaze direction accuracy."""
    
    def __init__(self, target_angular_error: float = 5.0):  # degrees
        super().__init__("Gaze Accuracy", target_angular_error)
        
    def evaluate(self, predicted_gaze: torch.Tensor, target_gaze: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate gaze direction accuracy."""
        if predicted_gaze.numel() > 0 and target_gaze.numel() > 0:
            # Compute angular error
            cos_sim = F.cosine_similarity(predicted_gaze, target_gaze, dim=-1)
            angular_error = torch.acos(cos_sim.clamp(-1, 1)) * 180 / np.pi
            avg_error = angular_error.mean().item()
        else:
            avg_error = 0.0
            
        passed = avg_error <= self.target_score
        message = f"Gaze error: {avg_error:.2f}°"
        if not passed:
            message += f" - Above target of {self.target_score}°"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=avg_error,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


class HeadPoseTest(BaseQualityTest):
    """Test for head pose accuracy."""
    
    def __init__(self, target_angular_error: float = 5.0):  # degrees
        super().__init__("Head Pose Accuracy", target_angular_error)
        
    def evaluate(self, predicted_pose: torch.Tensor, target_pose: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate head pose accuracy (rotation matrix or euler angles)."""
        if predicted_pose.numel() > 0 and target_pose.numel() > 0:
            if predicted_pose.size(-1) == 3:  # Euler angles
                angular_error = (predicted_pose - target_pose).abs().mean().item()
            else:  # Rotation matrix
                # Compute geodesic distance between rotations
                trace = (predicted_pose @ target_pose.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                angular_error = torch.acos((trace - 1) / 2).mean().item() * 180 / np.pi
        else:
            angular_error = 0.0
            
        passed = angular_error <= self.target_score
        message = f"Pose error: {angular_error:.2f}°"
        if not passed:
            message += f" - Above target of {self.target_score}°"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=angular_error,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


class ExpressionTransferTest(BaseQualityTest):
    """Test for facial expression transfer accuracy."""
    
    def __init__(self, target_accuracy: float = 0.8):
        super().__init__("Expression Transfer", target_accuracy)
        
    def evaluate(self, predicted_expression: torch.Tensor, target_expression: torch.Tensor,
                 **kwargs) -> TestResult:
        """Evaluate expression transfer accuracy."""
        if predicted_expression.numel() > 0 and target_expression.numel() > 0:
            # For continuous expression embeddings
            if predicted_expression.dim() >= 2:
                similarity = F.cosine_similarity(predicted_expression, target_expression, dim=-1)
                accuracy = similarity.mean().item()
            # For discrete expression classes
            else:
                accuracy = (predicted_expression == target_expression).float().mean().item()
        else:
            accuracy = 0.0
            
        passed = accuracy >= self.target_score
        message = f"Expression accuracy: {accuracy:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=accuracy,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


# ============================================================================
# PROGRESSIVE TRAINING TESTS
# ============================================================================

class StaticReconstructionTest(BaseQualityTest):
    """Test for single frame reconstruction (t=0, no noise)."""
    
    def __init__(self, target_loss: float = 0.1):
        super().__init__("Static Reconstruction", target_loss)
        
    def evaluate(self, reconstruction_loss: float, **kwargs) -> TestResult:
        """Evaluate static reconstruction quality."""
        passed = reconstruction_loss <= self.target_score
        message = f"Reconstruction loss: {reconstruction_loss:.4f}"
        if not passed:
            message += f" - Above target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=reconstruction_loss,
            target=self.target_score,
            message=message,
            timestamp=datetime.now()
        )
        
        self.log_result(result)
        return result


class ShortSequenceTest(BaseQualityTest):
    """Test for short sequence generation (3-5 frames)."""
    
    def __init__(self, target_dynamics_loss: float = 0.05):
        super().__init__("Short Sequence Generation", target_dynamics_loss)
        
    def evaluate(self, dynamics_loss: float, window_size: int, **kwargs) -> TestResult:
        """Evaluate short sequence generation quality."""
        if window_size > 5:
            # Not applicable for long sequences
            return TestResult(
                name=self.name,
                passed=True,
                score=0.0,
                target=self.target_score,
                message="Test skipped for long sequences",
                timestamp=datetime.now()
            )
            
        passed = dynamics_loss <= self.target_score
        message = f"Dynamics loss: {dynamics_loss:.4f} (window={window_size})"
        if not passed:
            message += f" - Above target of {self.target_score}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=dynamics_loss,
            target=self.target_score,
            message=message,
            timestamp=datetime.now(),
            metadata={'window_size': window_size}
        )
        
        self.log_result(result)
        return result


class ControlResponseTest(BaseQualityTest):
    """Test for control signal responsiveness."""
    
    def __init__(self, target_response: float = 0.7):
        super().__init__("Control Response", target_response)
        
    def evaluate(self, control_losses: Dict[str, float], **kwargs) -> TestResult:
        """Evaluate control signal responsiveness."""
        if control_losses:
            # Average normalized control response (inverse of loss)
            responses = [1.0 / (1.0 + loss) for loss in control_losses.values()]
            avg_response = np.mean(responses)
        else:
            avg_response = 0.0
            
        passed = avg_response >= self.target_score
        message = f"Control response: {avg_response:.4f}"
        if not passed:
            message += f" - Below target of {self.target_score}"
            message += f"\nPer-control: {control_losses}"
            
        result = TestResult(
            name=self.name,
            passed=passed,
            score=avg_response,
            target=self.target_score,
            message=message,
            timestamp=datetime.now(),
            metadata={'control_losses': control_losses}
        )
        
        self.log_result(result)
        return result


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class RegressionTestSuite:
    """Suite of regression tests for golden examples."""
    
    def __init__(self, golden_examples_path: Path):
        self.golden_examples_path = golden_examples_path
        self.golden_examples = self._load_golden_examples()
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        
    def _load_golden_examples(self) -> Dict[str, torch.Tensor]:
        """Load pre-computed golden examples."""
        examples = {}
        if self.golden_examples_path.exists():
            for example_file in self.golden_examples_path.glob("*.pt"):
                name = example_file.stem
                examples[name] = torch.load(example_file)
        return examples
    
    def test_canonical_poses(self, model: nn.Module) -> List[TestResult]:
        """Test model on canonical poses."""
        results = []
        canonical_poses = ['frontal', 'profile_left', 'profile_right', 'looking_up', 'looking_down']
        
        for pose_name in canonical_poses:
            if pose_name in self.golden_examples:
                golden = self.golden_examples[pose_name]
                with torch.no_grad():
                    prediction = model(golden['input'])
                
                # Compare with expected output
                error = F.l1_loss(prediction, golden['expected_output']).item()
                passed = error < golden.get('tolerance', 0.1)
                
                result = TestResult(
                    name=f"Canonical Pose: {pose_name}",
                    passed=passed,
                    score=error,
                    target=golden.get('tolerance', 0.1),
                    message=f"L1 error: {error:.4f}",
                    timestamp=datetime.now()
                )
                results.append(result)
                self.test_results[pose_name].append(result)
        
        return results
    
    def test_basic_expressions(self, model: nn.Module) -> List[TestResult]:
        """Test model on basic expressions."""
        results = []
        basic_expressions = ['neutral', 'smile', 'frown', 'surprise', 'anger']
        
        for expr_name in basic_expressions:
            if expr_name in self.golden_examples:
                golden = self.golden_examples[expr_name]
                with torch.no_grad():
                    prediction = model(golden['input'])
                
                # Check expression preservation
                if 'expression_classifier' in golden:
                    classifier = golden['expression_classifier']
                    pred_expr = classifier(prediction)
                    target_expr = golden['expected_expression']
                    
                    accuracy = (pred_expr.argmax(dim=-1) == target_expr).float().mean().item()
                    passed = accuracy >= 0.8
                    
                    result = TestResult(
                        name=f"Expression: {expr_name}",
                        passed=passed,
                        score=accuracy,
                        target=0.8,
                        message=f"Expression accuracy: {accuracy:.4f}",
                        timestamp=datetime.now()
                    )
                    results.append(result)
                    self.test_results[expr_name].append(result)
        
        return results
    
    def check_regression(self) -> bool:
        """Check if any regression has occurred."""
        has_regression = False
        
        for test_name, results in self.test_results.items():
            if len(results) >= 2:
                current = results[-1]
                previous = results[-2]
                
                if current.score > previous.score * 1.1:  # 10% degradation threshold
                    logger.warning(f"⚠️ Regression detected in {test_name}: "
                                 f"{previous.score:.4f} -> {current.score:.4f}")
                    has_regression = True
        
        return has_regression


# ============================================================================
# TEST RUNNER
# ============================================================================

class VASATestRunner:
    """Main test runner for VASA model testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.test_suites = self._initialize_test_suites()
        self.results_history = []
        self.output_dir = Path(config.get('test_output_dir', 'test_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_test_suites(self) -> Dict[str, List[BaseQualityTest]]:
        """Initialize all test suites based on configuration."""
        suites = {
            'image_fidelity': [
                PSNRTest(self.config.get('targets', {}).get('psnr', 28.0)),
                SSIMTest(self.config.get('targets', {}).get('ssim', 0.85)),
                LPIPSTest(self.config.get('targets', {}).get('lpips', 0.15))
            ],
            'motion_quality': [
                OpticalFlowConsistencyTest(self.config.get('targets', {}).get('flow_consistency', 0.9)),
                TemporalCoherenceTest(self.config.get('targets', {}).get('temporal_coherence', 0.8)),
                MotionSmoothnessTest(self.config.get('targets', {}).get('motion_smoothness', 0.85))
            ],
            'synchronization': [
                LipSyncTest(self.config.get('targets', {}).get('lip_sync_error', 40.0)),
                AudioVisualCorrelationTest(self.config.get('targets', {}).get('av_correlation', 0.7))
            ],
            'control_signals': [
                GazeAccuracyTest(self.config.get('targets', {}).get('gaze_error', 5.0)),
                HeadPoseTest(self.config.get('targets', {}).get('pose_error', 5.0)),
                ExpressionTransferTest(self.config.get('targets', {}).get('expression_accuracy', 0.8))
            ],
            'progressive_training': [
                StaticReconstructionTest(self.config.get('targets', {}).get('static_loss', 0.1)),
                ShortSequenceTest(self.config.get('targets', {}).get('dynamics_loss', 0.05)),
                ControlResponseTest(self.config.get('targets', {}).get('control_response', 0.7))
            ]
        }
        
        return suites
    
    def run_epoch_tests(self, epoch: int, model: nn.Module, 
                       data: Dict[str, Any]) -> Dict[str, TestResult]:
        """Run tests appropriate for current epoch."""
        results = {}
        
        # Determine which test suites to run based on epoch
        if epoch < 10:
            # Early training - focus on reconstruction
            suites_to_run = ['progressive_training']
        elif epoch < 20:
            # Add motion tests
            suites_to_run = ['progressive_training', 'motion_quality']
        elif epoch < 30:
            # Add control tests
            suites_to_run = ['progressive_training', 'motion_quality', 'control_signals']
        else:
            # Full test suite
            suites_to_run = list(self.test_suites.keys())
        
        # Run selected test suites
        for suite_name in suites_to_run:
            if suite_name in self.test_suites:
                suite_results = self.run_test_suite(suite_name, model, data)
                results.update(suite_results)
        
        # Save results
        self._save_results(epoch, results)
        
        return results
    
    def run_test_suite(self, suite_name: str, model: nn.Module,
                      data: Dict[str, Any]) -> Dict[str, TestResult]:
        """Run a specific test suite."""
        results = {}
        suite = self.test_suites.get(suite_name, [])
        
        logger.info(f"\nRunning {suite_name} tests...")
        
        for test in suite:
            try:
                # Extract relevant data for each test
                test_data = self._extract_test_data(test, data)
                
                # Run test
                result = test.evaluate(**test_data)
                results[test.name] = result
                
            except Exception as e:
                logger.error(f"Error running {test.name}: {str(e)}")
                results[test.name] = TestResult(
                    name=test.name,
                    passed=False,
                    score=0.0,
                    target=test.target_score,
                    message=f"Test failed with error: {str(e)}",
                    timestamp=datetime.now()
                )
        
        return results
    
    def _extract_test_data(self, test: BaseQualityTest, data: Dict[str, Any]) -> Dict:
        """Extract relevant data for specific test."""
        test_data = {}
        
        # Map test types to required data
        if isinstance(test, (PSNRTest, SSIMTest, LPIPSTest)):
            test_data['predictions'] = data.get('predictions')
            test_data['targets'] = data.get('targets')
            
        elif isinstance(test, (OpticalFlowConsistencyTest, TemporalCoherenceTest)):
            test_data['predictions'] = data.get('predictions')
            test_data['targets'] = data.get('targets')
            
        elif isinstance(test, MotionSmoothnessTest):
            test_data['motion_params'] = data.get('motion_params', {})
            
        elif isinstance(test, LipSyncTest):
            test_data['lip_features'] = data.get('lip_features')
            test_data['audio_features'] = data.get('audio_features')
            
        elif isinstance(test, AudioVisualCorrelationTest):
            test_data['visual_features'] = data.get('visual_features')
            test_data['audio_features'] = data.get('audio_features')
            
        elif isinstance(test, GazeAccuracyTest):
            test_data['predicted_gaze'] = data.get('predicted_gaze')
            test_data['target_gaze'] = data.get('target_gaze')
            
        elif isinstance(test, HeadPoseTest):
            test_data['predicted_pose'] = data.get('predicted_pose')
            test_data['target_pose'] = data.get('target_pose')
            
        elif isinstance(test, ExpressionTransferTest):
            test_data['predicted_expression'] = data.get('predicted_expression')
            test_data['target_expression'] = data.get('target_expression')
            
        elif isinstance(test, StaticReconstructionTest):
            test_data['reconstruction_loss'] = data.get('reconstruction_loss', 1.0)
            
        elif isinstance(test, ShortSequenceTest):
            test_data['dynamics_loss'] = data.get('dynamics_loss', 1.0)
            test_data['window_size'] = data.get('window_size', 5)
            
        elif isinstance(test, ControlResponseTest):
            test_data['control_losses'] = data.get('control_losses', {})
        
        return test_data
    
    def _save_results(self, epoch: int, results: Dict[str, TestResult]):
        """Save test results to file."""
        timestamp = datetime.now().isoformat()
        
        # Create results summary
        summary = {
            'epoch': epoch,
            'timestamp': timestamp,
            'results': {name: result.to_dict() for name, result in results.items()},
            'passed_count': sum(1 for r in results.values() if r.passed),
            'total_count': len(results)
        }
        
        # Save to JSON
        results_file = self.output_dir / f'epoch_{epoch:04d}_results.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Append to history
        self.results_history.append(summary)
        
        # Generate report
        self.generate_report(epoch, results)
    
    def generate_report(self, epoch: int, results: Dict[str, TestResult]):
        """Generate test report for epoch."""
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        
        report = [
            f"\n{'='*60}",
            f"TEST REPORT - Epoch {epoch}",
            f"{'='*60}",
            f"Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)\n"
        ]
        
        # Group results by suite
        for suite_name, suite_tests in self.test_suites.items():
            suite_results = [results.get(test.name) for test in suite_tests 
                            if test.name in results]
            
            if suite_results:
                suite_passed = sum(1 for r in suite_results if r and r.passed)
                suite_total = len(suite_results)
                
                report.append(f"\n{suite_name.upper()}:")
                report.append(f"  {suite_passed}/{suite_total} passed")
                
                for result in suite_results:
                    if result:
                        status = "✅" if result.passed else "❌"
                        report.append(f"  {status} {result.name}: {result.score:.4f} "
                                    f"(target: {result.target:.4f})")
        
        report.append(f"\n{'='*60}\n")
        
        # Print and save report
        report_text = '\n'.join(report)
        logger.info(report_text)
        
        report_file = self.output_dir / f'epoch_{epoch:04d}_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
    
    def plot_progress(self):
        """Plot test progress over epochs."""
        if not self.results_history:
            return
        
        # Extract metrics over time
        epochs = [r['epoch'] for r in self.results_history]
        pass_rates = [100 * r['passed_count'] / r['total_count'] 
                     for r in self.results_history]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, pass_rates, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Test Pass Rate (%)')
        plt.title('VASA Model Test Progress')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        
        # Save plot
        plt.savefig(self.output_dir / 'test_progress.png', dpi=100, bbox_inches='tight')
        plt.close()


# ============================================================================
# INTEGRATION WITH TRAINING
# ============================================================================

def integrate_with_training_loop(trainer, test_runner: VASATestRunner):
    """
    Integration helper to add TDD testing to existing training loop.
    
    Usage in training script:
        test_runner = VASATestRunner(test_config)
        integrate_with_training_loop(trainer, test_runner)
    """
    
    # Store original train_epoch method
    original_train_epoch = trainer.train_epoch
    
    def train_epoch_with_tests(self):
        """Modified train epoch with integrated testing."""
        # Run original training
        stats = original_train_epoch()
        
        # Prepare test data from training results
        test_data = {
            'predictions': getattr(self, 'last_predictions', None),
            'targets': getattr(self, 'last_targets', None),
            'motion_params': getattr(self, 'last_motion_params', {}),
            'reconstruction_loss': stats.get('reconstruction', 1.0),
            'dynamics_loss': stats.get('dynamics_loss', 1.0),
            'control_losses': {k: v for k, v in stats.items() if k.startswith('control_')},
            'window_size': self.training_state.get_window_size(),
            # Add more data as needed
        }
        
        # Run tests
        test_results = test_runner.run_epoch_tests(
            epoch=self.current_epoch,
            model=self.model,
            data=test_data
        )
        
        # Check for critical failures
        critical_tests = ['Static Reconstruction', 'Temporal Coherence']
        for test_name in critical_tests:
            if test_name in test_results and not test_results[test_name].passed:
                logger.warning(f"⚠️ Critical test '{test_name}' failed!")
                
                # Optionally adjust training
                if test_name == 'Static Reconstruction' and self.current_epoch > 10:
                    logger.info("Reducing learning rate due to reconstruction issues")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
        
        return stats
    
    # Replace method
    trainer.train_epoch = train_epoch_with_tests.__get__(trainer, trainer.__class__)
    
    logger.info("TDD testing integrated with training loop")


if __name__ == "__main__":
    # Example usage
    test_config = {
        'test_output_dir': 'test_results',
        'targets': {
            'psnr': 28.0,
            'ssim': 0.85,
            'lpips': 0.15,
            'flow_consistency': 0.9,
            'temporal_coherence': 0.8,
            'motion_smoothness': 0.85,
            'lip_sync_error': 40.0,
            'av_correlation': 0.7,
            'gaze_error': 5.0,
            'pose_error': 5.0,
            'expression_accuracy': 0.8,
            'static_loss': 0.1,
            'dynamics_loss': 0.05,
            'control_response': 0.7
        }
    }
    
    # Initialize test runner
    test_runner = VASATestRunner(test_config)
    
    # Example test data
    test_data = {
        'predictions': torch.randn(2, 10, 3, 256, 256),  # [B, T, C, H, W]
        'targets': torch.randn(2, 10, 3, 256, 256),
        'motion_params': {
            'theta': torch.randn(2, 10, 3, 4),
            'rotation': torch.randn(2, 10, 3),
            'translation': torch.randn(2, 10, 3)
        },
        'reconstruction_loss': 0.08,
        'dynamics_loss': 0.04,
        'control_losses': {
            'gaze': 0.02,
            'emotion': 0.03,
            'head_distance': 0.01
        },
        'window_size': 5
    }
    
    # Run tests for epoch 15
    results = test_runner.run_epoch_tests(epoch=15, model=None, data=test_data)
    
    # Plot progress
    test_runner.plot_progress()
    
    logger.info("TDD test suite example completed")