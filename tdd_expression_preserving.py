#!/usr/bin/env python3
"""
Expression-Preserving TDD Loss Module
======================================
Enforces strong expression preservation through reconstruction losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ExpressionPreservingTDDLoss(nn.Module):
    """TDD loss with strong expression preservation enforcement"""
    
    def __init__(
        self,
        expression_weight: float = 2.0,
        expression_codebook_weight: float = 1.5,
        expression_temporal_weight: float = 1.0,
        motion_weight: float = 0.8,
        lip_sync_weight: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        # Expression preservation weights (higher = stronger enforcement)
        self.weights = {
            'expression_reconstruction': expression_weight,
            'expression_codebook': expression_codebook_weight,
            'expression_temporal': expression_temporal_weight,
            'motion_quality': motion_weight,
            'lip_sync': lip_sync_weight,
        }
        
        # Expression codebook for discrete expression states
        self.expression_codebook = None
        self.codebook_size = 128
        
        # Test criteria for expression preservation
        self.expression_criteria = {
            'expression_consistency': {
                'target': 0.95,  # 95% similarity to original
                'threshold': 0.90,  # Must be > 90% similar
                'weight': 2.0
            },
            'expression_drift': {
                'target': 0.02,  # Max 2% drift
                'threshold': 0.05,  # Fail if > 5% drift
                'weight': 1.5
            },
            'codebook_alignment': {
                'target': 0.9,  # 90% codebook match
                'threshold': 0.8,  # Must be > 80% match
                'weight': 1.0
            }
        }
        
        logger.info(f"Initialized ExpressionPreservingTDDLoss with weights: {self.weights}")
    
    def build_expression_codebook(self, expressions: torch.Tensor):
        """Build a codebook of discrete expression states"""
        
        if self.expression_codebook is not None:
            return
        
        # Cluster expressions into discrete states
        from sklearn.cluster import KMeans
        
        # Flatten expressions for clustering
        B, T, D = expressions.shape
        expressions_flat = expressions.view(-1, D).cpu().numpy()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=42)
        kmeans.fit(expressions_flat)
        
        # Store codebook
        self.expression_codebook = torch.from_numpy(
            kmeans.cluster_centers_
        ).float().to(self.device)
        
        logger.info(f"Built expression codebook with {self.codebook_size} states")
    
    def expression_reconstruction_loss(
        self,
        pred_expression: torch.Tensor,
        target_expression: torch.Tensor
    ) -> torch.Tensor:
        """Strong L2 loss for expression reconstruction"""
        
        # Direct reconstruction loss
        recon_loss = F.mse_loss(pred_expression, target_expression)
        
        # Per-dimension loss to ensure all features are preserved
        dim_losses = F.mse_loss(pred_expression, target_expression, reduction='none')
        dim_loss = dim_losses.mean(dim=0).max()  # Penalize worst dimension
        
        # Cosine similarity loss for direction preservation
        cos_sim = F.cosine_similarity(
            pred_expression.view(-1, pred_expression.size(-1)),
            target_expression.view(-1, target_expression.size(-1)),
            dim=-1
        )
        cos_loss = 1.0 - cos_sim.mean()
        
        # Combined loss
        total_loss = recon_loss + 0.5 * dim_loss + 0.3 * cos_loss
        
        return total_loss
    
    def expression_codebook_loss(
        self,
        pred_expression: torch.Tensor,
        target_expression: torch.Tensor
    ) -> torch.Tensor:
        """Ensure expressions stay within learned codebook"""
        
        if self.expression_codebook is None:
            return torch.tensor(0.0, device=self.device)
        
        B, T, D = pred_expression.shape
        
        # Find nearest codebook entries for target
        target_flat = target_expression.view(-1, D)
        distances = torch.cdist(target_flat, self.expression_codebook)
        nearest_indices = distances.argmin(dim=1)
        nearest_codes = self.expression_codebook[nearest_indices]
        
        # Predicted should match nearest codes
        pred_flat = pred_expression.view(-1, D)
        codebook_loss = F.mse_loss(pred_flat, nearest_codes)
        
        # Commitment loss - predicted should commit to codebook entries
        pred_distances = torch.cdist(pred_flat, self.expression_codebook)
        commitment_loss = pred_distances.min(dim=1)[0].mean()
        
        total_loss = codebook_loss + 0.25 * commitment_loss
        
        return total_loss
    
    def expression_temporal_loss(
        self,
        pred_expression: torch.Tensor,
        target_expression: torch.Tensor
    ) -> torch.Tensor:
        """Preserve temporal dynamics of expressions"""
        
        # Temporal difference preservation
        pred_diff = torch.diff(pred_expression, dim=1)
        target_diff = torch.diff(target_expression, dim=1)
        
        temporal_loss = F.mse_loss(pred_diff, target_diff)
        
        # Velocity matching
        if pred_expression.shape[1] > 2:
            pred_vel = torch.diff(pred_diff, dim=1)
            target_vel = torch.diff(target_diff, dim=1)
            velocity_loss = F.mse_loss(pred_vel, target_vel)
        else:
            velocity_loss = 0.0
        
        # Smoothness constraint
        smoothness = torch.mean(torch.abs(pred_diff))
        target_smoothness = torch.mean(torch.abs(target_diff))
        smooth_loss = F.mse_loss(smoothness, target_smoothness)
        
        total_loss = temporal_loss + 0.3 * velocity_loss + 0.2 * smooth_loss
        
        return total_loss
    
    def test_expression_preservation(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Test if expression preservation criteria are met"""
        
        test_results = {}
        metrics = {}
        
        if 'expression_embed' not in outputs or 'expression_embed' not in targets:
            return {}, {}
        
        pred_expr = outputs['expression_embed']
        target_expr = targets['expression_embed']
        
        # Test 1: Expression consistency
        cos_sim = F.cosine_similarity(
            pred_expr.view(-1, pred_expr.size(-1)),
            target_expr.view(-1, target_expr.size(-1)),
            dim=-1
        ).mean().item()
        
        metrics['expression_consistency'] = cos_sim
        test_results['expression_consistency'] = (
            cos_sim > self.expression_criteria['expression_consistency']['threshold']
        )
        
        # Test 2: Expression drift
        drift = torch.norm(pred_expr - target_expr, dim=-1).mean().item()
        drift_ratio = drift / (torch.norm(target_expr, dim=-1).mean().item() + 1e-6)
        
        metrics['expression_drift'] = drift_ratio
        test_results['expression_drift'] = (
            drift_ratio < self.expression_criteria['expression_drift']['threshold']
        )
        
        # Test 3: Codebook alignment (if available)
        if self.expression_codebook is not None:
            pred_flat = pred_expr.view(-1, pred_expr.size(-1))
            distances = torch.cdist(pred_flat, self.expression_codebook)
            nearest_dist = distances.min(dim=1)[0].mean().item()
            
            # Normalize by average codebook distance
            codebook_spread = torch.cdist(
                self.expression_codebook,
                self.expression_codebook
            ).mean().item()
            
            alignment = 1.0 - (nearest_dist / codebook_spread)
            metrics['codebook_alignment'] = alignment
            test_results['codebook_alignment'] = (
                alignment > self.expression_criteria['codebook_alignment']['threshold']
            )
        
        return test_results, metrics
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        conditions: Dict[str, torch.Tensor] = None,
        stage: str = 'train'
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, bool]]:
        """
        Compute expression-preserving TDD losses
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss components
            test_results: Pass/fail for each test
        """
        
        losses = {}
        test_results = {}
        
        # 1. Expression Reconstruction Loss (PRIMARY)
        if 'expression_embed' in outputs and 'expression_embed' in targets:
            losses['expression_reconstruction'] = (
                self.expression_reconstruction_loss(
                    outputs['expression_embed'],
                    targets['expression_embed']
                ) * self.weights['expression_reconstruction']
            )
            
            # Build codebook if needed
            if self.expression_codebook is None and stage == 'train':
                self.build_expression_codebook(targets['expression_embed'])
            
            # 2. Codebook Loss
            losses['expression_codebook'] = (
                self.expression_codebook_loss(
                    outputs['expression_embed'],
                    targets['expression_embed']
                ) * self.weights['expression_codebook']
            )
            
            # 3. Temporal Loss
            losses['expression_temporal'] = (
                self.expression_temporal_loss(
                    outputs['expression_embed'],
                    targets['expression_embed']
                ) * self.weights['expression_temporal']
            )
            
            # Run preservation tests
            expr_tests, expr_metrics = self.test_expression_preservation(
                outputs, targets
            )
            test_results.update(expr_tests)
            
            # Log metrics
            for name, value in expr_metrics.items():
                losses[f'metric_{name}'] = value
        
        # 4. Motion Quality Loss (reduced weight to prioritize expression)
        if 'theta' in outputs and 'theta' in targets:
            motion_loss = F.mse_loss(outputs['theta'], targets['theta'])
            losses['motion_quality'] = motion_loss * self.weights['motion_quality']
        
        # 5. Lip Sync Loss (if audio conditions present)
        if conditions and 'audio_features' in conditions and 'lips' in outputs:
            # Simple amplitude-based lip sync
            audio_amp = torch.norm(conditions['audio_features'], dim=-1)
            lip_openness = torch.norm(outputs['lips'], dim=-1).mean(dim=-1)
            
            # Normalize
            audio_amp = (audio_amp - audio_amp.min()) / (audio_amp.max() - audio_amp.min() + 1e-6)
            lip_openness = (lip_openness - lip_openness.min()) / (lip_openness.max() - lip_openness.min() + 1e-6)
            
            lip_sync_loss = F.mse_loss(lip_openness, audio_amp)
            losses['lip_sync'] = lip_sync_loss * self.weights['lip_sync']
        
        # Compute total loss
        total_loss = sum(losses.values())
        
        # Log summary
        if stage == 'train' and len(test_results) > 0:
            passed = sum(test_results.values())
            total = len(test_results)
            logger.info(f"Expression Preservation: {passed}/{total} tests passed")
            
            if passed < total:
                failed = [k for k, v in test_results.items() if not v]
                logger.warning(f"Failed tests: {failed}")
        
        return total_loss, losses, test_results


def create_expression_preserving_trainer_config():
    """Create configuration for training with expression preservation"""
    
    config = {
        'loss': {
            'type': 'ExpressionPreservingTDD',
            'expression_weight': 2.0,  # Strong expression preservation
            'expression_codebook_weight': 1.5,
            'expression_temporal_weight': 1.0,
            'motion_weight': 0.8,  # Reduced for expression priority
            'lip_sync_weight': 1.0
        },
        'training': {
            'expression_warmup_epochs': 50,  # Focus on expressions first
            'expression_lr_mult': 2.0,  # Higher LR for expression params
            'use_curriculum': True,
            'curriculum_stages': [
                {
                    'epochs': 50,
                    'focus': 'expression_preservation',
                    'weights': {
                        'expression_reconstruction': 3.0,
                        'expression_codebook': 2.0,
                        'others': 0.5
                    }
                },
                {
                    'epochs': 100,
                    'focus': 'balanced',
                    'weights': {
                        'expression_reconstruction': 2.0,
                        'expression_codebook': 1.5,
                        'others': 1.0
                    }
                },
                {
                    'epochs': 200,
                    'focus': 'fine_tuning',
                    'weights': {
                        'all': 1.0
                    }
                }
            ]
        },
        'validation': {
            'expression_threshold': 0.90,  # Must maintain 90% similarity
            'max_drift': 0.05,  # Max 5% expression drift allowed
            'test_frequency': 10  # Test every 10 batches
        }
    }
    
    return config


if __name__ == "__main__":
    """Test expression preservation loss"""
    
    print("\n" + "="*70)
    print("Expression-Preserving TDD Loss Test")
    print("="*70)
    
    # Initialize loss module
    loss_module = ExpressionPreservingTDDLoss(
        expression_weight=2.0,
        expression_codebook_weight=1.5
    )
    
    # Create test data
    B, T, D = 2, 50, 128
    
    # Original expressions (what we want to preserve)
    target_expression = torch.randn(B, T, D).cuda()
    
    # Test different prediction scenarios
    scenarios = [
        ("Perfect Match", target_expression.clone()),
        ("Small Drift", target_expression + torch.randn_like(target_expression) * 0.1),
        ("Large Drift", target_expression + torch.randn_like(target_expression) * 0.5),
        ("Random", torch.randn_like(target_expression))
    ]
    
    for name, pred_expression in scenarios:
        print(f"\n{name}:")
        print("-" * 40)
        
        outputs = {'expression_embed': pred_expression}
        targets = {'expression_embed': target_expression}
        
        # Compute losses
        total_loss, losses, test_results = loss_module(
            outputs, targets, stage='test'
        )
        
        print(f"Total Loss: {total_loss.item():.4f}")
        print("\nLoss Components:")
        for loss_name, loss_value in losses.items():
            if not loss_name.startswith('metric_'):
                print(f"  {loss_name}: {loss_value.item():.4f}")
        
        print("\nExpression Preservation Tests:")
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            metric_name = f"metric_{test_name}"
            if metric_name in losses:
                metric_value = losses[metric_name]
                print(f"  {test_name}: {status} (value: {metric_value:.3f})")
            else:
                print(f"  {test_name}: {status}")
    
    print("\n" + "="*70)
    print("Expression preservation loss module ready!")
    print("Use this for training to maintain expression consistency")
    print("="*70)