"""
Expression Normalization Module for VASA Training

This module handles normalization of expression embeddings during training
and denormalization during inference to ensure compatibility with the
volumetric avatar model.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, Union
import numpy as np
from logger import logger


class ExpressionNormalizer:
    """Handles normalization and denormalization of expression embeddings."""
    
    def __init__(self, stats_path: Optional[Path] = None, device: str = 'cuda'):
        """
        Initialize the normalizer.
        
        Args:
            stats_path: Path to saved normalization statistics
            device: Device for tensor operations
        """
        self.device = device
        self.stats_path = stats_path or Path('expression_norm_stats.pt')
        self.stats_loaded = False
        
        # Initialize stats containers
        self.expression_mean = None
        self.expression_std = None
        self.expression_min = None
        self.expression_max = None
        
        # Additional stats for different components
        self.theta_stats = {}
        self.rotation_stats = {}
        self.translation_stats = {}
        self.scale_stats = {}
        self.landmark_stats = {}
        
        # Load existing stats if available
        if self.stats_path.exists():
            self.load_stats()
    
    def compute_stats_from_dataset(self, dataset, num_samples: Optional[int] = None):
        """
        Compute normalization statistics from a dataset.
        
        Args:
            dataset: VASAIntegratedDataset instance
            num_samples: Number of samples to use (None for all)
        """
        logger.info("Computing expression normalization statistics...")
        
        # Containers for accumulating stats
        expression_data = []
        theta_data = []
        rotation_data = []
        translation_data = []
        scale_data = []
        
        # Sample indices
        if num_samples:
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        else:
            indices = range(len(dataset))
        
        # Collect data
        for idx in indices:
            try:
                sample = dataset[idx]
                
                # Handle VASAIntegratedDataset structure with windows
                if 'windows' in sample and sample['windows']:
                    # Process each window in the sample
                    for window in sample['windows']:
                        # Extract expression embeddings if present
                        if 'expression_embed' in window:
                            expr = window['expression_embed']
                            if isinstance(expr, torch.Tensor):
                                expression_data.append(expr.cpu())
                        
                        # Extract motion parameters
                        if 'theta' in window:
                            theta_data.append(window['theta'].cpu())
                        if 'rotation' in window:
                            rotation_data.append(window['rotation'].cpu())
                        if 'translation' in window:
                            translation_data.append(window['translation'].cpu())
                        if 'scale' in window:
                            scale_data.append(window['scale'].cpu())
                else:
                    # Fallback to direct access if not windowed
                    # Extract expression embeddings if present
                    if 'expression_embed' in sample:
                        expr = sample['expression_embed']
                        if isinstance(expr, torch.Tensor):
                            expression_data.append(expr.cpu())
                    
                    # Extract motion parameters
                    if 'theta' in sample:
                        theta_data.append(sample['theta'].cpu())
                    if 'rotation' in sample:
                        rotation_data.append(sample['rotation'].cpu())
                    if 'translation' in sample:
                        translation_data.append(sample['translation'].cpu())
                    if 'scale' in sample:
                        scale_data.append(sample['scale'].cpu())
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        # Compute statistics for expression embeddings
        if expression_data:
            expressions = torch.stack(expression_data)
            self.expression_mean = expressions.mean(dim=(0, 1))  # Mean across batch and time
            self.expression_std = expressions.std(dim=(0, 1))
            self.expression_min = expressions.min(dim=0)[0].min(dim=0)[0]  # Min across all dimensions
            self.expression_max = expressions.max(dim=0)[0].max(dim=0)[0]
            
            logger.info(f"Expression stats computed from {len(expression_data)} samples")
            logger.info(f"  Mean range: [{self.expression_mean.min():.3f}, {self.expression_mean.max():.3f}]")
            logger.info(f"  Std range: [{self.expression_std.min():.3f}, {self.expression_std.max():.3f}]")
            logger.info(f"  Value range: [{self.expression_min.min():.3f}, {self.expression_max.max():.3f}]")
        
        # Compute stats for motion parameters
        if theta_data:
            thetas = torch.stack(theta_data)
            self.theta_stats = {
                'mean': thetas.mean(dim=(0, 1)),
                'std': thetas.std(dim=(0, 1)),
                'min': thetas.min(dim=0)[0].min(dim=0)[0],
                'max': thetas.max(dim=0)[0].max(dim=0)[0]
            }
            logger.info(f"Theta stats computed: range [{self.theta_stats['min'].min():.3f}, {self.theta_stats['max'].max():.3f}]")
        
        if rotation_data:
            rotations = torch.stack(rotation_data)
            self.rotation_stats = {
                'mean': rotations.mean(dim=(0, 1)),
                'std': rotations.std(dim=(0, 1)),
                'min': rotations.min(dim=0)[0].min(dim=0)[0],
                'max': rotations.max(dim=0)[0].max(dim=0)[0]
            }
            logger.info(f"Rotation stats computed: range [{self.rotation_stats['min'].min():.3f}, {self.rotation_stats['max'].max():.3f}]")
        
        if translation_data:
            translations = torch.stack(translation_data)
            self.translation_stats = {
                'mean': translations.mean(dim=(0, 1)),
                'std': translations.std(dim=(0, 1)),
                'min': translations.min(dim=0)[0].min(dim=0)[0],
                'max': translations.max(dim=0)[0].max(dim=0)[0]
            }
            logger.info(f"Translation stats computed: range [{self.translation_stats['min'].min():.3f}, {self.translation_stats['max'].max():.3f}]")
        
        if scale_data:
            scales = torch.stack(scale_data)
            self.scale_stats = {
                'mean': scales.mean(dim=(0, 1)),
                'std': scales.std(dim=(0, 1)),
                'min': scales.min(dim=0)[0].min(dim=0)[0],
                'max': scales.max(dim=0)[0].max(dim=0)[0]
            }
            logger.info(f"Scale stats computed: range [{self.scale_stats['min'].min():.3f}, {self.scale_stats['max'].max():.3f}]")
        
        self.stats_loaded = True
        self.save_stats()
    
    def normalize_expression(self, expr: torch.Tensor, mode: str = 'z-score') -> torch.Tensor:
        """
        Normalize expression embeddings.
        
        Args:
            expr: Expression tensor [B, T, D] or [B, D]
            mode: 'z-score' or 'min-max'
        
        Returns:
            Normalized expression tensor
        """
        if not self.stats_loaded:
            logger.warning("No normalization stats loaded, returning original expressions")
            return expr
        
        # Move stats to same device as input
        device = expr.device
        
        if mode == 'z-score':
            mean = self.expression_mean.to(device)
            std = self.expression_std.to(device)
            # Add small epsilon to avoid division by zero
            normalized = (expr - mean) / (std + 1e-8)
        elif mode == 'min-max':
            min_val = self.expression_min.to(device)
            max_val = self.expression_max.to(device)
            # Scale to [0, 1]
            normalized = (expr - min_val) / (max_val - min_val + 1e-8)
            # Scale to [-1, 1] if needed
            normalized = 2 * normalized - 1
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        
        return normalized
    
    def denormalize_expression(self, expr: torch.Tensor, mode: str = 'z-score') -> torch.Tensor:
        """
        Denormalize expression embeddings for inference.
        
        Args:
            expr: Normalized expression tensor [B, T, D] or [B, D]
            mode: 'z-score' or 'min-max'
        
        Returns:
            Denormalized expression tensor
        """
        if not self.stats_loaded:
            logger.warning("No normalization stats loaded, returning original expressions")
            return expr
        
        # Move stats to same device as input
        device = expr.device
        
        if mode == 'z-score':
            mean = self.expression_mean.to(device)
            std = self.expression_std.to(device)
            denormalized = expr * std + mean
        elif mode == 'min-max':
            min_val = self.expression_min.to(device)
            max_val = self.expression_max.to(device)
            # Scale from [-1, 1] to [0, 1]
            denormalized = (expr + 1) / 2
            # Scale to original range
            denormalized = denormalized * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        
        # Clip to original range to avoid outliers
        denormalized = torch.clamp(denormalized, self.expression_min.to(device), self.expression_max.to(device))
        
        return denormalized
    
    def normalize_motion_params(self, theta=None, rotation=None, translation=None, scale=None):
        """
        Normalize motion parameters.
        
        Returns:
            Dictionary of normalized parameters
        """
        normalized = {}
        
        if theta is not None and self.theta_stats:
            device = theta.device
            mean = self.theta_stats['mean'].to(device)
            std = self.theta_stats['std'].to(device)
            normalized['theta'] = (theta - mean) / (std + 1e-8)
        
        if rotation is not None and self.rotation_stats:
            device = rotation.device
            mean = self.rotation_stats['mean'].to(device)
            std = self.rotation_stats['std'].to(device)
            normalized['rotation'] = (rotation - mean) / (std + 1e-8)
        
        if translation is not None and self.translation_stats:
            device = translation.device
            mean = self.translation_stats['mean'].to(device)
            std = self.translation_stats['std'].to(device)
            normalized['translation'] = (translation - mean) / (std + 1e-8)
        
        if scale is not None and self.scale_stats:
            device = scale.device
            mean = self.scale_stats['mean'].to(device)
            std = self.scale_stats['std'].to(device)
            normalized['scale'] = (scale - mean) / (std + 1e-8)
        
        return normalized
    
    def denormalize_motion_params(self, theta=None, rotation=None, translation=None, scale=None):
        """
        Denormalize motion parameters for inference.
        
        Returns:
            Dictionary of denormalized parameters
        """
        denormalized = {}
        
        if theta is not None and self.theta_stats:
            device = theta.device
            mean = self.theta_stats['mean'].to(device)
            std = self.theta_stats['std'].to(device)
            denormalized['theta'] = theta * std + mean
        
        if rotation is not None and self.rotation_stats:
            device = rotation.device
            mean = self.rotation_stats['mean'].to(device)
            std = self.rotation_stats['std'].to(device)
            denormalized['rotation'] = rotation * std + mean
        
        if translation is not None and self.translation_stats:
            device = translation.device
            mean = self.translation_stats['mean'].to(device)
            std = self.translation_stats['std'].to(device)
            denormalized['translation'] = translation * std + mean
        
        if scale is not None and self.scale_stats:
            device = scale.device
            mean = self.scale_stats['mean'].to(device)
            std = self.scale_stats['std'].to(device)
            denormalized['scale'] = scale * std + mean
        
        return denormalized
    
    def save_stats(self):
        """Save normalization statistics to disk."""
        stats = {
            'expression_mean': self.expression_mean,
            'expression_std': self.expression_std,
            'expression_min': self.expression_min,
            'expression_max': self.expression_max,
            'theta_stats': self.theta_stats,
            'rotation_stats': self.rotation_stats,
            'translation_stats': self.translation_stats,
            'scale_stats': self.scale_stats,
            'landmark_stats': self.landmark_stats
        }
        
        # Convert to CPU before saving
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                stats[key] = value.cpu()
            elif isinstance(value, dict):
                stats[key] = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in value.items()}
        
        torch.save(stats, self.stats_path)
        logger.info(f"Normalization statistics saved to {self.stats_path}")
        
        # Also save a human-readable JSON summary
        summary_path = self.stats_path.with_suffix('.json')
        summary = {}
        
        if self.expression_mean is not None:
            summary['expression'] = {
                'mean_range': [float(self.expression_mean.min()), float(self.expression_mean.max())],
                'std_range': [float(self.expression_std.min()), float(self.expression_std.max())],
                'value_range': [float(self.expression_min.min()), float(self.expression_max.max())],
                'shape': list(self.expression_mean.shape)
            }
        
        for name, stats_dict in [('theta', self.theta_stats), ('rotation', self.rotation_stats),
                                  ('translation', self.translation_stats), ('scale', self.scale_stats)]:
            if stats_dict:
                summary[name] = {
                    'mean_range': [float(stats_dict['mean'].min()), float(stats_dict['mean'].max())],
                    'std_range': [float(stats_dict['std'].min()), float(stats_dict['std'].max())],
                    'value_range': [float(stats_dict['min'].min()), float(stats_dict['max'].max())]
                }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
    
    def load_stats(self):
        """Load normalization statistics from disk."""
        if not self.stats_path.exists():
            logger.warning(f"No stats file found at {self.stats_path}")
            return False
        
        try:
            stats = torch.load(self.stats_path, map_location='cpu')
            
            self.expression_mean = stats.get('expression_mean')
            self.expression_std = stats.get('expression_std')
            self.expression_min = stats.get('expression_min')
            self.expression_max = stats.get('expression_max')
            self.theta_stats = stats.get('theta_stats', {})
            self.rotation_stats = stats.get('rotation_stats', {})
            self.translation_stats = stats.get('translation_stats', {})
            self.scale_stats = stats.get('scale_stats', {})
            self.landmark_stats = stats.get('landmark_stats', {})
            
            self.stats_loaded = True
            logger.info(f"Normalization statistics loaded from {self.stats_path}")
            
            # Log summary
            if self.expression_mean is not None:
                logger.info(f"  Expression shape: {self.expression_mean.shape}")
                logger.info(f"  Value range: [{self.expression_min.min():.3f}, {self.expression_max.max():.3f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
            return False
    
    def get_stats_summary(self) -> Dict:
        """Get a summary of the normalization statistics."""
        summary = {
            'stats_loaded': self.stats_loaded,
            'has_expression_stats': self.expression_mean is not None,
            'has_motion_stats': bool(self.theta_stats)
        }
        
        if self.expression_mean is not None:
            summary['expression'] = {
                'shape': list(self.expression_mean.shape),
                'mean_range': [float(self.expression_mean.min()), float(self.expression_mean.max())],
                'std_range': [float(self.expression_std.min()), float(self.expression_std.max())],
                'value_range': [float(self.expression_min.min()), float(self.expression_max.max())]
            }
        
        return summary