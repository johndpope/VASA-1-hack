import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal
from dataclasses import dataclass

class FeatureMatchingLoss(nn.Module):
    """
    Calculates feature matching loss between real and fake features across multiple discriminators
    and multiple layers within each discriminator.
    
    Features are expected in format:
    [  # List of discriminators
        [  # List of layers for each discriminator
            [  # List of features for each input at this layer
                tensor  # Feature tensor
            ]
        ]
    ]
    """
    
    def __init__(self, loss_type: Literal['l1', 'l2'] = 'l1'):
        """
        Initialize feature matching loss module.
        
        Args:
            loss_type: Type of base loss to use - 'l1' for L1 loss or 'l2' for MSE loss
        """
        super().__init__()
        
        self.loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss
        
    def _compute_layer_loss(self, 
                           real_features: List[torch.Tensor],
                           fake_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss between real and fake features at a single layer.
        
        Args:
            real_features: List of real feature tensors
            fake_features: List of fake feature tensors
            
        Returns:
            torch.Tensor: Loss value for this layer
        """
        # If only one real feature, replicate it for each fake feature
        if len(real_features) == 1:
            real_features = [real_features[0]] * len(fake_features)
            
        # Compute loss between each real-fake pair
        layer_losses = [
            self.loss_fn(fake, real) 
            for fake, real in zip(fake_features, real_features)
        ]
        
        return sum(layer_losses) / len(layer_losses)
        
    def _compute_discriminator_loss(self,
                                  real_features: List[List[torch.Tensor]],
                                  fake_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute average loss across all layers of a single discriminator.
        
        Args:
            real_features: List of real feature lists per layer
            fake_features: List of fake feature lists per layer
            
        Returns:
            torch.Tensor: Average loss across all layers
        """
        # Compute loss for each layer
        layer_losses = [
            self._compute_layer_loss(real_layer, fake_layer)
            for real_layer, fake_layer in zip(real_features, fake_features)
        ]
        
        return sum(layer_losses) / len(layer_losses)

    def forward(self,
                real_features: List[List[List[torch.Tensor]]],
                fake_features: List[List[List[torch.Tensor]]]) -> torch.Tensor:
        """
        Compute total feature matching loss across all discriminators.
        
        Args:
            real_features: Nested list of real features [discriminators][layers][features]
            fake_features: Nested list of fake features [discriminators][layers][features]
            
        Returns:
            torch.Tensor: Total feature matching loss
            
        Raises:
            AssertionError: If real and fake feature dimensions don't match appropriately
        """
        # Validate inputs
        assert len(real_features) == len(fake_features), \
            f"Number of discriminators must match: {len(real_features)} != {len(fake_features)}"
            
        for i, (real_disc, fake_disc) in enumerate(zip(real_features, fake_features)):
            assert len(real_disc) == len(fake_disc), \
                f"Number of layers must match for discriminator {i}: {len(real_disc)} != {len(fake_disc)}"
                
        # Compute loss for each discriminator
        discriminator_losses = [
            self._compute_discriminator_loss(real_disc, fake_disc)
            for real_disc, fake_disc in zip(real_features, fake_features)
        ]
        
        # Return average loss across all discriminators
        return sum(discriminator_losses) / len(discriminator_losses)