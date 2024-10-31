import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Tuple
from VASAConfig import VASAConfig


class IdentityLoss(nn.Module):
    """
    Identity preservation loss using pretrained face recognition model.
    Ensures generated faces maintain the identity of the source image.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Initialize with pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()  # Remove classification layer
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add identity projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.register_buffer('center', torch.zeros(256))
        self.register_buffer('std', torch.ones(256))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract identity features from face images"""
        features = self.backbone(x)
        features = self.projection(features)
        # Normalize features
        features = (features - self.center) / self.std
        return F.normalize(features, p=2, dim=1)

    def forward(self, generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss between generated and source images
        Args:
            generated: Generated face images [B, C, H, W]
            source: Source face images [B, C, H, W]
        Returns:
            Identity loss value
        """
        gen_features = self.extract_features(generated)
        src_features = self.extract_features(source)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(gen_features, src_features, dim=1)
        identity_loss = 1.0 - cos_sim.mean()
        
        return identity_loss

class PoseExtractionNet(nn.Module):
    """
    Network for extracting head pose parameters from face images.
    """
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Modify final layer for pose parameters (rotation + translation)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 6)  # 3 for rotation, 3 for translation
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract pose parameters
        Returns:
            rotation: Rotation parameters [B, 3]
            translation: Translation parameters [B, 3]
        """
        pose_params = self.backbone(x)
        rotation = pose_params[:, :3]
        translation = pose_params[:, 3:]
        return rotation, translation

class ExpressionExtractionNet(nn.Module):
    """
    Network for extracting facial expression parameters.
    """
    def __init__(self, expression_dim: int = 64):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, expression_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract expression parameters"""
        return self.backbone(x)

class DPELoss(nn.Module):
    """
    Disentanglement of Pose and Expression (DPE) loss.
    Ensures effective disentanglement between pose and facial expressions.
    """
    def __init__(self, 
                 expression_dim: int = 64,
                 lambda_pose: float = 1.0,
                 lambda_expr: float = 1.0):
        super().__init__()
        self.pose_net = PoseExtractionNet()
        self.expression_net = ExpressionExtractionNet(expression_dim)
        
        # Loss weights
        self.lambda_pose = lambda_pose
        self.lambda_expr = lambda_expr
        
        # Feature reconstruction loss
        self.recon_loss = nn.MSELoss()
        
        # Freeze networks
        for param in self.pose_net.parameters():
            param.requires_grad = False
        for param in self.expression_net.parameters():
            param.requires_grad = False

    def compute_pose_consistency(self, 
                               I_i: torch.Tensor,
                               I_j: torch.Tensor,
                               I_i_pose_j: torch.Tensor) -> torch.Tensor:
        """Compute pose consistency loss"""
        # Extract poses
        rot_i, trans_i = self.pose_net(I_i)
        rot_j, trans_j = self.pose_net(I_j)
        rot_transferred, trans_transferred = self.pose_net(I_i_pose_j)
        
        # Pose should match target
        pose_loss = (
            F.mse_loss(rot_transferred, rot_j) +
            F.mse_loss(trans_transferred, trans_j)
        )
        return pose_loss

    def compute_expression_consistency(self,
                                    I_i: torch.Tensor,
                                    I_j: torch.Tensor,
                                    I_i_pose_j: torch.Tensor) -> torch.Tensor:
        """Compute expression consistency loss"""
        # Extract expressions
        expr_i = self.expression_net(I_i)
        expr_j = self.expression_net(I_j)
        expr_transferred = self.expression_net(I_i_pose_j)
        
        # Expression should remain the same after pose transfer
        expr_loss = F.mse_loss(expr_transferred, expr_i)
        return expr_loss

    def forward(self, 
                I_i: torch.Tensor,
                I_j: torch.Tensor,
                I_i_pose_j: torch.Tensor,
                I_j_pose_i: torch.Tensor,
                I_s: torch.Tensor,
                I_d: torch.Tensor,
                I_s_pose_d_dyn_d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DPE loss components
        Args:
            I_i, I_j: Source frames from same identity
            I_i_pose_j: I_i with I_j's pose
            I_j_pose_i: I_j with I_i's pose
            I_s: Source identity frame
            I_d: Different identity frame
            I_s_pose_d_dyn_d: Source frame with different identity's pose and dynamics
        """
        losses = {}
        
        # Pose consistency loss
        losses['pose_i'] = self.compute_pose_consistency(I_i, I_j, I_i_pose_j)
        losses['pose_j'] = self.compute_pose_consistency(I_j, I_i, I_j_pose_i)
        
        # Expression consistency loss
        losses['expr_i'] = self.compute_expression_consistency(I_i, I_j, I_i_pose_j)
        losses['expr_j'] = self.compute_expression_consistency(I_j, I_i, I_j_pose_i)
        
        # Cross-identity pose transfer loss
        losses['cross_pose'] = self.compute_pose_consistency(I_s, I_d, I_s_pose_d_dyn_d)
        
        # Cross-identity expression preservation
        losses['cross_expr'] = self.compute_expression_consistency(I_s, I_d, I_s_pose_d_dyn_d)
        
        # Total loss
        losses['total'] = (
            self.lambda_pose * (losses['pose_i'] + losses['pose_j'] + losses['cross_pose']) +
            self.lambda_expr * (losses['expr_i'] + losses['expr_j'] + losses['cross_expr'])
        )
        
        return losses

class CombinedVASALoss(nn.Module):
    """
    Combined loss function for VASA training
    """
    def __init__(self,
                 lambda_identity: float = 0.1,
                 lambda_dpe: float = 0.1):
        super().__init__()
        self.identity_loss = IdentityLoss()
        self.dpe_loss = DPELoss()
        
        self.lambda_identity = lambda_identity
        self.lambda_dpe = lambda_dpe

    def forward(self,
                generated: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        Args:
            generated: Dict containing generated images and intermediate results
            target: Dict containing ground truth images and attributes
        """
        losses = {}
        
        # Identity preservation loss
        losses['identity'] = self.identity_loss(
            generated['output'],
            target['source_image']
        )
        
        # DPE losses
        dpe_losses = self.dpe_loss(
            generated['source'], generated['target'],
            generated['source_pose_transfer'], generated['target_pose_transfer'],
            generated['source_identity'], generated['target_identity'],
            generated['cross_identity_transfer']
        )
        losses.update({f'dpe_{k}': v for k, v in dpe_losses.items()})
        
        # Total loss
        losses['total'] = (
            losses['identity'] * self.lambda_identity +
            dpe_losses['total'] * self.lambda_dpe
        )
        
        return losses

def test_losses():
    """Test loss computations"""
    batch_size = 4
    img_size = 256
    
    # Create dummy data
    dummy_data = {
        'source': torch.randn(batch_size, 3, img_size, img_size),
        'target': torch.randn(batch_size, 3, img_size, img_size),
        'source_pose_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'target_pose_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'source_identity': torch.randn(batch_size, 3, img_size, img_size),
        'target_identity': torch.randn(batch_size, 3, img_size, img_size),
        'cross_identity_transfer': torch.randn(batch_size, 3, img_size, img_size),
        'output': torch.randn(batch_size, 3, img_size, img_size)
    }
    
    target_data = {
        'source_image': torch.randn(batch_size, 3, img_size, img_size)
    }
    
    # Test loss computation
    loss_fn = CombinedVASALoss()
    losses = loss_fn(dummy_data, target_data)
    
    print("Loss components:")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")



class VASALossModule:
    """Loss module for VASA training"""
    def __init__(self, config: VASAConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize loss components
        self.identity_loss = IdentityLoss().to(device)
        self.dpe_loss = DPELoss(
            expression_dim=config.motion_dim,
            lambda_pose=config.lambda_pose,
            lambda_expr=config.lambda_expr
        ).to(device)
        self.combined_loss = CombinedVASALoss(
            lambda_identity=config.lambda_identity,
            lambda_dpe=config.lambda_dpe
        ).to(device)

    def compute_losses(
        self,
        generated_frames: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        face_components: Dict[str, torch.Tensor],
        diffusion_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses
        Args:
            generated_frames: Generated video frames
            batch: Training batch data
            face_components: Face encoder outputs
            diffusion_output: Diffusion model outputs
        """
        # Prepare inputs for loss computation
        loss_inputs = {
            'source': batch['frames'][:, 0],  # First frame is source
            'target': batch['frames'][:, 1:],  # Remaining frames are targets
            'source_pose_transfer': generated_frames[:, 0],
            'target_pose_transfer': generated_frames[:, 1:],
            'source_identity': face_components['identity'],
            'target_identity': face_components['identity'],
            'cross_identity_transfer': generated_frames,
            'output': generated_frames
        }
        
        # Ground truth data
        target_data = {
            'source_image': batch['frames'][:, 0]
        }
        
        # Compute main losses
        main_losses = self.combined_loss(loss_inputs, target_data)
        
        # Add additional losses
        losses = {
            'reconstruction': F.l1_loss(generated_frames, batch['frames']),
            'identity': main_losses['identity'],
            'dpe_total': main_losses['dpe_total']
        }
        
        # Add individual DPE losses for monitoring
        losses.update({
            f'dpe_{k}': v for k, v in main_losses.items() 
            if k.startswith('dpe_') and k != 'dpe_total'
        })
        
        # Add CFG losses
        if 'uncond' in diffusion_output:
            losses.update(self._compute_cfg_losses(diffusion_output))
        
        # Compute weighted total loss
        losses['total'] = (
            self.config.lambda_recon * losses['reconstruction'] +
            self.config.lambda_identity * losses['identity'] +
            self.config.lambda_dpe * losses['dpe_total'] +
            sum(self.config.lambda_cfg * losses[f'cfg_{k}'] 
                for k in ['audio', 'gaze'] 
                if f'cfg_{k}' in losses)
        )
        
        return losses

    def _compute_cfg_losses(
        self,
        diffusion_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute classifier-free guidance losses"""
        cfg_losses = {}
        
        # Base unconditional output
        uncond_output = diffusion_output['uncond']
        
        # Audio CFG loss
        if 'masked_audio' in diffusion_output:
            cfg_losses['cfg_audio'] = F.mse_loss(
                diffusion_output['masked_audio'],
                uncond_output
            )
        
        # Gaze CFG loss
        if 'masked_gaze' in diffusion_output:
            cfg_losses['cfg_gaze'] = F.mse_loss(
                diffusion_output['masked_gaze'],
                uncond_output
            )
            
        return cfg_losses