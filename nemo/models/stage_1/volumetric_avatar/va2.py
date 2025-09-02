
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from enum import Enum, auto
from networks import basic_avatar, volumetric_avatar
from aitypes import *


import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from networks.volumetric_avatar.utils import replace_bn_to_bcn,replace_bn_to_gn,replace_bn_to_in,ResBlock,norm_layers,activations


from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import point_transforms


@dataclass
class IdtEmbedConfig:
    """Configuration for Identity Embedding Network.

    Attributes:
        idt_backbone: Name of the backbone architecture (e.g., 'resnet18')
        num_source_frames: Number of source images per identity
        idt_output_size: Size of the output feature map
        idt_output_channels: Number of output channels
        num_gpus: Number of GPUs to use
        norm_layer_type: Type of normalization layer ('bn', 'in', 'gn', 'bcn')
        idt_image_size: Input image size
    """
    idt_backbone: str ="resnet50"
    num_source_frames: int=1
    idt_output_size: int=8
    idt_output_channels: int=512
    num_gpus: int=1
    norm_layer_type: str="gn"
    idt_image_size: int=256


class IdtEmbed(nn.Module):
    """Identity Embedding Network.
    
    This module processes source images to create identity embeddings using a backbone
    CNN architecture with customizable normalization layers.
    """

    def __init__(self, cfg: IdtEmbedConfig):
        """Initialize the Identity Embedding network.

        Args:
            cfg: Configuration object containing network parameters
        """
        super().__init__()
        self.cfg = cfg
        self._setup_network()
        self._register_normalization_values()

    def _setup_network(self) -> None:
        """Set up the network architecture including backbone and custom layers."""
        expansion = self._get_expansion_factor()
        self.net = self._create_backbone()
        self.net.avgpool = nn.AdaptiveAvgPool2d(self.cfg.idt_output_size)
        self.net.fc = nn.Conv2d(
            in_channels=512 * expansion,
            out_channels=self.cfg.idt_output_channels,
            kernel_size=1,
            bias=False
        )
        self._setup_normalization_layers()

    def _get_expansion_factor(self) -> int:
        """Determine the expansion factor based on backbone architecture."""
        return 1 if self.cfg.idt_backbone == 'resnet18' else 4

    def _create_backbone(self) -> nn.Module:
        """Create the backbone network."""
        return getattr(models, self.cfg.idt_backbone)(pretrained=True)

    def _setup_normalization_layers(self) -> None:
        """Set up normalization layers based on configuration."""
        norm_types = {
            'in': (replace_bn_to_in, 'Instance Normalization'),
            'gn': (replace_bn_to_gn, 'Group Normalization'),
            'bcn': (replace_bn_to_bcn, 'Batch-Channel Normalization'),
        }

        if self.cfg.norm_layer_type == 'bn':
            return
        
        if self.cfg.norm_layer_type not in norm_types:
            raise ValueError(f"Unsupported normalization type: {self.cfg.norm_layer_type}")
        
        replace_fn, norm_name = norm_types[self.cfg.norm_layer_type]
        self.net = replace_fn(self.net, 'IdtEmbed')

    def _register_normalization_values(self) -> None:
        """Register normalization values as buffers."""
        self.register_buffer('mean', 
            torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', 
            torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after passing through all network layers
        """
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.fc(x)
        x = self.net.avgpool(x)
        return x

    def forward_image(self, source_img: torch.Tensor) -> torch.Tensor:
        """Process source images to create identity embeddings.

        Args:
            source_img: Input source images tensor of shape (B*N, C, H, W)
                where N is num_source_frames

        Returns:
            Identity embedding tensor
        """
        source_img = F.interpolate(
            source_img, 
            size=(self.cfg.idt_image_size, self.cfg.idt_image_size), 
            mode='bilinear'
        )
        n = self.cfg.num_source_frames
        b = source_img.shape[0] // n

        inputs = (source_img - self.mean) / self.std
        idt_embed_tensor = self._forward_impl(inputs)
        return idt_embed_tensor.view(b, n, *idt_embed_tensor.shape[1:]).mean(1)

    def forward(self, source_img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            source_img: Input source images tensor

        Returns:
            Identity embedding tensor
        """
        return self.forward_image(source_img)


@dataclass(frozen=True)
class LocalEncoderConfig:
    """Hardcoded configuration for LocalEncoder."""
    # Generator parameters
    gen_upsampling_type: str = "trilinear"
    gen_downsampling_type: str = "avgpool"
    gen_input_image_size: int = 512
    gen_latent_texture_size: int = 64
    gen_latent_texture_depth: int = 16
    gen_latent_texture_channels: int = 96
    gen_num_channels: int = 32
    gen_max_channels: int = 512
    gen_activation_type: str = "relu"
    
    # Encoder parameters
    enc_channel_mult: float = 4.0
    enc_block_type: str = "res"
    
    # Normalization and other parameters
    norm_layer_type: str = "gn"
    num_gpus: int = 8
    warp_norm_grad: bool = False
    in_channels: int = 3


class LocalEncoder(nn.Module):
    """
    LocalEncoder with hardcoded configuration for processing input images into latent representations.
    """
    def __init__(self):
        super().__init__()
        self.cfg = LocalEncoderConfig()
        self._init_parameters()
        self._build_network()

    def _init_parameters(self):
        """Initialize model parameters and computed values."""
        self.ratio = self.cfg.gen_input_image_size // self.cfg.gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))
        self.init_depth = self.cfg.gen_latent_texture_depth
        self.spatial_size = self.cfg.gen_input_image_size
        
        # Determine normalization type
        self.norm_type = (
            self.cfg.norm_layer_type if self.cfg.norm_layer_type != 'bn'
            else 'sync_bn' if self.cfg.num_gpus >= 2 else 'bn'
        )

    def _build_network(self):
        """Construct the network architecture."""
        # Initialize grid sample if needed
        if self.cfg.warp_norm_grad:
            from . import GridSample
            self.grid_sample = GridSample(self.cfg.gen_latent_texture_size)
        else:
            self.grid_sample = lambda inputs, grid: F.grid_sample(
                inputs.float(), 
                grid.float(), 
                padding_mode='reflection'
            )

        # Initial convolution
        out_channels = int(self.cfg.gen_num_channels * self.cfg.enc_channel_mult)
        self.initial_conv = nn.Conv2d(
            in_channels=self.cfg.in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=3
        )

        # Build downsampling blocks
        self.down_blocks = nn.ModuleList()
        spatial_size = self.spatial_size
        
        for i in range(self.num_2d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.cfg.gen_max_channels)
            
            block = self._create_down_block(
                in_channels, 
                out_channels, 
                spatial_size
            )
            self.down_blocks.append(block)
            spatial_size //= 2

        # Final processing layers
        self.finale = self._create_finale_layers(out_channels)

    def _create_down_block(self, in_channels, out_channels, spatial_size):
        """Create a single downsampling block."""

        return ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            norm_layer_type=self.norm_type,
            activation_type=self.cfg.gen_activation_type,
            resize_layer_type=self.cfg.gen_downsampling_type
        )

    def _create_finale_layers(self, in_channels):
        """Create the final processing layers."""

        layers = []
        
        # Add normalization and activation for residual blocks
        if self.cfg.enc_block_type == 'res':
            layers.extend([
                norm_layers[self.norm_type](in_channels),
                activations[self.cfg.gen_activation_type](inplace=True)
            ])

        # Add final convolution
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.cfg.gen_latent_texture_channels * self.init_depth,
                kernel_size=1
            )
        )
        
        return nn.Sequential(*layers)

    def forward(self, source_img):
        """
        Forward pass of the encoder.
        
        Args:
            source_img: Input image tensor
            
        Returns:
            Encoded representation of the input image
        """
        x = self.initial_conv(source_img)
        
        # Apply downsampling blocks
        for block in self.down_blocks:
            x = block(x)
            
        # Apply final processing
        x = self.finale(x)
        
        return x


# Example usage:
def create_encoder():
    """Create an instance of the LocalEncoder with hardcoded config."""
    return LocalEncoder()


# Optional utility function to get the config without creating the model
def get_encoder_config():
    """Get the hardcoded configuration used by LocalEncoder."""
    return LocalEncoderConfig()


@dataclass
class ModelConfig:
    """Main configuration class for the volumetric avatar model"""
    
    # Basic model parameters
    image_size: int = 256
    latent_dim: int = 512
    num_source_frames: int = 1
    batch_size: int = 8
    
    # Network architecture
    activation_type: ActivationType = ActivationType.RELU
    norm_layer_type: NormLayerType = NormLayerType.BATCH_NORM
    use_spectral_norm: bool = True
    use_weight_standardization: bool = False
    
    # Encoder settings
    encoder_channels: int = 32
    encoder_max_channels: int = 512
    encoder_channel_multiplier: float = 2.0
    encoder_block_type: BlockType = BlockType.RESIDUAL
    
    # Volume settings
    volume_channels: int = 64
    volume_size: int = 64
    volume_depth: int = 16
    volume_blocks: int = 4
    use_volume_renderer: bool = False
    
    # Decoder settings
    decoder_channels: int = 32
    decoder_max_channels: int = 512
    decoder_blocks: int = 8
    decoder_channel_multiplier: float = 2.0
    decoder_predict_segmentation: bool = False
    
    # Training settings
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    adversarial_loss_weight: float = 1.0
    feature_matching_weight: float = 60.0
    vgg_loss_weight: float = 20.0
    
    # Discriminator settings
    discriminator_channels: int = 64
    discriminator_max_channels: int = 512
    discriminator_blocks: int = 4
    discriminator_scales: int = 2
    use_style_discriminator: bool = False
    
    # Advanced features
    use_adaptive_convolution: bool = False
    use_adaptive_normalization: bool = False
    use_mixing_generator: bool = True
    predict_cycle: bool = True
    
    # Volume renderer settings (if enabled)
    @dataclass
    class VolumeRendererConfig:
        depth_resolution: int = 48
        hidden_dim: int = 448
        num_layers: int = 2
        squeeze_dim: int = 0
        features_sigmoid: bool = True
    
    volume_renderer_config: VolumeRendererConfig = field(default_factory=VolumeRendererConfig)
    
    # Expression embedding settings
    @dataclass
    class ExpressionConfig:
        backbone: str = "resnet18"
        output_channels: int = 512
        output_size: int = 4
        dropout: float = 0.0
        use_smart_scaling: bool = False
        max_scale: float = 0.75
        max_angle_tolerance: float = 0.8
    
    expression_config: ExpressionConfig = field(default_factory=ExpressionConfig)
    
    # Identity embedding settings
    @dataclass
    class IdentityConfig:
        backbone: str = "resnet50"
        output_channels: int = 512
        output_size: int = 4
        image_size: int = 256
    
    identity_config: IdentityConfig = field(default_factory=IdentityConfig)
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        if self.num_source_frames != 1:
            raise ValueError("Multiple source frames are not supported")
        
        if self.volume_size % 2 != 0:
            raise ValueError("Volume size must be even")
            
        if self.volume_depth % 2 != 0:
            raise ValueError("Volume depth must be even")
    
    @property
    def device_count(self) -> int:
        """Get number of available GPU devices"""
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 1

    def get_optimizer_config(self, optimizer_type: str = "adam") -> Dict:
        """Get optimizer configuration"""
        return {
            "lr": self.learning_rate,
            "betas": (self.beta1, self.beta2),
            "eps": 1e-8
        }

    def get_scheduler_config(self, scheduler_type: str = "cosine") -> Dict:
        """Get learning rate scheduler configuration"""
        return {
            "T_max": 250000,
            "eta_min": 1e-6
        }
    
@dataclass
class NetworkConfig:
    """Configuration for network architectures"""
    local_encoder: Dict = None
    volume_renderer: Dict = None
    idt_embedder: Dict = None 
    exp_embedder: Dict = None
    warp_generator: Dict = None
    decoder: Dict = None
    discriminator: Dict = None
    unet3d: Dict = None
    
class Model(nn.Module):
    """Main model class for volumetric avatar generation"""

    def __init__(self, config: ModelConfig, training: bool = True, rank: int = 0, exp_dir: Optional[str] = None):
        super().__init__()
        self.config = config
        self.exp_dir = exp_dir
        self.rank = rank
        
        # Initialize basic parameters
        self.num_source_frames = 1  # Single source frame support
        self.background_net_input_channels = 64
        self.embed_size = 4

        # Initialize model flags
        self.pred_seg = False
        self.use_stylegan_d = False
        self.pred_flip = False
        self.pred_mixing = True
        self.pred_cycle = True
        
        # Initialize preprocessing and networks
        self._setup_preprocessing()
        self._init_networks(training)
        
        # Initialize training components if needed
        if training:
            self._init_discriminators()
            self._init_losses()
            
        # Register coordinate grids for warping
        self._register_coordinate_grids()
        
        # Initialize state tracking
        self.prev_targets = None
        self.thetas_pool = []
        
        # Apply weight initialization
        self.apply(self._weight_init)

        
    def _setup_preprocessing(self):
        """Setup preprocessing functions"""
        self.resize_d = lambda img: F.interpolate(
            img, 
            mode='bilinear',
            size=(224, 224), 
            align_corners=False
        )
        
        self.resize_u = lambda img: F.interpolate(
            img,
            mode='bilinear', 
            size=(256, 256),
            align_corners=False
        )
        
    def register_coordinate_grids(self):
        """Register 2D and 3D coordinate grids as buffers"""
        # 2D identity grid
        grid_s = torch.linspace(-1, 1, self.config.aug_warp_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer(
            'identity_grid_2d',
            torch.stack([u, v], dim=2).view(1, -1, 2),
            persistent=False
        )

        # 3D identity grid
        grid_s = torch.linspace(-1, 1, self.config.latent_volume_size)
        grid_z = torch.linspace(-1, 1, self.config.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer(
            'identity_grid_3d',
            torch.stack([u, v, w, e], dim=3).view(1, -1, 4),
            persistent=False
        )

    def _init_networks(self, training: bool = True):
        """Initialize all network components"""
        # Encoders

        self.local_encoder = LocalEncoder()

        # Initialize face parsing if needed
        if self.config.use_mix_mask:
            self.face_parsing = volumetric_avatar.FaceParsing()
            
        @dataclass
        class volConfig:
            z_dim: int = 16  # Input latent (Z) dimensionality.
            c_dim: int = 96  # Conditioning label (C) dimensionality.
            w_dim: int = 64  # Intermediate latent (W) dimensionality.
            img_resolution: int = 64  # Output resolution.
            dec_channels: int = 1024  # Number of output color channels
            img_channels: int = 384  # Number of output color channels
            features_sigm: int  = 1
            squeeze_dim: int =0 
            depth_resolution: int = 48
            hidden_vol_dec_dim: int = 448
            num_layers_vol_dec: int  = 2

        cfg = volConfig()
        # Volume renderer
        print("cfg:",cfg.squeeze_dim)
        self.volume_renderer = volumetric_avatar.VolumeRenderer(cfg)

            
        # Identity and Expression Embedders
        idtCfg = IdtEmbedConfig()
        self.idt_embedder = IdtEmbed(idtCfg)


      # OR using configuration with head network
        expr_cfg_with_head = volumetric_avatar.ExpressionEmbed.create_default(
            project_dir=self.config.project_dir,
            num_gpus=self.config.num_gpus,
            norm_layer_type=self.config.norm_layer_type,
            use_smart_scale=True
        )
        
        # Initialize expression embedder with chosen configuration
        self.expression_embedder = volumetric_avatar.ExpressionEmbed(expr_cfg_with_head)


        # Warp generators
        warp_gen_config = volumetric_avatar.WarpGenerator.create_default()
        warp_gen_config.input_channels=self.config.dec_max_channels
        self.xy_generator = volumetric_avatar.WarpGenerator(warp_gen_config)
        self.uv_generator = volumetric_avatar.WarpGenerator(warp_gen_config)

        # Initialize 3D processing networks

        vol = volumetric_avatar.Unet3D.create_default()

        self.volume_process = volumetric_avatar.Unet3D(vol)

        # Initialize decoder
        self.decoder = volumetric_avatar.Decoder(
            in_channels=self.config.volume_channels * self.config.volume_depth,
            out_channels=3,
            num_channels=self.config.decoder_channels,
            max_channels=self.config.decoder_max_channels,
            num_blocks=self.config.decoder_blocks,
            predict_segmentation=self.config.decoder_predict_segmentation
        )

    
    def _init_discriminators(self):
        """Initialize discriminator networks for training."""
        if not hasattr(self, 'discriminator'):
            self.discriminator = basic_avatar.MultiScaleDiscriminator(
                input_channels=3,
                num_channels=self.config.discriminator_channels,
                max_channels=self.config.discriminator_max_channels,
                num_blocks=self.config.discriminator_blocks,
                num_scales=self.config.discriminator_scales
            )
            
        if self.use_stylegan_d:
            self.style_discriminator = basic_avatar.DiscriminatorStyleGAN2(
                size=self.config.image_size,
                channel_multiplier=1
            )

    def _register_coordinate_grids(self):
        """Register 2D and 3D coordinate grids as model buffers."""
        # 2D identity grid
        grid_s = torch.linspace(-1, 1, self.config.volume_size)
        v, u = torch.meshgrid(grid_s, grid_s)
        self.register_buffer(
            'identity_grid_2d',
            torch.stack([u, v], dim=2).view(1, -1, 2),
            persistent=False
        )

        # 3D identity grid
        grid_s = torch.linspace(-1, 1, self.config.volume_size)
        grid_z = torch.linspace(-1, 1, self.config.volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.register_buffer(
            'identity_grid_3d',
            torch.stack([u, v, w, e], dim=3).view(1, -1, 4),
            persistent=False
        )

    def _weight_init(self, m):
        """Initialize network weights."""
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def init_losses(self):
        """Initialize loss functions"""
        self.init_additional_losses()

        
    def G_forward(self, data_dict: dict, visualize: bool, iteration: int = 0, epoch: int = 0) -> dict:
        """Main forward pass for generator following EMOPortraits architecture (Section 4).
        
        Implements the full generation pipeline as shown in Figure 4 of the paper:
        1. Face mask generation and preprocessing 
        2. Head pose estimation and warping
        3. Expression and identity embedding
        4. Canonical volume generation  
        5. Target pose warping
        6. Final image synthesis
        
        Args:
            data_dict: Dictionary containing source/target images and metadata
            visualize: Whether to generate visualization outputs
            iteration: Current training iteration 
            epoch: Current training epoch
            
        Returns:
            Updated data_dict with generated images and intermediate outputs
        """
        self.visualize = visualize
        
        # Get basic dimensions
        batch_size = data_dict['source_img'].shape[0]
        channels = self.args.latent_volume_channels
        spatial_size = self.args.latent_volume_size
        depth = self.args.latent_volume_depth

        # 1. Process face masks if enabled
        if self.args.use_mix_mask:
            data_dict = self._process_face_masks(data_dict)

        # 2. Head pose estimation and warping
        data_dict = self._estimate_head_pose(data_dict, batch_size)

        # 3. Generate embeddings
        # Extract features from masked source image
        latent_volume = self.local_encoder(
            data_dict['source_img'] * data_dict['source_mask']
        )
        
        # Get identity and expression embeddings
        data_dict['idt_embed'] = self.idt_embedder_nw(
            data_dict['source_img'] * data_dict['source_mask']
        )
        data_dict = self.expression_embedder_nw(
            data_dict, 
            self.args.estimate_head_pose_from_keypoints,
            self.use_masked_aug
        )

        # Generate warping embeddings
        source_warp_dict, target_warp_dict, mixing_warp_dict, embed_dict = self.predict_embed(
            data_dict
        )

        # 4. Generate canonical volume
        data_dict, canonical_volume = self._create_canonical_volume(
            data_dict=data_dict,
            latent_volume=latent_volume,
            source_warp_dict=source_warp_dict,
            target_warp_dict=target_warp_dict,
            embed_dict=embed_dict,
            iteration=iteration,
            epoch=epoch
        )

        # 5. Generate target warped volume
        aligned_target_volume = self._generate_aligned_volume(
            data_dict=data_dict,
            canonical_volume=canonical_volume,
            target_warp_dict=target_warp_dict,
            batch_size=batch_size,
            channels=channels,
            depth=depth,
            spatial_size=spatial_size
        )

        # 6. Generate final target image
        data_dict = self._generate_target_image(
            data_dict=data_dict,
            target_warp_dict=target_warp_dict,
            aligned_volume=aligned_target_volume,
            iteration=iteration
        )

        # 7. Optional: Handle neutral expression matching
        if self.args.match_neutral:
            data_dict = self._handle_neutral_matching(
                data_dict=data_dict,
                canonical_volume=canonical_volume,
                target_warp_dict=target_warp_dict,
                batch_size=batch_size,
                channels=channels,
                depth=depth,
                spatial_size=spatial_size,
                iteration=iteration,
                epoch=epoch
            )

        # 8. Optional: Handle mixing prediction for training
        if self.pred_mixing or not self.training:
            data_dict = self._handle_mixing_prediction(
                data_dict=data_dict,
                canonical_volume=canonical_volume,
                mixing_warp_dict=mixing_warp_dict,
                batch_size=batch_size,
                channels=channels,
                depth=depth,
                spatial_size=spatial_size,
                iteration=iteration,
                epoch=epoch
            )

        return data_dict

    def _create_canonical_volume(self, data_dict: dict, latent_volume: torch.Tensor,
                            source_warp_dict: dict, target_warp_dict: dict, 
                            embed_dict: dict, iteration: int, epoch: int) -> Tuple[dict, torch.Tensor]:
        """Create expression-free canonical volume following Section 4.2.
        
        Args:
            data_dict: Input data dictionary
            latent_volume: Initial latent volume from encoder
            source_warp_dict: Source warping embeddings
            target_warp_dict: Target warping embeddings
            embed_dict: Combined embedding dictionary
            iteration: Current iteration
            epoch: Current epoch
        
        Returns:
            Tuple of (updated data_dict, canonical_volume)
        """
        # Reshape latent volume
        batch_size = latent_volume.shape[0]
        latent_volume = latent_volume.view(
            batch_size, 
            self.args.latent_volume_channels,
            self.args.latent_volume_depth,
            self.args.latent_volume_size,
            self.args.latent_volume_size
        )

        # Process initial volume if needed
        if self.args.unet_first:
            latent_volume = self.volume_process_nw(latent_volume, embed_dict)
        elif self.args.source_volume_num_blocks > 0:
            latent_volume = self.volume_source_nw(latent_volume)

        # Handle volume detachment
        if self.args.detach_lat_vol > 0 and iteration % self.args.detach_lat_vol == 0:
            latent_volume = latent_volume.detach()

        # Handle processor network freezing
        if self.args.freeze_proc_nw > 0:
            for param in self.volume_process_nw.parameters():
                param.requires_grad = (iteration % self.args.freeze_proc_nw != 0)

        # Warp to canonical pose
        latent_volume = self.grid_sample(
            self.grid_sample(
                latent_volume, 
                data_dict['source_rotation_warp']
            ),
            data_dict['source_xy_warp_resize']
        )

        # Generate canonical volume
        if self.args.unet_first:
            if self.args.source_volume_num_blocks > 0:
                canonical_volume = self.volume_source_nw(latent_volume)
        else:
            canonical_volume = self.volume_process_nw(latent_volume, embed_dict)

        # Add average person tensor if enabled
        if self.args.use_tensor:
            canonical_volume = canonical_volume + self.avarage_tensor_ts

        return data_dict, canonical_volume

    def _generate_aligned_volume(self, data_dict: dict, canonical_volume: torch.Tensor,
                            target_warp_dict: dict, batch_size: int, channels: int,
                            depth: int, spatial_size: int) -> torch.Tensor:
        """Generate target-aligned volume through warping."""
        # Warp to target pose
        aligned_volume = self.grid_sample(
            self.grid_sample(
                canonical_volume,
                data_dict['target_uv_warp_resize']
            ),
            data_dict['target_rotation_warp']
        )
        
        # Process aligned volume if needed
        if self.args.pred_volume_num_blocks > 0:
            aligned_volume = self.volume_pred_nw(aligned_volume)

        # Reshape for decoder
        aligned_volume = aligned_volume.view(batch_size, channels * depth, spatial_size, spatial_size)

        return aligned_volume

    def _generate_target_image(self, data_dict: dict, target_warp_dict: dict,
                            aligned_volume: torch.Tensor, iteration: int) -> dict:
        """Generate final target image from aligned volume."""
        # Generate target image
        data_dict['pred_target_img'], _, _, _ = self.decoder_nw(
            data_dict,
            target_warp_dict,
            aligned_volume,
            False,
            iteration=iteration
        )

        # Apply masking if needed
        if not self.args.use_back:
            data_dict['target_img'] = (
                data_dict['target_img'] * 
                data_dict['target_mask'].detach()
            )
            
            # Add green background if enabled
            if self.args.green:
                green = torch.ones_like(data_dict['target_img']) * (
                    1 - data_dict['target_mask'].detach()
                )
                green[:, 0, :, :] = 0  # Zero red channel
                green[:, 2, :, :] = 0  # Zero blue channel
                data_dict['target_img'] += green

        return data_dict
    
    def _process_face_masks(self, data_dict: dict) -> dict:
        """Process face masks for source and target images.
        
        As described in Section 4 of the paper, face masks help isolate relevant facial regions
        for more accurate expression transfer.
        
        Args:
            data_dict: Dictionary containing source_img, target_img, and associated masks
            
        Returns:
            Updated data_dict with processed masks
        """
        threshold = 0.6
        
        if not self.args.use_ibug_mask:
            # Simple face parsing approach
            face_mask_source, _, _, _ = self.face_idt.forward(data_dict['source_img'])
            face_mask_target, _, _, _ = self.face_idt.forward(data_dict['target_img'])
            face_mask_source = (face_mask_source > threshold).float()
            face_mask_target = (face_mask_target > threshold).float()
            
            data_dict.update({
                'source_mask_modnet': data_dict['source_mask'],
                'target_mask_modnet': data_dict['target_mask'],
                'source_mask_face_pars': face_mask_source.float(),
                'target_mask_face_pars': face_mask_target.float(),
                'source_mask': (data_dict['source_mask'] * face_mask_source).float(),
                'target_mask': (data_dict['target_mask'] * face_mask_target).float()
            })
            return data_dict
            
        # Advanced face parsing with detailed features
        try:
            source_masks = []
            target_masks = []
            
            for i in range(data_dict['source_img'].shape[0]):
                # Get lips and face features
                _, _, source_logits, _ = self.face_parsing_bug.get_lips(data_dict['source_img'][i])
                _, _, target_logits, _ = self.face_parsing_bug.get_lips(data_dict['target_img'][i])
                
                # Detach gradients
                source_logits = source_logits.detach()
                target_logits = target_logits.detach()
                
                # Extract primary face region (index 0)
                source_masks.append(source_logits[:, 0:1])
                target_masks.append(target_logits[:, 0:1])
                
            face_mask_source = torch.cat(source_masks, dim=0)
            face_mask_target = torch.cat(target_masks, dim=0)
            
        except Exception as e:
            print(f"Falling back to basic face parsing: {e}")
            face_mask_source, _, _, _ = self.face_idt.forward(data_dict['source_img'])
            face_mask_target, _, _, _ = self.face_idt.forward(data_dict['target_img'])
        
        # Get additional features like hats
        _, _, hat_source, _ = self.face_idt.forward(data_dict['source_img'])
        _, _, hat_target, _ = self.face_idt.forward(data_dict['target_img'])
        
        face_mask_source += hat_source
        face_mask_target += hat_target
        
        # Process MODNET masks
        data_dict['source_mask_modnet'] = data_dict['source_mask'].clone()
        data_dict['target_mask_modnet'] = data_dict['target_mask'].clone()
        data_dict['source_mask_modnet'][:, :, -256:] *= 0
        data_dict['target_mask_modnet'][:, :, -256:] *= 0
        
        # Combine masks
        face_mask_source = (face_mask_source + data_dict['source_mask_modnet'] >= threshold).float()
        face_mask_target = (face_mask_target + data_dict['target_mask_modnet'] >= threshold).float()
        
        # Store results
        data_dict.update({
            'source_mask_face_pars_1': face_mask_source.float(),
            'target_mask_face_pars_1': face_mask_target.float(),
            'source_mask': (data_dict['source_mask'] * face_mask_source).float(),
            'target_mask': (data_dict['target_mask'] * face_mask_target).float()
        })
        
        return data_dict

    def _estimate_head_pose(self, data_dict: dict, batch_size: int) -> dict:
        """Estimate head pose and generate warping grids.
        
        Implements head pose estimation described in Section 4.2, generating transformation
        matrices and warping grids for source and target poses.
        
        Args:
            data_dict: Dictionary containing source/target images
            batch_size: Batch size for tensor operations
            
        Returns:
            Updated data_dict with pose estimates and warping grids
        """
        # Skip if not using keypoint-based estimation
        if not self.args.estimate_head_pose_from_keypoints:
            # Use 3DMM parameters instead
            data_dict['source_rotation_warp'] = point_transforms.world_to_camera(
                self.identity_grid_3d[..., :3],
                data_dict['source_params_3dmm']
            ).view(batch_size, self.args.latent_volume_depth, 
                self.args.latent_volume_size, self.args.latent_volume_size, 3)
                
            data_dict['target_rotation_warp'] = point_transforms.camera_to_world(
                self.identity_grid_3d[..., :3],
                data_dict['target_params_3dmm']
            ).view(batch_size, self.args.latent_volume_depth,
                self.args.latent_volume_size, self.args.latent_volume_size, 3)
            return data_dict
        
        # Estimate poses
        with torch.no_grad():
            data_dict['source_theta'], source_scale, data_dict['source_rotation'], source_tr = \
                self.head_pose_regressor.forward(data_dict['source_img'], return_srt=True)
            data_dict['target_theta'], target_scale, data_dict['target_rotation'], target_tr = \
                self.head_pose_regressor.forward(data_dict['target_img'], return_srt=True)
        
        # Generate warping grid
        grid = self.identity_grid_3d.repeat_interleave(batch_size, dim=0)
        
        # Process source pose
        inv_source_theta = data_dict['source_theta'].float().inverse().type(data_dict['source_theta'].type())
        data_dict['source_rotation_warp'] = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(
            -1, self.args.latent_volume_depth, self.args.latent_volume_size, 
            self.args.latent_volume_size, 3)
            
        # Compute keypoint transformations
        data_dict['source_warped_keypoints'] = data_dict['source_keypoints'].bmm(inv_source_theta[:, :3, :3])
        data_dict['source_warped_keypoints_n'] = self._normalize_keypoints(
            data_dict['source_warped_keypoints'].clone())
            
        return data_dict

    def _normalize_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Normalize keypoint positions to canonical pose.
        
        Args:
            keypoints: Source keypoints tensor
            
        Returns:
            Normalized keypoints
        """
        # Set canonical nose keypoints
        keypoints[:, 27:31] = torch.tensor([
            [-0.0000, -0.2,  0.22],
            [-0.0000, -0.13, 0.26],
            [-0.0000, -0.06, 0.307],
            [-0.0000, -0.008, 0.310]
        ]).to(keypoints.device)
        
        return keypoints

   

    
    