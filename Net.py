import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from math import cos, sin, pi
from typing import List, Tuple, Dict, Any
from camera import Camera
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from FaceHelper import FaceHelper
from PIL import Image
from torch.utils.data import Dataset
from modules.real3d.facev2v_warp.func_utils import transform_kp,make_coordinate_grid_2d
from modules.real3d.facev2v_warp.layers import ConvBlock2D, DownBlock2D, DownBlock3D, UpBlock2D, UpBlock3D, ResBlock2D, ResBlock3D, ResBottleneck
from modules.real3d.facev2v_warp.func_utils import (
    out2heatmap,
    heatmap2kp,
    kp2gaussian_2d,
    create_heatmap_representations,
    create_sparse_motions,
    create_deformed_source_image,
)


class Canonical3DVolumeEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 64, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        return x

class IdentityEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 512, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1)
        return x
    
class HeadPoseEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 3, 1, 1, 0)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1)
        return x

class FacialDynamicsEncoder(nn.Module):
    def __init__(self, model_scale='standard'):
        super().__init__()
        use_weight_norm = False
        down_seq = [3, 64, 128, 256, 512]
        n_res = 2
        self.in_conv = ConvBlock2D("CNA", 3, down_seq[0], 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(*[DownBlock2D(down_seq[i], down_seq[i + 1], use_weight_norm) for i in range(len(down_seq) - 1)])
        self.res = nn.Sequential(*[ResBlock2D(down_seq[-1], use_weight_norm) for _ in range(n_res)])
        self.out_conv = nn.Conv2d(down_seq[-1], 256, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.res(x)
        x = self.out_conv(x)
        x = x.view(x.shape[0], -1) 
        return x


class ExpressiveDisentangledFaceLatentSpace(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.canonical_3d_volume_encoder = Canonical3DVolumeEncoder()
        self.identity_encoder = IdentityEncoder()
        self.head_pose_encoder = HeadPoseEncoder()
        self.facial_dynamics_encoder = FacialDynamicsEncoder()
        
        # Decoder
        self.decoder = Decoder()
        self.fh = FaceHelper()
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.pairwise_transfer_loss = nn.L1Loss()
        self.identity_similarity_loss = nn.CosineSimilarity()

    def forward(self, img1, img2):
        # Extract latent variables for img1
        V_a1 = self.canonical_3d_volume_encoder(img1)
        z_id1 = self.identity_encoder(img1)
        z_pose1 = self.head_pose_encoder(img1)
        z_dyn1 = self.facial_dynamics_encoder(img1)
        
        # Extract latent variables for img2
        V_a2 = self.canonical_3d_volume_encoder(img2)
        z_id2 = self.identity_encoder(img2)
        z_pose2 = self.head_pose_encoder(img2)
        z_dyn2 = self.facial_dynamics_encoder(img2)
        
        # Reconstruct images
        img1_recon = self.decoder(V_a1, z_id1, z_pose1, z_dyn1)
        img2_recon = self.decoder(V_a2, z_id2, z_pose2, z_dyn2)
        
        # Pairwise head pose and facial dynamics transfer
        img1_pose_transfer = self.decoder(V_a1, z_id1, z_pose2, z_dyn1)
        img2_dyn_transfer = self.decoder(V_a2, z_id2, z_pose2, z_dyn1)
        
        # Cross-identity pose and facial motion transfer
        img1_cross_id_transfer = self.decoder(V_a1, z_id2, z_pose1, z_dyn1)
        img2_cross_id_transfer = self.decoder(V_a2, z_id1, z_pose2, z_dyn2)
        
        return img1_recon, img2_recon, img1_pose_transfer, img2_dyn_transfer, img1_cross_id_transfer, img2_cross_id_transfer

    def training_step(self, img1, img2):
        # Forward pass
        img1_recon, img2_recon, img1_pose_transfer, img2_dyn_transfer, img1_cross_id_transfer, img2_cross_id_transfer = self.forward(img1, img2)
        
        # Reconstruction loss
        loss_recon = self.reconstruction_loss(img1_recon, img1) + self.reconstruction_loss(img2_recon, img2)
        
        # Pairwise transfer loss
        loss_pairwise_transfer = self.pairwise_transfer_loss(img1_pose_transfer, img2_dyn_transfer)
        
        # Identity similarity loss

        id_feat1 = self.fh.extract_identity_features(img1)
        id_feat1_cross_id_transfer =  self.fh.extract_identity_features(img1_cross_id_transfer)
        id_feat2 =  self.fh.extract_identity_features(img2)
        id_feat2_cross_id_transfer =  self.fh.extract_identity_features(img2_cross_id_transfer)
        loss_id_sim = 1 - self.identity_similarity_loss(id_feat1, id_feat1_cross_id_transfer) + 1 - self.identity_similarity_loss(id_feat2, id_feat2_cross_id_transfer)
        
        # Total loss
        total_loss = loss_recon + loss_pairwise_transfer + loss_id_sim
        
        return total_loss

class FaceEncoder(nn.Module):
    def __init__(self, use_weight_norm=False):
        super(FaceEncoder, self).__init__()
        # Define encoder architecture for extracting latent variables
        # - Canonical 3D appearance volume (V_can)
        # - Identity code (z_id)
        # - 3D head pose (z_pose)
        # - Facial dynamics code (z_dyn)
        # ...
    
        self.in_conv = ConvBlock2D("CNA", 3, 64, 7, 1, 3, use_weight_norm)
        self.down = nn.Sequential(
            DownBlock2D(64, 128, use_weight_norm),
            DownBlock2D(128, 256, use_weight_norm),
            DownBlock2D(256, 512, use_weight_norm)
        )
        self.mid_conv = nn.Conv2d(512, 32 * 16, 1, 1, 0)
        self.res = nn.Sequential(
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm),
            ResBlock3D(32, use_weight_norm)
        )
        self.identity_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        self.head_pose_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )
        self.facial_dynamics_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down(x)
        x = self.mid_conv(x)
        N, _, H, W = x.shape
        x = x.view(N, 32, 16, H, W)
        appearance_volume = self.res(x)
        
        # Extract identity code, head pose, and facial dynamics code
        x = x.view(N, -1)
        identity_code = self.identity_encoder(x)
        head_pose = self.head_pose_encoder(x)
        facial_dynamics = self.facial_dynamics_encoder(x)
        
        return appearance_volume, identity_code, head_pose, facial_dynamics

class FaceDecoder(nn.Module):
    def __init__(self, use_weight_norm=True):
        super(FaceDecoder, self).__init__()
        self.in_conv = ConvBlock2D("CNA", 32 * 16, 256, 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.res = nn.Sequential(
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 256, 128, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 128, 64, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 64, 3, 7, 1, 3, use_weight_norm, activation_type="tanh")
        )

    def forward(self, appearance_volume, identity_code, head_pose, facial_dynamics):
        N, _, D, H, W = appearance_volume.shape
        x = appearance_volume.view(N, -1, H, W)
        x = self.in_conv(x)
        x = self.res(x)
        
        # Apply 3D warping based on head pose and facial dynamics
        # ...
        
        face_image = self.up(x)
        return face_image

# Define the diffusion transformer for holistic facial dynamics generation
'''
In the provided code snippet for the DiffusionTransformer class, the transformer architecture is implemented using the nn.TransformerEncoderLayer module from PyTorch. The queries, keys, and values are internally computed within each transformer layer based on the input features.

The nn.TransformerEncoderLayer module takes care of computing the queries, keys, and values from the input features using linear transformations. The attention mechanism in the transformer layer then uses these queries, keys, and values to compute the self-attention weights and update the input features.

Here's a breakdown of the transformer architecture in the code:

The DiffusionTransformer class is initialized with the number of layers (num_layers), number of attention heads (num_heads), hidden size (hidden_size), and dropout probability (dropout).
In the __init__ method, the class creates a nn.ModuleList called self.layers, which contains num_layers instances of nn.TransformerEncoderLayer. Each transformer layer has the specified hidden_size, num_heads, and dropout probability.
The forward method takes the input features x, audio_features, gaze_direction, head_distance, and emotion_offset.
The input features are concatenated along the last dimension using torch.cat to form a single tensor input_features.
The concatenated input_features tensor is then passed through each transformer layer in self.layers using a loop. Inside each transformer layer, the following operations are performed:
The input features are linearly transformed to compute the queries, keys, and values.
The attention mechanism computes the self-attention weights using the queries, keys, and values.
The self-attention weights are used to update the input features.
The updated features are passed through a feedforward neural network.
Residual connections and layer normalization are applied.
After passing through all the transformer layers, the output features are normalized using nn.LayerNorm in self.norm(x).
The final output x is returned, which represents the processed features after applying the transformer layers.
The transformer architecture in this code leverages the self-attention mechanism to capture dependencies and relationships among the input features. The queries, keys, and values are internally computed within each transformer layer based on the input features, allowing the model to learn and update the feature representations through the attention mechanism.
'''
class DiffusionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, dropout=0.1):
        super(DiffusionTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, audio_features, gaze_direction, head_distance, emotion_offset):
        # Concatenate input features
        input_features = torch.cat([x, audio_features, gaze_direction, head_distance, emotion_offset], dim=-1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(input_features)
        
        x = self.norm(x)
        return x

# Decoder
'''
we extract the head pose parameters (yaw, pitch, roll, and translation t) and facial dynamics (delta) from the input latent codes z_pose and z_dyn.
We then use the transform_kp function to transform the keypoints based on the head pose and facial dynamics. This function applies the necessary transformations to the canonical 3D volume V_can to obtain the transformed keypoints kp_pose.
Next, we create a 2D coordinate grid using make_coordinate_grid_2d and repeat it for each batch sample. We add the transformed keypoints kp_pose to the grid to obtain the transformed grid coordinates.
Finally, we use F.grid_sample to warp the feature volume x using the transformed grid coordinates. The warped feature volume x_warped is then passed through the upsampling layers to generate the final face image.
Note that you may need to adjust the dimensions and shapes of the tensors based on your specific implementation and the dimensions of V_can and the latent codes.
'''
class Decoder(nn.Module):
    def __init__(self, use_weight_norm=True):
        super(Decoder, self).__init__()
        self.in_conv = ConvBlock2D("CNA", 64, 256, 3, 1, 1, use_weight_norm, nonlinearity_type="leakyrelu")
        self.res = nn.Sequential(
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm),
            ResBlock2D(256, use_weight_norm)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 256, 128, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 128, 64, 3, 1, 1, use_weight_norm),
            nn.Upsample(scale_factor=2),
            ConvBlock2D("CNA", 64, 3, 7, 1, 3, use_weight_norm, activation_type="tanh")
        )

    def forward(self, V_can, z_id, z_pose, z_dyn):
        N, C, D, H, W = V_can.shape
        x = V_can.view(N, -1, H, W)
        x = self.in_conv(x)
        x = self.res(x)

        # Apply 3D warping based on head pose and facial dynamics
        yaw, pitch, roll = z_pose[:, 0], z_pose[:, 1], z_pose[:, 2]
        t = z_pose[:, 3:]
        delta = z_dyn

        # Transform keypoints based on head pose and facial dynamics
        kp_pose, R = transform_kp(V_can, yaw, pitch, roll, t, delta)

        # Warp the feature volume using the transformed keypoints
        grid = make_coordinate_grid_2d(x.shape[2:]).unsqueeze(0).repeat(N, 1, 1, 1).to(x.device)
        grid = grid.view(N, -1, 2)
        kp_pose = kp_pose.view(N, -1, 2)
        grid_transformed = grid + kp_pose
        grid_transformed = grid_transformed.view(N, x.shape[2], x.shape[3], 2)
        x_warped = F.grid_sample(x, grid_transformed, align_corners=True)

        face_image = self.up(x_warped)
        return face_image

