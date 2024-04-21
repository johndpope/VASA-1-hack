import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


import json
import os
from math import cos, sin, pi
from typing import List, Tuple, Dict, Any
from camera import Camera
import cv2
import decord
import librosa
import mediapipe as mp
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader,AVReader


from moviepy.editor import VideoFileClip
from PIL import Image
from torch.utils.data import Dataset

from transformers import Wav2Vec2Model, Wav2Vec2Processor


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
        id_feat1 = extract_identity_features(img1)
        id_feat1_cross_id_transfer = extract_identity_features(img1_cross_id_transfer)
        id_feat2 = extract_identity_features(img2)
        id_feat2_cross_id_transfer = extract_identity_features(img2_cross_id_transfer)
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


# The HeadPoseExpressionEstimator takes an input face image and estimates the head pose (yaw, pitch, roll) 
# and expression deformation parameters. It uses a series of residual bottleneck blocks followed by fully connected layers 
# to predict the head pose angles, translation vector, and deformation parameters.
class HeadPoseExpressionEstimator(nn.Module):
    def __init__(self, use_weight_norm=False):
        super(HeadPoseExpressionEstimator, self).__init__()
        self.pre_layers = nn.Sequential(
            ConvBlock2D("CNA", 3, 64, 7, 2, 3, use_weight_norm),
            nn.MaxPool2d(3, 2, 1)
        )
        self.res_layers = nn.Sequential(
            self._make_layer(0, 64, 256, 3, use_weight_norm),
            self._make_layer(1, 256, 512, 4, use_weight_norm),
            self._make_layer(2, 512, 1024, 6, use_weight_norm),
            self._make_layer(3, 1024, 2048, 3, use_weight_norm)
        )
        self.fc_yaw = nn.Linear(2048, 66)
        self.fc_pitch = nn.Linear(2048, 66)
        self.fc_roll = nn.Linear(2048, 66)
        self.fc_t = nn.Linear(2048, 3)
        self.fc_delta = nn.Linear(2048, 3 * 15)

    def _make_layer(self, i, in_channels, out_channels, n_block, use_weight_norm):
        stride = 1 if i == 0 else 2
        return nn.Sequential(
            ResBottleneck(in_channels, out_channels, stride, use_weight_norm),
            *[ResBottleneck(out_channels, out_channels, 1, use_weight_norm) for _ in range(n_block)]
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.res_layers(x)
        x = torch.mean(x, (2, 3))
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        t = self.fc_t(x)
        delta = self.fc_delta(x)
        delta = delta.view(x.shape[0], -1, 3)
        return yaw, pitch, roll, t, delta





