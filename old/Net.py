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
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from FaceHelper import FaceHelper
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from insightface.app import FaceAnalysis
from torchvision.models import resnet50
from model import CustomResNet50,Eapp,FEATURE_SIZE,COMPRESS_DIM,WarpGeneratorS2C,WarpGeneratorC2D,G3d,G2d,apply_warping_field
from resnet import resnet18


# these classes build from working MegaPortrait https://github.com/johndpope/MegaPortrait-hack
# these may need a re-write
class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.appearanceEncoder = Eapp()
        self.identityEncoder = CustomResNet50()
        self.headPoseEstimator = resnet18(pretrained=True)
        self.headPoseEstimator.fc = nn.Linear(self.headPoseEstimator.fc.in_features, 6)
        self.facialDynamicsEncoder = nn.Sequential(*list(resnet18(pretrained=False, num_classes=512).children())[:-1])
        self.facialDynamicsEncoder.adaptive_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE)
        self.facialDynamicsEncoder.fc = nn.Linear(2048, COMPRESS_DIM)

    def forward(self, x):
        appearance_volume = self.appearanceEncoder(x)[0]  # Get only the appearance volume
        identity_code = self.identityEncoder(x)
        head_pose = self.headPoseEstimator(x)
        rotation = head_pose[:, :3]
        translation = head_pose[:, 3:]
        facial_dynamics_features = self.facialDynamicsEncoder(x) # es
        facial_dynamics = self.facialDynamicsEncoder.fc(torch.flatten(facial_dynamics_features, start_dim=1))
        return appearance_volume, identity_code, rotation, translation, facial_dynamics


class FaceDecoder(nn.Module):
    def __init__(self):
        super(FaceDecoder, self).__init__()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512)
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512)
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

    def forward(self, appearance_volume, identity_code, rotation, translation, facial_dynamics):
        w_s2c = self.warp_generator_s2c(rotation, translation, facial_dynamics, identity_code)
        canonical_volume = apply_warping_field(appearance_volume, w_s2c)
        assert canonical_volume.shape[1:] == (96, 16, 64, 64)

        vc2d = self.G3d(canonical_volume)
        w_c2d = self.warp_generator_c2d(rotation, translation, facial_dynamics, identity_code)
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        assert vc2d_warped.shape[1:] == (96, 16, 64, 64)

        vc2d_projected = torch.sum(vc2d_warped, dim=2)
        xhat = self.G2d(vc2d_projected)
        return xhat



'''
DPE lossesl. For instance, inspired by [ DPE ], we add a pairwise head pose and facial dynamics transfer loss to improve their disentanglement.
'''
class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.identity_extractor = resnet50(pretrained=True)
        self.identity_extractor.fc = nn.Identity()

    def forward(self, x, y):
        x_feats = self.identity_extractor(x)
        y_feats = self.identity_extractor(y)
        return 1 - F.cosine_similarity(x_feats, y_feats, dim=1).mean()

class DPELoss(nn.Module):
    def __init__(self):
        super(DPELoss, self).__init__()
        self.identity_loss = IdentityLoss()
        self.recon_loss = nn.L1Loss()

    def forward(self, I_i, I_j, I_i_pose_j, I_j_pose_i, I_s, I_d, I_s_pose_d_dyn_d):
        # Pairwise head pose and facial dynamics transfer loss
        pose_dyn_loss = self.recon_loss(I_i_pose_j, I_j_pose_i)

        # Face identity similarity loss for cross-identity transfer
        identity_loss = self.identity_loss(I_s, I_s_pose_d_dyn_d)

        return pose_dyn_loss + identity_loss








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

    def forward(self, x, audio_features, gaze_direction, head_distance, emotion_offset, guidance_scale=1.0):
        # Concatenate input features
        input_features = torch.cat([x, audio_features, gaze_direction, head_distance, emotion_offset], dim=-1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(input_features)
        
        x = self.norm(x)
        
        # Apply Classifier-Free Guidance
        if guidance_scale != 1.0:
            uncond_input_features = torch.cat([x, audio_features, torch.zeros_like(gaze_direction), 
                                               torch.zeros_like(head_distance), torch.zeros_like(emotion_offset)], dim=-1)
            uncond_output = self.forward(uncond_input_features, audio_features, gaze_direction, head_distance, emotion_offset, guidance_scale=1.0)
            x = uncond_output + guidance_scale * (x - uncond_output)
        
        return x
    

'''
In this implementation:

The ClassifierFreeGuidance module takes a diffusion model (model) and a list of guidance scales (guidance_scales) as input.
The forward method computes the unconditional model output (unconditional_output) by passing None as the conditioning information.
It then computes the conditional model output (conditional_output) by passing the actual conditioning information (cond).
The Classifier-free Guidance is applied by computing a weighted sum of the difference between the conditional and unconditional outputs, using the provided guidance scales.
The final output is the sum of the weighted difference and the unconditional output.
During training or sampling, you can create an instance of the ClassifierFreeGuidance module with the desired guidance scales and use it like a regular diffusion model. The conditioning information (cond) should be provided based on your specific task (e.g., class labels, text embeddings, or other conditioning signals).

Note that this is a general implementation, and you may need to adjust it based on your specific diffusion model architecture and conditioning requirements.
'''
class ClassifierFreeGuidance(nn.Module):
    def __init__(self, model, guidance_scales):
        super().__init__()
        self.model = model
        self.guidance_scales = guidance_scales

    def forward(self, x, t, cond):
        # Compute the unconditional model output
        unconditional_output = self.model(x, t, None)

        # Compute the conditional model output
        conditional_output = self.model(x, t, cond)

        # Apply Classifier-free Guidance
        guidance_output = torch.zeros_like(unconditional_output)
        for scale in self.guidance_scales:
            guidance_output = guidance_output + scale * (conditional_output - unconditional_output)

        return guidance_output + unconditional_output

