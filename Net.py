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



class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, in_channels, out_channels):
        super().__init__()
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv2d_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.in_channels, self.out_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
    
#    @profile
    def forward(self, x):
        logging.debug(f"ResBlock_Custom > x.shape:  %s",x.shape)
        # logging.debug(f"x:",x)
        
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        # Assertions for shape and values
        assert output.shape[1] == self.out_channels, f"Expected {self.out_channels} channels, got {output.shape[1]}"
        assert output.shape[2] == x.shape[2] and output.shape[3] == x.shape[3], \
            f"Expected spatial dimensions {(x.shape[2], x.shape[3])}, got {(output.shape[2], output.shape[3])}"

        return output



# we need custom resnet blocks - so use the ResNet50  es.shape: torch.Size([1, 512, 1, 1])
# n.b. emoportraits reduced this from 512 -> 128 dim - these are feature maps / identity fingerprint of image 
class CustomResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        resnet = models.resnet50(*args, **kwargs)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
      #  self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # Remove the last residual block (layer4)
        # self.layer4 = resnet.layer4
        
        # Add an adaptive average pooling layer
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE_AVG_POOL)
        
        # Add a 1x1 convolutional layer to reduce the number of channels to 512
        self.conv_reduce = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Remove the forward pass through layer4
        # x = self.layer4(x)
        
        # Apply adaptive average pooling
        x = self.adaptive_avg_pool(x)
        
        # Apply the 1x1 convolutional layer to reduce the number of channels
        x = self.conv_reduce(x)
        
        return x

'''
Eapp Class:

The Eapp class represents the appearance encoder (Eapp) in the diagram.
It consists of two parts: producing volumetric features (vs) and producing a global descriptor (es).

Producing Volumetric Features (vs):

The conv layer corresponds to the 7x7-Conv-64 block in the diagram.
The resblock_128, resblock_256, resblock_512 layers correspond to the ResBlock2D-128, ResBlock2D-256, ResBlock2D-512 blocks respectively, with average pooling (self.avgpool) in between.
The conv_1 layer corresponds to the GN, ReLU, 1x1-Conv2D-1536 block in the diagram.
The output of conv_1 is reshaped to (batch_size, 96, 16, height, width) and passed through resblock3D_96 and resblock3D_96_2, which correspond to the two ResBlock3D-96 blocks in the diagram.
The final output of this part is the volumetric features (vs).

Producing Global Descriptor (es):

The resnet50 layer corresponds to the ResNet50 block in the diagram.
It takes the input image (x) and produces the global descriptor (es).

Forward Pass:

During the forward pass, the input image (x) is passed through both parts of the Eapp network.
The first part produces the volumetric features (vs) by passing the input through the convolutional layers, residual blocks, and reshaping operations.
The second part produces the global descriptor (es) by passing the input through the ResNet50 network.
The Eapp network returns both vs and es as output.

In summary, the Eapp class in the code aligns well with the appearance encoder (Eapp) shown in the diagram. The network architecture follows the same structure, with the corresponding layers and blocks mapped accurately. The conv, resblock_128, resblock_256, resblock_512, conv_1, resblock3D_96, and resblock3D_96_2 layers in the code correspond to the respective blocks in the diagram for producing volumetric features. The resnet50 layer in the code corresponds to the ResNet50 block in the diagram for producing the global descriptor.
'''


class Eapp(nn.Module):
    def __init__(self):
        super().__init__()
        
          # First part: producing volumetric features vs
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3).to(device)
        self.resblock_128 = ResBlock_Custom(dimension=2, in_channels=64, out_channels=128).to(device)
        self.resblock_256 = ResBlock_Custom(dimension=2, in_channels=128, out_channels=256).to(device)
        self.resblock_512 = ResBlock_Custom(dimension=2, in_channels=256, out_channels=512).to(device)

        # round 0
        self.resblock3D_96 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 1
        self.resblock3D_96_1 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_1_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        # round 2
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)
        self.resblock3D_96_2_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96).to(device)

        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0).to(device)

        # Adjusted AvgPool to reduce spatial dimensions effectively
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0).to(device)

        # Second part: producing global descriptor es
        self.custom_resnet50 = CustomResNet50().to(device)
        '''
        ### TODO 2: Change vs/es here for vector size
        According to the description of the paper (Page11: predict the head pose and expression vector), 
        zs should be a global descriptor, which is a vector. Otherwise, the existence of Emtn and Eapp is of little significance. 
        The output feature is a matrix, which means it is basically not compressed. This encoder can be completely replaced by a VAE.
        '''        
        filters = [64, 256, 512, 1024, 2048]
        outputs=COMPRESS_DIM
        self.fc = torch.nn.Linear(filters[4], outputs)       
       
    def forward(self, x):
        # First part
        logging.debug(f"image x: {x.shape}") # [1, 3, 256, 256]
        out = self.conv(x)
        logging.debug(f"After conv: {out.shape}")  # [1, 3, 256, 256]
        out = self.resblock_128(out)
        logging.debug(f"After resblock_128: {out.shape}") # [1, 128, 256, 256]
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_256(out)
        logging.debug(f"After resblock_256: {out.shape}")
        out = self.avgpool(out)
        logging.debug(f"After avgpool: {out.shape}")
        
        out = self.resblock_512(out)
        logging.debug(f"After resblock_512: {out.shape}") # [1, 512, 64, 64]
        out = self.avgpool(out) # at 512x512 image training - we need this  🤷 i rip this out so we can keep things 64x64 - it doesnt align to diagram though
        # logging.debug(f"After avgpool: {out.shape}") # [1, 256, 64, 64]
   
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)
        logging.debug(f"After conv_1: {out.shape}") # [1, 1536, 32, 32]
        
     # reshape 1546 -> C96 x D16
        vs = out.view(out.size(0), 96, 16, *out.shape[2:]) # 🤷 this maybe inaccurate
        logging.debug(f"reshape 1546 -> C96 x D16 : {vs.shape}") 
        
        
        # 1
        vs = self.resblock3D_96(vs)
        logging.debug(f"After resblock3D_96: {vs.shape}") 
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]

        # 2
        vs = self.resblock3D_96_1(vs)
        logging.debug(f"After resblock3D_96_1: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_1_2(vs)
        logging.debug(f"After resblock3D_96_1_2: {vs.shape}")

        # 3
        vs = self.resblock3D_96_2(vs)
        logging.debug(f"After resblock3D_96_2: {vs.shape}") # [1, 96, 16, 32, 32]
        vs = self.resblock3D_96_2_2(vs)
        logging.debug(f"After resblock3D_96_2_2: {vs.shape}")

        # Second part
        es_resnet = self.custom_resnet50(x)
        ### TODO 2
        # print(f"🍌 es:{es_resnet.shape}") # [1, 512, 2, 2]
        es_flatten = torch.flatten(es_resnet, start_dim=1)
        es = self.fc(es_flatten) # torch.Size([bs, 2048]) -> torch.Size([bs, COMPRESS_DIM])        
       
        return vs, es




class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super(AdaptiveGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        normalized = self.group_norm(x)
        return normalized * self.weight + self.bias


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
     

class ResBlock2D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='bilinear', align_corners=False)
        
        return out

class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super().__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_channels)
        self.norm2 = AdaptiveGroupNorm(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

#    @profile
    def forward(self, x):
        residual = x
        logging.debug(f"   🍒 ResBlock3D x.shape:{x.shape}")
        out = self.conv1(x)
        logging.debug(f"   conv1 > out.shape:{out.shape}")
        out = self.norm1(out)
        logging.debug(f"   norm1 > out.shape:{out.shape}")
        out = F.relu(out)
        logging.debug(f"   F.relu(out) > out.shape:{out.shape}")
        out = self.conv2(out)
        logging.debug(f"   conv2 > out.shape:{out.shape}")
        out = self.norm2(out)
        logging.debug(f"   norm2 > out.shape:{out.shape}")
        
        residual = self.residual_conv(residual)
        logging.debug(f"   residual > residual.shape:{residual.shape}",)
        
        out += residual
        out = F.relu(out)

        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out






class FlowField(nn.Module):
    def __init__(self):
        super(FlowField, self).__init__()
        
        self.conv1x1 = nn.Conv2d(512, 2048, kernel_size=1).to(device)


        
        # reshape the tensor from [batch_size, 2048, height, width] to [batch_size, 512, 4, height, width], effectively splitting the channels into a channels dimension of size 512 and a depth dimension of size 4.
        self.reshape_layer = lambda x: x.view(-1, 512, 4, *x.shape[2:]).to(device)

        self.resblock1 = ResBlock3D_Adaptive(in_channels=512, out_channels=256).to(device)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock2 = ResBlock3D_Adaptive( in_channels=256, out_channels=128).to(device)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2)).to(device)
        self.resblock3 =  ResBlock3D_Adaptive( in_channels=128, out_channels=64).to(device)
        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.resblock4 = ResBlock3D_Adaptive( in_channels=64, out_channels=32).to(device)
        self.upsample4 = nn.Upsample(scale_factor=(1, 2, 2)).to(device)
        self.conv3x3x3 = nn.Conv3d(32, 3, kernel_size=3, padding=1).to(device)
        self.gn = nn.GroupNorm(1, 3).to(device)
        self.tanh = nn.Tanh().to(device)
    
#    @profile
    def forward(self, zs,adaptive_gamma, adaptive_beta): # 
       # zs = zs * adaptive_gamma.unsqueeze(-1).unsqueeze(-1) + adaptive_beta.unsqueeze(-1).unsqueeze(-1)
        



        logging.debug(f"FlowField > zs sum.shape:{zs.shape}") #torch.Size([1, 512, 1, 1])
        x = self.conv1x1(zs)
        logging.debug(f"      conv1x1 > x.shape:{x.shape}") #  -> [1, 2048, 1, 1]
        x = self.reshape_layer(x)
        logging.debug(f"      reshape_layer > x.shape:{x.shape}") # -> [1, 512, 4, 1, 1]
        x = self.upsample1(self.resblock1(x))
        logging.debug(f"      upsample1 > x.shape:{x.shape}") # [1, 512, 4, 1, 1]
        x = self.upsample2(self.resblock2(x))
        logging.debug(f"      upsample2 > x.shape:{x.shape}") #[512, 256, 8, 16, 16]
        x = self.upsample3(self.resblock3(x))
        logging.debug(f"      upsample3 > x.shape:{x.shape}")# [512, 128, 16, 32, 32]
        x = self.upsample4(self.resblock4(x))
        logging.debug(f"      upsample4 > x.shape:{x.shape}")
        x = self.conv3x3x3(x)
        logging.debug(f"      conv3x3x3 > x.shape:{x.shape}")
        x = self.gn(x)
        logging.debug(f"      gn > x.shape:{x.shape}")
        x = F.relu(x)
        logging.debug(f"      F.relu > x.shape:{x.shape}")

        x = self.tanh(x)
        logging.debug(f"      tanh > x.shape:{x.shape}")

        # Assertions for shape and values
        assert x.shape[1] == 3, f"Expected 3 channels after conv3x3x3, got {x.shape[1]}"

        return x
 # produce a 3D warping field w𝑠→
    
    

'''
The ResBlock3D class represents a 3D residual block. It consists of two 3D convolutional layers (conv1 and conv2) with group normalization (norm1 and norm2) and ReLU activation. The residual connection is implemented using a shortcut connection.
Let's break down the code:

The init method initializes the layers of the residual block.

conv1 and conv2 are 3D convolutional layers with the specified input and output channels, kernel size of 3, and padding of 1.
norm1 and norm2 are group normalization layers with 32 groups and the corresponding number of channels.
If the input and output channels are different, a shortcut connection is created using a 1x1 convolutional layer and group normalization to match the dimensions.


The forward method defines the forward pass of the residual block.

The input x is stored as the residual.
The input is passed through the first convolutional layer (conv1), followed by group normalization (norm1) and ReLU activation.
The output is then passed through the second convolutional layer (conv2) and group normalization (norm2).
If a shortcut connection exists (i.e., input and output channels are different), the residual is passed through the shortcut connection.
The residual is added to the output of the second convolutional layer.
Finally, ReLU activation is applied to the sum.



The ResBlock3D class can be used as a building block in a larger 3D convolutional neural network architecture. It allows for the efficient training of deep networks by enabling the gradients to flow directly through the shortcut connection, mitigating the vanishing gradient problem.
You can create an instance of the ResBlock3D class by specifying the input and output channels:'''
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, scale_factors=(1, 1, 1)):
        super(ResBlock3D, self).__init__()
        self.upsample = upsample
        self.scale_factors = scale_factors
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factors, mode='trilinear', align_corners=False)
        
        return out
    
    
'''
G3d Class:
- The G3d class represents the 3D convolutional network (G3D) in the diagram.
- It consists of a downsampling path and an upsampling path.

Downsampling Path:
- The downsampling block in the code corresponds to the downsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D average pooling (nn.AvgPool3d) operations.
- The architecture of the downsampling path follows the structure shown in the diagram:
  - ResBlock3D(in_channels, 96) corresponds to the ResBlock3D-96 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-96.
  - ResBlock3D(96, 192) corresponds to the ResBlock3D-192 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 384) corresponds to the ResBlock3D-384 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 768) corresponds to the ResBlock3D-768 block.

Upsampling Path:
- The upsampling block in the code corresponds to the upsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D upsampling (nn.Upsample) operations.
- The architecture of the upsampling path follows the structure shown in the diagram:
  - ResBlock3D(768, 384) corresponds to the ResBlock3D-384 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 192) corresponds to the ResBlock3D-192 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 96) corresponds to the ResBlock3D-96 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-96.

Final Convolution:
- The final_conv layer in the code corresponds to the GN, ReLU, 3x3x3-Conv3D-96 block in the diagram.
- It takes the output of the upsampling path and applies a 3D convolution with a kernel size of 3 and padding of 1 to produce the final output.

Forward Pass:
- During the forward pass, the input tensor x is passed through the downsampling path, then through the upsampling path, and finally through the final convolution layer.
- The output of the G3d network is a tensor of the same spatial dimensions as the input, but with 96 channels.

In summary, the G3d class in the code aligns well with the 3D convolutional network (G3D) shown in the diagram. The downsampling path, upsampling path, and final convolution layer in the code correspond to the respective blocks in the diagram. The ResBlock3D and pooling/upsampling operations are consistent with the diagram, and the forward pass follows the expected flow of data through the network.
'''


class G3d(nn.Module):
    def __init__(self, in_channels):
        super(G3d, self).__init__()
        self.downsampling = nn.Sequential(
            ResBlock3D(in_channels, 96),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(96, 192),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(192, 384),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(384, 768),
        ).to(device)
        self.upsampling = nn.Sequential(
            ResBlock3D(768, 384),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(384, 192),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(192, 96),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        ).to(device)
        self.final_conv = nn.Conv3d(96, 96, kernel_size=3, padding=1).to(device)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        return x


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2D, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
'''
The G2d class consists of the following components:

The input has 96 channels (C96)
The input has a depth dimension of 16 (D16)
The output should have 1536 channels (C1536)

The depth dimension (D16) is present because the input to G2d is a 3D tensor 
(volumetric features) with shape (batch_size, 96, 16, height/4, width/4).
The reshape operation is meant to collapse the depth dimension and increase the number of channels.


The ResBlock2D layers have 512 channels, not 1536 channels as I previously stated. 
The diagram clearly shows 8 ResBlock2D-512 layers before the upsampling blocks that reduce the number of channels.
To summarize, the G2D network takes the orthographically projected 2D feature map from the 3D volumetric features as input.
It first reshapes the number of channels to 512 using a 1x1 convolution layer. 
Then it passes the features through 8 residual blocks (ResBlock2D) that maintain 512 channels. 
This is followed by upsampling blocks that progressively halve the number of channels while doubling the spatial resolution, 
going from 512 to 256 to 128 to 64 channels.
Finally, a 3x3 convolution outputs the synthesized image with 3 color channels.

'''

class G2d(nn.Module):
    def __init__(self, in_channels):
        super(G2d, self).__init__()
        self.reshape = nn.Conv2d(96, 1536, kernel_size=1).to(device)  # Reshape C96xD16 → C1536
        self.conv1x1 = nn.Conv2d(1536, 512, kernel_size=1).to(device)  # 1x1 convolution to reduce channels to 512

        self.res_blocks = nn.Sequential(
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        ).to(device)

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256)
        ).to(device)

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128)
        ).to(device)

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64)
        ).to(device)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        logging.debug(f"G2d > x:{x.shape}")
        x = self.reshape(x)
        x = self.conv1x1(x)  # Added 1x1 convolution to reduce channels to 512
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)
        return x


'''
In this expanded version of compute_rt_warp, we first compute the rotation matrix from the rotation parameters using the compute_rotation_matrix function. The rotation parameters are assumed to be a tensor of shape (batch_size, 3), representing rotation angles in degrees around the x, y, and z axes.
Inside compute_rotation_matrix, we convert the rotation angles from degrees to radians and compute the individual rotation matrices for each axis using the rotation angles. We then combine the rotation matrices using matrix multiplication to obtain the final rotation matrix.
Next, we create a 4x4 affine transformation matrix and set the top-left 3x3 submatrix to the computed rotation matrix. We also set the first three elements of the last column to the translation parameters.
Finally, we create a grid of normalized coordinates using F.affine_grid based on the affine transformation matrix. 
The grid size is assumed to be 64x64x64, but you can adjust it according to your specific requirements.
The resulting grid represents the warping transformations based on the given rotation and translation parameters, which can be used to warp the volumetric features or other tensors.
https://github.com/Kevinfringe/MegaPortrait/issues/4

'''
def compute_rt_warp(rotation, translation, invert=False, grid_size=64):
    """
    Computes the rotation/translation warpings (w_rt).
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        translation (torch.Tensor): The translation vector of shape (batch_size, 3).
        invert (bool): If True, invert the transformation matrix.
        
    Returns:
        torch.Tensor: The resulting transformation grid.
    """
    # Compute the rotation matrix from the rotation parameters
    rotation_matrix = compute_rotation_matrix(rotation)

    # Create a 4x4 affine transformation matrix
    affine_matrix = torch.eye(4, device=rotation.device).repeat(rotation.shape[0], 1, 1)

    # Set the top-left 3x3 submatrix to the rotation matrix
    affine_matrix[:, :3, :3] = rotation_matrix

    # Set the first three elements of the last column to the translation parameters
    affine_matrix[:, :3, 3] = translation

    # Invert the transformation matrix if needed
    if invert:
        affine_matrix = torch.inverse(affine_matrix)

    # # Create a grid of normalized coordinates 
    grid = F.affine_grid(affine_matrix[:, :3], (rotation.shape[0], 1, grid_size, grid_size, grid_size), align_corners=False)
    # # Transpose the dimensions of the grid to match the expected shape
    grid = grid.permute(0, 4, 1, 2, 3)
    return grid

def compute_rotation_matrix(rotation):
    """
    Computes the rotation matrix from rotation angles.
    
    Args:
        rotation (torch.Tensor): The rotation angles (in degrees) of shape (batch_size, 3).
        
    Returns:
        torch.Tensor: The rotation matrix of shape (batch_size, 3, 3).
    """
    # Assumes rotation is a tensor of shape (batch_size, 3), representing rotation angles in degrees
    rotation_rad = rotation * (torch.pi / 180.0)  # Convert degrees to radians

    cos_alpha = torch.cos(rotation_rad[:, 0])
    sin_alpha = torch.sin(rotation_rad[:, 0])
    cos_beta = torch.cos(rotation_rad[:, 1])
    sin_beta = torch.sin(rotation_rad[:, 1])
    cos_gamma = torch.cos(rotation_rad[:, 2])
    sin_gamma = torch.sin(rotation_rad[:, 2])

    # Compute the rotation matrix using the rotation angles
    zero = torch.zeros_like(cos_alpha)
    one = torch.ones_like(cos_alpha)

    R_alpha = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_alpha, -sin_alpha], dim=1),
        torch.stack([zero, sin_alpha, cos_alpha], dim=1)
    ], dim=1)

    R_beta = torch.stack([
        torch.stack([cos_beta, zero, sin_beta], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_beta, zero, cos_beta], dim=1)
    ], dim=1)

    R_gamma = torch.stack([
        torch.stack([cos_gamma, -sin_gamma, zero], dim=1),
        torch.stack([sin_gamma, cos_gamma, zero], dim=1),
        torch.stack([zero, zero, one], dim=1)
    ], dim=1)

    # Combine the rotation matrices
    rotation_matrix = torch.matmul(R_alpha, torch.matmul(R_beta, R_gamma))

    return rotation_matrix



'''
In the updated Emtn class, we use two separate networks (head_pose_net and expression_net) to predict the head pose and expression parameters, respectively.

The head_pose_net is a ResNet-18 model pretrained on ImageNet, with the last fully connected layer replaced to output 6 values (3 for rotation and 3 for translation).
The expression_net is another ResNet-18 model with the last fully connected layer adjusted to output the desired dimensions of the expression vector (e.g., 50).

In the forward method, we pass the input x through both networks to obtain the head pose and expression predictions. We then split the head pose output into rotation and translation parameters.
The Emtn module now returns the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
Note: Make sure to adjust the dimensions of the rotation, translation, and expression parameters according to your specific requirements and the details provided in the MegaPortraits paper.'''
class Emtn(nn.Module):
    def __init__(self):
        super().__init__()
        # https://github.com/johndpope/MegaPortrait-hack/issues/19
        # replace this with off the shelf SixDRepNet
        self.head_pose_net = resnet18(pretrained=True).to(device)
        self.head_pose_net.fc = nn.Linear(self.head_pose_net.fc.in_features, 6).to(device)  # 6 corresponds to rotation and translation parameters
        self.rotation_net =  SixDRepNet_Detector()

        model = resnet18(pretrained=False,num_classes=512).to(device)  # 512 feature_maps = resnet18(input_image) ->   Should print: torch.Size([1, 512, 7, 7])
        # Remove the fully connected layer and the adaptive average pooling layer
        self.expression_net = nn.Sequential(*list(model.children())[:-1])
        self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE)  # https://github.com/neeek2303/MegaPortraits/issues/3
        # self.expression_net.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) #OPTIONAL 🤷 - 16x16 is better?

        ## TODO 2
        outputs=COMPRESS_DIM ## 512,,方便后面的WarpS2C操作 512 -> 2048 channel
        self.fc = torch.nn.Linear(2048, outputs)

    def forward(self, x):
        # Forward pass through head pose network
        rotations,_ = self.rotation_net.predict(x)
        logging.debug(f"📐 rotation :{rotations}")
       

        head_pose = self.head_pose_net(x)

        # Split head pose into rotation and translation parameters
        # rotation = head_pose[:, :3]  - this is shit
        translation = head_pose[:, 3:]


        # Forward pass image through expression network
        expression_resnet = self.expression_net(x)
        ### TODO 2
        expression_flatten = torch.flatten(expression_resnet, start_dim=1)
        expression = self.fc(expression_flatten)  # (bs, 2048) ->>> (bs, COMPRESS_DIM)

        return rotations, translation, expression
    #This encoder outputs head rotations R𝑠/𝑑 ,translations t𝑠/𝑑 , and latent expression descriptors z𝑠/𝑑



'''
Rotation and Translation Warping (𝑤𝑟𝑡_wrt_):

For 𝑤𝑟𝑡→𝑑_wrt_→_d_​: This warping applies a transformation matrix (rotation and translation) to an identity grid.
For 𝑤𝑟𝑡𝑠→_wrts_→​: This warping applies an inverse transformation matrix to an identity grid.


Expression Warping (𝑤𝑒𝑚_wem_​):

Separate warping generators are used for source to canonical (𝑤𝑒𝑚𝑠→_wems_→​) and canonical to driver (𝑤𝑒𝑚→𝑑_wem_→_d_​).
Both warping generators share the same architecture, which includes several 3D residual blocks with Adaptive GroupNorms.
Inputs to these generators are the sums of the expression and appearance descriptors (𝑧𝑠+𝑒𝑠_zs_​+_es_​ for source and 𝑧𝑑+𝑒𝑠_zd_​+_es_​ for driver).
Adaptive parameters are generated by multiplying these sums with learned matrices.

'''
class WarpGeneratorS2C(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorS2C, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        # Adaptive parameters are generated by multiplying these sums with learned matrices.
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rs, ts, zs, es):
        # Assert shapes of input tensors
        assert Rs.shape == (zs.shape[0], 3), f"Expected Rs shape (batch_size, 3), got {Rs.shape}"
        assert ts.shape == (zs.shape[0], 3), f"Expected ts shape (batch_size, 3), got {ts.shape}"
        assert zs.shape == es.shape, f"Expected zs and es to have the same shape, got {zs.shape} and {es.shape}"

        # Sum es with zs
        zs_sum = zs + es

        # Generate adaptive parameters
        # adaptive_gamma = torch.matmul(zs_sum, self.adaptive_matrix_gamma.T)
        # adaptive_beta = torch.matmul(zs_sum, self.adaptive_matrix_beta.T)
        '''
        ### TODO 3: add adaptive_matrix_gamma
        According to the description of the paper (Page11: To generate adaptive parameters, 
        we multiply the foregoing sums and additionally learned matrices for each pair of parameters.), 
        adaptive_matrix_gamma should be retained. It is not used to change the shape, but can generate learning parameters, 
        which is more reasonable than just using sum.
        '''
        zs_sum = torch.matmul(zs_sum, self.adaptive_matrix_gamma) 
        zs_sum = zs_sum.unsqueeze(-1).unsqueeze(-1) ### TODO 3: add unsqueeze(-1).unsqueeze(-1) to match the shape of w_em_s2c

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_s2c = self.flowfield(zs_sum,adaptive_gamma,adaptive_beta) ### TODO 3: flowfield do not need them (adaptive_gamma,adaptive_beta)
        logging.debug(f"w_em_s2c:  :{w_em_s2c.shape}") # 🤷 this is [1, 3, 16, 16, 16] but should it be 16x16 or 64x64?  
        # Compute rotation/translation warping
        w_rt_s2c = compute_rt_warp(Rs, ts, invert=True, grid_size=64)
        logging.debug(f"w_rt_s2c: :{w_rt_s2c.shape}") 
        

        # 🤷 its the wrong dimensions - idk - 
        # Resize w_em_s2c to match w_rt_s2c
        w_em_s2c_resized = F.interpolate(w_em_s2c, size=w_rt_s2c.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_s2c_resized: {w_em_s2c_resized.shape}")
        w_s2c = w_rt_s2c + w_em_s2c_resized

        return w_s2c


class WarpGeneratorC2D(nn.Module):
    def __init__(self, num_channels):
        super(WarpGeneratorC2D, self).__init__()
        self.flowfield = FlowField()
        self.num_channels = COMPRESS_DIM ### TODO 3
        
        # Adaptive parameters are generated by multiplying these sums with learned matrices.
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device) ### TODO 3
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(self.num_channels, self.num_channels)).to(device)

#    @profile
    def forward(self, Rd, td, zd, es):
        # Assert shapes of input tensors
        assert Rd.shape == (zd.shape[0], 3), f"Expected Rd shape (batch_size, 3), got {Rd.shape}"
        assert td.shape == (zd.shape[0], 3), f"Expected td shape (batch_size, 3), got {td.shape}"
        assert zd.shape == es.shape, f"Expected zd and es to have the same shape, got {zd.shape} and {es.shape}"

        # Sum es with zd
        zd_sum = zd + es
        
        # Generate adaptive parameters
        # adaptive_gamma = torch.matmul(zd_sum, self.adaptive_matrix_gamma)
        # adaptive_beta = torch.matmul(zd_sum, self.adaptive_matrix_beta)
        '''
        ### TODO 3: add adaptive_matrix_gamma
        According to the description of the paper (Page11: To generate adaptive parameters, 
        we multiply the foregoing sums and additionally learned matrices for each pair of parameters.), 
        adaptive_matrix_gamma should be retained. It is not used to change the shape, but can generate learning parameters, 
        which is more reasonable than just using sum.
        '''
        zd_sum = torch.matmul(zd_sum, self.adaptive_matrix_gamma) 
        zd_sum = zd_sum.unsqueeze(-1).unsqueeze(-1) ### TODO 3 add unsqueeze(-1).unsqueeze(-1) to match the shape of w_em_c2d

        adaptive_gamma = 0
        adaptive_beta = 0
        w_em_c2d = self.flowfield(zd_sum,adaptive_gamma,adaptive_beta)

        # Compute rotation/translation warping
        w_rt_c2d = compute_rt_warp(Rd, td, invert=False, grid_size=64)

         # Resize w_em_c2d to match w_rt_c2d
        w_em_c2d_resized = F.interpolate(w_em_c2d, size=w_rt_c2d.shape[2:], mode='trilinear', align_corners=False)
        logging.debug(f"w_em_c2d_resized:{w_em_c2d_resized.shape}" )

        w_c2d = w_rt_c2d + w_em_c2d_resized

        return w_c2d


# Function to apply the 3D warping field
def apply_warping_field(v, warp_field):
    B, C, D, H, W = v.size()
    logging.debug(f"🍏 apply_warping_field v:{v.shape}", )
    logging.debug(f"warp_field:{warp_field.shape}" )

    device = v.device

    # Resize warp_field to match the dimensions of v
    warp_field = F.interpolate(warp_field, size=(D, H, W), mode='trilinear', align_corners=True)
    logging.debug(f"Resized warp_field:{warp_field.shape}" )

    # Create a meshgrid for the canonical coordinates
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # Shape: [D, H, W, 3]
    logging.debug(f"Canonical grid:{grid.shape}" )

    # Add batch dimension and repeat the grid for each item in the batch
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Batch grid:{grid.shape}" )

    # Apply the warping field to the grid
    warped_grid = grid + warp_field.permute(0, 2, 3, 4, 1)  # Shape: [B, D, H, W, 3]
    logging.debug(f"Warped grid:{warped_grid.shape}" )

    # Normalize the grid to the range [-1, 1]
    normalization_factors = torch.tensor([W-1, H-1, D-1], device=device)
    logging.debug(f"Normalization factors:{normalization_factors}" )
    warped_grid = 2.0 * warped_grid / normalization_factors - 1.0
    logging.debug(f"Normalized warped grid:{warped_grid.shape}" )

    # Apply grid sampling
    v_canonical = F.grid_sample(v, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
    logging.debug(f"v_canonical:{v_canonical.shape}" )

    return v_canonical




'''
The main changes made to align the code with the training stages are:

Introduced Gbase class that combines the components of the base model.
Introduced Genh class for the high-resolution model.
Introduced GHR class that combines the base model Gbase and the high-resolution model Genh.
Introduced Student class for the student model, which includes an encoder, decoder, and SPADE blocks for avatar conditioning.
Added separate training functions for each stage: train_base, train_hr, and train_student.
Demonstrated the usage of the training functions and saving the trained models.

Note: The code assumes the presence of appropriate dataloaders (dataloader, dataloader_hr, dataloader_avatars) and the implementation of the SPADEResBlock class for the student model. Additionally, the specific training loop details and loss functions need to be implemented based on the paper's description.

The main changes made to align the code with the paper are:

The Emtn (motion encoder) is now a single module that outputs the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
The warping generators (Ws2c and Wc2d) now take the rotation, translation, expression, and appearance features as separate inputs, as described in the paper.
The warping process is updated to match the paper's description. First, the volumetric features (vs) are warped using ws2c to obtain the canonical volume (vc). Then, vc is processed by G3d to obtain vc2d. Finally, vc2d is warped using wc2d to impose the driving motion.
The orthographic projection (denoted as P in the paper) is implemented as an average pooling operation followed by a squeeze operation to reduce the spatial dimensions.
The projected features (vc2d_projected) are passed through G2d to obtain the final output image (xhat).

These changes align the code more closely with the architecture and processing steps described in the MegaPortraits paper.


The volumetric features (vs) obtained from the appearance encoder (Eapp) are warped using the warping field ws2c generated by Ws2c. This warping transforms the volumetric features into the canonical coordinate space, resulting in the canonical volume (vc).
The canonical volume (vc) is then processed by the 3D convolutional network G3d to obtain vc2d.
The vc2d features are warped using the warping field wc2d generated by Wc2d. This warping imposes the driving motion onto the features.
The warped vc2d features are then orthographically projected using average pooling along the depth dimension (denoted as P in the paper). This step reduces the spatial dimensions of the features.
Finally, the projected features (vc2d_projected) are passed through the 2D convolutional network G2d to obtain the final output image (xhat).



'''

import matplotlib.pyplot as plt


class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.appearanceEncoder = Eapp()
        self.motionEncoder = Emtn()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512) # source-to-canonical
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512) # canonical-to-driving 
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)

#    @profile
    def forward(self, xs, xd):
        vs, es = self.appearanceEncoder(xs)
   
        # The motionEncoder outputs head rotations R𝑠/𝑑 ,translations t𝑠/𝑑 , and latent expression descriptors z𝑠/𝑑
        Rs, ts, zs = self.motionEncoder(xs)
        Rd, td, zd = self.motionEncoder(xd)

        logging.debug(f"es shape:{es.shape}")
        logging.debug(f"zs shape:{zs.shape}")


        w_s2c = self.warp_generator_s2c(Rs, ts, zs, es)


        logging.debug(f"vs shape:{vs.shape}") 
        # Warp vs using w_s2c to obtain canonical volume vc
        vc = apply_warping_field(vs, w_s2c)
        assert vc.shape[1:] == (96, 16, 64, 64), f"Expected vc shape (_, 96, 16, 64, 64), got {vc.shape}"

        # Process canonical volume (vc) using G3d to obtain vc2d
        vc2d = self.G3d(vc)

        # Generate warping field w_c2d
        w_c2d = self.warp_generator_c2d(Rd, td, zd, es)
        logging.debug(f"w_c2d shape:{w_c2d.shape}") 

        # Warp vc2d using w_c2d to impose driving motion
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        assert vc2d_warped.shape[1:] == (96, 16, 64, 64), f"Expected vc2d_warped shape (_, 96, 16, 64, 64), got {vc2d_warped.shape}"

        # Perform orthographic projection (P)
        vc2d_projected = torch.sum(vc2d_warped, dim=2)

        # Pass projected features through G2d to obtain the final output image (xhat)
        xhat = self.G2d(vc2d_projected)

        #self.visualize_warp_fields(xs, xd, w_s2c, w_c2d, Rs, ts, Rd, td)
        return xhat

    



    def plot_warp_field(self, ax, warp_field, title, sample_rate=3):
        # Convert the warp field to numpy array
        warp_field_np = warp_field.detach().cpu().numpy()[0]  # Assuming batch size of 1

        # Get the spatial dimensions of the warp field
        depth, height, width = warp_field_np.shape[1:]

        # Create meshgrids for the spatial dimensions with sample_rate
        x = np.arange(0, width, sample_rate)
        y = np.arange(0, height, sample_rate)
        z = np.arange(0, depth, sample_rate)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Extract the x, y, and z components of the warp field with sample_rate
        U = warp_field_np[0, ::sample_rate, ::sample_rate, ::sample_rate]
        V = warp_field_np[1, ::sample_rate, ::sample_rate, ::sample_rate]
        W = warp_field_np[2, ::sample_rate, ::sample_rate, ::sample_rate]

        # Create a mask for positive and negative values
        mask_pos = (U > 0) | (V > 0) | (W > 0)
        mask_neg = (U < 0) | (V < 0) | (W < 0)

        # Set colors for positive and negative values
        color_pos = 'red'
        color_neg = 'blue'

        # Plot the quiver3D for positive values
        ax.quiver3D(X[mask_pos], Y[mask_pos], Z[mask_pos], U[mask_pos], V[mask_pos], W[mask_pos],
                    color=color_pos, length=0.3, normalize=True)

        # Plot the quiver3D for negative values
        ax.quiver3D(X[mask_neg], Y[mask_neg], Z[mask_neg], U[mask_neg], V[mask_neg], W[mask_neg],
                    color=color_neg, length=0.3, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)







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
# Example usage
# guidance_scales = [1.0, 0.5]  # Adjust the scales as needed
# guided_model = ClassifierFreeGuidance(diffusion_model, guidance_scales)

# # During training or sampling
# x = ...  # Input noise or image
# t = ...  # Timestep
# cond = ...  # Conditioning information (e.g., class labels, text embeddings)
# output = guided_model(x, t, cond)

