import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,Bottleneck, resnet18
import torchvision.models as models
import math
import colored_traceback.auto
from torchsummary import summary
from resnet50 import ResNet50
from memory_profiler import profile
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import colored_traceback.auto
import cv2
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
from lpips import LPIPS


from mysixdrepnet import SixDRepNet_Detector
# Set this flag to True for DEBUG mode, False for INFO mode
debug_mode = False

# Configure logging
if debug_mode:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# keep the code in one mega class for copying and pasting into Claude.ai
FEATURE_SIZE_AVG_POOL = 2 # use 2 - not 4. https://github.com/johndpope/MegaPortrait-hack/issues/23
FEATURE_SIZE = (2, 2) 
COMPRESS_DIM = 512 # 🤷 TODO 1: maybe 256 or 512, 512 may be more reasonable for Emtn/app compression

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3)
        self.resblock_128 = ResBlock_Custom(dimension=2, in_channels=64, out_channels=128)
        self.resblock_256 = ResBlock_Custom(dimension=2, in_channels=128, out_channels=256)
        self.resblock_512 = ResBlock_Custom(dimension=2, in_channels=256, out_channels=512)

        # round 0
        self.resblock3D_96 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)

        # round 1
        self.resblock3D_96_1 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)
        self.resblock3D_96_1_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)

        # round 2
        self.resblock3D_96_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)
        self.resblock3D_96_2_2 = ResBlock3D_Adaptive(in_channels=96, out_channels=96)

        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0)

        # Adjusted AvgPool to reduce spatial dimensions effectively
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Second part: producing global descriptor es
        self.custom_resnet50 = CustomResNet50()
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
        es = self.fc(es_flatten) # torch.Size([bs, 2048]) -> torch.Size([bs, 2])        
        # print("es:",es.shape)
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
    """Network for generating flow fields from feature embeddings"""
    def __init__(self):
        super().__init__()
        
        # Initial 1x1 convolution
        self.conv1x1 = nn.Conv2d(512, 2048, kernel_size=1)
        
        # 3D processing branch
        self.resblock1 = ResBlock3D_Adaptive(in_channels=512, out_channels=256)
        self.resblock2 = ResBlock3D_Adaptive(in_channels=256, out_channels=128)
        self.resblock3 = ResBlock3D_Adaptive(in_channels=128, out_channels=64)
        self.resblock4 = ResBlock3D_Adaptive(in_channels=64, out_channels=32)
        
        # Final 3x3x3 convolution
        self.conv3x3x3 = nn.Conv3d(32, 3, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(1, 3)

    def forward(self, zs, adaptive_gamma, adaptive_beta):
        x = self.conv1x1(zs)
        
        # Reshape to 3D volume
        b = x.shape[0]
        x = x.view(b, 512, 4, *x.shape[2:])
        
        # Apply ResBlocks with upsampling
        x = F.interpolate(self.resblock1(x), scale_factor=(2, 2, 2))
        x = F.interpolate(self.resblock2(x), scale_factor=(2, 2, 2))
        x = F.interpolate(self.resblock3(x), scale_factor=(1, 2, 2))
        x = F.interpolate(self.resblock4(x), scale_factor=(1, 2, 2))
        
        # Final convolutions
        x = self.conv3x3x3(x)
        x = self.gn(x)
        x = F.relu(x)
        x = torch.tanh(x)

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
        )
        self.upsampling = nn.Sequential(
            ResBlock3D(768, 384),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(384, 192),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(192, 96),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        self.final_conv = nn.Conv3d(96, 96, kernel_size=3, padding=1)

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
This class, AntiAliasInterpolation2d, is a PyTorch module designed for band-limited downsampling of images, which helps preserve the input signal quality by applying a Gaussian filter before resizing. Here's an intuition breakdown of the code:
This approach ensures that the downsampled image retains more of the original signal's details by reducing high-frequency components that could cause aliasing.
'''
class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka


        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

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
        self.reshape = nn.Conv2d(96, 1536, kernel_size=1)  # Reshape C96xD16 → C1536
        self.conv1x1 = nn.Conv2d(1536, 512, kernel_size=1)  # 1x1 convolution to reduce channels to 512

        self.res_blocks = nn.Sequential(
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        )

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256)
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128)
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64)
        )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

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
        self.head_pose_net = resnet18(pretrained=True)
        self.head_pose_net.fc = nn.Linear(self.head_pose_net.fc.in_features, 6)  # 6 corresponds to rotation and translation parameters
        self.rotation_net =  SixDRepNet_Detector()

        model = resnet18(pretrained=False,num_classes=512)  # 512 feature_maps = resnet18(input_image) ->   Should print: torch.Size([1, 512, 7, 7])
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
    """Warping generator for source-to-canonical transformation"""
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.flowfield = FlowField()
        
        # Adaptive matrices for generating parameters
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(num_channels, num_channels))
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(num_channels, num_channels))

    def forward(self, Rs, ts, zs, es):
        # Validate input shapes
        assert Rs.shape[1] == 3, f"Expected Rs shape (batch_size, 3), got {Rs.shape}"
        assert ts.shape[1] == 3, f"Expected ts shape (batch_size, 3), got {ts.shape}"
        assert zs.shape == es.shape, f"Expected matching shapes for zs and es, got {zs.shape} vs {es.shape}"

        # Combine expression and identity features
        zs_sum = zs + es
        
        # Generate adaptive parameters using learned matrices
        zs_sum = torch.matmul(zs_sum, self.adaptive_matrix_gamma)
        zs_sum = zs_sum.unsqueeze(-1).unsqueeze(-1)

        # Generate warping field using FlowField
        w_em_s2c = self.flowfield(zs_sum, adaptive_gamma=0, adaptive_beta=0)
        
        # Generate rotation/translation warping
        w_rt_s2c = compute_rt_warp(Rs, ts, invert=True, grid_size=64)
        
        # Resize expression warping to match rotation warping
        w_em_s2c_resized = F.interpolate(
            w_em_s2c, 
            size=w_rt_s2c.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Combine both warpings
        w_s2c = w_rt_s2c + w_em_s2c_resized
        return w_s2c


class WarpGeneratorC2D(nn.Module):
    """Warping generator for canonical-to-driving transformation"""
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.flowfield = FlowField()
        
        # Adaptive matrices for generating parameters
        self.adaptive_matrix_gamma = nn.Parameter(torch.randn(num_channels, num_channels))
        self.adaptive_matrix_beta = nn.Parameter(torch.randn(num_channels, num_channels))

    def forward(self, Rd, td, zd, es):
        # Validate input shapes
        assert Rd.shape[1] == 3, f"Expected Rd shape (batch_size, 3), got {Rd.shape}"
        assert td.shape[1] == 3, f"Expected td shape (batch_size, 3), got {td.shape}"
        assert zd.shape == es.shape, f"Expected matching shapes for zd and es, got {zd.shape} vs {es.shape}"

        # Combine expression and identity features
        zd_sum = zd + es
        
        # Generate adaptive parameters using learned matrices
        zd_sum = torch.matmul(zd_sum, self.adaptive_matrix_gamma)
        zd_sum = zd_sum.unsqueeze(-1).unsqueeze(-1)

        # Generate warping field using FlowField
        w_em_c2d = self.flowfield(zd_sum, adaptive_gamma=0, adaptive_beta=0)
        
        # Generate rotation/translation warping
        w_rt_c2d = compute_rt_warp(Rd, td, invert=False, grid_size=64)
        
        # Resize expression warping to match rotation warping
        w_em_c2d_resized = F.interpolate(
            w_em_c2d, 
            size=w_rt_c2d.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Combine both warpings
        w_c2d = w_rt_c2d + w_em_c2d_resized
        return w_c2d


# Function to apply the 3D warping field
def apply_warping_field(v, warp_field):
    """Apply 3D warping field to volume"""
    B, C, D, H, W = v.size()
    device = v.device

    # Resize warp field to match volume dimensions
    warp_field = F.interpolate(
        warp_field, 
        size=(D, H, W), 
        mode='trilinear', 
        align_corners=True
    )

    # Create canonical coordinate grid
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)
    
    # Add batch dimension
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

    # Apply warping field
    warped_grid = grid + warp_field.permute(0, 2, 3, 4, 1)

    # Normalize grid coordinates
    normalization_factors = torch.tensor([W-1, H-1, D-1], device=device)
    warped_grid = 2.0 * warped_grid / normalization_factors - 1.0

    # Apply grid sampling
    v_warped = F.grid_sample(
        v, 
        warped_grid, 
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return v_warped




class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3 - oneshotview
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict





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
from repos.MODNet.src.models.modnet import MODNet



class Gbase(nn.Module):
    def __init__(self, use_mix_losses=True):
        super(Gbase, self).__init__()
        self.use_mix_losses = use_mix_losses
        self.appearanceEncoder = Eapp()
        self.motionEncoder = Emtn()
        self.warp_generator_s2c = WarpGeneratorS2C(num_channels=512)
        self.warp_generator_c2d = WarpGeneratorC2D(num_channels=512)
        self.G3d = G3d(in_channels=96)
        self.G2d = G2d(in_channels=96)
        
        if self.use_mix_losses:
            self.modnet = MODNet(backbone_pretrained=False)
            self.modnet = nn.DataParallel(self.modnet).cuda()
            self.modnet.load_state_dict(torch.load('/media/oem/12TB/EMOPortraits/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
            self.modnet.eval()


    def predict_mixing(self, vs, es, zs, zd, Rs, ts, Rd, td):
        """Generate mixing prediction using rolled expression"""
        # Roll the driving expression vector
        zd_rolled = torch.roll(zd, shifts=1, dims=0)
        
        # Generate warping with rolled expression
        w_c2d_mix = self.warp_generator_c2d(Rd, td, zd_rolled, es)
        vc2d_warped_mix = apply_warping_field(vs, w_c2d_mix)
        vc2d_projected_mix = torch.sum(vc2d_warped_mix, dim=2)
        xhat_mix = self.G2d(vc2d_projected_mix)
        
        return xhat_mix

    def get_mask(self, img):

        im_transform = transforms.Compose(
            [

                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        im = im_transform(img)
        ref_size = 512
        # add mini-batch dim

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self.modnet(im.cuda(), True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

        return matte

    @profile
    def forward(self, xs, xd):
        # Get appearance features
        vs, es = self.appearanceEncoder(xs)
   
        # Get motion parameters
        Rs, ts, zs = self.motionEncoder(xs)
        Rd, td, zd = self.motionEncoder(xd)

        # Generate source-to-canonical warping
        w_s2c = self.warp_generator_s2c(Rs, ts, zs, es)
        vc = apply_warping_field(vs, w_s2c)
        
        # Process canonical volume
        vc2d = self.G3d(vc)
        
        # Generate canonical-to-driving warping
        w_c2d = self.warp_generator_c2d(Rd, td, zd, es)
        vc2d_warped = apply_warping_field(vc2d, w_c2d)
        
        # Generate main output
        vc2d_projected = torch.sum(vc2d_warped, dim=2)
        xhat = self.G2d(vc2d_projected)

        # Prepare output dictionary
        output_dict = {
            'pred_target_img': xhat,
            'source_rotation': Rs,
            'target_rotation': Rd,
            'source_translation': ts,
            'target_translation': td,
            'source_expression': zs,
            'target_expression': zd,
            'source_appearance': es
        }

        # Add mixing-related outputs if enabled
        if self.use_mix_losses and self.training:
            # Generate mixing prediction
            xhat_mix = self.predict_mixing(vs, es, zs, zd, Rs, ts, Rd, td)
            
            # Generate mask for mixing output
            with torch.no_grad():
                mix_mask = self.get_mask(xhat_mix)
                xhat_mix_masked = xhat_mix * mix_mask
            
            # Update dictionary with mixing outputs
            mixing_outputs = {
                'pred_mixing_img': xhat_mix,
                'pred_mixing_mask': mix_mask,
                'pred_mixing_masked_img': xhat_mix_masked,
            }
            output_dict.update(mixing_outputs)

        return output_dict

    def get_mixing_expr_contrastive_data(self, mix_out_dict, xs, xd):
        """Generate data for expression contrastive loss"""
        with torch.no_grad():
            # Get expressions for mixing output
            _, _, z_mix = self.motionEncoder(mix_out_dict['pred_mixing_img'])
            _, _, z_target = self.motionEncoder(xd)
            _, _, z_source = self.motionEncoder(xs)

            mix_out_dict.update({
                'mixing_expr': z_mix,
                'target_expr': z_target,
                'source_expr': z_source
            })

        return mix_out_dict

    def get_cycle_consistency_data(self, mix_out_dict, xs, xd):
        """Generate data for cycle consistency"""
        if not self.use_mix_losses or not self.training:
            return mix_out_dict
            
        with torch.no_grad():
            # Roll mixing prediction
            rolled_mix = torch.roll(mix_out_dict['pred_mixing_img'], shifts=-1, dims=0)
            
            # Get motion for rolled mixing prediction
            Rm, tm, zm = self.motionEncoder(rolled_mix)
            
            # Get target appearance
            vs_target, es_target = self.appearanceEncoder(xd)
            
            # Process through warping sequence
            w_s2c_cycle = self.warp_generator_s2c(Rm, tm, zm, es_target)
            vc_cycle = apply_warping_field(vs_target, w_s2c_cycle)
            vc2d_cycle = self.G3d(vc_cycle)
            
            w_c2d_cycle = self.warp_generator_c2d(Rm, tm, zm, es_target)
            vc2d_warped_cycle = apply_warping_field(vc2d_cycle, w_c2d_cycle)
            vc2d_projected_cycle = torch.sum(vc2d_warped_cycle, dim=2)
            cycle_mix_pred = self.G2d(vc2d_projected_cycle)
            
            # Update dictionary with cycle outputs
            mix_out_dict.update({
                'cycle_mix_pred': cycle_mix_pred,
                'rolled_mix': rolled_mix
            })

        return mix_out_dict




'''
The high-resolution model (Genh) 
The encoder consists of a series of convolutional layers, residual blocks, and average pooling operations to downsample the input image.
The res_blocks section contains multiple residual blocks operating at the same resolution.
The decoder consists of upsampling operations, residual blocks, and a final convolutional layer to generate the high-resolution output.
'''
class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
        )
        self.res_blocks = nn.Sequential(
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

    def unsupervised_loss(self, x, x_hat):
        # Cycle consistency loss
        x_cycle = self.forward(x_hat)
        cycle_loss = F.l1_loss(x_cycle, x)

        # Other unsupervised losses can be added here

        return cycle_loss

    def supervised_loss(self, x_hat, y):
        # Supervised losses
        l1_loss = F.l1_loss(x_hat, y)
        perceptual_loss = self.perceptual_loss(x_hat, y)

        return l1_loss + perceptual_loss


# We load a pre-trained VGG19 network using models.vgg19(pretrained=True) and extract its feature layers using .features. We set the VGG network to evaluation mode and move it to the same device as the input images.
# We define the normalization parameters for the VGG network. The mean and standard deviation values are based on the ImageNet dataset, which was used to train the VGG network. We create tensors for the mean and standard deviation values and move them to the same device as the input images.
# We normalize the input images x and y using the defined mean and standard deviation values. This normalization is necessary to match the expected input format of the VGG network.
# We define the layers of the VGG network to be used for computing the perceptual loss. In this example, we use layers 1, 6, 11, 20, and 29, which correspond to different levels of feature extraction in the VGG network.
# We initialize a variable perceptual_loss to accumulate the perceptual loss values.
# We iterate over the layers of the VGG network using enumerate(vgg). For each layer, we pass the input images x and y through the layer and update their values.
# If the current layer index is in the perceptual_layers list, we compute the perceptual loss for that layer using the L1 loss between the features of x and y. We accumulate the perceptual loss values by adding them to the perceptual_loss variable.
# Finally, we return the computed perceptual loss.

    def perceptual_loss(self, x, y):
        # Load pre-trained VGG network
        vgg = models.vgg19(pretrained=True).features.eval().to(x.device)
        
        # Define VGG normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize input images
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Define perceptual loss layers
        perceptual_layers = [1, 6, 11, 20, 29]
        
        # Initialize perceptual loss
        perceptual_loss = 0.0
        
        # Extract features from VGG network
        for i, layer in enumerate(vgg):
            x = layer(x)
            y = layer(y)
            
            if i in perceptual_layers:
                # Compute perceptual loss for current layer
                perceptual_loss += nn.functional.l1_loss(x, y)
        
        return perceptual_loss

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base = self.Gbase(xs, xd)
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr



'''
In this expanded code, we have the SPADEResBlock class which represents a residual block with SPADE (Spatially-Adaptive Normalization) layers. The block consists of two convolutional layers (conv_0 and conv_1) with normalization layers (norm_0 and norm_1) and a learnable shortcut connection (conv_s and norm_s) if the input and output channels differ.
The SPADE class implements the SPADE layer, which learns to modulate the normalized activations based on the avatar embedding. It consists of a shared convolutional layer (conv_shared) followed by separate convolutional layers for gamma and beta (conv_gamma and conv_beta). The avatar embeddings (avatar_shared_emb, avatar_gamma_emb, and avatar_beta_emb) are learned for each avatar index and are added to the corresponding activations.
During the forward pass of SPADEResBlock, the input x is passed through the shortcut connection and the main branch. The main branch applies the SPADE normalization followed by the convolutional layers. The output of the block is the sum of the shortcut and the main branch activations.
The SPADE layer first normalizes the input x using instance normalization. It then applies the shared convolutional layer to obtain the shared embedding. The gamma and beta values are computed by adding the avatar embeddings to the shared embedding and passing them through the respective convolutional layers. Finally, the normalized activations are modulated using the computed gamma and beta values.
Note that this implementation assumes the presence of the avatar index tensor avatar_index during the forward pass, which is used to retrieve the corresponding avatar embeddings.
'''
class SPADEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_avatars):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_channels, num_avatars)
        self.norm_1 = SPADE(middle_channels, num_avatars)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, num_avatars)

    def forward(self, x, avatar_index):
        x_s = self.shortcut(x, avatar_index)

        dx = self.conv_0(self.actvn(self.norm_0(x, avatar_index)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, avatar_index)))

        out = x_s + dx

        return out

    def shortcut(self, x, avatar_index):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, avatar_index))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, num_avatars):
        super().__init__()
        self.num_avatars = num_avatars
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

        self.avatar_shared_emb = nn.Embedding(num_avatars, 128)
        self.avatar_gamma_emb = nn.Embedding(num_avatars, norm_nc)
        self.avatar_beta_emb = nn.Embedding(num_avatars, norm_nc)

    def forward(self, x, avatar_index):
        avatar_shared = self.avatar_shared_emb(avatar_index)
        avatar_gamma = self.avatar_gamma_emb(avatar_index)
        avatar_beta = self.avatar_beta_emb(avatar_index)

        x = self.norm(x)
        shared_emb = self.conv_shared(x)
        gamma = self.conv_gamma(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        beta = self.conv_beta(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        gamma = gamma + avatar_gamma.view(-1, self.norm_nc, 1, 1)
        beta = beta + avatar_beta.view(-1, self.norm_nc, 1, 1)

        out = x * (1 + gamma) + beta
        return out



'''
The encoder consists of convolutional layers, custom residual blocks, and average pooling operations to downsample the input image.
The SPADE (Spatially-Adaptive Normalization) blocks are applied after the encoder, conditioned on the avatar index.
The decoder consists of custom residual blocks, upsampling operations, and a final convolutional layer to generate the output image.

GDT
'''
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


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.avg_pool(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)
        return input
    

class Student(nn.Module):
    def __init__(self, num_avatars):
        super(Student, self).__init__()
        self.encoder = nn.Sequential(
            ResNet18(),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 96),
            ResBlock(96, 48),
            ResBlock(48, 24),
        )
        self.decoder = nn.Sequential(
            SPADEResBlock(24, 48, num_avatars),
            SPADEResBlock(48, 96, num_avatars),
            SPADEResBlock(96, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
        )
        self.final_layer = nn.Sequential(
            nn.InstanceNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 3, kernel_size=1),
        )

    def forward(self, xd, avatar_index):
        features = self.encoder(xd)
        features = self.decoder(features, avatar_index)
        output = self.final_layer(features)
        return output


'''

Gaze Loss:

The GazeLoss class computes the gaze loss between the output and target frames.
It uses a pre-trained GazeModel to predict the gaze directions for both frames.
The MSE loss is calculated between the predicted gaze directions.


Gaze Model:

The GazeModel class is based on the VGG16 architecture.
It uses the pre-trained VGG16 model as the base and modifies the classifier to predict gaze directions.
The classifier is modified to remove the last layer and add a fully connected layer that outputs a 2-dimensional gaze vector.


Encoder Model:

The Encoder class is a convolutional neural network that encodes the input frames into a lower-dimensional representation.
It consists of a series of convolutional layers with increasing number of filters, followed by batch normalization and ReLU activation.
The encoder performs downsampling at each stage to reduce the spatial dimensions.
The final output is obtained by applying adaptive average pooling and a 1x1 convolutional layer to produce the desired output dimensionality.



The GazeLoss can be used in the PerceptualLoss class to compute the gaze loss component. The Encoder model can be used in the contrastive_loss function to encode the frames into lower-dimensional representations for computing the cosine similarity.
'''
# Gaze Lossimport torch
# from rt_gene.estimate_gaze_pytorch import GazeEstimator
# import os
# class GazeLoss(nn.Module):
#     def __init__(self, model_path='./models/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'):
#         super(GazeLoss, self).__init__()
#         self.gaze_model = GazeEstimator("cuda:0", [os.path.expanduser(model_path)])
#         # self.gaze_model.eval()
#         self.loss_fn = nn.MSELoss()

#     def forward(self, output_frame, target_frame):
#         output_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(output_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(output_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         target_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(target_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(target_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         loss = self.loss_fn(output_gaze, target_gaze)
#         return loss

# Encoder Model
class PatchGanEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(PatchGanEncoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
'''
In this updated code:

We define a GazeBlinkLoss module that combines the gaze and blink prediction tasks.
The module consists of a backbone network (VGG-16), a keypoint network, a gaze prediction head, and a blink prediction head.
The backbone network is used to extract features from the left and right eye images separately. The features are then summed to obtain a combined eye representation.
The keypoint network takes the 2D keypoints as input and produces a latent vector of size 64.
The gaze prediction head takes the concatenated eye features and keypoint features as input and predicts the gaze direction.
The blink prediction head takes only the eye features as input and predicts the blink probability.
The gaze loss is computed using both MAE and MSE losses, weighted by w_mae and w_mse, respectively.
The blink loss is computed using binary cross-entropy loss.
The total loss is the sum of the gaze loss and blink loss.

To train this model, you can follow the training procedure you described:

Use Adam optimizer with the specified hyperparameters.
Train for 60 epochs with a batch size of 64.
Use the one-cycle learning rate schedule.
Treat the predictions from RT-GENE and RT-BENE as ground truth.

Note that you'll need to preprocess your data to provide the left eye image, right eye image, 2D keypoints, target gaze, and target blink for each sample during training.
This code provides a starting point for aligning the MediaPipe-based gaze and blink loss with the approach you described. You may need to make further adjustments based on your specific dataset and requirements.

'''
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class GazeBlinkLoss(nn.Module):
    def __init__(self, device, w_mae=15, w_mse=10):
        super(GazeBlinkLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.w_mae = w_mae
        self.w_mse = w_mse
        
        self.backbone = self._create_backbone()
        self.keypoint_net = self._create_keypoint_net()
        self.gaze_head = self._create_gaze_head()
        self.blink_head = self._create_blink_head()
        
    def _create_backbone(self):
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:1])
        return model
    
    def _create_keypoint_net(self):
        return nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def _create_gaze_head(self):
        return nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def _create_blink_head(self):
        return nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, left_eye, right_eye, keypoints, target_gaze, target_blink):
        # Extract eye features using the backbone
        left_features = self.backbone(left_eye)
        right_features = self.backbone(right_eye)
        eye_features = left_features + right_features
        
        # Extract keypoint features
        keypoint_features = self.keypoint_net(keypoints)
        
        # Predict gaze
        gaze_input = torch.cat((eye_features, keypoint_features), dim=1)
        predicted_gaze = self.gaze_head(gaze_input)
        
        # Predict blink
        predicted_blink = self.blink_head(eye_features)
        
        # Compute gaze loss
        gaze_mae_loss = nn.L1Loss()(predicted_gaze, target_gaze)
        gaze_mse_loss = nn.MSELoss()(predicted_gaze, target_gaze)
        gaze_loss = self.w_mae * gaze_mae_loss + self.w_mse * gaze_mse_loss
        
        # Compute blink loss
        blink_loss = nn.BCEWithLogitsLoss()(predicted_blink, target_blink)
        
        # Total loss
        total_loss = gaze_loss + blink_loss
        
        return total_loss, predicted_gaze, predicted_blink
    


# vanilla gazeloss using mediapipe
class MPGazeLoss(nn.Module):
    def __init__(self, device):
        super(MPGazeLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_gaze, target_gaze, face_image):
        # Ensure face_image has shape (C, H, W)
        if face_image.dim() == 4 and face_image.shape[0] == 1:
            face_image = face_image.squeeze(0)
        if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
            raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
        
        # Convert face image from tensor to numpy array
        face_image = face_image.detach().cpu().numpy()
        if face_image.shape[0] == 3:  # if channels are first
            face_image = face_image.transpose(1, 2, 0)
        face_image = (face_image * 255).astype(np.uint8)

        # Extract eye landmarks using MediaPipe
        results = self.face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return torch.tensor(0.0).to(self.device)

        eye_landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
            eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

        # Compute loss for each eye
        loss = 0.0
        h, w = face_image.shape[:2]
        for left_eye, right_eye in eye_landmarks:
            # Convert landmarks to pixel coordinates
            left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
            right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

            # Create eye mask
            left_mask = torch.zeros((1, h, w)).to(self.device)
            right_mask = torch.zeros((1, h, w)).to(self.device)
            cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
            cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

            # Compute gaze loss for each eye
            left_gaze_loss = self.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
            right_gaze_loss = self.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
            loss += left_gaze_loss + right_gaze_loss

        return loss / len(eye_landmarks)
        
# class Discriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3):
#         super(Discriminator, self).__init__()
        
#         layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
#                   nn.LeakyReLU(0.2, True)]
        
#         for i in range(1, n_layers):
#             layers += [nn.Conv2d(ndf * 2**(i-1), ndf * 2**i, kernel_size=4, stride=2, padding=1),
#                        nn.InstanceNorm2d(ndf * 2**i),
#                        nn.LeakyReLU(0.2, True)]
        
#         layers += [nn.Conv2d(ndf * 2**(n_layers-1), 1, kernel_size=4, stride=1, padding=1)]
        
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class PerceptualLoss(nn.Module):
    def __init__(self, device, weights={'vgg19': 20.0, 'vggface': 5.0, 'gaze': 4.0, 'lpips': 10.0}):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.weights = weights

        # VGG19 network
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg19 = nn.Sequential(*[vgg19[i] for i in range(30)]).to(device).eval()
        self.vgg19_layers = [1, 6, 11, 20, 29]

        # VGGFace network
        self.vggface = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        self.vggface_layers = [4, 5, 6, 7]

        # LPIPS
        self.lpips = LPIPS(net='vgg').to(device).eval()

        # Gaze loss
        self.gaze_loss = MPGazeLoss(device)

    def forward(self, predicted, target, use_fm_loss=False):
        # Normalize input images
        predicted = self.normalize_input(predicted)
        target = self.normalize_input(target)

        # Compute VGG19 perceptual loss
        vgg19_loss = self.compute_vgg19_loss(predicted, target)

        # Compute VGGFace perceptual loss
        vggface_loss = self.compute_vggface_loss(predicted, target)

        # Compute LPIPS loss
        lpips_loss = self.lpips(predicted, target).mean()

        # Compute gaze loss
        # gaze_loss = self.gaze_loss(predicted, target) - broken

        # Compute total perceptual loss
        total_loss = (
            self.weights['vgg19'] * vgg19_loss +
            self.weights['vggface'] * vggface_loss +
            self.weights['lpips'] * lpips_loss +
            self.weights['gaze'] * 1 #gaze_loss
        )

        if use_fm_loss:
            # Compute feature matching loss
            fm_loss = self.compute_feature_matching_loss(predicted, target)
            total_loss += fm_loss

        return total_loss

    def compute_vgg19_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target)

    def compute_vggface_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vggface, self.vggface_layers, predicted, target)

    def compute_feature_matching_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target, detach=True)

    def compute_perceptual_loss(self, model, layers, predicted, target, detach=False):
        loss = 0.0
        predicted_features = predicted
        target_features = target

        for i, layer in enumerate(model.children()):
            if isinstance(layer, nn.Conv2d):
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            elif isinstance(layer, nn.Linear):
                predicted_features = predicted_features.view(predicted_features.size(0), -1)
                target_features = target_features.view(target_features.size(0), -1)
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            else:
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)

            if i in layers:
                if detach:
                    loss += torch.mean(torch.abs(predicted_features - target_features.detach()))
                else:
                    loss += torch.mean(torch.abs(predicted_features - target_features))

        return loss

    def normalize_input(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std




'''
As for driving image, before sending it to Emtn we do a center crop around the face of
a person. Next, we augment it using a random warping based on
thin-plate-splines, which severely degrades the shape of the facial
features, yet keeps the expression intact (ex., it cannot close or open
eyes or change the eyes’ direction). Finally, we apply a severe color
jitter.
'''

from rembg import remove
import io
import os

def remove_background_and_convert_to_rgb(image_tensor):
    """
    Remove the background from an image tensor and convert the image to RGB format.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor.
        
    Returns:
        PIL.Image.Image: Image with background removed and converted to RGB.
    """
    # Convert the tensor to a PIL Image
    image = to_pil_image(image_tensor)
    
    # Remove the background from the image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    bg_removed_bytes = remove(img_byte_arr)
    bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
    
    # Convert the image to RGB format
    bg_removed_image_rgb = bg_removed_image.convert("RGB")
    
    return bg_removed_image_rgb
def crop_and_warp_face(image_tensor, video_name, frame_idx, output_dir="output_images", pad_to_original=False, apply_warping=True,warp_strength=0.05):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the file path
    output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")

    # Check if the file already exists
    if os.path.exists(output_path):
        # Load and return the existing image as a tensor
        existing_image = Image.open(output_path).convert("RGBA")
        return to_tensor(existing_image)
    
    # Check if the input tensor has a batch dimension and handle it
    if image_tensor.ndim == 4:
        # Assuming batch size is the first dimension, process one image at a time
        image_tensor = image_tensor.squeeze(0)
    
    # Convert the single image tensor to a PIL Image
    image = to_pil_image(image_tensor)
    
    # Remove the background from the image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    bg_removed_bytes = remove(img_byte_arr)
    bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
    
    # Convert the image to RGB format to make it compatible with face_recognition
    bg_removed_image_rgb = bg_removed_image.convert("RGB")

    # Detect the face in the background-removed RGB image using the numpy array
    face_locations = face_recognition.face_locations(np.array(bg_removed_image_rgb))

    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]

        # Crop the face region from the image
        face_image = bg_removed_image.crop((left, top, right, bottom))

        # Convert the face image to a numpy array
        face_array = np.array(face_image)

        # Generate random control points for thin-plate-spline warping
        rows, cols = face_array.shape[:2]
        src_points = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = src_points + np.random.randn(4, 2) * (rows * 0.1)

        # Create a PiecewiseAffineTransform object
        tps = PiecewiseAffineTransform()
        tps.estimate(src_points, dst_points)

        # Apply the thin-plate-spline warping to the face image
        warped_face_array = warp(face_array, tps, output_shape=(rows, cols))

        # Convert the warped face array back to a PIL image
        warped_face_image = Image.fromarray((warped_face_array * 255).astype(np.uint8))

        if pad_to_original:
            # Create a new blank image with the same size as the original image
            padded_image = Image.new('RGBA', bg_removed_image.size)

            # Paste the warped face image onto the padded image at the original location
            padded_image.paste(warped_face_image, (left, top))

            # Convert the padded PIL image back to a tensor
            return to_tensor(padded_image)
        else:
            # Convert the warped PIL image back to a tensor
            return to_tensor(warped_face_image)
    else:
        return None

'''
We load the pre-trained DeepLabV3 model using models.segmentation.deeplabv3_resnet101(pretrained=True). This model is based on the ResNet-101 backbone and is pre-trained on the COCO dataset.
We define the necessary image transformations using transforms.Compose. The transformations include converting the image to a tensor and normalizing it using the mean and standard deviation values specific to the model.
We apply the transformations to the input image using transform(image) and add an extra dimension to represent the batch size using unsqueeze(0).
We move the input tensor to the same device as the model to ensure compatibility.
We perform the segmentation by passing the input tensor through the model using model(input_tensor). The output is a dictionary containing the segmentation map.
We obtain the predicted segmentation mask by taking the argmax of the output along the channel dimension using torch.max(output['out'], dim=1).
We convert the segmentation mask to a binary foreground mask by comparing the predicted class labels with the class index representing the person class (assuming it is 15 in this example). The resulting mask will have values of 1 for foreground pixels and 0 for background pixels.
Finally, we return the foreground mask.
'''
def get_foreground_mask(image):
    # Load the pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check if the input is a PyTorch tensor
    if isinstance(image, torch.Tensor):
        # Assume the tensor is already in the range [0, 1]
        if image.dim() == 4 and image.shape[0] == 1:
            # Remove the extra dimension if present
            image = image.squeeze(0)
        input_tensor = transform(image).unsqueeze(0)
    else:
        # Convert PIL Image or NumPy array to tensor
        input_tensor = transforms.ToTensor()(image)
        input_tensor = transform(input_tensor).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor

    # Perform the segmentation
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted segmentation mask
    _, mask = torch.max(output['out'], dim=1)

    # Convert the segmentation mask to a binary foreground mask
    foreground_mask = (mask == 15).float()  # Assuming class 15 represents the person class

    return foreground_mask


from memory_profiler import profile

    
class IdentitySimilarityLoss(nn.Module):
    def __init__(self):
        super(IdentitySimilarityLoss, self).__init__()
        self.face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, Gbase, I3, I4):
        # Generate output with both pose and expression transferred
        full_transfer_output = Gbase(I3, I4)
        
        # Assume the first element of the tuple is the main output
        full_transfer = full_transfer_output[0] if isinstance(full_transfer_output, tuple) else full_transfer_output

        # Ensure inputs are in the correct format (B, C, H, W)
        def prepare_input(x):
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected a tensor, got {type(x)}")
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension if missing
            if x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
            return x.float()  # Ensure float type

        I3 = prepare_input(I3)
        full_transfer = prepare_input(full_transfer)

        # Extract identity features
        with torch.no_grad():
            try:
                id_features_source = self.face_recognition_model(I3)
                id_features_transfer = self.face_recognition_model(full_transfer)
            except RuntimeError as e:
                print(f"Error in face recognition model: {e}")
                print(f"I3 shape: {I3.shape}, full_transfer shape: {full_transfer.shape}")
                raise

        # Compute cosine similarity
        similarity = F.cosine_similarity(id_features_source, id_features_transfer)
        
        # We want to maximize similarity, so we minimize negative similarity
        loss = -similarity.mean()
        
        return loss
    
def make_grid(h, w, device, dtype):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h * w, 2)

    return grid

class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(n_scales):
            self.discriminators.append(self._make_discriminator_layers(in_channels))
    
    def _make_discriminator_layers(self, in_channels):
        """Creates a single-scale patch discriminator that returns intermediate features"""
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [
                nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1, bias=False))
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Store each layer separately to access intermediate features
        return nn.ModuleDict({
            'layer1': discriminator_block(in_channels, 64, normalize=False),
            'layer2': discriminator_block(64, 128),
            'layer3': discriminator_block(128, 256),
            'layer4': discriminator_block(256, 512),
            'layer5': nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, padding=1, bias=False))
        })

    def get_features(self, x):
        """Get intermediate features from all scales for feature matching loss"""
        features = []
        for i in range(self.n_scales):
            if i > 0:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            scale_features = []
            feat = x
            D = self.discriminators[i]
            
            # Collect features from each layer
            for layer in D.values():
                feat = layer(feat)
                scale_features.append(feat)
            
            features.append(scale_features)
        return features

    def forward(self, x):
        """Forward pass returns only final discriminator predictions"""
        predictions = []
        for i in range(self.n_scales):
            if i > 0:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            feat = x
            D = self.discriminators[i]
            for layer in D.values():
                feat = layer(feat)
            predictions.append(feat)
            
        return predictions