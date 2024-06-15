import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,input_channel = 3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
import torch
from torch import nn
import torch.nn.functional as F
from models.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from models.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)
        # return F.grid_sample(inp, deformation,align_corners = False)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
            output_dict['deformation'] = dense_motion['deformation']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict

from types import SimpleNamespace

import torch


try:
    # from torch.nn import BatchNorm2d as SyncBatchNorm
    from torch.nn import SyncBatchNorm
except ImportError:
    from torch.nn import BatchNorm2d as SyncBatchNorm
from torch import nn
from torch.nn import functional as F
from .conv import LinearBlock, Conv2dBlock, HyperConv2d, PartialConv2dBlock
from .misc import PartialSequential
import sync_batchnorm


class AdaptiveNorm(nn.Module):
    r"""Adaptive normalization layer. The layer first normalizes the input, then
    performs an affine transformation using parameters computed from the
    conditional inputs.
    Args:
        num_features (int): Number of channels in the input tensor.
        cond_dims (int): Number of channels in the conditional inputs.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``, or ``'weight_demod'``.
        projection (bool): If ``True``, project the conditional input to gamma
            and beta using a fully connected layer, otherwise directly use
            the conditional input as gamma and beta.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        input_dim (int): Number of dimensions of the input tensor.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self, num_features, cond_dims, weight_norm_type='',
                 projection=True,
                 separate_projection=False,
                 input_dim=2,
                 activation_norm_type='instance',
                 activation_norm_params=None):
        super().__init__()
        self.projection = projection
        self.separate_projection = separate_projection
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              input_dim,
                                              **vars(activation_norm_params))
        if self.projection:
            if self.separate_projection:
                self.fc_gamma = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
                self.fc_beta = \
                    LinearBlock(cond_dims, num_features,
                                weight_norm_type=weight_norm_type)
            else:
                self.fc = LinearBlock(cond_dims, num_features * 2,
                                      weight_norm_type=weight_norm_type)

        self.conditional = True

    def forward(self, x, y, **kwargs):
        r"""Adaptive Normalization forward.
        Args:
            x (N x C1 x * tensor): Input tensor.
            y (N x C2 tensor): Conditional information.
        Returns:
            out (N x C1 x * tensor): Output tensor.
        """
        if self.projection:
            if self.separate_projection:
                gamma = self.fc_gamma(y)
                beta = self.fc_beta(y)
                for _ in range(x.dim() - gamma.dim()):
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
            else:
                y = self.fc(y)
                for _ in range(x.dim() - y.dim()):
                    y = y.unsqueeze(-1)
                gamma, beta = y.chunk(2, 1)
        else:
            for _ in range(x.dim() - y.dim()):
                y = y.unsqueeze(-1)
            gamma, beta = y.chunk(2, 1)
        x = self.norm(x) if self.norm is not None else x
        out = x * (1 + gamma) + beta
        return out


class SpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.
    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
        separate_projection (bool): If ``True``, we will use two different
            layers for gamma and beta. Otherwise, we will use one layer. It
            matters only if you apply any weight norms to this layer.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
    """

    def __init__(self,
                 num_features,
                 cond_dims,
                 num_filters=128,
                 kernel_size=3,
                 weight_norm_type='',
                 separate_projection=False,
                 activation_norm_type='sync_batch',
                 activation_norm_params=None,
                 partial=False):
        super().__init__()
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace(affine=False)
        padding = kernel_size // 2
        self.separate_projection = separate_projection
        self.mlps = nn.ModuleList()
        self.gammas = nn.ModuleList()
        self.betas = nn.ModuleList()

        # Make cond_dims a list.
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        # Make num_filters a list.
        if not isinstance(num_filters, list):
            num_filters = [num_filters] * len(cond_dims)
        else:
            assert len(num_filters) >= len(cond_dims)

        # Make partial a list.
        if not isinstance(partial, list):
            partial = [partial] * len(cond_dims)
        else:
            assert len(partial) >= len(cond_dims)

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            conv_block = PartialConv2dBlock if partial[i] else Conv2dBlock
            sequential = PartialSequential if partial[i] else nn.Sequential

            if num_filters[i] > 0:
                mlp += [conv_block(cond_dim,
                                   num_filters[i],
                                   kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type,
                                   nonlinearity='relu')]
            mlp_ch = cond_dim if num_filters[i] == 0 else num_filters[i]

            if self.separate_projection:
                if partial[i]:
                    raise NotImplementedError(
                        'Separate projection not yet implemented for ' +
                        'partial conv')
                self.mlps.append(nn.Sequential(*mlp))
                self.gammas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
                self.betas.append(
                    conv_block(mlp_ch, num_features,
                               kernel_size,
                               padding=padding,
                               weight_norm_type=weight_norm_type))
            else:
                mlp += [conv_block(mlp_ch, num_features * 2, kernel_size,
                                   padding=padding,
                                   weight_norm_type=weight_norm_type)]
                self.mlps.append(sequential(*mlp))

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              **vars(activation_norm_params))
        self.conditional = True

    def forward(self, x, *cond_inputs, **kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.
        Args:
            x (N x C1 x H x W tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x) if self.norm is not None else x
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            label_map = F.interpolate(cond_inputs[i], size=x.size()[2:],
                                      mode='nearest')
            if self.separate_projection:
                hidden = self.mlps[i](label_map)
                gamma = self.gammas[i](hidden)
                beta = self.betas[i](hidden)
            else:
                affine_params = self.mlps[i](label_map)
                gamma, beta = affine_params.chunk(2, dim=1)
            output = output * (1 + gamma) + beta
        return output


class HyperSpatiallyAdaptiveNorm(nn.Module):
    r"""Spatially Adaptive Normalization (SPADE) initialization.
    Args:
        num_features (int) : Number of channels in the input tensor.
        cond_dims (int or list of int) : List of numbers of channels
            in the conditional input.
        num_filters (int): Number of filters in SPADE.
        kernel_size (int): Kernel size of the convolutional filters in
            the SPADE layer.
         weight_norm_type (str): Type of weight normalization.
             ``'none'``, ``'spectral'``, or ``'weight'``.
        activation_norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``.
        is_hyper (bool): Whether to use hyper SPADE.
    """

    def __init__(self, num_features, cond_dims,
                 num_filters=0, kernel_size=3,
                 weight_norm_type='',
                 activation_norm_type='sync_batch', is_hyper=True):
        super().__init__()
        padding = kernel_size // 2
        self.mlps = nn.ModuleList()
        if type(cond_dims) != list:
            cond_dims = [cond_dims]

        for i, cond_dim in enumerate(cond_dims):
            mlp = []
            if not is_hyper or (i != 0):
                if num_filters > 0:
                    mlp += [Conv2dBlock(cond_dim, num_filters, kernel_size,
                                        padding=padding,
                                        weight_norm_type=weight_norm_type,
                                        nonlinearity='relu')]
                mlp_ch = cond_dim if num_filters == 0 else num_filters
                mlp += [Conv2dBlock(mlp_ch, num_features * 2, kernel_size,
                                    padding=padding,
                                    weight_norm_type=weight_norm_type)]
                mlp = nn.Sequential(*mlp)
            else:
                if num_filters > 0:
                    raise ValueError('Multi hyper layer not supported yet.')
                mlp = HyperConv2d(padding=padding)
            self.mlps.append(mlp)

        self.norm = get_activation_norm_layer(num_features,
                                              activation_norm_type,
                                              2,
                                              affine=False)

        self.conditional = True

    def forward(self, x, *cond_inputs,
                norm_weights=(None, None), **kwargs):
        r"""Spatially Adaptive Normalization (SPADE) forward.
        Args:
            x (4D tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional maps for SPADE.
            norm_weights (5D tensor or list of tensors): conv weights or
            [weights, biases].
        Returns:
            output (4D tensor) : Output tensor.
        """
        output = self.norm(x)
        for i in range(len(cond_inputs)):
            if cond_inputs[i] is None:
                continue
            if type(cond_inputs[i]) == list:
                cond_input, mask = cond_inputs[i]
                mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear',
                                     align_corners=False)
            else:
                cond_input = cond_inputs[i]
                mask = None
            label_map = F.interpolate(cond_input, size=x.size()[2:])
            if norm_weights is None or norm_weights[0] is None or i != 0:
                affine_params = self.mlps[i](label_map)
            else:
                affine_params = self.mlps[i](label_map,
                                             conv_weights=norm_weights)
            gamma, beta = affine_params.chunk(2, dim=1)
            if mask is not None:
                gamma = gamma * (1 - mask)
                beta = beta * (1 - mask)
            output = output * (1 + gamma) + beta
        return output


class LayerNorm2d(nn.Module):
    r"""Layer Normalization as introduced in
    https://arxiv.org/abs/1607.06450.
    This is the usual way to apply layer normalization in CNNs.
    Note that unlike the pytorch implementation which applies per-element
    scale and bias, here it applies per-channel scale and bias, similar to
    batch/instance normalization.
    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional, default=1e-5): a value added to the
            denominator for numerical stability.
        affine (bool, optional, default=False): If ``True``, performs
            affine transformation after normalization.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        r"""
        Args:
            x (tensor): Input tensor.
        """
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def get_activation_norm_layer(num_features, norm_type,
                              input_dim, **norm_params):
    r"""Return an activation normalization layer.
    Args:
        num_features (int): Number of feature channels.
        norm_type (str):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        input_dim (int): Number of input dimensions.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the activation normalization.
    """
    input_dim = max(input_dim, 1)  # Norm1d works with both 0d and 1d inputs

    if norm_type == 'none' or norm_type == '':
        norm_layer = None
    elif norm_type == 'batch':
        # norm = getattr(nn, 'BatchNorm%dd' % input_dim)
        norm = getattr(sync_batchnorm, 'SynchronizedBatchNorm%dd' % input_dim)
        norm_layer = norm(num_features, **norm_params)
    elif norm_type == 'instance':
        affine = norm_params.pop('affine', True)  # Use affine=True by default
        norm = getattr(nn, 'InstanceNorm%dd' % input_dim)
        norm_layer = norm(num_features, affine=affine, **norm_params)
    elif norm_type == 'sync_batch':
        # There is a bug of using amp O1 with synchronize batch norm.
        # The lines below fix it.
        affine = norm_params.pop('affine', True)
        # Always call SyncBN with affine=True
        norm_layer = SyncBatchNorm(num_features, affine=True, **norm_params)
        norm_layer.weight.requires_grad = affine
        norm_layer.bias.requires_grad = affine
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(num_features, **norm_params)
    elif norm_type == 'layer_2d':
        norm_layer = LayerNorm2d(num_features, **norm_params)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_channels=num_features, **norm_params)
    elif norm_type == 'adaptive':
        norm_layer = AdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = SpatiallyAdaptiveNorm(num_features, **norm_params)
    elif norm_type == 'hyper_spatially_adaptive':
        if input_dim != 2:
            raise ValueError('Spatially adaptive normalization layers '
                             'only supports 2D input')
        norm_layer = HyperSpatiallyAdaptiveNorm(num_features, **norm_params)
    else:
        raise ValueError('Activation norm layer %s '
                         'is not recognized' % norm_type)
    return norm_layer
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from torch import nn


def get_nonlinearity_layer(nonlinearity_type, inplace):
    r"""Return a nonlinearity layer.

    Args:
        nonlinearity_type (str):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace (bool): If ``True``, set ``inplace=True`` when initializing
            the nonlinearity layer.
    """
    if nonlinearity_type == 'relu':
        nonlinearity = nn.ReLU(inplace=inplace)
    elif nonlinearity_type == 'leakyrelu':
        nonlinearity = nn.LeakyReLU(0.2, inplace=inplace)
    elif nonlinearity_type == 'prelu':
        nonlinearity = nn.PReLU()
    elif nonlinearity_type == 'tanh':
        nonlinearity = nn.Tanh()
    elif nonlinearity_type == 'sigmoid':
        nonlinearity = nn.Sigmoid()
    elif nonlinearity_type.startswith('softmax'):
        dim = nonlinearity_type.split(',')[1] if ',' in nonlinearity_type else 1
        nonlinearity = nn.Softmax(dim=int(dim))
    elif nonlinearity_type == 'none' or nonlinearity_type == '':
        nonlinearity = None
    else:
        raise ValueError('Nonlinearity %s is not recognized' %
                         nonlinearity_type)
    return nonlinearity

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial

import torch
import torch.nn as nn

from imaginaire.layers import Conv2dBlock


class NonLocal2dBlock(nn.Module):
    r"""Self attention Layer

    Args:
        in_channels (int): Number of channels in the input tensor.
        scale (bool, optional, default=True): If ``True``, scale the
            output by a learnable parameter.
        clamp (bool, optional, default=``False``): If ``True``, clamp the
            scaling parameter to (-1, 1).
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
    """

    def __init__(self,
                 in_channels,
                 scale=True,
                 clamp=False,
                 weight_norm_type='none'):
        super(NonLocal2dBlock, self).__init__()
        self.clamp = clamp
        self.gamma = nn.Parameter(torch.zeros(1)) if scale else 1.0
        self.in_channels = in_channels
        base_conv2d_block = partial(Conv2dBlock,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    weight_norm_type=weight_norm_type)
        self.theta = base_conv2d_block(in_channels, in_channels // 8)
        self.phi = base_conv2d_block(in_channels, in_channels // 8)
        self.g = base_conv2d_block(in_channels, in_channels // 2)
        self.out_conv = base_conv2d_block(in_channels // 2, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        r"""

        Args:
            x (tensor) : input feature maps (B X C X W X H)
        Returns:
            (tuple):
              - out (tensor) : self attention value + input feature
              - attention (tensor): B x N x N (N is Width*Height)
        """
        n, c, h, w = x.size()
        theta = self.theta(x).view(n, -1, h * w).permute(0, 2, 1)

        phi = self.phi(x)
        phi = self.max_pool(phi).view(n, -1, h * w // 4)

        energy = torch.bmm(theta, phi)
        attention = self.softmax(energy)

        g = self.g(x)
        g = self.max_pool(g).view(n, -1, h * w // 4)

        out = torch.bmm(g, attention.permute(0, 2, 1))
        out = out.view(n, c // 2, h, w)
        out = self.out_conv(out)

        if self.clamp:
            out = self.gamma.clamp(-1, 1) * out + x
        else:
            out = self.gamma * out + x
        return out

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

import torch
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from .conv import LinearBlock


class WeightDemodulation(nn.Module):
    r"""Weight demodulation in
    "Analyzing and Improving the Image Quality of StyleGAN", Karras et al.

    Args:
        conv (torch.nn.Modules): Convolutional layer.
        cond_dims (int): The number of channels in the conditional input.
        eps (float, optional, default=1e-8): a value added to the
            denominator for numerical stability.
        adaptive_bias (bool, optional, default=False): If ``True``, adaptively
            predicts bias from the conditional input.
        demod (bool, optional, default=False): If ``True``, performs
            weight demodulation.
    """

    def __init__(self, conv, cond_dims, eps=1e-8,
                 adaptive_bias=False, demod=True):
        super().__init__()
        self.conv = conv
        self.adaptive_bias = adaptive_bias
        if adaptive_bias:
            self.conv.register_parameter('bias', None)
            self.fc_beta = LinearBlock(cond_dims, self.conv.out_channels)
        self.fc_gamma = LinearBlock(cond_dims, self.conv.in_channels)
        self.eps = eps
        self.demod = demod
        self.conditional = True

    def forward(self, x, y):
        r"""Weight demodulation forward"""
        b, c, h, w = x.size()
        self.conv.groups = b
        gamma = self.fc_gamma(y)
        gamma = gamma[:, None, :, None, None]
        weight = self.conv.weight[None, :, :, :, :] * (gamma + 1)

        if self.demod:
            d = torch.rsqrt(
                (weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.conv.out_channels, *ws)
        x = self.conv.conv2d_forward(x, weight)

        x = x.reshape(-1, self.conv.out_channels, h, w)
        if self.adaptive_bias:
            x += self.fc_beta(y)[:, :, None, None]
        return x


def weight_demod(conv, cond_dims=256, eps=1e-8, demod=True):
    r"""Weight demodulation."""
    return WeightDemodulation(conv, cond_dims, eps, demod)


def get_weight_norm_layer(norm_type, **norm_params):
    r"""Return weight normalization.

    Args:
        norm_type (str):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        norm_params: Arbitrary keyword arguments that will be used to
            initialize the weight normalization.
    """
    if norm_type == 'none' or norm_type == '':  # no normalization
        return lambda x: x
    elif norm_type == 'spectral':  # spectral normalization
        return functools.partial(spectral_norm, **norm_params)
    elif norm_type == 'weight':  # weight normalization
        return functools.partial(weight_norm, **norm_params)
    elif norm_type == 'weight_demod':  # weight demodulation
        return functools.partial(weight_demod, **norm_params)
    else:
        raise ValueError(
            'Weight norm layer %s is not recognized' % norm_type)

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import functools

from torch import nn
from torch.nn import Upsample as NearestUpsample
from torch.utils.checkpoint import checkpoint

from .conv import (Conv1dBlock, Conv2dBlock, Conv3dBlock, HyperConv2dBlock,
                   LinearBlock, MultiOutConv2dBlock, PartialConv2dBlock,
                   PartialConv3dBlock)


class _BaseResBlock(nn.Module):
    r"""An abstract class for residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        super().__init__()
        if order == 'pre_act':
            order = 'NACNAC'
        if isinstance(bias, bool):
            # The bias for conv_block_0, conv_block_1, and conv_block_s.
            biases = [bias, bias, bias]
        elif isinstance(bias, list):
            if len(bias) == 3:
                biases = bias
            else:
                raise ValueError('Bias list must be 3.')
        else:
            raise ValueError('Bias must be either an integer or s list.')
        self.learn_shortcut = (in_channels != out_channels) or learn_shortcut
        if len(order) > 6 or len(order) < 5:
            raise ValueError('order must be either 5 or 6 characters')
        if hidden_channels_equal_out_channels:
            hidden_channels = out_channels
        else:
            hidden_channels = min(in_channels, out_channels)

        # Parameters that are specific for convolutions.
        conv_main_params = {}
        conv_skip_params = {}
        if block != LinearBlock:
            conv_base_params = dict(stride=1, dilation=dilation,
                                    groups=groups, padding_mode=padding_mode)
            conv_main_params.update(conv_base_params)
            conv_main_params.update(
                dict(kernel_size=kernel_size,
                     activation_norm_type=activation_norm_type,
                     activation_norm_params=activation_norm_params,
                     padding=padding))
            conv_skip_params.update(conv_base_params)
            conv_skip_params.update(dict(kernel_size=1))
            if skip_activation_norm:
                conv_skip_params.update(
                    dict(activation_norm_type=activation_norm_type,
                         activation_norm_params=activation_norm_params))

        # Other parameters.
        other_params = dict(weight_norm_type=weight_norm_type,
                            weight_norm_params=weight_norm_params,
                            apply_noise=apply_noise)

        # Residual branch.
        if order.find('A') < order.find('C') and \
                (activation_norm_type == '' or activation_norm_type == 'none'):
            # Nonlinearity is the first operation in the residual path.
            # In-place nonlinearity will modify the input variable and cause
            # backward error.
            first_inplace = False
        else:
            first_inplace = inplace_nonlinearity
        self.conv_block_0 = block(in_channels, hidden_channels,
                                  bias=biases[0],
                                  nonlinearity=nonlinearity,
                                  order=order[0:3],
                                  inplace_nonlinearity=first_inplace,
                                  **conv_main_params,
                                  **other_params)
        self.conv_block_1 = block(hidden_channels, out_channels,
                                  bias=biases[1],
                                  nonlinearity=nonlinearity,
                                  order=order[3:],
                                  inplace_nonlinearity=inplace_nonlinearity,
                                  **conv_main_params,
                                  **other_params)

        # Shortcut branch.
        if self.learn_shortcut:
            if skip_nonlinearity:
                skip_nonlinearity_type = nonlinearity
            else:
                skip_nonlinearity_type = ''
            self.conv_block_s = block(in_channels, out_channels,
                                      bias=biases[2],
                                      nonlinearity=skip_nonlinearity_type,
                                      order=order[0:3],
                                      **conv_skip_params,
                                      **other_params)

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(self.conv_block_0, 'conditional', False) or \
            getattr(self.conv_block_1, 'conditional', False)

    def conv_blocks(self, x, *cond_inputs, **kw_cond_inputs):
        r"""Returns the output of the residual branch.

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            dx (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, **kw_cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs, **kw_cond_inputs)
        return dx

    def forward(self, x, *cond_inputs, do_checkpoint=False, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            do_checkpoint (bool, optional, default=``False``) If ``True``,
                trade compute for memory by checkpointing the model.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        if do_checkpoint:
            dx = checkpoint(self.conv_blocks, x, *cond_inputs, **kw_cond_inputs)
        else:
            dx = self.conv_blocks(x, *cond_inputs, **kw_cond_inputs)

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs, **kw_cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output


class ResLinearBlock(_BaseResBlock):
    r"""Residual block with full-connected layers.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, None, None,
                         None, None, bias, None,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, LinearBlock, learn_shortcut)


class Res1dBlock(_BaseResBlock):
    r"""Residual block for 1D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv1dBlock, learn_shortcut)


class Res2dBlock(_BaseResBlock):
    r"""Residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv2dBlock, learn_shortcut)


class Res3dBlock(_BaseResBlock):
    r"""Residual block for 3D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv3dBlock, learn_shortcut)


class _BaseHyperResBlock(_BaseResBlock):
    r"""An abstract class for hyper residual blocks.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 hidden_channels_equal_out_channels,
                 order,
                 is_hyper_conv, is_hyper_norm, block, learn_shortcut):
        block = functools.partial(block,
                                  is_hyper_conv=is_hyper_conv,
                                  is_hyper_norm=is_hyper_norm)
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

    def forward(self, x, *cond_inputs, conv_weights=(None,) * 3,
                norm_weights=(None,) * 3, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            conv_weights (list of tensors): Convolution weights for
                three convolutional layers respectively.
            norm_weights (list of tensors): Normalization weights for
                three convolutional layers respectively.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            output (tensor): Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs, conv_weights=conv_weights[0],
                               norm_weights=norm_weights[0])
        dx = self.conv_block_1(dx, *cond_inputs, conv_weights=conv_weights[1],
                               norm_weights=norm_weights[1])
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs,
                                           conv_weights=conv_weights[2],
                                           norm_weights=norm_weights[2])
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output


class HyperRes2dBlock(_BaseHyperResBlock):
    r"""Hyper residual block for 2D input.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='', weight_norm_params=None,
                 activation_norm_type='', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', is_hyper_conv=False, is_hyper_norm=False,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, is_hyper_conv, is_hyper_norm,
                         HyperConv2dBlock, learn_shortcut)


class _BaseDownResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with downsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, pooling, down_factor, learn_shortcut):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)
        self.pooling = pooling(down_factor)

    def forward(self, x, *cond_inputs):
        r"""

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        dx = self.conv_block_0(x, *cond_inputs)
        dx = self.conv_block_1(dx, *cond_inputs)
        dx = self.pooling(dx)
        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        x_shortcut = self.pooling(x_shortcut)
        output = x_shortcut + dx
        return output


class DownRes2dBlock(_BaseDownResBlock):
    r"""Residual block for 2D input with downsampling.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        pooling (class, optional, default=nn.AvgPool2d): Pytorch pooling
            layer to be used.
        down_factor (int, optional, default=2): Downsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', pooling=nn.AvgPool2d, down_factor=2,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, Conv2dBlock, pooling,
                         down_factor, learn_shortcut)


class _BaseUpResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, upsample, up_factor, learn_shortcut):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)
        self.order = order
        self.upsample = upsample(scale_factor=up_factor)

    def forward(self, x, *cond_inputs):
        r"""Implementation of the up residual block forward function.
        If the order is 'NAC' for the first residual block, we will first
        do the activation norm and nonlinearity, in the original resolution.
        We will then upsample the activation map to a higher resolution. We
        then do the convolution.
        It is is other orders, then we first do the whole processing and
        then upsample.

        Args:
            x (tensor) : Input tensor.
            cond_inputs (list of tensors) : Conditional input.
        Returns:
            output (tensor) : Output tensor.
        """
        # In this particular upsample residual block operation, we first
        # upsample the skip connection.
        if self.learn_shortcut:
            x_shortcut = self.upsample(x)
            x_shortcut = self.conv_block_s(x_shortcut, *cond_inputs)
        else:
            x_shortcut = self.upsample(x)

        if self.order[0:3] == 'NAC':
            for ix, layer in enumerate(self.conv_block_0.layers.values()):
                if getattr(layer, 'conditional', False):
                    x = layer(x, *cond_inputs)
                else:
                    x = layer(x)
                if ix == 1:
                    x = self.upsample(x)
        else:
            x = self.conv_block_0(x, *cond_inputs)
            x = self.upsample(x)
        x = self.conv_block_1(x, *cond_inputs)

        output = x_shortcut + x
        return output


class UpRes2dBlock(_BaseUpResBlock):
    r"""Residual block for 2D input with downsampling.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        upsample (class, optional, default=NearestUpsample): PPytorch
            upsampling layer to be used.
        up_factor (int, optional, default=2): Upsampling factor.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', upsample=NearestUpsample, up_factor=2,
                 learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, Conv2dBlock,
                         upsample, up_factor, learn_shortcut)


class _BasePartialResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks with partial convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 multi_channel, return_mask,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        block = functools.partial(block,
                                  multi_channel=multi_channel,
                                  return_mask=return_mask)
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - output (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        if self.conv_block_0.layers.conv.return_mask:
            dx, mask_out = self.conv_block_0(x, *cond_inputs,
                                             mask_in=mask_in, **kw_cond_inputs)
            dx, mask_out = self.conv_block_1(dx, *cond_inputs,
                                             mask_in=mask_out, **kw_cond_inputs)
        else:
            dx = self.conv_block_0(x, *cond_inputs,
                                   mask_in=mask_in, **kw_cond_inputs)
            dx = self.conv_block_1(dx, *cond_inputs,
                                   mask_in=mask_in, **kw_cond_inputs)
            mask_out = None

        if self.learn_shortcut:
            x_shortcut = self.conv_block_s(x, mask_in=mask_in, *cond_inputs,
                                           **kw_cond_inputs)
            if type(x_shortcut) == tuple:
                x_shortcut, _ = x_shortcut
        else:
            x_shortcut = x
        output = x_shortcut + dx

        if mask_out is not None:
            return output, mask_out
        return output


class PartialRes2dBlock(_BasePartialResBlock):
    r"""Residual block for 2D input with partial convolution.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False,
                 hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, PartialConv2dBlock, learn_shortcut)


class PartialRes3dBlock(_BasePartialResBlock):
    r"""Residual block for 3D input with partial convolution.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, PartialConv3dBlock, learn_shortcut)


class _BaseMultiOutResBlock(_BaseResBlock):
    r"""An abstract class for residual blocks that can returns multiple outputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 skip_activation_norm, skip_nonlinearity,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, hidden_channels_equal_out_channels,
                 order, block, learn_shortcut):
        self.multiple_outputs = True
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         hidden_channels_equal_out_channels,
                         order, block, learn_shortcut)

    def forward(self, x, *cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
        Returns:
            (tuple):
              - output (tensor): Output tensor.
              - aux_outputs_0 (tensor): Auxiliary output of the first block.
              - aux_outputs_1 (tensor): Auxiliary output of the second block.
        """
        dx, aux_outputs_0 = self.conv_block_0(x, *cond_inputs)
        dx, aux_outputs_1 = self.conv_block_1(dx, *cond_inputs)
        if self.learn_shortcut:
            # We are not using the auxiliary outputs of self.conv_block_s.
            x_shortcut, _ = self.conv_block_s(x, *cond_inputs)
        else:
            x_shortcut = x
        output = x_shortcut + dx
        return output, aux_outputs_0, aux_outputs_1


class MultiOutRes2dBlock(_BaseMultiOutResBlock):
    r"""Residual block for 2D input. It can return multiple outputs, if some
    layers in the block return more than one output.

    Args:
        in_channels (int) : Number of channels in the input tensor.
        out_channels (int) : Number of channels in the output tensor.
        kernel_size (int, optional, default=3): Kernel size for the
            convolutional filters in the residual link.
        padding (int, optional, default=1): Padding size.
        dilation (int, optional, default=1): Dilation factor.
        groups (int, optional, default=1): Number of convolutional/linear
            groups.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        skip_activation_norm (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies activation norm to the
            learned shortcut connection.
        skip_nonlinearity (bool, optional, default=True): If ``True`` and
            ``learn_shortcut`` is also ``True``, applies nonlinearity to the
            learned shortcut connection.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function in the residual link.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layers.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        hidden_channels_equal_out_channels (bool, optional, default=False):
            If ``True``, set the hidden channel number to be equal to the
            output channel number. If ``False``, the hidden channel number
            equals to the smaller of the input channel number and the
            output channel number.
        order (str, optional, default='CNACNA'): Order of operations
            in the residual link.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
        learn_shortcut (bool, optional, default=False): If ``True``, always use
            a convolutional shortcut instead of an identity one, otherwise only
            use a convolutional one if input and output have different number of
            channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 skip_activation_norm=True, skip_nonlinearity=False,
                 nonlinearity='leakyrelu', inplace_nonlinearity=False,
                 apply_noise=False, hidden_channels_equal_out_channels=False,
                 order='CNACNA', learn_shortcut=False):
        super().__init__(in_channels, out_channels, kernel_size, padding,
                         dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         skip_activation_norm, skip_nonlinearity,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, hidden_channels_equal_out_channels,
                         order, MultiOutConv2dBlock, learn_shortcut)

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from .conv import LinearBlock, Conv1dBlock, Conv2dBlock, Conv3dBlock, \
    HyperConv2dBlock, MultiOutConv2dBlock, \
    PartialConv2dBlock, PartialConv3dBlock
from .residual import ResLinearBlock, Res1dBlock, Res2dBlock, Res3dBlock, \
    HyperRes2dBlock, MultiOutRes2dBlock, UpRes2dBlock, DownRes2dBlock, \
    PartialRes2dBlock, PartialRes3dBlock
# from .non_local import NonLocal2dBlock

__all__ = ['Conv1dBlock', 'Conv2dBlock', 'Conv3dBlock', 'LinearBlock',
           'HyperConv2dBlock', 'MultiOutConv2dBlock',
           'PartialConv2dBlock', 'PartialConv3dBlock',
           'Res1dBlock', 'Res2dBlock', 'Res3dBlock',
           'UpRes2dBlock', 'DownRes2dBlock',
           'ResLinearBlock', 'HyperRes2dBlock', 'MultiOutRes2dBlock',
           'PartialRes2dBlock', 'PartialRes3dBlock',]

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn


class ApplyNoise(nn.Module):
    r"""Add Gaussian noise to the input tensor."""

    def __init__(self):
        super().__init__()
        # scale of the noise
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        r"""

        Args:
            x (tensor): Input tensor.
            noise (tensor, optional, default=``None``) : Noise tensor to be
                added to the input.
        """
        if noise is None:
            sz = x.size()
            noise = x.new_empty(sz[0], 1, *sz[2:]).normal_()

        return x + self.weight * noise


class PartialSequential(nn.Sequential):
    r"""Sequential block for partial convolutions."""
    def __init__(self, *modules):
        super(PartialSequential, self).__init__(*modules)

    def forward(self, x):
        r"""

        Args:
            x (tensor): Input tensor.
        """
        act = x[:, :-1]
        mask = x[:, -1].unsqueeze(1)
        for module in self:
            act, mask = module(act, mask_in=mask)
        return act

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .misc import ApplyNoise


class _BaseConvBlock(nn.Module):
    r"""An abstract wrapper class that wraps a torch convolution or linear layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, order, input_dim):
        super().__init__()
        from .nonlinearity import get_nonlinearity_layer
        from .weight_norm import get_weight_norm_layer
        from .activation_norm import get_activation_norm_layer
        self.weight_norm_type = weight_norm_type

        # Convolutional layer.
        if weight_norm_params is None:
            weight_norm_params = SimpleNamespace()
        weight_norm = get_weight_norm_layer(
            weight_norm_type, **vars(weight_norm_params))
        conv_layer = weight_norm(self._get_conv_layer(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, input_dim))

        # Noise injection layer.
        noise_layer = ApplyNoise() if apply_noise else None

        # Normalization layer.
        conv_before_norm = order.find('C') < order.find('N')
        norm_channels = out_channels if conv_before_norm else in_channels
        if activation_norm_params is None:
            activation_norm_params = SimpleNamespace()
        activation_norm_layer = get_activation_norm_layer(
            norm_channels,
            activation_norm_type,
            input_dim,
            **vars(activation_norm_params))

        # Nonlinearity layer.
        nonlinearity_layer = get_nonlinearity_layer(
            nonlinearity, inplace=inplace_nonlinearity)

        # Mapping from operation names to layers.
        mappings = {'C': {'conv': conv_layer},
                    'N': {'norm': activation_norm_layer},
                    'A': {'nonlinearity': nonlinearity_layer}}

        # All layers in order.
        self.layers = nn.ModuleDict()
        for op in order:
            if list(mappings[op].values())[0] is not None:
                self.layers.update(mappings[op])
                if op == 'C' and noise_layer is not None:
                    # Inject noise after convolution.
                    self.layers.update({'noise': noise_layer})

        # Whether this block expects conditional inputs.
        self.conditional = \
            getattr(conv_layer, 'conditional', False) or \
            getattr(activation_norm_layer, 'conditional', False)

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        """
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                # Layers that require conditional inputs.
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            else:
                x = layer(x)
        return x

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        # Returns the convolutional layer.
        if input_dim == 0:
            layer = nn.Linear(in_channels, out_channels, bias)
        else:
            layer_type = getattr(nn, 'Conv%dd' % input_dim)

            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias)
        return layer

    def __repr__(self):
        main_str = self._get_name() + '('
        child_lines = []
        for name, layer in self.layers.items():
            mod_str = repr(layer)
            if name == 'conv' and self.weight_norm_type != 'none' and \
                    self.weight_norm_type != '':
                mod_str = mod_str[:-1] + \
                    ', weight_norm={}'.format(self.weight_norm_type) + ')'
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(mod_str)
        if len(child_lines) == 1:
            main_str += child_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(child_lines) + '\n'

        main_str += ')'
        return main_str

    @staticmethod
    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s


class LinearBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Linear`` with normalization and
    nonlinearity.

    Args:
        in_features (int): Number of channels in the input tensor.
        out_features (int): Number of channels in the output tensor.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, add
            Gaussian noise with learnable magnitude after the
            fully-connected layer.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: fully-connected,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_features, out_features, None, None,
                         None, None, None, bias,
                         None, weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, 0)


class Conv1dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv1d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, 1)


class Conv2dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 2)


class Conv3dBlock(_BaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False,
                 order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 3)


class _BaseHyperConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a hyper convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias,
                 padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity, apply_noise,
                 is_hyper_conv, is_hyper_norm,
                 order, input_dim):
        self.is_hyper_conv = is_hyper_conv
        if is_hyper_conv:
            weight_norm_type = 'none'
        if is_hyper_norm:
            activation_norm_type = 'hyper_' + activation_norm_type
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, input_dim)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        if input_dim == 0:
            raise ValueError('HyperLinearBlock is not supported.')
        else:
            name = 'HyperConv' if self.is_hyper_conv else 'nn.Conv'
            layer_type = eval(name + '%dd' % input_dim)
            layer = layer_type(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode)
        return layer


class HyperConv2dBlock(_BaseHyperConvBlock):
    r"""A Wrapper class that wraps ``HyperConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        is_hyper_conv (bool, optional, default=False): If ``True``, use
            ``HyperConv2d``, otherwise use ``torch.nn.Conv2d``.
        is_hyper_norm (bool, optional, default=False): If ``True``, use
            hyper normalizations.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 is_hyper_conv=False, is_hyper_norm=False,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         is_hyper_conv, is_hyper_norm, order, 2)


class HyperConv2d(nn.Module):
    r"""Hyper Conv2d initialization.

    Args:
        in_channels (int): Dummy parameter.
        out_channels (int): Dummy parameter.
        kernel_size (int or tuple): Dummy parameter.
        stride (int or tuple, optional, default=1):
            Stride of the convolution. Default: 1
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        padding_mode (string, optional, default='zeros'):
            ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True): If ``True``,
            adds a learnable bias to the output.
    """

    def __init__(self, in_channels=0, out_channels=0, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        self.conditional = True

    def forward(self, x, *args, conv_weights=(None, None), **kwargs):
        r"""Hyper Conv2d forward. Convolve x using the provided weight and bias.

        Args:
            x (N x C x H x W tensor): Input tensor.
            conv_weights (N x C2 x C1 x k x k tensor or list of tensors):
                Convolution weights or [weight, bias].
        Returns:
            y (N x C2 x H x W tensor): Output tensor.
        """
        if conv_weights is None:
            conv_weight, conv_bias = None, None
        elif isinstance(conv_weights, torch.Tensor):
            conv_weight, conv_bias = conv_weights, None
        else:
            conv_weight, conv_bias = conv_weights

        if conv_weight is None:
            return x
        if conv_bias is None:
            if self.use_bias:
                raise ValueError('bias not provided but set to true during '
                                 'initialization')
            conv_bias = [None] * x.size(0)
        if self.padding_mode != 'zeros':
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding

        y = None
        for i in range(x.size(0)):
            if self.stride >= 1:
                yi = F.conv2d(x[i: i + 1],
                              weight=conv_weight[i], bias=conv_bias[i],
                              stride=self.stride, padding=padding,
                              dilation=self.dilation, groups=self.groups)
            else:
                yi = F.conv_transpose2d(x[i: i + 1], weight=conv_weight[i],
                                        bias=conv_bias[i], padding=self.padding,
                                        stride=int(1 / self.stride),
                                        dilation=self.dilation,
                                        output_padding=self.padding,
                                        groups=self.groups)
            y = torch.cat([y, yi]) if y is not None else yi
        return y


class _BasePartialConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a partial convolutional layer
    with normalization and nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 multi_channel, return_mask,
                 apply_noise, order, input_dim):
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.partial_conv = True
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity, apply_noise,
                         order, input_dim)

    def _get_conv_layer(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias, padding_mode,
                        input_dim):
        if input_dim == 2:
            layer_type = PartialConv2d
        elif input_dim == 3:
            layer_type = PartialConv3d
        else:
            raise ValueError('Partial conv only supports 2D and 3D conv now.')
        layer = layer_type(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
            multi_channel=self.multi_channel, return_mask=self.return_mask)
        return layer

    def forward(self, x, *cond_inputs, mask_in=None, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Output tensor.
              - mask_out (tensor, optional): Masks the valid output region.
        """
        mask_out = None
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            elif getattr(layer, 'partial_conv', False):
                x = layer(x, mask_in=mask_in, **kw_cond_inputs)
                if type(x) == tuple:
                    x, mask_out = x
            else:
                x = layer(x)

        if mask_out is not None:
            return x, mask_out
        return x


class PartialConv2dBlock(_BasePartialConvBlock):
    r"""A Wrapper class that wraps ``PartialConv2d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask, apply_noise, order, 2)


class PartialConv3dBlock(_BasePartialConvBlock):
    r"""A Wrapper class that wraps ``PartialConv3d`` with normalization and
    nonlinearity.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
        multi_channel (bool, optional, default=False): If ``True``, use
            different masks for different channels.
        return_mask (bool, optional, default=True): If ``True``, the
            forward call also returns a new mask.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 multi_channel=False, return_mask=True,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         multi_channel, return_mask, apply_noise, order, 3)


class _MultiOutBaseConvBlock(_BaseConvBlock):
    r"""An abstract wrapper class that wraps a hyper convolutional layer with
    normalization and nonlinearity. It can return multiple outputs, if some
    layers in the block return more than one output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias,
                 padding_mode,
                 weight_norm_type, weight_norm_params,
                 activation_norm_type, activation_norm_params,
                 nonlinearity, inplace_nonlinearity,
                 apply_noise, order, input_dim):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, input_dim)
        self.multiple_outputs = True

    def forward(self, x, *cond_inputs, **kw_cond_inputs):
        r"""

        Args:
            x (tensor): Input tensor.
            cond_inputs (list of tensors) : Conditional input tensors.
            kw_cond_inputs (dict) : Keyword conditional inputs.
        Returns:
            (tuple):
              - x (tensor): Main output tensor.
              - other_outputs (list of tensors): Other output tensors.
        """
        other_outputs = []
        for layer in self.layers.values():
            if getattr(layer, 'conditional', False):
                x = layer(x, *cond_inputs, **kw_cond_inputs)
            if getattr(layer, 'multiple_outputs', False):
                x, other_output = layer(x)
                other_outputs.append(other_output)
            else:
                x = layer(x)
        return (x, *other_outputs)


class MultiOutConv2dBlock(_MultiOutBaseConvBlock):
    r"""A Wrapper class that wraps ``torch.nn.Conv2d`` with normalization and
    nonlinearity. It can return multiple outputs, if some layers in the block
    return more than one output.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional, default=1):
            Stride of the convolution.
        padding (int or tuple, optional, default=0):
            Zero-padding added to both sides of the input.
        dilation (int or tuple, optional, default=1):
            Spacing between kernel elements.
        groups (int, optional, default=1): Number of blocked connections
            from input channels to output channels.
        bias (bool, optional, default=True):
            If ``True``, adds a learnable bias to the output.
        padding_mode (string, optional, default='zeros'): Type of padding:
            ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        weight_norm_type (str, optional, default='none'):
            Type of weight normalization.
            ``'none'``, ``'spectral'``, ``'weight'``
            or ``'weight_demod'``.
        weight_norm_params (obj, optional, default=None):
            Parameters of weight normalization.
            If not ``None``, ``weight_norm_params.__dict__`` will be used as
            keyword arguments when initializing weight normalization.
        activation_norm_type (str, optional, default='none'):
            Type of activation normalization.
            ``'none'``, ``'instance'``, ``'batch'``, ``'sync_batch'``,
            ``'layer'``,  ``'layer_2d'``, ``'group'``, ``'adaptive'``,
            ``'spatially_adaptive'`` or ``'hyper_spatially_adaptive'``.
        activation_norm_params (obj, optional, default=None):
            Parameters of activation normalization.
            If not ``None``, ``activation_norm_params.__dict__`` will be used as
            keyword arguments when initializing activation normalization.
        nonlinearity (str, optional, default='none'):
            Type of nonlinear activation function.
            ``'none'``, ``'relu'``, ``'leakyrelu'``, ``'prelu'``,
            ``'tanh'`` , ``'sigmoid'`` or ``'softmax'``.
        inplace_nonlinearity (bool, optional, default=False): If ``True``,
            set ``inplace=True`` when initializing the nonlinearity layer.
        apply_noise (bool, optional, default=False): If ``True``, adds
            Gaussian noise with learnable magnitude to the convolution output.
        order (str, optional, default='CNA'): Order of operations.
            ``'C'``: convolution,
            ``'N'``: normalization,
            ``'A'``: nonlinear activation.
            For example, a block initialized with ``order='CNA'`` will
            do convolution first, then normalization, then nonlinearity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros',
                 weight_norm_type='none', weight_norm_params=None,
                 activation_norm_type='none', activation_norm_params=None,
                 nonlinearity='none', inplace_nonlinearity=False,
                 apply_noise=False, order='CNA'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode,
                         weight_norm_type, weight_norm_params,
                         activation_norm_type, activation_norm_params,
                         nonlinearity, inplace_nonlinearity,
                         apply_noise, order, 2)


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2d(nn.Conv2d):
    r"""Partial 2D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels,
                                                 self.in_channels,
                                                 self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1])

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``,
                it masks the valid input region.
        """
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    # If mask is not provided, create a mask.
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0],
                                          x.data.shape[1],
                                          x.data.shape[2],
                                          x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2],
                                          x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater,
                                            bias=None, stride=self.stride,
                                            padding=self.padding,
                                            dilation=self.dilation, groups=1)

                # For mixed precision training, eps from 1e-8 to 1e-6.
                eps = 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + eps)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv3d(nn.Conv3d):
    r"""Partial 3D convolution in
    "Image inpainting for irregular holes using partial convolutions."
    Liu et al., ECCV 2018
    """

    def __init__(self, *args, multi_channel=False, return_mask=True, **kwargs):
        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = \
                torch.ones(self.out_channels, self.in_channels,
                           self.kernel_size[0], self.kernel_size[1],
                           self.kernel_size[2])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0],
                                                 self.kernel_size[1],
                                                 self.kernel_size[2])
        self.weight_maskUpdater = self.weight_maskUpdater.to('cuda')

        shape = self.weight_maskUpdater.shape
        self.slide_winsize = shape[1] * shape[2] * shape[3] * shape[4]
        self.partial_conv = True

    def forward(self, x, mask_in=None):
        r"""

        Args:
            x (tensor): Input tensor.
            mask_in (tensor, optional, default=``None``) If not ``None``, it
                masks the valid input region.
        """
        assert len(x.shape) == 5

        with torch.no_grad():
            mask = mask_in
            update_mask = F.conv3d(mask, self.weight_maskUpdater, bias=None,
                                   stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=1)

            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = super(PartialConv3d, self).forward(torch.mul(x, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            if mask_in is not None:
                output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        if self.return_mask:
            return output, update_mask
        else:
            return output

from torch import nn
import torch
import torch.nn.functional as F
from models.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d



class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x,with_feature = False):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            out["jacobian_map"] = jacobian_map

            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])

            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian
        out["pred_feature"] = prediction
        if with_feature:
            out["feature_map"] = feature_map
        return out

from torch import nn
import torch.nn.functional as F
import torch
from models.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        # sparse_deformed = F.grid_sample(source_repeat, sparse_motions,align_corners = False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)#bs*(numkp+1)*1*h*w
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)#bs*(numkp+1)*h*w*2
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)#bs*num+1*4*w*h
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)#bs*numkp+1*1*h*w
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)# bs,2,64,64
        deformation = deformation.permute(0, 2, 3, 1)#bs*h*w*2

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

import torch.nn as nn
import torch
from models.util import MyResNet34

class audio2poseLSTM(nn.Module):
    def __init__(self):
        super(audio2poseLSTM,self).__init__()

        self.em_pose = MyResNet34(256, 1)
        self.em_audio = MyResNet34(256, 1)
        self.lstm = nn.LSTM(512,256,num_layers=2,bias=True,batch_first=True)

        self.output = nn.Linear(256,6)


    def forward(self,x):
        pose_em = self.em_pose(x["img"])
        bs = pose_em.shape[0]
        zero_state = torch.zeros((2, bs, 256), requires_grad=True).to(pose_em.device)
        cur_state = (zero_state, zero_state)
        img_em = pose_em
        bs,seqlen,num,dims = x["audio"].shape

        audio = x["audio"].reshape(-1, 1, num, dims)
        audio_em = self.em_audio(audio).reshape(bs, seqlen, 256)

        result = [self.output(img_em).unsqueeze(1)]

        for i in range(seqlen):

            img_em,cur_state = self.lstm(torch.cat((audio_em[:,i:i+1],img_em.unsqueeze(1)),dim=2),cur_state)
            img_em = img_em.reshape(-1, 256)

            result.append(self.output(img_em).unsqueeze(1))
        res = torch.cat(result,dim=1)
        return res

import torch.nn as nn
import torch
from models.util import mydownres2Dblock
import numpy as np
from models.util import AntiAliasInterpolation2d,make_coordinate_grid
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F
import copy


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, winsize):
        return self.pos_table[:, :winsize].clone().detach()

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,opt, src, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC

        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.permute(1, 0, 2)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)

        hs = self.decoder(tgt, memory,
                          pos=pos_embed, query_pos=query_embed)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None, pos = None):
        output = src+pos

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,  tgt_mask = None,  memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        output = tgt+pos+query_pos

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        # q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



class Audio2kpTransformer(nn.Module):
    def __init__(self,opt):
        super(Audio2kpTransformer, self).__init__()
        self.opt = opt


        self.embedding = nn.Embedding(41, opt.embedding_dim)
        self.pos_enc = PositionalEncoding(512,20)
        self.down_pose = AntiAliasInterpolation2d(1,0.25)
        input_dim = 2
        self.feature_extract = nn.Sequential(mydownres2Dblock(input_dim,32),
                                             mydownres2Dblock(32,64),
                                             mydownres2Dblock(64,128),
                                             mydownres2Dblock(128,256),
                                             mydownres2Dblock(256,512),
                                             nn.AvgPool2d(2))

        self.decode_dim = 70
        self.audio_embedding = nn.Sequential(nn.ConvTranspose2d(1, 8, (29, 14), stride=(1, 1), padding=(0, 11)),
                                             BatchNorm2d(8),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(8, 35, (13, 13), stride=(1, 1), padding=(6, 6)))
        self.decodefeature_extract = nn.Sequential(mydownres2Dblock(self.decode_dim,32),
                                             mydownres2Dblock(32,64),
                                             mydownres2Dblock(64,128),
                                             mydownres2Dblock(128,256),
                                             mydownres2Dblock(256,512),
                                             nn.AvgPool2d(2))

        self.transformer = Transformer()
        self.kp = nn.Linear(512,opt.num_kp*2)
        self.jacobian = nn.Linear(512,opt.num_kp*4)
        self.jacobian.weight.data.zero_()
        self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.opt.num_kp, dtype=torch.float))
        self.criterion = nn.L1Loss()

    def create_sparse_motions(self, source_image, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid
        if 'jacobian' in kp_source:
            jacobian = kp_source['jacobian']
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.opt.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)

        return sparse_motions.permute(0,1,4,2,3).reshape(bs,(self.opt.num_kp+1)*2,64,64)



    def forward(self,x, initial_kp = None):
        bs,seqlen = x["ph"].shape
        ph = x["ph"].reshape(bs*seqlen,1)
        pose = x["pose"].reshape(bs*seqlen,1,256,256)
        input_feature = self.down_pose(pose)

        phoneme_embedding = self.embedding(ph.long())
        phoneme_embedding = phoneme_embedding.reshape(bs*seqlen, 1, 16, 16)
        phoneme_embedding = F.interpolate(phoneme_embedding, scale_factor=4)
        input_feature = torch.cat((input_feature, phoneme_embedding), dim=1)

        input_feature = self.feature_extract(input_feature).unsqueeze(-1).reshape(bs,seqlen,512)

        audio = x["audio"].reshape(bs * seqlen, 1, 4, 41)
        decoder_feature = self.audio_embedding(audio)
        decoder_feature = F.interpolate(decoder_feature, scale_factor=2)
        decoder_feature = self.decodefeature_extract(torch.cat(
            (decoder_feature,
             initial_kp["feature_map"].unsqueeze(1).repeat(1, seqlen, 1, 1, 1).reshape(bs * seqlen, 35, 64, 64)),
            dim=1)).unsqueeze(-1).reshape(bs, seqlen, 512)

        posi_em = self.pos_enc(self.opt.num_w*2+1)


        out = {}

        output_feature = self.transformer(self.opt,input_feature,decoder_feature,posi_em)[-1,self.opt.num_w]

        out["value"] = self.kp(output_feature).reshape(bs,self.opt.num_kp,2)
        out["jacobian"] = self.jacobian(output_feature).reshape(bs,self.opt.num_kp,2,2)

        return out



from torch import nn

import torch.nn.functional as F
import torch
import cv2
import numpy as np

from models.resnet import resnet34
from models.layers.residual import Res2dBlock,Res1dBlock,DownRes2dBlock

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def myres2Dblock(indim,outdim,k_size = 3,padding = 1, normalize = "batch",nonlinearity = "relu",order = "NACNAC"):
    return Res2dBlock(indim,outdim,k_size,padding,activation_norm_type=normalize,nonlinearity=nonlinearity,inplace_nonlinearity=True,order = order)

def myres1Dblock(indim,outdim,k_size = 3,padding = 1, normalize = "batch",nonlinearity = "relu",order = "NACNAC"):
    return Res1dBlock(indim,outdim,k_size,padding,activation_norm_type=normalize,nonlinearity=nonlinearity,inplace_nonlinearity=True,order = order)

def mydownres2Dblock(indim,outdim,k_size = 3,padding = 1, normalize = "batch",nonlinearity = "leakyrelu",order = "NACNAC"):
    return DownRes2dBlock(indim,outdim,k_size,padding=padding,activation_norm_type=normalize,nonlinearity=nonlinearity,inplace_nonlinearity=True,order = order)

def gaussian2kp(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    value = (heatmap * grid).sum(dim=(2, 3))
    kp = {'value': value}

    return kp

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value'] #bs*numkp*2

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) #h*w*2
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape #1*1*h*w*2
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)  #bs*numkp*h*w*2

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out,inplace=True)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out,inplace=True)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        del x
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        del x
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

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

def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)



class up_sample(nn.Module):
    def __init__(self, scale_factor):
        super(up_sample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor,mode = 'linear',align_corners = True)
        return x



class MyResNet34(nn.Module):
    def __init__(self,embedding_dim,input_channel = 3):
        super(MyResNet34, self).__init__()
        self.resnet = resnet34(norm_layer = BatchNorm2d,num_classes=embedding_dim,input_channel = input_channel)
    def forward(self, x):
        return self.resnet(x)



class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
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

