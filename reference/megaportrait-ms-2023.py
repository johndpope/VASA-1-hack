import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


def get_nonspade_norm_layer(norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        subnorm_type = norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "sync_batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class LadderEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, need_feat=False, use_mask=False, label_nc=0, z_dim=512, norm_type="spectralinstance"):
        super().__init__()
        self.need_feat = need_feat
        ldmk_img_nc = 3

        nif = 3 + label_nc + 2 * ldmk_img_nc

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        nef = 64
        norm_layer = get_nonspade_norm_layer(norm_type)
        self.layer1 = norm_layer(nn.Conv2d(nif, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef * 1, nef * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef * 2, nef * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(nef * 4, nef * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))

        if need_feat:
            self.up_layer2 = norm_layer(
                nn.Conv2d(nef * 2, nef * 2, kw, stride=1, padding=pw)
            )
            self.up_layer3 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 4, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
            self.up_layer4 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            )
            self.up_layer5 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            )
            self.up_layer6 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            )

        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4
        self.fc = nn.Linear(nef * 8 * s0 * s0, z_dim)

    def forward(self, x):
        features = None
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear")

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        if self.need_feat:
            features = self.up_layer2(x)
        x = self.layer3(self.actvn(x))
        if self.need_feat:
            features = self.up_layer3(x) + features
        x = self.layer4(self.actvn(x))
        if self.need_feat:
            features = self.up_layer4(x) + features
        x = self.layer5(self.actvn(x))
        if self.need_feat:
            features = self.up_layer5(x) + features
        x = self.layer6(self.actvn(x))
        if self.need_feat:
            features = self.up_layer6(x) + features

        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x / (x.norm(dim=-1, p=2, keepdim=True) + 1e-5)

        return x, features
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by: algohunt
# Microsoft Research & Peking University
# lilingzhi@pku.edu.cn
# Copyright (c) 2019

import torch
from torch.nn import init
import torch.nn.functional as F
from torch import nn
from math import sqrt


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class Blur(nn.Module):
    def __init__(self):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(
            input,
            self.weight.repeat(input.shape[1], 1, 1, 1),
            padding=1,
            groups=input.shape[1],
        )


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return outfrom torch import nn
import torch
import functools
from modules.util import (
    Hourglass,
    make_coordinate_grid,
    LayerNorm2d,
)


class DenseMotionNetworkReg(nn.Module):
    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        Lwarp=False,
        AdaINc=0,
        dec_lease=0,
        label_nc=0,
        ldmkimg=False,
        occlusion=False,
    ):
        super(DenseMotionNetworkReg, self).__init__()
        in_c = 3 + label_nc + 2 * 3 if ldmkimg else 3 + label_nc
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=in_c,
            max_features=max_features,
            num_blocks=num_blocks,
            Lwarp=Lwarp,
            AdaINc=AdaINc,
            dec_lease=dec_lease,
        )

        self.occlusion = occlusion
        if dec_lease > 0:
            norm_layer = functools.partial(LayerNorm2d, affine=True)
            self.reger = nn.Sequential(
                norm_layer(self.hourglass.out_filters),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    self.hourglass.out_filters, 2, kernel_size=7, stride=1, padding=3
                ),
            )
            if occlusion:
                self.occlusion_net = nn.Sequential(
                    norm_layer(self.hourglass.out_filters),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(
                        self.hourglass.out_filters,
                        1,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                )
        else:
            self.reger = nn.Conv2d(
                self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3)
            )

    def forward(self, source_image, drv_deca):
        prediction = self.hourglass(source_image, drv_exp=drv_deca)

        out_dict = {}
        flow = self.reger(prediction)
        bs, _, h, w = flow.shape
        flow_norm = 2 * torch.cat(
            [flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1
        )
        out_dict["flow"] = flow_norm
        grid = make_coordinate_grid((h, w), type=torch.FloatTensor).to(flow_norm.device)
        deformation = grid + flow_norm.permute(0, 2, 3, 1)
        out_dict["deformation"] = deformation

        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion_net(prediction))
            _, _, h_old, w_old = occlusion_map.shape
            _, _, h, w = source_image.shape
            if h_old != h or w_old != w:
                occlusion_map = torch.nn.functional.interpolate(
                    occlusion_map, size=(h, w), mode="bilinear", align_corners=False
                )
            out_dict["occlusion_map"] = occlusion_map
        return out_dict
from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian
import torch
from torch.nn.utils import spectral_norm


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(
        self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False
    ):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size
        )

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(
        self,
        num_channels=3,
        block_expansion=64,
        num_blocks=4,
        max_features=512,
        sn=False,
        use_kp=False,
        num_kp=10,
        kp_variance=0.01,
        AdaINc=0,
        **kwargs
    ):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    num_channels + num_kp * use_kp
                    if i == 0
                    else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                    sn=sn,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(
            self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1
        )
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

        self.AdaINc = AdaINc
        if AdaINc > 0:
            self.to_exp = nn.Sequential(
                nn.Linear(block_expansion * (2 ** num_blocks), 256),
                nn.LeakyReLU(256),
                nn.Linear(256, AdaINc),
            )

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        if self.AdaINc > 0:
            feat = F.adaptive_avg_pool2d(out, 1)
            exp = self.to_exp(feat.squeeze(-1).squeeze(-1))
        else:
            exp = None

        return feature_maps, prediction_map, exp


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        self.use_kp = kwargs["use_kp"]
        for scale in scales:
            discs[str(scale).replace(".", "-")] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        gain = 0.02
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace("-", ".")
            key = "prediction_" + scale
            feature_maps, prediction_map, exp = disc(x[key], kp)
            out_dict["feature_maps_" + scale] = feature_maps
            out_dict["prediction_map_" + scale] = prediction_map
            out_dict["exp_" + scale] = exp
        return out_dict
import torch
import torch.nn.functional as F
from torch import nn

from modules.dense_motion import DenseMotionNetworkReg
from modules.LadderEncoder import LadderEncoder
from modules.spade import SPADEGenerator
from modules.util import Hourglass, kp2gaussian


def Generator(arch, **kwarg):
    return OcclusionAwareSPADEGenerator(**kwarg, hasid=True)


class OcclusionAwareSPADEGenerator(nn.Module):
    """
    Generator that given source image, source ldmk image and driving ldmk image try to transform image according to movement trajectories
    according to the ldmks.
    """

    def __init__(
        self,
        num_channels,
        block_expansion,
        max_features,
        dense_motion_params=None,
        with_warp_im=False,
        hasid=False,
        with_gaze_htmap=False,
        with_ldmk_line=False,
        with_mouth_line=False,
        with_ht=False,
        ladder=None,
        use_IN=False,
        use_SN=True
    ):
        super(OcclusionAwareSPADEGenerator, self).__init__()
        self.with_warp_im = with_warp_im
        self.with_gaze_htmap = with_gaze_htmap
        self.with_ldmk_line = with_ldmk_line
        self.with_mouth_line = with_mouth_line
        self.with_ht = with_ht
        self.ladder = ladder
        self.use_IN = use_IN
        self.use_SN = use_SN

        ladder_norm_type = "spectralinstance" if use_SN else "instance"
        self.ladder_network = LadderEncoder(**ladder, norm_type=ladder_norm_type)
        self.dense_motion_network = DenseMotionNetworkReg(
            **dense_motion_params
        )

        num_blocks = 3
        self.feature_encoder = Hourglass(
                block_expansion=block_expansion,
                in_features=3,
                max_features=max_features,
                num_blocks=num_blocks,
                Lwarp=False,
                AdaINc=0,
                dec_lease=0,
                use_IN=use_IN
            )
        self.fuse_high_res = nn.Conv2d(block_expansion + 3, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        norm = "spectral" if self.use_SN else ""
        norm += "spadeinstance3x3" if self.use_IN else "spadebatch3x3"
        if hasid:
            norm += "id"
        class_dim = 256
        label_nc_offset = 0  # if with_warp_im else 256
        label_nc_offset = label_nc_offset + 8 if with_gaze_htmap else label_nc_offset
        label_nc_offset = label_nc_offset + 6 if with_ldmk_line else label_nc_offset
        label_nc_offset = label_nc_offset + 3 if with_mouth_line else label_nc_offset
        label_nc_offset = label_nc_offset + 59 if with_ht else label_nc_offset
        label_nc_offset = label_nc_offset + 1  # For occlusion map
        label_nc_list = [512, 512, 512, 512, 256, 128, 64]
        label_nc_list = [ln + label_nc_offset for ln in label_nc_list]
        self.SPDAE_G = SPADEGenerator(
            conv_dim=32,
            label_nc=label_nc_list,
            norm_G=norm,
            class_dim=class_dim,
        )

        self.num_channels = num_channels

        self.apply(self._init_weights)

    def _init_weights(self, m):
        gain = 0.02
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(
            inp.to(deformation.dtype), deformation, padding_mode="reflection"
        )

    def get_gaze_ht(self, source_image, kp_driving):
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(
            kp_driving, spatial_size=spatial_size, kp_variance=0.005
        )
        return gaussian_driving[:, 29:37]

    def forward_warp(
        self,
        source_image,
        ldmk_line=None
    ):
        output_dict = {}

        input_t = (
            source_image
            if ldmk_line is None
            else torch.cat((source_image, ldmk_line), dim=1)
        )
        
        style_feat, _ = self.ladder_network(input_t)
        drv_exp = style_feat
        dense_motion = self.dense_motion_network(input_t, drv_exp)

        output_dict["deformation"] = dense_motion["deformation"]
        output_dict["deformed"] = self.deform_input(
            source_image, dense_motion["deformation"]
        )
        output_dict["occlusion_map"] = dense_motion["occlusion_map"]
        output_dict["prediction"] = output_dict["deformed"]
        output_dict["flow"] = dense_motion["flow"]
        return output_dict

    def foward_refine(
        self,
        source_image,
        src_id,
        ldmk_line,
        mouth_line,
        warp_out,
        kp_driving=None,
    ):
        _, out_list = self.feature_encoder(source_image, return_all=True)
        out_list[-1] = self.fuse_high_res(out_list[-1])
        out_list = [self.deform_input(out, warp_out["deformation"]) for out in out_list]

        feature_list = []
        for out in out_list:
            if self.with_gaze_htmap:
                gaze_htmap = self.get_gaze_ht(out, kp_driving)

            inputs = out
            if out.shape[2] != warp_out["occlusion_map"].shape[2] or out.shape[3] != warp_out["occlusion_map"].shape[3]:
                occlusion_map = F.interpolate(
                    warp_out["occlusion_map"], size=out.shape[2:], mode="bilinear"
                )
            else:
                occlusion_map = warp_out["occlusion_map"]
            inputs = torch.cat((inputs, occlusion_map), dim=1)
            
            if self.with_gaze_htmap:
                inputs = torch.cat((inputs, gaze_htmap), dim=1)
            if self.with_ldmk_line:
                ldmk_line = F.interpolate(ldmk_line, size=inputs.shape[2:], mode="bilinear")
                inputs = torch.cat((inputs, ldmk_line), dim=1)
            if self.with_mouth_line:
                mouth_line = F.interpolate(
                    mouth_line, size=inputs.shape[2:], mode="bilinear"
                )
                inputs = torch.cat((inputs, mouth_line), dim=1)
            
            feature_list.append(inputs)

        outs = self.SPDAE_G(feature_list, class_emb=src_id)

        warp_out["prediction"] = outs
        return warp_out

    def forward(
        self,
        source_image,
        kp_driving=None,
        src_id=None,
        stage=None,
        ldmk_line=None,
        mouth_line=None,
        warp_out=None,
    ):
        if stage == "Warp":
            return self.forward_warp(source_image, ldmk_line)
        elif stage == "Refine":
            return self.foward_refine(
                source_image,
                src_id,
                ldmk_line,
                mouth_line,
                warp_out,
                kp_driving,
            )
        else:
            raise Exception("Unknown stage.")
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from modules.util import AntiAliasInterpolation2d


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


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


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, arch=None, rank=0, conf=None):
        super(GeneratorFullModel, self).__init__()
        self.arch = arch
        self.conf = conf
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        if conf['model']['discriminator'].get('type', 'MultiPatchGan') == 'MultiPatchGan':
            self.disc_scales = self.discriminator.scales
        
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.to(rank)

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.to(rank)
            self.vgg.eval()

        if self.loss_weights.get('warp_ce', 0) > 0:
            self.ce_loss = nn.CrossEntropyLoss().to(rank)
        
        if self.loss_weights.get('l1', 0) > 0:
            self.l1_loss = nn.L1Loss()

    def nist_prec(self, x):
        x = (x.clone() - 0.5) * 2 # -1 ~ 1
        x = x[:, :, 25:256, 25:256]
        x = torch.flip(x,[1]) # RGB -> BGR
        return x
    
    def forward_warp(self, x, cal_loss=True):
        if self.conf['dataset'].get('ldmkimg', False):
            ldmk_line = torch.cat((x['source_line'], x['driving_line']), dim=1)
        else:
            ldmk_line = None
        generated = self.generator(x['source'], ldmk_line=ldmk_line, stage='Warp')

        loss_values = {}
        if cal_loss:
            pyramide_real = self.pyramid(x['driving'])
            pyramide_generated = self.pyramid(generated['deformed'])
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value

            loss_values['warp_perceptual'] = value_total

        return loss_values, generated

    def forward_refine(self, x, warp_out, loss_values, inference=False):
        kp_driving = {'value': x['driving_ldmk']}
        embed_id = x['source_id']
   
        ldmk_line = torch.cat((x['source_line'], x['driving_line']), dim=1) if self.conf['dataset'].get('ldmkimg', False) else None
        if self.loss_weights.get('mouth_enhance', 0) > 0:
            mouth_line = x['driving_line'] * x['mouth_mask']
        else:
            mouth_line = None

        generated = self.generator(x['source'], kp_driving=kp_driving, src_id=embed_id, ldmk_line=ldmk_line, mouth_line=mouth_line, warp_out=warp_out, stage='Refine')
        if inference:
            return generated

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            eye_total = 0
            mouth_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value

                if self.loss_weights.get('eye_enhance', 0) > 0:
                    eye_scale = F.interpolate(x['eye_mask'], size=pyramide_generated['prediction_' + str(scale)].shape[2:], mode='nearest')
                    eye_total += ((pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)]) ** 2 * eye_scale).sum() / (eye_scale.sum() + 1e-6)

                if self.loss_weights.get('mouth_enhance', 0) > 0:
                    mouth_scale = F.interpolate(x['mouth_mask'], size=pyramide_generated['prediction_' + str(scale)].shape[2:], mode='nearest')
                    mouth_total += ((pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)]) ** 2 * mouth_scale).sum() / (mouth_scale.sum() + 1e-6)

            loss_values['perceptual'] = value_total
            if self.loss_weights.get('eye_enhance', 0) > 0:
                loss_values['eye'] = eye_total * self.loss_weights['eye_enhance']
            if self.loss_weights.get('mouth_enhance', 0) > 0:
                loss_values['mouth'] = mouth_total * self.loss_weights['mouth_enhance']
        
        if self.loss_weights.get('l1', 0) > 0:
            loss_values['l1'] = self.l1_loss(generated['prediction'], x['driving']) * self.loss_weights['l1']

        # if self.loss_weights.get('id', 0) > 0:
        #     gen_grid = F.affine_grid(x['driving_theta'], [x['driving_theta'].shape[0], 3, 256,256], align_corners=True)
        #     gen_nist = F.grid_sample(F.interpolate(generated['prediction'], (256, 256), mode='bilinear'), gen_grid, align_corners=True)

        #     gen_id = self.id_classifier(self.nist_prec(gen_nist))
        #     gen_id = F.normalize(gen_id, dim=1)
        #     tgt_id = F.normalize(embed_id, dim=1)

        #     loss_values['id'] = (1 - (gen_id * tgt_id).sum(1).mean()) * self.loss_weights['id']

        if self.loss_weights['generator_gan'] != 0:
            if self.conf['model']['discriminator'].get('type', 'MultiPatchGan') == 'MultiPatchGan':
                if self.conf['model']['discriminator'].get('use_kp', False):
                    discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
                    discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
                else:
                    discriminator_maps_generated = self.discriminator(pyramide_generated, kp=x['driving_line'])
                    discriminator_maps_real = self.discriminator(pyramide_real, kp=x['driving_line'])
                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total

            else:
                discriminator_maps_generated = self.discriminator(pyramide_generated['prediction_1'])
                value = ((1 - discriminator_maps_generated) ** 2).mean()
                loss_values['gen_gan'] = self.loss_weights['generator_gan'] * value

        return loss_values, generated


    def forward(self, x, stage=None, inference=False):
        if stage == 'Warp':
            return self.forward_warp(x, cal_loss=not inference)
        elif stage == 'Full':
            warp_loss, warp_out = self.forward_warp(x)
            return self.forward_refine(x, warp_out, warp_loss, inference=inference)
        else:
            raise Exception("Unknown stage.")

    def get_gaze_loss(self, deformation, gaze):
        mask = (gaze != 0).detach().float()
        up_deform = F.interpolate(deformation.permute(0,3,1,2), size=gaze.shape[1:3], mode='bilinear').permute(0,2,3,1)
        gaze_loss = (torch.abs(up_deform - gaze) * mask).sum() / (mask.sum() + 1e-6)
        return gaze_loss

    def get_ldmk_loss(self, mask, ldmk_gt):
        pred = F.interpolate(mask, size=ldmk_gt.shape[1:], mode='bilinear')
        ldmk_loss = F.cross_entropy(pred, ldmk_gt, ignore_index=0)
        return ldmk_loss


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.use_kp = discriminator.use_kp
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        if self.use_kp:
            kp_driving = generated['kp_driving']
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        else:
            kp_driving = x['driving_line']
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=kp_driving)
            discriminator_maps_real = self.discriminator(pyramide_real, kp=kp_driving)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        if self.loss_weights.get('D_exp', 0) > 0:
            loss_values['exp'] = F.mse_loss(discriminator_maps_real['exp_1'], x['driving_exp']) * self.loss_weights['D_exp'] + \
                F.mse_loss(discriminator_maps_generated['exp_1'], x['driving_exp']) * self.loss_weights['D_exp']

        return loss_values

import re
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from modules.adain import AdaptiveInstanceNorm


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, style_nc):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d(\D*)", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.hasid = parsed.group(3) == "id"
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif "batch" in param_free_norm_type:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.label_nc = label_nc
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        if self.hasid:
            self.mlp_attention = nn.Sequential(
                nn.Conv2d(norm_nc, 1, kernel_size=ks, padding=pw), nn.Sigmoid(),
            )
            self.adain = AdaptiveInstanceNorm(norm_nc, style_nc)

    def forward(self, x, attr_map, id_emb):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(attr_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        spade_out = normalized * (1 + gamma) + beta
        if self.hasid:
            attention = self.mlp_attention(x)
            adain_out = self.adain(x, id_emb)

            out = attention * spade_out + (1 - attention) * adain_out
        else:
            out = spade_out
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc, style_nc, norm_G):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if "spectral" in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace("spectral", "")
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc, style_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc, style_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc, style_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, class_emb):
        x_s = self.shortcut(x, seg, class_emb)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, class_emb)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, class_emb)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, class_emb):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, class_emb))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


class SPADEGenerator(nn.Module):
    def __init__(
        self,
        label_nc=256,
        class_dim=256,
        conv_dim=64,
        norm_G="spectralspadebatch3x3",
    ):
        super().__init__()

        nf = conv_dim
        self.nf = conv_dim
        self.norm_G = norm_G

        self.conv1 = spectral_norm(nn.ConvTranspose2d(class_dim, nf * 16, 4)) if "spectral" in norm_G else nn.ConvTranspose2d(class_dim, nf * 16, 4)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, label_nc[0], class_dim, norm_G)

        self.G_middle_0 = SPADEResnetBlock(
            16 * nf, 16 * nf, label_nc[1], class_dim, norm_G
        )
        self.G_middle_1 = SPADEResnetBlock(
            16 * nf, 16 * nf, label_nc[2], class_dim, norm_G
        )

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, label_nc[3], class_dim, norm_G)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, label_nc[4], class_dim, norm_G)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, label_nc[5], class_dim, norm_G)

        final_nc = nf
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, label_nc[6], class_dim, norm_G)

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

    def forward(self, attr_pyramid, class_emb=None):
        if class_emb is None:
            x = torch.randn(
                (attr_pyramid[0].size(0), 256, 1, 1), device=attr_pyramid[0].device
            )
        else:
            x = class_emb.view(class_emb.size(0), class_emb.size(1), 1, 1)
        x = self.conv1(x)
        style4 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.head_0(x, style4, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style8 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.G_middle_0(x, style8, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style16 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.G_middle_1(x, style16, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style32 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.up_0(x, style32, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style64 = F.interpolate(attr_pyramid[1], size=x.shape[2:], mode="bilinear")
        x = self.up_1(x, style64, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style128 = F.interpolate(attr_pyramid[2], size=x.shape[2:], mode="bilinear")
        x = self.up_2(x, style128, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style256 = F.interpolate(attr_pyramid[3], size=x.shape[2:], mode="bilinear")
        x = self.up_3(x, style256, class_emb)

        x = F.leaky_relu(x, 2e-1, inplace=True)

        x = self.conv_img(x)
        x = torch.tanh(x)

        return x
from torch import nn

import torch.nn.functional as F
import torch

from torch.nn import BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp["value"]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

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
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        padding=1,
        groups=1,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(UpBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)

        self.Lwarp = Lwarp
        if Lwarp:
            self.SameBlock2d = SameBlock2d(
                out_features, out_features, groups, kernel_size, padding, AdaINc=AdaINc
            )

    def forward(self, x, drv_exp=None):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        if self.Lwarp:
            out = self.SameBlock2d(out, drv_exp=drv_exp)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        padding=1,
        groups=1,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(DownBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        self.Lwarp = Lwarp
        if Lwarp:
            self.SameBlock2d = SameBlock2d(
                out_features, out_features, groups, kernel_size, padding, AdaINc=AdaINc
            )

    def forward(self, x, drv_exp=None):
        out = self.conv(x)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        if self.Lwarp:
            out = self.SameBlock2d(out, drv_exp=drv_exp)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(
        self, in_features, out_features, groups=1, kernel_size=3, padding=1, AdaINc=0, use_IN=False
    ):
        super(SameBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x, drv_exp=None):
        out = self.conv(x)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        return out


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias), nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1, 1)
        beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_features
                    if i == 0
                    else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                    Lwarp=Lwarp,
                    AdaINc=AdaINc,
                    use_IN=use_IN
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x, drv_exp=None):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1], drv_exp=drv_exp))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        dec_lease=0,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(dec_lease, num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(
                UpBlock2d(
                    in_filters,
                    out_filters,
                    kernel_size=3,
                    padding=1,
                    Lwarp=Lwarp,
                    AdaINc=AdaINc,
                    use_IN=use_IN
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = (
            out_filters + in_features if dec_lease == 0 else out_filters * 2
        )

    def forward(self, x, drv_exp=None, return_all=False):
        out = x.pop()
        if return_all:
            out_list = [out]
        for up_block in self.up_blocks:
            out = up_block(out, drv_exp=drv_exp)
            if return_all:
                out_list.append(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        if return_all:
            out_list.pop()
            out_list.append(out)
            return out, out_list
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        dec_lease=0,
        use_IN=False
    ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(
            block_expansion, in_features, num_blocks, max_features, Lwarp, AdaINc, use_IN
        )
        self.decoder = Decoder(
            block_expansion,
            in_features,
            num_blocks,
            dec_lease,
            max_features,
            Lwarp,
            AdaINc,
            use_IN
        )
        self.out_filters = self.decoder.out_filters

    def forward(self, x, drv_exp=None, return_all=False):
        return self.decoder(self.encoder(x, drv_exp=drv_exp), drv_exp=drv_exp, return_all=return_all)


class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(
                x,
                normalized_shape,
                self.weight.expand(normalized_shape),
                self.bias.expand(normalized_shape),
            )

        else:
            return F.layer_norm(x, normalized_shape)


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
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


if __name__ == '__main__':
    model = Hourglass(
            block_expansion=64,
            in_features=3,
            max_features=512,
            num_blocks=3,
            Lwarp=False,
            AdaINc=0,
            dec_lease=0,
        )
    print(model)
    x = torch.zeros((2, 3, 256, 256))
    out, out_list = model(x, return_all=True)
    print(out.shape)
    for t in out_list:
        print(t.shape)
