import torch
from torch import nn

from typing import Union, List
from ..common import layers
from dataclasses import dataclass


class Discriminator(nn.Module):
    def __init__(self,
                 num_channels: int,
                 max_channels: int,
                 num_blocks: int,
                 input_channels: int,
                 norm_layer='in'):
        super(Discriminator, self).__init__()
        self.num_blocks = num_blocks

        self.in_channels = [min(num_channels * 2 ** (i - 1), max_channels) for i in range(self.num_blocks)]
        self.in_channels[0] = input_channels

        self.out_channels = [min(num_channels * 2 ** i, max_channels) for i in range(self.num_blocks)]
        self.norm_layer = norm_layer
        self.init_networks()

        # print(self.in_channels, self.out_channels)

    def init_networks(self) -> None:
        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            self.blocks.append(
                layers.blocks['conv'](
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels[i],
                    kernel_size=3,
                    padding=1,
                    stride=2 if i < self.num_blocks - 1 else 1,
                    norm_layer_type=self.norm_layer,
                    activation_type='lrelu'))

        self.to_scores = nn.Conv2d(
            in_channels=self.out_channels[-1],
            out_channels=1,
            kernel_size=1)

    def forward(self, inputs):
        outputs = inputs
        features = []

        for block in self.blocks:
            outputs = block(outputs)
            features.append(outputs)

        scores = self.to_scores(outputs)

        return scores, features


class MultiScaleDiscriminator(nn.Module):

    @dataclass
    class Config:
        min_channels: int
        max_channels: int
        num_blocks: int
        input_channels: int
        input_size: int
        num_scales: int
        norm_layer: str = 'in'

    @classmethod
    def create_default(
        cls,
        input_channels: int = 3,
        input_size: int = 256,
    ) -> 'MultiScaleDiscriminatorConfig':
        """Create a default configuration with commonly used settings.
        
        Args:
            input_channels: Number of input channels (default: 3)
            input_size: Input image size (default: 256)
            
        Returns:
            Default configuration instance
        """
        return cls.Config(
            min_channels=64,
            max_channels=512,
            num_blocks=4,
            input_channels=input_channels,
            input_size=input_size,
            num_scales=3,
            norm_layer='in'
        )


    def __init__(self, cfg:Config) -> None:
        super(MultiScaleDiscriminator, self).__init__()

        self.cfg = cfg

        self.input_size = self.cfg.input_size
        self.num_scales = self.cfg.num_scales

        spatial_size = self.cfg.input_size
        self.nets = []

        for i in range(self.cfg.num_scales):
            net = Discriminator(self.cfg.min_channels, self.cfg.max_channels, self.cfg.num_blocks, self.cfg.input_channels, norm_layer=self.cfg.norm_layer)

            setattr(self, 'net_%04d' % spatial_size, net)
            self.nets.append(net)

            spatial_size //= 2

        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, inputs: torch.Tensor): # -> (List[List[torch.Tensor]], List[List[List[torch.Tensor]]])
        spatial_size = self.input_size
        scores, features = [], []

        for i in range(self.num_scales):
            net = getattr(self, 'net_%04d' % spatial_size)
            scores_i, features_i = net(inputs)
            
            scores.append([scores_i])
            features.append([[features_i_block] for features_i_block in features_i])

            spatial_size //= 2
            inputs = self.down(inputs)

        return scores, features
