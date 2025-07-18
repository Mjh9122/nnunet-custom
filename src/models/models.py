from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """Most basic 3D Unet building block
    (3D convolution) -> (instance normalization) -> (leaky ReLU)


    Args:
        in_channels (int): Number of channels in input
        out_channels (int): Number of channels in output
        kernel_size (int, optional): Convolution kernel size. Defaults to 3
        stride (int, optional): Stride for convolution blocks. Defaults to 1
        padding (int, optional): Padding size. Defaults to 1
        negative_slope (float, optional): LeakyReLU negative slope. Defaults to 0.01
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DownBlock3D(nn.Module):
    """Down block for the encoder side of the 3D Unet.
    Applies two 3D convolution blocks followed by a 3d max pooling operation.

    Args:
        in_channels (int): Number of channels in input
        out_channels (int): Number of channels in output
        pool_size (int, optional): size of pooling operation. Defaults to 2

    Returns:
        Tuple (Tensor, Tensor): pooled output, skip connection feature map
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 2):
        super().__init__()

        self.conv_block1 = Conv3DBlock(in_channels, out_channels)
        self.conv_block2 = Conv3DBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(pool_size)

    def forward(self, x):
        x = self.conv_block1(x)
        skip_features = self.conv_block2(x)
        pooled_output = self.pool(skip_features)
        return pooled_output, skip_features


class UpBlock3D(nn.Module):
    """Up block for the decoder side of the 3D Unet.
    Applies transposed conv 3d, concatenates the skip connection feature map, and applies two 3D convolution blocks

    Args:
        in_channels (int): Number of channels in input
        skip_channels (int): Number of channels from skip connection
        out_channels (int): Number of channels in output
        scale_factor (int, optional): How much to upscale image. Defaults to 2.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=scale_factor,
            stride=scale_factor,
        )
        self.conv_block1 = Conv3DBlock(in_channels + skip_channels, out_channels)
        self.conv_block2 = Conv3DBlock(out_channels, out_channels)

    def forward(self, x, skip_con):
        x = self.upconv(x)
        x = torch.cat([x, skip_con], dim=1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class Unet3D(nn.Module):
    """Implementation of 3D U-net following nnu-net implementation.
    Takes tensors with dims (batch, channels, depth, height, width)
    Follows the general structure of the 2016 3D u-net paper, with modifications from the 2018 nnu-net paper.
    Uses instance norm and leaky relu as per the nnu-net paper.

    Built using a series of down convolution blocks and up convolution blocks with skip connections.
    Final class logits are computed by a 1x1x1 convolution across all channels left by the final upconv block.

    Args:
        channels (List[int, ...]): Number of channels expected after each layer of downconv blocks.
                                   Channels are expected to double each layer up to a specified cap.
                                   Upconv blocks utilize the listed channels in reverse.
        num_classes (int): Number of classes to predict in the final convolution layer

    TODO variable pooling layers by dimension

    """

    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [
                DownBlock3D(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-2], channels[1:-1])
            ]
        )
        self.bottom_conv1 = Conv3DBlock(channels[-2], channels[-1])
        self.bottom_conv2 = Conv3DBlock(channels[-1], channels[-1])
        self.up_blocks = nn.ModuleList(
            [
                UpBlock3D(in_channels, skip_channels, out_channels)
                for (in_channels, skip_channels, out_channels) in zip(
                    channels[-1:1:-1], channels[-2:0:-1], channels[-2:0:-1]
                )
            ]
        )
        self.final_conv = nn.Conv3d(
            in_channels=channels[1], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        skip_con_features = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_con_features.append(skip)

        x = self.bottom_conv1(x)
        x = self.bottom_conv2(x)

        for up_block, feature_map in zip(self.up_blocks, skip_con_features[::-1]):
            x = up_block(x, feature_map)

        x = self.final_conv(x)

        return x
