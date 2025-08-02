from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint


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
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
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
        pool_shape (Tuple[int, int, int], optional): size of pooling operation. Defaults to (2, 2, 2)
        pool_stride (Tuple[int, int, int], optional): stride of pooling operation. Defaults to (2, 2, 2)

    Returns:
        Tuple (Tensor, Tensor): pooled output, skip connection feature map
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: Tuple[int, int, int] = (2, 2, 2),
        pool_stride: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()

        self.conv_block1 = Conv3DBlock(in_channels, out_channels)
        self.conv_block2 = Conv3DBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride)

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
        tconv_kernel_shape (Tuple[int, int, int], optional): Transposed Conv. shape.S Defaults to (2, 2, 2).
        tconv_stride (Tuple[int, int, int], optional): Transposed Conv. stride. Defaults to (2, 2, 2).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        tconv_kernel_shape: Tuple[int, int, int] = (2, 2, 2),
        tconv_stride: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=tconv_kernel_shape,
            stride=tconv_stride,
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
        pooling_ops (Tuple[int, int, int]): Number of pooling ops along each spacial dimension. Ops are applied the the first n blocks.
        num_classes (int): Number of classes to predict in the final convolution layer

    """

    def __init__(
        self, channels: List[int], pooling_ops: Tuple[int, int, int], num_classes: int
    ):
        super().__init__()

        pool_ops_arr = np.array(
            [
                np.concat((np.ones(n, int), np.zeros(np.max(pooling_ops) - n, int)))
                for n in pooling_ops
            ]
        )
        pool_ops_arr += 1
        shapes_and_strides = [tuple(ops) for ops in pool_ops_arr.T]

        self.down_blocks = nn.ModuleList(
            [
                DownBlock3D(in_channels, out_channels, shape_stride, shape_stride)
                for (in_channels, out_channels, shape_stride) in zip(
                    channels[:-2], channels[1:-1], shapes_and_strides
                )
            ]
        )
        self.bottom_conv1 = Conv3DBlock(channels[-2], channels[-1])
        self.bottom_conv2 = Conv3DBlock(channels[-1], channels[-1])
        self.up_blocks = nn.ModuleList(
            [
                UpBlock3D(
                    in_channels, skip_channels, out_channels, shape_stride, shape_stride
                )
                for (in_channels, skip_channels, out_channels, shape_stride) in zip(
                    channels[-1:1:-1],
                    channels[-2:0:-1],
                    channels[-2:0:-1],
                    shapes_and_strides[::-1],
                )
            ]
        )
        self.final_conv = nn.Conv3d(
            in_channels=channels[1], out_channels=num_classes, kernel_size=1
        )

    def _checkpoint_up_block(self, up_block, x, skip_features):
        return up_block(x, skip_features)

    def _checkpoint_down_block(self, down_block, x):
        return down_block(x)

    def _checkpoint_conv_block(self, conv_block, x):
        return conv_block(x)

    def forward(self, x):
        skip_con_features = []
        for down_block in self.down_blocks:
            x, skip = checkpoint(
                self._checkpoint_down_block, down_block, x, use_reentrant=False
            )
            skip_con_features.append(skip)

        x = checkpoint(
            self._checkpoint_conv_block, self.bottom_conv1, x, use_reentrant=False
        )
        x = checkpoint(
            self._checkpoint_conv_block, self.bottom_conv2, x, use_reentrant=False
        )

        for up_block in self.up_blocks:
            feature_map = skip_con_features.pop()
            x = checkpoint(
                self._checkpoint_up_block, up_block, x, feature_map, use_reentrant=False
            )
            del feature_map

        x = checkpoint(
            self._checkpoint_conv_block, self.final_conv, x, use_reentrant=False
        )

        return x
