import sys
from pathlib import Path

import numpy as np
import pytest
import torch

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.models import Conv3DBlock, DownBlock3D, Unet3D, UpBlock3D

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


@pytest.mark.parametrize(
    "input_shape, in_channels, out_channels, expected_shape",
    (
        ((4, 1, 64, 64, 64), 1, 2, (4, 2, 64, 64, 64)),
        ((4, 2, 64, 64, 64), 2, 4, (4, 4, 64, 64, 64)),
        ((4, 4, 64, 64, 64), 4, 8, (4, 8, 64, 64, 64)),
        ((4, 8, 64, 64, 64), 8, 16, (4, 16, 64, 64, 64)),
        ((4, 16, 64, 64, 64), 16, 32, (4, 32, 64, 64, 64)),
        ((4, 16, 64, 32, 32), 16, 32, (4, 32, 64, 32, 32)),
        ((4, 256, 64, 64, 64), 256, 1, (4, 1, 64, 64, 64)),
        ((4, 1, 64, 64, 64), 1, 256, (4, 256, 64, 64, 64)),
    ),
)
def test_Conv3DBlock(input_shape, in_channels, out_channels, expected_shape):
    input = torch.randn(input_shape).to(device)
    block = Conv3DBlock(in_channels, out_channels).to(device)
    output = block(input)
    assert output.shape == expected_shape


def test_Conv3DBlock_bad_args():
    with pytest.raises(RuntimeError):
        Conv3DBlock(-1, 16)

    with pytest.raises(RuntimeError):
        Conv3DBlock(16, -1)


@pytest.mark.parametrize(
    "input_shape, in_channels, out_channels, kernel_size, stride, expected_output_shape, expected_skip_shape",
    (
        (
            (4, 1, 128, 128, 128),
            1,
            32,
            (2, 2, 2),
            (2, 2, 2),
            (4, 32, 64, 64, 64),
            (4, 32, 128, 128, 128),
        ),
        (
            (4, 32, 64, 64, 64),
            32,
            64,
            (2, 2, 2),
            (2, 2, 2),
            (4, 64, 32, 32, 32),
            (4, 64, 64, 64, 64),
        ),
        (
            (4, 64, 32, 32, 32),
            64,
            128,
            (2, 2, 2),
            (2, 2, 2),
            (4, 128, 16, 16, 16),
            (4, 128, 32, 32, 32),
        ),
        (
            (4, 128, 16, 16, 16),
            128,
            256,
            (2, 2, 2),
            (2, 2, 2),
            (4, 256, 8, 8, 8),
            (4, 256, 16, 16, 16),
        ),
        (
            (4, 1, 64, 64, 64),
            1,
            2,
            (4, 2, 1),
            (4, 2, 1),
            (4, 2, 16, 32, 64),
            (4, 2, 64, 64, 64),
        ),
        (
            (4, 1, 64, 64, 64),
            1,
            256,
            (4, 4, 4),
            (4, 4, 4),
            (4, 256, 16, 16, 16),
            (4, 256, 64, 64, 64),
        ),
    ),
)
def test_DownBlock3D(
    input_shape,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    expected_output_shape,
    expected_skip_shape,
):
    input = torch.randn(input_shape).to(device)
    block = DownBlock3D(in_channels, out_channels, kernel_size, stride).to(device)
    out, skip = block(input)
    assert out.shape == expected_output_shape
    assert skip.shape == expected_skip_shape


@pytest.mark.parametrize(
    "input_shape, in_channels, skip_channels, out_channels, kernel_size, stride, expected_shape",
    (
        ((4, 512, 8, 8, 8), 512, 256, 256, (2, 2, 2), (2, 2, 2), (4, 256, 16, 16, 16)),
        (
            (4, 256, 16, 16, 16),
            256,
            128,
            128,
            (2, 2, 2),
            (2, 2, 2),
            (4, 128, 32, 32, 32),
        ),
        ((4, 128, 32, 32, 32), 128, 64, 64, (2, 2, 2), (2, 2, 2), (4, 64, 64, 64, 64)),
        # ((4, 64, 64, 64, 64), 64, 32, 32, 2, (4, 32, 128, 128, 128)), # TOO LARGE FOR LAPTOP CPU MEM
        ((4, 2, 32, 32, 32), 2, 1, 1, (4, 2, 1), (4, 2, 1), (4, 1, 128, 64, 32)),
    ),
)
def test_UpBlock3D(
    input_shape,
    in_channels,
    skip_channels,
    out_channels,
    kernel_size,
    stride,
    expected_shape,
):
    input = torch.randn(input_shape).to(device)

    skip_shape = list(input.shape)
    skip_shape[-3:] = [sf * s for sf, s in zip(kernel_size, skip_shape[-3:])]
    skip_shape[-4] = skip_channels

    skip = torch.randn(*skip_shape).to(device)

    block = UpBlock3D(in_channels, skip_channels, out_channels, kernel_size, stride).to(
        device
    )
    out = block(input, skip)
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape, channels, pooling_ops, num_classes, expected_shape",
    (
        ((4, 1, 64, 64, 64), [1, 16, 32, 64, 128], [3, 3, 3], 3, (4, 3, 64, 64, 64)),
        ((4, 1, 32, 64, 64), [1, 16, 32, 64, 128], [2, 3, 3], 3, (4, 3, 32, 64, 64)),
        ((4, 1, 16, 64, 64), [1, 16, 32, 64, 128], [1, 3, 3], 3, (4, 3, 16, 64, 64)),
        ((4, 1, 8, 64, 64), [1, 16, 32, 64, 128], [0, 4, 4], 3, (4, 3, 8, 64, 64)),
        ((4, 1, 64, 16, 16), [1, 16, 32, 64, 128], [3, 1, 1], 3, (4, 3, 64, 16, 16)),
        ((4, 1, 64, 64, 64), [1, 16, 32, 64, 128], [3, 3, 3], 16, (4, 16, 64, 64, 64)),
        ((4, 1, 64, 64, 64), [1, 16, 32, 64, 128], [3, 3, 3], 1, (4, 1, 64, 64, 64)),
    ),
)
def test_Unet3D(
    input_shape,
    channels,
    pooling_ops,
    num_classes,
    expected_shape,
):
    input = torch.randn(input_shape).to(device)

    net = Unet3D(channels, pooling_ops, num_classes).to(device)
    out = net(input)
    assert out.shape == expected_shape
