import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from topology_generation.topology_generation import (
    estimate_model_vram,
    estimate_batch_vram,
    determine_3d_patch_batch,
    determine_channels_per_layer,
    determine_pooling_operations,
    MAX_3D_PATCH_VOXELS
)

from models.models import Unet3D


@pytest.mark.parametrize(
    "dims, expected",
    (
        ((192, 160), (5, 5)),
        ((128, 128, 128), (5, 5, 5)),
        ((320, 256), (6, 6)),
        ((80, 192, 128), (4, 5, 5)),
        ((64, 160, 128), (4, 5, 5)),
        ((512, 512), (6, 6)),
        ((56, 40), (3, 3)),
        ((40, 56, 40), (3, 3, 3)),
        ((320, 320), (6, 6)),
        ((20, 192, 192), (2, 5, 5)),
        ((112, 128, 128), (4, 5, 5)),
        ((96, 160, 128), (4, 5, 5)),
    ),
)
def test_determine_pooling_ops(dims, expected):
    result = determine_pooling_operations(dims)
    assert result == expected


@pytest.mark.parametrize(
    "pooling_ops, expected",
    (
        ((3, 3, 3), [32, 64, 128]),
        ((3, 3), [32, 64, 128]),
        ((5, 5, 5), [32, 64, 128, 256, 256]),
        ((5, 5), [32, 64, 128, 256, 512]),
        ((4, 5, 5), [32, 64, 128, 256, 256]),
        ((6, 6), [32, 64, 128, 256, 512, 512]),
        ((2, 5, 5), [32, 64, 128, 256, 256]),
    ),
)
def test_channels_per_layer(pooling_ops, expected):
    result = determine_channels_per_layer(pooling_ops)
    assert result == expected


@pytest.mark.parametrize(
    "batch_size, channels, image_size, dtype",
    (
        (2, 96, (128, 128, 128), torch.float32),
        (2, 96, (80, 192, 128), torch.float32),
        (9, 96, (40, 56, 40), torch.float32),
        (4, 96, (20, 192, 192), torch.float32),
        (2, 96, (96, 160, 128), torch.float32),
    ),
)
def test_estimate_batch_vram(batch_size, channels, image_size, dtype):
    tensor = torch.randn(size=(batch_size, channels, *image_size), dtype=dtype)

    actual_mem = tensor.numel() * tensor.itemsize

    estimated_mem = estimate_batch_vram(batch_size, channels, image_size, dtype)

    assert actual_mem == estimated_mem


@pytest.mark.parametrize(
    "channels, classes",
    (
        ([1, 16, 32, 64, 128, 256, 512], 2),
        ([1, 16, 32, 64, 128, 256, 512], 5),
        ([1, 16, 32, 64, 128, 256, 512], 64),
        ([1, 16, 32, 64, 128, 256], 3),
        ([1, 16, 32, 64, 128], 3),
        ([1, 16, 32, 64], 3),
        ([1, 16, 32], 2),
    ),
)
def test_estimate_model_vram(channels, classes):
    pooling_ops = tuple([len(channels) - 2 for _ in range(3)])
    model = Unet3D(channels, pooling_ops, classes)

    result = estimate_model_vram(channels, classes)
    expected = sum(p.numel() for p in model.parameters()) * torch.float32.itemsize

    assert result == expected


@pytest.mark.parametrize(
    "image_size, total_voxels, mem_target_gb",
    (
        ((138, 169, 138), 750 * 138 * 169 * 138, 8),
        ((115, 320, 232), 30 * 115 * 320 * 232, 8),
        ((20, 320, 319), 48 * 20 * 320 * 319, 8),
        ((482, 512, 512), 201 * 482 * 512 * 512, 8),
        ((36, 50, 35), 394 * 36 * 50 * 35, 8),
        ((252, 512, 512), 96 * 252 * 512 * 512, 8),
        ((96, 512, 512), 420 * 96 * 512 * 512, 8),
    ),
)
def test_determine_3d_patch_batch(image_size, total_voxels, mem_target_gb):
    result_patch, result_batch = determine_3d_patch_batch(image_size, total_voxels, mem_target_gb)
    
    result_pool_ops = determine_pooling_operations(result_patch)
    result_channels = determine_channels_per_layer(result_pool_ops)
    # Test all specs from paper

    assert np.prod(result_patch) * result_batch <= .05 * total_voxels
    assert np.prod(result_batch) <= MAX_3D_PATCH_VOXELS

    for dim, n in zip(result_patch, result_pool_ops):
        assert dim % (2 ** n) == 0

    assert estimate_batch_vram(result_batch, channels = 96, spatial_dims = result_patch) + \
           estimate_model_vram(result_channels, classes = 10) < mem_target_gb * 1024 ** 3
