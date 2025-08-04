from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import os
import torch
import pickle as pkl
from pathlib import Path

MAX_3D_PATCH_VOXELS = 128**3
MIN_3D_BATCH_SIZE = 2
MAX_POOLING_LAYERS_2D = 6
MAX_POOLING_LAYERS_3D = 5
MAX_SINGLE_BATCH_VOXEL_PERCENTAGE = 0.05
INITIAL_CHANNELS = 32
MAX_3D_CHANNELS = 256
MAX_2D_CHANNELS = 512
STOPPING_SIZE = 8


def estimate_model_vram(channels: List[int], classes: int, dtype=torch.float32):
    """Estimates the amount of memory used by a 3d Unet model based on channel and classes. Will be slightly off
    when estimating parameter count on nets that have unequal pooling operations along spacial dimensions

    Args:
        channels (List[int]): Channels after each down block. channels[0] should have the number of channels in orig. image
        classes (int): Number of output classes
        dtype (_type_, optional): Datatype of parameter weights. Defaults to torch.float32.

    Returns:
        int: number of bytes used up by a model with these specs
    """
    param_size = dtype.itemsize

    def params_per_down_block(in_channels, out_channels):
        filter_params1 = 3 * 3 * 3 * in_channels + 1 + 2
        filter_params2 = 3 * 3 * 3 * out_channels + 1 + 2

        return (filter_params1 + filter_params2) * out_channels

    def params_per_up_block(in_channels, out_channels):
        transposed_conv_params = (2 * 2 * 2 * in_channels + 1) * in_channels
        filter_params1 = int(3 * 3 * 3 * in_channels * 1.5) + 1 + 2
        filter_params2 = int(3 * 3 * 3 * out_channels) + 1 + 2

        return (filter_params1 + filter_params2) * out_channels + transposed_conv_params

    ins = channels[:-1]
    outs = channels[1:]
    params_down = sum(params_per_down_block(i, o) for i, o in zip(ins, outs))

    ins = channels[-1:1:-1]
    outs = channels[-2:0:-1]
    params_up = sum(params_per_up_block(i, o) for i, o in zip(ins, outs))
    final_conv_params = (channels[1] + 1) * classes

    return (params_down + params_up + final_conv_params) * param_size


def estimate_batch_vram(
    batch_size: int,
    channels: int,
    input_channels: int,
    spatial_dims: Tuple[int, int, int],
    c: float = 3.0,
    dtype: torch.dtype = torch.float32,
):
    """Estimates (calculates) the size of the largest data tensor as it passes through the U-net. This should occur when the
    feature map from the first down block passes through the skip connection and is concatenated to the corresponding conv block input
    the final up block. Typically the channels will be 32 at the end of the first down block, and 64 entering the final up block.

    Args:
        batch_size (int): number of patches per batch
        channels (int): number of channels (total) in largest tensor (skip + up block)
        input_channels (int): number of channels in the original image (stored concurrently and scales with batch)
        spatial_dims (Tuple[int, int, int]): spacial dimensions of the largest tensor (typically patch size)
        c (float, optional): constant multiplier to adjust for workspace memory as a factor of input tensor size. Defaults to 3.0
        dtype (torch.dtype, optional): dtype of tensors for single element size calculation. Defaults to torch.float32.

    Returns:
        int: total number of bytes in the largest tensor
    """
    conv_input_tensor = channels * np.prod(spatial_dims) * dtype.itemsize
    conv_output_tensor = 1 / 3 * channels * np.prod(spatial_dims) * dtype.itemsize
    net_input_tensor = input_channels * np.prod(spatial_dims) * dtype.itemsize
    work_space_memory = c * conv_input_tensor

    return batch_size * (
        conv_input_tensor + conv_output_tensor + net_input_tensor + work_space_memory
    )


def determine_3d_patch_batch(
    image_shape: Tuple[int, int, int],
    image_channels: int,
    total_voxels: int,
    mem_target_gb: int = 8,
    output_classes: int = 8,
) -> Tuple[Tuple[int, int, int], int]:
    """Determines an appropriate patch and batch size for the 3d Unet
    Starts with (128, 128, 128) and 2.

    If # of voxels in image > MAX_3D_PATCH_VOXELS then choose largest patch size matching input aspect ratio s.t. # of voxels in image < MAX_3D_PATCH_VOXELS
    If # of voxels in image < MAX_3D_PATCH_VOXELS then use image_shape as patch size and grow batch size
    Minimum batch size is 2
    Patch size x Batch size < .05 x total_voxels

    Args:
        image_shape (Tuple[int, int, int]): dimensions of incoming image
        image_channels (int): channels of incoming image
        total_voxels (int): total number of voxels in dataset (estimated by median size * num images)
        mem_target_gp (int: optional): available vram in Gb for training. Defaults to 8
        output_classes (int: optional): number of classes in target. Defaults to 10 (overestimate)
    Returns:
        Tuple[Tuple[int, int, int], int]: (patch size, batch size)
    """
    batch_size = 2

    # make size under cap
    if np.prod(image_shape) <= MAX_3D_PATCH_VOXELS:
        patch_size = image_shape
    else:
        needed_reduction = (np.prod(image_shape) / MAX_3D_PATCH_VOXELS) ** (1 / 3)
        patch_size = tuple(round(dim / needed_reduction) for dim in image_shape)

    pooling_ops = determine_pooling_operations(patch_size)
    channels = determine_channels_per_layer(pooling_ops)
    channels.insert(0, image_channels)
    max_mem_channels = 96 if len(channels) < 3 else (channels[2] * 3 / 2)

    new_patch_size = []
    for dim, n in zip(patch_size, pooling_ops):
        if dim % (2**n) != 0:
            if dim % (2**n) >= 2 ** (n - 1):
                new_patch_size.append(dim + (2**n - dim % (2**n)))
            else:
                new_patch_size.append(dim - (dim % (2**n)))
        else:
            new_patch_size.append(dim)

    while np.prod(new_patch_size) > MAX_3D_PATCH_VOXELS:
        min_channel = np.argmin(new_patch_size)
        new_patch_size[min_channel] -= 2**n

    patch_size = tuple(new_patch_size)

    for dim, n in zip(patch_size, pooling_ops):
        assert dim % 2**2 == 0

    # Arbitrarily cap memory at 95% of goal, to allow some wiggle room
    memory_target = (mem_target_gb * 1024**3) * 0.95
    model_vram = estimate_model_vram(channels, output_classes)
    model_vram *= 4  # adam stores momentum and velocity for each param and we need gradients for all

    mem_for_data = memory_target - model_vram
    single_image_memory = estimate_batch_vram(
        1, max_mem_channels, image_channels, patch_size
    )

    memory_max_batch = int(mem_for_data / single_image_memory)
    data_vram = estimate_batch_vram(
        memory_max_batch, max_mem_channels, image_channels, patch_size
    )

    assert model_vram + data_vram < memory_target
    batch_size = memory_max_batch

    # Hard upper bound on batch size
    max_batch_5_percent = int((0.05 * total_voxels) / np.prod(patch_size))
    if batch_size > max_batch_5_percent:
        batch_size = max_batch_5_percent

    return patch_size, batch_size


def determine_pooling_operations(
    median_image_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Determines the number of pooling operations applied to each axis. Stops when axis length is less than STOPPING_SIZE or the
    number of pooling operations is higher than 5 or 6 depending on 2d vs 3d nets.

    Args:
        median_image_shape (Union[Tuple[int, int], Tuple[int, int, int]]): Median image input shape, provides starting size before pooling

    Returns:
        Tuple[int, ...]: pooling operators per axis
    """
    dims = len(median_image_shape)

    if dims == 2:
        pools = tuple(
            [
                int(min(np.log2(x) - 2, MAX_POOLING_LAYERS_2D))
                for x in median_image_shape
            ]
        )
    else:
        pools = tuple(
            [
                int(min(np.log2(x) - 2, MAX_POOLING_LAYERS_2D))
                for x in median_image_shape
            ]
        )

    return pools


def determine_channels_per_layer(pooling_operations: Tuple[int, ...]) -> List[int]:
    """Determines number of channels per layer
    Number of channels double each layer starting at the number of initial channels with a maximum of 30 channels.

    Args:
        initial_channels (int): Number of channels in the original image
        pooling_operations (Tuple[int, ...]): Pooling operations per axis

    Returns:
        Tuple[int, ...]: list of channels per level
    """
    num_dims = len(pooling_operations)

    if num_dims == 2:
        max_channels = MAX_2D_CHANNELS
    else:
        max_channels = MAX_3D_CHANNELS

    return [
        min(max_channels, INITIAL_CHANNELS * 2**i)
        for i in range(max(pooling_operations) + 1)
    ]


def load_dataset_stats(pickle_path: Path) -> Tuple[Tuple[int, int, int], int]:
    """Gets median image shape and estimated total voxels from dataset stat pickle

    Args:
        pickle_path (Path): path to dataset stat pickle

    Returns:
        Tuple[Tuple[int, int, int], int]: median image shape (int, int, int), estimated total voxels (int)
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file missing at {pickle_path}")

    with open(pickle_path, "rb") as f:
        stats = pkl.load(f)

    assert "num_images" in stats
    assert "post_resample_shape" in stats

    total_voxels = np.prod(stats.get("post_resample_shape")) * stats.get("num_images")

    return stats.get("post_resample_shape"), total_voxels
