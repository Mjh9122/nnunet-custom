from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

MAX_3D_PATCH_VOXELS = 128**3
MIN_3D_BATCH_SIZE = 2
MAX_POOLING_LAYERS_2D = 6
MAX_POOLING_LAYERS_3D = 5
MAX_SINGLE_BATCH_VOXEL_PERCENTAGE = 0.05
INITIAL_CHANNELS = 32
MAX_3D_CHANNELS = 256
MAX_2D_CHANNELS = 512
STOPPING_SIZE = 8


def determine_2d_patch_batch(
    image_shape: Tuple[int, int, int],
    image_spacing: Tuple[float, float, float],
    total_voxels: int,
) -> Tuple[Tuple[int, int], int]:
    """Determines an appropriate patch and batch size for the 2d Unet.
       Starts with (256, 256) and 42 and adjusts based on gpu memory available and maximum voxels per step

       Adapts to median plane size of the image (using smallest in plane spacing, corrisponding to the highest resolution).
       Tries to train entire slices by applying the above rule.

       Patch size x Batch size < .05 x total_voxels

    Args:
        image_shape (Tuple[int, int, int]): dimmensions of incoming image
        image_spacing (Tuple[float, float, float]): spacing of incoming image
        total_voxels (int): total number of voxels in dataset

    Returns:
        Tuple[Tuple[int, int], int]: (patch size, batch size)
    """
    aspect_ratio = 1 / np.array(image_spacing)


def determine_3d_patch_batch(
    image_shape: Tuple[int, int, int], total_voxels: int
) -> Tuple[Tuple[int, int, int], int]:
    """Determines an appropriate patch and batch size for the 3d Unet
    Starts with (128, 128, 128) and 2.

    If # of voxels in image > MAX_3D_PATCH_VOXELS then choose largest patch size matching input aspect ratio s.t. # of voxels in image < MAX_3D_PATCH_VOXELS
    If # of voxels in image < MAX_3D_PATCH_VOXELS then use image_shape as patch size and grow batch size
    Minimum batch size is 2
    Patch size x Batch size < .05 x total_voxels

    Args:
        image_shape (Tuple[int, int, int]): dimmensions of incoming image
        total_voxels (int): total number of voxels in dataset

    Returns:
        Tuple[Tuple[int, int, int], int]: (patch size, batch size)
    """
    pass


def determine_pooling_operations(
    median_image_shape: Tuple[int, ...]
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


def determine_channels_per_layer(
    pooling_operations: Tuple[int, ...]
) -> Tuple[int, ...]:
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

    return tuple(
        [
            min(max_channels, INITIAL_CHANNELS * 2**i)
            for i in range(max(pooling_operations))
        ]
    )


def generate_network_topologies(
    median_image_shape: Tuple[int, int, int],
    cascade_image_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Dict, Dict, Dict]:
    """Generates network topologies for 2d, 3d, and 3d cascade (if necessary) unets. Orchistrates other topology generation functions.
    Logic for cascade shape is applied in preprocessing.

    Args:
        median_image_shape (Tuple[int, int, int]): Median dataset image shape
        cascade_image_shape (Optional[Tuple[int, int, int]], optional): Median shape of low res image for cascade. Defaults to None.

    Returns:
        Tuple[Dict, Dict, Dict]: Configs for 2D, 3D, and Cascade Unets
        Each dict contains:
        'model_type': 'Unet2d' | 'Unet3d' | 'CascadeUnet3d',
        'batch_size': int,
        'patch_size': Tuple[int, ...],
        'pooling_ops': Tuple[int, ...],
        'channels': List[int]

        In case that cascade should not be trained, batch + patch size, pooling ops and channels will be None
    """
    pass
