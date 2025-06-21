import numpy as np
import numpy.typing as npt

from typing import Union, Optional, Dict, List, Tuple, Any

MAX_3D_PATCH_VOXELS = 128**3
MIN_3D_BATCH_SIZE = 2
MAX_POOLING_LAYERS_2D = 6
MAX_POOLING_LAYERS_3D = 5
MAX_SINGLE_BATCH_VOXEL_PERCENTAGE = 0.05
MAX_CHANNELS = 30
STOPPING_SIZE = 8


def determine_2d_patch_batch(
    image_shape: Tuple[int, int, int], total_voxels: int
) -> Tuple[Tuple[int, int], int]:
    """Determines an appropriate patch and batch size for the 2d Unet.
       Starts with (256, 256) and 42 and adjusts based on gpu memory available and maximum voxels per step

       Adapts to median plane size of the image (using smallest in plane spacing, corrisponding to the highest resolution).
       Tries to train entire slices by applying the above rule.

       Patch size x Batch size < .05 x total_voxels

    Args:
        image_shape (Tuple[int, int, int]): dimmensions of incoming image
        total_voxels (int): total number of voxels in dataset

    Returns:
        Tuple[Tuple[int, int], int]: (patch size, batch size)
    """
    pass


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
    pass


def determine_channels_per_layer(pooling_operations: Tuple[int, ...]) -> List[int]:
    """Determines number of channels per layer
    Number of channels double each layer starting at 1 with a maximum of 30 channels.

    Args:
        pooling_operations (Tuple[int, ...]): Pooling operations per axis

    Returns:
        List[int]: list of channels per level
    """
    pass


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
