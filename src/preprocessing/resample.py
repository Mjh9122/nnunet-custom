import numpy as np
import numpy.typing as npt
from typing import Union, Optional, Dict, List, Tuple, Any

NDArray = npt.NDArray[np.float32]
from skimage.transform import resize


def resample_image(
    image: NDArray,
    old_spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float],
    is_segmentation: bool,
) -> NDArray:
    """Resamples image to new voxel spacing using cubic spline interpolation or nearest
    neighbor (cublic spline order 0) based on whether an image or segmentation mask is passed in

    Args:
        image (NDArray): original image
        old_spacing (Tuple[float, float, float]): original voxel spacing (X, Y, Z)
        new_spacing (Tuple[float, float, float]): new voxel spacing (X, Y, Z)
        is_segmentation: True if segmentation map is inputed

    Returns:
        NDArray: resampled image
    """
    if len(old_spacing) != len(new_spacing):
        raise (Exception("New and old spacings must have same length"))

    old_dims = np.array(image.shape)

    if len(old_dims) != len(new_spacing):
        raise (Exception("New spacing and image shape must have same length"))

    new_dims = np.array(
        [
            int(dim * old / new)
            for dim, old, new in zip(old_dims, old_spacing, new_spacing)
        ]
    )

    if is_segmentation:
        reshaped = np.round(
            resize(image, new_dims, 0, mode="edge", anti_aliasing=False)
        ).astype(int)
    else:
        reshaped = resize(image, new_dims, 3, mode="edge", anti_aliasing=False)
    return reshaped
