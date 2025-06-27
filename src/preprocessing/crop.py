import numpy as np
import numpy.typing as npt
from typing import Union, Optional, Dict, List, Tuple, Any
NDArray = npt.NDArray[np.float32]

def crop_zeros(image: NDArray, mask: NDArray) -> Tuple[NDArray, NDArray]:
    """Creates bounding box of nonzero values in an image. Returns cropped image
    Args:
        image (NDArray): original image
        mask (NDArray): segmentation mask of image

    Returns:
        Tuple[NDArray, NDArray]: cropped image, cropped mask
    """
    if image.shape != mask.shape:
        raise Exception(f'Image shape {image.shape} must match mask shape {mask.shape}')

    # Find indeces of all nonzero values
    nonzero = np.array(np.where(image != 0))
    # Min and max nonzeros for each axis, transpose to get min/max pairs
    nonzero = np.array([np.min(nonzero, 1), np.max(nonzero, 1)]).T

    slices = tuple(slice(a, b + 1) for a, b in nonzero)
    
    cropped_img = image[slices]
    cropped_mask = mask[slices]

    return cropped_img, cropped_mask