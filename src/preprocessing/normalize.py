import numpy as np
import numpy.typing as npt
from typing import Union, Optional, Dict, List, Tuple, Any
NDArray = npt.NDArray[np.float32]

def normalize(
    image: NDArray,
    modality: str,
    cropping_threshold_met: bool,
    dataset_stats: Optional[Tuple[float, float]] = None,
    clipping_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5),
) -> NDArray:
    """Normalizes images using nnU-net normalization strategy.

    For CT images: values are clipped using [.5 - 99.5] percentile values of non-background values,
    followed by z-score normilization using dataset wide statistics. Dataset stats cannot be None when
    modality is 'CT'

    For other modalities: z-score normilization applied to each sample

    If cropping reduced number of voxels by > 25% then only nonzero values are used for normalization

    Args:
        image: original image
        modality: image modality ('CT', 'mri', etc) found via dataset JSON
        cropping_threshold_met: whether the cropping threshold for alternate normalization is met
        dataset_stats: (mean, std) dataset statistics for dataset wide CT normalization
        clipping_percentiles: (.5 - 99.5) preconfigured clipping percentages for CT normalization
    Returns:
        NDArray: normalized image
    """
    if modality == "CT":
        mean, std = dataset_stats
        low, high = clipping_percentiles
        if cropping_threshold_met:
            # CT with cropping threshold met -> clip and nonzeros
            image[image != 0] = np.clip(image[image != 0], low, high)
            image[image != 0] = (image[image != 0] - mean) / std
        else:
            # CT without cropping threshold met -> clip only
            image = np.clip(image, low, high)
            image = (image - mean) / std
    else:
        if cropping_threshold_met:
            # Non-CT with cropping threshold met -> nonzeros
            nonzero = image[image != 0]
            mean, std = nonzero.mean(), nonzero.std()
            image[image != 0] = (image[image != 0] - mean) / std
        else:
            # Non-CT without cropping threshold met -> nothing special
            mean, std = image.mean(), image.std()
            image = (image - mean) / std
    return image