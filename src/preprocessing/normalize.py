import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import os

from typing import Union, Optional, Dict, List, Tuple, Any
from pathlib import Path

NDArray = npt.NDArray[np.float32]


def normalize(
    image: NDArray,
    modality: str,
    cropping_threshold_met: bool,
    dataset_stats: Optional[Tuple[float, float]] = None,
    clipping_percentiles: Optional[Tuple[float, float]] = None,
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
        if dataset_stats is None:
            raise Exception("dataset_stats cannot be None for CT datasets")
        mean, std = dataset_stats

        if clipping_percentiles is None:
            raise Exception("clipping_percentiles cannot be None for CT datasets")
        low, high = clipping_percentiles

        # Ignore cropping threshold for CT images (normalize based on targets)
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

def normalize_dataset(
    dataset_dir: Path, output_dir: Path, dataset_stats: Dict
) -> None:
    """ Normalize all the cropped images in a dataset

    Args:
        dataset_dir (Path): path to images
        output_dir (Path): path to place normalized images
        modality (str): image modality
        cropping_threshold_met (bool): whether 25% threshold for cropping was met
        dataset_stats (Dict): statistics of dataset
    """
    images = [file for file in os.listdir(dataset_dir) if file[-3:] != 'pkl']
    modality = dataset_stats['modality']
    cropping_threshold_met = dataset_stats['cropping_threshold_met']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for image in images:
        img = sitk.ReadImage(dataset_dir / image)
        img_np = sitk.GetArrayFromImage(img)
        if modality == 'CT':
            img_normalized_np = normalize(img_np, modality, cropping_threshold_met, dataset_stats['stats'], dataset_stats['percentiles'])
        else:
            img_normalized_np = normalize(img_np, modality, cropping_threshold_met)

        img_normalized = sitk.GetImageFromArray(img_normalized_np)

        sitk.WriteImage(img_normalized, output_dir / image)