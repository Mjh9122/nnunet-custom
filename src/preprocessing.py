import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import json
import os

from skimage.transform import resize
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

np.random.seed(42)

NDArray = npt.NDArray[np.float32]
CT_CLIP_VALUES = (0.5, 99.5)
MAX_3D_PATCH_SIZE = (256, 256, 256)
MIN_3D_BATCH_SIZE = 2
CROPPING_NORMALIZATION_THRESHOLD = 0.25


def load_data(filepath: Path) -> NDArray:
    """Loads data from image file into numpy array for preprocessing

    Args:
        filepath (Path): filepath to image

    Returns:
        NDArray: image
    """
    if not os.path.exists(filepath):
        raise (FileNotFoundError)

    img = sitk.ReadImage(filepath)
    img_arr = sitk.GetArrayFromImage(img)

    return img_arr


def crop_zeros(image: NDArray) -> Tuple[NDArray, bool]:
    """Creates bounding box of nonzero values in an image. Also returns whether the total # of voxels
    is reduced by more than 25%, triggering different normalization strategy later on. This function borrows
    heavily from the batchgenerators example here: https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_preprocessing.py

    Args:
        image (NDArray): original image

    Returns:
        Tuple[NDArray, bool]: cropped image, reduction > 25%
    """
    # Find indeces of all nonzero values
    nonzero = np.array(np.where(image != 0))
    # Min and max nonzeros for each axis, transpose to get min/max pairs
    nonzero = np.array([np.min(nonzero, 1), np.max(nonzero, 1)]).T

    slices = tuple(slice(a, b + 1) for a, b in nonzero)
    cropped = image[slices]

    reduction_ratio = cropped.size / image.size

    return cropped, reduction_ratio < 0.75


def resample_image(
    image: NDArray,
    old_spacing: Tuple[float, float, float],
    new_spacing: Tuple[float, float, float],
    is_segmentation: bool
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
        raise(Exception('New and old spacings must have same length'))
    
    old_dims = np.array(image.shape)

    if len(old_dims) != len(new_spacing):
        raise(Exception('New spacing and image shape must have same length'))
    
    new_dims = np.array([int(dim * old / new) for dim, old, new in zip(old_dims, old_spacing, new_spacing)])

    if is_segmentation:
        reshaped = np.round(resize(image, new_dims, 0, mode = 'edge', anti_aliasing=False)).astype(int)
    else:
        reshaped = resize(image, new_dims, 3, mode = 'edge', anti_aliasing=False)
    return reshaped

def normalize(
    image: NDArray,
    modality: str,
    cropping_threshold_met: bool,
    dataset_stats: Optional[Tuple[float, float]] = None,
    clipping_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5),
    mask: Optional[NDArray] = None,
) -> NDArray:
    """Normalizes images using nnU-net normalization strategy.

    For CT images: values are clipped using [.5 - 99.5] percentile values of non-background values,
    followed by z-score normilization using dataset wide statistics. Dataset stats cannot be None when
    modality is 'ct'

    For other modalities: z-score normilization applied to each sample

    If cropping reduced number of voxels by > 25% then only nonzero values are used for normalization

    Args:
        image: original image
        modality: image modality ('ct', 'mri', etc) found via dataset JSON
        cropping_threshold_met: whether the cropping threshold for alternate normalization is met
        dataset_stats: (mean, std) dataset statistics for dataset wide CT normalization
        clipping_percentiles: (.5 - 99.5) preconfigured clipping percentages for CT normalization
        mask: Optional mask normalization within ROI
    Returns:
        NDArray: normalized image
    """
    pass


def compute_dataset_stats(
    dataset_dir: Path, modality: str, image_dir: str, mask_dir: str
) -> Dict[str, Any]:
    """Calculate key statistics from datasets.
    From CT datasets: mean, std, .5, 99.5 values from foreground classes. Median shape and voxel spacing.
    From non-CT datasets: Median shape and voxel spacing.

    Args:
        dataset_dir: path to dataset images
        modality: modality of images
        image_suffix: suffix to identify images in dir
        mask_suffix: suffix to identify masks in dir

    Returns:
        Dict[str, Tuple[float, float]]: 'percentiles':(low:float, high:float),
                                        'stats':(mean:float, std:float),
                                        'shape':(int, int, int),
                                        'spacing':(float, float, float)

    """
    if not os.path.exists(dataset_dir):
        raise(FileNotFoundError)
    
    image_path = dataset_dir / image_dir
    label_path = dataset_dir / mask_dir

    images = os.listdir(image_path)
    labels = os.listdir(label_path)

    dims = []
    spacings = []

    stats = {}

    if modality == 'CT':
        for image in images:
            if image not in labels:
                raise(Exception(f'All files in image directory must have a corrisponding segmentation mask. {image} was not found in {label_path}'))
        
        count, total, squares = 0, 0, 0
        percentile_pool = []
        max_pool_len = 1_000_000

        for image in tqdm(images):
            img = sitk.ReadImage(image_path / image)
            mask = sitk.ReadImage(label_path / image)

            img_np = sitk.GetArrayFromImage(img)
            mask_np = sitk.GetArrayFromImage(mask)

            masked_voxels =  img_np[mask_np != 0]

            n = len(masked_voxels)

            count += n
            total += np.sum(masked_voxels)
            squares += np.sum(masked_voxels ** 2)

            dims.append(img.GetSize())
            spacings.append(img.GetSpacing())

            if len(percentile_pool) < max_pool_len:
                percentile_pool.extend(masked_voxels)
            else:
                indices = np.random.randint(0, len(percentile_pool), size = min(n, len(percentile_pool)))
                for i, idx in enumerate(indices):
                    percentile_pool[idx] = masked_voxels[i]

        mean = total/count
        var = squares/count - mean ** 2
        std = np.sqrt(var)

        pool = np.array(percentile_pool)
        low = np.percentile(pool, .5)
        high = np.percentile(pool, 99.5)

        stats['stats'] = (mean, std)
        stats['percentiles'] = (low, high)

    else:
        for image in images:
            img = sitk.ReadImage(image_path / image)
            dims.append(img.GetSize())
            spacings.append(img.GetSpacing())

    dims = np.array(dims).T
    stats['shape'] = np.median(dims, 1)

    spacings = np.array(spacings).T
    stats['spacing'] = np.median(spacings, 1)
    return stats
            


def determine_cascade_necessity(median_shape: Tuple[float, float, float]) -> bool:
    """Determines if U-Net Cascade needed based on nnU-Net heuristics.

    Returns True if median shape contains >4x voxels than can be processed
    by 3D U-Net with patch size 128³ and batch size 2. Otherwise 3d cascade network is not trained.

    Args:
        median_shape (Tuple[float, float, float]): Median shape of post-resampled images

    Returns:
        bool: whether the cascade is determined to be necessary
    """


def lower_resolution(
    median_shape: Tuple[float, float, float],
    voxel_spacing: Tuple[float, float, float],
    image: NDArray,
) -> Tuple[NDArray, Tuple[float, float, float]]:
    """Follows nnU-net algorithm for lowering image resolution for input into 3d cascade.
    Determines appropriate size and voxel spacing by resampling images.

    For anisotropic images, higher resolution axes are resampled first.
    Once resolution on all axis is the same, all axes are resampled simultaniously.

    Args:
        median_shape (Tuple[float, float, float]): Median shape of dataset
        voxel_spacing (Tuple[float, float, float]): voxel spacing of dataset
        image (NDArray): original image

    Returns:
        Tuple[NDArray, Tuple[float, float, float]]: low res image, new voxel spacing
    """


def modality_detection(json_path: Path) -> str:
    """Detects image modality by searching json file for 'ct' substring to determine appropriate preprocessing steps

    Args:
        json_path (str): path of dataset's json file

    Returns:
        str: image modality detected
    """
    if not os.path.exists(json_path):
        raise (FileNotFoundError)
    with open(json_path, "r") as f:
        dataset_json = json.load(f)
        modality = dataset_json.get("modality", None)
        if modality is None:
            return "No Modality Detected"
        if "CT" in modality.values():
            return "CT"
        else:
            return "Not CT"
