import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import json
import os
import pickle as pkl

from .crop import crop_zeros
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

np.random.seed(42)

NDArray = npt.NDArray[np.float32]
CT_CLIP_VALUES = (0.5, 99.5)
MAX_3D_PATCH_SIZE = (128, 128, 128)
MIN_3D_BATCH_SIZE = 2
CROPPING_NORMALIZATION_THRESHOLD = 0.25

def crop_dataset(dataset_dir: Path, output_dir:Path) -> Path:
    """Crops all images and labels in dataset, placing them in a temp folder in the output dir

    Args:
        dataset_dir (Path): Dataset directory should have dataset.json, imagesTr, labelsTr
        output_dir (Path): Directory to place completed preprocessed images
    """
    image_path = dataset_dir / 'imagesTr'
    label_path = dataset_dir / 'labelsTr'

    images = os.listdir(image_path)

    cropped_path = output_dir / 'crops'
    cropped_images_path = cropped_path / 'imagesTr'
    cropped_labels_path = cropped_path / 'labelsTr'

    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)
    if not os.path.exists(cropped_images_path):
        os.mkdir(cropped_images_path)
    if not os.path.exists(cropped_labels_path):
        os.mkdir(cropped_labels_path)

    for image in tqdm(images):
        img = sitk.ReadImage(image_path / image)
        mask = sitk.ReadImage(label_path / image)

        spacing = img.GetSpacing()
        img_np = sitk.GetArrayFromImage(img)
        mask_np = sitk.GetArrayFromImage(mask)

        pre_crop_size = img_np.shape

        img_crop_np, mask_crop_np = crop_zeros(img_np, mask_np)

        post_crop_size = img_crop_np.shape

        img_crop = sitk.GetImageFromArray(img_crop_np)
        sitk.WriteImage(img_crop, cropped_images_path / image)

        mask_crop = sitk.GetImageFromArray(mask_crop_np)
        sitk.WriteImage(mask_crop, cropped_labels_path / image)

        stats = {
            'pre_crop_shape' : pre_crop_size,
            'post_crop_shape' : post_crop_size,
            'spacing' : spacing
        }

        stats_pkl  = image.split('.')[0] + ".pkl"
        with open(cropped_images_path / stats_pkl, 'wb') as file:
            pkl.dump(stats, file)

    return cropped_path



def compute_dataset_stats(
    dataset_dir: Path, modality: str
) -> Dict[str, Any]:
    """Calculate key statistics from datasets.
    From CT datasets: mean, std, .5, 99.5 values from foreground classes. Median shape and voxel spacing.
    From non-CT datasets: Median shape and voxel spacing.

    Args:
        dataset_dir: path to dataset images
        modality: modality of images

    Returns:
        Dict[str, Tuple[float, float]]: 'percentiles':(low:float, high:float),
                                        'stats':(mean:float, std:float),
                                        'shape':(int, int, int),
                                        'spacing':(float, float, float)

    """
    if not os.path.exists(dataset_dir):
        raise (FileNotFoundError)

    image_path = dataset_dir / 'imagesTr'
    label_path = dataset_dir / 'labelsTr'

    images = os.listdir(image_path)
    labels = os.listdir(label_path)

    dims = []
    spacings = []

    stats = {}
    stats['pre_crop_voxels'] = 0

    if modality == "CT":
        for image in images:
            if image not in labels:
                raise (
                    Exception(
                        f"All files in image directory must have a corrisponding segmentation mask. {image} was not found in {label_path}"
                    )
                )

        count, total, squares = 0, 0, 0
        percentile_pool = []
        max_pool_len = 1_000_000

        for image in tqdm(images):
            img = sitk.ReadImage(image_path / image)
            mask = sitk.ReadImage(label_path / image)

            img_np = sitk.GetArrayFromImage(img)
            mask_np = sitk.GetArrayFromImage(mask)

            masked_voxels = img_np[mask_np != 0]

            n = len(masked_voxels)

            count += n
            total += np.sum(masked_voxels)
            squares += np.sum(masked_voxels**2)

            dims.append(img.GetSize())
            spacings.append(img.GetSpacing())

            if len(percentile_pool) < max_pool_len:
                percentile_pool.extend(masked_voxels)
            else:
                indices = np.random.randint(
                    0, len(percentile_pool), size=min(n, len(percentile_pool))
                )
                for i, idx in enumerate(indices):
                    percentile_pool[idx] = masked_voxels[i]
            
            stats['pre_crop_voxels'] += np.prod(img_np.shape)

        mean = total / count
        var = squares / count - mean**2
        std = np.sqrt(var)

        pool = np.array(percentile_pool)
        low = np.percentile(pool, 0.5)
        high = np.percentile(pool, 99.5)

        stats["stats"] = (mean, std)
        stats["percentiles"] = (low, high)
        

    else:
        for image in images:
            img = sitk.ReadImage(image_path / image)
            dims.append(img.GetSize())
            spacings.append(img.GetSpacing())
            stats['pre_crop_voxels'] += np.prod(img.GetSize())

    dims = np.array(dims).T
    stats["shape"] = np.median(dims, 1)

    spacings = np.array(spacings).T
    stats["spacing"] = np.median(spacings, 1)
    return stats


def determine_cascade_necessity(median_shape: Tuple[int, int, int]) -> bool:
    """Determines if U-Net Cascade needed based on nnU-Net heuristics.

    Returns True if median shape contains >4x voxels than can be processed
    by 3D U-Net with patch size 128³ and batch size 2. Otherwise 3d cascade network is not trained.

    Args:
        median_shape (Tuple[int, int, int]): Median shape of post-resampled images

    Returns:
        bool: whether the cascade is determined to be necessary
    """
    max_voxels = np.array(MAX_3D_PATCH_SIZE).prod()
    current_voxels = np.array(median_shape).prod()

    return current_voxels > 4 * max_voxels


def lower_resolution(
    median_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Follows nnU-net algorithm for lowering image resolution for input into 3d cascade.
    Determines appropriate size and voxel spacing for resampling images.

    For anisotropic images, higher resolution axes are resampled first.
    Once resolution on all axis is the same, all axes are resampled simultaniously.

    Args:
        median_shape (Tuple[float, float, float]): Median shape of dataset
        voxel_spacing (Tuple[float, float, float]): voxel spacing of dataset

    Returns:
        Tuple[NDArray, Tuple[float, float, float]]: low res image, new voxel spacing
    """
    current_size = np.array(median_shape)
    current_spacing = np.array(voxel_spacing)
    max_voxels = np.array(MAX_3D_PATCH_SIZE).prod()

    total_voxels = current_size.prod()

    while total_voxels > 4 * max_voxels:
        high_res_axis = current_spacing == min(current_spacing)
        current_spacing[high_res_axis] *= 2
        current_size[high_res_axis] //= 2
        total_voxels = current_size.prod()

    return current_spacing


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


def preprocess_dataset(dataset_dir: Path, output_dir: Path) -> Dict[Any, Any]:
    """Preprocesses an entire dataset. preprocessed images are placed in the output directory
    and dataset statistics are returned for use in downstream tasks

    Args: 
        dataset_dir (Path): Path to dataset directory should contain a dataset.json file, imagesTr, and labelsTr dirs
        output_dir (Path): Path to output directory. Finished numpy arrays are stored alongside metadata.

    Returns:
        Dict
    """

    if not os.path.exists(dataset_dir):
        raise(FileNotFoundError)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir):
        raise(Exception('Output directory is not a directory'))
    
    stats = {}

    # 1. Determine modality
    modality = modality_detection(dataset_dir / 'dataset.json')
    stats['modality'] = modality

    # 2. Crop data 
    cropped_dir = crop_dataset(dataset_dir)

    # 3. Normalize each image
    # 4. Resample to median voxel spacing
    # 5. Determine Cascade necessity
    # 6. Generate low resolution image if needed


    return stats