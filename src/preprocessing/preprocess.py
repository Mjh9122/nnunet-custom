import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import json
import os
import pickle as pkl
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from tqdm import tqdm

from preprocessing.crop import crop_dataset
from preprocessing.normalize import normalize_dataset
from preprocessing.resample import resample_dataset

np.random.seed(42)

NDArray = npt.NDArray[np.float32]
CT_CLIP_VALUES = (0.5, 99.5)
MAX_3D_PATCH_SIZE = (128, 128, 128)
MIN_3D_BATCH_SIZE = 2


def compute_dataset_stats(dataset_dir: Path, modality: str) -> Dict[str, Any]:
    """Calculate key statistics from cropped dataset.
    From CT datasets: mean, std, .5, 99.5 values from foreground classes. Median shape and voxel spacing.
    From non-CT datasets: Median shape and voxel spacing.

    Args:
        dataset_dir: path to dataset images
        modality: modality of images

    Returns:
        Dict[str, Tuple[float, float]]: 'percentiles':(low:float, high:float),
                                        'stats':(mean:float, std:float),
                                        'shape':(int, int, int),
                                        'spacing':(float, float, float),
                                        'num_images':int

    """
    if not os.path.exists(dataset_dir):
        raise (FileNotFoundError)

    image_path = dataset_dir / "imagesTr"
    label_path = dataset_dir / "labelsTr"
    pickle_path = dataset_dir / "picklesTr"

    images = os.listdir(image_path)
    labels = os.listdir(label_path)
    pkls = os.listdir(pickle_path)

    dataset_stats = {}

    dataset_stats["num_images"] = len(images)
    # Aggregate dimensions and spacing for median calculations
    precrop_dims = []
    postcrop_dims = []
    spacings = []

    for f in pkls:
        with open(pickle_path / f, "rb") as file:
            stats = pkl.load(file)

            precrop_dims.append(stats["pre_crop_shape"])
            postcrop_dims.append(stats["post_crop_shape"])
            spacings.append(stats["spacing"])

    precrop_dims = np.array(precrop_dims).T
    dataset_stats["pre_crop_shape"] = np.median(precrop_dims, 1)

    postcrop_dims = np.array(postcrop_dims).T
    dataset_stats["post_crop_shape"] = np.median(postcrop_dims, 1)

    spacings = np.array(spacings).T
    dataset_stats["spacing"] = np.median(spacings, 1)

    dataset_stats["cropping_threshold_met"] = (
        dataset_stats["pre_crop_shape"].prod() * 3 / 4
        > dataset_stats["post_crop_shape"].prod()
    )

    if modality == "CT":
        for image in images:
            if image not in labels:
                raise (
                    Exception(
                        f"All files in image directory must have a corrisponding segmentation mask. {image} was not found in {label_path}"
                    )
                )

        count, total, M2 = 0, 0, 0
        percentile_pool = []
        max_pool_len = 1_000_000

        for image in tqdm(images, desc="Stats"):
            img = sitk.ReadImage(image_path / image)
            mask = sitk.ReadImage(label_path / image)

            img_np = sitk.GetArrayFromImage(img)
            mask_np = sitk.GetArrayFromImage(mask)

            masked_voxels = img_np[mask_np != 0]

            n = len(masked_voxels)

            if n > 0:
                old_count = count
                if count > 0:
                    old_mean = total / count
                else:
                    old_mean = 0

                count += n
                total += np.sum(masked_voxels)
                new_mean = total / count

                if old_count > 0:
                    delta = new_mean - old_mean
                    M2 += (
                        np.sum((masked_voxels - new_mean) ** 2)
                        + old_count * n * delta**2 / count
                    )
                else:
                    M2 += np.sum((masked_voxels - new_mean) ** 2)

                if len(percentile_pool) < max_pool_len:
                    percentile_pool.extend(masked_voxels)
                else:
                    indices = np.random.randint(
                        0, len(percentile_pool), size=min(n, len(percentile_pool))
                    )
                    for i, idx in enumerate(indices):
                        percentile_pool[idx] = masked_voxels[i]

        mean = total / count
        var = M2 / count
        std = np.sqrt(var)

        pool = np.array(percentile_pool)
        low = np.percentile(pool, 0.5)
        high = np.percentile(pool, 99.5)

        dataset_stats["stats"] = (mean, std)
        dataset_stats["percentiles"] = (low, high)

    return dataset_stats


def select_cv_fold(dataset_dir: Path, images_list: List[str], output_dir: Path):
    """Copies selected images to separate directory for preprocessing

    Args:
        dataset_dir (Path): Dataset dir (should follow medical segmentaion decathalon format)
        images_list (List[str]): Images to be preprocessed (CV fold)
        output_dir (Path): Directory to put selected images and labels
    """

    all_images = dataset_dir / "imagesTr"
    all_labels = dataset_dir / "labelsTr"
    selected_images = output_dir / "imagesTr"
    selected_labels = output_dir / "labelsTr"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir / "imagesTr"):
        os.mkdir(output_dir / "imagesTr")
    if not os.path.exists(output_dir / "labelsTr"):
        os.mkdir(output_dir / "labelsTr")

    # Check all images are present in dataset_dir/imagesTr
    if not set(images_list).issubset(os.listdir(all_images)):
        raise (
            Exception(
                "All images in images_list must be present in dataset_dir/imagesTr"
            )
        )

    # Check all images are present in dataset_dir/labelsTr
    if not set(images_list).issubset(os.listdir(all_labels)):
        raise (
            Exception(
                "All images in images_list must be present in dataset_dir/labelsTr"
            )
        )

    # Copy all images and labels to output_dir/images(labels)Tr
    for img in images_list:
        shutil.copy(all_images / img, selected_images / img)
        shutil.copy(all_labels / img, selected_labels / img)


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

    return current_voxels >= 4 * max_voxels


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


def preprocess_dataset(dataset_dir: Path, cv_split: List[str], output_dir: Path):
    """Preprocesses an entire dataset. preprocessed images are placed in the output directory
    and dataset statistics are returned for use in downstream tasks

    Args:
        dataset_dir (Path): Path to dataset directory should contain a dataset.json file, imagesTr, and labelsTr dirs
        cv_split (List[str]): List of image names in the dataset dir to preprocess
        output_dir (Path): Path to output directory. Finished numpy arrays are stored alongside metadata.
    """

    if not os.path.exists(dataset_dir):
        raise (FileNotFoundError)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir):
        raise (Exception("Output directory is not a directory"))

    stats = {}

    # 1. Determine modality
    modality = modality_detection(dataset_dir / "dataset.json")
    stats["modality"] = modality

    # 2. Select CV -> place in output_dir
    select_cv_fold(dataset_dir, cv_split, output_dir / "original")

    # 2. Crop data + generate pickles -> replace output_dir
    crop_dataset(output_dir / "original", output_dir / "cropped")

    # 3. Calculate dataset stats
    stats.update(compute_dataset_stats(output_dir / "cropped", modality))

    # 4. Normalize each image place new images in normalized
    normalize_dataset(
        output_dir / "cropped" / "imagesTr", output_dir / "normalized", stats
    )

    # 5. Resample to median voxel spacing -> place in dataset_dir / high_res
    stats["post_resample_shape"] = resample_dataset(
        output_dir / "normalized",
        output_dir / "cropped" / "labelsTr",
        output_dir / "cropped" / "picklesTr",
        output_dir / "high_res",
        stats["spacing"],
    )
    # 6. Determine Cascade necessity
    cascade_needed = determine_cascade_necessity(stats["post_resample_shape"])

    # 7. Generate low resolution image if needed -> place in dataset_dir / low_res
    if cascade_needed:
        low_res = lower_resolution(stats["post_resample_shape"], stats["spacing"])
        stats["low_res_spacing"] = low_res

        resample_dataset(
            output_dir / "normalized",
            output_dir / "cropped" / "labelsTr",
            output_dir / "cropped" / "picklesTr",
            output_dir / "low_res",
            stats["low_res_spacing"],
        )

        stats["low_res_path"] = dataset_dir / "low_res"

    stats["high_res_path"] = dataset_dir / "high_res"

    # clean up output directory
    shutil.rmtree(output_dir / "original")
    shutil.rmtree(output_dir / "cropped")
    shutil.rmtree(output_dir / "normalized")

    # Save dataset stats for later
    with open(output_dir / "dataset_stats.pkl", "wb") as f:
        pkl.dump(stats, f)
