import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import os
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from skimage.transform import resize

NDArray = npt.NDArray[np.float32]


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


def resample_dataset(
    images_dir: Path,
    labels_dir: Path,
    pickles_dir: Path,
    output_dir: Path,
    dataset_stats: Dict,
):
    """Resample entire dataset to median spacing

    Args:
        images_dir (Path): path to normalized images
        labels_dir (Path): path to cropped segmentation masks
        pickles_dir (Path): path to individualized spacing info
        output_dir (Path): path to place resampled images/masks
        dataset_stats (Dict): dataset_stats
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir / "imagesTr"):
        os.mkdir(output_dir / "imagesTr")

    if not os.path.exists(output_dir / "labelsTr"):
        os.mkdir(output_dir / "labelsTr")

    images = [file for file in os.listdir(images_dir) if file[-3:] != "pkl"]
    pickles = [file for file in os.listdir(pickles_dir) if file[-3:] == "pkl"]
    masks = os.listdir(labels_dir)

    new_spacing = dataset_stats["spacing"]

    for image in images:
        pickle = image.split(".")[0] + ".pkl"
        if pickle not in pickles:
            raise Exception("WARNING NO PICKLES --> missing {image} pickle")
        if image not in masks:
            raise Exception("WARNING NO MASK --> missing {image} mask")

        with open(pickles_dir / pickle, "rb") as file:
            stats = pkl.load(file)
            orig_spacing = stats["spacing"]

        img = sitk.ReadImage(images_dir / image)
        img_np = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(labels_dir / image)
        mask_np = sitk.GetArrayFromImage(mask)

        if img_np.shape != mask_np.shape:
            raise Exception(
                f"images and masks must be the same shape {img_np.shape} {mask_np.shape}"
            )

        img_resampled_np = resample_image(
            img_np, orig_spacing, new_spacing, is_segmentation=False
        )
        mask_resampled_np = resample_image(
            img_np, orig_spacing, new_spacing, is_segmentation=True
        )

        if img_resampled_np.shape != mask_resampled_np.shape:
            raise Exception(
                f"Resampled images and masks should be the same shape {img_resampled_np.shape} {mask_resampled_np.shape}"
            )

        img_resampled = sitk.GetImageFromArray(img_resampled_np)
        mask_resampled = sitk.GetImageFromArray(mask_resampled_np)

        sitk.WriteImage(img_resampled, output_dir / "imagesTr" / image)
        sitk.WriteImage(mask_resampled, output_dir / "labelsTr" / image)
