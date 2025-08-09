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
from tqdm import tqdm

NDArray = npt.NDArray[np.float32]


def resample_image(
    image: NDArray,
    old_spacing: Tuple[float, ...],
    new_spacing: Tuple[float, ...],
    is_segmentation: bool,
) -> NDArray:
    """Resamples image to new voxel spacing using cubic spline interpolation or nearest
    neighbor (cublic spline order 0) based on whether an image or segmentation mask is passed in

    Args:
        image (NDArray): original image
        old_spacing (Tuple[float, ...]): original voxel spacing (X, Y, Z)
        new_spacing (Tuple[float, ...]): new voxel spacing (X, Y, Z)
        is_segmentation: True if segmentation map is inputed

    Returns:
        NDArray: resampled image
    """
    assert image.ndim == 4
    assert len(old_spacing) == 3
    assert len(new_spacing) == 3

    old_dims = np.array(image.shape)
    old_spacing = (1,) + old_spacing
    new_spacing = (1,) + new_spacing

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
    target_spacing: Tuple[float, ...],
):
    """Resample entire dataset to median spacing

    Args:
        images_dir (Path): path to normalized images
        labels_dir (Path): path to cropped segmentation masks
        pickles_dir (Path): path to individualized spacing info
        output_dir (Path): path to place resampled images/masks
        target_spacing (Tuple[float, ...]): spacing to resample images to
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir / "imagesTr"):
        os.mkdir(output_dir / "imagesTr")

    if not os.path.exists(output_dir / "labelsTr"):
        os.mkdir(output_dir / "labelsTr")

    images = os.listdir(images_dir)
    pickles = os.listdir(pickles_dir)
    masks = os.listdir(labels_dir)

    new_shapes = []

    for image in tqdm(images, "Resampling"):
        pickle = image.split(".")[0] + ".pkl"
        if pickle not in pickles:
            raise Exception(f"WARNING NO PICKLES --> missing {image} pickle")
        if image not in masks:
            raise Exception(f"WARNING NO MASK --> missing {image} mask")

        with open(pickles_dir / pickle, "rb") as file:
            stats = pkl.load(file)
            orig_spacing = stats["spacing"]

        img = sitk.ReadImage(images_dir / image)
        img_np = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(labels_dir / image)
        mask_np = sitk.GetArrayFromImage(mask)

        if img_np.shape != mask_np.shape:
            raise ValueError(
                f"images and masks must be the same shape {img_np.shape} {mask_np.shape}"
            )

        img_resampled_np = resample_image(
            img_np, orig_spacing, target_spacing, is_segmentation=False
        )
        mask_resampled_np = resample_image(
            mask_np, orig_spacing, target_spacing, is_segmentation=True
        )

        new_shapes.append(img_resampled_np.shape)

        img_resampled = sitk.GetImageFromArray(img_resampled_np)
        mask_resampled = sitk.GetImageFromArray(mask_resampled_np)

        sitk.WriteImage(img_resampled, output_dir / "imagesTr" / image)
        sitk.WriteImage(mask_resampled, output_dir / "labelsTr" / image)

    new_shapes = np.array(new_shapes).T
    return np.median(new_shapes, axis=1)
