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
from tqdm import tqdm

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
        raise ValueError(
            f"Image shape {image.shape} must match mask shape {mask.shape}"
        )

    if not np.abs(image).sum() > 0:
        raise ValueError(f"Image must contain non-zero values")

    # Find indeces of all nonzero values
    nonzero = np.array(np.where(image != 0))
    # Min and max nonzeros for each axis, transpose to get min/max pairs
    nonzero = np.array([np.min(nonzero, 1), np.max(nonzero, 1)]).T

    slices = tuple(slice(a, b + 1) for a, b in nonzero)

    cropped_img = image[slices]
    cropped_mask = mask[slices]

    return cropped_img, cropped_mask


def crop_dataset(dataset_dir: Path, output_dir: Path):
    """Crops all images and labels in dataset, placing them in a temp folder in the output dir

    Args:
        dataset_dir (Path): Dataset directory should have dataset.json, imagesTr, labelsTr
        output_dir (Path): Directory to place completed preprocessed images
    """
    image_path = dataset_dir / "imagesTr"
    label_path = dataset_dir / "labelsTr"

    images = os.listdir(image_path)

    cropped_images_path = output_dir / "imagesTr"
    cropped_labels_path = output_dir / "labelsTr"
    pickle_path = output_dir / "picklesTr"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(cropped_images_path):
        os.mkdir(cropped_images_path)

    if not os.path.exists(cropped_labels_path):
        os.mkdir(cropped_labels_path)

    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    for image in tqdm(images, desc="Cropping"):
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
            "pre_crop_shape": pre_crop_size,
            "post_crop_shape": post_crop_size,
            "spacing": spacing,
        }

        stats_pkl = image.split(".")[0] + ".pkl"
        with open(pickle_path / stats_pkl, "wb") as file:
            pkl.dump(stats, file)
