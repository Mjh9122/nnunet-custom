import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import sys
from pathlib import Path

import numpy as np
import os
import pytest
import SimpleITK as sitk
import shutil


src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.crop import crop_zeros, crop_dataset


TEST_DATA_DIR = Path(__file__).parent / "crop_data"

np.random.seed(42)


@pytest.mark.parametrize(
    "input_dims, ones_slices, output_dims",
    (
        ((4, 4), [(0, 4), (0, 4)], (4, 4)),
        ((1, 2, 3), [(0, 1), (0, 2), (0, 3)], (1, 2, 3)),
        ((16, 16, 16), [(0, 16), (0, 16), (0, 16)], (16, 16, 16)),
        ((2, 2, 2, 2, 2), [(0, 2) for _ in range(5)], (2, 2, 2, 2, 2)),
        ((64, 64, 64), [(1, 64), (1, 64), (1, 64)], (63, 63, 63)),
        ((64, 64, 64), [(10, 40), (20, 50), (0, 64)], (31, 31, 64)),
        ((64, 64), [(4, 64), (0, 60)], (60, 61)),
        ((64, 64), [(5, 9), (5, 39)], (5, 35)),
    ),
)
def test_crop_bool(input_dims, ones_slices, output_dims):
    arr = np.zeros(input_dims, np.float32)
    if ones_slices is not None:
        slices = tuple(slice(a, b + 1) for a, b in ones_slices)
        arr[slices] = 1

    seg_mask = arr.copy()

    cropped_arr, cropped_mask = crop_zeros(arr, seg_mask)
    np.testing.assert_array_equal(cropped_arr, np.ones(output_dims, np.float32))
    np.testing.assert_array_equal(cropped_mask, np.ones(output_dims, np.float32))


def setup_dataset():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "imagesTr")
        os.mkdir(TEST_DATA_DIR / "labelsTr")

    arrs = [np.zeros((10, 10, 10), np.float32) for _ in range(10)]
    masks = [np.zeros((10, 10, 10), np.float32) for _ in range(10)]

    ones = np.random.randint(low=0, high=10, size=(10, 3, 3))

    for arr, spots in zip(arrs, ones):
        for spot in spots:
            arr[spot[0], spot[1], spot[2]] = 1

    crops = [[(min(spots[:, n]), max(spots[:, n])) for n in range(3)] for spots in ones]

    imgs = [sitk.GetImageFromArray(arr) for arr in arrs]
    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f"imagesTr/test0{i}.nii.gz")

    imgs = [sitk.GetImageFromArray(mask) for mask in masks]

    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f"labelsTr/test0{i}.nii.gz")

    return crops


def tear_down_dataset():
    """crop_data
    -> crops
        -> imagesTr
        -> labelsTr
    -> imagesTr
    -> labelsTr
    """
    shutil.rmtree(TEST_DATA_DIR)


def test_crop_dataset():
    crops = setup_dataset()
    results = crop_dataset(TEST_DATA_DIR, TEST_DATA_DIR)

    images = sorted([f for f in os.listdir(results / "imagesTr") if f[-3:] != "pkl"])
    for crops, img in zip(crops, images):
        cropped_img = sitk.ReadImage(results / "imagesTr" / img)
        cropped_mask = sitk.ReadImage(results / "labelsTr" / img)

        img_size = np.array(cropped_img.GetSize()[::-1])
        mask_size = np.array(cropped_mask.GetSize()[::-1])

        correct_size = [(b - a + 1) for a, b in crops]

        np.testing.assert_array_equal(img_size, mask_size)
        np.testing.assert_array_equal(img_size, correct_size)

    tear_down_dataset()
