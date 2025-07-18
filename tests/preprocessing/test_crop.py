import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.crop import crop_dataset, crop_zeros

TEST_DATA_DIR = Path(__file__).parent / "crop_data"

np.random.seed(42)


@pytest.mark.parametrize(
    "input_dims, ones_slices, output_dims",
    (
        ((4, 4), [(0, 4), (0, 4)], (4, 4)),  # All non zero 2d
        ((1, 2, 3), [(0, 1), (0, 2), (0, 3)], (1, 2, 3)),  # All non zero odd shape
        ((16, 16, 16), [(0, 16), (0, 16), (0, 16)], (16, 16, 16)),  # All non zero 3d
        (
            (2, 2, 2, 2, 2),
            [(0, 2) for _ in range(5)],
            (2, 2, 2, 2, 2),
        ),  # All non zero 5d
        (
            (64, 64, 64),
            [(1, 64), (1, 64), (1, 64)],
            (63, 63, 63),
        ),  # Single zero x, y, z
        (
            (64, 64, 64),
            [(10, 40), (20, 50), (0, 64)],
            (31, 31, 64),
        ),  # Non zero rectangle 3d
        ((64, 64), [(4, 64), (0, 60)], (60, 61)),  # Non zero rectangel 2d
        ((64, 64), [(5, 5), (5, 5)], (1, 1)),  # Single non zero element 2d
        ((1, 1, 1), [(0, 0), (0, 0), (0, 0)], (1, 1, 1)),  # 1 x 1 x 1 non zero
        (
            (1, 512, 512),
            [(0, 0), (0, 512), (0, 512)],
            (1, 512, 512),
        ),  # 1 x 512 x 512 non zero
        (
            (512, 512, 512),
            [(100, 200), (445, 446), (30, 480)],
            (101, 2, 451),
        ),  # Very large
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


def test_crop_all_zeros():
    with pytest.raises(Exception):
        crop_zeros(np.zeros((10, 10)), np.ones((10, 10)))  # All zeros


def test_crop_non_matching_dims():
    with pytest.raises(Exception):
        crop_zeros(np.ones((10, 10)), np.ones((10, 9)))  # Missmatched mask dims


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
    shutil.rmtree(TEST_DATA_DIR)


def test_crop_dataset():
    crops = setup_dataset()
    crop_dataset(TEST_DATA_DIR, TEST_DATA_DIR / "cropped")

    images = sorted(os.listdir(TEST_DATA_DIR / "cropped" / "imagesTr"))
    for crops, img in zip(crops, images):
        cropped_img = sitk.ReadImage(TEST_DATA_DIR / "cropped" / "imagesTr" / img)
        cropped_mask = sitk.ReadImage(TEST_DATA_DIR / "cropped" / "labelsTr" / img)

        img_size = np.array(cropped_img.GetSize()[::-1])
        mask_size = np.array(cropped_mask.GetSize()[::-1])

        correct_size = [(b - a + 1) for a, b in crops]

        np.testing.assert_array_equal(img_size, mask_size)
        np.testing.assert_array_equal(img_size, correct_size)

    tear_down_dataset()
