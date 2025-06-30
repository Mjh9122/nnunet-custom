import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)


import os
import pickle as pkl
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.resample import resample_dataset, resample_image

TEST_DATA_DIR = TEST_DATA_DIR = Path(__file__).parent / "resample_data"

np.random.seed(42)


@pytest.mark.parametrize(
    "image, old_spacing, new_spacing",
    (
        (np.ones((4, 4, 4)), np.array((1.0, 1.0, 1.0)), np.array((2.0, 2.0))),
        (np.ones((4, 4, 4, 4)), np.array((1.0, 1.0, 1.0)), np.array((2.0, 2.0, 2.0))),
    ),
)
def test_resample_image_bad_spacings(image, old_spacing, new_spacing):
    with pytest.raises(Exception):
        resample_image(image, old_spacing, new_spacing, is_segmentation=False)


@pytest.mark.parametrize(
    "old_dims, old_spacing, new_spacing, expected",
    (
        ((16, 32, 32), (2.0, 1.0, 1.0), (2.0, 2.0, 2.0), (16, 16, 16)),
        ((16, 16, 16), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (16, 16, 16)),
        ((100, 100, 50), (1.0, 1.0, 2.0), (0.5, 0.5, 1.0), (200, 200, 100)),
        ((100, 100, 50), (1.0, 1.0, 2.0), (10.0, 10.0, 10.0), (10, 10, 10)),
        ((10, 10, 10), (10.0, 10.0, 10.0), (1.0, 1.0, 2.0), (100, 100, 50)),
    ),
)
def test_resample_image_expected_dims(old_dims, old_spacing, new_spacing, expected):
    img = np.random.rand(*old_dims)
    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation=False)

    assert expected == reshaped.shape


def test_rasample_image_range_preservation():
    old_dims = (64, 32, 32)
    old_spacing = (1.0, 0.5, 0.5)
    new_spacing = (1.5, 0.75, 0.75)
    img = np.random.rand(*old_dims) * 1000

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation=False)

    tol = 0.05 * (img.max() - img.min())
    assert reshaped.min() > img.min() - tol
    assert reshaped.max() < img.max() + tol


def test_resample_image_energy_preservation():
    old_dims = (64, 32, 32)
    old_spacing = (1.0, 0.5, 0.5)
    new_spacing = (1.0, 1.0, 1.0)
    img = np.random.rand(*old_dims) * 1000

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation=False)

    tol = 0.005
    sum_diff = abs(
        img.sum() * np.prod(old_spacing) - reshaped.sum() * np.prod(new_spacing)
    )

    assert sum_diff / (img.sum() * np.prod(old_spacing)) < tol


def test_resample_image_uniform():
    const = 1000
    old_dims = (64, 32, 32)
    old_spacing = (5.0, 0.1, 0.1)
    new_spacing = (1.0, 1.0, 1.0)
    img = np.ones(old_dims) * const

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation=False)

    err = np.abs(reshaped - const)
    rtol = 0.01

    assert np.max(err) < rtol * const


def test_resample_image_seg_mask_up():
    img = np.array([[[1], [0]], [[0], [1]]])
    old_spacing = (2, 2, 2)
    new_spacing = (1, 1, 2)

    expected = np.array(
        [
            [[1], [1], [0], [0]],
            [[1], [1], [0], [0]],
            [[0], [0], [1], [1]],
            [[0], [0], [1], [1]],
        ]
    )

    reshaped = resample_image(img, old_spacing, new_spacing, True)

    np.testing.assert_array_equal(expected, reshaped)


def test_resample_image_seg_mask_down():
    img = np.array(
        [
            [[1], [1], [0], [0]],
            [[1], [1], [0], [0]],
            [[0], [0], [1], [1]],
            [[0], [0], [1], [1]],
        ]
    )
    old_spacing = (1, 1, 2)
    new_spacing = (2, 2, 2)

    expected = np.array([[[1], [0]], [[0], [1]]])

    reshaped = resample_image(img, old_spacing, new_spacing, True)

    np.testing.assert_array_equal(expected, reshaped)


def test_bad_spacings():
    with pytest.raises(Exception):
        resample_image(np.ones((10, 10)), (0.1, 0.1), (0.1, 0.1, 0.1), True)

    with pytest.raises(Exception):
        resample_image(np.ones((10, 10)), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), True)


def setup_dataset():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "images")
        os.mkdir(TEST_DATA_DIR / "labels")
        os.mkdir(TEST_DATA_DIR / "pickles")
        os.mkdir(TEST_DATA_DIR / "resampled")

    arrs = [
        np.random.randint(0, 1000, (16, 16, 16)).astype(np.float32) for _ in range(10)
    ]
    masks = [np.random.choice([0, 1], (16, 16, 16), p=[0.2, 0.8]) for _ in range(10)]
    pickles = [
        {"spacing": ((i + 1) * 1.0, (i + 1) * 1.0, (i + 1) * 1.0)} for i in range(10)
    ]

    for i, (arr, mask_np, pickle) in enumerate(zip(arrs, masks, pickles)):
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, TEST_DATA_DIR / "images" / f"test{i}.nii.gz")

        mask = sitk.GetImageFromArray(mask_np)
        sitk.WriteImage(mask, TEST_DATA_DIR / "labels" / f"test{i}.nii.gz")

        with open(TEST_DATA_DIR / "pickles" / f"test{i}.pkl", "wb") as file:
            pkl.dump(pickle, file)

    energies = [
        arr.sum() * np.prod(stats["spacing"]) for arr, stats in zip(arrs, pickles)
    ]

    return energies


def teardown_dataset():
    shutil.rmtree(TEST_DATA_DIR)


def test_resample_directory():
    energies = setup_dataset()

    images = sorted(os.listdir(TEST_DATA_DIR / "images"))

    spacing = (3.0, 2.0, 2.0)

    resample_dataset(
        TEST_DATA_DIR / "images",
        TEST_DATA_DIR / "labels",
        TEST_DATA_DIR / "pickles",
        TEST_DATA_DIR / "resampled",
        spacing,
    )

    for i, image in enumerate(images):
        img = sitk.ReadImage(TEST_DATA_DIR / "resampled" / "imagesTr" / image)
        img_np = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(TEST_DATA_DIR / "resampled" / "labelsTr" / image)
        mask_np = sitk.GetArrayFromImage(mask)

        assert img_np.shape == mask_np.shape
        assert abs(img_np.sum() * np.prod(spacing) - energies[i]) < energies[i] * 0.1

    teardown_dataset()


def test_bad_directories():
    if not os.path.exists(TEST_DATA_DIR):
        os.mkdir(TEST_DATA_DIR)
    os.mkdir(TEST_DATA_DIR / "bad_images")
    os.mkdir(TEST_DATA_DIR / "bad_labels")
    os.mkdir(TEST_DATA_DIR / "bad_pickles")
    os.mkdir(TEST_DATA_DIR / "bad_resampled")

    arr = np.ones((10, 10))
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, TEST_DATA_DIR / "bad_images" / "img.nii.gz")
    sitk.WriteImage(img, TEST_DATA_DIR / "bad_labels" / "img.nii.gz")

    with pytest.raises(Exception):
        resample_dataset(
            TEST_DATA_DIR / "bad_images",
            TEST_DATA_DIR / "bad_labels",
            TEST_DATA_DIR / "bad_pickles",
            TEST_DATA_DIR / "bad_resampled",
            (1.0, 1.0, 1.0),
        )

    os.remove(TEST_DATA_DIR / "bad_labels" / "img.nii.gz")
    with open(TEST_DATA_DIR / "bad_pickles" / "img.pkl", "wb") as file:
        pkl.dump({"pickle": "pickle"}, file)

    with pytest.raises(Exception):
        resample_dataset(
            TEST_DATA_DIR / "bad_images",
            TEST_DATA_DIR / "bad_labels",
            TEST_DATA_DIR / "bad_pickles",
            TEST_DATA_DIR / "bad_resampled",
            (1.0, 1.0, 1.0),
        )

    teardown_dataset()
