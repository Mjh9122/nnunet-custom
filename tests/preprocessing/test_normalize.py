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

from preprocessing.normalize import normalize, normalize_dataset

TEST_DATA_DIR = TEST_DATA_DIR = Path(__file__).parent / "normalize_data"


def test_normalize_ct_withzeros():
    img = np.zeros((4, 4), np.float32)
    img[1:3, 1:3] = np.arange(1, 5).reshape((2, 2))

    result = normalize(img, "CT", False, (2.0, 2.0), (1.0, 3.0))

    expected = np.ones((4, 4), np.float32) * -1 / 2
    expected[1, 2] = 0
    expected[2, [1, 2]] = 1 / 2

    np.testing.assert_allclose(expected, result)


def test_normalize_non_ct_nonzeros():
    img = np.zeros((4, 4), np.float32)
    img[1:3, 1:3] = np.arange(1, 5).reshape((2, 2))

    result = normalize(img, "MRI", True)

    expected = np.zeros((4, 4), np.float32)
    expected[1, 1] = -1.5 / np.sqrt(5 / 4)
    expected[1, 2] = -0.5 / np.sqrt(5 / 4)
    expected[2, 1] = 0.5 / np.sqrt(5 / 4)
    expected[2, 2] = 1.5 / np.sqrt(5 / 4)

    np.testing.assert_allclose(expected, result)


def test_normalize_non_ct_withzeros():
    img = np.zeros((4, 4), np.float32)
    img[1:3, 1:3] = np.arange(1, 5).reshape((2, 2))

    result = normalize(img, "MRI", False)

    expected = np.ones((4, 4), np.float32) * -(10 / 16) / np.sqrt(95 / 64)
    expected[1, 1] = (6 / 16) / np.sqrt(95 / 64)
    expected[1, 2] = (22 / 16) / np.sqrt(95 / 64)
    expected[2, 1] = (38 / 16) / np.sqrt(95 / 64)
    expected[2, 2] = (54 / 16) / np.sqrt(95 / 64)

    np.testing.assert_allclose(expected, result)


def test_incorrect_params():
    with pytest.raises(Exception):
        normalize(np.ones((10, 10, 10)), "CT", True, dataset_stats=(1, 1))

    with pytest.raises(Exception):
        normalize(np.ones((10, 10, 10)), "CT", True, clipping_percentiles=(1, 1))


def setup_dataset():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "crops")

    arr1 = np.zeros((3, 4), np.float32)
    arr1[1, [0, 3]] = 1
    arr1[1, [1, 2]] = 3

    arr2 = np.zeros((3, 4), np.float32)
    arr2[1, 2] = 1
    arr2[1, 3] = 5

    arr3 = np.zeros((3, 4), np.float32)
    arr3[1, 2] = 4
    arr3[1, 3] = 6

    arrs = [arr1, arr2, arr3]

    imgs = [sitk.GetImageFromArray(arr) for arr in arrs]
    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / "crops" / f"test0{i}.nii.gz")

    # CT args
    mean, std, low, high = 2, 2, 1, 4
    CT_map = np.vectorize(
        {i: ((min(max(i, low), high) - mean) / std) for i in range(7)}.get
    )
    CT_solutions = [CT_map(arr) for arr in arrs]

    threshold_solutions = []
    for arr in arrs:
        sol = arr.copy()
        sol[sol != 0] = (sol[sol != 0] - sol[sol != 0].mean()) / sol[sol != 0].std()
        threshold_solutions.append(sol)

    no_threshold_solutions = [(arr - arr.mean()) / arr.std() for arr in arrs]

    return CT_solutions, threshold_solutions, no_threshold_solutions


def tear_down_dataset():
    shutil.rmtree(TEST_DATA_DIR)


def check_solutions(solutions, images_path):
    images = sorted([f for f in os.listdir(images_path)])

    for solution, img in zip(solutions, images):
        normalized_img = sitk.ReadImage(images_path / img)
        normalized_np = sitk.GetArrayFromImage(normalized_img)
        np.testing.assert_allclose(normalized_np, solution)


def test_normalize_dataset():
    ct_solutions, threshold_solutions, no_threshold_solutions = setup_dataset()
    normalize_dataset(
        TEST_DATA_DIR / "crops",
        TEST_DATA_DIR / "CT",
        {
            "stats": (2.0, 2.0),
            "percentiles": (1.0, 4.0),
            "modality": "CT",
            "cropping_threshold_met": True,
        },
    )

    check_solutions(ct_solutions, TEST_DATA_DIR / "CT")

    normalize_dataset(
        TEST_DATA_DIR / "crops",
        TEST_DATA_DIR / "threshold",
        {"modality": "MRI", "cropping_threshold_met": True},
    )

    check_solutions(threshold_solutions, TEST_DATA_DIR / "threshold")

    normalize_dataset(
        TEST_DATA_DIR / "crops",
        TEST_DATA_DIR / "no_threshold",
        {"modality": "MRI", "cropping_threshold_met": False},
    )

    check_solutions(no_threshold_solutions, TEST_DATA_DIR / "no_threshold")

    tear_down_dataset()
