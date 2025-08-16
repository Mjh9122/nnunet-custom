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
    img = np.zeros((1, 4, 4, 4), dtype=np.float32)
    nonzeros = np.arange(16).reshape(1, 4, 2, 2)
    img[:, :, 1:3, 1:3] = nonzeros

    clipping_thresh = (2.0, 14.0)
    mean_std = (5.0, 6.0)

    result = normalize(
        img,
        "CT",
        cropping_threshold_met=False,
        dataset_stats=mean_std,
        clipping_percentiles=clipping_thresh,
    )

    expected = np.clip(img, *clipping_thresh)
    expected -= mean_std[0]
    expected /= mean_std[1]

    np.testing.assert_allclose(expected, result)


def test_normalize_non_ct_nonzeros():
    img = np.zeros((1, 4, 4, 4), dtype=np.float32)
    nonzeros = np.arange(16).reshape(1, 4, 2, 2)
    img[:, :, 1:3, 1:3] = nonzeros

    result = normalize(img, "MRI", cropping_threshold_met=False)

    expected = img.copy()
    expected -= img.mean()
    expected /= img.std()

    np.testing.assert_allclose(expected, result)


def test_normalize_multichannel():
    img = np.zeros((3, 4, 4, 4), dtype=np.float32)
    nonzeros = np.arange(24).reshape((3, 2, 2, 2))
    img[:, 1:3, 1:3, 1:3] = nonzeros

    result = normalize(img, "MRI", cropping_threshold_met=False)

    expected = img.copy()
    expected -= img.mean(axis=(1, 2, 3)).reshape((3, 1, 1, 1))
    expected /= img.std(axis=(1, 2, 3)).reshape((3, 1, 1, 1))

    np.testing.assert_allclose(expected, result)


def test_threshold_single_channel():
    img = np.zeros((1, 4, 4, 4), dtype=np.float32)
    nonzeros = np.arange(16).reshape(1, 4, 2, 2)
    img[:, :, 1:3, 1:3] = nonzeros

    result = normalize(img, "MRI", cropping_threshold_met=True)

    expected = img.copy()
    expected -= img[img != 0].mean()
    expected /= img[img != 0].std()

    np.testing.assert_allclose(expected, result)


def test_threshold_multi_channel():
    img = np.zeros((3, 4, 4, 4), dtype=np.float32)
    nonzeros = np.arange(48).reshape(3, 4, 2, 2)
    img[:, :, 1:3, 1:3] = nonzeros

    result = normalize(img, "MRI", cropping_threshold_met=True)

    for i, channel in enumerate(img):
        expected = channel.copy()
        expected -= channel[channel != 0].mean()
        expected /= channel[channel != 0].std()
        np.testing.assert_allclose(result[i], expected)


def test_incorrect_params():
    with pytest.raises(Exception):
        normalize(np.ones((1, 10, 10, 10)), "CT", True, dataset_stats=(1, 1))

    with pytest.raises(Exception):
        normalize(np.ones((1, 10, 10, 10)), "CT", True, clipping_percentiles=(1, 1))


def setup_dataset():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "crops")

    arr1 = np.random.randint(-128, 1000, (4, 64, 64, 64))
    arr2 = np.random.randint(-128, 1000, (4, 64, 64, 64))
    arr3 = np.random.randint(-128, 1000, (4, 64, 64, 64))

    arrs = [arr1, arr2, arr3]


    arrs_t = [np.transpose(arr, (3, 2, 1, 0)) for arr in arrs]

    imgs = [sitk.GetImageFromArray(arr) for arr in arrs_t]
    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / "crops" / f"test0{i}.nii.gz")

    # CT args
    mean, std, low, high = 250, 100, 0, 256
    CT_map = np.vectorize(
        {i: ((min(max(i, low), high) - mean) / std) for i in range(-128, 1001)}.get
    )
    CT_solutions = [CT_map(arr) for arr in arrs]

    threshold_solutions = []
    for img in arrs:
        solution = np.zeros_like(img)
        for i, channel in enumerate(img):
            expected = channel.astype(float)
            expected -= channel[channel != 0].mean()
            expected /= channel[channel != 0].std()
            solution[i] = expected
        threshold_solutions.append(solution)

    no_threshold_solutions = []
    for img in arrs:
        solution = np.zeros_like(img)
        for i, channel in enumerate(img):
            expected = channel.astype(float)
            solution[i] = (expected - channel.mean()) / channel.std()
    return CT_solutions, threshold_solutions, no_threshold_solutions


def tear_down_dataset():
    shutil.rmtree(TEST_DATA_DIR)


def check_solutions(solutions, images_path):
    images = sorted([f for f in os.listdir(images_path)])

    for solution, img in zip(solutions, images):
        normalized_img = sitk.ReadImage(images_path / img)
        normalized_np = sitk.GetArrayFromImage(normalized_img)
        normalized_np = np.transpose(normalized_np, (3, 2, 1, 0))
        np.testing.assert_allclose(normalized_np, solution)


def test_normalize_dataset():
    ct_solutions, threshold_solutions, no_threshold_solutions = setup_dataset()
    normalize_dataset(
        TEST_DATA_DIR / "crops",
        TEST_DATA_DIR / "CT",
        {
            "stats": (250, 100),
            "percentiles": (0, 256),
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

    for image in os.listdir(TEST_DATA_DIR / "no_threshold"):
        normed = sitk.ReadImage(TEST_DATA_DIR / "no_threshold" / image)
        normed_np = sitk.GetArrayFromImage(normed)
        normed_np = np.transpose(normed_np, (3, 2, 1, 0))

        np.testing.assert_allclose(
            normed_np.mean(axis=(1, 2, 3)), np.zeros(normed_np.shape[0]), atol=1e-7
        )
        np.testing.assert_allclose(
            normed_np.std(axis=(1, 2, 3)), np.ones(normed_np.shape[0]), atol=1e-7
        )

    tear_down_dataset()
