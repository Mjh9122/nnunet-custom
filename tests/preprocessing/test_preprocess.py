import os
import pickle as pkl
import shutil
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import numpy as np
import SimpleITK as sitk

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.preprocess import (compute_dataset_stats,
                                      determine_cascade_necessity,
                                      lower_resolution, modality_detection,
                                      preprocess_dataset)

TEST_DATA_DIR = Path(__file__).parent / "preprocess_data"
TEST_JSON_DIR = Path(__file__).parent / "json_data"

np.random.seed(42)


@pytest.mark.parametrize(
    "input, expected",
    (
        ("dataset_01.json", "Not CT"),
        ("dataset_02.json", "CT"),
        ("dataset_03.json", "CT"),
        ("dataset_04.json", "No Modality Detected"),
    ),
)
def test_modality_detection(input, expected):
    if os.path.exists(TEST_JSON_DIR / input):
        assert modality_detection(TEST_JSON_DIR / input) == expected


def test_modality_dectection_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        modality_detection(TEST_JSON_DIR / "dataset_05.json")


@pytest.mark.parametrize(
    "input, expected",
    (
        ((115, 320, 232), True),
        ((482, 512, 512), True),
        ((138, 169, 138), False),
        ((382, 512, 512), True),
        ((36, 50, 35), False),
        ((20, 320, 319), False),
        ((252, 512, 512), True),
        ((96, 512, 512), True),
    ),
)
def test_cascade_necessity(input, expected):
    result = determine_cascade_necessity(input)

    assert result == expected


@pytest.mark.parametrize(
    "dims, spacing, expected",
    (
        ((512, 512, 400), (1.0, 1.0, 1.0), (4.0, 4.0, 4.0)),
        ((320, 320, 100), (0.5, 0.5, 2.0), (1.0, 1.0, 2.0)),
        ((1024, 1024, 64), (0.125, 0.125, 1.0), (0.5, 0.5, 1.0)),
        ((400, 400, 150), (0.8, 0.8, 2.5), (1.6, 1.6, 2.5)),
    ),
)
def test_lower_resolution(dims, spacing, expected):
    result = lower_resolution(dims, spacing)
    np.testing.assert_allclose(np.array(result), np.array(expected))


def setup_dataset():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "crops")
        os.mkdir(TEST_DATA_DIR / "crops" / "imagesTr")
        os.mkdir(TEST_DATA_DIR / "crops" / "labelsTr")

    precrop_dims = [
        [64, 32, 32],
        [64, 32, 32],
        [32, 16, 16],
        [32, 16, 16],
        [16, 16, 16],
    ]
    postcrop_dims = [[32, 32, 32], [16, 16, 16], [16, 16, 16], [16, 16, 16], [8, 8, 8]]
    spacings = [
        [0.5, 0.5, 0.5],
        [0.4, 0.3, 0.2],
        [0.4, 0.4, 0.4],
        [1.0, 1.0, 1.0],
        [0.2, 0.2, 0.2],
    ]

    solutions = {
        "pre_crop_shape": np.array([32, 16, 16]),
        "post_crop_shape": np.array([16, 16, 16]),
        "spacing": np.array([0.4, 0.4, 0.4]),
    }

    for i, (pre, post, space) in enumerate(zip(precrop_dims, postcrop_dims, spacings)):
        stats = {
            "pre_crop_shape": pre,
            "post_crop_shape": post,
            "spacing": space,
        }
        pkl_file = f"test0{i + 1}.pkl"
        with open(TEST_DATA_DIR / "crops" / "imagesTr" / pkl_file, "wb") as file:
            pkl.dump(stats, file)

    arrs = [np.random.randint(low=0, high=100, size=dims) for dims in postcrop_dims]
    masks = [
        np.random.choice([0, 1], size=dims, p=[0.2, 0.8]) for dims in postcrop_dims
    ]

    masked = np.concat([arr[mask != 0].flatten() for arr, mask in zip(arrs, masks)])
    mean = masked.mean()
    std = masked.std()
    low = np.percentile(masked, 0.5)
    high = np.percentile(masked, 99.5)

    solutions.update({"stats": (mean, std), "percentiles": (low, high)})

    imgs = [sitk.GetImageFromArray(arr) for arr in arrs]
    for img, spacing in zip(imgs, spacings):
        img.SetSpacing(spacing)

    for i, img in enumerate(imgs):
        sitk.WriteImage(
            img, TEST_DATA_DIR / "crops" / "imagesTr" / f"test0{i + 1}.nii.gz"
        )

    imgs = [sitk.GetImageFromArray(mask) for mask in masks]

    for i, img in enumerate(imgs):
        sitk.WriteImage(
            img, TEST_DATA_DIR / "crops" / "labelsTr" / f"test0{i + 1}.nii.gz"
        )

    return solutions


def tear_down_dataset():
    shutil.rmtree(TEST_DATA_DIR)


def test_compute_dataset_stats_no_CT():
    solution = setup_dataset()

    stats = compute_dataset_stats(TEST_DATA_DIR / "crops", "MRI")

    np.testing.assert_allclose(stats["pre_crop_shape"], solution["pre_crop_shape"])
    np.testing.assert_allclose(stats["post_crop_shape"], solution["post_crop_shape"])
    np.testing.assert_allclose(stats["spacing"], solution["spacing"])


def test_compute_dataset_CT():
    solution = setup_dataset()
    sol_mean, sol_std = solution["stats"]
    sol_low, sol_high = solution["percentiles"]

    stats = compute_dataset_stats(TEST_DATA_DIR / "crops", "CT")
    mean, std = stats["stats"]
    low, high = stats["percentiles"]

    assert abs(sol_mean - mean) < (sol_mean * 0.001)
    assert abs(sol_std - std) < (sol_std * 0.001)

    assert abs(sol_low - low) < (sol_low * 0.001) + 1e-9
    assert abs(sol_high - high) < (sol_high * 0.001) + 1e-9

    np.testing.assert_allclose(stats["pre_crop_shape"], solution["pre_crop_shape"])
    np.testing.assert_allclose(stats["post_crop_shape"], solution["post_crop_shape"])
    np.testing.assert_allclose(stats["spacing"], solution["spacing"])

    tear_down_dataset()


def test_reservoir_CT():
    os.mkdir(TEST_DATA_DIR)
    os.mkdir(TEST_DATA_DIR / "imagesTr")
    os.mkdir(TEST_DATA_DIR / "labelsTr")

    images_path = TEST_DATA_DIR / "imagesTr"
    masks_path = TEST_DATA_DIR / "labelsTr"

    # Generate 0 - 100_000_000
    n = 10_000_000
    nums = np.arange(n)
    np.random.shuffle(nums)
    nums = nums.reshape((100, 100, 100, 10))

    for i, arr in enumerate(nums):
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, images_path / f"test_img{i}.nii.gz")

        mask = np.ones((100, 100, 10))
        mask_img = sitk.GetImageFromArray(mask)

        sitk.WriteImage(mask_img, masks_path / f"test_img{i}.nii.gz")

        stats = {
            "pre_crop_shape": np.array((100, 100, 10)),
            "post_crop_shape": np.array((100, 100, 10)),
            "spacing": np.array((1, 1, 1), np.float32),
        }

        pkl_file = f"test_img{i}.pkl"
        with open(images_path / pkl_file, "wb") as file:
            pkl.dump(stats, file)

    results = compute_dataset_stats(TEST_DATA_DIR, "CT")

    true_mean = (n - 1) / 2
    true_std = np.sqrt(n * (n - 2) / 12)

    np.testing.assert_allclose((true_mean, true_std), results["stats"])

    low, high = results["percentiles"]

    assert low > n * (0.4 / 100)
    assert low < n * (0.6 / 100)
    assert high > n * (99.4 / 100)
    assert high < n * (99.6 / 100)

    tear_down_dataset()
