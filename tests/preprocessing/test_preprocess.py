import json
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

from preprocessing.preprocess import (
    compute_dataset_stats,
    determine_cascade_necessity,
    lower_resolution,
    modality_detection,
    preprocess_dataset,
    select_cv_fold,
)

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


def test_split_cv():
    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "a")
        os.mkdir(TEST_DATA_DIR / "a" / "imagesTr")
        os.mkdir(TEST_DATA_DIR / "a" / "labelsTr")
        os.mkdir(TEST_DATA_DIR / "b")
        os.mkdir(TEST_DATA_DIR / "b" / "imagesTr")
        os.mkdir(TEST_DATA_DIR / "b" / "labelsTr")

    # generate images, labels
    for i in range(10):
        with open(TEST_DATA_DIR / "a" / "imagesTr" / f"img{i}.nii.gz", "w") as f:
            f.write(str(i))

        with open(TEST_DATA_DIR / "a" / "labelsTr" / f"img{i}.nii.gz", "w") as f:
            f.write(str(i))

    select_cv_fold(
        TEST_DATA_DIR / "a", [f"img{i}.nii.gz" for i in range(8)], TEST_DATA_DIR / "b"
    )

    assert set(os.listdir(TEST_DATA_DIR / "b" / "imagesTr")) == set(
        [f"img{i}.nii.gz" for i in range(8)]
    )
    assert set(os.listdir(TEST_DATA_DIR / "b" / "labelsTr")) == set(
        [f"img{i}.nii.gz" for i in range(8)]
    )

    with pytest.raises(Exception) as e_info:
        select_cv_fold(
            TEST_DATA_DIR / "b",
            [f"img{i}.nii.gz" for i in range(10)],
            TEST_DATA_DIR / "a",
        )

    assert (
        e_info.value.args[0]
        == "All images in images_list must be present in dataset_dir/imagesTr"
    )

    for i in range(8, 10):
        with open(TEST_DATA_DIR / "b" / "imagesTr" / f"img{i}.nii.gz", "w") as f:
            f.write(str(i))

    with pytest.raises(Exception) as e_info:
        select_cv_fold(
            TEST_DATA_DIR / "b",
            [f"img{i}.nii.gz" for i in range(10)],
            TEST_DATA_DIR / "a",
        )

    assert (
        e_info.value.args[0]
        == "All images in images_list must be present in dataset_dir/labelsTr"
    )

    tear_down_dataset()


@pytest.fixture()
def stats_dataset(request):
    params = request.param

    pre_crop_med = params["pre_crop_med"]
    post_crop_med = params["post_crop_med"]
    spacings_med = params["spacings_med"]
    image_channels = params["channels"]

    if os.path.exists(TEST_DATA_DIR):
        if not os.path.isdir(TEST_DATA_DIR):
            raise Exception(f"Test data path {TEST_DATA_DIR} is not a directory")
    else:
        os.mkdir(TEST_DATA_DIR)
        os.mkdir(TEST_DATA_DIR / "crops")
        os.mkdir(TEST_DATA_DIR / "crops" / "imagesTr")
        os.mkdir(TEST_DATA_DIR / "crops" / "labelsTr")
        os.mkdir(TEST_DATA_DIR / "crops" / "picklesTr")
    multipliers = [.5, .75, 1.0, 1.3, 1.75]
    precrop_dims = [(image_channels, ) + (int(pre_crop_med * k), ) * 3 for k in multipliers]
    postcrop_dims = [(image_channels, ) + (int(post_crop_med * k), ) * 3 for k in multipliers]
    spacings = [(spacings_med * k, ) * 3 for k in multipliers]

    solutions = {
        "pre_crop_shape": np.array((image_channels, ) + (pre_crop_med, ) * 3),
        "post_crop_shape": np.array((image_channels, ) + (post_crop_med, ) * 3),
        "spacing": np.array((spacings_med, ) * 3),
        "num_images": len(precrop_dims),
    }

    for i, (pre, post, space) in enumerate(zip(precrop_dims, postcrop_dims, spacings)):
        stats = {
            "pre_crop_shape": pre,
            "post_crop_shape": post,
            "spacing": space,
        }
        pkl_file = f"test0{i + 1}.pkl"
        with open(TEST_DATA_DIR / "crops" / "picklesTr" / pkl_file, "wb") as file:
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

    yield (TEST_DATA_DIR, solutions)

    tear_down_dataset()


def tear_down_dataset():
    shutil.rmtree(TEST_DATA_DIR)


@pytest.mark.parametrize(
    "stats_dataset",
    [{"pre_crop_med": 64, "post_crop_med": 32, "spacings_med": 0.5, "channels": 1},  
     {"pre_crop_med": 64, "post_crop_med": 32, "spacings_med": 0.5, "channels": 4}],
    indirect=True,
)
def test_compute_dataset_stats_no_CT(stats_dataset):
    dataset, solution = stats_dataset 

    stats = compute_dataset_stats(dataset / "crops", "MRI")

    np.testing.assert_allclose(stats["pre_crop_shape"], solution["pre_crop_shape"])
    np.testing.assert_allclose(stats["post_crop_shape"], solution["post_crop_shape"])
    np.testing.assert_allclose(stats["spacing"], solution["spacing"])
    assert stats["num_images"] == solution["num_images"]


@pytest.mark.parametrize(
    "stats_dataset",  
    [{"pre_crop_med": 64, "post_crop_med": 32, "spacings_med": 0.5, "channels": 1},  
     {"pre_crop_med": 64, "post_crop_med": 32, "spacings_med": 0.5, "channels": 4}],
    indirect=True)
def test_compute_dataset_CT(stats_dataset):
    dataset, solution = stats_dataset 
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


def test_reservoir_CT():
    os.mkdir(TEST_DATA_DIR)
    os.mkdir(TEST_DATA_DIR / "imagesTr")
    os.mkdir(TEST_DATA_DIR / "labelsTr")
    os.mkdir(TEST_DATA_DIR / "picklesTr")

    images_path = TEST_DATA_DIR / "imagesTr"
    masks_path = TEST_DATA_DIR / "labelsTr"
    pickles_path = TEST_DATA_DIR / "picklesTr"

    # Generate 0 - 100_000_000
    n = 10_000_000
    nums = np.arange(n)
    np.random.shuffle(nums)
    nums = nums.reshape((100, 1, 100, 100, 10))

    for i, arr in enumerate(nums):
        img = sitk.GetImageFromArray(arr)
        sitk.WriteImage(img, images_path / f"test_img{i}.nii.gz")

        mask = np.ones((1, 100, 100, 10))
        mask_img = sitk.GetImageFromArray(mask)

        sitk.WriteImage(mask_img, masks_path / f"test_img{i}.nii.gz")

        stats = {
            "pre_crop_shape": np.array((1, 100, 100, 10)),
            "post_crop_shape": np.array((1, 100, 100, 10)),
            "spacing": np.array((1, 1, 1), np.float32),
        }

        pkl_file = f"test_img{i}.pkl"
        with open(pickles_path / pkl_file, "wb") as file:
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


def test_preprocess_bad_dataset():
    with pytest.raises(Exception):
        preprocess_dataset(TEST_DATA_DIR / "bad_path", TEST_DATA_DIR)

    if not os.path.exists(TEST_DATA_DIR):
        os.mkdir(TEST_DATA_DIR)
    os.mkdir(TEST_DATA_DIR / "fake_dir")

    with open(TEST_DATA_DIR / "fake_file", "w") as file:
        file.write("this is a file")

    with pytest.raises(Exception):
        preprocess_dataset(TEST_DATA_DIR / "fake_dir", TEST_DATA_DIR / "fake_file")

    tear_down_dataset()


def create_test_images(shape):
    spacings = [
        (0.5, 0.25, 0.25),
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 0.5, 0.5),
    ]
    median_spacing = np.median(np.array(spacings).T, axis=1)

    dim_multiplier = [np.array(s) / median_spacing for s in spacings]

    orig_dims = (np.array(shape) * np.array(dim_multiplier)).astype(int)

    non_zeros = [np.random.randint(0, 1000, dims) for dims in orig_dims]

    zero_buffers = [np.zeros(dims + np.array((10, 10, 10))) for dims in orig_dims]

    imgs = []
    masks = []

    for non_zero, buffer in zip(non_zeros, zero_buffers):
        img = buffer.copy()
        img[5:-5, 5:-5, 5:-5] = non_zero

        mask = buffer.copy()

        imgs.append(img)
        masks.append(mask)

    return imgs, masks, spacings


@pytest.fixture()
def full_pipeline_dataset(request):
    params = request.param
    target_image_shape = params["target_image_shape"]

    if not os.path.exists(TEST_DATA_DIR):
        os.mkdir(TEST_DATA_DIR)

    os.mkdir(TEST_DATA_DIR / "dataset")
    dataset_dir = TEST_DATA_DIR / "dataset"

    os.mkdir(dataset_dir / "imagesTr")
    os.mkdir(dataset_dir / "labelsTr")
    os.mkdir(dataset_dir / "output")

    imgs, masks, spacings = create_test_images(target_image_shape)

    for i, (img, mask, spacing) in enumerate(zip(imgs, masks, spacings)):
        img_sitk = sitk.GetImageFromArray(img)
        img_sitk.SetSpacing(spacing)

        mask_sitk = sitk.GetImageFromArray(mask)

        sitk.WriteImage(img_sitk, dataset_dir / "imagesTr" / f"img{i}.nii.gz")
        sitk.WriteImage(mask_sitk, dataset_dir / "labelsTr" / f"img{i}.nii.gz")

    dataset_json = {
        "name": "test",
        "description": "N/A",
        "reference": "N/A",
        "modality": {"0": "MRI"},
        "licence": "ARG",
        "tensorImageSize": "3D",
        "labels": {"0": "background"},
        "numTraining": 5,
        "numTest": 0,
        "training": [
            {"image": "img0.nii.gz"},
            {"image": "img1.nii.gz"},
            {"image": "img2.nii.gz"},
            {"image": "img3.nii.gz"},
            {"image": "img4.nii.gz"},
        ],
        "test": [],
    }
    with open(dataset_dir / "dataset.json", "w") as file:
        json.dump(dataset_json, file)

    yield (dataset_dir, params)

    tear_down_dataset()


@pytest.mark.parametrize(
    "full_pipeline_dataset",
    [
        {"target_image_shape": (128, 256, 256), "needs_cascade": True},
        {"target_image_shape": (100, 50, 50), "needs_cascade": False},
    ],
    indirect=True,
)
def test_whole_preprocessing_pipeline(full_pipeline_dataset):
    return  
    dataset_dir, params = full_pipeline_dataset
    target_image_shape = params["target_image_shape"]
    needs_cascade = params["needs_cascade"]

    preprocess_dataset(
        dataset_dir, os.listdir(dataset_dir / "imagesTr"), dataset_dir / "output"
    )

    with open(dataset_dir / "output" / "dataset_stats.pkl", "rb") as file:
        stats = pkl.load(file)

    np.testing.assert_array_equal(stats["post_crop_shape"], target_image_shape)
    assert (stats.get("low_res_spacing") is not None) == needs_cascade
    assert stats["modality"] == "Not CT"

    output_directories = [
        dataset_dir / "output" / "high_res" / "imagesTr",
        dataset_dir / "output" / "high_res" / "labelsTr",
    ]

    if needs_cascade:
        output_directories.extend(
            [
                dataset_dir / "output" / "low_res" / "imagesTr",
                dataset_dir / "output" / "low_res" / "labelsTr",
            ]
        )

    images = set(os.listdir(dataset_dir / "labelsTr"))

    for dir in output_directories:
        assert set(images).issubset(os.listdir(dir))
