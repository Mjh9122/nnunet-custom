import os
import pickle as pkl
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
    crop_dataset,
    determine_cascade_necessity,
    lower_resolution,
    modality_detection,
    normalize_dataset,
    preprocess_dataset,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
CT_DATASET_DIR = Path(
    "/mnt/d/dummy_CT-SKIPTEST"
)  # Takes a long time, remove -SKIPTEST to run test

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
    if os.path.exists(TEST_DATA_DIR / input):
        assert modality_detection(TEST_DATA_DIR / input) == expected


def test_modality_dectection_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        modality_detection(TEST_DATA_DIR / "dataset_05.json")


def setup_dataset():
    arr1 = np.zeros((10, 10, 10), np.float32)
    arr1[3:7, 3:7, 3:7] = np.arange(64).reshape(4, 4, 4)
    arr2 = np.ones((3, 3, 3), np.float32) * 10
    arr3 = np.zeros((10, 10, 10), np.float32)
    arr3[2:8, 2:8, 2:8] = 5
    arr4 = np.arange(75)
    arr4 = arr4.reshape((5, 3, 5)).astype(np.float32)
    arr5 = np.arange(50)
    arr5 = arr5.reshape((5, 5, 2)).astype(np.float32)
    arrs = [arr1, arr2, arr3, arr4, arr5]

    mask1 = np.zeros((10, 10, 10))
    mask1[3:8, 4:7, 4:7] = 1
    mask2 = np.ones((3, 3, 3))
    mask3 = np.zeros((10, 10, 10))
    mask3[0, 0, 1] = 1
    mask4 = np.zeros((5, 3, 5))
    mask4[1, :, :] = 1
    mask5 = np.zeros((5, 5, 2))
    mask5[:, :, 0] = 1
    masks = [mask1, mask2, mask3, mask4, mask5]

    imgs = [sitk.GetImageFromArray(arr) for arr in arrs]
    spacings = [
        (1.0, 1.0, 1.0),
        (0.325, 0.1, 0.1),
        (10.0, 5.0, 3.0),
        (0.5, 1.0, 1.0),
        (0.4, 0.2, 0.2),
    ]
    for img, spacing in zip(imgs, spacings):
        img.SetSpacing(spacing)

    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f"imagesTr/test0{i + 1}.nii.gz")

    imgs = [sitk.GetImageFromArray(mask) for mask in masks]

    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f"labelsTr/test0{i + 1}.nii.gz")

    return arrs, masks


def test_compute_dataset_stats_no_CT():
    setup_dataset()

    stats = compute_dataset_stats(TEST_DATA_DIR / "crops", "NOT CT")

    np.testing.assert_array_equal(stats["pre_crop_shape"], np.array([5, 5, 5]))
    np.testing.assert_array_equal(stats["post_crop_shape"], np.array([5, 4, 4]))
    np.testing.assert_array_equal(stats["spacing"], np.array([0.5, 1.0, 1.0]))


def test_compute_dataset_CT():
    arrs, masks = setup_dataset()

    # Manually crop zeros
    arrs[0] = arrs[0][3:7, 3:7, 3:7]
    masks[0] = masks[0][3:7, 3:7, 3:7]
    arrs[2] = arrs[2][2:8, 2:8, 2:8]
    masks[2] = masks[2][2:8, 2:8, 2:8]

    # only take values in foreground
    masked_vals = np.concat([arr[mask != 0] for arr, mask in zip(arrs, masks)])

    stats = compute_dataset_stats(TEST_DATA_DIR / "crops", "CT")
    mean, std = stats["stats"]
    low, high = stats["percentiles"]

    assert abs(masked_vals.mean() - mean) < (masked_vals.mean() * 0.001)
    assert abs(masked_vals.std() - std) < (masked_vals.std() * 0.001)

    assert (
        abs(np.percentile(masked_vals, 0.5) - low)
        < (np.percentile(masked_vals, 0.5) * 0.001) + 1e-9
    )
    assert (
        abs(np.percentile(masked_vals, 99.5) - high)
        < (np.percentile(masked_vals, 99.5) * 0.001) + 1e-9
    )

    np.testing.assert_array_equal(stats["pre_crop_shape"], np.array([5, 5, 5]))
    np.testing.assert_array_equal(stats["post_crop_shape"], np.array([5, 4, 4]))
    np.testing.assert_array_equal(stats["spacing"], np.array([0.5, 1.0, 1.0]))


def test_reservoir_on_real_ct_masks():
    if not os.path.exists(CT_DATASET_DIR):
        return
    image_path = CT_DATASET_DIR / "imagesTr"
    label_path = CT_DATASET_DIR / "labelsTr"

    images = os.listdir(image_path)

    all_voxels = []
    shapes = []
    spacings = []

    for image in images:
        img = sitk.ReadImage(image_path / image)
        mask = sitk.ReadImage(label_path / image)

        img_np = sitk.GetArrayFromImage(img)
        mask_np = sitk.GetArrayFromImage(mask)

        masked_voxels = img_np[mask_np != 0]

        all_voxels.extend(masked_voxels)

        shapes.append(img.GetSize())
        spacings.append(img.GetSpacing())

    stats = compute_dataset_stats(CT_DATASET_DIR, "CT")

    all_voxels = np.array(all_voxels)
    mean = all_voxels.mean()
    std = all_voxels.std()

    shapes = np.array(shapes).T
    shape = np.median(shapes, 1)

    spacings = np.array(spacings).T
    spacing = np.median(spacings, 1)

    assert abs(all_voxels.mean() - mean) < (all_voxels.mean() * 0.001)
    assert abs(all_voxels.std() - std) < (all_voxels.std() * 0.001)
    mean, std = stats["stats"]
    low, high = stats["percentiles"]

    assert low > np.percentile(all_voxels, 0.25)
    assert low < np.percentile(all_voxels, 0.75)
    assert high > np.percentile(all_voxels, 99.25)
    assert high < np.percentile(all_voxels, 99.75)

    np.testing.assert_array_equal(stats["precrop_shape"], shape)
    np.testing.assert_array_equal(stats["spacing"], spacing)


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


def test_preprocessing():
    pass


# def test_crop_dataset():
#     setup_dataset()

#     cropped_dir = crop_dataset(TEST_DATA_DIR, TEST_DATA_DIR)
#     cropped_imgs_dir = cropped_dir / "imagesTr"
#     cropped_labels_dir = cropped_dir / "labelsTr"

#     original_shapes = []
#     cropped_shapes = []
#     spacings = []
#     images = []

#     for f in os.listdir(cropped_imgs_dir):
#         if f[-3:] == "pkl":
#             with open(cropped_imgs_dir / f, "rb") as file:
#                 img_stat = pkl.load(file)
#                 original_shapes.append(img_stat["pre_crop_shape"])
#                 cropped_shapes.append(img_stat["post_crop_shape"])
#                 spacings.append(img_stat["spacing"])
#         else:
#             images.append(f)

#     assert all(
#         [np.prod(a) >= np.prod(b) for a, b in zip(original_shapes, cropped_shapes)]
#     )
#     assert len(spacings) == len(original_shapes) and len(spacings) == len(
#         cropped_shapes
#     )
#     assert all([lbl in images for lbl in os.listdir(cropped_labels_dir)])


# def test_normalize_dataset():
#     normalize_dataset(
#         TEST_DATA_DIR / 'crops' / 'imagesTr',
#         TEST_DATA_DIR / 'normalized' / 'imagesTr',
#         {'modality': 'CT',
#          'cropping_threshold_met':True,
#          'stats':(1, 1),
#          'percentiles':(0, 10)}
#     )
