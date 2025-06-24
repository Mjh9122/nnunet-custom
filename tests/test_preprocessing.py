import pytest
import sys
from pathlib import Path
import os

import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="importlib._bootstrap"
)

import SimpleITK as sitk
import numpy as np


src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import (
    load_data,
    crop_zeros,
    modality_detection,
    resample_image,
    compute_dataset_stats,
    normalize,
    determine_cascade_necessity,
    lower_resolution,
)

TEST_DATA_DIR = Path(__file__).parent / "test_data"
CT_DATASET_DIR = Path(
    "/mnt/d/dummy_CT-SKIPTEST"
)  # Takes a long time, remove -SKIPTEST to run test

np.random.seed(42)


def test_load_data_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        load_data(TEST_DATA_DIR / "test001.nii.gz")


# def test_load_dir_path():
#     with pytest.raises(RuntimeError):
#         load_data(TEST_DATA_DIR)


def test_load_simple_file():
    arr = np.ones((2, 2, 2), np.float32)
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, TEST_DATA_DIR / "test002.nii.gz")
    np.testing.assert_array_equal(arr, load_data(TEST_DATA_DIR / "test002.nii.gz"))
    os.remove(TEST_DATA_DIR / "test002.nii.gz")


def test_load_large_file():
    arr = np.random.random((256, 256, 256))
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, TEST_DATA_DIR / "test003.nii.gz")
    np.testing.assert_array_equal(arr, load_data(TEST_DATA_DIR / "test003.nii.gz"))
    os.remove(TEST_DATA_DIR / "test003.nii.gz")


@pytest.mark.parametrize(
    "input_dims, ones_slices, output_dims, reduction",
    (
        ((4, 4), [(0, 4), (0, 4)], (4, 4), False),
        ((1, 2, 3), [(0, 1), (0, 2), (0, 3)], (1, 2, 3), False),
        ((16, 16, 16), [(0, 16), (0, 16), (0, 16)], (16, 16, 16), False),
        ((2, 2, 2, 2, 2), [(0, 2) for _ in range(5)], (2, 2, 2, 2, 2), False),
        ((64, 64, 64), [(1, 64), (1, 64), (1, 64)], (63, 63, 63), False),
        ((64, 64, 64), [(10, 40), (20, 50), (0, 64)], (31, 31, 64), True),
        ((64, 64), [(4, 64), (0, 60)], (60, 61), False),
        ((64, 64), [(5, 9), (5, 39)], (5, 35), True),
    ),
)
def test_crop_bool(input_dims, ones_slices, output_dims, reduction):
    arr = np.zeros(input_dims, np.float32)
    if ones_slices is not None:
        slices = tuple(slice(a, b + 1) for a, b in ones_slices)
        arr[slices] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones(output_dims, np.float32))
    assert norm_bool == reduction


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

    assert sum_diff / img.sum() * np.prod(old_spacing) < tol


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


def setup_dataset():
    arr1 = np.arange(64)
    arr1 = arr1.reshape((4, 4, 4)).astype(np.float32)
    arr2 = np.ones((3, 3, 3), np.float32) * 10
    arr3 = np.zeros((2, 2, 2), np.float32)
    arr4 = np.arange(30)
    arr4 = arr4.reshape((2, 3, 5)).astype(np.float32)
    arr5 = np.arange(50)
    arr5 = arr5.reshape((5, 5, 2)).astype(np.float32)
    arrs = [arr1, arr2, arr3, arr4, arr5]

    mask1 = np.zeros((4, 4, 4))
    mask1[1:3, 1:3, 1:3] = 1
    mask2 = np.ones((3, 3, 3))
    mask3 = np.zeros((2, 2, 2))
    mask3[0, 0, 1] = 1
    mask4 = np.zeros((2, 3, 5))
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

    stats = compute_dataset_stats(TEST_DATA_DIR, "NOT CT", "imagesTr", "labelsTr")

    np.testing.assert_array_equal(stats["shape"], np.array([3, 3, 3]))
    np.testing.assert_array_equal(stats["spacing"], np.array([0.5, 1.0, 1.0]))


def test_compute_dataset_CT():
    arrs, masks = setup_dataset()

    masked_vals = np.concat([arr[mask == 1] for arr, mask in zip(arrs, masks)])

    stats = compute_dataset_stats(TEST_DATA_DIR, "CT", "imagesTr", "labelsTr")
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

    np.testing.assert_array_equal(stats["shape"], np.array([3, 3, 3]))
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

    stats = compute_dataset_stats(CT_DATASET_DIR, "CT", "imagesTr", "labelsTr")

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

    np.testing.assert_array_equal(stats["shape"], shape)
    np.testing.assert_array_equal(stats["spacing"], spacing)


def test_normalize_ct_nonzero():
    img = np.zeros((4, 4), np.float32)
    img[1:3, 1:3] = np.arange(1, 5).reshape((2, 2))

    result = normalize(img, "CT", True, (2.0, 2.0), (2.0, 3.0))

    expected = np.zeros((4, 4), np.float32)
    expected[2, [1, 2]] = 1 / 2

    np.testing.assert_allclose(expected, result)


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
