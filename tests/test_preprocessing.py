import pytest
import sys
from pathlib import Path
import os
import SimpleITK as sitk
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import load_data, crop_zeros, modality_detection, resample_image

TEST_DATA_DIR = Path(__file__).parent / "test_data"

np.random.seed(42)


def test_load_data_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        load_data(TEST_DATA_DIR / "test001.nii.gz")


def test_load_dir_path():
    with pytest.raises(RuntimeError):
        load_data(TEST_DATA_DIR)


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
    assert modality_detection(TEST_DATA_DIR / input) == expected


def test_modality_dectection_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        modality_detection(TEST_DATA_DIR / "dataset_05.json")

@pytest.mark.parametrize(
        'image, old_spacing, new_spacing',
        (
            (np.ones((4, 4, 4)), np.array((1., 1., 1.)), np.array((2., 2.))),
            (np.ones((4, 4, 4, 4)), np.array((1., 1., 1.)), np.array((2., 2., 2.)))
        )
)
def test_resample_image_bad_spacings(image, old_spacing, new_spacing):
    with pytest.raises(Exception):
        resample_image(image, old_spacing, new_spacing, is_segmentation = False)

@pytest.mark.parametrize(
    'old_dims, old_spacing, new_spacing, expected',
    (
        ((16, 32, 32), (2., 1., 1.), (2., 2., 2.), (16, 16, 16)),
        ((16, 16, 16), (.33, .33, .33), (.33, .33, .33), (16, 16, 16)),
        ((100, 100, 50), (1., 1., 2.), (.5, .5, 1.), (200, 200, 100)),
        ((100, 100, 50), (1., 1., 2.), (10., 10., 10.), (10, 10, 10)),
        ((10, 10, 10), (10., 10., 10.), (1., 1., 2.), (100, 100, 50))
    )
)
def test_resample_image_expected_dims(old_dims, old_spacing, new_spacing, expected):
    img = np.random.rand(*old_dims)
    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation = False)

    assert expected == reshaped.shape

def test_rasample_image_range_preservation():
    old_dims = (64, 32, 32)
    old_spacing = (1., .5, .5)
    new_spacing = (1.5, .75, .75)
    img = np.random.rand(*old_dims) * 1000

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation = False)
    
    tol = .05 * (img.max() - img.min())
    assert reshaped.min() > img.min() - tol
    assert reshaped.max() < img.max() + tol

def test_resample_image_energy_preservation():
    old_dims = (64, 32, 32)
    old_spacing = (1., .5, .5)
    new_spacing = (1., 1., 1.)
    img = np.random.rand(*old_dims) * 1000

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation = False)
    
    tol = .005
    sum_diff = abs(img.sum() * np.prod(old_spacing) - reshaped.sum() * np.prod(new_spacing))

    assert  sum_diff / img.sum() * np.prod(old_spacing) < tol

def test_resample_image_uniform():
    const = 1000
    old_dims = (64, 32, 32)
    old_spacing = (5., .1, .1)
    new_spacing = (1., 1., 1.)
    img = np.ones(old_dims) * const

    reshaped = resample_image(img, old_spacing, new_spacing, is_segmentation = False)

    err = np.abs(reshaped - const)
    rtol = .01

    assert np.max(err) < rtol * const

def test_resample_image_seg_mask_up():
    img = np.array([[[1], [0]], [[0], [1]]])
    old_spacing  = (2, 2, 2)
    new_spacing = (1, 1, 2)

    expected = np.array([
        [[1], [1], [0], [0]], 
        [[1], [1], [0], [0]],
        [[0], [0], [1], [1]],
        [[0], [0], [1], [1]]
    ])

    reshaped = resample_image(img, old_spacing, new_spacing, True)

    np.testing.assert_array_equal(expected, reshaped)

def test_resample_image_seg_mask_down():
    img = np.array([
        [[1], [1], [0], [0]], 
        [[1], [1], [0], [0]],
        [[0], [0], [1], [1]],
        [[0], [0], [1], [1]]
    ])
    old_spacing  = (1, 1, 2)
    new_spacing = (2, 2, 2)

    expected = np.array([[[1], [0]], [[0], [1]]])

    reshaped = resample_image(img, old_spacing, new_spacing, True)

    np.testing.assert_array_equal(expected, reshaped)

