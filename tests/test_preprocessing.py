import pytest
import sys
from pathlib import Path
import os
import SimpleITK as sitk
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import load_data, crop_zeros, modality_detection, resample_image, compute_dataset_stats

TEST_DATA_DIR = Path(__file__).parent / "test_data"

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
    spacings = [(1., 1., 1.), (.325, .1, .1), (10., 5., 3.), (.5, 1., 1.), (.4, .2, .2)]
    for img, spacing in zip(imgs, spacings):
        img.SetSpacing(spacing)

    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f'imagesTr/test0{i + 1}.nii.gz')

    imgs = [sitk.GetImageFromArray(mask) for mask in masks]

    for i, img in enumerate(imgs):
        sitk.WriteImage(img, TEST_DATA_DIR / f'labelsTr/test0{i + 1}.nii.gz')

    return arrs, masks
    


def test_compute_dataset_stats_no_CT():
    setup_dataset()

    stats = compute_dataset_stats(TEST_DATA_DIR, 'NOT CT', 'imagesTr', 'labelsTr')

    np.testing.assert_array_equal(stats['shape'], np.array([3, 3, 3]))
    np.testing.assert_array_equal(stats['spacing'], np.array([.5, 1., 1.]))
    
def test_compute_dataset_CT():
    arrs, masks = setup_dataset()

    masked_vals = np.concat([arr[mask == 1] for arr, mask in zip(arrs, masks)])

    stats = compute_dataset_stats(TEST_DATA_DIR, 'NOT CT', 'imagesTr', 'labelsTr')
    #mean, std = stats['stats']
    #low, high = stats['percentiles']

    #np.testing.assert_almost_equal(masked_vals.mean(), mean)
    #np.testing.assert_almost_equal(masked_vals.std(), std)
    #np.testing.assert_almost_equal(np.percentile(masked_vals, .5), low)
    #np.testing.assert_almost_equal(np.percentile(masked_vals, 99.5), high)
    np.testing.assert_array_equal(stats['shape'], np.array([3, 3, 3]))
    np.testing.assert_array_equal(stats['spacing'], np.array([.5, 1., 1.]))
    