import pytest
import sys
from pathlib import Path
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.resample import resample_image

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