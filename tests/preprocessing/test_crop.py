import sys
from pathlib import Path

import numpy as np
import pytest

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.crop import crop_zeros


@pytest.mark.parametrize(
    "input_dims, ones_slices, output_dims",
    (
        ((4, 4), [(0, 4), (0, 4)], (4, 4)),
        ((1, 2, 3), [(0, 1), (0, 2), (0, 3)], (1, 2, 3)),
        ((16, 16, 16), [(0, 16), (0, 16), (0, 16)], (16, 16, 16)),
        ((2, 2, 2, 2, 2), [(0, 2) for _ in range(5)], (2, 2, 2, 2, 2)),
        ((64, 64, 64), [(1, 64), (1, 64), (1, 64)], (63, 63, 63)),
        ((64, 64, 64), [(10, 40), (20, 50), (0, 64)], (31, 31, 64)),
        ((64, 64), [(4, 64), (0, 60)], (60, 61)),
        ((64, 64), [(5, 9), (5, 39)], (5, 35)),
    ),
)
def test_crop_bool(input_dims, ones_slices, output_dims):
    arr = np.zeros(input_dims, np.float32)
    if ones_slices is not None:
        slices = tuple(slice(a, b + 1) for a, b in ones_slices)
        arr[slices] = 1

    seg_mask = arr.copy()

    cropped_arr, cropped_mask = crop_zeros(arr, seg_mask)
    np.testing.assert_array_equal(cropped_arr, np.ones(output_dims, np.float32))
    np.testing.assert_array_equal(cropped_mask, np.ones(output_dims, np.float32))
