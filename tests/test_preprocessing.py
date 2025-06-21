import pytest
import sys
from pathlib import Path
import os
import SimpleITK as sitk
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import load_data, crop_zeros

TEST_DATA_DIR = Path(__file__).parent / "test_data"


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
        'input_dims, ones_slices, output_dims, reduction',
        (((4, 4), [(0, 4), (0, 4)], (4, 4), False), 
         ((1,2,3), [(0, 1), (0, 2), (0, 3)], (1, 2, 3), False), 
         ((16,16,16), [(0, 16), (0, 16), (0, 16)], (16, 16, 16), False), 
         ((2, 2, 2, 2, 2), [(0, 2) for _ in range(5)] , (2, 2, 2, 2, 2), False), 
         ((64, 64, 64), [(1, 64), (1, 64), (1, 64)], (63, 63, 63), False),
         ((64, 64, 64), [(10, 40), (20, 50), (0, 64)], (31, 31, 64), True),
         ((64, 64), [(4, 64), (0, 60)], (60, 61), False),
         ((64, 64), [(5, 9), (5, 39)], (5, 35), True)
         )
)
def test_crop_bool(input_dims, ones_slices, output_dims, reduction):
    arr = np.zeros(input_dims, np.float32)
    if ones_slices is not None:
        slices = tuple(slice(a, b + 1) for a, b in ones_slices)
        arr[slices] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones(output_dims, np.float32))
    assert norm_bool == reduction


