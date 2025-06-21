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
        'input',
        ((np.ones((4,4), dtype=np.float32)), 
         (np.ones((1,2,3), dtype=np.float32)), 
         (np.ones((16,16,16), dtype=np.float32)), 
         )
)
def test_crop_bool_no_crop(input):
    cropped, norm_bool = crop_zeros(input)
    np.testing.assert_array_equal(cropped, input, np.float32)
    assert norm_bool == False

def test_crop_bool_3d_false():
    arr = np.zeros((64, 64, 64), np.float32)
    arr[1:, 1:, 1:] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones((63, 63, 63), np.float32))
    assert norm_bool == False

def test_crop_bool_3d_true():
    arr = np.zeros((64, 64, 64), np.float32)
    arr[10:40, 40:60, 0:64] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones((30, 20, 64), np.float32))
    assert norm_bool == True

def test_crop_bool_2d_false():
    arr = np.zeros((64, 64), np.float32)
    arr[4:, :60] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones((60, 60), np.float32))
    assert norm_bool == False

def test_crop_bool_2d_true():
    arr = np.zeros((64, 64), np.float32)
    arr[5:10, 5:40] = 1
    cropped, norm_bool = crop_zeros(arr)
    np.testing.assert_array_equal(cropped, np.ones((5, 35), np.float32))
    assert norm_bool == True


