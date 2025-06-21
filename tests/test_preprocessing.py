import pytest
import sys
from pathlib import Path
import os
import SimpleITK as sitk
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import load_data

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

