import pytest
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing import load_data

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_load_data_nonexistant_file():
    with pytest.raises(FileNotFoundError):
        load_data(TEST_DATA_DIR / "test001.nii.gz")


def test_load_file():
    assert np.ones(2, 2) == load_data()
