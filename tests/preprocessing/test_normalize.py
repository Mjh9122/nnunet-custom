import sys
from pathlib import Path
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from preprocessing.normalize import normalize

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