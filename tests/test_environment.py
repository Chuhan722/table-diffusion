import numpy as np


def test_numpy_array_sum():
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
