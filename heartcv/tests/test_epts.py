import pytest
import numpy as np
import vuba

import heartcv as hcv


@pytest.fixture
def data():
    arr = np.random.randint(0, 255, size=(300, 600, 600), dtype=np.uint8)
    return hcv.epts(arr, fs=30, binsize=16)


@pytest.fixture
def expected_shape():
    # Expected output for mpx grid made with x16 binning
    return [(151, 38, 38), (151, 38, 38)]


@pytest.fixture
def expected_frequencies():
    return (0, 15)


def test_shape(data, expected_shape):
    for shape, ret in zip(expected_shape, data):
        assert shape == ret.shape


def test_frequencies(data, expected_frequencies):
    freq, power = data
    frequencies = freq[..., 0, 0]
    fmin, fmax = (frequencies.min(), frequencies.max())

    for expected, actual in zip(expected_frequencies, (fmin, fmax)):
        assert expected == actual
