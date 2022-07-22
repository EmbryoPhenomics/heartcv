import pytest
import numpy as np
import vuba

import heartcv as hcv


@pytest.fixture
def data():
    arr = np.random.randint(0, 255, size=(300, 600, 600), dtype=np.uint8)
    ept = hcv.epts(arr, fs=30, binsize=16)  # example fps
    return ept


@pytest.fixture
def expected_shape():
    return (38, 38)


@pytest.mark.parametrize("frequencies", [4, (1, 4), (None, 4), (1, None), [1, 2, 3, 4]])
def test_shape(data, expected_shape, frequencies):
    np.seterr(
        divide="ignore", invalid="ignore"
    )  # Ignore true-divide warnings in normalization of heatmap
    out = hcv.spectral_map(data, frequencies)
    assert out.shape == expected_shape
