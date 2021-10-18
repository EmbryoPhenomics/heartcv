import pytest
import numpy as np

import heartcv as hcv

@pytest.fixture
def data():
	return np.ones((100, 600, 600), dtype=np.uint8)


@pytest.fixture
def expected_shapes():
	# Expected output for different binning factors for test video
	return {4 : (100, 150, 150), # x4
			6 : (100, 100, 100), # x6
			8 : (100, 75, 75),   # x8
			16 : (100, 37, 37)}  # x16


@pytest.mark.parametrize("binsize", [4, 6, 8, 16])
def test_mpx_grid(data, expected_shapes, binsize) -> None:
	out = hcv.mpx_grid(data, binsize=binsize)
	
	assert expected_shapes[binsize] == out.shape

