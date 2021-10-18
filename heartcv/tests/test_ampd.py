import pytest
import numpy as np
from scipy import signal

import heartcv as hcv

# Parametrizing fixtures not currently accepted in pytest so the following
# solution is used
def sinusoidal():
    return np.sin(np.arange(0, 100, 0.1))


def stochastic(v):
    # Incorporate some stochasticity to simulate noise in biological signals
    random = np.random.randint(-1, 1, size=(1000))
    return np.mean((v, v, v, v, random), axis=0)


@pytest.fixture
def expected_shape():
    # Shape should be same regardless of signal type above
    n = len(sinusoidal())
    return (n // 2, n)


@pytest.fixture
def expected_peaks():
    v = sinusoidal()
    p, _ = signal.find_peaks(v)
    return p[1:]  # AMPD excludes first peaks


@pytest.mark.parametrize("data", [sinusoidal(), stochastic(sinusoidal())])
def test_shape(expected_shape, data):
    # No need to linearly detrend signal here as the only shape being tested
    actual_shape = hcv.heartcv._lms(data).shape
    assert expected_shape == actual_shape


@pytest.mark.parametrize("data", [sinusoidal(), stochastic(sinusoidal())])
def test_range(data):
    actual_min = np.min(hcv.heartcv._lms(data))
    actual_max = np.max(hcv.heartcv._lms(data))
    assert actual_min == 0.0 and actual_max <= 2.0


@pytest.mark.parametrize("data", [sinusoidal(), stochastic(sinusoidal())])
def test_peaks(expected_peaks, data):
    actual_peaks = hcv.find_peaks(data)
    assert expected_peaks.all() == actual_peaks.all()


@pytest.mark.parametrize("data", [sinusoidal(), stochastic(sinusoidal())])
def test_minmax_scale(data):
    out = hcv.minmax_scale(data)
    assert out.min() == 0 and out.max() <= 1.0
