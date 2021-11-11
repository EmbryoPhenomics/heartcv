import pytest
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt

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


@pytest.fixture
def ampd_components():
    return hcv.heartcv._ampd(sinusoidal())


@pytest.fixture
def ampd_plot(ampd_components):
    yield hcv.heartcv._ampd_plot(*ampd_components, show=False)

    plt.close("all")


class TestPlot:
    @pytest.fixture(autouse=True)
    def setup(self, ampd_plot):
        self.fig = ampd_plot

    def text_in_fig(self):
        # Taken from xarray testing suite
        return [t.get_text() for t in plt.gcf().findobj(mpl.text.Text)]

    def test_num_axes(self):
        assert 6 == len(self.fig.axes) # Includes colorbar

    def test_lms_axes(self):
        axes = self.fig.axes[0]

        assert "Local maxima scalogram (LMS)" == axes.get_title() 
        assert "Index" == axes.get_xlabel()
        assert "Scale [No]" == axes.get_ylabel()
        assert "a)" in self.text_in_fig()

        assert axes.has_data()

    def test_rowsums_axes(self):
        axes = self.fig.axes[1]

        assert "Row-wise summation of LMS" == axes.get_title() 
        assert "Scale [No]" == axes.get_xlabel()
        assert "b)" in self.text_in_fig()

        assert axes.has_data()

    def test_G_axes(self):
        axes = self.fig.axes[2]

        assert "Rescaled LMS" == axes.get_title() 
        assert "Index" == axes.get_xlabel()
        assert "Scale [No]" == axes.get_ylabel()
        assert "c)" in self.text_in_fig()

        assert axes.has_data()

    def test_S_axes(self):
        axes = self.fig.axes[3]

        assert "Column-wise standard deviation \n of the rescaled LMS" == axes.get_title() 
        assert "Index" == axes.get_xlabel()
        assert "σ" == axes.get_ylabel()
        assert "d)" in self.text_in_fig()

        assert axes.has_data()

    def test_peaks_axes(self):
        axes = self.fig.axes[4]

        assert "Detected peaks" == axes.get_title() 
        assert "Index" == axes.get_xlabel()
        assert "Amplitude" == axes.get_ylabel()
        assert "e)" in self.text_in_fig()

        assert axes.has_data()


