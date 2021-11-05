import pytest
import numpy as np

import heartcv as hcv


@pytest.fixture
def data():
    peaks = np.asarray([2, 4, 8, 12, 16])
    data_length = 20
    fs = 5
    return peaks, data_length, fs


@pytest.fixture
def expected_bpm(data):
    p, l, fs = data
    return (len(p) / (l / fs)) * 60


@pytest.fixture
def expected_b2b(data):
    _, _, fs = data
    return np.asarray([2, 4, 4, 4]) / fs


@pytest.fixture
def expected_stats(expected_bpm, expected_b2b):
    return dict(
        bpm=[expected_bpm],
        min_b2b=[expected_b2b.min()],
        mean_b2b=[expected_b2b.mean()],
        median_b2b=[np.median(expected_b2b)],
        max_b2b=[expected_b2b.max()],
        sd_b2b=[np.std(expected_b2b)],
        range_b2b=[expected_b2b.max() - expected_b2b.min()],
        rmssd=[np.sqrt(np.nanmean((expected_b2b[1:] - expected_b2b[:-1]) ** 2))]
    )


def test_bpm(expected_bpm, data):
    p, l, fs = data
    assert expected_bpm == hcv.bpm(len(p), l, fs)


def test_b2b_intervals(expected_b2b, data):
    p, l, fs = data
    assert expected_b2b.all() == hcv.b2b_intervals(p, fs).all()


def test_stats(expected_stats, data):
    stats = hcv.stats(*data)
    for key in expected_stats.keys():
        assert expected_stats[key] == stats[key]
