import pytest
import numpy as np

import heartcv as hcv


@pytest.fixture
def sample():
    data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return np.asarray(data, dtype=np.uint8)


@pytest.fixture
def segmented():
    data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    return np.asarray(data, dtype=np.uint8)


@pytest.fixture
def expected_contour():
    data = [
        [1, 4],
        [1, 5],
        [1, 6],
        [2, 6],
        [3, 6],
        [4, 6],
        [4, 5],
        [4, 4],
        [3, 4],
        [2, 4],
    ]
    return np.asarray(data)


@pytest.fixture
def sample_sequence(sample):
    return np.asarray([sample for i in range(10)])


@pytest.fixture
def segmented_sequence(segmented):
    return np.asarray([segmented for i in range(10)])


def test_contour_detection(expected_contour, sample):
    assert expected_contour.all() == hcv.detect_largest(sample).all()


@pytest.mark.parametrize("invert", [True, False])
def test_segmentation(sample_sequence, segmented_sequence, invert):
    if invert:
        np.invert(segmented_sequence)

    roi = hcv.detect_largest(sample_sequence[0])
    actual_sequence = np.asarray(
        [out for out in hcv.segment(sample_sequence, roi, invert)]
    )

    assert segmented_sequence.all() == actual_sequence.all()
