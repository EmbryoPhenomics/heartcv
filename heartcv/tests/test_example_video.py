import pytest
import os
import glob

import heartcv as hcv


@pytest.fixture
def expected_result():
    expected_parameters = dict(resolution=(600, 600), fps=30, codec="MJPG")
    expected_length = 600

    return expected_length, expected_parameters


@pytest.fixture
def video():
    yield hcv.load_example_video()

    for file in glob.glob("./*.avi"):
        os.remove(file)


def test_example_video(expected_result, video):
    expected_length, expected_parameters = expected_result

    for key in expected_parameters.keys():
        assert expected_parameters[key] == getattr(video, key)

    assert expected_length == len(video)
