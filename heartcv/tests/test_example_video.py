import pytest
import os
import glob

import heartcv as hcv


def test_example_video():
    expected_parameters = dict(resolution=(600, 600), fps=30, codec="MJPG")
    expected_length = 600

    video = hcv.load_example_video()

    for key in expected_parameters.keys():
        assert expected_parameters[key] == getattr(video, key)

    assert expected_length == len(video)

    for file in glob.glob("./*.avi"):
        os.remove(file)
