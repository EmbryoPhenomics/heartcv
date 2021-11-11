import pytest
import cv2
import os
import glob

import heartcv as hcv


@pytest.fixture
def video():
    yield hcv.load_example_video()

    for file in glob.glob("./*.avi"):
        os.remove(file)


@pytest.fixture
def gui(video):
    frames = video.read(0, len(video), grayscale=True)
    mpx = hcv.mpx_grid(frames, binsize=16)
    ept = hcv.epts(mpx, fs=30)  # example fps
    gui = hcv.identify_frequencies(video, ept, run=False)
    return gui


def test_gui_components(gui, video):
    assert "EPT GUI" == gui.title

    assert len(gui.trackbars) == 2

    assert gui.trackbars['frames'].min == 0
    assert gui.trackbars['frames'].max == len(video)
    assert gui.trackbars['frames'].current == 0

    assert gui.trackbars['freq_ind'].min == 0
    assert gui.trackbars['freq_ind'].max == (len(video)/2) + 1
    assert gui.trackbars['freq_ind'].current == 0

    cv2.destroyAllWindows() # cleanup
