import cv2
import time
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import base64
from dataclasses import dataclass
import numpy as np
import os

import vuba

# Text defaults for opencv
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2


@dataclass
class DataStore:
    """Backend container that holds data from various ui. """

    __slots__ = ["ds_frame", "ss_frame"]
    ds_frame: list
    ss_frame: list


class VideoStore:
    """Backend container for holding video capture instances. """

    def __init__(self):
        self.raw_video = None
        self.current_frame = None

    def initiate(self, raw_video):
        (self.raw_video, self.raw_video_release) = vuba.open_video(raw_video)

    def close(self):
        if self.raw_video_release:
            self.raw_video.close()


if os.name == "posix":
    class Keys:
        """Key codes for keyboard keys (as tested on linux). """

        right = 83
        left = 81
        d = 100
        s = 115
        space = 32
        esc = 27
elif os.name == "nt":
    class Keys:
        """Key codes for keyboard keys (as tested on Windows). """

        right = 46
        left = 44
        d = 100
        s = 115
        space = 32
        esc = 27
else:
    raise OSError(
        "Operating system not currently supported, currently only Windows 10 and Linux are supported."
    )


def frame_recorder(filename, queue):
    """
    GUI for labelling frames that correspond to various stages in the
    cardiac cycle.

    Note that the keys bound here may have different codes between systems,
    so this might have to be changed on a system by system basis.

    Parameters
    ----------
    filename : str
        Filename to process.
    queue : multiprocessing.Queue
        Queue for appending with data from frame recorder.

    """

    video = vuba.Video(filename)
    frame_count = len(video)
    fps = video.fps
    window_title = "Frame recorder"
    frames = list(range(frame_count))

    def display(val):
        frame = video.read(index=val)
        cv2.putText(
            frame,
            "Frame: {}".format(val),
            tuple(map(int, (0.025 * frame.shape[1], 0.975 * frame.shape[0]))),
            font,
            fontScale,
            fontColor,
            lineType,
        )

        cv2.imshow(window_title, frame)

    def stream(val):
        _current = val
        for f in range(val, frame_count):
            key = cv2.waitKey(1)

            if key is Keys.space:
                break
            else:
                display(f)
                _current = f
                time.sleep(1 / fps)

        return _current

    current = 0
    diastoleFrameRecord = []
    systoleFrameRecord = []

    display(current)
    while True:
        key = cv2.waitKey(1)

        val = 0
        if key is Keys.right:
            if current < (frame_count - 1):
                val = 1
        elif key is Keys.left:
            if current > 0:
                val = -1
        elif key is Keys.d:
            diastoleFrameRecord.append(current)
        elif key is Keys.s:
            systoleFrameRecord.append(current)
        elif key is Keys.space:
            current = stream(current)
        elif key is Keys.esc:
            break

        if val != 0:
            current += val
            display(current)
        else:
            pass

        time.sleep(0.01)

    video.close()
    queue.put((diastoleFrameRecord, systoleFrameRecord))
