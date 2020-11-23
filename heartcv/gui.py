import functools
import cv2
import numpy as np
from typing import Callable
from dataclasses import dataclass
import cvu

from heartcv import location 

# Implementations for the semi-automated API ------------------------------------

def location_gui(video, method, size=None):
    '''
    Launch a gui for testing an embryo location method.

    Keyword arguments:
        video    HeartCV.Video.   HeartCV.Video object to the video to test (Required).

        method   Callable.        Callable to execute on each frame to
                                  locate the embryo (Required).

        size     Tuple.           Size of GUI window. Default is the size of
                                  the first image in the video.

    Returns:
        Numpy ndarray.   Embryo outline determined by the supplied method.

        Dict.            Dict of trackbar names and their current values on exit.               

    '''
    gui = cvu.VideoGUI(video=video, title='Embryo location GUI')

    @gui.process
    def locate(gui):
        frame = gui.frame.copy()
        frame_proc = cvu.gray(frame)

        if method.preprocess is location.binary_thresh:
            gui.embryo_outline = method(frame_proc, gui['thresh'])
        else:
            gui.embryo_outline = method(frame_proc)

        cvu.draw_contours(frame, gui.embryo_outline, -1, (0,255,0), 1)
        return frame

    if method.preprocess is location.binary_thresh:
        @gui.trackbar('Threshold', id='thresh', min=0, max=255)
        def on_thresh(gui, val):
            gui['thresh'] = val
            img = gui.process()
            cv2.imshow(gui.title, img)

    gui.run()

    return gui.embryo_outline, gui.values()

def activity_gui(frame, diff_img, size=None):
    '''
    Launch a gui for finding a bounding box from the output of the activity location methods.

    Keyword arguments:
        frame     Numpy ndarray.    RGB image from footage to be processed (Required).

        diff_img  Numpy ndarray.    Grayscale image produced from HeartCV.sum_abs_diff() (Required).
    
        size      Tuple.            Size of the interactive window in pixels, default is the
                                    shape of the image supplied.
    
    Returns:
        Tuple.    Bounding box dimensions for the heart (x,y,w,h).

        Tuple.    Binary threshold and gaussian kernel values (thresh, gauss).

    '''
    gui = cvu.FrameGUI(frame=(frame, diff_img), title='ROI viewer')

    @gui.process
    def find(gui):
        frame, diff = gui.frame
        frame_proc, diff_proc = (frame.copy(), diff.copy())
        if len(frame_proc.shape) == 2:
            frame_proc = cvu.bgr(frame)

        _thresh = gui['thresh']
        _gauss = gui['gauss']
        if _gauss%2 == 0:
            _gauss = _gauss + 1

        bbox, blur = location._roi_filter(diff_proc, _thresh, _gauss)
        if bbox:
            gui.bbox = bbox
            cvu.draw_rectangles(frame_proc, bbox, (0,255,0), 1)

        diff_proc = cvu.bgr(blur)
        allImg = np.hstack((frame_proc, diff_proc))

        return allImg

    @gui.trackbar('Threshold', id='thresh', min=0, max=255)
    def on_thresh(gui, val):
        gui['thresh'] = val
        frame = gui.process()
        cv2.imshow(gui.title, frame) 

    @gui.trackbar('Gaussian', id='gauss', min=1, max=101)
    def on_gauss(gui, val):
        gui['gauss'] = val
        frame = gui.process()
        cv2.imshow(gui.title, frame) 

    gui.run()

    return gui.bbox, gui.values()
