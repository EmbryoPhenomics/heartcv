import cv2
import numpy as np
from heartcv.gui import base
from heartcv.core.location import binary_thresh
from heartcv import util

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
    if not size:
        first = video.read(0)
        size = np.swapaxes(first,0,1).shape # required to get correct dims for cv2.resize()

    gui = base.VideoGUI(video=video, title='Embryo location GUI', size=size)

    @gui.main_process
    def locate(gui):
        frame = gui.frame.copy()
        frameProc = util.gray(frame)

        if method.preprocess is binary_thresh:
            gui.embryoOutline = method(frameProc, gui['thresh'])
        else:
            gui.embryoOutline = method(frameProc)

        util.draw_contours(frame, gui.embryoOutline, -1, (0,255,0), 1)
        return frame

    if method.preprocess is binary_thresh:
        @gui.trackbar('Threshold', id='thresh', min=0, max=255)
        def on_thresh(gui, val):
            gui['thresh'] = val
            img = gui.process()
            cv2.imshow(gui.title, img)

    gui.run()

    return gui.embryoOutline, gui.values()

def activity_gui(frame, diffImg, size=None):
    '''
    Launch a gui for testing the final bounding box used for subsequent methods.

    Keyword arguments:
        frame     Numpy ndarray.    RGB image from footage to be processed (Required).

        diffImg   Numpy ndarray.    Grayscale image produced from absDiff() (Required).
    
        size      Tuple.            Size of the interactive window in pixels, default is the
                                    shape of the image supplied.
    
    Returns:
        Tuple.    Bounding box dimensions for the heart (x,y,w,h).

        Tuple.    Binary threshold and gaussian kernel values (thresh, gauss).

    '''
    if not size:
        size = (diffImg.shape[1]*2, diffImg.shape[0])

    gui = base.FrameGUI(frame=(frame, diffImg), title='Localisation GUI', size=size)

    @gui.main_process
    def find(gui):
        frame, diff = gui.frame
        frameProc, diffProc = (frame.copy(), diff.copy())
        if len(frameProc.shape) == 2:
            frameProc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        _thresh = gui['thresh']
        _gauss = gui['gauss']
        if _gauss%2 == 0:
            _gauss = _gauss + 1

        gaussKernel = (_gauss, _gauss)

        _, thresh = cv2.threshold(diffProc, _thresh, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh, gaussKernel, 0, 0)

        contours, hierarchy = util.find_contours(blur)

        if contours:
            contour = util.largest(contours)
            x,y,w,h = cv2.boundingRect(contour)
            gui.bbox = (x,y,w,h)

            cv2.rectangle(frameProc, (x,y), (x+w, y+h), (0,255,0), 1)

        diffProc = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        allImg = np.hstack((frameProc, diffProc))

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
