import cv2
import numpy as np
import vuba

from heartcv import location

# Implementations for the semi-automated API ------------------------------------


def location_gui(video, method):
    """
    Launch a gui for testing an embryo location method.
    Keyword arguments:
        video    vuba.Video.       vuba.Video object to the video to test (Required).
        method   Callable.        Callable to execute on each frame to
                                  locate the embryo (Required).
    Returns:
        Numpy ndarray.   Embryo outline determined by the supplied method.
        Dict.            Dict of trackbar names and their current values on exit.
    """
    gui = vuba.VideoGUI(video=video, title="Embryo location GUI")

    @gui.method
    def locate(gui):
        frame = gui.frame.copy()
        frame_proc = vuba.gray(frame)

        if method.preprocess is location.binary_thresh:
            gui.embryo_outline = method(frame_proc, gui["thresh"])
        else:
            gui.embryo_outline = method(frame_proc)

        vuba.draw_contours(frame, gui.embryo_outline, -1, (0, 255, 0), 1)
        return frame

    if method.preprocess is location.binary_thresh:

        @gui.trackbar("Threshold", id="thresh", min=0, max=255)
        def on_thresh(gui, val):
            gui["thresh"] = val
            img = gui.process()
            cv2.imshow(gui.title, img)

    gui.run()

    return gui.embryo_outline, gui.values()


def activity_gui(video, diff_img, rotate=False):
    """
    Launch a gui for finding a bounding box from the output of the activity location methods.
    Keyword arguments:
        video     vuba.Video.        vuba.Video object to the video to test (Required).
        diff_img  Numpy ndarray.    Grayscale image produced from HeartCV.sum_abs_diff() (Required).
        rotate    Bool.             Whether to rotate bounding boxes, default is False.
    Returns:
        Tuple.    Bounding box dimensions for the heart (x,y,w,h).
        Tuple.    Binary threshold and gaussian kernel values (thresh, gauss).
    """
    gui = vuba.VideoGUI(video=video, title="ROI viewer")

    @gui.method
    def find(gui):
        diff_proc = diff_img.copy()
        frame_proc = gui.frame.copy()

        _thresh = gui["thresh"]
        _gauss = gui["gauss"]
        if _gauss % 2 == 0:
            _gauss = _gauss + 1

        bbox, contour, blur = location.roi_filter(
            diff_proc, _thresh, _gauss, rotate=rotate
        )
        if bbox is not None:
            vuba.draw_contours(frame_proc, contour, -1, (0, 0, 255), 1)

            gui.bbox = bbox
            gui.contour = contour
            if not rotate:
                vuba.draw_rectangles(frame_proc, bbox, (0, 255, 0), 1)
            else:
                bbox = np.int0(cv2.boxPoints(bbox))
                vuba.draw_contours(frame_proc, bbox, -1, (0, 255, 0), 1)

        diff_proc = vuba.bgr(blur)
        allImg = np.hstack((frame_proc, diff_proc))

        return allImg

    @gui.trackbar("Threshold", id="thresh", min=0, max=255)
    def on_thresh(gui, val):
        gui["thresh"] = val
        frame = gui.process()
        cv2.imshow(gui.title, frame)

    @gui.trackbar("Gaussian", id="gauss", min=1, max=101)
    def on_gauss(gui, val):
        gui["gauss"] = val
        frame = gui.process()
        cv2.imshow(gui.title, frame)

    gui.run()

    return (gui.bbox, gui.contour, gui.values())
