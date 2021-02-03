import cv2
import numpy as np
from more_itertools import pairwise

from heartcv import util
from heartcv.util import hcv_logger

def _abs_diffs(frames, mask):
    '''
    Function for computing the absolute differences between consecutive frames,
    used in both individual and sum frame differencing below.

    '''
    first = util.take_first(frames)
    if mask is None:
        mask = np.ones(first.shape, dtype='uint8')
    mask_ = util.Mask(mask)

    for prev,next_ in pairwise(map(mask_, frames)):
        yield cv2.absdiff(prev, next_)

def abs_diffs(frames, mask=None, thresh_val=10):
    '''
    Compute the absolute differences between consecutive frames.

    Unlike conventional background subtraction, this method computes the absolute
    differences in consecutive frames. This generally results in much better localisation
    of heart activity, since any non-heart noise and changes in animal orientation
    are not registered to the same extent.

    Keyword arguments:
        frames       List.                  List of rgb frames (Required).

        mask         Numpy ndarray.         Mask to filter footage to.

        thresh_val   Integer.               Binary threhsold value to threshold difference images (Required).
                                            Default is None.

    Returns:
        List.    Sequence of difference frames.

    '''
    hcv_logger.info('Computing the absolute differences for footage...')

    diff_frames = []
    with util.pgbar(total=len(frames)-1) as pgbar:
        for diff in _abs_diffs(frames, mask):
            _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            diff_frames.append(thresh)
            pgbar.update(1)

    return diff_frames

def sum_abs_diff(frames, mask=None, thresh_val=10):
    '''
    Compute the sum of absolute differences between consecutive frames.

    Unlike conventional background subtraction, this method computes the absolute
    differences in consecutive frames. This generally results in much better localisation
    of heart activity, since any non-heart noise and changes in animal orientation
    are not registered to the same extent.

    Keyword arguments:
        frames       List.                  List of rgb frames (Required).

        mask         Numpy ndarray.         Mask to filter footage to.

        thresh_val   Integer.               Binary threhsold value to threshold difference images (Required).
                                            Default is None.

    Returns:
        Numpy ndarray.    Sum difference image, this is the cumulative of all consecutive
                          difference frames.

    '''
    hcv_logger.info('Computing sum of the absolute differences for footage...')

    sum_diff = np.zeros_like(util.take_first(frames))
    with util.pgbar(total=len(frames)-1) as pgbar:
        for diff in _abs_diffs(frames, mask):
            _, thresh = cv2.threshold(diff, thresh_val, 1, cv2.THRESH_BINARY)
            sum_diff += thresh
            pgbar.update(1)   

    return sum_diff

def max_optic_flow(frames):
    '''
    Localise the point of most activity using dense optical flow.

    Similar to the method implemented for absolute differences, this function
    also only executes on consecutive frames.  

    Keyword arguments:
        frames       List.    List of rgb frames (Required).

    Returns:
        Tuple.    Coordinates of point with most activity according to dense
                  optical flow.

    '''
    hcv_logger.info('Finding area of most activity (optic flow)...')
    coords = []

    with util.pgbar(len(frames)) as pgbar:
        for prev, next_ in pairwise(frames):
            hsv = np.zeros((*prev.shape, 3), dtype='uint8')
            hsv[...,1] = 255

            flow = cv2.calcOpticalFlowFarneback(prev, next_, None, 0.5, 1, 13, 1, 15, 4, 3)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            h,s,v = cv2.split(hsv)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(v) 

            if maxLoc != (0,0):
                x,y = maxLoc[0], maxLoc[1]
                coords.append((x, y))
            
            pgbar.update(1)

    X,Y = zip(*coords) 
    return tuple(map(np.nanmean, (X,Y)))


