import cv2
import numpy as np
from more_itertools import pairwise

from heartcv.util import HeartCVError 
from heartcv import util 
from heartcv.util import hcv_logger

# Built-in embryo location methods -------------------------------------------------
'''
    These methods are apart of the semi-automated API to be used with HeartCV.location_gui(...).
    They can be useful in creating a more precise embryo mask to segment to if needed.    

'''

def default(img):
    '''
    Simplest embryo location method. 
    
    The method applies an OTSU threshold, median filter and finally retrieves
    the largest contour by area. 

    '''
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = util.find_contours(median)

    return contours, hierarchy

def two_stage(img):
    '''
    Default method but with two stages of contour detection. 

    Both stages of contour filtering involve subsetting to the largest 
    contour by area. The image is masked to the bounding box of the largest 
    contour detected first, and then the same default method is applied to the
    masked image.

    '''
    firstContour = default(img)

    bbox = cv2.boundingRect(firstContour)
    firstRectMask = util.rect_mask(img, bbox)
    onlyFirst = cv2.bitwise_and(img, img, mask=firstRectMask)

    return default(img)

def binary_thresh(img, thresh):
    '''
    Default method but with a variable binary threshold instead of an
    OTSU threshold.

    '''
    _, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = util.find_contours(median)

    return contours, hierarchy

class Location:
    '''
    Location wrapper class. 

    This allows the application of any number of image pre-processing methods
    and subsequent contour filters to extract a suitable embryo outline from 
    an image.

    '''
    def __init__(self, preprocess, contour_filter):
        '''
        Keyword arguments:
            preprocess       Callable.    Pre-processing method required for producing 
                                          image contours.

            contour_filter   Callable.    Contour filtering method for applying to contours
                                          produced from the pre-processing method.

        '''
        self.preprocess = preprocess
        self.contour_filter = contour_filter

    def __call__(self, img, *args, **kwargs):
        '''
        Callable for processing an image and producing an embryo outline. 

        Keyword arguments:
            img    Numpy.ndarray.    Image to process.

        Returns:
            Numpy.ndarray or List.    Contour(s) outlining embryo.

        '''
        contours, hierarchy = self.preprocess(img, *args, **kwargs)
        if contours is None:
            raise HeartCVError(f'{self.preprocess} did not find contours.')

        try:
            filtContours = self.contour_filter(contours, hierarchy)
        except TypeError:
            filtContours = self.contour_filter(contours)
        except:
            raise
        return filtContours

# Heart location methods ----------------------------------------------------------------------
'''
    These are methods used in both semi- and fully automated API's for locating the heart. They are
    relatively simple in that they use consecutive frame subtraction to locate the activity of the
    heart. The output of these methods can be used both with HeartCV.activity_gui(...) and 
    HeartCV.roi_search(...) to locate the heart.

'''

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

def _roi_filter(diff_img, thresh_val, gauss_kernel):
    '''Find an roi given a sum difference image. '''
    _, thresh = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (gauss_kernel,gauss_kernel), 0, 0)

    contours, hierarchy = util.find_contours(blur)

    bbox = None
    if contours:
        contour = util.largest(contours)
        bbox =  cv2.boundingRect(contour)

    return bbox

def roi_search(diff_img, thresh_range, gauss_range):
    '''
    Find an roi for the heart via grid search across pre-defined binary
    threshold and gaussian blur ranges.

    Keyword arguments:
        diff_img        Numpy ndarray.    Grayscale image produced from HeartCV.sum_abs_diff() (Required).

        thresh_range    Tuple.            Binary threshold values to conduct roi search over (Required).

        gauss_range     Tuple.            Gaussian kernel sizes to conduct roi search over (Required).

    Returns:
        Tuple.    Bounding box dimensions of the final roi.

        List.     List of all bounding box dimensions found.

    '''
    hcv_logger.info('Computing the heart roi...')

    bboxs = []
    for t in range(*thresh_range):
        for g in range(*gauss_range):
                bbox = _roi_filter(diff_img, t, g)
                if bbox:
                    bboxs.append(bbox)

    x,y,w,h = map(np.median, zip(*bboxs))

    return (tuple(map(int, (x,y,w,h))), bboxs)