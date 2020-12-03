import cv2
import numpy as np
from more_itertools import pairwise
import cvu

from heartcv.util import HeartCVError 
from heartcv import util 
from heartcv.util import hcv_logger

def default(img):
    '''
    Simplest embryo location method. 
    
    The method applies an OTSU threshold, median filter and finally retrieves
    the largest contour by area. 

    '''
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = cvu.find_contours(medi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
    firstRectMask = cvu.rect_mask(img, bbox)
    onlyFirst = cv2.bitwise_and(img, img, mask=firstRectMask)

    return default(img)

def binary_thresh(img, thresh):
    '''
    Default method but with a variable binary threshold instead of an
    OTSU threshold.

    '''
    _, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = cvu.find_contours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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


def _abs_diffs(frames, mask):
    '''
    Function for computing the absolute differences between consecutive frames,
    used in both individual and sum frame differencing below.

    '''
    first = cvu.take_first(frames)
    if mask is None:
        mask = np.ones(first.shape, dtype='uint8')
    mask_ = cvu.Mask(mask)

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

    sum_diff = np.zeros_like(cvu.take_first(frames))
    with util.pgbar(total=len(frames)-1) as pgbar:
        for diff in _abs_diffs(frames, mask):
            _, thresh = cv2.threshold(diff, thresh_val, 1, cv2.THRESH_BINARY)
            sum_diff += thresh
            pgbar.update(1)   

    return sum_diff

def _roi_filter(diff_img, thresh_val, gauss_kernel, rotate):
    '''Find an roi given a sum difference image. '''
    _, thresh = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (gauss_kernel,gauss_kernel), 0, 0)

    contours, hierarchy = cvu.find_contours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bbox = None
    if contours:
        contour = cvu.largest(contours)

        if rotate:
            bbox = cv2.minAreaRect(contour)
        else:
            bbox = cv2.boundingRect(contour)

    return (bbox, contour, blur)

def roi_search(diff_img, thresh_range, gauss_range, rotate=True):
    '''
    Find an roi for the heart via grid search across pre-defined binary
    threshold and gaussian blur ranges.

    Keyword arguments:
        diff_img        Numpy ndarray.    Grayscale image produced from HeartCV.sum_abs_diff() (Required).

        thresh_range    Tuple.            Binary threshold values to conduct roi search over (Required).

        gauss_range     Tuple.            Gaussian kernel sizes to conduct roi search over (Required).

        rotate          Bool.             Whether to rotate bounding boxes, default is True.

    Returns:
        Tuple.    Bounding box dimensions of the median roi.

        List.     List of all bounding box dimensions found.

    '''
    hcv_logger.info('Computing the heart roi...')

    bboxs = []
    for t in range(*thresh_range):
        for g in range(*gauss_range):
                bbox, _, _ = _roi_filter(diff_img, t, g, rotate=rotate)

                if rotate:
                    (x,y),(w,h),a = bbox
                    bbox = (x,w,y,h,a)

                if bbox:
                    bboxs.append(bbox)

    bbox = tuple(map(np.median, zip(*bboxs)))

    return (tuple(map(int, bbox)), bboxs)