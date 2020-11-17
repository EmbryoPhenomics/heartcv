import cv2
import numpy as np
import functools

from heartcv.util import HeartCVError 
from heartcv import util 

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
