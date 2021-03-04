import cv2
import numpy as np
from more_itertools import pairwise
import vuba

from heartcv.util import HeartCVError, hcv_logger
from heartcv import util


def default(img):
    """
    Simplest animal location method.

    The method applies an OTSU threshold, median filter and finally retrieves
    the largest contour by area.

    Parameters
    ----------
    img : ndarray
        Grayscale image to perform animal localisation on.

    Returns
    -------
    contours : ndarray
        Array of contour(s) detected.
    hierarchy : ndarray
        Associated hierarchy information for the contours detected.

    See Also
    --------
    two_stage
    binary_thresh
    Location

    """
    vuba._channel_check(img, 2)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = vuba.find_contours(medi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours, hierarchy


def two_stage(img, rotate=False):
    """
    Default method but with two stages of contour detection.

    Both stages of contour filtering involve subsetting to the largest
    contour by area. The image is masked to the bounding box of the largest
    contour detected first, and then the same default method is applied to the
    masked image.

    Parameters
    ----------
    img : ndarray
        Grayscale image to perform animal localisation on.
    rotate : bool
        Whether to fit a rotated bounding box to the largest contour
        detected in the first stage. Default is False.
    
    Returns
    -------
    contours : ndarray
        Array of contour(s) detected.
    hierarchy : ndarray
        Associated hierarchy information for the contours detected.

    See Also
    --------
    two_stage
    binary_thresh
    Location

    """
    vuba._channel_check(img, 2)

    first_contour = default(img)

    if rotate:
        box = cv2.minAreaRect(first_contour)
        box = cv2.boxPoints(box)
        box = np.int0(box)
        first_rect_mask = vuba.contour_mask(img, box)
    else:
        bbox = cv2.boundingRect(first_contour)
        first_rect_mask = vuba.rect_mask(img, bbox)

    only_first = cv2.bitwise_and(img, img, mask=first_rect_mask)

    return default(img)


def binary_thresh(img, thresh):
    """
    Default animal location method but with a variable binary threshold instead of an
    OTSU threshold.

    Parameters
    ----------
    img : ndarray
        Grayscale image to perform animal localisation on.
    thresh : int
        Binary threshold value.

    Returns
    -------
    contours : ndarray or list
        Array of contour(s) detected.
    hierarchy : ndarray or list
        Associated hierarchy information for the contours detected.

    See Also
    --------
    two_stage
    binary_thresh
    Location

    """
    vuba._channel_check(img, 2)

    _, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(thresh, 3)
    contours, hierarchy = vuba.find_contours(
        median, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    return contours, hierarchy


class Location:
    """
    Location wrapper class.

    This allows the application of any number of image pre-processing methods
    and subsequent contour filters to extract a suitable animal outline from
    an image.

    Parameters
    ----------
    preprocess : callable.    
        Pre-processing method required for producing image contours.
    contour_filter : callable.    
        Contour filtering method for applying to contours produced 
        from the pre-processing method.

    Returns
    -------
    handler : Location
        Class object for performing localisation using the supplied methods.

    See Also
    --------
    default
    two_stage
    binary_thresh

    """

    def __init__(self, preprocess, contour_filter):
        self.preprocess = preprocess
        self.contour_filter = contour_filter

    def find(self, img, *args, **kwargs) -> 'Location':
        """
        Callable for processing an image and producing an animal outline.

        Parameters
        ----------
        img : ndarray
            Grayscale image to perform animal localisation on.
        *args : tuple or list
            Arguments corresponding to additional parameters required by the
            supplied callables at initiation.
        **kwargs : dict
            Keyword arguments corresponding to additional parameters required by the
            supplied callables at initiation

        Returns
        -------
        contours : ndarray or list
            Array of contour(s) detected.

        See Also
        --------
        default
        two_stage
        binary_thresh

        """
        vuba._channel_check(img, 2)

        contours, hierarchy = self.preprocess(img, *args, **kwargs)
        if contours is None:
            raise HeartCVError(f"{self.preprocess} did not find contours.")

        try:
            filt_contours = self.contour_filter(contours, hierarchy)
        except TypeError:
            filt_contours = self.contour_filter(contours)
        except:
            raise
        return filt_contours


def abs_diffs(frames, mask=None, thresh_val=10):
    """
    Compute the absolute differences between consecutive frames.

    Parameters
    ----------
    frames : list or ndarray or Frames
        Sequence of grayscale frames to process.
    mask : ndarray
        Optional mask to filter footage to, default is None.
    thresh_val : int
        Binary threshold value to apply to difference images. Adjusting
        this parameter is useful at removing background noise in footage.
        Default is n=10.

    Returns
    -------
    difference_frames : generator
        Generator that will supply difference frames.

    See Also
    --------
    sum_abs_diff

    """
    hcv_logger.info("Computing the absolute differences for footage...")

    first = vuba.take_first(frames)
    if mask is None:
        mask = np.ones(first.shape, dtype="uint8")
    mask_ = vuba.Mask(mask)

    with util.pgbar(total=len(frames) - 1) as pgbar:
        for prev, next_ in pairwise(map(mask_, frames)):
            diff = cv2.absdiff(prev, next_)
            _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            
            yield thresh

            pgbar.update(1)


def sum_abs_diff(frames, mask=None, thresh_val=10):
    """
    Compute the sum of absolute differences between consecutive frames.

    Parameters
    ----------
    frames : list or ndarray or Frames
        Sequence of grayscale frames to process.
    mask : ndarray
        Optional mask to filter footage to, default is None.
    thresh_val : int
        Binary threshold value to apply to difference images. Adjusting
        this parameter is useful at removing background noise in footage.
        Default is n=10.

    Returns
    -------
    sum_diff : ndarray
        Sum difference image, this is the cumulative of all difference frames.    

    See Also
    --------
    abs_diffs

    """
    hcv_logger.info("Computing sum of the absolute differences for footage...")

    sum_diff = np.zeros_like(vuba.take_first(frames))
    with util.pgbar(total=len(frames) - 1) as pgbar:
        for diff in abs_diffs(frames, mask, thresh_val):
            sum_diff += diff
            pgbar.update(1)

    return sum_diff


def roi_filter(diff_img, thresh_val, gauss_kernel, rotate):
    """Find an roi given a sum difference image. """
    vuba._channel_check(diff_img, 2)

    _, thresh = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (gauss_kernel, gauss_kernel), 0, 0)

    contours, hierarchy = vuba.find_contours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bbox = None
    if contours:
        contour = vuba.largest(contours)

        if rotate:
            bbox = cv2.minAreaRect(contour)
        else:
            bbox = cv2.boundingRect(contour)

        return (bbox, contour, blur)


def roi_search(diff_img, thresh_range, gauss_range, area_range, rotate=False):
    """
    Find an roi for the heart via grid search across pre-defined binary
    threshold and gaussian blur ranges.

    Parameters
    ----------
    diff_img : ndarray
        Grayscale image produced from HeartCV.sum_abs_diff().
    thresh_range : tuple
        Binary threshold values to conduct roi search over.
    gauss_range : tuple
        Gaussian kernel sizes to conduct roi search over.
    area_range : tuple 
        Area limits to filter roi's on.
    rotate : bool
        Whether to rotate bounding boxes, default is False.

    Returns
    -------
    bbox : tuple
        Bounding box dimensions of the median roi.
    bboxs : list
        List of all bounding box dimensions found.

    Notes
    -----
    This method will compute the median roi dimensions after grid-search
    across the parameters supplied. Also note that rotated bounding box
    dimensions are not kept in the same format as that produced by minAreaRect, 
    where they would be: ((x,y), (w,h), a) vs here where they are: (x,y,w,h,a).

    See Also
    --------
    roi_filter

    """
    hcv_logger.info("Computing the heart roi...")

    bboxs = []
    for t in range(*thresh_range):
        for g in range(*gauss_range):
            ret = roi_filter(diff_img, t, g, rotate=rotate)
            if ret:
                bbox, _, _ = ret

                if rotate:
                    (x, y), (w, h), a = bbox
                    bbox = (x, w, y, h, a)

                if bbox:
                    bboxs.append(bbox)

    bbox = tuple(map(np.median, zip(*bboxs)))

    return (tuple(map(int, bbox)), bboxs)
