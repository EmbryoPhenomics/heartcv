import cv2
import numpy as np

# Convenience function for various opencv versions
cvVers = int(cv2.__version__[0])
if cvVers == 4:
    def find_contours(img):
        return cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
else:
    def find_contours(img):
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy

def draw_contours(img, contours, *args, **kwargs):
    '''Convenience function for drawing contour(s) on an image. '''
    if isinstance(contours, list):
        for c in contours: cv2.drawContours(img, [c], *args, **kwargs)
    else:
        cv2.drawContours(img, [contours], *args, **kwargs)

def gray(frame):
    ''' 
    Grayscale a frame.

    Keyword arguments:
        frame    Numpy.ndarray.    Frame to grayscale (Required).

    Returns:
        Numpy.ndarray.   Grayscale frame.

    '''
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

class Mask:
    '''
    Convenience class for performing segmentation to a mask.

    '''
    def __init__(self, mask):
        '''
        Keyword arguments:
            mask    Numpy.ndarray.   Mask to segment frame(s) to (Required).

        '''
        self.mask = mask

    def __call__(self, frame):
        ''' 
        Mask a frame using the pre-defined mask.

        Keyword arguments:
            frame    Numpy.ndarray.    Frame to mask (Required).

        Returns:
            Numpy.ndarray.   Masked frame. 

        '''
        return cv2.bitwise_and(frame, frame, mask=self.mask)

def square(bbox):
    ''' 
    Ensure a bounding box is square. 

    Keyword Arguments:
        bbox    Tuple.    Bounding box dimensions to ensure are square (Required).
                          Must be in format: x,y,w,h.

    Returns:
        Bounding box dimensions adjusted to a square shape.

    '''
    def _expand(Min, Max, diff):
        expand = round(diff/2)
        Min = Min - expand
        Max = Max + expand
        return Min, Max

    x,y,w,h = bbox
    minX, maxX, minY, maxY = x, x+w, y, y+h
    diffX = maxX - minX
    diffY = maxY - minY

    if diffX < diffY:
        diff = diffY - diffX
        minX, maxX = _expand(minX, maxX, diff)
    else:
        diff = diffX - diffY
        minY, maxY = _expand(minY, maxY, diff)

    return map(int, (minX, minY, maxX-minX, maxY-minY))

def shrink(img, by=50):
    ''' 
    Mask an image to a new roi.

    Note that not all sides of the image are contracted. This function will 
    only mask the image to a new roi where the mask is reduced in the minY 
    and maxX dimensions.

    Keyword arguments:
        img    Grayscale image.    Grayscale image to reduce the roi for (Required).

        by     Integer.            Number of pixels to reduce roi in minY and maxX
                                   dimensions. Default = 50.

    Returns:
        An image with a reduced roi, by a constant number of pixels.

    '''
    imgMask = np.zeros(img.shape, dtype='uint8')
    cv2.rectangle(imgMask, (0,by), (int(img.shape[1]-by), img.shape[0]), 255, -1)
    img = cv2.bitwise_and(img, img, mask=imgMask)
    return img

def rect_mask(img, bbox):
    ''' 
    Create a rectangular mask at the bounding box dimensions.

    Keyword arguments:
        img     Numpy ndarray.      Grayscale image to produce the rectangular mask
                                    for (Required).

        bbox    Tuple.              Bounding box dimensions to create
                                    rectangular mask at (x,y,w,h) (Required).

    Returns:
        2D numpy ndarray.    Rectangular mask at dimensions of bounding box.

    '''
    x,y,w,h = bbox
    rect_mask_ = np.zeros(img.shape, dtype='uint8')
    cv2.rectangle(rect_mask_, (x,y), (x+w, y+h), 255, -1)
    return rect_mask_

def circle_mask(img, x, y, r):
    ''' 
    Create a circular mask at x,y with radius r.

    Keyword arguments:
        img    Numpy ndarray.      Grayscale image to produce circular mask for (Required).

        x      Integer.            X coordinate (Required).

        y      Integer.            Y coordinate (Required).

        r      Integer.            Radius of circle to draw (Required).

    Returns:
        2D numpy ndarray.    Circular mask at x,y with radius r. 

    '''
    circle_mask_ = np.zeros(img.shape, dtype='uint8')
    cv2.circle(circle_mask_, (x,y), r, 255, -1)
    return circle_mask_

def contour_mask(img, contours):
    ''' 
    Create a mask at the dimensions of the contour.

    Keywords arguments:
        img        Numpy ndarray.    Grayscale image to produce contour mask for (Required).

        contours   List or           Contour(s) to create mask (Required).
                   Numpy ndarray.    

    Returns:
        2D numpy ndarray.    Contour mask at the dimensions of the contour supplied.
    
    '''
    cnt_mask = np.zeros(img.shape, dtype='uint8')
    draw_contours(cnt_mask, contours, -1, 255, -1)
    return cnt_mask

