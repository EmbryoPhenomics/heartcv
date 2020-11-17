import cv2
import numpy as np

def _contours_area(contours):
    '''Convenience function for computing the area of contour(s). '''
    if isinstance(contours, np.ndarray):     
        areas = cv2.contourArea(contours)
    else: 
        areas = [cv2.contourArea(c) for c in contours]
    return areas

def smallest(contours):
    '''Contour filter that returns the smallest contour by area. '''
    areas = _contours_area(contours)
    return contours[np.argmin(areas)]

def largest(contours):
    '''Contour filter that returns the largest contour by area. '''
    areas = _contours_area(contours)
    return contours[np.argmax(areas)]

def parents(contours, hierarchy):
    '''Contour filter that returns only parent contours. '''
    parentLvl = hierarchy[0,:,3].tolist()
    return [contours[i] for i,e in enumerate(parentLvl) if e == -1]

class _ByLimit:
    '''Filter values based on pre-defined limits. '''
    def __init__(self, min=None, max=None):
        '''
        Keyword arguments: 
            min    Int or float.   Lower limit to filter values.
            max    Int or float.   Upper limit to filter values.

        '''
        if not min and not max:
            raise ValueError('Limit(s) required for filtering contours by area.')

        self.min = min
        self.max = max

    def __call__(self, x):
        '''
        Filter x based on  a lower or upper limit, or both. If x is between either
        of the pre-defined limits then it will be returned.

        '''
        min, max = (self.min, self.max)
        if min and max:
            if x >= min and x <= max:
                return x
        else:
            if min and not max:
                if x >= min:
                    return x
            elif max and not min:
                if x <= max:
                    return x    

class Area:
    '''Filter contours by area based on a lower or upper limit, or both. '''
    def __init__(self, min=None, max=None):
        '''
        Keyword arguments: 
            min    Int or float.   Lower limit to filter contour areas.
            max    Int or float.   Upper limit to filter contour areas.

        '''
        self._filter = _ByLimit(min, max)

    def __call__(self, contours):
        '''
        Filter contours by area.

        Note that contour areas are computed within this method.

        Keyword arguments:
            contours    List.   List of contours to filter (Required).

        Returns:
            List.    Filtered contours.

        '''
        areas = _contours_area(contours)
        _filtered = []

        for i,a in enumerate(areas):
            if self._filter(a):
                _filtered.append(contours[i])

        return _filtered

class Eccentricity:
    '''Filter contours by eccentricity based on a lower or upper limit, or both. '''
    def __init__(self, min=None, max=None):
        '''
        Keyword arguments: 
            min    Int or float.   Lower limit to filter contour areas.
            max    Int or float.   Upper limit to filter contour areas.

        '''
        self._filter = _ByLimit(min, max)

    def _eccentricity(self, contours):
        '''Convenience method for computing the eccentricity of contours. '''
        for i,c in enumerate(contours):
            if c.shape[0] >= 5: # filter to this size for below calc
                center,axes,orientation = cv2.fitEllipse(c)
                major, minor = (max(axes), min(axes))
                eccentricity = (np.sqrt(1-(minor/major)**2))
                yield (i, eccentricity) 

    def __call__(self, contours):
        '''
        Filter contours by eccentricity.

        Note that the eccentricity of contours are computed within this method.

        Keyword arguments:
            contours    List.   List of contours to filter (Required).

        Returns:
            List.    Filtered contours.

        '''
        _filtered = []
        for i,e in self._eccentricity(contours):
            if self._filter(e):
                _filtered.append(contours[i])

        return _filtered

class Solidity:
    '''Filter contours by solidity based on a lower or upper limit, or both. '''
    def __init__(self, min=None, max=None):
        '''
        Keyword arguments: 
            min    Int or float.   Lower limit to filter contour areas.
            max    Int or float.   Upper limit to filter contour areas.

        '''
        self._filter = _ByLimit(min, max)

    def _solidity(self, contours):
        '''Convenience method for computing the eccentricity of contours. '''
        for i,c in enumerate(contours):
            c_area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hullarea = cv2.contourArea(hull)
            
            try:
                yield (i, c_area/hullarea)
            except ZeroDivisionError:
                pass

    def __call__(self, contours):
        '''
        Filter contours by solidity.

        Note that the solidity of contours are computed within this method. Also 
        note that here the solidity of a contour is based on its area relative
        to its corresponding convex hull.

        Keyword arguments:
            contours    List.   List of contours to filter (Required).

        Returns:
            List.    Filtered contours.

        '''
        _filtered = []
        for i,s in self._solidity(contours):
            if self._filter(s):
                _filtered.append(contours[i])

        return _filtered

