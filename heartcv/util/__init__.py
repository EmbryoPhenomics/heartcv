''' 
Sub-package with general utilities for use both in the internal and external API's.

'''

__all__ = [
    'show_progress',
    'pgbar',
    'take_first',
    'open_video',
    'Video',
    'Writer',
    'find_contours',
    'gray',
    'Mask',
    'square',
    'shrink',
    'rect_mask',
    'circle_mask',
    'contour_mask',
    'smallest',
    'largest',
    'parents',
    'Area',
    'Eccentricity',
    'Solidity'
]

# Custom exceptions included here as they don't warrant a separate module
class HeartCVError(Exception):
    ''' Raise for HeartCV specific runtime errors'''

class HeartCVWarning(Warning):
    ''' Raise for HeartCV specific warnings'''

from ._logging import *
from .imgio import *
from .imgops import *
from .contourfilters import *