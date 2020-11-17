from contextlib import contextmanager
import logging
from tqdm import tqdm
import cv2
import numpy as np
import cvu

# Signal operations -----------------------------------------------------------------

def minmax_scale(vals):
    '''Convenience function for performing min/max normalization. '''
    vals = np.asarray(vals)
    m = np.nanmin(vals)
    M = np.nanmax(vals)
    return (vals-m)/(M-m)

def mse(truth, act):
    '''Compute the mean square error between truth and actual values. '''
    truth, act = map(np.asarray, (truth, act))
    truth, act = map(scale, (truth, act))
    diff = truth - act
    return np.nanmean(diff**2)

def rmse(truth, act):
    '''Compute the root mean square error between truth and actual values. '''
    return np.sqrt(mse(truth, act))

# Image operations ------------------------------------------------------------------

def subset(frames, x, y, w, h):
    '''Subset a sequence of frames to a particular roi. '''
    for frame in frames:
        yield frame[y:y+h, x:x+w, ...]

# Logging utilities -----------------------------------------------------------------

class HeartCVError(Exception):
    ''' Raise for HeartCV specific runtime errors'''

class HeartCVWarning(Warning):
    ''' Raise for HeartCV specific warnings'''

SHOW_PROGRESS = True

if SHOW_PROGRESS:
    logging.basicConfig(level='INFO')
    hcv_logger = logging.getLogger('HeartCV')
    hcv_logger.setLevel('INFO')
else:
    logging.basicConfig(level='WARNING')
    hcv_logger = logging.getLogger('HeartCV')
    hcv_logger.setLevel('WARNING') 

def show_progress(on):
    '''
    Show progress information for functions in HeartCV.

    This will only print progress for functions which have long running
    computation, much of the convenience functions will not print any 
    progress. 

    Keyword arguments:
        on    Bool.    Whether to enable package wide progress monitoring (Required). 

    '''
    global SHOW_PROGRESS
    SHOW_PROGRESS = on

class EmptyPGbar:
    '''Dummy progress bar for progress bar context.'''
    @staticmethod
    def update(n):
        '''Dummy update for progress bar context.'''
        pass

@contextmanager
def pgbar(total):
    '''
    Create a progress bar to be used in a 'with' context.

    Note that this is displayed depending on the global 
    variable SHOW_PROGRESS. For VideoCapture objects with infinite 
    feeds, such as camera streams, a progress bar will not 
    be produced.

    Keyword arguments:
        total    Integer.    Length of progress bar (Required).

    Returns:
        tqdm progress bar.

    '''

    if SHOW_PROGRESS:
        if total is not None or total != -1:
            _pgbar = tqdm(total=total)
            try:
                yield _pgbar
            finally:
                _pgbar.close()
        else:
            yield EmptyPGbar
    else:
        yield EmptyPGbar
