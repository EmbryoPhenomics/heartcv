from contextlib import contextmanager
import logging
from tqdm import tqdm
from heartcv.util import imgio

SHOW_PROGRESS = True

# Logging utilities
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
    Show progress information for functions in heartcv.

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