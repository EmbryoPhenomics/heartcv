from contextlib import contextmanager
import logging
from tqdm import tqdm
import numpy as np

# Stats operations -----------------------------------------------------------------


def minmax_scale(vals):
    """
    Convenience function for performing min/max normalization. 

    Parameters
    ----------
    vals : list or ndarray
        Sequence of values to perform normalization on.

    Returns
    -------
    vals : ndarray
        Sequence of normalized values.

    See Also
    --------
    mse
    rmse

    """
    vals = np.asarray(vals)
    m = np.nanmin(vals)
    M = np.nanmax(vals)
    return (vals - m) / (M - m)


def mse(truth, act):
    """
    Compute the mean square error between truth and actual values. 

    Parameters
    ----------
    truth : list or ndarray
        Sequence of truth values.
    act : list or ndarray
        Sequence of actual values.

    Returns
    -------
    ret : int or float
        Mean square error of the values of provided.

    See Also
    --------
    minmax_scale
    rmse

    """
    truth, act = map(np.asarray, (truth, act))
    truth, act = map(scale, (truth, act))
    diff = truth - act
    return np.nanmean(diff ** 2)


def rmse(truth, act):
    """
    Compute the root mean square error between truth and actual values. 

    Parameters
    ----------
    truth : list or ndarray
        Sequence of truth values.
    act : list or ndarray
        Sequence of actual values.

    Returns
    -------
    ret : int or float
        Root mean square error of the values of provided.

    See Also
    --------
    minmax_scale
    mse

    """
    return np.sqrt(mse(truth, act))


# Logging utilities -----------------------------------------------------------------


class HeartCVError(Exception):
    """ Raise for HeartCV specific runtime errors"""


class HeartCVWarning(Warning):
    """ Raise for HeartCV specific warnings"""


SHOW_PROGRESS = True

if SHOW_PROGRESS:
    logging.basicConfig(level="INFO")
    hcv_logger = logging.getLogger("HeartCV")
    hcv_logger.setLevel("INFO")
else:
    logging.basicConfig(level="WARNING")
    hcv_logger = logging.getLogger("HeartCV")
    hcv_logger.setLevel("WARNING")


def show_progress(on):
    """
    Show progress information for functions in HeartCV.

    Parameters
    ----------
    on : bool
        Whether to enable package wide progress information.

    See Also
    --------
    pgbar

    Notes
    -----
    This will only print progress for functions which have long running
    computation, much of the convenience functions will not print any
    progress.

    """
    global SHOW_PROGRESS
    SHOW_PROGRESS = on


class EmptyPGbar:
    """Dummy progress bar for progress bar context."""

    @staticmethod
    def update(n):
        """Dummy update for progress bar context."""
        pass


@contextmanager
def pgbar(total):
    """
    Create a progress bar to be used in a 'with' context.

    Parameters
    ----------
    total : int
        Length of progress bar.

    Returns
    -------
    pgbar : tqdm.tqdm
        Progress bar.

    See Also
    --------
    show_progress

    Notes 
    -----
    This is displayed depending on the global variable SHOW_PROGRESS, 
    changed for HeartCV.show_progress.

    """

    if SHOW_PROGRESS:
        _pgbar = tqdm(total=total)
        try:
            yield _pgbar
        finally:
            _pgbar.close()
    else:
        yield EmptyPGbar
