import numpy as np
from scipy import signal


def _merge(vals1, vals2):
    """Helper function for merging two lists. """
    vals = [(i, j) for i, j in zip(vals1, vals2)]
    return np.asarray(vals).reshape(1, len(vals) * 2).tolist()[0]


def find_events(areas, *args, **kwargs):
    """
    Find cardiac events in a signal.

    Parameters
    ----------
    areas : list or ndarray
        Sequence of values.
    *args : tuple or list
        Arguments corresponding to additional parameters to be supplied to
        ``scipy.signal.find_peaks``.
    **kwargs : dict
        Keyword arguments corresponding to additional parameters to be 
        supplied to ``scipy.signal.find_peaks``.

    Returns
    -------
    all : tuple
        Both peaks and troughs identified through ``scipy.signal.find_peaks``. 
        With indices taking up the first slot and the associated values taking up
        the second slot. The same applies to both peaks and troughs below.
    peaks : tuple
        Peaks identified through ``scipy.signal.find_peaks``. 
    troughs : tuple
        Troughs identified through ``scipy.signal.find_peaks``. 

    Notes
    -----
    Note that this method is built around scipy's find_peaks, and so any
    keyword arguments that you would supply to that function can be supplied
    here as well.

    """

    arr = np.asarray(areas)
    arr_inv = np.max(arr) - arr

    peaks, _ = signal.find_peaks(arr, *args, **kwargs)
    troughs, _ = signal.find_peaks(arr_inv, *args, **kwargs)

    # Specific events
    peaks_ = (peaks, arr[peaks])
    troughs_ = (troughs, arr[troughs])

    # Global trend
    indices = _merge(peaks, troughs)
    vals_ = _merge(peaks_[1], troughs_[1])
    all_ = (indices, vals_)

    return (all_, peaks_, troughs_)
