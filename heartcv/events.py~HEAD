import numpy as np
from scipy import signal


def _merge(vals1, vals2):
    """Helper function for merging two lists. """
    vals = [(i, j) for i, j in zip(vals1, vals2)]
    return np.asarray(vals).reshape(1, len(vals) * 2).tolist()[0]


def find_events(areas, *args, **kwargs):
    """
    Find cardiac events in a signal outputted from heartcv.heartArea().

    Note that this method is built around scipy's find_peaks, and so any
    keyword arguments that you would supply to that function can be supplied
    here as well.

    Keyword arguments:
        areas     List.     Sequence of area values computed using heartcv (Required).

        *args               Any number of arguments that you would
        **kwargs            ordinarily supply to find_peaks().

    Returns:
        Tuple.    End diastole, systole and both supplied as lists of indices with
                  their corresponding values in the following structure:
                      - total        = indices, areas
                      - end diastole =      ''
                      - end systole  =      ''
    """

    arr = np.asarray(areas)
    arr_inv = np.max(arr) - arr

    peaks, _ = signal.find_peaks(arr, *args, **kwargs)
    troughs, _ = signal.find_peaks(arr_inv, *args, **kwargs)

    # Specific events
    endDs = (peaks, arr[peaks])
    endSs = (troughs, arr[troughs])

    # Global trend
    indices = _merge(peaks, troughs)
    _areas = _merge(endDs[1], endSs[1])
    total = (indices, _areas)

    return (total, endDs, endSs)
