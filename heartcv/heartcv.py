import vuba
from scipy import signal
from skimage.util import view_as_blocks
import math
from tqdm import tqdm
import cv2
import numpy as np
from more_itertools import pairwise, prepend
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import wget

def load_example_video():
    '''
    Load example video.

    Returns
    -------
    video : vuba.Video
        Video instance of example video

    '''
    path = 'https://zenodo.org/record/4645805/files/20C_E3_10d.avi?download=1'
    fn = wget.download(path)
    return vuba.Video(f'./{fn}')


def mpx_grid(frames, binsize):
    """
    Compute a mean pixel value grid for a sequence of frames at a given binsize.

    This function is a required step in calculating energy proxy traits as per the
    method outlined in Tills et al., 2021.

    Parameters
    ----------
    frames : list or ndarray or vuba.Frames
        Sequence of frames to compute mean pixel values for.
    binsize : int
        Binning resolution to create the mean pixel value grid from.

    Returns
    -------
    mpxs : ndarray
        Array of the mean pixel value grid.

    Notes
    -----

    Here, binsize simply refers to the size of individual bins to compute mean pixel
    values for, e.g. a binsize of 2 would mean that each 2x2 block of pixels in a frame
    will be reduced to a mean.

    See Also
    --------
    epts
    spectral_map

    References
    ----------

    Tills, O., Spicer, J.I., Ibbini, Z. et al. Spectral phenotyping of embryonic development reveals
    integrative thermodynamic responses. BMC Bioinformatics 22, 232 (2021). https://doi.org/10.1186/s12859-021-04152-1

    """
    # Crop to new size required for binning
    first_frame = vuba.take_first(frames)
    new_x, new_y = map(lambda v: math.floor(v / binsize) * binsize, first_frame.shape)

    mpxs = [
        view_as_blocks(frame[:new_x, :new_y], (binsize, binsize)).mean(axis=(2, 3))
        for frame in frames
    ]

    return np.asarray(mpxs)


def epts(mpx, fs):
    """
    Compute the energy proxy traits for a mean pixel value grid.

    Parameters
    ----------
    mpx : ndarray
        Mean pixel value grid obtained using ``mpx_grid``.
    fs : int or float
        Sampling frequency for video.

    Returns
    -------
    ept : tuple
        Energy proxy traits, split into frequency and power arrays respectively.

    See Also
    --------
    mpx_grid
    spectral_map

    References
    ----------

    Tills, O., Spicer, J.I., Ibbini, Z. et al. Spectral phenotyping of embryonic development reveals
    integrative thermodynamic responses. BMC Bioinformatics 22, 232 (2021). https://doi.org/10.1186/s12859-021-04152-

    """
    freq = np.empty((int((mpx.shape[0] / 2) + 1), mpx.shape[1], mpx.shape[2]))
    power = freq.copy()

    with tqdm(total=mpx.shape[1]*mpx.shape[2]) as pg:
        for i in range(mpx.shape[1]):
            for j in range(mpx.shape[2]):
                freq[:, i, j], power[:, i, j] = signal.welch(
                    mpx[:, i, j], fs=fs, scaling="spectrum", nfft=len(mpx[:, i, j])
                )
                pg.update(1)

    return (freq, power)


def _parse_args(freq, args):
    """
    Parse frequency argument for ``spectral_map`` and produce a condition
    for indexing EPT arrays.

    """
    arg_type = type(args)

    if arg_type == tuple:
        fmin, fmax = args

        if fmin and fmax:
            condition = (freq >= fmin) & (freq <= fmax)
        else:
            if fmin and not fmax:
                condition = freq >= fmin
            elif fmax and not fmin:
                condition = freq <= fmax

    elif arg_type == int or arg_type == float:
        condition = freq == args

    elif arg_type == list or np.ndarray:
        if arg_type == list:
            args = np.asarray(args)

        condition = np.asarray([np.where(freq == i)[0][0] for i in args])

    return condition


def spectral_map(epts, frequencies):
    """
    Create a heatmap based on the spectral energy in the supplied energy proxy traits (EPTs)

    Parameters
    ----------
    epts : ndarray
        Energy proxy traits computed using ``epts``.
    Frequencies : int or float or tuple
        Frequencies to keep in the EPTs supplied to create the heatmap.

    Returns
    -------
    heatmap : ndarray
        Two dimensional heatmap.

    See also
    --------
    epts
    mpx_grid

    """
    freq, power = epts
    power_map = np.empty_like(vuba.take_first(power))

    for i in range(freq.shape[1]):
        for j in range(freq.shape[2]):
            freq_, power_ = freq[:, i, j], power[:, i, j]

            keep = _parse_args(freq_, frequencies)
            power_lim = power_[keep]
            power_map[i, j] = np.sum(power_lim)

    power_map = minmax_scale(power_map) * 255
    power_map = power_map.astype("uint8")

    return power_map


def identify_frequencies(video, epts, indices=(None, None)):
    """
    Launch a user interface to identify frequencies of interest in the supplied
    energy proxy traits.

    Parameters
    ----------
    video : vuba.Video
        Video instance of a desired footage.
    epts : ndarray
        Energy proxy traits produced using ``epts``.
    indices : tuple
         Frame indices to limit GUI to, especially useful when analysing long sequences of footage.
         First index will always be interpreted as the minimum index and the second as the maximum index.
         If None is specified to either limit, then that limit will be ignored. Default is for no limits, i.e. (None, None).

    Returns
    -------
    frequency : int or float
        Last selected frequency in user interface.

    See also
    --------
    mpx_grid
    epts
    spectral_map

    """
    freq, power = epts
    length = len(freq)

    gui = vuba.VideoGUI(video=video, indices=indices, title="EPT GUI")

    @gui.method
    def locate(gui):
        frame = gui.frame.copy()
        frame_gray = vuba.gray(frame)

        at = freq[:, 0, 0][gui["freq_ind"]]
        power_map = spectral_map((freq, power), at)
        power_map = cv2.resize(power_map, video.resolution)
        roi = detect_largest(power_map)
        rect = vuba.fit_rectangles(roi, rotate=True)
        c = cv2.boxPoints(rect)
        c = np.int0(c)
        vuba.draw_contours(frame, c, 0, (0, 255, 0), 1)

        both = np.hstack((frame, vuba.bgr(power_map)))

        return both

    gui.trackbar("Frequency index", id="freq_ind", min=0, max=length)(None)

    gui.run()

    return freq[:, 0, 0][gui["freq_ind"]]


def detect_largest(map):
    """
    Detect the largest polygon in a heatmap produced by ``spectral_map``.

    Parameters
    ----------
    map : ndarray
        Heatmap produced by ``spectral_map``.

    Returns
    -------
    roi : tuple
        Bounding box for largest shape detected in the supplied heatmap.

    See also
    --------
    segment

    """
    ret, thresh = cv2.threshold(map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = vuba.find_contours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    largest = vuba.largest(contours)

    return largest


def segment(frames, roi, invert=False):
    """
    Segment to the supplied ROI.

    Parameters
    ----------
    frames : list or ndarray or vuba.Frames
        Frames to segment.
    roi : ndarray
        Region of interest to segment to (contour).
    invert : bool
        Whether to invert the final segmentation of the
        supplied frames.

    Returns
    -------
    frames : generator
        Generator that supplies segmented frames.

    See also
    --------
    detect_largest

    """
    if type(roi) == tuple:
        mask = vuba.rectangle_mask(vuba.take_first(frames), roi)
    else:
        mask = vuba.contour_mask(vuba.take_first(frames), roi)
    
    if invert:
        mask = cv2.bitwise_not(mask)

    segm = vuba.Mask(mask)

    for frame in frames:
        yield segm(frame)


@njit
def _lms(x_dtr):
    """
    Numba optimised method for computing the Local Maxima Scalogram (LMS)
    from a linearly detrended signal.

    """

    N = len(x_dtr)
    L = N // 2

    LMS = np.random.random((L, N)) + 1
    for k in range(1, L):
        for i in range(k + 2, N - k + 1):
            if (x_dtr[i - 1] > x_dtr[i - k - 1]) & (x_dtr[i - 1] > x_dtr[i + k - 1]):
                LMS[k, i] = 0

    return LMS


def _ampd_visualoutput(x, LMS, row_sums, G, S, peaks):
    """
    Equivalent visual output for the AMPD algorithm to that
    presented in the original paper.

    """
    scale = [i for i in range(len(x) // 2)]
    index = np.asarray([i for i in range(len(x))])

    fig = plt.figure()

    gs = GridSpec(2, 4)  # 2 rows, 4 columns

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, :])

    im = ax1.imshow(LMS, cmap="hot", aspect="auto")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Scale [No]")
    ax1.text(-0.1, 1.1, "a)", transform=ax1.transAxes, size=12)
    ax1.set_title("Local maxima scalogram (LMS)")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    ax2.plot(scale, row_sums, "k", linewidth=1.0)
    ax2.axvline(x=scale[np.argmin(row_sums)], color="r")
    ax2.set_xlabel("Scale [No]")
    ax2.set_title("Row-wise summation of LMS")
    ax2.text(-0.1, 1.1, "b)", transform=ax2.transAxes, size=12)

    ax3.imshow(G, cmap="hot", aspect="auto")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Scale [No]")
    ax3.set_title("Rescaled LMS")
    ax3.text(-0.1, 1.1, "c)", transform=ax3.transAxes, size=12)

    ax4.plot(S, "k", linewidth=0.5)
    ax4.set_xlim([0, len(S)])
    ax4.set_ylim([0, 1])
    ax4.set_xlabel("Index")
    ax4.set_ylabel("σ")
    ax4.set_title("Column-wise standard deviation \n of the rescaled LMS")
    ax4.text(-0.1, 1.1, "d)", transform=ax4.transAxes, size=12)

    ax5.plot(x, "k")
    ax5.set_xlim([0, len(x)])
    for v in peaks:
        ax5.axvline(x=v, color="k", linestyle="--", alpha=0.25)
    ax5.plot(peaks, x[peaks], "or")
    ax5.set_xlabel("Index")
    ax5.set_ylabel("Amplitude")
    ax5.set_title("Detected peaks")
    ax5.text(-0.025, 1.05, "e)", transform=ax5.transAxes, size=12)

    plt.show()


def find_peaks(x, visual_output=False):
    """
    Python implementation of automatic multiscale peak detection (AMPD).

    Parameters
    ----------
    x : list or ndarray
        Signal to detect peaks for.
    visual_output : bool
        Whether to produce visual output for the signal provided. If False,
        only the detected peaks will be returned. Default is False.

    Returns
    -------
    peaks : ndarray
        Detected peaks.

    References
    ----------

    Scholkmann, F., Boss, J. and Wolf, M., 2012. An Efficient Algorithm for Automatic
    Peak Detection in Noisy Periodic and Quasi-Periodic Signals. Algorithms, 5(4), pp.588-603.

    """
    x = np.asarray(x)

    # Linearly detrend signal
    x_dtr = signal.detrend(x)

    # Compute LSM
    LMS = _lms(x_dtr)

    # Find scale with most local maxima
    row_sums = LMS.sum(axis=1)

    # Rescale LMS
    G = LMS[1 : np.argmin(row_sums), :]

    # Column-wise standard deviation of rescaled LMS
    S = np.std(G, axis=0)

    # Extract peaks
    peaks = np.where(S == 0)[0] - 1

    if visual_output:
        _ampd_visualoutput(x, LMS, row_sums, G, S, peaks)

    return peaks


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

    """
    vals = np.asarray(vals)
    m = np.nanmin(vals)
    M = np.nanmax(vals)
    return (vals - m) / (M - m)


def bpm(num_peaks, sample_length, fs):
    """
    Calculate beats per minute based on detected peaks.

    Parameters
    ----------
    num_peaks : int or float
        Number of peaks detected.
    sample_length : int or float
        Length of time-series data.
    fs : int or float
        Sampling frequency of footage.

    Returns
    -------
    bpm : int or float
        Beats per minute.

    """
    return (num_peaks / (sample_length / fs)) * 60


def b2b_intervals(peaks, fs):
    """
    Calculate beat to beat intervals based on peaks detected.

    Parameters
    ----------
    peaks : ndarray
        Detected peaks.
    fs : int or float
        Sampling frequency of footage.

    Returns
    -------
    intervals : ndarray
        Beat to beat intervals.

    """
    return (peaks[1:] - peaks[:-1]) / fs


def stats(peaks, sample_length, fs):
    """
    Calculate cardiac stats for the peaks detected.

    Parameters
    ----------
    peaks : ndarray
        Detected peaks.
    sample_length : int
        Length of time-series data.
    fs : int or float
        Sampling frequency of footage.

    Returns
    -------
    stats : dict
        Dictionary containing cardiac stats based on the data
        provided. The following keys can be accessed:

        - 'bpm': Beats per minute.
        - 'min_b2b': Minimum beat to beat interval.
        - 'mean_b2b': Mean beat to beat interval.
        - 'median_b2b': Median beat to beat interval.
        - 'max_b2b': Maximum beat to beat interval.
        - 'sd_b2b': Standard deviation in beat to beat interval.
        - 'range_b2b': Range in beat to beat intervals.
        - 'rmssd': Root mean square of successive differences in beat
           to beat intervals.

    """
    b2b = b2b_intervals(peaks, fs)

    return dict(
        bpm=bpm(len(peaks), sample_length, fs),
        min_b2b=b2b.min(),
        mean_b2b=b2b.mean(),
        median_b2b=np.median(b2b),
        max_b2b=b2b.max(),
        sd_b2b=np.std(b2b),
        range_b2b=b2b.max() - b2b.min(),
        rmssd=np.sqrt(np.nanmean((b2b[1:] - b2b[:-1]) ** 2)),
    )
