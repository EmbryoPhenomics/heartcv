# Example script showcasing the use of HeartCV on a single
# timepoint with supervision over the localisation stage

# For visualisation of results
import matplotlib.pyplot as plt

# For image and signal utilities
import vuba
import numpy as np
import cv2

# For exporting results into csv
import pandas as pd

import heartcv as hcv

from scipy import signal

# Input parameters to change
use_example_video = True  # Change to False and add file path to source_video below to use your own video
source_video = "/run/media/z/Expansion/Pleopods/Labelled pleopod videos/25.15.3_settle1_1.mp4"
output_filename = "./output.csv"
# ---------------------------

def identify_frequencies(video, epts, rotate=True, indices=(None, None)):
    """
    Launch a user interface to identify frequencies of interest in the supplied
    energy proxy traits.

    Parameters
    ----------
    video : vuba.Video
        Video instance of a desired footage.
    epts : ndarray
        Energy proxy traits produced using ``epts``.
    rotate : bool
        Whether to fit a rotated bounding box to the largest shape detected in the EPT
        heatmap. If False, a non-rotated bounding box will be fit. Default is True. 
    indices : tuple
         Frame indices to limit GUI to, especially useful when analysing long sequences of footage.
         First index will always be interpreted as the minimum index and the second as the maximum index.
         If None is specified to either limit, then that limit will be ignored. Default is for no limits, i.e. (None, None).

    Returns
    -------
    bbox : ndarray or tuple
        Bounding box array or tuple describing the coordinates or dimensions over
        which the box has been fit respectively. 
    frequency : int or float
        Last selected frequency in user interface.
    gui : vuba.VideoGUI or None
        If run is False then an instance of VideoGUI will be returned.

    See also
    --------
    mpx_grid
    epts
    spectral_map

    """
    freq, power = epts
    length = len(freq)

    gui = vuba.FramesGUI(video, indices=indices, title="EPT GUI")
    gui.roi = None

    @gui.method
    def locate(gui):
        frame = gui.frame.copy()

        at = freq[:, 0, 0][gui["freq_ind"]]

        power_map = hcv.spectral_map((freq, power), at)
        power_map = cv2.resize(power_map, (frame.shape[1], frame.shape[0]))
        roi = hcv.detect_largest(power_map)

        frame = vuba.bgr(frame)
        rect = vuba.fit_rectangles(roi, rotate=rotate)

        if rotate:
            c = cv2.boxPoints(rect)
            c = np.int0(c)
            vuba.draw_contours(frame, c, 0, (0, 255, 0), 1)
            gui.roi = c
        else:
            vuba.draw_rectangles(frame, rect, (0, 255, 0), 1)
            gui.roi = rect

        both = np.hstack((frame, vuba.bgr(power_map)))

        return both

    gui.trackbar("Frequency index", id="freq_ind", min=0, max=length)(None)

    gui.run()
    return gui.roi, freq[:, 0, 0][gui["freq_ind"]]

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

frames = video.read(
    start=0,  # Initial frame index to import frames from
    stop=len(video),  # End index to stop importing frames
    grayscale=True,
    low_memory=False
)  # Grayscale images upon reading them into memory

frames = [cv2.resize(frame, (960, 540)) for frame in frames]

# Localisation
ept = hcv.epts(frames, fs=video.fps, binsize=16)  # Compute energy proxy traits (EPTs) - video is downsampled in this function.
roi, _ = identify_frequencies(frames, ept, indices=(0, len(video)))  # Supervision of localisation

# Segment all images to this cardiac region
segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

# Peak detection
v = np.interp(
    [i / 2 for i in range(len(v) * 2)], np.arange(0, len(v)), v
)  # upsample by a factor of 3 to improve peak detection

peaks = hcv.find_peaks(v)  # Find peaks using AMPD

# Plot the results
time = np.asarray([i / (video.fps * 3) for i in range(len(v))])

plt.plot(time, v, "k")
plt.plot(time[peaks], v[peaks], "or")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean pixel value (px)")
plt.show()

# Data output
cardiac_measures = hcv.stats(
    peaks, len(video) * 3, video.fps * 3
)  # Length and fps multiplied by 3 to match upsampling of the MPV signal

# df = pd.DataFrame(data=cardiac_measures)
# df.to_csv(output_filename)
