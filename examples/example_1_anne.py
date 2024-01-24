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

# Input parameters to change
use_example_video = False  # Change to False and add file path to source_video below to use your own video
source_video = "/run/media/z/rootfs/home/pi/Turner-salinity/Pos38/timepoint24.avi"
output_filename_hr = "/run/media/z/rootfs/home/pi/Turner-salinity/Pos38/timepoint24_hr.csv"
output_filename_raw = "/run/media/z/rootfs/home/pi/Turner-salinity/Pos38/timepoint24_signal.csv"
# ---------------------------

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

frames = video.read(
    start=30,  # Initial frame index to import frames from
    stop=len(video),  # End index to stop importing frames
    grayscale=True,
)  # Grayscale images upon reading them into memory

# Localisation
ept = hcv.epts(frames, fs=video.fps, binsize=8)  # Compute energy proxy traits (EPTs) - video is downsampled in this function.
roi, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation

# Segment all images to this cardiac region
segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

# v = v.max() - v

# Peak detection
v = np.interp(
    [i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v
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

for k,i in cardiac_measures.items():
    cardiac_measures[k] = [i]

df = pd.DataFrame(data=cardiac_measures)
df.to_csv(output_filename_hr)

empty = v.copy()
empty[:] = 0
empty[peaks] = 1
df = pd.DataFrame(data=dict(time=time, px_value=v, heartbeats=empty))
df.to_csv(output_filename_raw)