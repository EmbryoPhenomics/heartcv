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
use_example_video = True  # Change to False and add file path to source_video below to use your own video
source_video = "/path/to/video"
output_filename = "./output.csv"
# ---------------------------

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

frames = video.read(
    start=0,  # Initial frame index to import frames from
    stop=len(video),  # End index to stop importing frames
    grayscale=True)  # Grayscale images upon reading them into memory

# Localisation --------------------------------
mpx = hcv.mpx_grid(frames, binsize=16)  # Downsample images
ept = hcv.epts(mpx, fs=video.fps)  # Compute energy proxy traits (EPTs)
roi, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation

# Segment all images to this cardiac region
segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

# Peak detection ------------------------------
v = v.max() - v  # invert signal
v = np.interp([i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v) # upsample by a factor of 3 to improve peak detection

peaks = hcv.find_peaks(v)  # Find peaks using AMPD

# Plot the results ----------------------------
time = np.asarray([i / (video.fps * 3) for i in range(len(v))])

plt.plot(time, v, "k")
plt.plot(time[peaks], v[peaks], "or")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean pixel value (px)")
plt.show()

# Data output ---------------------------------
cardiac_measures = hcv.stats(peaks, len(video) * 3, video.fps * 3) # Length and fps multiplied by 3 to match upsampling of the MPV signal

df = pd.DataFrame(data=cardiac_measures)
df.to_csv(output_filename)
