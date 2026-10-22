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
# Localisation
ept = hcv.epts(frames, fs=video.fps, binsize=2)  # Compute energy proxy traits (EPTs) - video is downsampled in this function.
roi, freq, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation

# Segment all images to this cardiac region
segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

# Peak detection
peaks = hcv.find_peaks(v, plot=False)  # Find peaks using AMPD

# Plot the results
time = np.asarray([i / video.fps for i in range(len(v))])

# Plot the results ---------------------------- 
plt.plot(time, v, "k")
plt.plot(time[peaks], v[peaks], "or")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean pixel value (px)")
plt.show()

# Data output ---------------------------------
cardiac_measures = hcv.stats(peaks, len(video), video.fps) 

df = pd.DataFrame(data=cardiac_measures)
df.to_csv(output_filename)