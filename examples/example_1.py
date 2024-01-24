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
from matplotlib.animation import FuncAnimation
from matplotlib import animation

from scipy import signal

# Input parameters to change
use_example_video = False  # Change to False and add file path to source_video below to use your own video
source_video = "/run/media/z/thumb1/Analysis_Tests/Individual_Heart.avi"
output_filename = "./output.csv"
# ---------------------------

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

frames = video.read(
    start=0,  # Initial frame index to import frames from
    stop=len(video)-10,  # End index to stop importing frames
    grayscale=True,
    low_memory=False
)  # Grayscale images upon reading them into memory

# Localisation
ept = hcv.epts(frames, fs=video.fps, binsize=2)  # Compute energy proxy traits (EPTs) - video is downsampled in this function.
roi, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation

# Segment all images to this cardiac region
segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

# # Peak detection
# v = np.interp(
#     [i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v
# )  # upsample by a factor of 3 to improve peak detection

peaks = hcv.find_peaks(v)  # Find peaks using AMPD

# Plot the results
time = np.asarray([i / video.fps for i in range(len(v))])

annotated_frames = []
for f in frames:
    rect = vuba.fit_rectangles(roi)
    f = vuba.bgr(f).astype(np.uint8)
    vuba.draw_rectangles(f, rect, (0,255,0), 1)
    annotated_frames.append(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

im1 = ax1.imshow(annotated_frames[0])
ax2_line, = ax2.plot(time, v, 'k')
ax2_peaks, = ax2.plot(time[peaks], v[peaks], 'ro')

ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Mean pixel value (px)")

ax1.set_title('HeartCV Localisation')
ax2.set_title('Mean pixel value time-series')

def animate(i):
    im1.set_array(annotated_frames[i])

    at_time = time[:i]
    at_v = v[:i]
    at_peaks = peaks[peaks < i]

    ax2_line.set_data(at_time, at_v)
    ax2_peaks.set_data(at_time[at_peaks], at_v[at_peaks])

    return im1, ax2_line, ax2_peaks

ani = FuncAnimation(fig, animate, frames=[i for i in range(len(annotated_frames))])
writer = animation.FFMpegWriter(
    fps=video.fps, metadata=dict(artist='Me'), bitrate=3200)
ani.save("./FEC_heartcv_test.gif", writer=writer)



# # Data output
# cardiac_measures = hcv.stats(
#     peaks, len(video) * 3, video.fps * 3
# )  # Length and fps multiplied by 3 to match upsampling of the MPV signal

# df = pd.DataFrame(data=cardiac_measures)
# df.to_csv(output_filename)
