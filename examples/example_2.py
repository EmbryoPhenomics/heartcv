# Example script showcasing the use of HeartCV on multiple
# timepoints with user supervision

# For visualisation of results
import matplotlib.pyplot as plt

# For image and signal utilities
import vuba
import numpy as np
import cv2

# For exporting results into csv
import pandas as pd

import heartcv as hcv


# Input parameters to change --
use_example_video = True  # Change to False and add file path to source_video below to use your own video
source_video = "/path/to/video"
output_filename = "./output.csv"
# -----------------------------

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

cardiac_stats = dict(
    bpm=[],
    min_b2b=[],
    mean_b2b=[],
    median_b2b=[],
    max_b2b=[],
    sd_b2b=[],
    range_b2b=[],
    rmssd=[],
)

timepoint_interval = 300  # Here we've set this arbritrarily, but this is calculated as fps x length of timepoint in seconds

for i in range(0, len(video), timepoint_interval):
    frames = video.read(
        start=i,  # Initial frame index to import frames from
        stop=i + timepoint_interval,  # End index to stop importing frames
        grayscale=True)  # Grayscale images upon reading them into memory

    # Localisation ----------------------------------
    mpx = hcv.mpx_grid(frames, binsize=16)  # Downsample images (same binning factor as used in the paper)
    ept = hcv.epts(mpx, fs=video.fps)  # Compute energy proxy traits (EPTs)
    roi, _ = hcv.identify_frequencies(video, ept, indices=(i, i + timepoint_interval))  # Supervision of localisation

    segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))  # Segment all images to this cardiac region
    v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

    # Peak detection --------------------------------
    v = v.max() - v  # invert signal
    v = np.interp([i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v)  # upsample by a factor of 3 to improve peak detection

    peaks = hcv.find_peaks(v)
    stats = hcv.stats(peaks, len(v) * 3, video.fps * 3)

    for key in cardiac_stats.keys():
        cardiac_stats[key].append(stats[key])

    # Plot the results -------------------------------
    time = np.asarray([(j + (i * 3)) / (video.fps * 3) for j in range(0, len(v))])

    n = int(i / timepoint_interval)
    plt.plot(time, v, color=f"C{n}", label=f"Timepoint {n}")
    plt.plot(time[peaks], v[peaks], "o", color=f"C{n}")

plt.legend(loc="lower right")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean pixel value (px)")
plt.show()

# Data output ----------------------------------------
df = pd.DataFrame(data=cardiac_stats)
df.to_csv(output_filename)
