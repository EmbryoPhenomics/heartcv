# Example script showcasing the use of HeartCV on multiple
# timepoints with user supervision but with export of all HeartCV
# outputs including localisation maps.

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import vuba
import numpy as np
import cv2
import pandas as pd
import heartcv as hcv

# Input parameters to change ---------
use_example_video = False  # Change to False and add file path to source_video below to use your own video
source_video = "/run/media/z/fast2/lymnaea_backup_mjpg/20C/A_A2.avi" # Path to multi-timepoint concatenated video
output_filename = "./output.csv" # Cardiac statistics
timepoint_interval = 600
# ------------------------------------

# Import video
if use_example_video:
    video = hcv.load_example_video()
else:
    video = vuba.Video(source_video)

cardiac_stats = dict(
    timepoint=[],
    bpm=[],
    min_b2b=[],
    mean_b2b=[],
    median_b2b=[],
    max_b2b=[],
    sd_b2b=[],
    range_b2b=[],
    rmssd=[],
)

# Matplotlib interactivity
class Buttons:
    answer = None

    def yes(self, event):
        self.answer = True
        plt.close()

    def no(self, event):
        self.answer = False
        plt.close()

for i in range(0, len(video), timepoint_interval):
    timepoint = round(i / timepoint_interval)

    frames = video.read(
        start=i,  # Initial frame index to import frames from
        stop=i + timepoint_interval,  # End index to stop importing frames
        grayscale=True, # Grayscale images upon reading them into memory
    )  

    # Localisation
    ept = hcv.epts(frames, fs=video.fps, binsize=16)  # Compute energy proxy traits (EPTs)

    # Supervision of localisation
    roi, _, lmap = hcv.identify_frequencies(
        video, ept, (i, i + timepoint_interval)
    )

    ext = str.split(source_video, '.')[-1]
    cv2.imwrite(str.replace(source_video, f'.{ext}', f'_timepoint_{timepoint}.png'), lmap)

    # Segment all images to this cardiac region
    segmented_frames = np.asarray(
        list(hcv.segment(frames, vuba.fit_rectangles(roi)))
    )  

    v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

    # Peak detection
    v = v.max() - v  # invert signal

    # Upsample by a factor of 3 to improve peak detection
    v = np.interp(
        [i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v
    )

    peaks = hcv.find_peaks(v)
    stats = hcv.stats(peaks, len(v) * 3, video.fps * 3)

    # Plot the results
    time = np.asarray([(j + (i * 3)) / (video.fps * 3) for j in range(0, len(v))])

    fig = plt.figure(figsize=(7, 5))
    plt.plot(time, v, color='k')
    plt.plot(time[peaks], v[peaks], 'ro')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean pixel value (px)")
    plt.title(f'Timepoint: {timepoint}')
    plt.suptitle('Check peaks and choose whether to save cardiac statistics:')

    callback = Buttons()
    axno = plt.axes([0.7, 0.88, 0.1, 0.055])
    axyes = plt.axes([0.81, 0.88, 0.1, 0.055])
    byes = Button(axyes, 'Decline')
    byes.on_clicked(callback.no)
    bno = Button(axno, 'Accept')
    bno.on_clicked(callback.yes)
    plt.tight_layout()

    plt.show()

    cardiac_stats['timepoint'].append(timepoint)
    for key in cardiac_stats.keys():
        if key == 'timepoint':
            continue

        if callback.answer:
            cardiac_stats[key].append(stats[key])
        else:
            cardiac_stats[key].append(np.nan)

# Data output
df = pd.DataFrame(data=cardiac_stats)
df.to_csv(output_filename)
