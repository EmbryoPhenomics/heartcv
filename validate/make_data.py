oimport vuba
import cv2
import pandas as pd
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

import heartcv as hcv

def lowess_smooth(x, frac=0.015):
    x = np.asarray(x)
    return lowess(x, [i for i in range(len(x))], it=0, frac=frac)[:,1]

def process(files):
    for file in files:
        video = vuba.Video(file)
        frames = video.read(0, 300, low_memory=False, grayscale=True)
        first = vuba.take_first(frames)

        if 'radix' in file or 'paleomon' in file:
            frames = np.asarray([vuba.shrink(frame, by=50) for frame in frames])

        # Localisation
        mpx = hcv.mpx_grid(frames, binsize=16)  # Downsample images
        ept = hcv.epts(mpx, fs=video.fps)  # Compute energy proxy traits (EPTs)
        roi, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation

        # Segment all images to this cardiac region
        segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))
        v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal

        if 'ciona' in file:
            v = lowess_smooth(v) # Smooth for missed frames in that experiment.

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

        # Data output ---------------------------------
        cardiac_measures = hcv.stats(peaks, len(video) * 3, video.fps * 3) # Length and fps multiplied by 3 to match upsampling of the MPV signal

        for c in cardiac_measures.keys():
          cardiac_measures[c] = [cardiac_measures[c]]

        df_stats = pd.DataFrame(data=cardiac_measures)
        df_raw = pd.DataFrame(data=dict(mpv=v))

        df_stats.to_csv(re.sub('.avi', '.csv', file))
        df_raw.to_csv(re.sub('.avi', '_raw.csv', file))


ciona_files = glob.glob('./video/ciona/*.avi')
radix_files = glob.glob('./video/radix/*.avi')
paleomon_files = glob.glob('./video/paleomon/*.avi')

process(ciona_files)
process(radix_files)
process(paleomon_files)
