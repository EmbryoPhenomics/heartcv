import pandas as pd
import heartcv as hcv
import vuba
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from statsmodels.nonparametric.smoothers_lowess import lowess

def lowess_smooth(x, frac=0.01):
	x = np.asarray(x)
	return lowess(x, [i for i in range(len(x))], it=0, frac=frac)[:,1]

file = '/home/ziad/Documents/paleomon_validate/young_D5.avi'
video = vuba.Video(file)

frames = video.read(
    start=0,  # Initial frame index to import frames from
    stop=300,  # End index to stop importing frames
    grayscale=True)  # Grayscale images upon reading them into memory

frames = [vuba.shrink(f, by=50) for f in frames]

# Localisation --------------------------------
mpx = hcv.mpx_grid(frames, binsize=8)  # Downsample images
ept = hcv.epts(mpx, fs=video.fps)  # Compute energy proxy traits (EPTs)

roi, _ = hcv.identify_frequencies(video, ept)  # Supervision of localisation
print(roi.shape)

# segmented_frames = np.asarray(list(hcv.segment(frames, vuba.fit_rectangles(roi))))  # Segment all images to this cardiac region
# v = segmented_frames.mean(axis=(1, 2))  # Compute MPV signal
# # v = lowess_smooth(v, frac=0.04)

# # Peak detection --------------------------------
# # v = v.max() - v  # invert signal
# v = np.interp([i / 3 for i in range(len(v) * 3)], np.arange(0, len(v)), v)  # upsample by a factor of 3 to improve peak detection

# peaks = hcv.find_peaks(v, plot=True)

# manual = pd.read_csv('/home/ziad/Documents/paleomon_validate/manual_young_D5.csv')
# manual = np.asarray(manual['end_diastole_frame'])

# print(hcv.stats(manual, 300, 25))
# print(hcv.stats(peaks, 300*3, 25*3))

# cardiac_measures = hcv.stats(peaks, 300*3, 25*3)

# for c in cardiac_measures.keys():
# 	cardiac_measures[c] = [cardiac_measures[c]]

# df_stats = pd.DataFrame(data=cardiac_measures)
# df_raw = pd.DataFrame(data=dict(mpv=v))

# df_stats.to_csv(re.sub('.avi', '.csv', file))
# df_raw.to_csv(re.sub('.avi', '_raw.csv', file))

# # cardiac_measures = hcv.stats(peaks, len(v) * 3, video.fps * 3)

# # # Data output ---------------------------------
# # cardiac_measures = hcv.stats(peaks, len(video) * 3, video.fps * 3) # Length and fps multiplied by 3 to match upsampling of the MPV signal

# # print(cardiac_measures)
