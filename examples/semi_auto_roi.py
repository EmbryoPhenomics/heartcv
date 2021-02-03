import heartcv as hcv
import vuba
import numpy as np
import matplotlib.pyplot as plt


def subset(frames, x, y, w, h):
    if frames.in_mem:
        return frames.ndarray[:, y : y + h, x : x + w, ...]
    else:
        for frame in frames:
            yield frame[y : y + h, x : x + w, ...]


hcv.show_progress(True)

video = vuba.Video("./data/test.avi")
frames = video.read(start=0, stop=300, grayscale=True)

frame = video.read(index=0, grayscale=True)
ones = np.ones_like(frame)
mask = vuba.shrink(ones, by=50)

# Find heart roi through the gui
diff_img = hcv.sum_abs_diff(frames, mask)
bbox, _, _ = hcv.activity_gui(video, diff_img)

# Subset frames to roi (imported into memory here)
frames = list(subset(frames, *bbox))

# Compute sum frame differences at roi
diff_vals = np.asarray([np.sum(diff) for diff in hcv.abs_diffs(frames)])

# Find peaks and troughs
t, dia, sys = hcv.find_events(diff_vals, prominence=0.1)

# Exclude second peaks so we only have contraction phase
dia_vals, dia_peaks = dia
idx = 1  # change to one if peaks shifted by one due to a peak at start of sequence
dia_vals = np.asarray([d for i, d in enumerate(dia_vals) if i % 2 == idx])

plt.plot(diff_vals)
plt.plot(dia_vals, diff_vals[dia_vals], "x")
plt.show()

video.close()
