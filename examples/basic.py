import heartcv as hcv
import vuba
import numpy as np
import matplotlib.pyplot as plt

hcv.show_progress(True)

video = vuba.Video("./data/test.avi")
frames = video.read(start=0, stop=100, grayscale=True)

frame = video.read(0)
frame_gray = vuba.gray(frame)
mask = vuba.shrink(np.ones_like(frame_gray), by=50)

# Compute sum frame differences at roi
diff_vals = hcv.minmax_scale([np.sum(diff) for diff in hcv.abs_diffs(frames)])

# Find peaks and troughs
t,dia,sys = hcv.find_events(diff_vals, prominence=0.05)

# Exclude second peaks so we only have contraction phase
dia_vals, dia_peaks = dia
idx = 1 # change to one if peaks shifted by one due to a peak at start of sequence
dia_vals = np.asarray([d for i,d in enumerate(dia_vals) if i%2 == idx])

plt.plot(diff_vals)
plt.plot(dia_vals, diff_vals[dia_vals], 'x')
plt.show()
