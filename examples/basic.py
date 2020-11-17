import heartcv as hcv
import cvu
import numpy as np
import matplotlib.pyplot as plt

hcv.show_progress(True)

video = cvu.Video('./paleomon/video/test.avi')
frames = video.read(start=0, stop=100, grayscale=True)

frame = video.read(0)
frame_gray = cvu.gray(frame)
mask = cvu.shrink(np.ones_like(frame_gray), by=50)

diff_vals = [np.sum(diff) for diff in hcv.abs_diffs(frames, mask=mask)]
mpx_vals = [np.mean(frame) for frame in frames]

diff_vals, mpx_vals = map(hcv.minmax_scale, (diff_vals, mpx_vals))

plt.plot(mpx_vals)
plt.plot(diff_vals)
plt.show()

video.close()