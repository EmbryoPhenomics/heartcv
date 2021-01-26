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

video = vuba.Video("./paleomon/video/test.avi")
frames = video.read(start=0, stop=100, grayscale=True)

frame = video.read(index=0)

diff_img = hcv.sum_abs_diff(frames)
bbox, _, _ = hcv.activity_gui(video, diff_img)
frames = list(subset(frames, *bbox))

diff_vals = [np.sum(diff) for diff in hcv.abs_diffs(frames)]
mpx_vals = [np.mean(frame) for frame in frames]

diff_vals, mpx_vals = map(hcv.minmax_scale, (diff_vals, mpx_vals))

plt.plot(mpx_vals)
plt.plot(diff_vals)
plt.show()

video.close()
