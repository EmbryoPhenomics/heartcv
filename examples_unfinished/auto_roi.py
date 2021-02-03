import heartcv as hcv
import vuba
import cv2
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
frames = video.read(start=0, stop=300, grayscale=True)

frame = video.read(0, grayscale=True)
mask = vuba.shrink(np.ones_like(frame), by=50)

# Find all roi's between binary threshold range (1,150,2) and gaussian blur (1,61,2)
# for the sum difference image
diff_img = hcv.sum_abs_diff(frames, mask=mask)
_, bboxs = hcv.roi_search(diff_img, (1, 150, 2), (1, 61, 2))
bboxs = np.asarray(bboxs)

# Compute corresponding areas for each roi
box_area = lambda x, y, w, h: w * h
bboxs_areas = np.asarray([box_area(*bbox) for bbox in bboxs])

# For some reason a tuple is returned from np.where here, hence the zero index
# Area limits (px) will need to be adjusted depending on the developmental stage
indices = np.where((bboxs_areas >= 1000) & (bboxs_areas <= 20000))[0]

# This can probably be cleaned up but works for now
good_bboxs = bboxs[indices]
x, y, w, h = map(np.median, zip(*good_bboxs))
bbox = tuple(map(int, (x, y, w, h)))

# Draw the roi to make sure it's correct
frame = vuba.bgr(frame)
vuba.draw_rectangles(frame, bbox, (0, 255, 0), 1)
cv2.imshow("ROI view", frame)
cv2.waitKey()

# Subset frames to roi (imported into memory here)
frames = list(subset(frames, *bbox))

# Compute image statistics
diff_vals = [np.sum(diff) for diff in hcv.abs_diffs(frames)]
mpx_vals = [np.mean(frame) for frame in frames]

# Normalise to same scale (0-1)
diff_vals, mpx_vals = map(hcv.minmax_scale, (diff_vals, mpx_vals))

plt.plot(mpx_vals)
plt.plot(diff_vals)
plt.show()

video.close()
