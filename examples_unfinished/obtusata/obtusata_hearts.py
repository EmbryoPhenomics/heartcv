import heartcv as hcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import cvu

from heartcv import minmax_scale as scale

hcv.show_progress(True)

video = cvu.Video("/home/z/Downloads/Obtusatamp4.mp4")
frame = video.read(0, grayscale=True)

frames = video.read(start=0, stop=len(video), grayscale=True)

loc = hcv.Location(hcv.binary_thresh, cvu.largest)

contour, _ = hcv.location_gui(video, loc)

mask = cvu.contour_mask(frame, contour)
diffImg = hcv.sum_abs_diff(frames, mask)

bbox, _, _ = hcv.activity_gui(frame, diffImg, rotate=False)
first_rect_mask = cv2.bitwise_not(cvu.rect_mask(frame, bbox))
first_rect_mask = cv2.bitwise_and(first_rect_mask, first_rect_mask, mask=mask)
diffImg = hcv.sum_abs_diff(frames, first_rect_mask)

bbox2, _, _ = hcv.activity_gui(frame, diffImg, rotate=False)
second_rect_mask = cv2.bitwise_not(cvu.rect_mask(frame, bbox2))
second_rect_mask = cv2.bitwise_and(
    second_rect_mask, second_rect_mask, mask=first_rect_mask
)
diffImg = hcv.sum_abs_diff(frames, second_rect_mask)

bbox3, _, _ = hcv.activity_gui(frame, diffImg, rotate=False)
heart1 = cvu.Mask(cvu.rect_mask(frame, bbox3))
third_rect_mask = cv2.bitwise_not(cvu.rect_mask(frame, bbox3))
third_rect_mask = cv2.bitwise_and(
    third_rect_mask, third_rect_mask, mask=second_rect_mask
)
diffImg = hcv.sum_abs_diff(frames, third_rect_mask)

bbox, _, _ = hcv.activity_gui(frame, diffImg, rotate=False)
heart2 = cvu.Mask(cvu.rect_mask(frame, bbox))

heart1_fr = [heart1(frame) for frame in frames]
heart2_fr = [heart2(frame) for frame in frames]

mpxvals = scale([np.mean(frame) for frame in heart1_fr])
diff_frames = hcv.abs_diffs(heart1_fr)
diff_vals = scale([np.sum(diff) for diff in diff_frames])

plt.plot(mpxvals)
plt.plot(diff_vals)
plt.show()

mpxvals = scale([np.mean(frame) for frame in heart2_fr])
diff_frames = hcv.abs_diffs(heart2_fr)
diff_vals = scale([np.sum(diff) for diff in diff_frames])

plt.plot(mpxvals)
plt.plot(diff_vals)
plt.show()

writer = cvu.Writer("./radix/hcv_visualoutput/obtusata.avi", video)
with hcv.pgbar(len(diff_frames)) as pgbar:
    for diff, bgr in zip(diff_frames, video.read(start=0, stop=len(video))):
        blobs = list(cvu.find_contours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))[0]
        if blobs is not None:
            cvu.draw_contours(bgr, blobs, -1, (0, 255, 255), -1)

        x, y, w, h = bbox3
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        x, y, w, h = bbox
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)

        writer.write(bgr)
        pgbar.update(1)

writer.close()
video.close()
