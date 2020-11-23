import heartcv as hcv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import cvu

from heartcv import minmax_scale as scale

hcv.show_progress(True)

video = cvu.Video('/home/z/Documents/ciona.avi')
frame = video.read(0, grayscale=True)

frames = video.read(start=0, stop=600, grayscale=True)

loc = hcv.Location(hcv.binary_thresh, cvu.largest)

contour, _ = hcv.location_gui(video, loc)

mask = cvu.contour_mask(frame, contour)
diffImg = hcv.sum_abs_diff(frames, mask)

bbox, contour, _ = hcv.activity_gui(frame, diffImg)
first_rect_mask = cv2.bitwise_not(cvu.contour_mask(frame, np.int0(cv2.boxPoints(bbox))))
first_rect_mask = cv2.bitwise_and(first_rect_mask, first_rect_mask, mask=mask)
diffImg = hcv.sum_abs_diff(frames, first_rect_mask)

bbox, contour, _ = hcv.activity_gui(frame, diffImg)
box = np.int0(cv2.boxPoints(bbox))
final_mask = cvu.contour_mask(frame, contour)
final_mask = cv2.bitwise_and(final_mask, final_mask, mask=mask)
masker = cvu.Mask(final_mask)
masked_frames = [masker(frame) for frame in frames]

mpxvals = scale([np.mean(frame) for frame in masked_frames])
diff_frames = hcv.abs_diffs(masked_frames)
diff_vals = scale([np.sum(diff) for diff in diff_frames])

plt.plot(mpxvals)
plt.plot(diff_vals)
plt.show()

writer = cvu.Writer('/home/z/Documents/ciona_blood_vessel.avi', video)
with hcv.pgbar(len(diff_frames)) as pgbar:
	for diff, bgr in zip(diff_frames, video.read(start=1, stop=600)):
		blobs = list(cvu.find_contours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))[0]
		if blobs is not None:
			cvu.draw_contours(bgr, blobs, -1, (0,255,255), -1)
		cvu.draw_contours(bgr, contour, -1, (0,0,255), 1)
		cvu.draw_contours(bgr, box, -1, (0,255,0), 1)

		writer.write(bgr)
		pgbar.update(1)

writer.close()
video.close()
