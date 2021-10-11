import heartcv as hcv
import vuba
import cv2
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt

def process(files):
    for file in files:
        vid = vuba.Video(file)
        frames = vid.read(0, 300, low_memory=False, grayscale=True)
        first = vuba.take_first(frames)
        frames = np.asarray([vuba.shrink(frame, by=50) for frame in frames])

        mpx = hcv.mpx_grid(frames, binsize=8)

        ept = hcv.epts(mpx_, vid.fps)
        freq = hcv.identify_frequencies(vid, ept)
        map_ = hcv.spectral_map(ept, freq)
        map_ = cv2.resize(map_, vid.resolution)

        roi = hcv.detect_largest(map_)
        rect = vuba.fit_rectangles(roi, rotate=True)
        c = cv2.boxPoints(rect)
        c = np.int0(c)

        at_roi = np.asarray(list(hcv.segment(frames, c)))

        mpx = at_roi.mean(axis=(1, 2))

        plt.plot(mpx)
        plt.show()

        df = pd.DataFrame(data=dict(area=mpx_))
        out = re.sub('.avi', '.csv', file)
        df.to_csv(re.sub('video', 'hcv_data', out))


ciona_files = glob.glob('./video/ciona/*.avi')
radix_files = glob.glob('/home/z/Documents/radix_raw/last/*.avi')
paleomon_files = glob.glob('./video/paleomon/*.avi')

process(ciona_files)
process(radix_files)
process(paleomon_files)
