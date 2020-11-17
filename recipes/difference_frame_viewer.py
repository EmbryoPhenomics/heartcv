import heartcv as hcv
import numpy as np
import cv2
from more_itertools import pairwise

# ---------------------------------------------------
def diff_viewer(frames):
    '''
    Interactive gui for viewing differences frames.

    '''
    first = hcv.take_first(frames)
    size = np.swapaxes(first,0,1).shape # required to get correct dims for cv2.resize()

    gui = hcv.FramesGUI(frames, 'Frame viewer', size)

    # Here we do nothing except supply the current frame since we don't want 
    # to do any image processing on the difference frames
    @gui.main_process
    def segment(gui):
        frameProc = gui.frame.copy()
        return frameProc

    gui.run()

# ---------------------------------------------------

# Request the library to show progress information
hcv.show_progress(True)

# Filename to footage
filename = '../examples/data/10.avi'

# Initiate a Video instance to open the video
video = hcv.Video(filename)

# Create a handler for importing frames
frame = video.read(0, grayscale=False)
frames = video.read(start=0, stop=len(video))

# Compute the absolute differences between consecutive frames in the footage
diff_frames = hcv.abs_diffs(frames=frames)
diff_viewer(diff_frames)

video.close()



