import heartcv as hcv
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Helpers for plotting and saving -------------------
def plot(area, pxvals):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(area)
    ax1.set_ylabel('Heart area proxy')

    ax2.plot(pxvals)
    ax2.set_ylabel('Mean pixel values')
    ax2.set_xlabel('Frames')

    plt.show()

def save(area, pxvals, filename):
    df = pd.DataFrame(data=dict(heartArea=area, meanPxVal=pxvals))
    df.to_csv(filename)
# ---------------------------------------------------

# Request the library to show progress information
hcv.show_progress(True)

# Filenames to footage and to save outputs
filename = './data/10.avi'
visual_output = './output10.avi'
data_output = './output10.csv'

# Initiate a Video instance to open the video
video = hcv.Video(filename)
frame = video.read(0)

# Locate the embryo using the supplied method binary_thresh in combination with 
# a contour filter that returns the largest contour by area
contour, _ = hcv.location_gui(video, hcv.Location(hcv.binary_thresh, hcv.largest))

# Create a handler for reading frames
frames = video.read(start=0, stop=len(video))

# Compute the sum of absolute differences for the provided footage
diffImg = hcv.sum_abs_diff(frames=frames, contour=contour)

# Find the bounding box for the heart from the sum image above
bbox, _ = hcv.activity_gui(frame, diffImg)

# Request the handler to subset all frames to the roi of the bounding box
# for faster computations below
frames.subset(*bbox)

# Localise a tighter roi with dense optical flow
coords = hcv.max_optic_flow(frames)

# Cast the coordinates to the original image dimensions
X,Y,_,_ = bbox
x,y = coords
coords = (x+X, y+Y, 40) # here we supply a radius as a third variable for gui below

# Initiate an instance of heartcv.Segmentation with the embryo outline and heart bounding box 
# found above.
segment = hcv.Segmentation(video, contour, bbox, coords)

# Here we launch a gui to tune various image processing parameters using the hcv.largest()
# contour filter. This contour filter simply filters to the largest contour by area. Other
# contour methods are available in the contourfilters module in heartcv.util. Here we invert
# footage to select darker areas within the bounding box found above. After closing this gui
# we get a job that describes a series of keyword arguments to pass to the underlying
# segmentation method
layer = segment.gui(inv=True, contour_method=hcv.largest)
print(layer)

# To use the job we created using the above gui, we need to add it to the instance
# of Segmentation we created.
segment.add_layer(layer)

# Finally we calculate the proxy for heart area using the job we defined. This outputs
# the area proxy and mean pixel values at the bounding box 
areas, pxvals = segment.calc(frames, visual_output)

# Plot and save the above output
plot(areas, pxvals)
save(areas, pxvals, data_output)

# Release the Video instance
video.close()
