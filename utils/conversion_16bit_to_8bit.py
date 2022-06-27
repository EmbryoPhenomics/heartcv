# Python script for conversion between 16-bit to 8-bit image depth, based on a constant scaling factor.
# Note that this script operates on image sequences and so produces concatenated video as output.

import imageio
import vuba
import tqdm

# Input parameters
source_video = "/path/to/*.tif"
output_video = "path/to/out.avi"
output_fps = 10
scaling_factor = 25
# --------------------------

video = vuba.Video(source_video)
writer = vuba.Writer(output_video, video, fps=output_fps, codec='RGBA', grayscale=True)

for file in tqdm.tqdm(video.filenames):
    frame = imageio.imread(file)

    img_scaled = frame * scaling_factor
    img_scaled = (img_scaled >> 8).astype("uint8")

    writer.write(img_scaled)

writer.close()
video.close()