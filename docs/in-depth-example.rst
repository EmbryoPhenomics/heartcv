.. _in-depth-example:

In-depth example of using HeartCV
=================================

Here we provide a detailed example of using HeartCV and explain each step of the workflow in depth.

The following imports are necessary for this example:

.. ipython:: python

	import heartcv as hcv
	import matplotlib.pyplot as plt
	import cv2
	import numpy as np
	import vuba

For this example, we will use video specifically for the freshwater pond snail, *Radix balthica*:

.. ipython:: python

	video = hcv.load_example_video()

This returns a `vuba.Video`__ instance which we can read frames from like so:

__ https://vuba.readthedocs.io/en/latest/generated/vuba.Video.html#vuba.Video

.. ipython:: python

	# Grayscale frames are required for HeartCV
	frames = video.read(0, len(video), grayscale=True)

For a thorough overview of the capabilities of **vuba**, please refer to the corresponding documentation_.

.. _documentation: https://vuba.readthedocs.io/en/latest/.

The first step of HeartCV involves the computation of a mean pixel value blockwise grid for the video provided:

.. ipython:: python

	mpx = hcv.mpx_grid(frames, binsize=2)

Here we are binning each frame in the video by a factor of 2, i.e. 2 x 2 blocks of pixels are averaged. We can change this binning factor to as little or as much as we want - this will largely be dependent on what resolution you require and/or the speed you need for your application, respectively. 

Once we've computed our mean pixel value grid for our video, we can compute the energy proxy traits present within the footage. Energy proxy traits (EPTs) are essentially power spectra derived from mean pixel value time-series present within video of live biological material. They are a holistic measure of the observable phenotype that animals exhibit (see Tills et al., 2021). However, here we are using EPTs for a different purpose: localising regions of cardiac activity. Because mean pixel value time-series of cardiac regions are quasi-periodic, they can be detected at specific frequency bands when such time-series are transformed to the frequency domain. To compute EPTs from the mean pixel value grid above we can use the following:

.. ipython:: python
	
	ept = hcv.epts(mpx, video.fps)

The output of ``heartcv.epts`` is two ``numpy`` arrays corresponding to the frequency and power outputs from performing welch's method at each mean pixel value block. If we collapse these arrays to yield the total energy at each block, we get a heatmap coloured by the relative energy present within a block at all frequency bands: 

.. ipython:: python

	# Frequency range
	freq, power = ept
	frequencies = freq[..., 0, 0]
	fmin, fmax = frequencies.min(), frequencies.max()

	print(fmin, fmax)

	heatmap = hcv.spectral_map(ept, frequencies=(fmin, fmax))

	@savefig all_frequencies.png width=8in
	plt.imshow(heatmap, cmap='jet')

However, for most instances we would not want information from all frequency bands since cardiac activity is likely to only be reflected at specific frequencies. Consequently, we can use a bandpass filter to attenuate frequencies at which we would not ordinarily expect cardiac activity for the animal at hand like so:

.. ipython:: python

	# 2-6 Hz is generally where most cardiac activity can be observed in hippo stage Radix balthica
	heatmap = hcv.spectral_map(ept, frequencies=(2, 6))

	@savefig hr_frequencies.png width=8in
	plt.imshow(heatmap, cmap='jet')

Now that we've performed this bandpass filter, we find that we actually have only a single bright spot in the heatmap, corresponding to the heart. Because these heatmaps are at a resolution smaller than the original video, we need to resize them so that we can segment to the desired regions:

.. ipython:: python
	
	heatmap = cv2.resize(heatmap, video.resolution)

With our heatmap at the appropriate resolution, we can now perform segmentation via OTSU thresholding and contour filtering operations:

.. ipython:: python

	roi = hcv.detect_largest(heatmap)

This gives a polygon that is fit to the largest shape detected by OTSU thresholding, which in this case is the heart. However, for most applications it is preferable to segment to a bounding box. To convert this polygon to a bounding box we can simply do the following:

.. ipython:: python
	
	rectangle = vuba.fit_rectangles(roi, rotate=True)

	contour = cv2.boxPoints(rectangle)
	contour = np.int0(contour)

	first_frame = vuba.bgr(vuba.take_first(frames))
	vuba.draw_contours(first_frame, contour, -1, (0,255,0), 1)

	@savefig detected_roi.png width=8in
	plt.imshow(first_frame, cmap='jet')

Note that here we specified that the bounding box fit should be by minimum area, and thus will be rotated (rotate=True). This generally results in much better segmentation to the region of interest as most applications will not have the heart perfectly oriented. 

Now we can perform segmentation to this region using the following:

.. ipython:: python

	at_roi = np.asarray(list(hcv.segment(frames, contour)))

We can validate that this is indeed the heart using an orthogonal view of the segmented frames:

.. ipython:: python
	
	# Taken from: https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
	def get_sub_image(img, rect):
	    center, size, theta = rect
	    center, size = tuple(map(int, center)), tuple(map(int, size))
	    M = cv2.getRotationMatrix2D( center, theta, 1)
	    dst = cv2.warpAffine(img, M, img.shape[:2])
	    out = cv2.getRectSubPix(dst, size, center)
	    return out

	at_roi_sub = np.asarray([get_sub_image(frame, rectangle) for frame in frames])

	length, x, y = at_roi_sub.shape
	ix,iy = x // 2, y // 2

	x = at_roi_sub[:, ix, :]
	y = at_roi_sub[:, :, iy]

	fig, (ax1, ax2) = plt.subplots(2, 1)

	ax1.imshow(x.T, cmap='gray')
	ax1.set_title('Horizontal view')
	ax2.imshow(y.T, cmap='gray')
	ax2.set_title('Vertical view')

	@savefig orthogonal_view.png width=8in
	plt.draw()

As we can see there is a clear rhythmic signal in the data, very similar to the m-modes one finds from videos of hearts obtained through other techniques (e.g. `Fink et al., 2009`__).

__ https://www.future-science.com/doi/full/10.2144/000113078?rfr_dat=cr_pub++0pubmed&url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org

Now that we've localised the cardiac region, the next step is to extract a signal that enables us to quantify when heart beats occur. In HeartCV, we do this by collapsing the segmented images above to a one dimensional vector by averaging each segmented frame, creating a mean pixel value time-series:

.. ipython:: python

	v = at_roi.mean(axis=(1, 2))
	time = np.asarray([i/video.fps for i in range(len(v))])

	plt.plot(time, v, 'k')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Mean pixel value (px)')

	@savefig mpx.png width=8in
	plt.draw()

Because this signal is oscillatory in nature, we can leverage a multitude of peak detection methods to retrieve the peaks that correspond to a heart beat. We've found that automatic multiscale peak detection (AMPD_) to perform particularly well on such data and so it is the one we provide with HeartCV: 

.. _AMPD: https://www.mdpi.com/1999-4893/5/4/588

.. ipython:: python

	v = v.max() - v # invert signal
    v = np.interp([i/3 for i in range(len(v)*3)], np.arange(0, len(v)), v) # upsample by a factor of 3 to improve peak detection

    peaks = hcv.find_peaks(v)

    time = np.asarray([i/(video.fps*3) for i in range(len(v))])

    plt.plot(time, v, 'k')
    plt.plot(time[peaks], v[peaks], 'or')

	plt.xlabel('Time (seconds)')
	plt.ylabel('Mean pixel value (px)')

	@savefig detected_peaks.png width=8in
	plt.draw()

Note that we invert and upsample the mean pixel value signal, this both improves peak detection performance but has also provided much more accurate results in comparison to manual quantification. 

We can now use these peaks to compute various metrics of cardiac function as follows:

.. ipython:: python

	# Beat to beat intervals (seconds)
	hcv.b2b_intervals(peaks, video.fps*3)

	# Various cardiac statistics
	hcv.stats(peaks, len(video)*3, video.fps*3)