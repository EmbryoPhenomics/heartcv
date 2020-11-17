import cv2
import numpy as np
from heartcv import util
from heartcv.gui import base
from heartcv.util import hcv_logger
from heartcv.util import contourfilters as cf

small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

def _segment(img, mask, inv, contour_method, thresh_val, e_iter, d_iter):
    '''
    Main method for performing segmentation to retrieve the heart in an image.

    Note this should not be called separately, instead address it through the
    Segmentation class.

    Keyword argumnets:
        img                 Numpy ndarray.    Grayscale image to segment (Required).
        
        mask                Numpy ndarray.    Mask to image footage to (Required).
        
        inv                 Boolean.          Whether to invert the image (Required).
        
        contour_method      Callable.         Function to filter contours (Required).
        
        thresh_val          Int or float.     Binary threshold value (Required).
        
        e_iter              Int.              Number of small elliptical erosions 
                                              to apply (Required).

        d_iter              Int.              Number of small elliptical dilations
                                              to apply (Required).

    Returns:
        List or numpy ndarray.    Contours found in the segmented image according to 
                                  contour_method.

        Numpy ndarray.            Mask based on the filtered contours above for subsequent
                                  segmentations.

    '''
    if inv:
        img = cv2.bitwise_not(img)

    masked = cv2.bitwise_and(img, img, mask=mask)
    _, thresh = cv2.threshold(masked, thresh_val, 255, cv2.THRESH_BINARY)

    morphed = thresh
    if e_iter > 0:
        morphed = cv2.erode(morphed, small_ellipse, iterations=e_iter)
    if d_iter > 0:
        morphed = cv2.dilate(morphed, small_ellipse, iterations=d_iter)

    contours, hierarchy = util.find_contours(morphed)
    
    try:
        try:
            filtContours = contour_method(contours, hierarchy)
        except TypeError:
            filtContours = contour_method(contours)
        mask = util.contour_mask(img, filtContours)
    except:
        filtContours = None 

    return filtContours, mask

def _cast_contours(contours, x, y):
    '''Convenience function to cast contour(s) to a given x,y. '''
    if isinstance(contours, list):
        for i,c in enumerate(contours):
            c[:,0,0] += x
            c[:,0,1] += y
            contours[i] = c
    else:
        contours[:,0,0] += x
        contours[:,0,1] += y
    return contours

def _minmax_scale(vals):
    '''Convenience function for performing min/max normalization. '''
    m = np.nanmin(vals)
    M = np.nanmax(vals)
    return (vals-m)/(M-m)

def _segmentation_gui(video, main_mask, layers, inv, contour_method, bbox, coords, size):
    '''
    Launch a gui for testing a mask for the heart.

    Note this should not be called separately, instead address it through the
    Segmentation class.

    Keyword arguments:
        video             HeartCV.Video.         HeartCV.Video object to the video to test (Required).

        main_mask         Numpy.ndarray.         Main mask derived from location and activity
                                                 segmentation (Required).

        layers              List.                  User provided layers, can be None if none have
                                                 have been declared (Required).

        inv               Boolean.               Whether to invert incoming frames (Required).

        contour_method    Callable.              Contour filter to apply to those detected in
                                                 the final mask (Required).

        bbox              Tuple.                 Bounding box dimensions to area of most activity 
                                                 found using sum_abs_diff(). Note that all footage
                                                 will be filtered to this to reduce computational 
                                                 load (Required).    

        coords            Tuple.                 Coordinates and radius to area of most activity
                                                 found using max_optic_flow(), default is None.

        size              Tuple.                 Size of interactive window (Required). 

    Returns:
        Dict.    Latest layer based on the last trackbar values and any user provided variables.
                 This should be added to Segmentation through Segmentation.add_layer(layer_to_add) so that
                 it is used in subsequent segmentations.

    '''
    x,y,w,h = bbox
    def create_layer(gui):
        layer = dict(inv=inv, contour_method=contour_method)
        for key,val in gui.values().items():
            if key is not 'frames':
                layer[key] = val
        return layer    

    gui = base.VideoGUI(video=video, title='Heart area GUI', size=size)

    @gui.main_process
    def segment(gui):
        frame = gui.frame.copy()
        frameProc = util.gray(frame)
        at_roi = frameProc[y:y+h, x:x+w]

        mask = main_mask
        if len(layers):
            for layer in layers:
                contours, mask = _segment(at_roi, mask, **layer)

        latest_layer = create_layer(gui)
        contours, mask = _segment(at_roi, mask, **latest_layer)

        try:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)
            if coords:
                x_,y_,r = coords
                cv2.circle(frame, (x_,y_), r, (255,0,0), 1)

            contours = _cast_contours(contours, x, y)
            util.draw_contours(frame, contours, -1, (0,0,255), 1)
        except:
            pass 

        return frame

    @gui.trackbar('Threshold', id='thresh_val', min=0, max=255)
    def on_thresh(gui, val):
        gui['thresh_val'] = val
        img = gui.process()
        cv2.imshow(gui.title, img)

    @gui.trackbar('Erosions', id='e_iter', min=0, max=20)
    def on_erode(gui, val):
        gui['e_iter'] = val
        img = gui.process()
        cv2.imshow(gui.title, img)

    @gui.trackbar('Dilations', id='d_iter', min=0, max=20)
    def on_dilate(gui, val):
        gui['d_iter'] = val
        img = gui.process()
        cv2.imshow(gui.title, img)

    gui.run()

    return create_layer(gui)

class Segmentation:
    '''
    Class for iteratively segmenting footage to a trackable heart component.

    '''
    def __init__(self, video, contour, bbox, coords=None):
        '''
        Keyword arguments:
            video      HeartCV.Video.         HeartCV.Video object to the video to test (Required).

            contour    Numpy.ndarray.         Contour of embryo to initially segment to (Required).  

            bbox       Tuple.                 Bounding box dimensions to area of most activity 
                                              found using sum_abs_diff(). Note that all footage
                                              will be filtered to this to reduce computational 
                                              load (Required).    

            coords     Tuple.                 Coordinates and radius to area of most activity
                                              found using max_optic_flow(), default is None.

        '''
        self.video, self.release = util.open_video(video)
        self.contour = contour
        self.bbox = bbox
        self.layers = []

        self.first = self.video.read(0)
        embryoMask = util.contour_mask(self.first, contour)
       
        self.coords = coords
        if self.coords:
            self.coords = tuple(map(int, coords))
            roiMask = util.circle_mask(self.first, *self.coords)
        else:
            roiMask = util.rect_mask(self.first, bbox)

        self.main_mask = cv2.bitwise_and(roiMask, roiMask, mask=embryoMask)
        x,y,w,h = self.bbox
        self.main_mask = self.main_mask[y:y+h, x:x+w]

    def gui(self, inv, contour_method, size=None):
        '''
        Launch a gui for testing a combination of layers. 

        Note that the segmentation performed in this gui is based on all the
        previously declared layers, and then the latest layer based on the supplied variables.
        Because of this, you will need to call this method each time you'd like to create
        a new segmentation layer. 

        Keyword arguments:
            inv              Boolean.   Whether to invert the image (Required).
            
            contour_method   Callable.  Function to filter contours (Required).

            size             Tuple.     Size of interactive window, default 
                                        is the size of the first image in the video.
        
        Returns:
            Dict.    Latest layer based on the last trackbar values and any user provided variables.
                     This should be added to Segmentation through Segmentation.add_layer(layer_to_add) so that
                     it is used in subsequent segmentations.

        '''

        if not size:
            size = np.swapaxes(self.first,0,1).shape # required to get correct dims for cv2.resize()

        return _segmentation_gui(self.video, self.main_mask, self.layers, inv,
                                 contour_method, self.bbox, self.coords, size)

    def add_layer(self, layer):
        '''
        Add a layer to Segmentation obtained from Segmentation.gui(...)

        Keyword arguments:
            layer    Dict.    Dictionary of keyword arguments describing the layer to add (Required).

        '''
        self.layers.append(layer)

    def optimise(self, frames, layer, thresh_range=(0,255), erosions_range=(0,20), dilations_range=(0,20)):
        '''
        Optimise with provided layers to objective of mean trend. Note that it is currently only
        supported for single layers. (Unfinished).

        '''
        hcv_logger.info('Computing ideal trend to optimise for...')
        mean_pxvals = []
        with util.pgbar(len(frames)) as pgbar:
            for frame in frames: 
                if len(frame.shape) > 2:
                    frame = util.gray(frame)
                mean_pxvals.append(frame.mean())
                pgbar.update(1)

        # Invert to get trend that will match heart area output
        # Normalize to get uniform scale for optimization
        mean_pxvals = np.asarray(mean_pxvals)
        mean_pxvals = _minmax_scale(np.nanmax(mean_pxvals) - mean_pxvals)

        combinations_totest = []

        # For grid-search
        for t in range(*thresh_range): # Threshold values
            for e in range(*erosions_range): # Erosion iterations
                for d in range(*dilations_range): # Dilation iterations
                    inv = layer['inv']
                    method = layer['contour_method']
                    new_layer = dict(inv=inv, contour_method=method, thresh_val=t, e_iter=e, d_iter=d)

                    new_layers = new_layer
                    if len(self.layers) >= 2:                    
                        new_layers = self.layers
                        new_layers[new_layers.index(layer)] = new_layer

                    combinations_totest.append(new_layers)

        results = []
        hcv_logger.info('Finding optimal combination with grid-search...')
        with util.pgbar(len(combinations_totest)) as pgbar:
            for comb in combinations_totest:
                areas = []
                for frame in frames:
                    if len(frame.shape) > 2:
                        frame = util.gray(frame)

                    mask = self.main_mask
                    contours, mask = _segment(frame, mask, **comb)

                    try:
                        _areas = cf._contours_area(contours)
                        try:
                            areas.append(sum(_areas))
                        except TypeError:
                            areas.append(_areas)
                    except:
                        areas.append(np.nan)

                if np.nan not in areas:
                    areas = np.asarray(areas)
                    norm_areas = _minmax_scale(areas)
                    diff = mean_pxvals - norm_areas

                    # Here we use RMS to get a measure of both mean difference and the magnitude of differences
                    results.append(np.sqrt(np.nanmean(diff**2))) 
                else:
                    results.append(np.nan)
                pgbar.update(1)

        # Find the value closest to zero as this will be the optimimum with 
        # respect to the mean trend.
        results = np.asarray(results)
        opt_idx = np.nanargmin(results)
        opt_layer = combinations_totest[opt_idx]
        return opt_layer, opt_idx, results

    def calc(self, frames, writer):
        '''
        Perform segmentation based on the provided layers.
        
        Note this exports both the visual output for later comparisons/validation and
        also returns area obtained for each frame. If the supplied writer is None
        then no visual output will be created.

        Keyword arguments:
            frames    List.              List of rgb frames (Required).

            writer    HeartCV.Writer.    Writer to export visual output with (Required).

        Returns:
            List.    Area values found via segmentation. 

            List.    Mean pixel values at the bounding box dimensions.

        '''
        if not len(self.layers):
            raise HeartCVError('Cannot calculate heart area with no declared layers.')

        hcv_logger.info('Computing heart area...')

        x,y,w,h = self.bbox
        areas = []
        contours_all = []
        with util.pgbar(len(frames)) as pgbar:
            for frame in frames:
                if len(frame.shape) > 2:
                    frame = util.gray(frame)

                mask = self.main_mask
                for layer in self.layers:
                    contours, mask = _segment(frame, mask, **layer)

                try:
                    _areas = cf._contours_area(contours)
                    try:
                        areas.append(sum(_areas))
                    except TypeError:
                        areas.append(_areas)
 
                     # Visual output ---------
                    contours_all.append(_cast_contours(contours, x, y))
                except:
                    contours_all.append(None)
                    areas.append(np.nan)

                pgbar.update(1)

        if writer:
            hcv_logger.info('Exporting visual output...')
            with util.pgbar(len(frames)) as pgbar:
                for i,frame in enumerate(self.video.read(*frames.idxs, grayscale=False)):
                        util.draw_contours(frame, contours_all[i], -1, (0,0,255), 1)

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)
                        if self.coords:
                            x,y,r = self.coords
                            cv2.circle(frame, (x,y), r, (255,0,0), 1)     

                        writer.write(frame)
                        pgbar.update(1)

            writer.close()
        if self.release: self.video.close()

        return areas

