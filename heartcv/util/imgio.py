import cv2
import glob
import os
import numpy as np
from natsort import natsorted, ns
from more_itertools import prepend
import math

from heartcv import util

def take_first(it):
    '''
    Retrieve a value from an iterable object.

    Keyword arguments:
        it    Iterable object (Required).

    Returns:
        First value from an iterable object.

    '''

    it = iter(it)
    try:
        return next(it)
    except StopIteration:
        return None

def open_video(video):
    '''Convenience function for creating a heartcv.Video instance. '''
    release = False
    if not isinstance(video, Video):
        video = Video(video)
        release = True
    
    return video, release

class Frames:
    '''
    Container for frames that is used in the Video class for supplying to
    other methods in the lib.

    '''
    def __init__(self, reader, start, stop, step, grayscale):
        ''' 
        Keyword arguments:
            reader       Callable.    Frame reader method supplied from Video class.

            start        Int.         Index of first frame.

            stop         Int.         Index of last frame.

            step         Int.         Step size.

            grayscale    Boolean.     Whether to convert frames to grayscale.

        '''
        self._reader = reader
        self.idxs = (start, stop, step)
        self.grayscale = grayscale
        self.in_mem = False
        self._roi = None

    def raise_mem_exc(self):
        if not self.in_mem:
            raise TypeError('Can only retrieve values using array slicing when the frames are imported into memory.')

    def __getitem__(self, shape):
        self.raise_mem_exc()
        return self._frames[shape]

    def __setitem__(self, shape, data):
        self.raise_mem_exc()
        self._frames[shape] = data

    def import_to_nparray(self):
        '''
        Import the declared frames into a contiguous numpy array.

        '''
        gen_fr = self._reader(*self.idxs, self.grayscale)
        first = take_first(gen_fr)
        self._frames = np.ascontiguousarray(np.empty((len(self), *first.shape), dtype='uint8'))
        gen_fr = prepend(first, gen_fr)

        with util.pgbar(total=len(self)) as pgbar:      
            for i,frame in enumerate(gen_fr):
                self._frames[i,:] = frame[:]
                pgbar.update(1)

        self.in_mem = True

    def subset(self, x, y, w, h):
        '''
        Filter all footage to a given roi (x,y,w,h).

        '''
        if self.in_mem:
            self._frames = self._frames[:, y:y+h, x:x+w, ...]
        else:
            self._roi = (x,y,w,h)

    def _subset_frame(self, frame, x, y, w, h):
        '''Subset a frame to a given roi. '''
        return frame[y:y+h, x:x+w, ...]

    def __len__(self):
        '''
        Retrieve the length of the provided frames without having to iterate
        across them.
    
        '''
        (start, stop, step) = self.idxs
        return math.floor((stop-start)/step)

    def __iter__(self):
        '''
        Iterator for retrieving the declared frames.

        '''
        if self.in_mem:
            for frame in self._frames:
                yield frame
        else:
            for frame in self._reader(*self.idxs, self.grayscale):
                if self._roi:
                    yield self._subset_frame(frame, *self._roi)
                else:
                    yield frame

class Video:
    '''
    Wrapper around various image readers and writers that provides a simple API
    to achieve the same functions regardless of format. 
    
    '''
    def __init__(self, video):
        '''
        Keyword arguments:
            video      String or         Full filename or VideoCapture object to
                       VideoCapture or   a video (e.g. AVI or MP4), or a glob string
                       glob string.      to a series of individual images. Note that
                                         all individual filenames will be sorted prior
                                         to being read if a glob string is supplied. 
        '''
        def open_videocv(video):
            self.video = video
            self.video_release = False
            if not isinstance(self.video, cv2.VideoCapture):
                self.video = cv2.VideoCapture(self.video)
                self.video_release = True
            self.filenames = None

        try:
            # Somewhat hacky but provides consistent behaviour
            if '*' in video:
                files = glob.glob(video)
                self.filenames = natsorted(files, alg=ns.IGNORECASE)
                self.video = self.video_release = None
            else:
                open_videocv(video)
        except TypeError:
            open_videocv(video)
        except:
            raise

    def _to_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def __len__(self):
        '''
        Retrieve the length of the provided footage without having to iterate
        across it.
    
        '''
        if self.video:
            return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return len(self.filenames)

    def read(self, *args, **kwargs):
        '''
        Read single or multiple frames from the provided footage.

        Note that reading multiple frames follows slice behaviour, 
        whereby there is a 'start', 'stop' and 'step'. However steps will
        only give a performance uplift for reading images stored as 
        individual files (e.g. multiple pngs). If you are reading from 
        videos, the read time will be the same as if you were reading 
        all the frames between 'start' and 'stop'. This is because it is
        faster to pass on frames that don't match the step size, than it is
        to repeatedly locate the correct frames by index in the video file.

        If you don't supply an argument to this method, all frames will be
        read from the supplied video footage and a generator object will be 
        created for iterating through them. 

        Keyword arguments:
            Single frames:
                index    Int.    Index of frame in footage to read.

            Multiple frames:
                start       Int.    Index of frame to start reading from.
    
                stop        Int.    Index of frame to stop reading at.
    
                step        Int.    Step size.
    
                low_memory  Bool.   Whether to import frames into RAM, default is
                                    True i.e. not to.

        Returns:
            Single frames:
                Numpy.ndarray.    Frame at the given index.

            Multiple frames:
                HeartCV.Frames.   Container that will either supply the frames from 
                                  a Numpy.ndarray or from a generator. Note that this
                                  container contains both a len and iter method. For the 
                                  latter, a new generator is created upon calling the 
                                  method if the frames have not been imported into memory. 
                                  This is to maintain implementation parity with the 
                                  in-memory container.

        '''
        try:
            frame = self._read_single(*args, **kwargs)
            return frame
        except TypeError:
            low_mem = kwargs.pop('low_memory', True)            
            args = self._prep_args_kwargs(*args, **kwargs)
            frames = Frames(self._read_multi, *args)

            if not low_mem:
                util.hcv_logger.info('Importing frames into memory...')
                frames.import_to_nparray()

            return frames
        except:
            raise

    def _prep_args_kwargs(self, start=0, stop=None, step=1, grayscale=True):
        '''
        Prep any args and kwargs supplied from Video.read(...).

        '''
        def raise_ind_exc(index):
            if index:
                if index < 0 or index > len(self):
                    raise IndexError('Indices out of available range for provided footage.')

        raise_ind_exc(start)
        raise_ind_exc(stop)

        if not stop: stop = len(self)

        return (start, stop, step, grayscale)

    def _read_single(self, index, grayscale=True):
        '''
        Read a single frame at a given location in the requested footage. 

        '''
        if self.video:
            self.video.set(1,index)
            success, frame = self.video.read()
        else:
            frame = cv2.imread(self.filenames[index])

        if grayscale:
            frame = self._to_gray(frame)

        return frame

    def _read_multi(self, start, stop, step, grayscale):
        '''
        Read multiple frames from the requested footage at the given indices.

        '''
        if self.video:
            self.video.set(1, start)

            if step:
                if step > 1:
                    # Floor is used here to stop out-of-range indices being created
                    number = math.floor((stop-start)/step) 
                    frames_to_yield = [round((step*n)+start) for n in range(number)]
                    steps = True
                else:
                    steps = False

            for fr in range(start,stop):
                success, frame = self.video.read()
                if not success:
                    break

                if grayscale:
                    frame = self._to_gray(frame)

                if steps:
                    if fr in frames_to_yield:
                        yield frame
                    else:
                        pass
                else:
                    yield frame
        else:
            for fn in self.filenames[slice(start, stop, step)]:
                frame = cv2.imread(fn)
                if grayscale:
                    frame = self._to_gray(frame)
                yield frame         

    def close(self):
        '''
        Close attached video handlers (if any).

        '''
        if self.video:
            self.video.release()
        elif self.filenames:
            self.filenames = None

class Writer:
    '''
    Create a writer for exporting frames at a given path.

    Note that this will create an OpenCV VideoWriter regardless of the input
    image formats (individual images or avi's for e.g.). 

    '''
    def __init__(self, footage, path, fps=None, resolution=None, fourcc=None, grayscale=False):
        '''
        Keyword arguments:
            footage      HeartCV.Video   Instance of Video to create writer based on (Required).

            path         String.         Path to export frames (Required).

            fps          Float.          Framerate to export footage at, default is the fps of 
                                         the video supplied. Note that this argument must be supplied
                                         if working with individual images.

            resolution   Tuple.          Width and height to export footage at (both must be supplied 
                                         as integers). Default is the resolution of the footage supplied.

            fourcc       String.         Fourcc codec to encode footage with. Default is to encode with MJPG
                                         codec.

            grayscale    Boolean.        Whether to export footage as grayscale. Default is False.


        '''
        footage, self.release = open_video(footage)
        self.valid_writer = True

        isColor = True
        if grayscale: isColor = False
        if not fourcc: 
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*fourcc)

        if footage.video:
            if not fps: fps = int(footage.video.get(cv2.CAP_PROP_FPS))
            if not resolution:
                width = int(footage.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(footage.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resolution = (width, height)
            else:
                self.resolution = resolution
        else:
            if not fps:
                raise ValueError('When working with individual images you must supply a framerate to export footage at.')
            if not resolution:
                frame = footage.read(0)
                self.resolution = (frame.shape[1], frame.shape[0])

        self.cvwriter = cv2.VideoWriter(path, fourcc, fps, self.resolution, isColor)

    def write(self, frame):
        '''
        Write a frame using the declared writer. Note that if the frames supplied for encoding
        are not at the same resolution as that set for the Writer upon initiation, the frames
        will be resized accordingly. 

        Keyword arguments:
            frame.    Numpy.ndarray.    Frame to export (Required).

        '''
        size = (frame.shape[1], frame.shape[0])
        if size != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        self.cvwriter.write(frame)

    def close(self):
        '''Close/cleanup declared writers. '''
        self.cvwriter.release()
        if self.release: self.footage.close()