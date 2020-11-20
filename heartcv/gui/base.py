import functools
import cv2
from typing import Callable
from dataclasses import dataclass

from heartcv import util

@dataclass
class TrackbarMethod:
    '''Container for a trackbar method and it's associated variables. '''
    __slots__ = ['id', 'min', 'max', 'method', 'current']
    id: str
    min: int or float
    max: int or float
    method: Callable
    current: int or float

class BaseGUI:
    '''
    Class for creating interfaces for image manipulation. 

    '''
    def __init__(self, title, size):
        '''
        Keyword arguments:
            title    String.          Title of the interface window (Required).

            size     Tuple.           Size of the interactive window in pixels (Required).
        
        Returns:
            BaseGUI    Class object for creating the interactive window.

        '''
        self.title = title
        self.size = size

        self.trackbars = {}
        cv2.namedWindow(self.title)

    def values(self):
        '''
        Retrieve all current trackbar values from the gui. Returns
        None if no trackbars declared. 

        '''
        if self.trackbars:
            vals = {}
            for k in self.trackbars: 
                vals[k] = self.trackbars[k].current
        else:
            vals = None
        return vals

    def __getitem__(self, key):
        '''Getter for retrieving a trackbar's current value. '''
        return self.trackbars[key].current

    def __setitem__(self, key, val):
        '''Setter for changing a trackbar's current value. '''
        self.trackbars[key].current = val

    def main_process(self, func):
        ''' 
        Add a main processing function to be executed on every call.
        
        Note that this is usually called through the use of a decorater,
        for e.g. @gui.main_process
                 def function(gui):
                    # Various image processing
                    return image(s)

        All trackbar functions should call the function that is wrapped
        in this decorator. As such changes made to a single trackbar, will
        propagate to the main processing function and the changes will be 
        returned to the named window. Note that this function will return silently.

        Keyword arguments:
            func    Callable.    Main processing function of the interface (Required).

        '''
        def wrap_to_proc():
            img = func(self)
            return cv2.resize(img, self.size)

        self.process = wrap_to_proc
        return wrap_to_proc

    def trackbar(self, name, id, min, max):
        '''
        Add a trackbar to the interactive window. 

        Just like the above, this is usually called through the use 
        of a decorator. Note that this function will return silently.

        Keyword arguments:
            name    String.    Name of trackbar to add (Required).

            id      String.    Id of trackbar to add. This will be the key
                               associated with the trackbar (Required).

            min     Integer.   Minimum limit of trackbar (Required).

            max     Integer.   Maximum limit of trackbar (Required).

        '''
        def wrap_to_trackbar(func):
            @functools.wraps(func)
            def on_exe(val):
                return func(self, val)

            self.trackbars[id] = TrackbarMethod(id, min, max, on_exe, min)
            cv2.createTrackbar(name, self.title, min, max, on_exe)

            return on_exe
        return wrap_to_trackbar

    def run(self):        
        '''
        Launch the interface 

        Note that you can access any variables from the class that you
        added/manipulated through the trackbars attribute. This contains 
        a dict of the trackbars, with each key containing associated 
        variables in a dataclass:
            e.g. gui = BaseGUI(...)
                 ...
                 @gui.trackbar('Threshold', 0, 255)
                 def on_threshold(gui, val):
                    ...
                 ...
                 gui.run()

                 # Retrieve all associated vars
                 name, min, max, method, current = gui.trackbars['Threshold']

                 # Only the current value
                 current = gui['Threshold']

        '''
        # Execute first method to launch the gui
        firstfunc = self.trackbars[[*self.trackbars][0]]
        func = firstfunc.method
        min = firstfunc.min
        func(min)

        cv2.waitKey()
        cv2.destroyAllWindows()   


class FrameGUI(BaseGUI):
    '''
    Class for creating interfaces for individual image manipulation. 

    '''
    def __init__(self, frame, *args, **kwargs):
        '''
        Keyword arguments:
            frame    Numpy ndarray    Frame(s) to manipulate within the interface (Required).

            title    String.          Title of the interface window (Required).

            size     Tuple.           Size of the interactive window in pixels (Required).
        
        Returns:
            HeartCV.FrameGUI    Class object for creating the interactive window.

        '''
        self.frame = frame
        super(FrameGUI, self).__init__(*args, **kwargs)

class FramesGUI(BaseGUI):
    '''
    Class for creating interfaces for manipulating a sequence of frames.

    Note that any video gui will always have a frame trackbar and corresponding callback
    for scrolling through the frames. 

    '''
    def __init__(self, frames, *args, **kwargs):
        '''
        Keyword arguments:
            frames   List or          Images to manipulate within the interface (Required).
                     Numpy ndarray.                

            title    String.          Title of the interface window (Required).

            size     Tuple.           Size of the interactive window in pixels (Required).
        
        Returns:
            HeartCV.FramesGUI    Class object for creating the interactive window.

        '''
        self.frames = frames
        super(FramesGUI, self).__init__(*args, **kwargs)

        # Create frame-reader trackbar
        len_frames = len(self.frames)
        cv2.createTrackbar('Frames', self.title, 0, len_frames, self.read)
        self.trackbars['frames'] = TrackbarMethod('frames', 0, len_frames, self.read, 0)

    def read(self, val):
        '''
        Callback for reading and displaying a frame from the provided frames.

        Any image processing in the main_process method will be executed prior 
        to displaying the frame.
        
        Keyword arguments:
            val    Integer.    Frame id in the requested video.

        '''
        self.frame = self.frames[val]
        frameProc = self.process()
        cv2.imshow(self.title, frameProc)


class VideoGUI(BaseGUI):
    '''
    Class for creating interfaces for video manipulation. 

    Note that any video gui will always have a frame trackbar and corresponding callback
    for scrolling through the footage. 

    '''
    def __init__(self, video, *args, **kwargs):
        '''
        Keyword arguments:
            video    HeartCV.Video.         HeartCV.Video object to the video to test (Required).

            title    String.                Title of the interface window (Required).

            size     Tuple.                 Size of the interactive window in pixels (Required).       

        Returns:
            HeartCV.VideoGUI    Class object for creating the interactive window.

        '''
        self.video, self.release = util.open_video(video)
        super(VideoGUI, self).__init__(*args, **kwargs)
        
        # Create frame-reader trackbar
        cv2.createTrackbar('Frames', self.title, 0, len(self.video), self.read)
        self.trackbars['frames'] = TrackbarMethod('frames', 0, len(self.video), self.read, 0)

    def read(self, val):
        '''
        Callback for reading and displaying a frame from the requested video.

        Any image processing in the main_process method will be executed prior 
        to displaying the frame.
        
        Keyword arguments:
            val    Integer.    Frame id in the requested video.

        '''
        self.frame = self.video.read(val, grayscale=False)
        frameProc = self.process()
        cv2.imshow(self.title, frameProc)

    def run(self):
        '''
        Launch the interactive video interface.

        '''
        try:
            super().run()
        finally:
            if self.release:
                self.video.close()
