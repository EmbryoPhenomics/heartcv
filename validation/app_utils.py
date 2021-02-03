import cv2
import time
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import base64
from dataclasses import dataclass
import numpy as np
import os

import vuba

# Text defaults for opencv
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255,255,255)
lineType = 2

@dataclass
class DataStore:
    '''Backend container that holds data from various ui. '''
    __slots__ = ['ds_frame', 'ds_area', 'ss_frame', 'ss_area']
    ds_frame: list
    ds_area: list
    ss_frame: list
    ss_area: list

class VideoStore:
    '''Backend container for holding video capture instances. '''
    def __init__(self):
        self.raw_video = None
        self.hcv_video = None
        self.current_frame = None

    def add_raw(self, raw_video):
        (self.raw_video, self.raw_video_release) = vuba.open_video(raw_video)

    def add_hcv(self, hcv_video):
        (self.hcv_video, self.hcv_video_release) = vuba.open_video(hcv_video)

    def close(self):
        def release(video, video_release):
            if video_release:
                video.release()

        map(release, (self.raw_video, self.hcv_video))

class ContourStore:
    '''
    Backend container that holds contours and their corresponding frames. Used 
    primarily for exporting both manual and heartcv frames for later comparison.
    '''
    def __init__(self):
        self.contour_info = []
        self.contour_fr = []
        self.raw_video = None
        self.raw_video_release = False
        self.hcv_video = None
        self.hcv_video_release = False

    def refresh(self):
        self.contour_info = []
        self.contour_fr = []
        self.raw_video = None
        self.raw_video_release = False
        self.hcv_video = None
        self.hcv_video_release = False

    def add_raw(self, raw_video):
        (self.raw_video, self.raw_video_release) = vuba.open_video(raw_video)
        frame = self.raw_video.read(index=0)
        x,y,_ = frame.shape
        self.full = (x,y)

    def add_hcv(self, hcv_video):
        (self.hcv_video, self.hcv_video_release) = vuba.open_video(hcv_video)

    def add_contour(self, event, frame, contour):
        if event and frame and contour.all():
            if frame in self.contour_fr:
                ind = self.contour_fr.index(frame)
                self.contour_info[ind] = (event,frame,contour)
            else:
                self.contour_info.append((event, frame, contour))

    def export(self, outpath):
        for (e,fr,c) in self.contour_info:
            raw_f = self.raw_video.read(index=fr)
            vuba.draw_contours(raw_f, c, -1, (0,0,255), 1)

            hcv_f = self.hcv_video.read(index=fr)
            both = np.hstack((raw_f, hcv_f)) 
            cv2.imwrite(os.path.join(outpath, f'{fr}_{e}.png'), both)

if os.name == 'posix':
    class Keys:
        '''Key codes for keyboard keys (as tested on linux). '''
        right = 83
        left = 81
        d = 100
        s = 115
        space = 32
        esc = 27
elif os.name == 'nt':
    class Keys:
        '''Key codes for keyboard keys (as tested on Windows). '''
        right = 46
        left = 44
        d = 100
        s = 115
        space = 32
        esc = 27
else:
    raise OSError('Operating system not currently supported, currently only Windows 10 and Linux are supported.')

def frameLabeler(filename, queue):
    '''
    GUI for labelling frames that correspond to various stages in the
    cardiac cycle.
    Note that the keys bound here may have different codes between systems,
    so this might have to be changed on a system by system basis. 
    Keyword arguments:
        filename    String.          Filename to process (Required).
        queue       Multiprocessing  Queue for appending with data from the     
                    queue object.    GUI (Required).
    '''

    video = vuba.Video(filename)
    frame_count = len(video)
    fps = video.fps
    window_title = 'Frame recorder'
    frames = list(range(frame_count))

    def display(val):
        frame = video.read(index=val)
        cv2.putText(
            frame, 'Frame: {}'.format(val), 
            tuple(map(int, (0.025*frame.shape[1], 0.975*frame.shape[0]))), font, fontScale, fontColor, lineType)

        cv2.imshow(window_title, frame)

    def stream(val):
        _current = val
        for f in range(val,frame_count):
            key = cv2.waitKey(1)

            if key is Keys.space:
                break
            else:
                display(f)
                _current = f
                time.sleep(1/fps)

        return _current

    current = 0
    diastoleFrameRecord = []
    systoleFrameRecord = []

    display(current)
    while True:
        key = cv2.waitKey(1)

        val = 0
        if key is Keys.right:
            if current < (frame_count-1):
                val = 1
        elif key is Keys.left:
            if current > 0:
                val = -1
        elif key is Keys.d:
            diastoleFrameRecord.append(current)
        elif key is Keys.s:
            systoleFrameRecord.append(current)
        elif key is Keys.space:
            current = stream(current)
        elif key is Keys.esc:
            break

        if val != 0:
            current += val
            display(current)
        else:
            pass

        time.sleep(0.01)

    video.close()
    queue.put((diastoleFrameRecord, systoleFrameRecord))

def interactiveImage(id, img):
    '''
    Create an interactive plotly graph from an image.
    Keyword arguments:
        id    String.         Component id for use in other callbacks (Required).
        img   Numpy.ndarray   Image to render (Required).
    Returns:
        dash_core_components.Graph component containing the supplied image.
    '''
    _, img_np = cv2.imencode('.png', img)
    byteStr = img_np.tobytes()

    encodedImg = base64.b64encode(byteStr)
    height, width = img.shape[0], img.shape[1]

    source = 'data:image/png;base64,{}'.format(encodedImg.decode())

    return dcc.Graph(
        id=id,
        figure={
            'data': [],
            'layout': {
                'margin': go.layout.Margin(l=0, b=0, t=0, r=0),
                'xaxis': {
                    'range': (0, width),
                    'showgrid': False,
                    'zeroline': False,
                    'visible': True,
                    'scaleanchor': 'y',
                    'scaleratio': 1
                },
                'yaxis': {
                    'range': (0, height),
                    'showgrid': False,
                    'zeroline': False,
                    'visible': True
                },
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': 0,
                    'y': 0,
                    'yanchor': 'bottom',
                    'sizing': 'stretch',
                    'sizex': width,
                    'sizey': height,
                    'layer': 'below',
                    'source': source,
                }],
                'dragmode': 'lasso',
            }
        },
        config={
            'modeBarButtonsToRemove': [
                'sendDataToCloud',
                'autoScale2d',
                'toggleSpikelines',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'zoom2d'
            ],
            'displaylogo': False
        },
        style={
            'width': '900px', 
            'height': '800px'}
    )

def blank():
    '''Convenience function to return a blank image. '''
    img = cv2.imread('./assets/blankImg.png')

    return html.Div(children=[
        interactiveImage(
            id='still-image-graph', 
            img=img)
        ])