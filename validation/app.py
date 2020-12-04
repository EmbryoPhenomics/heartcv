import cv2
import multiprocessing as mp
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import os
import numpy as np
import pandas as pd
from flask import request

import cvu
import app_utils, app_layout

external_stylesheets = ['./assets/app.css'] # courtesy of dadamson
app = dash.Dash('heartcv_validation_app', external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = app_layout.app_layout()

# Python session containers used instead of passing around data through html divs
dstore = app_utils.DataStore([0],[0],[0],[0])
vstore = app_utils.VideoStore()
cstore = app_utils.ContourStore()

# For callbacks where no return is needed
trigger = {'dummydata': True}

@app.callback(
    Output('filepath-check', 'children'),
    [Input('filename-input', 'value')])
def check_raw_path(path):
    if path:
        if not os.path.exists(path):
            return 'Specified path does not exist.'
        else:
            cstore.refresh()
            vstore.add_raw(path)
            cstore.add_raw(vstore.raw_video) 
            return True

@app.callback(
    Output('hcv-filepath-check', 'children'),
    [Input('hcv-filename-input', 'value')])
def check_hcv_path(path):
    if path:
        if not os.path.exists(path):
            return 'Specified path does not exist.'
        else:
            vstore.add_hcv(path)
            cstore.add_hcv(vstore.hcv_video)
            return True

@app.callback(
    Output('recorded-data', 'children'),
    [Input('cardiac-frame-recorder', 'n_clicks')],
    state=[State('filename-input', 'value'),
           State('filepath-check', 'children'),
           State('overwrite-check', 'value')])
def init_frame_recorder(n_clicks, filename, check, switch):
    if n_clicks:
        if check is True:
            queue = mp.Queue()
            proc = mp.Process(target=app_utils.frameLabeler, args=(filename,queue))
            proc.start()
            proc.join()

            _dsfr_,_ssfr_ = queue.get()

            if switch is False:
                dsfr = dstore.ds_frame + _dsfr_
                dsar = dstore.ds_area + [0 for i in _dsfr_]
                ssfr = dstore.ss_frame + _ssfr_
                ssar = dstore.ss_area + [0 for i in _ssfr_]
            else:
                dsfr = _dsfr_
                dsar = [0 for i in _dsfr_]
                ssfr = _ssfr_
                ssar = [0 for i in _ssfr_]

            dstore.ds_frame = dsfr
            dstore.ds_area = dsar
            dstore.ss_frame = ssfr
            dstore.ss_area = ssar

            return trigger

@app.callback(
    Output('frame-number', 'options'),
    [Input('cardiac-cycle', 'value'),
     Input('frame-table', 'data')])
def update_dropdowns(input_type, data):
    if input_type == 'diastole':
        frlist = dstore.ds_frame
    elif input_type == 'systole':
        frlist = dstore.ss_frame

    return [dict(label=fr,value=fr) for fr in frlist]

@app.callback(
    Output('update-callback', 'children'),
    [Input('filename-input', 'value'),
     Input('frame-number', 'value')])
def update_still_image(filename, fr):
    if not filename:
        return None
    else:
        if not fr:
            return None
        else:
            frame = vstore.raw_video.read(index=fr)
            vstore.current_frame = frame

            return trigger

@app.callback(
    [Output('selected-data', 'children'),
     Output('compute-callback', 'children')],
    [Input('still-image-graph', 'selectedData'),
     Input('cardiac-cycle', 'value'),
     Input('frame-number', 'value'),
     Input('draw-ellipse', 'value')])
def compute_area(selected_data, input_type, fr, ellipse):
    if selected_data:
        selectedPoints = selected_data['lassoPoints']
        x = [int(x) for x in selectedPoints['x']]
        y = [int(y) for y in selectedPoints['y']]

        contour = np.empty((len(x),2), dtype=np.int)

        # Invert y for subsequent exports
        maxx, maxy = cstore.full
        aspect = maxx/maxy
        diff = maxy*aspect
        for i,v in enumerate(y): y[i] = diff - v    

        contour[:,0],contour[:,1] = x,y

        if ellipse:
            arr = np.zeros((maxy, maxx), dtype='uint8')
            arr = cv2.ellipse(arr, (cv2.fitEllipse(contour)), 255, -1)
            contour, _ = cvu.find_contours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if isinstance(contour, list):
                contour = hcv.largest(contour)

        cstore.add_contour(input_type, fr, contour)
        area = cv2.contourArea(contour)

        if fr:
            if input_type == 'diastole':
                frlist = dstore.ds_frame
            elif input_type == 'systole':
                frlist = dstore.ss_frame

            index = frlist.index(fr)

            if input_type == 'diastole':
                dstore.ds_area[index] = area
            elif input_type == 'systole':
                dstore.ss_area[index] = area

            frame = vstore.raw_video.read(index=fr)
            cvu.draw_contours(frame, contour, -1, (0,0,255), 1)
            vstore.current_frame = frame

            return trigger, trigger
        else:
            return None, None
    else:
        return None, None

@app.callback(
    Output('still-image', 'children'),
    [Input('update-callback', 'children'),
     Input('compute-callback', 'children'),
     Input('frame-number', 'value')])
def update_image(update, compute, fr):
    if fr:
        frame = vstore.current_frame
        if frame is not None:
            return html.Div(children=[
                app_utils.interactiveImage(
                    id='still-image-graph',
                    img=frame)
                ])
        else:
            return app_utils.blank()
    else:
        return app_utils.blank()

@app.callback(
    Output('data-load-div', 'children'),
    [Input('load-data', 'n_clicks')],
    state=[State('output-filename', 'value')])
def load_data(n_clicks, filename):
    if n_clicks:
        if filename:
            df = pd.read_csv(filename)
            dstore.ds_frame = list(df['EndDiastoleFrame'])
            dstore.ds_area = list(df['EndDiastoleArea'])
            dstore.ss_frame = list(df['EndSystoleFrame'])
            dstore.ss_area = list(df['EndSystoleArea'])

            return trigger

@app.callback(
    Output('data-save-div', 'children'),
    [Input('save-data', 'n_clicks')],
    state=[State('output-filename', 'value')])
def save_data(n_clicks, filename):
    if n_clicks:
        if filename:
            df = pd.DataFrame(data=dict(
                                    EndDiastoleFrame=dstore.ds_frame,
                                    EndDiastoleArea=dstore.ds_area,
                                    EndSystoleFrame=dstore.ss_frame,
                                    EndSystoleArea=dstore.ss_area)
                            )
            df.to_csv(filename)
            return 'Finished.'

@app.callback(
    Output('frame-export-div', 'children'),
    [Input('export-frames', 'n_clicks')],
    state=[State('frame-output-filename', 'value')])
def export_frames(n_clicks, filename):
    if n_clicks:
        if filename:
            if not os.path.exists(filename):
                os.mkdir(filename)

            cstore.export(filename)
            return 'Finished.'

@app.callback(
    Output('frame-table', 'data'),
    [Input('recorded-data', 'children'),
     Input('compute-callback', 'children'),
     Input('data-load-div', 'children')])
def update_data_table(data, selected_data, loaded_data):
    dsfr = dstore.ds_frame
    dsar = dstore.ds_area
    ssfr = dstore.ss_frame
    ssar = dstore.ss_area

    data = []
    for d,da,s,sa in zip(dsfr,dsar,ssfr,ssar):
        data.append(dict(ds_frame=d, ds_area=da, ss_frame=s, ss_area=sa))

    return data

@app.callback(
    Output('close-app-div', 'children'),
    [Input('close-app', 'n_clicks')])
def shutdown_app(n_clicks):
    if n_clicks:
        shutdown = request.environ.get('werkzeug.server.shutdown')

        try:
            vstore.close()
            shutdown()
        except:
            raise RuntimeError('Not running with the Werkzeug Server')

        return 'App has been shutdown.'

if __name__ == '__main__':
    app.run_server(debug=False)