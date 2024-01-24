import cv2
import multiprocessing as mp
import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
import os
import numpy as np
import pandas as pd
from flask import request
import signal

import vuba
import app_utils, app_layout


external_stylesheets = ["./assets/app.css"]  # courtesy of dadamson
app = dash.Dash("heartcv_validation_app", external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = app_layout.app_layout()


# Python session containers used instead of passing around data through html divs
dstore = app_utils.DataStore([0], [0])
vstore = app_utils.VideoStore()


# For callbacks where no return is needed
trigger = {"dummydata": True}


@app.callback(
    Output("filepath-check", "children"), 
    [Input("filename-input", "value")])
def check_raw_path(path):
    if path:
        if not os.path.exists(path):
            return "Specified path does not exist."
        else:
            vstore.initiate(path)
            return True


@app.callback(
    Output("recorded-data", "children"),
    [Input("cardiac-frame-recorder", "n_clicks")],
    state=[
        State("filename-input", "value"),
        State("filepath-check", "children"),
        State("overwrite-check", "value"),
    ],
)
def init_frame_recorder(n_clicks, filename, check, switch):
    if n_clicks:
        if check is True:
            queue = mp.Queue()
            proc = mp.Process(target=app_utils.frame_recorder, args=(filename, queue))
            proc.start()
            proc.join()

            _dsfr_, _ssfr_ = queue.get()
            print(_dsfr_, _ssfr_)

            if switch is False:
                dsfr = dstore.ds_frame + _dsfr_
                ssfr = dstore.ss_frame + _ssfr_
            else:
                dsfr = _dsfr_
                ssfr = _ssfr_

            dstore.ds_frame = dsfr
            dstore.ss_frame = ssfr

            return trigger


@app.callback(
    Output("data-load-div", "children"),
    [Input("load-data", "n_clicks")],
    state=[State("output-filename", "value")],
)
def load_data(n_clicks, filename):
    if n_clicks:
        if filename:
            df = pd.read_csv(filename)
            dstore.ds_frame = list(df["end_diastole_frame"])
            dstore.ss_frame = list(df["end_systole_frame"])

            return trigger


@app.callback(
    Output("data-save-div", "children"),
    [Input("save-data", "n_clicks")],
    state=[State("output-filename", "value")],
)
def save_data(n_clicks, filename):
    if n_clicks:
        if filename:
            df = pd.DataFrame(
                data=dict(
                    end_diastole_frame=dstore.ds_frame,
                    end_systole_frame=dstore.ss_frame,
                )
            )
            df.to_csv(filename)
            return "Finished."


@app.callback(
    Output("frame-table", "data"),
    [
        Input("recorded-data", "children"),
        Input("data-load-div", "children"),
    ],
)
def update_data_table(data, loaded_data):
    dsfr = dstore.ds_frame
    ssfr = dstore.ss_frame

    data = []
    for d, s in zip(dsfr, ssfr):
        data.append(dict(ds_frame=d, ss_frame=s))

    return data


@app.callback(Output("close-app-div", "children"), [Input("close-app", "n_clicks")])
def shutdown_app(n_clicks):
    if n_clicks:
        vstore.close()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    app.run_server(debug=False)
