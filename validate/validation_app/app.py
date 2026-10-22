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

# Cardiac statistic functions
def bpm(peaks, fs):
    """
    Calculate beats per minute based on detected peaks.

    Parameters
    ----------
    num_peaks : int or float
        Number of peaks detected.
    sample_length : int or float
        Length of time-series data.
    fs : int or float
        Sampling frequency of footage.

    Returns
    -------
    bpm : int or float
        Beats per minute.

    """
    b2b = b2b_intervals(peaks, fs)
    return 60 / np.nanmean(b2b)


def b2b_intervals(peaks, fs):
    """
    Calculate beat to beat intervals based on peaks detected.

    Parameters
    ----------
    peaks : ndarray
        Detected peaks.
    fs : int or float
        Sampling frequency of footage.

    Returns
    -------
    intervals : ndarray
        Beat to beat intervals.

    """
    peaks = np.asarray(peaks)
    return (peaks[1:] - peaks[:-1]) / fs


external_stylesheets = ["./assets/app.css"]  # courtesy of dadamson
app = dash.Dash("heartcv_validation_app", external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = app_layout.app_layout()


# Python session containers used instead of passing around data through html divs
dstore = app_utils.DataStore([0], [0], 0)
vstore = app_utils.VideoStore()


# For callbacks where no return is needed
trigger = ''


@app.callback(
    output=Output("filepath-check", "children"), 
    inputs=[Input("filename-input", "value")])
def check_raw_path(path):
    if path:
        if not os.path.exists(path):
            return "Specified path does not exist."
        else:
            vstore.initiate(path)
            return True


@app.callback(
    output=[
        Output("recorded-data", "children"),
        Output("cardiac_stats", "children"),
    ],
    inputs=[Input("cardiac-frame-recorder", "n_clicks")],
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

            video = vuba.Video(filename)
            fps = video.fps
            video.close()

            dstore.video_fps = fps

            hr = bpm(dsfr, fps)
            b2b = b2b_intervals(dsfr, fps)

            stats = f'**HR (bpm)**: {round(hr, 1)}, **Mean beat-to-beat (sec)**: {round(np.nanmean(b2b), 1)}'                

            return trigger, stats


@app.callback(
    output=Output("data-load-div", "children"),
    inputs=[Input("load-data", "n_clicks")],
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
    output=Output("data-save-div", "children"),
    inputs=[Input("save-data", "n_clicks")],
    state=[State("output-filename", "value")],
)
def save_data(n_clicks, filename):
    if n_clicks:
        if filename:
            if not filename.endswith('.csv'):
                return "Filename must be .csv format."

            df = pd.DataFrame(
                data=dict(
                    end_diastole_frame=dstore.ds_frame,
                    end_systole_frame=dstore.ss_frame,
                )
            )
            df.to_csv(filename)

            stats = []
            b2b = b2b_intervals(dstore.ds_frame, dstore.video_fps)
            stats.append([
                        bpm(dstore.ds_frame, dstore.video_fps),
                        b2b.min(),
                        b2b.mean(),
                        np.median(b2b),
                        b2b.max(),
                        np.std(b2b),
                        b2b.max() - b2b.min(),
                        np.sqrt(np.nanmean((b2b[1:] - b2b[:-1]) ** 2))
                    ]
                )

            keys = ['bpm', 'min_b2b', 'mean_b2b', 'median_b2b', 'max_b2b', 'sd_b2b', 'range_b2b', 'rmssd']
            avg_stats = {}
            for key, stat in zip(keys, zip(*stats)):
                avg_stats[key] = [np.nanmean(np.asarray(stat), axis=0)]


            df_stats = pd.DataFrame(avg_stats)
            df_stats.to_csv(filename.replace('.csv', '_stats.csv'))

            return "Finished."


@app.callback(
    output=Output("frame-table", "data"),
    inputs=[
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


@app.callback(
    output=Output("close-app-div", "children"), 
    inputs=[Input("close-app", "n_clicks")]
)
def shutdown_app(n_clicks):
    if n_clicks:
        vstore.close()
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    app.run_server(debug=False)
