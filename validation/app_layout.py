import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table


def app_layout():
    """ Convenience function for creating the app layout. """
    return html.Div(
        children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="four columns",
                        children=[html.H3(children="HeartCV manual validation tool")],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            html.Br(),  # forces columns to be created
                        ],
                    ),
                    html.Div(
                        className="two columns",
                        children=[
                            html.Br(),
                            html.Button("Close", "close-app"),
                            html.Div(id="close-app-div"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Div(className="one column", children=[]),  # spacer
                    html.Div(
                        className="four columns",
                        children=[
                            # UI sidebar
                            html.Br(),
                            dcc.Input(
                                id="filename-input",
                                type="text",
                                placeholder="Specify a name to a video-file...",
                                size=40,
                            ),
                            html.Div(id="filepath-check"),
                            html.Br(),
                            dcc.Input(
                                id="hcv-filename-input",
                                type="text",
                                placeholder="Specify a name to heartcv visual output...",
                                size=40,
                            ),
                            html.Div(id="hcv-filepath-check"),
                            html.Br(),
                            html.Label("Overwrite data-table?"),
                            dcc.RadioItems(
                                id="overwrite-check",
                                options=[
                                    {"label": "Yes", "value": True},
                                    {"label": "No", "value": False},
                                ],
                                value=True,
                                labelStyle={"display": "inline-block"},
                            ),
                            html.Button(
                                "Record cardiac cycle", id="cardiac-frame-recorder"
                            ),
                            html.Br(),
                            html.Br(),
                            dcc.Markdown(
                                """
                    **Frame recorder usage:**
                    * `Left`/`Right` keys move one frame `backwards`/`forwards` in the video respectively (Linux), or < and > keys in Windows.
                    * `D`/`S` keys record the frame number for end `diastole`/`systole` respectively.
                    * The `SpaceBar` plays the video in realtime and the `Esc` key closes the GUI.
                """
                            ),
                            html.Br(),
                            html.Label(children="Image to load for validation:"),
                            dcc.RadioItems(
                                id="cardiac-cycle",
                                options=[
                                    {"label": "Diastole", "value": "diastole"},
                                    {"label": "Systole", "value": "systole"},
                                ],
                                value="diastole",
                                labelStyle={"display": "inline-block"},
                            ),
                            dcc.Dropdown(
                                id="frame-number",
                                placeholder="Please select a frame number to load...",
                            ),
                            html.Br(),
                            html.Div(
                                children=[
                                    html.Div(
                                        children=[html.Button("load", id="load-data")],
                                        style={"width": "10%", "display": "table-cell"},
                                    ),
                                    html.Div(
                                        children=[html.Button("save", id="save-data")],
                                        style={"width": "10%", "display": "table-cell"},
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Input(
                                                id="output-filename",
                                                type="text",
                                                placeholder="Specify a filename to save data...",
                                                size=35,
                                            )
                                        ],
                                        style={"width": "60%", "display": "table-cell"},
                                    ),
                                    html.Div(
                                        children=[],
                                        style={"width": "10%", "display": "table-cell"},
                                    ),
                                ],
                                style={"width": "100%", "display": "table"},
                            ),
                            html.Div(id="data-load-div"),
                            html.Div(id="data-save-div"),
                            html.Br(),
                            html.Div(
                                children=[
                                    html.Div(
                                        children=[
                                            html.Button("Export", id="export-frames"),
                                        ],
                                        style={"width": "20%", "display": "table-cell"},
                                    ),
                                    html.Div(
                                        children=[],
                                        style={"width": "10%", "display": "table-cell"},
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Input(
                                                id="frame-output-filename",
                                                type="text",
                                                placeholder="Specify a filename to save frames...",
                                                size=45,
                                            )
                                        ],
                                        style={"width": "70%", "display": "table-cell"},
                                    ),
                                ],
                                style={"width": "100%", "display": "table"},
                            ),
                            html.Div(id="frame-export-div"),
                            html.Hr(),
                            dash_table.DataTable(
                                id="frame-table",
                                columns=[
                                    dict(name="Diastole-frame", id="ds_frame"),
                                    dict(name="Diastole-area", id="ds_area"),
                                    dict(name="Systole-frame", id="ss_frame"),
                                    dict(name="Systole-area", id="ss_area"),
                                ],
                                style_cell={"textAlign": "left"},
                                style_cell_conditional=[
                                    {"if": {"column_id": col}, "width": "25%"}
                                    for col in [
                                        "ds_frame",
                                        "ds_area",
                                        "ss_frame",
                                        "ss_area",
                                    ]
                                ],
                                fixed_rows={"headers": True},
                                style_table={"height": 250},
                                editable=True,
                            ),
                            html.Div(id="recorded-data"),
                        ],
                    ),
                    html.Div(
                        className="seven columns",
                        children=[
                            html.Label(children="Draw ellipses?"),
                            dcc.RadioItems(
                                id="draw-ellipse",
                                options=[
                                    dict(label="Yes", value=True),
                                    dict(label="No", value=False),
                                ],
                                value=True,
                                labelStyle={"display": "inline-block"},
                            ),
                            html.Br(),
                            html.Div(id="still-image", children=[]),
                            html.Div(id="selected-data"),
                            html.Div(id="update-callback"),
                            html.Div(id="compute-callback"),
                        ],
                    ),
                ],
            ),
        ]
    )
