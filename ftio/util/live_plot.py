"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import base64
import json
import threading

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize Dash app
app = dash.Dash(__name__)

# Global variable to store the current file's data
current_data = pd.DataFrame()


# Function to read the JSONL file and return the data as a DataFrame
def read_jsonl_file(file_content):
    data = []
    for line in file_content.decode("utf-8").splitlines():
        data.append(json.loads(line))
    return pd.DataFrame(data)


# Create Dash layout
app.layout = html.Div(
    [
        html.H1("Live Data Plot from Selected JSONL File"),
        # File Upload component
        dcc.Upload(
            id="upload-data",
            children=html.Button("Upload File"),
            multiple=False,  # Allow only one file at a time
        ),
        # Graph to display the plot
        dcc.Graph(id="live-graph"),
        # Interval to update the graph every second
        dcc.Interval(
            id="graph-update",
            interval=1000,  # in milliseconds, i.e., update every second
            n_intervals=0,
        ),
    ]
)


# Callback to update the graph based on the file's latest data
@app.callback(
    Output("live-graph", "figure"),
    Input("graph-update", "n_intervals"),
    Input("upload-data", "contents"),
)
def update_graph(n, uploaded_file):
    global current_data

    # If a file is uploaded, update the data
    if uploaded_file:
        content_type, content_string = uploaded_file.split(",")
        decoded = base64.b64decode(content_string)
        current_data = read_jsonl_file(decoded)

    # Assuming the data in the JSONL has at least two columns, 'x' and 'y'
    # If the data structure is different, adjust the column names accordingly
    fig = go.Figure()
    if not current_data.empty:
        fig.add_trace(
            go.Scatter(
                x=current_data["x"],
                y=current_data["y"],
                mode="markers",
                name="Data Points",
            )
        )

        # Customize the layout of the plot
        fig.update_layout(
            title="Live Data Plot",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
        )

    return fig


if __name__ == "__main__":
    # Run the Dash app in a separate thread for live updates
    def run_app():
        app.run_server(debug=True, use_reloader=False)

    # Start the Dash app in a separate thread
    thread = threading.Thread(target=run_app)
    thread.start()

    # Optionally, you can add other functionality to monitor new data coming in.
