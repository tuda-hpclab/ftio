"""
This functions plots error bars (normed and not normed)

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ftio.plot.helper import format_plot


def make_sub():
    f = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
    )
    f.update_xaxes(minor={"ticklen": 6, "tickcolor": "black", "showgrid": True})

    return f


def plot_error_bar(df, s, f=None):
    if f is None:
        f = []
    if not f:
        f = make_sub()
    modes = ["max", "median"]
    # modes = ['max', 'min', 'median', 'hmean', 'amean', 'whmean']
    x_unqiue = pd.unique(df["number_of_ranks"])
    colors = px.colors.qualitative.Plotly
    if "T" in s:
        symbols = [
            "circle",
            "square",
            "cross",
            "star-triangle-down",
            "hourglass",
        ]
        dash = "dash"
    else:
        symbols = ["hexagon", "diamond", "x", "star-triangle-up", "bowtie"]
        # colors = colors[5:]
        colors = colors[2:]
        dash = None

    markeredgecolor = "DarkSlateGrey"
    y = np.zeros((len(modes), len(x_unqiue)))
    y_plus = np.zeros((len(modes), len(x_unqiue)))
    y_minus = np.zeros((len(modes), len(x_unqiue)))
    y_plus_normed = np.zeros((len(modes), len(x_unqiue)))
    y_minus_normed = np.zeros((len(modes), len(x_unqiue)))

    for j in modes:
        for k in range(0, len(x_unqiue)):
            data = df[df["number_of_ranks"] == x_unqiue[k]][j]
            data = data[data != 0]
            if not data.empty:
                y[modes.index(j), k] = data.mean()
                tmp = data - y[modes.index(j), k]
                y_plus[modes.index(j), k] = abs(max(tmp))
                y_minus[modes.index(j), k] = abs(min(tmp))
            if y[modes.index(j), k] != 0:
                y_plus_normed[modes.index(j), k] = (
                    abs(max(tmp)) / y[modes.index(j), k] * 100
                )
                y_minus_normed[modes.index(j), k] = (
                    abs(min(tmp)) / y[modes.index(j), k] * 100
                )
            else:
                y_plus_normed[modes.index(j), k] = 0
                y_minus_normed[modes.index(j), k] = 0
        f.add_trace(
            go.Scatter(
                x=x_unqiue,
                y=y[modes.index(j)],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": y_plus[modes.index(j)],
                    "arrayminus": y_minus[modes.index(j)],
                },
                mode="lines+markers",
                name=f"$\\text{{{j}}}({s})$",
                legendgroup=f"{j}_{s}",
                marker={
                    "symbol": symbols[modes.index(j)],
                    "line": {"width": 1, "color": markeredgecolor},
                },
                marker_color=colors[modes.index(j)],
                line={"dash": dash},
            ),
            row=1,
            col=1,
        )

    f.update_yaxes(title_text="Transfer rate (B/s)", row=1, col=1)
    f.update_xaxes(type="log", row=1, col=1, showspikes=True)
    # f.update_xaxes(type="log",row=1, col=1, showspikes=True) #, tickmode = 'array', tickvals = x_unqiue, ticktext = x_unqiue.astype(str))
    f.update_layout(
        hovermode="x",
        legend_tracegroupgap=1,
        width=900,
        height=400,
        title="Bandwidth with error bars",
    )

    for j in modes:
        f.add_trace(
            go.Scatter(
                x=x_unqiue,
                y=np.repeat(0, len(y[modes.index(j)])),
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": y_plus_normed[modes.index(j)],
                    "arrayminus": y_minus_normed[modes.index(j)],
                },
                mode="lines+markers",
                name=f"$\\text{{{j}}}({s})$",
                showlegend=False,
                legendgroup=f"{j}_{s}",
                marker={
                    "symbol": symbols[modes.index(j)],
                    "line": {"width": 1, "color": markeredgecolor},
                },
                marker_color=colors[modes.index(j)],
                line={"dash": dash},
            ),
            row=2,
            col=1,
        )

    f.update_xaxes(
        type="log", title_text="Ranks", row=2, col=1, showspikes=True
    )  # ,tickmode = 'array', tickvals = x_unqiue, ticktext = x_unqiue.astype(str))
    f.update_yaxes(title_text="Rel.<br>dev. (%)", row=2, col=1)
    f.update_yaxes(
        minor={"ticklen": 4, "tickcolor": "black", "showgrid": True, "ticks": "inside"}
    )
    f.update_layout(
        hovermode="x",
        legend_tracegroupgap=1,
        width=900,
        height=400,
        title="Bandwidth with error bars (normed)",
    )

    if "A" in s:
        title = "Error Bar: Average"
    elif "S" in s:
        title = "Error Bar: Sum"
    else:
        title = "Error Bar: Ind"

    f.update_layout(
        yaxis_title="Transfer Rate (B/s)",
        title=title,
        width=900,
        height=560,
    )
    f = format_plot(f, 17)
    # if 'B' in s and 'E' in s:
    # 	f[-1].show()
    # 	f[-1].write_image("%s.pdf"%s)
    return f


# **********************************************************************
# *                       2. plot_time_error_bar
# **********************************************************************
def plot_time_error_bar(df_time, modes, names, colors, symbols, markeredgecolor):
    f = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.2,
        row_heights=[0.68, 0.32],
    )
    x_unqiue = pd.unique(df_time["number_of_ranks"])
    y = np.zeros((len(modes), len(x_unqiue)))
    y_plus = np.zeros((len(modes), len(x_unqiue)))
    y_minus = np.zeros((len(modes), len(x_unqiue)))
    y_plus_normed = np.zeros((len(modes), len(x_unqiue)))
    y_minus_normed = np.zeros((len(modes), len(x_unqiue)))
    for j in modes:
        for count, _ in enumerate(x_unqiue):
            if "+" in j:
                values = j.split("+")
                values = [x.strip() for x in values]
                for q in values:
                    if values.index(q) == 0:
                        data = df_time[df_time["number_of_ranks"] == x_unqiue[count]][q]
                    else:
                        data = (
                            data
                            + df_time[df_time["number_of_ranks"] == x_unqiue[count]][q]
                        )
            elif "/" in j:
                values = j.split("/")
                values = [x.strip() for x in values]
                for q in values:
                    if values.index(q) == 0:
                        data = df_time[df_time["number_of_ranks"] == x_unqiue[count]][q]
                    else:
                        data = (
                            data
                            / df_time[df_time["number_of_ranks"] == x_unqiue[count]][q]
                        )
            else:
                data = df_time[df_time["number_of_ranks"] == x_unqiue[count]][j]
            data = data[data != 0]
            if not data.empty:
                y[modes.index(j), count] = data.mean()
                tmp = data - y[modes.index(j), count]
                y_plus[modes.index(j), count] = abs(max(tmp))
                y_minus[modes.index(j), count] = abs(min(tmp))
                if y[modes.index(j), count] != 0:
                    y_plus_normed[modes.index(j), count] = (
                        abs(max(tmp)) / y[modes.index(j), count] * 100
                    )
                    y_minus_normed[modes.index(j), count] = (
                        abs(min(tmp)) / y[modes.index(j), count] * 100
                    )
                else:
                    y_plus_normed[modes.index(j), count] = 0
                    y_minus_normed[modes.index(j), count] = 0
        f.add_trace(
            go.Scatter(
                x=x_unqiue,
                y=y[modes.index(j)],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": y_plus[modes.index(j)],
                    "arrayminus": y_minus[modes.index(j)],
                },
                mode="lines+markers",
                name=names[modes.index(j)],
                legendgroup=names[modes.index(j)],
                marker={
                    "symbol": symbols[modes.index(j)],
                    "line": {"width": 1, "color": markeredgecolor},
                    "size": 8 if "Total" not in names[modes.index(j)] else 9,
                },
                marker_color=colors[modes.index(j)],
            ),
            row=1,
            col=1,
        )
    f.update_yaxes(title_text="Time (s)", row=1, col=1, showspikes=True)
    f.update_xaxes(
        type="log", title_text="Ranks", row=1, col=1, showspikes=True
    )  # ,tickmode = 'array', tickvals = x_unqiue, ticktext = x_unqiue.astype(str))
    f.update_layout(
        hovermode="x",
        legend_tracegroupgap=1,
        width=900,
        height=640,
        title="I/O time with error bars",
    )

    #! Error bar normed
    for j in modes:
        f.add_trace(
            go.Scatter(
                x=x_unqiue,
                y=np.repeat(0, len(y[modes.index(j)])),
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": y_plus_normed[modes.index(j)],
                    "arrayminus": y_minus_normed[modes.index(j)],
                },
                mode="lines+markers",
                name=names[modes.index(j)],
                showlegend=False,
                legendgroup=names[modes.index(j)],
                marker={
                    "symbol": symbols[modes.index(j)],
                    "line": {"width": 1, "color": markeredgecolor},
                    "size": 8,
                },
                marker_color=colors[modes.index(j)],
            ),
            row=2,
            col=1,
        )
    f.update_xaxes(type="log", title_text="Ranks", row=2, col=1, showspikes=True)
    f.update_yaxes(title_text="Rel. dev. (%)", row=2, col=1, showspikes=True)
    f.update_layout(
        hovermode="x",
        legend_tracegroupgap=1,
        width=990,
        height=640,
        title="I/O time with error bars (normed)",
    )
    f = format_plot(f, 17)
    return f
