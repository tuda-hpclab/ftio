"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Sep 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ftio.plot.helper import format_plot
from ftio.plot.plot_core import PlotCore


def main(args=sys.argv):
    plotter = PlotCore(args)

    plotter.nRun = len(pd.unique(plotter.data.df_time["file_index"]))
    symbols = ["circle", "square", "cross", "star-triangle-down", "hourglass"]
    markeredgecolor = "DarkSlateGrey"

    if plotter.nRun == 1:
        x = plotter.data.df_time["number_of_ranks"].astype(str)

    else:
        plotter.data.df_time = plotter.data.df_time.sort_values(
            by=["number_of_ranks", "file_index"]
        )
        x = [
            plotter.data.df_time["number_of_ranks"].astype(str),
            plotter.data.df_time["file_index"].astype(str),
        ]

    fig = go.Figure()
    ranks = plotter.data.df_time["number_of_ranks"]
    # sets_names = ["Limit", "No Limit"]
    # sets = [[0,1],[2]]
    sets = [[0, 1], [2, 3], [4, 5], [6, 7]]
    sets_names = ["Direct", "Up-only", "Adaptive", "No Limit"]

    metric = [
        plotter.data.df_time["delta_t_agg"],
        plotter.data.df_time["delta_t_agg_io"],
    ]
    modes = ["Total", "IO"]

    for i, _ in enumerate(sets):
        group = plotter.data.df_time["file_index"].isin(values=sets[i])
        x = ranks[group]
        x_unqiue = pd.unique(x)
        y = np.zeros((len(modes), len(x_unqiue)))
        y_plus = np.zeros((len(modes), len(x_unqiue)))
        y_minus = np.zeros((len(modes), len(x_unqiue)))

        # fill the array
        for count0, _ in enumerate(modes):
            for count1, _ in enumerate(x_unqiue):
                scope_metric = metric[count0]
                scope_metric = scope_metric[group]
                data = scope_metric[x == x_unqiue[count1]]
                y[count0, count1] = data.mean()
                y_plus[count0, count1] = abs(max(data - y[count0, count1]))
                y_minus[count0, count1] = abs(min(data - y[count0, count1]))

            fig.add_trace(
                go.Scatter(
                    x=x_unqiue,
                    y=y[count0],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": y_plus[count0],
                        "arrayminus": y_minus[count0],
                    },
                    mode="lines+markers",
                    name=modes[count0],
                    legendgroup=sets_names[i],
                    legendgrouptitle_text=sets_names[i],
                    marker={
                        "symbol": symbols[i],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    # marker_color=colors[i]
                    showlegend=True,
                )
            )

    fig.update_xaxes(title_text="Ranks")
    fig.update_layout(xaxis_title="Ranks", yaxis_title="Aggregated Time (s)")
    fig.update_layout(legend_groupclick="toggleitem")
    fig = format_plot(fig)
    fig.update_layout(
        hovermode="x",
        legend_tracegroupgap=1,
        width=1200,
        height=500,
    )
    fig.show()
    print(plotter.names)


if __name__ == "__main__":
    main(sys.argv)
