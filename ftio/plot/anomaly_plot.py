"""
Outlier plot function

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Sep 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import DecisionBoundaryDisplay

from ftio.freq.freq_html import create_html
from ftio.plot.helper import format_plot
from ftio.plot.spectrum import plot_both_spectrums


# ?#################################
# ? Plot outliers
# ?#################################
def plot_outliers(
    args,
    freq_arr: np.ndarray,
    amp: np.ndarray,
    indecies: np.ndarray,
    conf: np.ndarray,
    dominant_index: np.ndarray,
    d: np.ndarray = np.array([]),
    eps: float = 0,
) -> None:
    """Plots outliers for DB scan

    Args:
        freq_arr (np.ndarray): _description_
        amp (np.ndarray): aplitude or power
        indecies (np.ndarray): _description_
        conf (np.ndarray): _description_
        dominant_index (np.ndarray): _description_
        d (np.ndarray, optional): _description_. Defaults to np.array([]).
    """
    colorscale = [
        (0, "rgb(0,50,150)"),
        (0.5, "rgb(150,50,150)"),
        (1, "rgb(255,50,0)"),
    ]
    labels = [(f"cluster {i}") if (i >= 0) else "outliers" for i in dominant_index]
    # prepare figures
    figs = []
    for _ in np.arange(0, 5):
        f = go.Figure()
        f = format_plot(f)
        figs.append(f)

    # prepare symbols
    if dominant_index.size > 0 and dominant_index.max() < 20:
        symbol = dominant_index.copy()
        symbol[symbol >= 5] = symbol[symbol >= 5] + 1
        symbol[symbol == -1] = 5  # x marker
        symbol = symbol - 1
    else:
        symbol = np.ones(dominant_index.size)
    labels = np.array(labels)

    if d.size == 0 and len(freq_arr) != 0:
        d = np.vstack(
            (
                freq_arr[indecies] / freq_arr[indecies].max(),
                amp[indecies] / amp[indecies].sum(),
            )
        ).T
    elif len(freq_arr) == 0:
        return
    else:
        pass

    all_colors = [
        px.colors.qualitative.Alphabet[0],
        px.colors.qualitative.Plotly[0],
    ] + px.colors.qualitative.Plotly[2:]
    all_colors = np.array(
        all_colors
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Dark24
        + [px.colors.qualitative.Plotly[1]]
    )

    # Only for DB SCAN
    if args.outlier.lower() in ["dbscan", "db-scan", "db"]:
        if dominant_index.max() < 20:
            color = all_colors[dominant_index]
        else:
            color = np.array(["blue" if (x >= 0) else "red" for x in dominant_index])
        # draw the circles
        for i in range(0, len(d)):
            figs[0].add_shape(
                {
                    "type": "circle",
                    "x0": d[i, 0] - eps,
                    "y0": d[i, 1] - eps,
                    "x1": d[i, 0] + eps,
                    "y1": d[i, 1] + eps,
                    "opacity": 0.3,
                },
                name=labels[i],
                line_color=color[i],
            )
    else:
        color = all_colors[dominant_index]

    for i in np.unique(labels):
        figs[0].add_trace(
            go.Scatter(
                x=d[labels == i, 0],
                y=d[labels == i, 1],
                mode="markers",
                marker={"color": color[labels == i], "colorscale": colorscale},
                text=conf[labels == i],
                hovertemplate="<b>Noremed values<br><br>Freq: %{x:.4f}  <br>"
                + "Amplitude: %{y:.2f}<br>"
                + "conf: %{text}",
                showlegend=True,
                name=i,
            )
        )

        figs[1].add_trace(
            go.Scatter(
                x=d[labels == i, 0],
                y=d[labels == i, 1],
                mode="markers",
                marker={
                    "color": conf[labels == i],
                    "colorscale": colorscale,
                    "coloraxis": "coloraxis",
                    "symbol": symbol[labels == i],
                    "size": 12,
                },
                text=labels[labels == i],
                hovertemplate="<b>Noremed values<br><br>Freq: %{x:.4f}  <br>"
                + "Amplitude: %{y:.2f}<br>"
                + "cluster: %{text}",
                showlegend=True,
                name=i,
            )
        )

        figs[2].add_trace(
            go.Scatter(
                x=freq_arr[indecies[labels == i]],
                y=2 * amp[indecies[labels == i]],
                mode="markers",
                marker={"color": color[labels == i], "colorscale": colorscale},
                text=labels[labels == i],
                hovertemplate="<b>Freq: %{x:.4f}    Hz<br>"
                + "Amplitude: %{y:.2f}<br>"
                + "cluster: %{text}",
                showlegend=True,
                name=i,
            )
        )

    figs[1].update_layout(
        coloraxis={
            "colorbar": {
                "x": 0.92,
                "len": 0.91,
                "y": 0.57,
                "title": "conf",
                "thickness": 10,
                "tickfont": {"size": 14},
            },
            "colorscale": colorscale,
        }
    )

    counter = 2
    spec_figs, plt_names = plot_both_spectrums(args, freq_arr, amp, full=False)
    for trace in list(spec_figs.select_traces()):
        counter += 1
        trace.update(marker={"coloraxis": "coloraxis"})
        figs[counter].add_trace(trace)
        if figs[counter].data and "xaxis" in figs[counter].data[0]:
            figs[counter].data[0]["xaxis"] = "x1"
            figs[counter].data[0]["yaxis"] = "y1"

    for i in np.arange(3, 5):
        figs[i].update_layout(coloraxis={"colorscale": colorscale})

    y_title = "Power" if args.psd else "Amplitude"
    figs[0].update_xaxes(title_text="Normed Frequency", range=[-0.01, 1.01])
    figs[0].update_yaxes(title_text=f"Normed {y_title}")
    figs[1].update_xaxes(title_text="Normed Frequency ", range=[-0.01, 1.01])
    figs[1].update_yaxes(title_text=f"Normed {y_title}")
    figs[2].update_xaxes(title_text="Frequency (Hz)", range=[0, freq_arr[indecies].max()])
    figs[2].update_yaxes(title_text=f"{y_title}")
    figs[3].update_xaxes(title_text="Frequency (Hz)")
    figs[3].update_yaxes(title_text=plt_names[0])
    figs[4].update_xaxes(title_text="Frequency (Hz)")
    figs[4].update_yaxes(title_text=plt_names[1])
    for fig in figs:
        fig.update_layout(width=1300, height=400)

    configuration = {"toImageButtonOptions": {"format": "png", "scale": 4}}
    plot_name = "anomaly"
    if "plot_name" in args:
        plot_name += "_" + args.plot_name

    create_html(figs, args.render, configuration, plot_name)


def plot_decision_boundaries(model, d, conf):
    disp = DecisionBoundaryDisplay.from_estimator(
        model,
        d,
        response_method="decision_function",
        alpha=0.5,
    )
    disp.ax_.scatter(d[:, 0], d[:, 1], c=conf, s=20, edgecolor="k")
    disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
    plt.axis("square")
    plt.legend(labels=["outliers", "inliers"], title="true class")
    plt.show()
