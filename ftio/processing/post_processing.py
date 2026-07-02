"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ftio.freq.prediction import Prediction
from ftio.plot.helper import format_plot


def label_phases(
    prediction: Prediction,
    args,
    b0: np.ndarray = np.array([]),
    t0: np.ndarray = np.array([]),
):
    """Labels the phases using the result from FTIO

    Args:
        prediction (dict): prediction from FTIO
        args (Argparse): arguments passed to FTIO
        b0 (np.ndarray, optional): original bandwidth. Defaults to np.array([]).
        t0 (np.ndarray, optional): original time. Defaults to np.array([]).

    Returns:
        _type_: _description_
    """
    # just take the dominant or consider more than one
    only0ne = False
    n_waves = 1

    t = np.arange(prediction.t_start, prediction.t_end, 1 / prediction.freq)

    if only0ne:
        dominant_index = np.argmax(prediction.conf)
        # conf = prediction.conf[dominant_index]
        f = prediction.dominant_freq[dominant_index]
        amp = prediction.amp[dominant_index]
        phi = prediction.phi[dominant_index]
        n = np.floor((prediction.t_end - prediction.t_start) * prediction.freq)

        ## create cosine wave
        cosine_wave = 2 * amp / n * np.cos(2 * np.pi * f * t + phi)
    else:
        cosine_wave = np.zeros(len(t))
        n_waves = (
            len(prediction.conf)
            if args.n_freq == 0
            else len(prediction.top_freqs["freq"])
        )
        print(f"merging {n_waves} frequencies")

        # iterate over all dominant frequencies or take the top_frequencies if args.n_freq is above 0
        for index, _ in enumerate(
            prediction.conf if args.n_freq == 0 else prediction.top_freqs["freq"]
        ):
            # conf = prediction.conf[index]
            f = (
                prediction.dominant_freq[index]
                if args.n_freq == 0
                else prediction.top_freqs["freq"][index]
            )
            amp = (
                prediction.amp[index]
                if args.n_freq == 0
                else prediction.top_freqs["amp"][index]
            )
            phi = (
                prediction.phi[index]
                if args.n_freq == 0
                else prediction.top_freqs["phi"][index]
            )
            n = np.floor((prediction.t_end - prediction.t_start) * prediction.freq)

            # skip frequency at 0
            if f == 0 and args.n_freq != 0:
                continue

            ## create cosine wave
            cosine_wave = cosine_wave + 2 * amp / n * np.cos(2 * np.pi * f * t + phi)

    ## make square signal
    square_wave = np.zeros(len(cosine_wave))
    top = np.max(cosine_wave)
    for i, x in enumerate(cosine_wave):
        if x > 0:
            square_wave[i] = top
        else:
            square_wave[i] = -top

    phases = []
    tmp = []
    for i, x in enumerate(square_wave):
        if i == 0:
            counter = 0
            mem = x
            phases.append(counter)
            tmp.append(t[i])
        else:
            if mem == x:
                pass
            else:
                mem = x
                counter += 1
                phases.append(counter)
                tmp.append(t[i])

    # Assign in not empty
    if tmp:
        # add end time
        tmp.append(t[-1])
        time = {"t_s": tmp[0:-1], "t_e": tmp[1:]}
    else:
        time = {"t_s": [], "t_e": []}

    ## Plot
    _ = plot_classification(args, cosine_wave, t, time, n_waves, b0, t0)

    return phases, time


def plot_classification(
    args, cosine_wave, t, time, n_waves=0, b0=np.array([]), t0=np.array([])
) -> go.Figure | None:
    fig = None
    if any(x in args.engine for x in ["mat", "plot"]):
        fig = go.Figure()
        if n_waves == 1:
            name = "Dominant wave"
        elif args.n_freq == 0:
            name = "Dominant waves"
        else:
            name = f"{n_waves} superposed <br>cosine waves"
        fig.add_trace(go.Scatter(x=t, y=cosine_wave, name=name, marker_color="red"))
        # fig.add_trace(go.Scatter(x=t, y=square_wave, name="square wave"))
        fig.add_hline(y=0, line_width=1, line_color="gray")
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.G10
        for i, _ in enumerate(time["t_s"]):
            if i == len(colors):
                print("Not enough colors")
                break
            fig.add_vrect(
                x0=time["t_s"][i],
                x1=time["t_e"][i],
                annotation_text=f"phase {i}",
                annotation_position="top",
                fillcolor=colors[i],
                opacity=0.25,
                line_width=0,
            )

        if b0.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=t0,
                    y=b0,
                    mode="lines+markers",
                    name="Original signal",
                    marker_color="rgb(0,150,250)",
                )
            )

        fig.update_layout(
            xaxis_title="Ranks",
            yaxis_title="Transfer Rate (B/s)",
            width=1800,
            height=600,
        )
        fig = format_plot(fig)
        fig.show()

        return fig
