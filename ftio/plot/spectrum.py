"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Sep 2024

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


def plot_spectrum(
    amp: np.ndarray,
    freq: np.ndarray,
    mode: str = "Amplitude",
    percent: bool = False,
):
    template = "plotly"
    name = "Amplitude" if "amp" in mode.lower() else "Power"
    unit = ""
    if percent:
        name = "Normed " + name
        unit = unit + " (%)"
        if amp.sum() > 0:
            amp = 100 * amp / amp.sum()

    df_tmp = pd.DataFrame(
        data={
            "A": amp,
            "freq": freq,
        }
    )
    # fig_tmp = px.bar(df_tmp, x="freq", y="A", color="A", color_continuous_scale='Bluered')
    # fig_tmp = px.bar(df_tmp, x="freq", y="A", color="A", color_continuous_scale=["rgb(0,10,130)", "rgb(130,30,130)", "rgb(255,50,50)"])
    # fig_tmp = px.bar(df_tmp, x="freq", y="A", color="A", color_continuous_scale="plasma")
    # cyan - purple
    # fig_tmp = px.bar(df_tmp, x="freq", y="A", color="A", color_continuous_scale=[ "rgb(90, 40, 90)", "rgb(113, 221, 255)" ])
    fig_tmp = px.bar(
        df_tmp,
        x="freq",
        y="A",
        color="A",
        color_continuous_scale=[
            "rgb(0,50,150)",
            "rgb(150,50,150)",
            "rgb(255,50,0)",
        ],
    )

    fig_tmp.update_traces(
        marker_line={"width": 0.2, "color": "black"},
        hovertemplate="<b>Frequency</b>: %{x:.2e} Hz <br><b>"
        + f"{name}"
        + "</b>: %{y:.2e}"
        + f"{unit}",
    )
    fig_tmp.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title=name + unit,
        coloraxis_colorbar={"yanchor": "top", "y": 1, "x": 0, "ticks": "outside"},
        template=template,
    )
    fig_tmp = format_plot(fig_tmp)
    return fig_tmp, name + unit


def plot_both_spectrums(args, freq: np.ndarray, amp: np.ndarray, full: bool = True):
    fig_1 = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{"colspan": 2}, None]],
    )

    start = 1

    if not full:
        indecies = np.arange(start, int(len(amp) / 2) + 1)
        freq = freq[indecies]
        amp = 2 * amp[indecies]
        amp[-1] = amp[-1] / 2
        if start == 0:
            amp[0] = amp[0] / 2

    if args.psd:
        power = amp
        amplitude = np.sqrt(amp) * len(amp)
    else:
        power = amp * amp / len(amp)
        amplitude = amp

    fig_tmp, name_plt = plot_spectrum(amplitude, freq, "Amplitude", False)
    layout = fig_tmp.layout
    for trace in list(fig_tmp.select_traces()):
        trace.update(marker={"coloraxis": "coloraxis"})
        if hasattr(fig_1, "add_trace"):
            fig_1.add_trace(trace, row=1, col=1)
        else:
            fig_1.append_trace(trace, 1, 1)
    fig_tmp, name_plt1 = plot_spectrum(power, freq, "Power", True)
    for trace in list(fig_tmp.select_traces()):
        trace.update(marker={"coloraxis": "coloraxis2"})
        if hasattr(fig_1, "add_trace"):
            fig_1.add_trace(trace, row=2, col=1)
        else:
            fig_1.append_trace(trace, 2, 1)
    fig_1.update_layout(layout)
    fig_1.update_layout(
        coloraxis={
            "colorbar": {
                "x": 1,
                "len": 0.4,
                "y": 0.8,
            },
        },
        coloraxis2={
            "colorbar": {
                "x": 1,
                "len": 0.4,
                "y": 0.2,
            },
        },
    )

    return fig_1, [name_plt, name_plt1]


def plot_one_spectrum(
    psd_flag: bool,
    freq: np.ndarray,
    amp: np.ndarray,
    full: bool = True,
    percent: bool = False,
) -> go.Figure:
    if not full:
        n_samples = len(freq)
        last = int(n_samples / 2) - 1
        amp = 2 * amp[0 : last + 1]
        amp[last] = amp[last] / 2
        amp[0] = amp[0] / 2
        freq = freq[0 : int(n_samples / 2)]

    if psd_flag:
        amp = amp * amp / len(amp)
        mode = "Power"
    else:
        mode = "Amplitude"

    fig_tmp, _ = plot_spectrum(amp, freq, mode, percent)
    return fig_tmp
