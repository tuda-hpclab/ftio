"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

from argparse import Namespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rich.console import Console

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.prediction import Prediction
from ftio.plot.helper import format_plot
from ftio.plot.spectrum import plot_one_spectrum
from ftio.plot.units import set_unit

# import plotly.io as pio
matplotlib.rcParams["backend"] = "TkAgg"


def plot_dft(
    args: Namespace,
    prediction: Prediction = None,
    analysis_figures: AnalysisFigures = None,
):
    if not any(x in args.engine for x in ["mat", "plot"]):
        return
    console = Console()
    console.print(
        f"\n[underline cyan]Plotting:[/]\n"
        f"[cyan]Plot render:[/] {args.render}\n"
        f"[cyan]Plot engine:[/] {args.engine}\n"
    )

    amp = analysis_figures.amp
    freq = analysis_figures.freqs
    phi = analysis_figures.phi
    t_sampled = analysis_figures.t_sampled
    b_sampled = analysis_figures.b_sampled
    b = analysis_figures.b
    t = analysis_figures.t

    # set the unit and order
    unit, order = set_unit(b_sampled)
    # unit, order = set_unit(b_sampled*1000) #compatibility with old version (For BWLIMIT)

    N = len(freq)

    sum_all_components = np.zeros_like(t_sampled)
    samples = np.arange(N)
    for k in range(N // 2):
        a = amp[k] / N
        if k != 0 and not (N % 2 != 0 and k == N - 1):
            a *= 2
        x = a * np.cos(2 * np.pi * samples * k / N + phi[k])
        x = a * np.cos(2 * np.pi * freq[k] * (t_sampled) + phi[k])
        sum_all_components += x

    # dominant freq
    dominant_freq, dominant_amp, dominant_phi = prediction.get_dominant_freq_amp_phi()
    dominant_signal = None
    dominant_name = ""
    if not np.isnan(dominant_freq):
        x, wave_name = prediction.get_wave_and_name(
            dominant_freq, dominant_amp, dominant_phi
        )
        if not dominant_name:
            dominant_signal = x
            dominant_name = wave_name
        else:
            dominant_signal += x
            dominant_name += " + " + wave_name

    # For reconstruction
    sum_top = {}
    top_freqs = prediction.top_freqs
    if args.reconstruction:
        for top_index in args.reconstruction:
            sum_top[f"Recon. top {top_index} signal"] = np.zeros_like(t_sampled)
            for k in range(top_index):
                x = prediction.get_wave(
                    top_freqs["freq"][k], top_freqs["amp"][k], top_freqs["phi"][k]
                )
                sum_top[f"Recon. top {top_index} signal"] += x

    # For plotting top 3 signals isolated
    n = 3
    top_signals = []
    top_names = []
    if len(freq) > n:
        if args.n_freq and args.n_freq >= n:
            top_freqs = prediction.top_freqs
        else:
            top_candidates = np.argsort(-amp)
            ids = top_candidates[:n]
            top_freqs = {"freq": freq[ids], "amp": amp[ids], "phi": phi[ids]}
        for k in range(n):
            wave, wave_name = prediction.get_wave_and_name(
                top_freqs["freq"][k], top_freqs["amp"][k], top_freqs["phi"][k]
            )
            top_signals.append(wave)
            top_names.append(wave_name)

    if "mat" in args.engine:
        from ftio.freq.duty_cycle import overlay_burst_widths_matplotlib

        f_top = plot_dft_matplotlib_top(
            args,
            order,
            unit,
            b_sampled,
            t_sampled,
            b,
            t,
            sum_all_components,
            top_signals,
            top_names,
        )
        overlay_burst_widths_matplotlib(prediction)

        f_dom = plot_dft_matplotlib_dominant(
            args,
            order,
            unit,
            b_sampled,
            t_sampled,
            b,
            t,
            dominant_signal,
            dominant_name,
            sum_top,
        )
        overlay_burst_widths_matplotlib(prediction)

        f = [f_top, f_dom, plot_dft_matplotlib_spectrum(args, freq, amp)]

    else:
        f = [
            plot_dft_plotly_top(
                args,
                order,
                unit,
                b_sampled,
                t_sampled,
                b,
                t,
                sum_all_components,
                top_signals,
                top_names,
            ),
            plot_dft_plotly_dominant(
                args,
                order,
                unit,
                b_sampled,
                t_sampled,
                b,
                t,
                dominant_signal,
                dominant_name,
                sum_top,
            ),
            plot_dft_plotly_spectrum(args, freq, amp),
        ]

        # create_html(f, args.render, {"toImageButtonOptions": {"format": "png", "scale": 4}}, args.transformation)

    analysis_figures.add_figure_and_show(f, "dft")


def plot_dft_matplotlib_top(
    args,
    order,
    unit,
    b_sampled,
    t_sampled,
    b,
    t,
    sum_all_components,
    top_signals,
    top_names,
):
    f = plt.figure(figsize=(10, 4))
    for i, signal in enumerate(top_signals):
        plt.plot(t_sampled, order * signal, linewidth=0.9, label=top_names[i])

    plt.plot(
        t_sampled,
        order * sum_all_components,
        linestyle="-",
        marker=".",
        color="deepskyblue",
        label="Reconstructed signal",
    )
    plt.xlim(t_sampled[0], t_sampled[-1])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(f"Bandwidth ({unit})", fontsize=17)
    plt.xlabel("Time (s)", fontsize=17)
    plt.grid(True)
    plt.legend(loc="upper left", ncol=2, fontsize=13)
    plt.tight_layout()
    return f


def plot_dft_matplotlib_dominant(
    args,
    order,
    unit,
    b_sampled,
    t_sampled,
    b,
    t,
    dominant_signal,
    dominant_name,
    sum_top,
):
    f = plt.figure(figsize=(10, 4))
    plt.fill_between(
        t,
        0,
        b * order,
        label="Original signal",
        alpha=0.5,
        step="post",
        color="royalblue",
    )
    plt.plot(
        t,
        b * order,
        drawstyle="steps-post",
        color="royalblue",
    )
    plt.fill_between(
        t_sampled,
        0,
        b_sampled * order,
        label="Sampled signal",
        alpha=0.6,
        step="post",
        color="red",
    )
    plt.plot(
        t_sampled,
        b_sampled * order,
        drawstyle="steps-post",
        color="red",
    )
    if dominant_name:
        plt.fill_between(
            t_sampled,
            0,
            y2=dominant_signal * order,
            label=dominant_name,
            alpha=0.6,
            step="post",
            color="limegreen",
        )
        plt.plot(
            t_sampled,
            dominant_signal * order,
            drawstyle="steps-post",
            color="limegreen",
        )

    if args.reconstruction:
        colors = [
            "gold",
            "cyan",
            "purple",
            "slategrey",
            "darkorange",
            "green",
            "blue",
            "red",
            "magenta",
            "brown",
            "pink",
            "yellowgreen",
            "lightblue",
            "indigo",
            "teal",
            "orchid",
            "crimson",
        ]

        for i, (name, signal) in enumerate(sum_top.items()):
            color = colors[i % len(colors)]  # cycle colors

            # Fill the area between the curve and the x-axis
            plt.fill_between(
                t_sampled,
                0,
                y2=signal * order,
                label=name,
                alpha=0.6,
                step="post",
                color=color,
            )

            # Plot the line on top of the fill
            plt.plot(t_sampled, signal * order, drawstyle="steps-post", color=color)
    plt.xlim(t_sampled[0], t_sampled[-1])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(f"Bandwidth ({unit})", fontsize=17)
    plt.xlabel("Time (s)", fontsize=17)
    plt.grid(True)
    plt.legend(loc="upper left", ncol=1, fontsize=13)
    plt.title("FTIO DFT results", fontsize=17)
    plt.tight_layout()
    return f


def plot_dft_matplotlib_spectrum(args, freq, amp):
    f = plt.figure(figsize=(10, 4))
    # settings
    percent = True

    mode = "Amplitude"
    if args.psd:
        amp = amp * amp / len(amp)
        mode = "Power"

    if percent:
        mode = "Normed " + mode + " (%)"
        amp = 100 * amp / amp.sum()

    plt.bar(
        freq,
        amp,
        edgecolor="k",
        width=freq[1] - freq[0],
    )
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(mode, fontsize=17)
    plt.xlabel("Frequency (Hz)", fontsize=17)
    plt.grid(True)
    plt.legend(loc="upper left", ncol=2, fontsize=13)
    plt.tight_layout()
    return f


def plot_dft_plotly_spectrum(args, freq, amp):
    f = plot_one_spectrum(args.psd, freq, amp, False, True)
    f.update_layout(
        font={
            "family": "Courier New, monospace",
            "size": 24,
            "color": "black",
        },
        width=1100,
        height=600,
        coloraxis_colorbar={
            "yanchor": "top",
            "y": 1,
            "x": 1,
            "ticks": "outside",
            "title": "",
        },
        # title="Spectrum (Ranks %i)" % r
    )
    # fig_tmp.show(config=conf)
    return f


def plot_dft_plotly_top(
    args,
    order,
    unit,
    b_sampled,
    t_sampled,
    b,
    t,
    sum_all_components,
    top_signals,
    top_names,
):
    # For faster scatter plot
    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    template = "plotly"
    if "dark" in template:
        pass
    else:
        pass
    paper = True
    if "no_paper" in args.engine:
        paper = False
    colors = list(px.colors.qualitative.Plotly)
    colors.pop(1)
    width = 1100
    height = 600  # 600
    f = go.Figure()
    for i, signal in enumerate(top_signals):
        f.add_trace(
            Scatter(
                x=t_sampled,
                y=order * signal,
                mode="lines",
                name=top_names[i],
                hovertemplate="<b>Time</b>: %{x:.2f} s"
                + "<br><b>Amplitude</b>: %{y}"
                + "<br><b>T</b>: %{text} s",
                text=len(signal) * [f"{1/args.freq:.2f}"],
                marker_color=(colors[i] if i != 1 else "rgb(70,220,70)"),
            )
        )
    f.add_trace(
        Scatter(
            x=t_sampled,
            y=sum_all_components * order,
            mode="lines+markers",
            name="Sampled signal",
            line={"shape": "hv"},
            marker_color="rgb(180,30,30)",
        )
    )
    f.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        width=width if paper else 1.5 * width,
        height=height / 1.1,
        template=template,
    )
    if paper:
        f.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 0.01,
            }
        )
    f.update_xaxes(range=[t_sampled[0], t_sampled[-1]])
    f = format_plot(f)
    return f


def plot_dft_plotly_dominant(
    args,
    order,
    unit,
    b_sampled,
    t_sampled,
    b,
    t,
    dominant_signal,
    dominant_name,
    sum_top,
):
    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    template = "plotly"
    if "dark" in template:
        pass
    else:
        pass
    paper = True
    if "no_paper" in args.engine:
        paper = False
    colors = list(px.colors.qualitative.Plotly)
    colors.pop(1)
    width = 1100
    height = 600  # 600
    font_settings = {
        "family": "Courier New, monospace",
        "size": 24,
        "color": "black",
    }
    f = go.Figure()
    fill = "tozeroy"
    console = Console()
    if args.render == "dynamic" or len(t_sampled) < 1e3:
        f.add_trace(
            Scatter(
                x=t,
                y=b * order,
                mode="lines+markers",
                name="Original signal",
                fill=fill,
                line={"shape": "hv"},
                marker_color="rgb(0,150,250)",
            )
        )
    else:
        console.print(
            "[orange]Too many data points for original signal[/]\n"
            "[orange]  skipping plotting it[/]\n"
            "[orange]  use ftio ... -e mat to show this plot or set ftio ... -re static[/]\n"
        )
    # f.add_trace(Scatter(x=time,y=sum,mode='lines+markers',name="Reconstructed signal",fill=fill, line={"shape": 'hv'},visible=visible))
    f.add_trace(
        Scatter(
            x=t_sampled,
            y=b_sampled * order,
            mode="lines+markers",
            name="Sampled signal",
            fill=fill,
            line={"shape": "hv"},
            marker_color="rgb(180,30,30)",
        )
    )
    if len(t_sampled) > 1000:
        fill = None
    if dominant_name:
        f.add_trace(
            Scatter(
                x=t_sampled,
                y=dominant_signal * order,
                mode="lines+markers",
                name=dominant_name,
                fill=fill,
                line={"shape": "hv"},
                marker_color="rgb(70,220,70)",
            )
        )
    if args.reconstruction:
        color_counter = 0
        for name, signal in sum_top.items():
            f.add_trace(
                trace=Scatter(
                    x=t_sampled,
                    y=signal * order,
                    mode="lines+markers",
                    name=name,
                    fill=fill,
                    line={"shape": "hv"},
                    visible="legendonly",
                    marker_color=(
                        colors[color_counter] if color_counter != 1 else "rgb(70,220,70)"
                    ),
                )
            )

    f.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        font=font_settings,
        # width=1.05*width,
        width=width if paper else 1.5 * width,
        height=height / 1.1,
        # title="Time Plot (Ranks %i)" % r,
        template=template,
    )
    if paper:
        f.update_layout(
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01}
        )
    # else:
    #     f.update_layout(
    #         legend=dict(yanchor="top", y=0.99, xanchor="right", x=.99)
    #     )

    f.update_xaxes(range=[t_sampled[0], t_sampled[-1]])
    # if isinstance(sum_dominant, list):
    #     f.update_yaxes(
    #         range=[
    #             min(sum_dominant),
    #             1.2 *
    #             b.max(),
    #         ]
    #     )
    return format_plot(f)
