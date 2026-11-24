"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ftio.api.metric_proxy.phasemode import PhaseMode
from ftio.freq.helper import MyConsole
from ftio.plot.helper import format_plot_and_ticks
from ftio.plot.print_html import PrintHtml

CONSOLE = MyConsole()
CONSOLE.set(True)


def phases_and_timeseries(metrics, data, argv=None):
    if argv is None:
        argv = []
    phasemode_list, t = classify_waves(data, True)

    if argv and "-n" in argv:
        n = argv[argv.index("-n") + 1]
        plot_waves_and_timeseries(argv, metrics, phasemode_list, t, n)
    else:
        plot_waves_and_timeseries(argv, metrics, phasemode_list, t)


def phases(data, argv=None):
    if argv is None:
        argv = []
    phasemode_list, t = classify_waves(data)

    if argv and "-n" in argv:
        n = argv[argv.index("-n") + 1]
        plot_waves(argv, phasemode_list, t, n)
    else:
        plot_waves(argv, phasemode_list, t)


def classify_waves(data, normed=True):
    phasemode_list = []
    phasemode_list.append(
        PhaseMode(
            "Network",
            [
                "network",
                "mpi_bcast",
                "sock",
                "send",
                "listen",
                "recieve",
                "gather",
                "reduce",
                "cast",
            ],
        )
    )
    phasemode_list.append(PhaseMode("write", ["write"]))
    phasemode_list.append(PhaseMode("read", ["read"]))
    phasemode_list.append(
        PhaseMode(
            "I/O",
            [
                "io",
                "disk",
                "stat",
                "cntl",
                "read",
                "write",
                "openat",
                "mmap",
                "seek",
                "close",
            ],
        )
    )
    phasemode_list.append(PhaseMode("Temperature", ["temp"]))
    phasemode_list.append(PhaseMode("CPU", ["cpu"]))
    phasemode_list.append(PhaseMode("Wait", ["poll", "barrier"]))
    seen = []
    for mode in phasemode_list:
        seen.extend(mode.matches)
    other = [x.metric for x in data if all(y not in x.metric for y in seen)]
    phasemode_list.append(PhaseMode("Other", other))

    sampling_freq = np.nan
    t_s = np.inf
    t_e = 0
    for prediction in data:
        if len(prediction.dominant_freq) > 0 and len(prediction.conf) > 0:
            if np.isnan(sampling_freq):
                sampling_freq = prediction.freq
                t_s = min(prediction.t_start, t_s)
                t_e = max(prediction.t_end, t_e)

            add_metric(phasemode_list, prediction)

    if not np.isnan(sampling_freq):
        t = np.arange(t_s, t_e, 1 / sampling_freq)
        add_time(phasemode_list, t)
    else:
        CONSOLE.print(
            "[red] No metrics classified. Try increasing the sampling frequency[/]"
        )
        exit(0)

    text = print_len(phasemode_list, data)
    CONSOLE.print(f"[blue]{text}[/]")
    aggregate_wave_for_all_modes(phasemode_list, t, normed)

    return phasemode_list, t


def plot_waves(argv: list, arr: list[PhaseMode], t, n=None):
    plot_mode = False
    if argv and "-e" in argv:
        plot_mode = "mat" in argv[argv.index("-e") + 1]

    if plot_mode:
        # Create a figure and axis using variables
        fig, ax = plt.subplots()

        # Loop through each mode in arr and plot
        for mode in arr:
            ax.plot(t, mode.wave, marker="o", label=mode.name)  # Plot with markers

        # Set axis titles
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normed metrics")

        # Set the legend title if n is provided
        if n:
            ax.legend(title=f"{n} Frequencies")
        else:
            ax.legend()

        plt.show()
    else:
        fig = go.Figure()
        for mode in arr:
            fig.add_trace(
                go.Scatter(x=t, y=mode.wave, mode="lines+markers", name=mode.name)
            )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Normed metrics",
        )
        if n:
            fig.update_layout(legend_title_text=f"{n} Frequencies")
        fig = format_plot_and_ticks(fig)
        fig.show()
        # fig.write_image("waves.png")


def plot_waves_and_timeseries(argv: list, metrics: dict, arr: list[PhaseMode], t, n=None):
    names = []  # get_names(arr)

    out = PrintHtml("./", names, outdir="phase_plots")
    out.generate_html_start()
    # t = []
    for mode in arr:
        f = plot_mode(mode, metrics, t, n, True)
        # for fig in f:
        #     fig.show
        out.generate_html_core(mode.name + ".html", f)

    out.generate_html_end()


def plot_mode(mode, metrics, t, n, subfig=False) -> list[go.Figure]:

    f = []
    spec = [{}, {}]
    if subfig:
        f.append(make_subplots(rows=2, cols=1))
        spec = [{"row": 1, "col": 1}, {"row": 2, "col": 1}]
    else:
        f.append(go.Figure())

    f[-1].add_trace(
        go.Scatter(x=t, y=mode.wave, mode="lines+markers", name=mode.name),
        **spec[0],
    )

    if not subfig:
        f.append(go.Figure())

    for metric, arrays in metrics.items():
        if len(arrays[0]) > 1 and any(n in metric for n in mode.matches):
            f[-1].add_trace(
                go.Scatter(
                    x=arrays[1],
                    y=arrays[0],
                    mode="lines+markers",
                    name=metric,
                    hovertemplate="<i>Time </i>: %{x} s" + "<br><b>Metric</b>: %{y}<br>",
                    legendgroup=metric,
                ),
                **spec[1],
            )
            wave, name = mode.get_wave(metric)
            f[-1].add_trace(
                go.Scatter(
                    x=mode.get("t"),
                    y=wave,
                    mode="lines+markers",
                    name=name,
                    hovertemplate="<i>Time </i>: %{x} s" + "<br><b>Metric</b>: %{y}<br>",
                    legendgroup=metric,
                ),
                **spec[0],
            )

    for fig in f:
        fig = format_plot_and_ticks(fig, font=False)
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title=f"{mode.name.capitalize()} Metrics",
            width=1400,
            height=800,
            showlegend=True,
        )
    if subfig:
        fig.update_xaxes(title_text="Time (s)", **spec[1])
        fig.update_yaxes(title_text=f"{mode.name.capitalize()} Metrics", **spec[1])

        if f.index(fig) == 0:
            if n:
                fig.update_layout(
                    title=f"{mode.name.capitalize()}: Using {n} Frequencies per metric"
                )
            else:
                fig.update_layout(
                    title=f"{mode.name.capitalize()}: Using Dominant Frequencies"
                )
    return f


def add_metric(arr: list[PhaseMode], d):
    for mode in arr:
        if mode.match(d):
            mode.add(d)
            break


def add_time(arr: list[PhaseMode], t):
    for mode in arr:
        mode.set_time(t)


def get_names(arr: list[PhaseMode]) -> list[str]:
    names = []
    for mode in arr:
        names.append(mode.name)
    return names


def aggregate_wave_for_all_modes(arr: list[PhaseMode], t, normed=True):
    for mode in arr:
        mode.aggregates_waves(t, normed)


def print_len(arr: list[PhaseMode], data) -> str:
    text = "\n"
    total_metrics = 0
    for mode in arr:
        text += f"{len(mode.data):3} {mode.name} metrics\n"
        total_metrics += len(mode.data)

    text += f"---------------------\n{total_metrics:3}/{len(data):3} total metrics with dominant frequencies\n"

    return text
