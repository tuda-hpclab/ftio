"""
Plotting functions for ASTFT, VMD, and EFD (AMD-based methods).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ftio.plot.helper import format_plot
from ftio.plot.units import set_unit

FONT_SIZE = 17
FONT_SIZE_ANNOTATION = int(FONT_SIZE * 2 / 3)


def plot_amd_components(
    args, signal, t, components, title="AMD components", b_orig=None, t_orig=None
):
    """Plots original signal, sampled signal and reconstructed components."""
    if not any(x in args.engine for x in ["mat", "plot"]):
        return None

    if "mat" in args.engine:
        return [
            plot_amd_components_matplotlib(signal, t, components, title, b_orig, t_orig),
            plot_reconstructed_waves_matplotlib(
                signal, t, components, "Identified waves"
            ),
        ]
    elif "plotly" in args.engine:
        return [
            plot_amd_components_plotly(
                args, signal, t, components, title, b_orig, t_orig
            ),
            plot_reconstructed_waves_plotly(
                args, signal, t, components, "Identified waves"
            ),
        ]
    return None


def plot_imfs(args, signal, t, u, title="Intrinsic mode functions"):
    """Plots the signal and its decomposition into IMFs."""
    if not any(x in args.engine for x in ["mat", "plot"]):
        return None

    if "mat" in args.engine:
        return plot_imfs_matplotlib(signal, t, u, title)
    elif "plotly" in args.engine:
        return plot_imfs_plotly(args, signal, t, u, title)
    return None


def plot_amd_components_matplotlib(
    signal, t, components, title, b_orig=None, t_orig=None
):
    unit, order = set_unit(signal if b_orig is None else b_orig)
    fig = plt.figure(figsize=(10, 6))

    # Original signal (same as DFT/STFT)
    if b_orig is not None and t_orig is not None:
        min_len_orig = min(len(t_orig), len(b_orig))
        plt.fill_between(
            t_orig[:min_len_orig],
            0,
            b_orig[:min_len_orig] * order,
            label="Original signal",
            alpha=0.5,
            step="post",
            color="royalblue",
        )
        plt.plot(
            t_orig[:min_len_orig],
            b_orig[:min_len_orig] * order,
            drawstyle="steps-post",
            color="royalblue",
        )

    # Sampled signal (same as DFT/STFT - Red and Filled)
    plt.fill_between(
        t, 0, signal * order, label="Sampled signal", alpha=0.6, step="post", color="red"
    )
    plt.plot(t, signal * order, drawstyle="steps-post", color="red")

    colors = ["limegreen", "orange", "magenta", "cyan", "gold", "pink", "brown"]
    for i, p in enumerate(components):
        if hasattr(p, "freq"):
            freq, amp, phase, start, end = p.freq, p.amp, p.phase, p.start, p.end
        else:
            start, end, freq = p[0][0], p[0][1], p[2]
            amp, phase = 0, 0

        color = colors[i % len(colors)]
        label = f"f_{i} = {freq:.3e} Hz"
        if amp > 0:
            estimate = amp * np.cos(2 * np.pi * freq * t + phase)
            plt.plot(
                t[start:end],
                estimate[start:end] * order,
                label=label,
                linewidth=2,
                color=color,
            )
        else:
            plt.axvspan(
                t[start], t[end - 1], alpha=0.1, color=color, label=f"{label} (range)"
            )

    plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(f"Bandwidth ({unit})", fontsize=FONT_SIZE)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    return fig


def plot_amd_components_plotly(
    args, signal, t, components, title, b_orig=None, t_orig=None
):
    unit, order = set_unit(signal if b_orig is None else b_orig)
    fig = go.Figure()

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    # Original signal
    if b_orig is not None and t_orig is not None:
        min_len_orig = min(len(t_orig), len(b_orig))
        fig.add_trace(
            Scatter(
                x=t_orig[:min_len_orig],
                y=b_orig[:min_len_orig] * order,
                mode="lines",
                name="Original signal",
                line={"color": "rgb(0,150,250)", "shape": "hv"},
                fill="tozeroy",
            )
        )

    # Sampled signal (same as DFT - Red and Filled)
    fig.add_trace(
        Scatter(
            x=t,
            y=signal * order,
            mode="lines",
            name="Sampled signal",
            line={"color": "rgb(180,30,30)", "shape": "hv"},
            fill="tozeroy",
        )
    )

    colors = list(px.colors.qualitative.Plotly)
    colors = colors[2:] + colors[:2]

    for i, p in enumerate(components):
        if hasattr(p, "freq"):
            freq, amp, phase, start, end = p.freq, p.amp, p.phase, p.start, p.end
        else:
            start, end, freq = p[0][0], p[0][1], p[2]
            amp = 0

        color = colors[i % len(colors)]
        label = f"f_{i} = {freq:.3e} Hz"
        if amp > 0:
            estimate = amp * np.cos(2 * np.pi * freq * t + phase)
            fig.add_trace(
                Scatter(
                    x=t[start:end],
                    y=estimate[start:end] * order,
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 3, "shape": "hv"},
                )
            )
        else:
            fig.add_vrect(
                x0=t[start],
                x1=t[end - 1],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=FONT_SIZE_ANNOTATION,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        template="plotly",
    )
    return format_plot(fig)


def plot_reconstructed_waves_matplotlib(signal, t, components, title):
    unit, order = set_unit(signal)
    fig = plt.figure(figsize=(10, 6))

    colors = ["limegreen", "orange", "magenta", "cyan", "gold", "pink", "brown"]
    any_plotted = False
    for i, p in enumerate(components):
        if hasattr(p, "freq") and p.amp > 0:
            estimate = p.amp * np.cos(2 * np.pi * p.freq * t + p.phase)
            plt.plot(
                t[p.start : p.end],
                estimate[p.start : p.end] * order,
                label=f"f_{i} = {p.freq:.3e} Hz",
                linewidth=2,
                color=colors[i % len(colors)],
            )
            any_plotted = True

    if not any_plotted:
        plt.text(
            0.5,
            0.5,
            "No sinusoidal waves identified",
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
        )

    plt.title(title, fontsize=FONT_SIZE)
    plt.ylabel(f"Bandwidth ({unit})", fontsize=FONT_SIZE)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    if any_plotted:
        plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    return fig


def plot_reconstructed_waves_plotly(args, signal, t, components, title):
    unit, order = set_unit(signal)
    fig = go.Figure()

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    colors = list(px.colors.qualitative.Plotly)
    colors = colors[2:] + colors[:2]

    for i, p in enumerate(components):
        if hasattr(p, "freq") and p.amp > 0:
            estimate = p.amp * np.cos(2 * np.pi * p.freq * t + p.phase)
            fig.add_trace(
                Scatter(
                    x=t[p.start : p.end],
                    y=estimate[p.start : p.end] * order,
                    mode="lines",
                    name=f"f_{i} = {p.freq:.3e} Hz",
                    line={"color": colors[i % len(colors)], "width": 3, "shape": "hv"},
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        template="plotly",
    )
    return format_plot(fig)


def plot_imfs_matplotlib(signal, t, u, title):
    K = u.shape[0]
    fig, axes = plt.subplots(K + 1, 1, figsize=(10, 2 * (K + 1)), sharex=True)

    axes[0].plot(t, signal, color="royalblue")
    axes[0].set_title("Original signal", fontsize=FONT_SIZE)

    for i in range(K):
        t_plot = t[: len(u[i])]
        axes[i + 1].plot(t_plot, u[i], color="black")
        axes[i + 1].set_ylabel(f"IMF {i}", fontsize=FONT_SIZE)

    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    fig.suptitle(title, fontsize=FONT_SIZE + 2)
    plt.tight_layout()
    return fig


def plot_imfs_plotly(args, signal, t, u, title):
    K = u.shape[0]
    fig = make_subplots(
        rows=K + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Original signal"] + [f"IMF {i}" for i in range(K)],
    )

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    fig.add_trace(
        Scatter(x=t, y=signal, name="Original", line={"color": "royalblue"}), row=1, col=1
    )

    for i in range(K):
        t_plot = t[: len(u[i])]
        fig.add_trace(
            Scatter(x=t_plot, y=u[i], name=f"IMF {i}", line={"color": "black"}),
            row=i + 2,
            col=1,
        )

    fig.update_layout(height=200 * (K + 1), title_text=title, template="plotly")
    return format_plot(fig)
