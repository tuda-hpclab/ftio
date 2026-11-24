"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mär 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from argparse import Namespace

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ftio.plot.helper import format_plot


def plot_autocorr_results(
    args: Namespace,
    acorr: np.ndarray,
    peaks: np.ndarray,
    outliers: np.ndarray,
    flag: bool = False,
):
    """
    Dispatches autocorrelation plot rendering based on the specified engine.

    Args:
        args (Namespace): Command line arguments, must contain 'engine'.
        acorr (np.ndarray): Autocorrelation values.
        peaks (np.ndarray): Indices of the peaks in the autocorrelation.
        outliers (np.ndarray): Indices of the outliers in the peaks.
        flag (bool, optional): Whether to highlight relevant peaks. Defaults to False.

    Returns:
        Figure: Plotly or Matplotlib figure object depending on the engine.
    """

    fig = None
    if any(x in args.engine for x in ["mat", "plot"]):
        if "mat" in args.engine:
            fig = plot_matplotlib_autocorr_results(acorr, peaks, outliers, flag)
        elif "plot" in args.engine:
            fig = plot_plotly_autocorr_results(acorr, peaks, outliers, flag)

    return fig


def plot_plotly_autocorr_results(
    acorr: np.ndarray,
    peaks: np.ndarray,
    outliers: np.ndarray,
    flag: bool = False,
) -> go.Figure:
    """
    Creates a Plotly figure for autocorrelation data.

    Args:
        acorr (np.ndarray): Autocorrelation values.
        peaks (np.ndarray): Indices of the peaks.
        outliers (np.ndarray): Indices of outlier peaks.
        flag (bool, optional): Whether to plot relevant peaks. Defaults to False.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()
    fig.add_scatter(
        y=acorr,
        mode="markers+lines",
        name="ACF",
        marker={
            "color": acorr,
            "colorscale": ["rgb(0,50,150)", "rgb(150,50,150)", "rgb(255,50,0)"],
            "showscale": True,
        },
    )

    fig.update_layout(
        font={
            "family": "Courier New, monospace",
            "size": 24,
            "color": "black",
        },
        xaxis_title="Lag (Samples)",
        yaxis_title="ACF",
        width=1100,
        height=400,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        coloraxis_colorbar={
            "yanchor": "top",
            "y": 1,
            "x": 0,
            "ticks": "outside",
            "ticksuffix": " bills",
        },
    )

    # Plot peaks
    fig.add_scatter(
        x=peaks,
        y=acorr[peaks],
        marker={
            "color": "rgba(20, 220, 70, 0.9)",
            "size": 14,
            "symbol": "star-triangle-up",
            "line": {"width": 1, "color": "DarkSlateGrey"},
        },
        mode="markers",
        name="peaks",
    )

    # Plot relevant peaks (filtered)
    if flag:
        val = np.delete(peaks, outliers)
        fig.add_scatter(
            x=val,
            y=acorr[val],
            marker={
                "color": "rgba(220, 20, 70, 0.9)",
                "size": 21,
                "symbol": "circle-open-dot",
                "line": {"width": 2, "color": "DarkSlateGrey"},
            },
            mode="markers",
            name="relevant peaks",
        )

    return format_plot(fig)


def plot_matplotlib_autocorr_results(
    acorr: np.ndarray,
    peaks: np.ndarray,
    outliers: np.ndarray,
    flag: bool = False,
) -> matplotlib.figure.Figure:
    """
    Creates a Matplotlib figure for autocorrelation data.

    Args:
        acorr (np.ndarray): Autocorrelation values.
        peaks (np.ndarray): Indices of the peaks.
        outliers (np.ndarray): Indices of outlier peaks.
        flag (bool, optional): Whether to plot relevant peaks. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure.
    """
    fig = plt.figure(figsize=(11, 4))
    ax = plt.gca()
    ax.plot(acorr, marker="o", linestyle="-", label="ACF", color="purple")

    # Plot peaks
    ax.scatter(
        peaks,
        acorr[peaks],
        color="green",
        s=100,
        marker="*",
        label="peaks",
        edgecolors="black",
    )

    # Plot relevant peaks
    if flag:
        val = np.delete(peaks, outliers)
        ax.scatter(
            val,
            acorr[val],
            facecolors="none",
            edgecolors="red",
            s=150,
            label="relevant peaks",
        )

    ax.set_xlabel("Lag (Samples)")
    ax.set_ylabel("ACF")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig
