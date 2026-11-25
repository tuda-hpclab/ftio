"""
This file provides functions to compute similarities between time series

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 26 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from ftio.freq.helper import MyConsole
from ftio.plot.helper import format_plot
from ftio.plot.units import set_unit


def correlation(x, y, method="pearson"):
    """
    Compute global correlation between two signals.

    Parameters:
        x, y   : Input 1D signals (must be same length)
        method : 'pearson' or 'spearman'

    Returns:
        r      : Correlation coefficient
    """
    if len(x) != len(y):
        raise ValueError("Signals must have the same length.")

    if method == "pearson":
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0, 1]

    elif method == "spearman":
        rho, _ = spearmanr(x, y)
        return rho if not np.isnan(rho) else 0

    else:
        raise ValueError("Unsupported method: choose 'pearson' or 'spearman'")


def sliding_correlation(x, y, window_size, method="pearson"):
    """
    Compute local correlation (Pearson or Spearman) in a sliding window.

    Parameters:
        x, y        : Input 1D signals (must be same length)
        window_size : Size of window in samples
        method      : 'pearson' or 'spearman'

    Returns:
        corrs       : Array of local correlation values
    """
    n = len(x)
    w = window_size
    corrs = np.zeros(n - w + 1)

    for i in range(len(corrs)):
        x_win = x[i : i + w]
        y_win = y[i : i + w]
        corrs[i] = correlation(x_win, y_win, method)

    return corrs


def plot_correlation(
    args,
    t,
    signal_1,
    signal_2,
    corrs,
    window_duration=None,
    name=None,
    prediction=None,
    b_orig=None,
    t_orig=None,
):
    """
    Plot input signals, sliding correlation, and their product.

    Parameters:
        args            : FTIO arguments
        t               : Time vector (1D array)
        signal_1        : First input signal (1D array)
        signal_2        : Second input signal (1D array)
        corrs           : Sliding correlation values (1D array)
        window_duration : Optional duration of the correlation window (float)
        name            : List of strings indicating the labels
        prediction      : Prediction object to extract ranges and waves
        b_orig          : Original bandwidth
        t_orig          : Original time
    """
    if "mat" in args.engine:
        return plot_correlation_matplotlib(
            t,
            signal_1,
            signal_2,
            corrs,
            window_duration,
            name,
            prediction,
            b_orig,
            t_orig,
        )
    elif "plotly" in args.engine:
        return plot_correlation_plotly(
            args,
            t,
            signal_1,
            signal_2,
            corrs,
            window_duration,
            name,
            prediction,
            b_orig,
            t_orig,
        )
    return None


def plot_correlation_matplotlib(
    t,
    signal_1,
    signal_2,
    corrs,
    window_duration=None,
    name=None,
    prediction=None,
    b_orig=None,
    t_orig=None,
):
    if name is None:
        name = ["Cosine", "Logical"]
    min_len = min(len(signal_1), len(signal_2), len(corrs))
    signal_1 = signal_1[:min_len]
    signal_2 = signal_2[:min_len]
    t_corr = t[:min_len]
    corrs = corrs[:min_len]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Signals
    unit, order = set_unit(signal_2 if b_orig is None else b_orig)

    if b_orig is not None and t_orig is not None:
        min_orig = min(len(b_orig), len(t_orig))
        axes[0].fill_between(
            t_orig[:min_orig],
            0,
            b_orig[:min_orig] * order,
            label="Original signal",
            alpha=0.3,
            color="royalblue",
            step="post",
        )
        axes[0].plot(
            t_orig[:min_orig],
            b_orig[:min_orig] * order,
            color="royalblue",
            alpha=0.4,
            drawstyle="steps-post",
        )

    axes[0].plot(t_corr, order * signal_1, label=f"{name[0]}", color="blue")

    # Sampled/Upsampled signal style (same as DFT)
    axes[0].fill_between(
        t_corr, 0, signal_2 * order, label=name[1], alpha=0.6, step="post", color="red"
    )
    axes[0].plot(
        t_corr,
        order * signal_2,
        label=f"{name[1]} (line)",
        drawstyle="steps-post",
        color="red",
    )

    # Masked Cosine / Identified Waves
    if prediction is not None and len(prediction.ranges) > 0:
        masked_signal = np.zeros_like(signal_1)
        for start, end in prediction.ranges:
            mask = (t_corr >= start) & (t_corr <= end)
            masked_signal[mask] = signal_1[mask]
        axes[0].plot(
            t_corr,
            order * masked_signal,
            color="limegreen",
            label="Masked cosine",
            linewidth=2,
        )

    axes[0].set_title("Signals from DWT and DFT", fontsize=17)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_ylabel(f"Bandwidth ({unit})", fontsize=15)
    axes[0].grid(True)

    # Plot 2: Correlation
    axes[1].plot(t_corr, corrs, color="purple")
    axes[1].axhline(0, color="gray", linestyle="--")
    title = "Sliding Window Correlation"
    if window_duration:
        title += f" (Window = {window_duration:.1f} s)"
    axes[1].set_title(title, fontsize=17)
    axes[1].set_ylabel("Correlation", fontsize=15)
    axes[1].grid(True)

    plt.xlabel("Time (s)", fontsize=15)
    plt.tight_layout()
    return fig


def plot_correlation_plotly(
    args,
    t,
    signal_1,
    signal_2,
    corrs,
    window_duration=None,
    name=None,
    prediction=None,
    b_orig=None,
    t_orig=None,
):
    if name is None:
        name = ["Cosine", "Logical"]
    min_len = min(len(signal_1), len(signal_2), len(corrs))
    signal_1 = signal_1[:min_len]
    signal_2 = signal_2[:min_len]
    t_corr = t[:min_len]
    corrs = corrs[:min_len]

    unit, order = set_unit(signal_2 if b_orig is None else b_orig)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Signals from DWT and DFT", "Sliding Window Correlation"),
    )

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    # Subplot 1: Signals
    # Original
    if b_orig is not None and t_orig is not None:
        min_orig = min(len(b_orig), len(t_orig))
        fig.add_trace(
            Scatter(
                x=t_orig[:min_orig],
                y=b_orig[:min_orig] * order,
                name="Original signal",
                line={"color": "rgb(0,150,250)", "shape": "hv"},
                fill="tozeroy",
                visible="legendonly",
            ),
            row=1,
            col=1,
        )

    # DFT Signal
    fig.add_trace(
        Scatter(x=t_corr, y=signal_1 * order, name=name[0], line={"color": "blue"}),
        row=1,
        col=1,
    )
    # DWT Sampled/Upsampled
    fig.add_trace(
        Scatter(
            x=t_corr,
            y=signal_2 * order,
            name=name[1],
            fill="tozeroy",
            line={"shape": "hv"},
            marker_color="rgb(180,30,30)",
        ),
        row=1,
        col=1,
    )

    # Masked Cosine
    if prediction is not None and len(prediction.ranges) > 0:
        masked_signal = np.zeros_like(signal_1)
        for start, end in prediction.ranges:
            mask = (t_corr >= start) & (t_corr <= end)
            masked_signal[mask] = signal_1[mask]

        fig.add_trace(
            Scatter(
                x=t_corr,
                y=masked_signal * order,
                name="Masked cosine",
                line={"color": "rgb(70,220,70)", "width": 3},
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(
        title_text=f"Bandwidth ({unit})", title_font={"size": 15}, row=1, col=1
    )

    # Subplot 2: Correlation
    fig.add_trace(
        Scatter(x=t_corr, y=corrs, name="Correlation", line={"color": "purple"}),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", title_font={"size": 15}, row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", title_font={"size": 15}, row=2, col=1)

    fig.update_layout(height=600, template="plotly")
    fig.update_layout(legend={"font": {"size": 10}})
    return format_plot(fig)


def extract_correlation_ranges(
    t: np.ndarray,
    corrs: np.ndarray,
    threshold_low: float = 0.12,
    threshold_high: float = 1.0,
    min_duration: float = 0.0,
    verbose: bool = False,
) -> list[tuple[float, float]]:
    """
    Extract continuous time ranges where correlation is between threshold_low and threshold_high,
    and discard segments shorter than min_duration.

    Parameters:
        t              : Time vector (1D array)
        corrs          : Correlation values (same length as t)
        threshold_low  : Lower threshold (inclusive)
        threshold_high : Upper threshold (inclusive)
        min_duration   : Minimum duration of a valid segment (in seconds)
        verbose        : If set, prints the result on the console

    Returns:
        List of (start_time, end_time) tuples
    """
    assert len(t) == len(corrs), "Time and correlation arrays must be the same length"
    mask = (corrs >= threshold_low) & (corrs <= threshold_high)

    # Find rising and falling edges of the mask
    edges = np.diff(mask.astype(int))
    start_indices = np.where(edges == 1)[0] + 1
    end_indices = np.where(edges == -1)[0] + 1

    # Handle edge cases where mask starts or ends True
    if mask[0]:
        start_indices = np.r_[0, start_indices]
    if mask[-1]:
        end_indices = np.r_[end_indices, len(mask)]

    # Create ranges and filter by duration
    ranges = []
    for start, end in zip(start_indices, end_indices, strict=False):
        t_start, t_end = t[start], t[end - 1]
        if (t_end - t_start) >= min_duration:
            ranges.append((t_start, t_end))

    # Step 2: Merge nearby ranges
    if ranges:
        merged_ranges = [ranges[0]]
        for curr_start, curr_end in ranges[1:]:
            prev_start, prev_end = merged_ranges[-1]
            if curr_start - prev_end < min_duration:
                # Merge with previous
                merged_ranges[-1] = (prev_start, curr_end)
            else:
                merged_ranges.append((curr_start, curr_end))
    else:
        merged_ranges = []

    if merged_ranges:
        console = MyConsole(verbose)
        for start, end in merged_ranges:
            console.info(
                f"Correlation between {threshold_low} and {threshold_high} from {start:.2f}s to {end:.2f}s"
            )

    return merged_ranges
