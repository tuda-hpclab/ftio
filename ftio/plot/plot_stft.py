"""
STFT Plotting functions for FTIO.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ftio.freq._stft import compute_stft
from ftio.plot.helper import format_plot
from ftio.plot.units import set_unit

FONT_SIZE = 17
FONT_SIZE_ANNOTATION = int(FONT_SIZE * 2 / 3)


def plot_stft(args, prediction, b_sampled, fs, stft_data=None):
    """
    Plots the STFT of the signal, its reconstruction, and the dominant frequency tracking
    using the specified engine (Plotly or Matplotlib).

    Args:
        args (Namespace): Parsed arguments.
        prediction (Prediction): Prediction object containing detected frequencies.
        b_sampled (np.ndarray): Uniformly sampled bandwidth data.
        fs (float): Sampling frequency in Hz.
        stft_data (tuple, optional): Precomputed (f, t, Zxx) from STFT. Defaults to None.

    Returns:
        list: A list of figure objects (Plotly or Matplotlib).
    """
    if stft_data is not None:
        f, t_stft, Zxx = stft_data
    else:
        f, t_stft, Zxx = compute_stft(b_sampled, fs)

    Zxx_mag = np.abs(Zxx)
    amp_mean = np.mean(Zxx_mag, axis=1)

    if "mat" in args.engine:
        return [
            plot_stft_matplotlib_spectrogram(f, t_stft, Zxx_mag),
            plot_stft_matplotlib_reconstruction(args, prediction, b_sampled, fs, t_stft),
            plot_stft_matplotlib_dominant_freq(args, prediction, b_sampled, fs, t_stft),
            plot_stft_matplotlib_all_spectra(f, t_stft, Zxx_mag, amp_mean),
        ]
    elif "plotly" in args.engine:
        return [
            plot_stft_plotly_spectrogram(f, t_stft, Zxx_mag),
            plot_stft_plotly_reconstruction(args, prediction, b_sampled, fs, t_stft),
            plot_stft_plotly_dominant_freq(args, prediction, b_sampled, fs, t_stft),
            plot_stft_plotly_all_spectra(f, t_stft, Zxx_mag, amp_mean),
        ]


def plot_stft_matplotlib_spectrogram(f, t, Zxx_mag):
    """Generates an STFT magnitude spectrogram using Matplotlib."""
    # Reduced width by 1/3 (10 * 2/3 = 6.67)
    fig = plt.figure(figsize=(6.67, 6))
    plt.pcolormesh(t, f, Zxx_mag, shading="gouraud", cmap="viridis")
    plt.title("STFT magnitude", fontsize=FONT_SIZE)
    plt.ylabel("Frequency (Hz)", fontsize=FONT_SIZE)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    cbar = plt.colorbar(label="Magnitude")
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.set_label("Magnitude", size=FONT_SIZE)
    return fig


def plot_stft_plotly_spectrogram(f, t, Zxx_mag):
    """Generates an STFT magnitude spectrogram using Plotly."""
    fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=f,
            z=Zxx_mag,
            colorscale="viridis",
            hovertemplate="<b>Time</b>: %{x:.2f} s<br>"
            + "<b>Frequency</b>: %{y:.3e} Hz<br>"
            + "<b>Magnitude</b>: %{z:.3e}<extra></extra>",
        )
    )

    fig.update_layout(
        title="STFT magnitude",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
    )
    return format_plot(fig)


def reconstruct_stft_signal(prediction, fs, n_samples, t_stft):
    """
    Reconstructs the time-domain signal by stitching together the dominant
    sinusoids identified in each STFT window.

    Args:
        prediction (Prediction): Object containing 'dominant_freq', 'amp', and 'phi' arrays.
        fs (float): Sampling frequency in Hz.
        n_samples (int): Total number of samples in the original signal.
        t_stft (np.ndarray): Array of center times for each STFT window.

    Returns:
        tuple: (t_full, reconstructed_signal)
    """
    t_full = prediction.t_start + np.arange(n_samples) / fs
    reconstructed = np.zeros(n_samples)

    # Note: prediction.dominant_freq[0] is global, 1: are windows
    # windows start at index 1
    win_freqs = prediction.dominant_freq[1:]
    win_amps = prediction.amp[1:]
    win_phis = prediction.phi[1:]
    win_ranges = prediction.ranges[1:]

    for i in range(len(win_freqs)):
        start_time, end_time = win_ranges[i]
        mask = (t_full >= start_time) & (t_full < end_time)
        if np.any(mask):
            freq = win_freqs[i]
            amp = win_amps[i]
            phi = win_phis[i]
            reconstructed[mask] = amp * np.cos(2 * np.pi * freq * t_full[mask] + phi)

    return t_full, reconstructed


def plot_stft_plotly_reconstruction(args, prediction, b_sampled, fs, t_stft):
    """Plots the original signal vs the STFT-reconstructed signal using Plotly."""
    unit, order = set_unit(b_sampled)
    t_full, reconstructed = reconstruct_stft_signal(
        prediction, fs, len(b_sampled), t_stft
    )

    # Calculate window duration from the first window range (index 1)
    win_duration = 0
    if len(prediction.ranges) > 1:
        win_duration = prediction.ranges[1][1] - prediction.ranges[1][0]

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    fig = go.Figure()

    # Original signal
    fig.add_trace(
        Scatter(
            x=t_full,
            y=b_sampled * order,
            mode="lines",
            name="Original Signal",
            line={"color": "rgba(0,150,250, 0.5)", "shape": "hv"},
            fill="tozeroy",
        )
    )

    # Reconstructed signal
    fig.add_trace(
        Scatter(
            x=t_full,
            y=reconstructed * order,
            mode="lines",
            name="STFT Reconstructed (Dominant)",
            line={"color": "rgb(70,220,70)", "shape": "hv"},
        )
    )

    # Add window segments (starting from index 1)
    if len(prediction.ranges) > 1:
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        for i, (start, end) in enumerate(prediction.ranges[1:]):
            color = colors[i % len(colors)]
            # Draw both boundaries for each window with matching color
            if start > prediction.t_start + 0.01:
                fig.add_vline(
                    x=start,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.4,
                    annotation_text=f"W{i}s",
                    annotation_position="top left",
                    annotation_font_size=FONT_SIZE_ANNOTATION,
                )
            if end < prediction.t_end - 0.01:
                fig.add_vline(
                    x=end,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.4,
                    annotation_text=f"W{i}e",
                    annotation_position="top left",
                    annotation_font_size=FONT_SIZE_ANNOTATION,
                )

    fig.update_layout(
        title=f"STFT time-varying reconstruction ({win_duration:.2f} s)",
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        template="plotly",
    )

    return format_plot(fig)


def plot_stft_matplotlib_reconstruction(args, prediction, b_sampled, fs, t_stft):
    """Plots the original signal vs the STFT-reconstructed signal using Matplotlib."""
    unit, order = set_unit(b_sampled)
    t_full, reconstructed = reconstruct_stft_signal(
        prediction, fs, len(b_sampled), t_stft
    )

    # Calculate window duration from the first window range (index 1)
    win_duration = 0
    if len(prediction.ranges) > 1:
        win_duration = prediction.ranges[1][1] - prediction.ranges[1][0]

    fig = plt.figure(figsize=(10, 4))
    plt.fill_between(
        t_full,
        0,
        b_sampled * order,
        label="Original signal",
        alpha=0.5,
        step="post",
        color="royalblue",
    )
    plt.plot(t_full, b_sampled * order, drawstyle="steps-post", color="royalblue")

    plt.plot(
        t_full,
        reconstructed * order,
        label="STFT Reconstructed",
        color="limegreen",
        drawstyle="steps-post",
    )

    # Add window segments (starting from index 1)
    if len(prediction.ranges) > 1:
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        ylim = plt.ylim()
        for i, (start, end) in enumerate(prediction.ranges[1:]):
            color = colors[i % len(colors)]
            if start > prediction.t_start + 0.01:
                plt.axvline(x=start, linestyle="--", color=color, alpha=0.4)
                plt.text(
                    start,
                    ylim[1],
                    f"W{i}s",
                    color=color,
                    verticalalignment="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                )
            if end < prediction.t_end - 0.01:
                plt.axvline(x=end, linestyle="--", color=color, alpha=0.4)
                plt.text(
                    end,
                    ylim[1],
                    f"W{i}e",
                    color=color,
                    verticalalignment="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                )

    from ftio.freq.duty_cycle import overlay_burst_widths_matplotlib

    overlay_burst_widths_matplotlib(prediction)

    plt.title(
        f"STFT time-varying reconstruction ({win_duration:.2f} s)", fontsize=FONT_SIZE
    )
    plt.ylabel(f"Bandwidth ({unit})", fontsize=FONT_SIZE)
    plt.xlabel("Time (s)", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    return fig


def plot_stft_plotly_dominant_freq(args, prediction, b_sampled, fs, t_stft):
    """
    Plots the dominant frequency tracking over time using Plotly.
    Includes a secondary axis for period (T=1/f).
    """

    def Scatter(**kwargs):
        if args.render == "dynamic":
            return go.Scatter(kwargs)
        else:
            return go.Scattergl(kwargs)

    fig = go.Figure()

    # Per-window data starts at index 1
    win_freqs = prediction.dominant_freq[1:]

    # Calculate periods for hover info and secondary axis
    periods = np.zeros_like(win_freqs)
    mask = win_freqs > 0
    periods[mask] = 1.0 / win_freqs[mask]

    # Dominant Frequency (primary axis)
    fig.add_trace(
        Scatter(
            x=t_stft,
            y=win_freqs,
            mode="lines+markers",
            name="Dominant Frequency",
            line={"color": "rgb(220,50,50)"},
            marker={"size": 6},
            hovertemplate="<b>Time</b>: %{x:.2f} s"
            + "<br><b>Freq</b>: %{y:.3e} Hz"
            + "<br><b>T</b>: %{customdata:.3f} s",
            customdata=periods,
            yaxis="y1",
        )
    )

    # Period (secondary axis, visible only in legend/hover)
    fig.add_trace(
        Scatter(
            x=t_stft,
            y=periods,
            mode="markers",
            name="Period (s)",
            visible="legendonly",
            marker={"color": "rgb(50,200,50)"},
            yaxis="y2",
        )
    )

    # Add window segments (index 1:)
    if len(prediction.ranges) > 1:
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        for i, (start, end) in enumerate(prediction.ranges[1:]):
            color = colors[i % len(colors)]
            if start > prediction.t_start + 0.01:
                fig.add_vline(
                    x=start,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.4,
                    annotation_text=f"W{i}s",
                    annotation_position="top left",
                    annotation_font_size=FONT_SIZE_ANNOTATION,
                )
            if end < prediction.t_end - 0.01:
                fig.add_vline(
                    x=end,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.4,
                    annotation_text=f"W{i}e",
                    annotation_position="top left",
                    annotation_font_size=FONT_SIZE_ANNOTATION,
                )

    fig.update_layout(
        title="Dominant frequency in each STFT window",
        xaxis_title="Time (s)",
        yaxis={
            "title": {"text": "Frequency (Hz)"},
        },
        yaxis2={
            "title": {"text": "Period (s)"},
            "anchor": "x",
            "overlaying": "y",
            "side": "right",
        },
        showlegend=True,
        template="plotly",
    )

    # Put legend inside
    fig.update_layout(legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99})

    return format_plot(fig)


def plot_stft_matplotlib_dominant_freq(args, prediction, b_sampled, fs, t_stft):
    """
    Plots the dominant frequency tracking over time using Matplotlib.
    Includes a secondary axis for period (T=1/f).
    """
    # Reduced width by 1/3
    fig, ax1 = plt.subplots(figsize=(6.67, 4))

    # Per-window data starts at index 1
    win_freqs = prediction.dominant_freq[1:]

    ax1.plot(t_stft, win_freqs, "r-o", label="Dominant Frequency", markersize=4)
    ax1.set_xlabel("Time (s)", fontsize=FONT_SIZE)
    ax1.set_ylabel("Frequency (Hz)", color="black", fontsize=FONT_SIZE)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=FONT_SIZE)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE)

    # Add secondary axis for Period
    ax2 = ax1.twinx()
    periods = np.zeros_like(win_freqs)
    mask = win_freqs > 0
    periods[mask] = 1.0 / win_freqs[mask]

    ax2.set_ylabel("Period (s)", color="black", fontsize=FONT_SIZE)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=FONT_SIZE)
    ax2.plot(t_stft, periods, "g--", label="Period", alpha=0.5)

    # Add window segments
    if len(prediction.ranges) > 1:
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        ylim = ax1.get_ylim()
        for i, (start, end) in enumerate(prediction.ranges[1:]):
            color = colors[i % len(colors)]
            if start > prediction.t_start + 0.01:
                ax1.axvline(x=start, linestyle="--", color=color, alpha=0.4)
                ax1.text(
                    start,
                    ylim[1],
                    f"W{i}s",
                    color=color,
                    verticalalignment="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                )
            if end < prediction.t_end - 0.01:
                ax1.axvline(x=end, linestyle="--", color=color, alpha=0.4)
                ax1.text(
                    end,
                    ylim[1],
                    f"W{i}e",
                    color=color,
                    verticalalignment="bottom",
                    fontsize=FONT_SIZE_ANNOTATION,
                )

    fig.suptitle("Dominant frequency in each STFT window", fontsize=FONT_SIZE)
    ax1.grid(True)
    # Put legend inside
    ax1.legend(loc="upper left", fontsize=FONT_SIZE)
    fig.tight_layout()
    return fig


def plot_stft_plotly_all_spectra(f, t, Zxx_mag, amp_mean, stft_global=True):
    """Plots the amplitude spectrum for each window and global spectrum using Plotly bars."""
    fig = go.Figure()
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    # Exclude DC component (index 0)
    f_plot = f[1:]
    mag_plot = Zxx_mag[1:, :]
    amp_mean_plot = amp_mean[1:]

    # Add global spectrum if requested
    if stft_global:
        fig.add_trace(
            go.Bar(
                x=f_plot,
                y=amp_mean_plot,
                name="Global summary",
                marker_color="black",
                opacity=0.6,
                visible=True,
            )
        )

    # Add each window spectrum
    for i in range(mag_plot.shape[1]):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Bar(
                x=f_plot,
                y=mag_plot[:, i],
                name=f"Window {i} ({t[i]:.2f} s)",
                marker_color=color,
                visible=True if i == 0 and not stft_global else "legendonly",
            )
        )

    fig.update_layout(
        title="Amplitude spectrum in each window",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        barmode="overlay",
        template="plotly",
    )
    return format_plot(fig)


def plot_stft_matplotlib_all_spectra(f, t, Zxx_mag, amp_mean, stft_global=True):
    """Plots the amplitude spectrum for each window and global spectrum using Matplotlib bars."""
    fig = plt.figure(figsize=(10, 6))
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    # Exclude DC component (index 0)
    f_plot = f[1:]
    mag_plot = Zxx_mag[1:, :]
    amp_mean_plot = amp_mean[1:]

    # Width for bars
    width = (f_plot[1] - f_plot[0]) * 0.8 if len(f_plot) > 1 else 0.1

    # Add global spectrum if requested
    if stft_global:
        plt.bar(
            f_plot,
            amp_mean_plot,
            width=width,
            label="Global summary",
            color="black",
            alpha=0.4,
        )

    # Add window spectra (only if few windows, otherwise it's too messy)
    if mag_plot.shape[1] < 10:
        for i in range(mag_plot.shape[1]):
            color = colors[i % len(colors)]
            plt.bar(
                f_plot, mag_plot[:, i], width=width, label=f"W{i}", color=color, alpha=0.5
            )

    plt.title("Amplitude spectrum in each window", fontsize=FONT_SIZE)
    plt.ylabel("Magnitude", fontsize=FONT_SIZE)
    plt.xlabel("Frequency (Hz)", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.legend(fontsize=FONT_SIZE)
    return fig
