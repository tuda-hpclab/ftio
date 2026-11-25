"""
Wavelet plot methods

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

####################################################################################################
# Deprecated functions
####################################################################################################


def plot_wave_cont(
    b_sampled: np.ndarray,
    frequencies: np.ndarray,
    freq: float,
    t: np.ndarray,
    coefficients: np.ndarray,
) -> plt.Figure:
    """
    Plot continuous wavelet transform.

    Args:
        b_sampled (np.ndarray): Sampled signal.
        frequencies (np.ndarray): Frequencies array.
        freq (float): Sampling frequency.
        t (np.ndarray): Time array.
        coefficients (np.ndarray): Wavelet coefficients.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    time_disc = t[0] + 1 / freq * np.arange(0, len(b_sampled))
    power = (
        abs(coefficients)
    ) ** 2  # probably error on https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
    counter = 0
    for i in range(
        1, len(coefficients)
    ):  # see Continuous wavelet transform properties @ https://en.wikipedia.org/wiki/Continuous_wavelet_transform
        power[counter] = 1 / i * power[counter]
        counter = counter + 1
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.contourf(
        time_disc,
        frequencies,
        power,
        np.arange(0, power.max(), power.max() / 10),
        extend="neither",
        cmap=plt.cm.seismic,
    )
    # fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    fig.colorbar(im, ax=ax)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.tight_layout()
    plt.show()
    return fig


####################################################################################################
# Matplotlib plotting functions
####################################################################################################
def matplot_dominant_scale(
    t: np.ndarray,
    dominant_power: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
    subplot=None,
) -> plt.Figure:
    """
    Plot dominant wavelet scale using Matplotlib.

    Args:
        t (np.ndarray): Time array.
        dominant_power (np.ndarray): Power array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.
        subplot: Subplot configuration.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    if subplot is None:
        subplot = []
    if subplot:
        fig = plt.subplot(subplot[0], subplot[1], subplot[2])
    else:
        fig = plt.figure(figsize=(12, 4))

    plt.plot(
        t,
        dominant_power,
        label=f"Power at Dominant Scale {label if label else ''}",
        color="orange",
    )
    if peaks is not None:
        plt.scatter(
            t[peaks],
            dominant_power[peaks],
            color="red",
            label="Detected Peaks",
        )
    plt.title("Wavelet Power at Dominant Scale")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    return fig


def matplot_wave_cont_spectrum(
    t: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    subplot=None,
) -> plt.Figure:
    """
    Plot wavelet power spectrum using Matplotlib.

    Args:
        t (np.ndarray): Time array.
        power_spectrum (np.ndarray): Power spectrum array.
        frequencies (np.ndarray): Frequencies array.
        subplot: Subplot configuration.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    if subplot is None:
        subplot = []
    if subplot:
        fig = plt.subplot(subplot[0], subplot[1], subplot[2])
    else:
        fig = plt.figure(figsize=(12, 4))

    # plt.imshow(power_spectrum, aspect='auto', extent=[t[0], t[-1], scales[-1], scales[0]], cmap='viridis')
    plt.imshow(
        power_spectrum,
        aspect="auto",
        cmap="viridis",
        extent=[t[0], t[-1], frequencies[0], frequencies[-1]],
    )
    plt.colorbar(label="Power")
    plt.title("Wavelet Power Spectrum (All Scales)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequencies (Hz)")
    plt.grid()

    return fig


def matplot_wave_cont_and_spectrum(
    t: np.ndarray,
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    dominant_power_spectrum: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
) -> plt.Figure:
    """
    Plot continuous wavelet transform and power spectrum using Matplotlib.

    Args:
        t (np.ndarray): Time array.
        frequencies (np.ndarray): Frequencies array.
        power_spectrum (np.ndarray): Power spectrum array.
        dominant_power_spectrum (np.ndarray): Dominant power spectrum array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig = plt.figure(figsize=(16, 12))
    matplot_wave_cont_spectrum(t, power_spectrum, frequencies, [2, 1, 1])
    matplot_dominant_scale(t, dominant_power_spectrum, label, peaks, [2, 1, 2])

    return fig


def matplot_plot_scales(
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
    common_xaxis: bool = True,
) -> plt.Figure:
    """
    Plot wavelet power spectrum at different scales using Matplotlib.

    Args:
        t (np.ndarray): Time array.
        b (np.ndarray): Original signal.
        power_spectrum (np.ndarray): Power spectrum array (shape: [num_scales, num_time_points]).
        frequencies (np.ndarray): Frequencies array.
        scales (np.ndarray): Scale array matching the frequencies.
        common_xaxis (bool): If True, enables a common x-axis for zooming.

    Returns:
        plt.Figure: Matplotlib figure object.
    """

    # Ensure power_spectrum is a 2D array (scales, time)
    if len(power_spectrum.shape) != 2:
        raise ValueError("Power spectrum must be a 2D array (scales, time)")

    # Set up the figure and subplots
    fig, axes = plt.subplots(len(scales) + 2, 1, figsize=(10, 6), sharex=common_xaxis)
    axes = axes.flatten()

    # Plot the time series on the first subplot
    axes[0].plot(t, b, color="blue", label="Time Series")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Bandwidth over Time")
    axes[0].legend()

    # Plot the wavelet power spectrum (all scales)
    # Make sure the power_spectrum has the correct shape: [num_scales, num_time_points]
    axes[1].imshow(
        power_spectrum,
        aspect="auto",
        extent=[t[0], t[-1], frequencies[-1], frequencies[0]],
        cmap="jet",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Wavelet Power Spectrum (All Scales)")

    # Plot the power spectrum at each scale
    for i, scale in enumerate(scales):
        axes[i + 2].plot(
            t,
            power_spectrum[i, :],
            label=f"Scale {scale:.3e} (Frequency {frequencies[i]:.3e})",
        )
        axes[i + 2].set_xlabel("Time")
        axes[i + 2].set_ylabel("Power")
        axes[i + 2].legend()
        axes[i + 2].set_title(
            f"Power at Scale {scale:.3e} (Frequency {frequencies[i]:.3e})"
        )

    # Adjust layout to avoid overlapping
    # plt.tight_layout()

    return fig


def matplotlib_plot_scales_all_in_one(
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
) -> plt.Figure:
    """
    Creates a Matplotlib figure showing the bandwidth over time and the power spectrum
    at different scales.

    Parameters:
    - t: np.ndarray
        Time array.
    - b: np.ndarray
        Bandwidth or signal values over time.
    - power_spectrum: np.ndarray
        2D array where each row corresponds to the power spectrum at a given scale.
    - frequencies: np.ndarray
        Array of frequencies corresponding to the scales.
    - scales: np.ndarray
        Array of scales used for the wavelet transform.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(t, b, label="Bandwidth over time", color="blue")

    for i, scale in enumerate(scales):
        label = f"Power at Scale {scale:.3e} (freq. {frequencies[i]:.3e} -- period {1/frequencies[i] if frequencies[i] > 0 else 0:.3e})"
        ax.plot(t, power_spectrum[i, :], label=label)

    ax.set_title("Wavelet Power Spectrum and Time Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude / Power")
    ax.legend()
    plt.show()

    return fig


####################################################################################################
# Plotly plotting functions
####################################################################################################
def plotly_dominant_scale(
    t: np.ndarray,
    dominant_power: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
    subplot=None,
    fig: go.Figure = None,
) -> go.Figure:
    """
    Plot dominant wavelet scale using Plotly.

    Args:
        t (np.ndarray): Time array.
        dominant_power (np.ndarray): Power array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.
        subplot: Subplot configuration.
        fig (go.Figure): Plotly figure object.

    Returns:
        go.Figure: Plotly figure object.
    """
    if fig is None:
        fig = go.Figure()

    trace = go.Scatter(
        x=t,
        y=dominant_power,
        mode="lines",
        name=f"Power at {label if label else 'Dominant Scale'}",
        line={"color": "orange"},
    )
    if subplot:
        fig.add_trace(trace, row=subplot[0], col=subplot[1])
    else:
        fig.add_trace(trace)

    if peaks is not None:
        peak_trace = go.Scatter(
            x=t[peaks],
            y=dominant_power[peaks],
            mode="markers",
            marker={"color": "red"},
            name="Detected Peaks",
        )

        if subplot:
            fig.add_trace(peak_trace, row=subplot[0], col=subplot[1])
        else:
            fig.add_trace(peak_trace)

    if subplot:
        fig.update_xaxes(title_text="Time (seconds)", row=subplot[0], col=subplot[1])
        fig.update_yaxes(title_text="Power", row=subplot[0], col=subplot[1])
    else:
        fig.update_layout(
            title=f"Wavelet Power at {label if label else 'Dominant Scale'}",
            xaxis_title="Time (seconds)",
            yaxis_title="Power",
        )

    return fig


def plotly_wave_cont_spectrum(
    t: np.ndarray,
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    subplot=None,
    fig: go.Figure = None,
) -> go.Figure:
    """
    Plots a wavelet power spectrum using Plotly.

    Args:
        t (np.ndarray): Array of time values.
        frequencies (np.ndarray): Array of frequency values.
        power_spectrum (np.ndarray): 2D array of power spectrum values.
        subplot (tuple, optional): Tuple specifying the subplot position (row, col). Defaults to None.
        fig (go.Figure, optional): Existing Plotly figure to add the heatmap to. Defaults to None.

    Returns:
        go.Figure: The Plotly figure object with the wavelet power spectrum heatmap.
    """
    if fig is None:
        fig = go.Figure()

    heatmap = go.Heatmap(x=t, y=frequencies, z=power_spectrum, colorscale="viridis")
    if subplot:
        fig.add_trace(heatmap, row=subplot[0], col=subplot[1])
        fig.update_xaxes(title_text="Time (seconds)", row=subplot[0], col=subplot[1])
        fig.update_yaxes(title_text="Frequencies (Hz)", row=subplot[0], col=subplot[1])
    else:
        fig.add_trace(heatmap)
        fig.update_layout(
            title="Scaleogram",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequencies (Hz)",
        )

    return fig


def plotly_wave_cont_and_spectrum(
    t: np.ndarray,
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    dominant_power: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
) -> go.Figure:
    """
    Plot continuous wavelet transform and power spectrum using Plotly.

    Args:
        t (np.ndarray): Time array.
        frequencies (np.ndarray): Frequencies array.
        power_spectrum (np.ndarray): Power spectrum array.
        dominant_power (np.ndarray): Dominant power spectrum array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.

    Returns:
        go.Figure: Plotly figure object.
    """
    rows = 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=[
            "Wavelet Power Spectrum (All Scales)",
            "Wavelet Power at Dominant Scale",
        ],
        vertical_spacing=0.2,
        shared_xaxes=True,
    )

    fig = plotly_wave_cont_spectrum(
        t, frequencies, power_spectrum, subplot=[1, 1], fig=fig
    )
    fig = plotly_dominant_scale(t, dominant_power, label, peaks, subplot=[2, 1], fig=fig)

    for i in range(1, rows):
        fig.update_xaxes(showticklabels=True, row=i, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=i, col=1)
        fig.update_yaxes(title_text="Power", row=i, col=1)

    fig.update_layout(height=1200, width=1600, showlegend=True)

    return fig


def plotly_plot_scales(
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
    common_xaxis: bool = True,
) -> go.Figure:
    """
    Plot wavelet power spectrum at different scales using Plotly.

    Args:
        t (np.ndarray): Time array.
        b (np.ndarray): Original signal.
        power_spectrum (np.ndarray): Power spectrum array.
        frequencies (np.ndarray): Frequencies array.
        scales (np.ndarray): Scale array matching the frequencies.
        common_xaxis (bool): If True, enables a common x-axis for zooming.

    Returns:
        go.Figure: Plotly figure object.
    """

    names = ["bandwidth over time", "Wavelet Power Spectrum (All Scales)"]
    names.extend(
        [
            f"Power at Scale {scale:.3e} (freq. {frequencies[i]:.3e} -- period {1/frequencies[i] if frequencies[i] > 0 else 0:.3e})"
            for i, scale in enumerate(scales)
        ]
    )
    fig = make_subplots(
        rows=len(scales) + 2,
        cols=1,
        subplot_titles=names,
        shared_xaxes=common_xaxis,
    )
    fig.add_trace(
        go.Scatter(x=t, y=b, mode="lines", name="Time Series", line={"color": "blue"}),
        row=1,
        col=1,
    )
    fig = plotly_wave_cont_spectrum(
        t, frequencies, power_spectrum, subplot=[2, 1], fig=fig
    )
    for i, scale in enumerate(scales):
        fig = plotly_dominant_scale(
            t,
            power_spectrum[i, :],
            f"Scale {scale} (Frequency {frequencies[i]:.3e})",
            subplot=[i + 3, 1],
            fig=fig,
        )

    # Adjust layout to allow scrolling and zooming
    for i in range(1, len(scales) + 3):  # 1 to (number of scales + 2)
        fig.update_xaxes(showticklabels=True, row=i, col=1)

    fig.update_layout(
        showlegend=True,
        height=400 * (len(scales) + 2),  # Makes the figure scrollable
        xaxis_title="Time",
        yaxis_title="Amplitude",
    )

    return fig


def plotly_plot_scales_all_in_one(
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
    peaks: np.ndarray = np.array([]),
) -> go.Figure:
    """
    Creates a Plotly figure showing the bandwidth over time and the power spectrum
    at different scales.

    Parameters:
    - t: np.ndarray
        Time array.
    - b: np.ndarray
        Bandwidth or signal values over time.
    - power_spectrum: np.ndarray
        2D array where each row corresponds to the power spectrum at a given scale.
    - frequencies: np.ndarray
        Array of frequencies corresponding to the scales.
    - scales: np.ndarray
        Array of scales used for the wavelet transform.

    Returns:
        go.Figure: Plotly figure object.
    """
    names = [
        "bandwidth over time",
    ]
    names.extend(
        [
            f"Power at Scale {scale:.3e} (freq. {frequencies[i]:.3e} -- period {1/frequencies[i] if frequencies[i] > 0 else 0:.3e})"
            for i, scale in enumerate(scales)
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=t, y=b, mode="lines", name=names[0], line={"color": "blue"})
    )
    for i, _scale in enumerate(scales):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=power_spectrum[i, :],
                mode="lines",
                name=names[i + 1],
            )
        )
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=t[peaks[i]],
                    y=power_spectrum[i, peaks[i]],
                    mode="markers",
                    marker={"color": "red"},
                    name=names[i + 1].replace("Power", "Peaks"),
                )
            )

    fig.update_layout(
        title="Wavelet Power Spectrum and Time Series",
        xaxis_title="Time",
        yaxis_title="Amplitude / Power",
        showlegend=True,
    )
    return fig


####################################################################################################
# Plotting functions
####################################################################################################
def plot_wave_cont_and_spectrum(
    args: Namespace,
    t: np.ndarray,
    frequencies: np.ndarray,
    power_spectrum_all_scales: np.ndarray,
    dominant_power_spectrum: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
) -> plt.Figure:
    """
    Plot continuous wavelet transform and power spectrum using the specified plotting engine.

    Args:
        args (Namespace): Arguments containing the plotting engine.
        t (np.ndarray): Time array.
        frequencies (np.ndarray): Frequencies array.
        power_spectrum_all_scales (np.ndarray): Power spectrum array.
        dominant_power_spectrum (np.ndarray): Dominant power spectrum array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_wave_cont_and_spectrum(
            t,
            frequencies,
            power_spectrum_all_scales,
            dominant_power_spectrum,
            label,
            peaks,
        )
    else:
        fig = matplot_wave_cont_and_spectrum(
            t,
            frequencies,
            power_spectrum_all_scales,
            dominant_power_spectrum,
            label,
            peaks,
        )

    return fig


def plot_wave_cont_spectrum(
    args: Namespace,
    t: np.ndarray,
    frequencies: np.ndarray,
    power_spectrum_all_scales: np.ndarray,
) -> plt.Figure:
    """
    Plot continuous wavelet transform and power spectrum using the specified plotting engine.

    Args:
        args (Namespace): Arguments containing the plotting engine.
        t (np.ndarray): Time array.
        frequencies (np.ndarray): Frequencies array.
        power_spectrum_all_scales (np.ndarray): Power spectrum array.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_wave_cont_spectrum(t, frequencies, power_spectrum_all_scales)
    else:
        fig = matplot_wave_cont_spectrum(t, frequencies, power_spectrum_all_scales)

    return fig


def plot_dominant_scale(
    args: Namespace,
    t: np.ndarray,
    dominant_power_spectrum: np.ndarray,
    label: str = None,
    peaks: np.ndarray = None,
) -> plt.Figure:
    """
    Plot dominant power spectrum using the specified plotting engine.

    Args:
        args (Namespace): Arguments containing the plotting engine.
        t (np.ndarray): Time array.
        dominant_power_spectrum (np.ndarray): Dominant power spectrum array.
        label (str): Label for the plot.
        peaks (np.ndarray): Detected peaks.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_dominant_scale(t, dominant_power_spectrum, label, peaks)
    else:
        fig = matplot_dominant_scale(t, dominant_power_spectrum, label, peaks)

    return fig


def plot_scales(
    args: Namespace,
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
    common_xaxis: bool = True,
) -> plt.Figure:
    """
    Plots the wavelet scales using either Plotly or Matplotlib based on the specified engine.

    Args:
        args (Namespace): Arguments containing the plotting engine information.
        t (np.ndarray): Time array.
        b (np.ndarray): Data array.
        power_spectrum (np.ndarray): Power spectrum array.
        frequencies (np.ndarray): Frequencies array.
        scales (np.ndarray): Scales array.
        common_xaxis (bool, optional): Whether to use a common x-axis for the plots. Defaults to True.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_plot_scales(t, b, power_spectrum, frequencies, scales, common_xaxis)
    else:
        fig = matplot_plot_scales(t, b, power_spectrum, frequencies, scales, common_xaxis)

    return fig


def plot_scales_all_in_one(
    args: Namespace,
    t: np.ndarray,
    b: np.ndarray,
    power_spectrum: np.ndarray,
    frequencies: np.ndarray,
    scales: np.ndarray,
    peaks: np.ndarray = np.array([]),
) -> plt.Figure:
    """
    Plots the wavelet scales using either Plotly or Matplotlib based on the specified engine.

    Args:
        args (Namespace): Arguments containing the plotting engine information.
        t (np.ndarray): Time array.
        b (np.ndarray): Data array.
        power_spectrum (np.ndarray): Power spectrum array.
        frequencies (np.ndarray): Frequencies array.
        scales (np.ndarray): Scales array.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_plot_scales_all_in_one(
            t, b, power_spectrum, frequencies, scales, peaks
        )
    else:
        fig = matplotlib_plot_scales_all_in_one(t, b, power_spectrum, frequencies, scales)

    return fig
