"""
This module contains functions for plotting discrete wavelet transforms and their spectra
using Matplotlib and Plotly.

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
import pywt
from plotly.subplots import make_subplots

####################################################################################################
#! Deprecated functions
####################################################################################################


def plot_wave_disc(
    b_sampled: np.ndarray,
    coffs,
    t: np.ndarray,
    freq: float,
    level: int,
    wavelet: str,
    b: np.ndarray,
):
    """
    Plot discrete wavelet transform.

    Args:
        b_sampled (np.ndarray): Sampled signal.
        coffs: Wavelet coefficients.
        t (np.ndarray): Time array.
        freq (float): Sampling frequency.
        level (int): Decomposition level.
        wavelet (str): Wavelet type.
        b (np.ndarray): Original signal.

    Returns:
        Tuple[np.ndarray, list[plt.Figure]]: Coefficients and list of Matplotlib figure objects.
    """
    n = len(b_sampled)
    time_disc = t[0] + 1 / freq * np.arange(0, n)
    reconstructed_signal = pywt.waverec(coffs, wavelet, "smooth")
    f = []
    f1 = plt.figure(figsize=(12, 4))
    plt.plot(t, b, label="signal")
    plt.plot(
        time_disc,
        reconstructed_signal[0 : len(time_disc)],
        label=f"reconstructed levels {level}",
        linestyle="--",
    )
    # plt.plot(time,reconstructed_signal[0:len(time)], label="reconstructed levels %d", linestyle="--")
    plt.legend(loc="upper right")
    plt.title("single reconstruction", fontsize=20)
    plt.xlabel("time axis", fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)

    #! use recon or coffs to either create the partiality reconstructed signal or the coefficients of the DTW:
    #! recon -> idea is that all values are extracted just before reconstructing the signal. This adds less resolution to the upper frequencies
    #! x ---->.g -->↓ 2 --> C1 --> ↑2 -->* g-1  #   capture   # -->+ --> x
    #!     "->.h -->↓ 2 --> C2 --> ↑2 -->* h-1  #      here    #  --^
    use = "recon"
    # use = "coffs" #hold the signal, not upsample!
    cc = np.zeros((level + 1, n))
    for i in np.arange(start=level, stop=0, step=-1):
        # coffs ->  [cA_n, cD_n, cD_n-1, …, cD2, cD1]
        # reconstruction by 1) up-sampling and 2) multiplying with the appropriate filter
        # page 29 on https://www.corsi.univr.it/documenti/OccorrenzaIns/matdid/matdid358630.pdf
        # GoH0+G1H1 = 1 -> Multiply a with H1 and d with G1
        # https://medium.com/@shouke.wei/process-of-discrete-wavelet-transform-iii-wavelet-partial-reconstruction-ca7a8f9420dc
        if "recon" in use:
            cc[level - i + 1] = pywt.upcoef("d", coffs[level - i + 1], wavelet, level=i)[
                :n
            ]
        else:
            # Wrong: only upsample: ‘↑ 2’ denotes ‘upsample by 2’ (put 0’s before values)
            # cc[level-i+1,::2**(i)] = coffs[level-i+1]
            # Correct: upsample is used for reconstruction, we only want to visualize -> hold the data constant during sample step
            counter = 0
            for j in range(0, n):
                if j % 2**i == 0:
                    cc[level - i + 1, j] = coffs[level - i + 1][counter]
                    counter = counter + 1
                else:
                    cc[level - i + 1, j] = cc[level - i + 1, j - 1]

    if "recon" in use:
        cc[0] = pywt.upcoef("a", coffs[0], wavelet, level=level)[:n]
        fig, ax = plt.subplots(nrows=level + 2, ncols=1, figsize=(12, 12))
        ax[0].plot(t, b, label="signal")
        for i in range(0, level + 1):
            if i == 0:
                sum_cc = cc[0]
            else:
                sum_cc = sum_cc + cc[i]
        ax[0].plot(time_disc, sum_cc, label="sum")
        for i in np.arange(start=level, stop=0, step=-1):
            ax[level - i + 2].plot(time_disc, cc[level - i + 1])
            ax[level - i + 2].set_title(
                f"reconstruction at level {i} from detail coefficients -> [{freq / 2 ** (i + 1)}, {freq / 2**i}] Hz"
            )
        ax[1].plot(time_disc, cc[0])
        ax[1].set_title(
            f"reconstruction at level {level} from approximation coefficients -> [0, {freq / 2 ** (level + 1)}] Hz"
        )
        fig.tight_layout()
        f.append(fig)

    else:
        counter = 0
        for j in range(0, n):
            if j % 2**level == 0:
                cc[0, j] = coffs[0][counter]
                counter = counter + 1
            else:
                cc[0, j] = cc[0, j - 1]

    # Use freq or level to plot either the frequency or the level in the y-axis
    show = "freq"
    f_2 = plt.figure(figsize=(12, 4))
    plt.xlabel("Time (s)", fontsize=18)
    # plt.title("Discrete Wavelet Transform with %d decompositions"%level)
    if "freq" in show:
        plt.ylabel("Frequency (Hz)", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        y = np.concatenate(
            [
                np.array([0]),
                freq / 2 ** np.arange(start=level + 1, stop=0, step=-1),
            ]
        )
        x = (
            -2 / freq + t[0] + 1 / freq * np.arange(0, len(b_sampled) + 1)
        )  # ? add corner shifted by half a sample step
        X, Y = np.meshgrid(x, y)
        # plt.pcolormesh(X, Y,abs(cc),cmap=plt.cm.coolwarm,shading="flat")
        plt.pcolormesh(X, Y, abs(cc), cmap=plt.cm.seismic, shading="flat")
    else:
        y = np.arange(start=level, stop=-1, step=-1)
        x = t[0] + 1 / freq * np.arange(0, len(b_sampled))
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, abs(cc), cmap=plt.cm.coolwarm)
        plt.ylabel("decomposition level")
        # cc = np.flip(cc,0)
        plt.gca().invert_yaxis()

    # plt.pcolormesh(X, Y, cc,cmap=plt.cm.seismic)
    # plt.pcolormesh(X, Y,abs(cc),cmap=plt.cm.seismic,shading="flat")
    # plt.plot(X.flat, Y.flat, "x", color="m")
    plt.colorbar()
    f.append(f_2)
    plt.tight_layout()
    f.append(f1)
    return cc, f


####################################################################################################
#! Plotting helpers
####################################################################################################


def get_names(freq_bands: np.ndarray, n: int):
    """
    Generate names for the plot subplots based on frequency bands.

    Args:
        freq_bands (np.ndarray): Array of frequency ranges for each level.
        n (int): Number of decomposition levels.

    Returns:
        list: List of subplot names.
    """
    names = ["Application Bandwidth"] + [
        f"RS{n-i} from CD (Freq: [{freq_bands[i, 0]:.3e} Hz - {freq_bands[i, 1]:.3e}] Hz)"
        for i in range(n)
    ]
    names[1] = (
        f"RS{n - 1} from CA (Freq: [{freq_bands[0, 0]:.3e} Hz - {freq_bands[0, 1]:.3e}] Hz)"
    )

    return names


####################################################################################################
#! Matplotlib plotting functions
####################################################################################################
def matplot_coeffs_reconst_signal(
    t: np.ndarray,
    b: np.ndarray,
    t_sampled: np.ndarray,
    b_sampled: np.ndarray,
    coeffs: np.ndarray,
    freq_bands: np.ndarray,
    common_xaxis: bool = True,
):
    """
    Plot the original signal and wavelet decomposition levels using Matplotlib.

    Args:
        t (np.ndarray): Time array.
        b (np.ndarray): Bandwidth.
        t_sampled (np.ndarray): Sampled time array.
        b_sampled (np.ndarray): Sampled bandwidth.
        coeffs (list): Wavelet decomposition coefficients.
        freq_bands (np.ndarray): Frequency ranges for each level.
        common_xaxis (bool): If True, enables a common x-axis for zooming.

    Returns:
        fig: Matplotlib figure object.
    """

    num_levels = len(coeffs)
    names = get_names(freq_bands, num_levels)
    fig, axes = plt.subplots(
        num_levels + 1,
        1,
        figsize=(10, 2.5 * (num_levels + 1)),
        sharex=common_xaxis,
    )

    # Plot the original and reconstructed signals in the first subplot
    axes[0].plot(t, b, label="Bandwidth", color="b", linestyle="dashed")
    axes[0].plot(
        t_sampled,
        b_sampled,
        label="Sampled Bandwidth",
        color="b",
        linestyle="dashed",
    )
    axes[0].plot(
        t_sampled,
        np.sum(coeffs, axis=0),
        label="Reconstructed Signal",
        color="g",
    )
    # axes[0].set_title(names[0])
    axes[0].set_ylabel("Amplitude", fontsize=17)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[0].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1e"))
    # axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].set_xlim(t_sampled[0], t_sampled[-1])

    # Plot each coefficient with corresponding frequency range
    for i in range(num_levels):
        axes[i + 1].plot(
            t_sampled,
            coeffs[i],
            label=names[i + 1],
        )
        #         axes[i + 1].set_title(names[i + 1])
        axes[i + 1].set_ylabel("Amplitude", fontsize=17)
        axes[i + 1].legend(loc="upper right")
        axes[i + 1].grid(True, which="both", linestyle="--", alpha=0.6)
        axes[i + 1].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1e"))
        #         axes[i + 1].ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
        axes[i + 1].ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
        axes[i + 1].tick_params(axis="both", labelsize=12)
        axes[i + 1].set_xlim(t_sampled[0], t_sampled[-1])

    axes[-1].set_xlabel("Time (s)", fontsize=17)
    fig.tight_layout()  # comment this out for the paper
    return fig


def matplot_wavelet_disc_spectrum(
    t_sampled: np.ndarray, coeffs: np.ndarray, freq_ranges: np.ndarray
):
    """
    Plot the wavelet spectrum (frequency decomposition) using Matplotlib with pcolormesh.

    Args:
        t_sampled (np.ndarray): Sampled time array.
        coeffs (np.ndarray): Wavelet decomposition coefficients (2D array).
        freq_ranges (np.ndarray): Frequency ranges for each level.

    Returns:
        fig: Matplotlib figure object.
    """

    # Compute magnitude of wavelet coefficients
    coeffs_magnitude = np.abs(coeffs)

    # Define frequency edges (including a 0 Hz boundary for spacing)
    freq_edges = np.concatenate([[0], freq_ranges[:, 1]])

    dt = t_sampled[1] - t_sampled[0]  # Assume uniform time step
    time_edges = np.concatenate([[t_sampled[0] - dt / 2], t_sampled + dt / 2])

    # Generate time-frequency mesh grid
    X, Y = np.meshgrid(time_edges, freq_edges)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot using pcolormesh
    cax = ax.pcolormesh(X, Y, coeffs_magnitude, cmap="seismic", shading="flat")

    # Formatting

    plt.ylabel("Frequency (Hz)", fontsize=17)
    plt.xlabel("Time (s)", fontsize=17)
    # ax.set_title("Wavelet Spectrum")
    ax.tick_params(axis="both", labelsize=12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))

    # Invert y-axis to have higher frequencies at the top
    # plt.gca().invert_yaxis()

    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax, location="bottom", aspect=50)
    cbar.set_label("Amplitude", fontsize=17)
    plt.tight_layout()
    return fig


####################################################################################################
#! Plotly plotting functions
####################################################################################################
def ploty_coeffs_reconst_signal(
    t: np.ndarray,
    b: np.ndarray,
    t_sampled: np.ndarray,
    b_sampled: np.ndarray,
    coeffs: np.ndarray,
    freq_bands: np.ndarray,
    common_xaxis: bool = True,
):
    """
    Plot the original signal and wavelet decomposition levels using Plotly.

    Args:
        t (np.ndarray): Time array.
        b (np.ndarray): Bandwidth.
        t_sampled (np.ndarray): Sampled time array.
        b_sampled (np.ndarray): Sampled bandwidth.
        coeffs (list): Wavelet decomposition coefficients.
        freq_bands (np.ndarray): Frequency ranges for each level.
        common_xaxis (bool): If True, enables a common x-axis for zooming.

    Returns:
        fig: Plotly figure object.
    """

    names = get_names(freq_bands, len(coeffs))
    fig = make_subplots(
        rows=len(coeffs) + 2,
        cols=1,
        subplot_titles=names,
        shared_xaxes=common_xaxis,
    )

    # Plot the original bandwidth
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="Bandwidth"), row=1, col=1)

    # Plot the original bandwidth
    fig.add_trace(
        go.Scatter(x=t_sampled, y=b_sampled, mode="lines", name="Sampled Bandwidth"),
        row=1,
        col=1,
    )
    # Plot the reconstructed signal
    fig.add_trace(
        go.Scatter(
            x=t_sampled,
            y=np.sum(coeffs, axis=0),
            mode="lines",
            name="Reconstructed Signal",
        ),
        row=1,
        col=1,
    )

    # Plot each decomposition level
    for i, coeff in enumerate(coeffs):
        fig.add_trace(
            go.Scatter(x=t_sampled, y=coeff, mode="lines", name=names[i + 1]),
            row=i + 2,
            col=1,
        )
        fig.update_yaxes(title_text="Amplitude", row=i + 2, col=1)
        fig.update_xaxes(showticklabels=True, title_text="Time (s)", row=i + 2, col=1)

    # Final layout adjustments
    fig.update_layout(
        showlegend=True,
        height=400 * (len(coeffs) + 1),  # Makes the figure scrollable
        yaxis_title="Bandwidth (B/s)",
        legend_title="Frequency Range",
    )

    return fig


def plotly_wavelet_disc_spectrum(
    t_sampled: np.ndarray, coeffs: np.ndarray, freq_ranges: np.ndarray
):
    """
    Plot the wavelet spectrum (frequency decomposition) using Plotly.

    Args:
        t_sampled (np.ndarray): Sampled time array.
        coeffs (np.ndarray): Wavelet decomposition coefficients (2D array).
        freq_ranges (np.ndarray): Frequency ranges for each level.

    Returns:
        fig: Plotly figure object.
    """

    # Compute the magnitude of the coefficients for visualization
    coeffs_magnitude = np.abs(coeffs)

    # Reverse the frequency order
    freq_labels = [f"{low:.3e} - {high:.3e} Hz" for low, high in freq_ranges]
    # coeffs_magnitude = np.flipud(coeffs_magnitude)  # Flip data to match reversed y-axis

    fig = go.Figure()

    # Create heatmap for wavelet spectrum
    fig.add_trace(
        go.Heatmap(
            x=t_sampled,  # Time
            y=freq_labels,  # Frequency bands (now reversed)
            z=coeffs_magnitude,  # Magnitude
            colorscale="Viridis",  # Color mapping
            colorbar={"title": "Magnitude"},
        )
    )

    # Update layout
    fig.update_layout(
        title="Wavelet Spectrum",
        height=500,
        xaxis={
            "title": "Time (s)",
            # range=[t[0], t[-1]],  # Ensures full time range
        },
        yaxis={
            "title": "Frequency ranges ",
            # autorange=False,
            # range=[freq_labels[0], freq_labels[-1]],
        },
    )

    return fig


####################################################################################################
#! Plotting functions
####################################################################################################
def plot_coeffs_reconst_signal(
    args: Namespace,
    t: np.ndarray,
    b: np.ndarray,
    t_sampled: np.ndarray,
    b_sampled: np.ndarray,
    coeffs_upsampled: np.ndarray,
    freq_bands: np.ndarray,
    common_xaxis: bool = True,
) -> plt.Figure:
    """
    Plot the reconstructed signal from the discrete wavelet coefficients at each level
    using either Plotly or Matplotlib based on the specified engine.

    Args:
        args (Namespace): Arguments containing the plotting engine information.
        t (np.ndarray): Time array.
        b (np.ndarray): Bandwidth array.
        t_sampled (np.ndarray): Sampled time array.
        b_sampled (np.ndarray): Sampled bandwidth array.
        coeffs_upsampled (np.ndarray): Upsampled coefficients from the DWT.
        freq_bands (np.ndarray): Frequency bands from the DWT.
        common_xaxis (bool, optional): Whether to use a common x-axis for the plots. Defaults to True.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = ploty_coeffs_reconst_signal(
            t,
            b,
            t_sampled,
            b_sampled,
            coeffs_upsampled,
            freq_bands,
            common_xaxis,
        )
        # create_html([fig], args.render, {"toImageButtonOptions": {"format": "png", "scale": 4}}, args.transformation)

    else:
        fig = matplot_coeffs_reconst_signal(
            t,
            b,
            t_sampled,
            b_sampled,
            coeffs_upsampled,
            freq_bands,
            common_xaxis,
        )

    return fig


def plot_wavelet_disc_spectrum(
    args: Namespace,
    t_sampled: np.ndarray,
    coeffs: np.ndarray,
    freq_bands: np.ndarray,
) -> plt.Figure:
    """
    Plot the wavelet spectrum from the discrete wavelet coefficients at each level
    using either Plotly or Matplotlib based on the specified engine.

    Args:
        args (Namespace): Arguments containing the plotting engine information.
        t_sampled (np.ndarray): Sampled time array.
        coeffs (np.ndarray): Wavelet decomposition coefficients (2D array).
        freq_bands (np.ndarray): Frequency bands from the DWT.

    Returns:
        plt.Figure: Matplotlib or Plotly figure object.
    """
    if "plotly" in args.engine:
        fig = plotly_wavelet_disc_spectrum(t_sampled, coeffs, freq_bands)
    else:
        fig = matplot_wavelet_disc_spectrum(t_sampled, coeffs, freq_bands)

    return fig
