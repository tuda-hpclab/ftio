"""
Contains DFT methods and accuracy calculation

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#!################
#! DFT amplitude and phase
#!################
def compute_dft_spectrum(b: np.ndarray, fs: float):
    """
    Compute the amplitude and phase of the Discrete Fourier Transform (DFT) of the signal.

    Parameters:
    - b: np.ndarray, input signal in the time domain.
    - fs: float, sampling frequency.

    Returns:
    - amp: np.ndarray, amplitudes of the frequency components.
    - phi: np.ndarray, phases of the frequency components.
    - freqs: np.ndarray, corresponding frequency bins.
    """
    # Compute DFT of the signal
    X = dft(b)
    n = len(X)

    # Calculate the amplitude (magnitude) of the frequency components
    amp = np.abs(X)

    # Calculate the phase (angle) of the frequency components
    phi = np.angle(X)

    # Compute the frequencies
    freqs = fs * np.arange(0, n) / n

    # Only keep the positive frequencies (half of the DFT result)
    indices = np.arange(0, int(len(amp) / 2) + 1)
    amp = amp[indices]  # Keep the amplitude
    amp[
        1:
    ] *= 2  # Double the amplitude for the positive frequencies (except the DC component)
    phi = phi[indices]  # Keep the corresponding phase values
    freqs = freqs[indices]

    return (
        amp,
        phi,
        freqs,
    )  # Return the amplitude, phase, and corresponding frequencies


#!################
#! DFT flavors
#!################
#! Wrapper
def dft(b: np.ndarray) -> np.ndarray:
    """
    Wrapper function to compute the DFT using numpy's FFT implementation.

    Parameters:
    - b: np.ndarray, input signal in the time domain.

    Returns:
    - np.ndarray, DFT of the input signal.
    """
    return numpy_dft(b)


#! 1) Custom implementation
def dft_fast(b: np.ndarray) -> np.ndarray:
    """
    Custom implementation of the Discrete Fourier Transform (DFT).

    Parameters:
    - b: np.ndarray, input signal in the time domain.

    Returns:
    - np.ndarray, DFT of the input signal.
    """
    N = len(b)
    X = np.repeat(complex(0, 0), N)  # np.zeros(N)
    for k in range(0, N):
        for n in range(0, N):
            X[k] = X[k] + b[n] * np.exp((-2 * np.pi * n * k / N) * 1j)

    return X


#! 2) numpy DFT
def numpy_dft(b: np.ndarray) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) using numpy's FFT implementation.

    Parameters:
    - b: np.ndarray, input signal in the time domain.

    Returns:
    - np.ndarray, DFT of the input signal.
    """
    return np.fft.fft(b)


#! 3) DFT with complex
def dft_slow(b: np.ndarray) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) using a slower, more explicit method.

    Parameters:
    - b: np.ndarray, input signal in the time domain.

    Returns:
    - np.ndarray, DFT of the input signal.
    """
    N = len(b)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, b)

    return X


#!################
#! DFT Precision
#!################
def precision_dft(
    amp: np.ndarray,
    phi: np.ndarray,
    dominant_index: np.ndarray,
    b_sampled: np.ndarray,
    t_disc: np.ndarray,
    freq_arr: np.ndarray,
    plt_engine: str,
) -> str:
    """calculates the precision of the dft

    Args:
        amp (np.ndarray): amplitude array from DFT
        phi (np.ndarray): phase array from DFT
        dominant_index (np.ndarray): index/indices of dominant frequency/frequencies
        b_sampled (np.ndarray): discretized bandwidth
        t_disc (np.ndarray): discretized time (constant step size). Start at t_0
        freq_arr (np.ndarray): frequency array
        plt_engine (str): command line specific plot engine

    Returns:
        str: precision
    """
    #
    showplot = False
    text = ""
    if showplot and ("mat" in plt_engine or "plotly" in plt_engine):
        plt.figure(figsize=(10, 5))
    dc_offset = np.zeros(len(amp))
    for index in dominant_index:
        x = dc_offset + 2 * (1 / len(amp)) * amp[index] * np.cos(
            2 * np.pi * np.arange(0, len(amp)) * (index) / (len(amp)) + phi[index]
        )
        x[x < 0] = 0
        x_2 = x.copy()
        total = np.sum(b_sampled)
        for i, _ in enumerate(x):
            if x[i] > b_sampled[i]:
                x_2[i] = b_sampled[i]
                x[i] = -x[i] + b_sampled[i]

        text += f"Precision of [cyan]{freq_arr[index]:.2f}[/] Hz is [cyan]{float(np.sum(x)) / total * 100:.2f}% [/]"
        text += f"(Positive only: [cyan]{float(np.sum(x_2)) / total * 100:.2f}%[/])\n"

        if showplot and ("mat" in plt_engine or "plotly" in plt_engine):
            plt.plot(t_disc, x, label=f"f = {freq_arr[index]:.2f} Hz")

    if showplot and ("mat" in plt_engine or "plotly" in plt_engine):
        plt.plot(t_disc, b_sampled, label="b_sampled")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel("Amplitude (B/s)", fontsize=17)
        plt.xlabel("Time (s)", fontsize=17)
        plt.grid(True)
        plt.legend(loc="upper left", ncol=2, fontsize=13)
        plt.tight_layout()
        plt.show()

    return text


def prepare_plot_dft(
    freq_arr: np.ndarray,
    conf: np.ndarray,
    dominant_index: list[float],
    amp: np.ndarray,
    phi: np.ndarray,
    b_sampled: np.ndarray,
    ranks: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Prepares data for plotting the Discrete Fourier Transform (DFT) by creating two DataFrames.

    Args:
        freq_arr: An array of frequency values.
        conf: An array of confidence values corresponding to the frequencies.
        dominant_index: The index (or indices) of the dominant frequency component.
        amp: An array of amplitudes corresponding to the frequencies.
        phi: An array of phase values corresponding to the frequencies.
        b_sampled: An array of sampled bandwidth values.
        ranks: The number of ranks.

    Returns:
        tuple[list[pd.DataFrame], list[pd.DataFrame]]: A tuple containing two lists of DataFrames:
            - The first list contains a DataFrame with amplitude (A), phase (phi), sampled bandwidth (b_sampled), ranks, frequency (freq),
            period (T), and confidence (conf) values.
            - The second list contains a DataFrame with the dominant frequency, its index (k), its confidence, and the ranks.
    """
    df0 = []
    df1 = []

    df0.append(
        pd.DataFrame(
            {
                "A": amp,
                "phi": phi,
                "b_sampled": b_sampled,
                "ranks": np.repeat(ranks, len(b_sampled)),
                "k": np.arange(0, len(b_sampled)),
                "freq": freq_arr,
                "T": np.concatenate([np.array([0]), 1 / freq_arr[1:]]),
                "conf": conf,
            }
        )
    )
    df1.append(
        pd.DataFrame(
            {
                "dominant": freq_arr[dominant_index],
                "k": dominant_index,
                "conf": conf[dominant_index],
                "ranks": np.repeat(ranks, len(dominant_index)),
            }
        )
    )
    return df0, df1
