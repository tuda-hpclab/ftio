"""
Wavelet functions (continuous and discrete)

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
import pywt
from scipy import signal

# from ftio.freq.helper import MyConsole


def wavelet_cont(
    b_sampled: np.ndarray, wavelet: str, scales: np.ndarray, freq: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform continuous wavelet transformation on a given signal.

    Args:
        b_sampled (np.ndarray): The input signal to be transformed.
        wavelet (str): The type of wavelet to use. E.g., 'morlet', 'cmor', etc.
        scales (np.ndarray): Array of scales
        freq (float): The sampling frequency of the input signal in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Coefficients (np.ndarray): The wavelet coefficients obtained after the transformation.
            - Frequencies (np.ndarray): The corresponding frequencies for each scale used in the transformation.
    """

    sampling_period = 1 / freq
    # console = MyConsole(True)
    # console.print(pywt.scale2frequency(wavelet, scale) / sampling_period)
    coefficients, frequencies = pywt.cwt(b_sampled, scales, wavelet, sampling_period)

    return coefficients, frequencies


def wavelet_disc(b_sampled: np.ndarray, wavelet: str, level: int) -> list[np.ndarray]:
    """
    Perform a discrete wavelet transformation (DWT) on the input signal.

    Args:
        b_sampled (np.ndarray): The input signal to be transformed as a 1D NumPy array.
        wavelet (str): The type of wavelet to use for the transformation.
        level (int): The decomposition level for the wavelet transformation.

    Returns:
        List[np.ndarray]: A list of NumPy arrays, where each array contains the wavelet coefficients
        for a specific level of decomposition.
    """
    # Perform Discrete Wavelet Transformation
    coefficients = pywt.wavedec(b_sampled, wavelet, level=level)
    return coefficients


def welch(b, freq):
    """Welch method for spectral density estimation.

    Args:
        b (list[float]): bandwidth over time
        freq (float): sampling frequency
    """
    # f, pxx_den = signal.welch(b, freq, 'flattop', 10000, scaling='spectrum', average='mean')
    f, pxx_den = signal.welch(
        b, freq, "flattop", freq * 256, scaling="spectrum", average="mean"
    )
    plt.semilogy(f, np.sqrt(pxx_den))
    plt.xlabel("frequency [Hz]")
    # plt.ylabel('PSD [V**2/Hz]')
    plt.ylabel("Linear spectrum [V RMS]")
    plt.show()
