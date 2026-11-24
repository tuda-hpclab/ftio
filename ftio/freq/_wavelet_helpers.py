"""
Helper functions for wavelet transformation in the FTIO package.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from argparse import Namespace

import numpy as np
import pywt

from ftio.freq.helper import MyConsole


def get_scales(args: Namespace, b_sampled: np.ndarray) -> np.ndarray:
    """
    Calculate the scales for wavelet transformation based on the given arguments.

    Args:
        args (Namespace): A namespace object containing the arguments, including the decomposition level.
        b_sampled (np.ndarray): The sampled data array.

    Returns:
        np.ndarray: An array of scales to be used for the wavelet transformation.
    """
    if args.level == 0:
        args.level = decomposition_level(args, len(b_sampled))

    scales = np.arange(1, args.level)  # 2** mimics the DWT
    # scale = np.arange(1, args.level,0.1)  # 2** mimics the DWT

    return scales


def decomposition_level(args: Namespace, n: int) -> int:
    """
    Determine the decomposition level for wavelet transformation.

    Args:
        args (Namespace): Parsed arguments containing `level` and `transformation` attributes.
            - `args.level` (int): Specifies the decomposition level. If set to 0, the level is determined automatically.
            - `args.transformation` (str): Specifies the type of transformation ('wave_cont' for continuous or 'wave_disc' for discrete).
        n (int): The length of the input signal.

    Returns:
        int: The decomposition level to be used for the wavelet transformation.
    """
    level = args.level
    console = MyConsole(True)
    console.print(f"[green]Decomposition level is {level}[/]")

    if args.level == 0:
        if "wave_cont" in args.transformation:
            level = 10
            console.print(f"[green]Decomposition level set to {level}[/]")
        else:
            level = pywt.dwt_max_level(n, args.wavelet)
            console.print(f"[green]Decomposition level optimally adjusted to {level}[/]")

    if "wave_cont" in args.transformation:
        check_wavelet(args.wavelet, "continuous")
    elif "wave_disc" in args.transformation:
        check_wavelet(args.wavelet, "discrete")
    else:
        pass

    return level


def check_wavelet(wavelet: str, mode: str = "discrete") -> None:
    """
    Check that the wavelet selected is supported.

    Args:
        wavelet (str): Wavelet name.
        mode (str, optional): Wavelet mode. Defaults to "discrete".

    Raises:
        ValueError: If the specified wavelet is not supported.
    """
    check = False
    for supported_wavelet in pywt.wavelist(kind=mode):
        if supported_wavelet in wavelet:
            check = True
            break
    if check is False:
        raise ValueError(
            f"Unsupported wavelet specified, supported modes are: {pywt.wavelist(kind=mode)}\nsee: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html "
        )
    else:
        return


def wavelet_freq_bands(f_s: float, levels: int):
    """
    Compute frequency ranges for each wavelet decomposition level.

    Parameters:
        f_s (float): Sampling frequency (Hz)
        levels (int): Number of decomposition levels

    Returns:
        np.ndarray: 2D array with low and high frequency ranges for each level.
    """

    low_freq = f_s / 2 ** (np.arange(start=levels, stop=0, step=-1) + 1)
    high_freq = f_s / 2 ** np.arange(start=levels, stop=0, step=-1)
    if len(low_freq) > 0:
        high_freq = np.insert(high_freq, 0, low_freq[0])
        low_freq = np.insert(low_freq, 0, 0)

    return np.column_stack(
        (low_freq, high_freq)
    )  # High frequencies are higher at finer levels


def upsample_coefficients(
    coefficients: list[np.ndarray], wavelet="db1", signal_length: int = 0
) -> np.ndarray:
    """
    Extend wavelet coefficients to the same length as the original signal using upcoef.

    Parameters:
        coefficients (list): List of wavelet coefficients (approximations and details) for each level.
        wavelet (str): The type of wavelet used (default is 'db1').
        signal_length (int): The length of the original signal to extend the coefficients to.

    Returns:
        list: List of coefficients extended to the length of the original signal.
    """
    coefficients_stretched = []
    level = len(coefficients)  # the low pass component doesn't count as a level
    for i in np.arange(start=0, stop=level, step=1):
        # Extend the coefficients using pywt.upcoef
        if i == 0:
            # Approximation coefficients (cA)
            reconstructed = pywt.upcoef("a", coefficients[0], wavelet, level=level - 1)[
                :signal_length
            ]
        else:
            # Detail coefficients (cD)
            reconstructed = pywt.upcoef("d", coefficients[i], wavelet, level=level - i)[
                :signal_length
            ]

        coefficients_stretched.append(reconstructed)

    return np.array(coefficients_stretched)
