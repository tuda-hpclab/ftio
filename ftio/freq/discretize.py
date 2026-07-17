"""
Module contains function functions to discretize the data.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
from numba import jit
from rich.panel import Panel

from ftio.freq.helper import MyConsole


def sample_data(
    b: np.ndarray,
    t: np.ndarray,
    args: Namespace = None,
) -> tuple[np.ndarray, float]:
    """
    Samples the data at equal time steps  according to the specified frequency.

    Args:
        b (np.ndarray): Bandwidth values.
        t (np.ndarray): Time points corresponding to the bandwidth values.
        args (Namespace): Parsed arguments (see io_args.py) containing:
            - freq (float, optional): Sampling frequency. Defaults to -1, which triggers
                automatic calculation of the optimal sampling frequency.
            - memory_limit (float, optional): memory limit in case freq is -1.
            - verbose (bool, optional): Flag to indicate if information about the sampling
                process, including the time window, frequency step, and abstraction error is
                printed

    Returns:
        tuple: A tuple containing:
            - b_sampled (np.ndarray): Uniform sampled bandwidth values.
            - freq (float): The frequency used for sampling (calculated if freq was -1).

    Raises:
        RuntimeError: If no data is found in the sampled bandwidth.
    """
    if args is not None:
        freq = args.freq
        memory_limit = args.memory_limit * 1000**3  # args.memory_limit GB
        verbose = args.verbose
    else:
        freq = -1
        memory_limit = 2 * 1000**3  # 2 GB
        verbose = False

    duration = t[-1] - t[0] if t[-1] != t[0] else 0.0
    text = (
        f"Time window: {t[-1]-t[0]:.2f} s\n"
        f"Frequency step: {1/ duration if duration > 0 else 0:.3e} Hz\n"
    )

    # Callers unpack two values.
    if len(t) == 0:
        return np.empty(0), 0

    # Calculate recommended frequency:
    if freq == -1:
        # Auto-detect frequency based on smallest time delta
        t_rec = find_lowest_time_change(t)
        freq = 2 / t_rec
        text += f"Recommended sampling frequency: {freq:.3e} Hz\n"
        # Apply limit if freq is negative
        N = int(np.floor((t[-1] - t[0]) * freq))
        limit_N = int(memory_limit // np.dtype(np.float64).itemsize)
        text += f"memory limit: {memory_limit/ 1000**3:.3e} GB ({limit_N} samples)\n"
        if limit_N < N:
            N = limit_N
            freq = N / duration if duration > 0 else 10
            text += f"[yellow]Adjusted sampling frequency due to memory limit: {freq:.3e} Hz[/])\n"
    else:
        text += f"Sampling frequency:  {freq:.3e} Hz\n"
        # Compute number of samples
        N = int(np.floor((t[-1] - t[0]) * freq))

    text += f"Expected samples: {N}\n"

    if N <= 0:
        raise RuntimeError(
            f"Invalid sample count N={N} (freq={freq:.3e} Hz, duration={t[-1]-t[0]:.3f} s). "
            "Timestamps may be non-monotonic or the time window is too short."
        )

    #  sample the data with the recommended frequency
    # t_sampled = np.zeros(N) #t[0]+np.arange(N)*1/freq
    b_sampled = np.zeros(N)
    n = len(t)
    counter = 0
    n_old = 0
    t_step = t[0]
    # error   = 0
    # errorStep   = 0
    for _ in range(0, N):
        for i in range(min(n_old, n - 1), n):
            if (t_step >= t[i]) and (t_step < t[i + 1]) if i < n - 1 else True:
                n_old = i  # no need to iterate over entire array
                b_sampled[counter] = b[i]
                counter = counter + 1
                break
        t_step = t_step + 1 / freq

    #! Abstraction error
    v_a = np.sum(np.abs(b_sampled * np.repeat(1 / freq, len(b_sampled))))
    # Ensure lengths match for v_0 calculation
    min_len = min(len(b), len(t))
    b_sub = b[:min_len]
    t_sub = t[:min_len]
    v_0 = np.sum(b_sub * (np.concatenate([t_sub[1:], t_sub[-1:]]) - t_sub))

    # E = (abs(error))/(V0) if V0 > 0 else 0
    error = (abs(v_a - v_0)) / v_0 if v_0 > 0 else 0
    text += f"Abstraction error: {error:.5e}\n"

    if len(b_sampled) == 0:
        raise RuntimeError(
            "No data in sampled bandwidth.\n Try increasing the sampling frequency"
        )

    console = MyConsole(verbose)
    console.print(
        Panel.fit(
            text[:-1],
            style="white",
            border_style="yellow",
            title="Discretization",
            title_align="left",
        )
    )

    return b_sampled, freq


def sample_data_same_size(
    b: np.ndarray, t: np.ndarray, freq=-1, n_bins=-1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discretize the data according to the specified frequency, ensuring the sampled data has the same
    number of bins as specified.

    This function samples the bandwidth data at the given frequency and returns the sampled
    data along with the corresponding time points, ensuring the number of bins is consistent with the provided `n_bins` value.

    Args:
        b (np.ndarray): Bandwidth values.
        t (np.ndarray): Time points corresponding to the bandwidth values.
        freq (float, optional): Sampling frequency. Defaults to -1, which triggers automatic calculation of the optimal sampling frequency.
        n_bins (int, optional): The desired number of bins for the sampled data.


    Returns:
        tuple: A tuple containing:
            - b_sampled (np.ndarray): Sampled bandwidth values.
            - t_sampled (np.ndarray): Time points corresponding to the sampled data.
    """
    print(f"    '-> \033[1;34mBins: {n_bins}  \033[1;0m")
    b_sampled = np.zeros(n_bins)
    n_old = 0
    counter = 0
    t_step = 0
    n = len(b)
    for _k in range(0, n_bins):
        if (t_step < t[0]) or (t_step > t[-1]):
            counter = counter + 1
        else:
            for i in range(n_old, n):
                if (t_step >= t[i]) and (t_step < t[i + 1]) or i == n - 1:
                    n_old = i  # no need to iterate over an entire array
                    b_sampled[counter] = b[i]
                    counter = counter + 1
                    break
        t_step = t_step + 1 / freq

    t = np.arange(0, n_bins) * 1 / freq
    return b_sampled, t


@jit(nopython=True, cache=True)
def find_lowest_time_change(t: np.ndarray) -> float:
    """finds the lowest time change

    Args:
        t (np.ndarray): array of time stamps

    Returns:
        float: smallest time change
    """
    t_rec = np.inf
    for i in range(0, len(t) - 1):
        delta = t[i + 1] - t[i]
        if 0 < delta < t_rec:
            t_rec = delta

    # no need, as we now limit per memory
    # if t_rec <= 0.001:
    #     t_rec = 0.001

    return t_rec
