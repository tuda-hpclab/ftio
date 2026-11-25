"""
Module for Dynamic Time Warping (DTW) calculations.

This module provides functionality for computing DTW distances using either
a NumPy-based implementation or FastDTW (if available). It also supports
multi-threaded DTW evaluation for improved performance.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Sep 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import importlib.util
from threading import Thread

import numpy as np
from scipy.spatial.distance import euclidean

# Check if fastdtw is available
FASTDTW_AVAILABLE: bool = importlib.util.find_spec("fastdtw") is not None
if FASTDTW_AVAILABLE:
    from fastdtw import fastdtw


def fill_dtw_cost_matrix(s1: np.ndarray, s2: np.ndarray) -> tuple:
    """
    Compute the DTW distance between two sequences using a cost matrix.

    Args:
        s1 (np.ndarray): First sequence.
        s2 (np.ndarray): Second sequence.

    Returns:
        tuple: The DTW distance and None (for compatibility with FastDTW return type).
    """
    l_s_1, l_s_2 = len(s1), len(s2)
    # Initialize the cost matrix
    cost_matrix = np.full((l_s_1 + 1, l_s_2 + 1), np.inf)
    cost_matrix[0, 0] = 0  # Starting point

    # Fill the cost matrix
    for i in range(1, l_s_1 + 1):
        for j in range(1, l_s_2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])  # Euclidean distance
            # Take the minimum of three previous entries (top, left, diagonal)
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],  # from top
                cost_matrix[i, j - 1],  # from left
                cost_matrix[i - 1, j - 1],  # diagonal
            )

    return cost_matrix[-1, -1], None


def fdtw(s1: np.ndarray, s2: np.ndarray) -> tuple:
    """
    Compute the DTW distance using FastDTW if available, otherwise fallback to a basic implementation.

    Args:
        s1 (np.ndarray): First sequence.
        s2 (np.ndarray): Second sequence.

    Returns:
        tuple: DTW distance and warping path (if FastDTW is used) or None (if using the fallback method).
    """
    if FASTDTW_AVAILABLE:
        return fastdtw(s1.reshape(-1, 1), s2.reshape(-1, 1), dist=euclidean)
    else:
        return fill_dtw_cost_matrix(s1, s2)


def evaluate_dtw(discret_arr, original_discret_signal, freq):
    """
    Evaluate the DTW distance between a discretized signal and its original.

    Args:
        discret_arr (array-like): The discretized signal.
        original_discret_signal (array-like): The original discretized signal.
        freq (float): The frequency associated with the signal.

    Prints:
        The computed DTW distance.
    """
    dtw_k1, _ = fdtw(discret_arr, original_discret_signal)
    print(f"    '-> \033[1;32mfreq {freq:.2f} Hz\033[1;0m --> dtw: {dtw_k1}")


def threaded_dtw(
    sum_all_components, df, dominant_X1, dominant_k1, dominant_X2, dominant_k2
):
    """
    Compute DTW distances for dominant components in parallel using threads.

    Args:
        sum_all_components (array-like): The summed signal components.
        df: DataFrame-like object containing frequency information.
        dominant_X1 (array-like): First dominant component.
        dominant_k1 (int): Index of the first dominant component in `df`.
        dominant_X2 (array-like): Second dominant component.
        dominant_k2 (int): Index of the second dominant component in `df`.

    Returns:
        list: List of active threads.
    """
    threads = []
    print("    '-> \033[1;35mCalculating DTW\033[1;0m")
    threads.append(
        Thread(
            target=evaluate_dtw,
            args=(
                dominant_X1,
                sum_all_components,
                df.iloc[dominant_k1]["freq"],
            ),
        )
    )
    threads.append(
        Thread(
            target=evaluate_dtw,
            args=(
                dominant_X2,
                sum_all_components,
                df.iloc[dominant_k2]["freq"],
            ),
        )
    )

    for thread in threads:
        thread.start()

    return threads
