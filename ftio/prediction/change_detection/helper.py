"""
Change point detection algorithms helpers

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: January 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np

from ftio.prediction.helper import get_dominant


def get_frequencies(data):
    """
    Extract dominant frequencies from prediction data.

    Args:
        data (list): Iterable of prediction objects or structures.

    Returns:
        list[float]: List of dominant frequencies.
    """
    return [get_dominant(d) for d in data]


def get_timestamps(data):
    """
    Extract end timestamps from prediction data.

    Args:
        data (list): Iterable of prediction dictionaries.

    Returns:
        list[float]: List of end timestamps.
    """
    return [d["t_end"] for d in data]


def safe_float(x: float | None) -> float:
    """
    Convert a value to float, returning 0.0 for None or NaN.

    Args:
        x (float | None): Input value.

    Returns:
        float: Safe float value.
    """
    return float(x) if x is not None and not np.isnan(x) else 0.0


def change_post_processing(args, prediction_count, t_e, t_s, adaptive_start_time, text):
    """
    Safely update the start time after successful change detection.

    The start time is modified only if online change detection
    produced output text.

    Args:
        args (Namespace): Command-line arguments.
        prediction_count (int): Current prediction index.
        t_e (float): End time of the prediction window.
        t_s (float): Current start time of the prediction window.
        adaptive_start_time (float): Proposed new start time.
        text (str): Accumulated log text.

    Returns:
        tuple[float, str]: Updated start time and log text.
    """
    if text:
        # Safe adaptive start
        algorithm_name = args.window_adaptation.upper()
        min_window_size = 1.0
        safe_adaptive_start = min(adaptive_start_time, t_e - min_window_size)
        if safe_adaptive_start >= 0 and (t_e - safe_adaptive_start) >= min_window_size:
            t_s = safe_adaptive_start
            text += f"[purple][PREDICTOR] (#{prediction_count}):[/][green] {algorithm_name} adapted window to start at {t_s:.3f}s (window size: {t_e - t_s:.3f}s)[/]\n"
        else:
            t_s = max(0, t_e - min_window_size)
            text += f"[purple][PREDICTOR] (#{prediction_count}):[/][yellow] {algorithm_name} adaptation would create unsafe window, using conservative {min_window_size}s window[/]\n"
    return t_s, text


def create_change_point_info(
    prediction_count, t_e, old_freq, new_freq, adaptive_start_time, frequency_count
):
    """
    Create structured information describing a detected change point.

    Args:
        prediction_count (int): Prediction identifier.
        t_e (float): Timestamp of the change.
        old_freq (float | None): Previous dominant frequency.
        new_freq (float | None): New dominant frequency.
        adaptive_start_time (float): Window start time after adaptation.
        frequency_count (int): Number of frequency samples.

    Returns:
        dict: Change point metadata.
    """

    old_freq = safe_float(old_freq)
    new_freq = safe_float(new_freq)
    frequency_change_percent = (
        abs(new_freq - old_freq) / old_freq * 100 if old_freq > 0 else 0.0
    )
    change_point_info = {
        "prediction_id": prediction_count,
        "timestamp": float(t_e),
        "old_frequency": old_freq,
        "new_frequency": new_freq,
        "frequency_change_percent": frequency_change_percent,
        "frequency_count": frequency_count,
        "cut_position": frequency_count - 1 if frequency_count > 0 else 0,
        "total_samples": frequency_count,
        "start_time": float(adaptive_start_time),
    }
    return change_point_info
