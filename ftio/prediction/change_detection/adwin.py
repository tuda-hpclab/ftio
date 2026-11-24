"""
Change point detection algorithms for FTIO online predictor.

This module provides functional ADWIN change point detection algorithms for detecting
I/O pattern changes in streaming data.

Author: Amine Aherbil
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import math
from argparse import Namespace
from typing import Any

import numpy as np

from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.prediction.change_detection.helper import get_frequencies, get_timestamps
from ftio.prediction.helper import get_dominant
from ftio.prediction.shared_resources import SharedResources

CONSOLE = MyConsole()
CONSOLE.set(True)


def adwin_step(
    freq: float,
    timestamp: float,
    state: dict[str, Any],
    delta: float = 0.05,
    verbose: bool = False,
) -> tuple[int | None, float | None, dict[str, Any]]:
    """
    Perform one step of the ADWIN algorithm.

    Returns:
        A tuple of (change_point_index, change_timestamp, new_state).
    """
    CONSOLE.set(verbose)
    new_state = state.copy()
    min_window_size = 2

    if np.isnan(freq) or freq <= 0:
        if verbose:
            CONSOLE.print(
                "[yellow][ADWIN] No frequency found - resetting window history[/]"
            )
        return None, None, {"frequencies": [], "timestamps": [], "frequencies_found": 0}

    frequencies = new_state.get("frequencies", [])
    timestamps = new_state.get("timestamps", [])
    frequencies.append(freq)
    timestamps.append(timestamp)
    frequencies_found = new_state.get("frequencies_found", 0) + 1

    if len(frequencies) < 2 * min_window_size:
        new_state.update(
            {
                "frequencies": frequencies,
                "timestamps": timestamps,
                "frequencies_found": frequencies_found,
            }
        )
        return None, None, new_state

    # Detect change
    change_point = None
    n = len(frequencies)
    for cut in range(min_window_size, n - min_window_size + 1):
        if _test_cut_point(frequencies, cut, delta, verbose):
            if verbose:
                CONSOLE.print(
                    f"[blue][ADWIN] Change detected at position {cut}/{n}, "
                    f"time={timestamps[cut]:.3f}s[/]"
                )
            change_point = cut
            break

    if change_point is not None:
        change_time = timestamps[change_point]

        # Trim window
        new_frequencies = frequencies[change_point:]
        new_timestamps = timestamps[change_point:]

        new_state = {
            "frequencies": new_frequencies,
            "timestamps": new_timestamps,
            "frequencies_found": len(new_frequencies),
            "last_change_point": change_point,
            "last_change_time": change_time,
        }
        return change_point, change_time, new_state

    new_state.update(
        {
            "frequencies": frequencies,
            "timestamps": timestamps,
            "frequencies_found": frequencies_found,
        }
    )
    return None, None, new_state


def _test_cut_point(
    frequencies: list[float], cut: int, delta: float, verbose: bool
) -> bool:
    left_data = frequencies[:cut]
    right_data = frequencies[cut:]

    n0 = len(left_data)
    n1 = len(right_data)

    mean0 = np.mean(left_data)
    mean1 = np.mean(right_data)
    mean_diff = abs(mean1 - mean0)

    n_harmonic = (n0 * n1) / (n0 + n1)
    confidence_term = math.log(2.0 / delta) / (2.0 * n_harmonic)
    threshold = math.sqrt(2.0 * confidence_term)

    if verbose:
        CONSOLE.print(
            f"[dim blue]ADWIN Cut={cut}: Δ={mean_diff:.3f}, ε={threshold:.3f}[/]"
        )

    return mean_diff > threshold


def get_adaptive_start_time(
    current_time: float, last_change_time: float | None, original_start_time: float
) -> float:
    if last_change_time is not None:
        min_window = 0.5
        max_lookback = 10.0
        window_span = current_time - last_change_time

        if window_span < min_window:
            return max(0.0, current_time - min_window)
        elif window_span > max_lookback:
            return max(0.0, current_time - max_lookback)
        else:
            return last_change_time

    return original_start_time


def detect_pattern_change_adwin(
    args: Namespace,
    shared_resources: SharedResources,
    current_prediction: Prediction,
    counter: int,
) -> tuple[bool, str | None, float, float | None, float | None]:
    """Functional wrapper for ADWIN detection."""
    current_freq = get_dominant(current_prediction)
    current_time = current_prediction.t_end

    # Restore state
    state = shared_resources.online_detection.get("state", {})
    if "frequencies" not in state:
        state["frequencies"] = get_frequencies(list(shared_resources.data))
        state["timestamps"] = get_timestamps(list(shared_resources.data))

    change_idx, change_time, new_state = adwin_step(
        current_freq, current_time, state, delta=0.05, verbose=args.verbose
    )

    change_detected = False
    new_start_time = current_prediction.t_start
    change_log, old_freq = None, None

    if change_idx is not None:
        change_detected = True
        old_freq = (
            np.mean(state["frequencies"][:change_idx]) if change_idx > 0 else current_freq
        )

        freq_change_pct = (
            abs(current_freq - old_freq) / old_freq * 100 if old_freq > 0 else 0
        )

        change_log = (
            f"[magenta bold][CHANGE_POINT] t_s={change_time:.3f} sec[/]\n"
            f"[purple][PREDICTOR] (#{counter}):[/][yellow] "
            f"ADWIN detected pattern change: {old_freq:.3f} → {current_freq:.3f} Hz "
            f"({freq_change_pct:.1f}% change)[/]\n"
            f"[dim blue]ADWIN ANALYSIS: Statistical significance detected using Hoeffding bounds[/]"
        )

        new_start_time = get_adaptive_start_time(
            current_time, change_time, current_prediction.t_start
        )

        shared_resources.online_detection["change_count"] = (
            shared_resources.online_detection.get("change_count", 0) + 1
        )
        shared_resources.online_detection["last_change_time"] = change_time

        if args.gui:
            try:
                from ftio.gui.socket_logger import get_socket_logger

                logger = get_socket_logger()
                if logger is not None:
                    logger.send_log(
                        "change_point",
                        "ADWIN Change Point Detected",
                        {
                            "exact_time": change_time,
                            "old_freq": old_freq,
                            "new_freq": current_freq,
                            "adaptive_start": new_start_time,
                            "counter": counter,
                        },
                    )
            except ImportError:
                pass
    else:
        old_freq = (
            np.mean(new_state["frequencies"])
            if new_state["frequencies"]
            else current_freq
        )

    # Save state
    shared_resources.online_detection["state"] = new_state

    return change_detected, change_log, new_start_time, old_freq, current_freq
