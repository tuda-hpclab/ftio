"""
Change point detection algorithms for FTIO online predictor.

This module provides functional CUSUM change point detection algorithms for detecting
I/O pattern changes in streaming data.
It includes AV-CUSUM: Adaptive-Variance Cumulative Sum.

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


def cusum_step(
    freq: float,
    timestamp: float,
    state: dict[str, Any],
    window_size: int = 50,
    verbose: bool = False,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """
    Perform one step of the AV-CUSUM algorithm.

    Args:
        freq: Current frequency.
        timestamp: Current timestamp.
        state: Current algorithm state.
        window_size: Window size for rolling statistics.
        verbose: Enable verbose logging.

    Returns:
        A tuple of (change_detected, change_info, new_state).
    """
    CONSOLE.set(verbose)
    new_state = state.copy()

    if np.isnan(freq) or freq <= 0:
        if verbose:
            CONSOLE.print(
                "[yellow][AV-CUSUM] No frequency found - resetting algorithm state[/]"
            )
        return (
            False,
            {},
            {
                "sum_pos": 0.0,
                "sum_neg": 0.0,
                "reference": None,
                "frequencies": [],
                "timestamps": [],
            },
        )

    # Initialize or update buffers
    frequencies = new_state.get("frequencies", [])
    timestamps = new_state.get("timestamps", [])
    frequencies.append(freq)
    timestamps.append(timestamp)

    # Update adaptive parameters
    rolling_std = 0.0
    adaptive_threshold = 0.0
    adaptive_drift = 0.0

    recent_frequencies = []
    if len(frequencies) > 1:
        recent_frequencies = frequencies[-window_size - 1 : -1]

    if len(recent_frequencies) >= 3:
        rolling_std = np.std(np.array(recent_frequencies))
        std_factor = max(rolling_std, 0.01)
        adaptive_threshold = 2.0 * std_factor
        adaptive_drift = 0.5 * std_factor

        if verbose:
            CONSOLE.print(
                f"[dim cyan][CUSUM] σ={rolling_std:.3f}, "
                f"h_t={adaptive_threshold:.3f} (2σ threshold), "
                f"k_t={adaptive_drift:.3f} (0.5σ drift)[/]"
            )

    # Detect change logic
    min_init_samples = 3
    change_detected = False

    sum_pos = new_state.get("sum_pos", 0.0)
    sum_neg = new_state.get("sum_neg", 0.0)
    reference = new_state.get("reference", None)

    # Initialize reference if not present
    if len(frequencies) >= min_init_samples and reference is None:
        reference = np.mean(frequencies[:min_init_samples])
        if verbose:
            CONSOLE.print(
                f"[cyan][AV-CUSUM] Reference established: {reference:.3f} Hz "
                f"(from first {min_init_samples} observations)[/]"
            )

    # Build initial change info dict
    change_info = {
        "timestamp": timestamp,
        "frequency": freq,
        "reference": reference,
        "sum_pos": sum_pos,
        "sum_neg": sum_neg,
        "threshold": adaptive_threshold,
        "rolling_std": rolling_std,
        "change_type": "none",
    }

    if len(frequencies) < min_init_samples or adaptive_threshold <= 0:
        if verbose:
            CONSOLE.print(
                f"[dim yellow][AV-CUSUM] Collecting calibration data "
                f"({len(frequencies)}/{min_init_samples})[/]"
            )
        new_state.update(
            {
                "sum_pos": sum_pos,
                "sum_neg": sum_neg,
                "reference": reference,
                "frequencies": frequencies,
                "timestamps": timestamps,
            }
        )
        return False, change_info, new_state

    # Compute deviation and update cumulative sums
    deviation = freq - reference
    sum_pos = max(0, sum_pos + deviation - adaptive_drift)
    sum_neg = max(0, sum_neg - deviation - adaptive_drift)

    if verbose:
        current_window_size = len(frequencies)
        CONSOLE.print(
            f"[dim yellow][AV-CUSUM DEBUG] Observation #{current_window_size}:[/]\n"
            f"  [dim]• Current freq: {freq:.3f} Hz[/]\n"
            f"  [dim]• Reference: {reference:.3f} Hz[/]\n"
            f"  [dim]• Deviation: {deviation:.3f}[/]\n"
            f"  [dim]• Sum_pos: {sum_pos:.3f}[/]\n"
            f"  [dim]• Sum_neg: {sum_neg:.3f}[/]\n"
            f"  [dim]• Threshold: {adaptive_threshold:.3f}[/]"
        )

    # Detect change
    change_detected = sum_pos > adaptive_threshold or sum_neg > adaptive_threshold
    change_type = (
        "increase"
        if sum_pos > adaptive_threshold
        else "decrease" if sum_neg > adaptive_threshold else "none"
    )
    change_percent = abs(deviation / reference * 100) if reference else 0

    change_info = {
        "timestamp": timestamp,
        "frequency": freq,
        "reference": reference,
        "sum_pos": sum_pos,
        "sum_neg": sum_neg,
        "threshold": adaptive_threshold,
        "rolling_std": rolling_std,
        "deviation": deviation,
        "change_type": change_type,
    }

    if change_detected:
        if verbose:
            CONSOLE.print(
                f"[bold yellow][AV-CUSUM] CHANGE DETECTED! {reference:.3f}Hz → {freq:.3f}Hz ({change_percent:.1f}% {change_type})[/]"
            )

        # Reset for next time
        new_state = {
            "sum_pos": 0.0,
            "sum_neg": 0.0,
            "reference": freq,
            "frequencies": [freq],
            "timestamps": [timestamp],
        }
    else:
        new_state.update(
            {
                "sum_pos": sum_pos,
                "sum_neg": sum_neg,
                "reference": reference,
                "frequencies": frequencies,
                "timestamps": timestamps,
            }
        )

    return change_detected, change_info, new_state


def detect_pattern_change_cusum(
    args: Namespace,
    shared_resources: SharedResources,
    current_prediction: Prediction,
    counter: int,
) -> tuple[bool, str | None, float, float | None, float | None]:
    """
    Functional wrapper for CUSUM change point detection.
    """
    current_freq = get_dominant(current_prediction)
    current_time = current_prediction.t_end
    new_start_time = current_prediction.t_start

    # Restore state from shared resources
    state = shared_resources.online_detection.get("state", {})
    # Also ensure frequencies/timestamps are synced if not in state
    if "frequencies" not in state:
        state["frequencies"] = get_frequencies(list(shared_resources.data))
        state["timestamps"] = get_timestamps(list(shared_resources.data))

    change_detected, change_info, new_state = cusum_step(
        current_freq, current_time, state, window_size=50, verbose=args.verbose
    )

    change_log = None
    reference = None

    if change_detected:
        change_type = change_info["change_type"]
        reference = change_info["reference"]
        threshold = change_info["threshold"]
        sum_pos = change_info["sum_pos"]
        sum_neg = change_info["sum_neg"]

        magnitude = abs(current_freq - reference)
        percent_change = (magnitude / reference * 100) if reference > 0 else 0

        change_log = (
            f"[bold magenta][CUSUM] CHANGE DETECTED! "
            f"{reference:.1f}Hz → {current_freq:.1f}Hz "
            f"(Δ={magnitude:.1f}Hz, {percent_change:.1f}% {change_type}) "
            f"at sample {len(shared_resources.data)}, "
            f"time={current_time:.3f}s[/]\n"
            f"[magenta][CUSUM] CUSUM stats: sum_pos={sum_pos:.2f}, "
            f"sum_neg={sum_neg:.2f}, threshold={threshold}[/]\n"
            f"[magenta][CUSUM] Cumulative sum exceeded threshold -> "
            f"Starting fresh analysis[/]"
        )

        min_window_size = (
            0.5 if percent_change > 100 else 1.0 if percent_change > 50 else 2.0
        )
        new_start_time = max(0, current_time - min_window_size)

        shared_resources.online_detection["change_count"] = (
            shared_resources.online_detection.get("change_count", 0) + 1
        )
        shared_resources.online_detection["last_change_time"] = current_time

        if args.gui:
            try:
                from ftio.prediction.online_analysis import get_socket_logger

                logger = get_socket_logger()
                if logger is not None:
                    logger.send_log(
                        "change_point",
                        "CUSUM Change Point Detected",
                        {
                            "algorithm": "CUSUM",
                            "detection_time": current_time,
                            "change_type": change_type,
                            "frequency": current_freq,
                            "reference": reference,
                            "magnitude": magnitude,
                            "percent_change": percent_change,
                            "threshold": threshold,
                            "counter": counter,
                        },
                    )
            except ImportError:
                pass

    # Save state back to shared resources
    shared_resources.online_detection["state"] = new_state

    # Ensure reference is returned even if no change (for display/logging in caller)
    if not reference:
        reference = new_state.get("reference")

    return change_detected, change_log, new_start_time, reference, current_freq
