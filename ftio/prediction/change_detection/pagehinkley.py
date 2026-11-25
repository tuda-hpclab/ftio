"""
Change point detection algorithms for FTIO online predictor.
Self-Tuning Page-Hinkley test for sequential change point detection

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


def pagehinkley_step(
    freq: float,
    timestamp: float,
    state: dict[str, Any],
    window_size: int = 50,
    verbose: bool = False,
) -> tuple[bool, float, dict[str, Any], dict[str, Any]]:
    """
    Perform one step of the Page-Hinkley algorithm.

    Returns:
        A tuple of (change_detected, triggering_sum, change_info, new_state).
    """
    CONSOLE.set(verbose)
    new_state = state.copy()

    if np.isnan(freq) or freq <= 0:
        if verbose:
            CONSOLE.print(
                "[yellow][STPH] No frequency found - resetting Page-Hinkley state[/]"
            )
        return False, 0.0, {}, {"initialized": False}

    # Initialize or update buffers
    frequencies = new_state.get("frequencies", [])
    timestamps = new_state.get("timestamps", [])
    frequencies.append(freq)
    timestamps.append(timestamp)

    # Adaptive parameters
    rolling_std = 0.0
    adaptive_threshold = 0.0
    adaptive_delta = 0.0

    recent_frequencies = []
    if len(frequencies) > 1:
        recent_frequencies = frequencies[-window_size - 1 : -1]

    if len(recent_frequencies) >= 3:
        rolling_std = np.std(np.array(recent_frequencies))
        std_factor = max(rolling_std, 0.01)
        adaptive_threshold = 2.0 * std_factor
        adaptive_delta = 0.5 * std_factor

        if verbose:
            CONSOLE.print(
                f"[dim magenta][Page-Hinkley] σ={rolling_std:.3f}, "
                f"λ_t={adaptive_threshold:.3f} (2σ threshold), "
                f"δ_t={adaptive_delta:.3f} (0.5σ delta)[/]"
            )

    # State variables
    sum_pos = new_state.get("cumulative_sum_pos", 0.0)
    sum_neg = new_state.get("cumulative_sum_neg", 0.0)
    ref_mean = new_state.get("reference_mean", 0.0)
    sum_samples = new_state.get("sum_of_samples", 0.0)
    sample_count = new_state.get("sample_count", 0)

    # Update mean baseline
    if sample_count == 0:
        sample_count = 1
        ref_mean = freq
        sum_samples = freq
        if verbose:
            CONSOLE.print(
                f"[yellow][STPH] Reference mean initialized: {ref_mean:.3f} Hz[/]"
            )
    else:
        sample_count += 1
        sum_samples += freq
        ref_mean = sum_samples / sample_count

    # Differences
    pos_difference = freq - ref_mean - adaptive_delta
    sum_pos = max(0, sum_pos + pos_difference)

    neg_difference = ref_mean - freq - adaptive_delta
    sum_neg = max(0, sum_neg + neg_difference)

    if verbose:
        CONSOLE.print(
            f"[dim cyan][STPH DEBUG] Sample #{sample_count}:[/]\n"
            f"  • Current freq: {freq:.3f} Hz\n"
            f"  • Ref mean: {ref_mean:.3f} Hz\n"
            f"  • Sum_pos: {sum_pos:.3f}, Sum_neg: {sum_neg:.3f}\n"
            f"  • Threshold: {adaptive_threshold:.3f}"
        )

    # Detection logic
    change_detected = False
    triggering_sum = 0.0
    change_type = "none"

    if len(frequencies) >= 3 and adaptive_threshold > 0:
        upward = sum_pos > adaptive_threshold
        downward = sum_neg > adaptive_threshold
        change_detected = upward or downward

        if upward:
            change_type = "increase"
            triggering_sum = sum_pos
        elif downward:
            change_type = "decrease"
            triggering_sum = sum_neg
        else:
            triggering_sum = max(sum_pos, sum_neg)

    change_info = {
        "cumulative_sum_pos": sum_pos,
        "cumulative_sum_neg": sum_neg,
        "triggering_sum": triggering_sum,
        "change_type": change_type,
        "reference_mean": ref_mean,
        "frequency": freq,
        "window_size": len(frequencies),
        "threshold": adaptive_threshold,
        "adaptive_delta": adaptive_delta,
        "rolling_std": rolling_std,
    }

    if change_detected:
        if verbose:
            CONSOLE.print(
                f"[bold cyan][STPH] CHANGE DETECTED! {ref_mean:.3f}Hz → {freq:.3f}Hz ({change_type})[/]"
            )
        # Reset state on change
        new_state = {
            "cumulative_sum_pos": 0.0,
            "cumulative_sum_neg": 0.0,
            "reference_mean": freq,
            "sum_of_samples": freq,
            "sample_count": 1,
            "frequencies": [freq],
            "timestamps": [timestamp],
            "initialized": True,
        }
    else:
        new_state = {
            "cumulative_sum_pos": sum_pos,
            "cumulative_sum_neg": sum_neg,
            "reference_mean": ref_mean,
            "sum_of_samples": sum_samples,
            "sample_count": sample_count,
            "frequencies": frequencies,
            "timestamps": timestamps,
            "initialized": True,
        }

    return change_detected, triggering_sum, change_info, new_state


def detect_pattern_change_pagehinkley(
    args: Namespace,
    shared_resources: SharedResources,
    prediction: Prediction,
    counter: int,
) -> tuple[bool, str | None, float, float | None, float | None]:
    """Functional wrapper for Page-Hinkley detection."""
    dominant_freq = get_dominant(prediction)
    current_time = prediction.t_end

    if dominant_freq is None or np.isnan(dominant_freq):
        return False, None, prediction.t_start, None, None

    # Restore state
    state = shared_resources.online_detection.get("state", {})
    if "frequencies" not in state:
        state["frequencies"] = get_frequencies(list(shared_resources.data))
        state["timestamps"] = get_timestamps(list(shared_resources.data))

    change_detected, triggering_sum, metadata, new_state = pagehinkley_step(
        dominant_freq, current_time, state, window_size=50, verbose=args.verbose
    )

    log_message = None
    reference_mean = None
    frequency = None

    if change_detected:
        change_type = metadata.get("change_type", "unknown")
        frequency = metadata.get("frequency", dominant_freq)
        reference_mean = metadata.get("reference_mean", 0.0)
        window_size = metadata.get("window_size", 0)

        magnitude = abs(frequency - reference_mean)
        percent_change = (magnitude / reference_mean * 100) if reference_mean > 0 else 0

        direction_arrow = (
            "increasing"
            if change_type == "increase"
            else "decreasing" if change_type == "decrease" else "stable"
        )
        log_message = (
            f"[bold magenta][Page-Hinkley] PAGE-HINKLEY CHANGE DETECTED! {direction_arrow} "
            f"{reference_mean:.1f}Hz → {frequency:.1f}Hz "
            f"(Δ={magnitude:.1f}Hz, {percent_change:.1f}% {change_type}) "
            f"at sample {window_size}, time={current_time:.3f}s[/]\n"
            f"[magenta][Page-Hinkley] Page-Hinkley stats: sum_pos={metadata.get('cumulative_sum_pos', 0):.2f}, "
            f"sum_neg={metadata.get('cumulative_sum_neg', 0):.2f}, threshold={metadata.get('threshold', 0):.3f}[/]\n"
            f"[magenta][Page-Hinkley] Cumulative sum exceeded threshold -> Starting fresh analysis[/]"
        )

        adaptive_start_time = current_time
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
                        "Page-Hinkley Change Point Detected",
                        {
                            "algorithm": "PageHinkley",
                            "frequency": frequency,
                            "reference_mean": reference_mean,
                            "magnitude": magnitude,
                            "percent_change": percent_change,
                            "triggering_sum": triggering_sum,
                            "change_type": change_type,
                            "position": window_size,
                            "timestamp": current_time,
                            "threshold": metadata.get("threshold", 0),
                            "delta": metadata.get("adaptive_delta", 0),
                            "prediction_counter": counter,
                        },
                    )
            except ImportError:
                pass
    else:
        adaptive_start_time = prediction.t_start
        reference_mean = new_state.get("reference_mean")
        frequency = dominant_freq

    # Save state
    shared_resources.online_detection["state"] = new_state

    return change_detected, log_message, adaptive_start_time, reference_mean, frequency
