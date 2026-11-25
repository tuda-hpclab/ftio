"""
Burst-width (duty-cycle) estimation for periodic I/O signals.

For each detected period the shortest contiguous time window that contains
`energy_fraction` of the period's total energy is found using an O(N)
two-pointer sweep.  No amplitude threshold is required.

Computational cost: O(N) where N = len(b_sampled).  The sweep runs once
per period (each of length T_samples), so the total work is proportional
to the length of the resampled signal — negligible compared with the DFT
or wavelet steps that precede it.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jun 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np


def _min_contiguous_window(power: np.ndarray, fraction: float) -> tuple[int, int]:
    """Return (width, left_index) of the shortest contiguous window whose
    energy sum >= fraction * total_energy.  O(N) two-pointer sweep."""
    target = fraction * float(np.sum(power))
    if target == 0.0:
        return 0, 0

    left = 0
    window_sum = 0.0
    min_width = len(power)
    min_left = 0

    for right in range(len(power)):
        window_sum += power[right]
        while window_sum >= target:
            w = right - left + 1
            if w < min_width:
                min_width = w
                min_left = left
            window_sum -= power[left]
            left += 1

    return min_width, min_left


def estimate_burst_widths(
    b_sampled: np.ndarray,
    prediction,
    energy_fraction: float = 0.95,
) -> np.ndarray:
    """Estimate per-period burst width in seconds.

    Slices *b_sampled* into complete periods of length T = 1/f_dom and finds
    the shortest contiguous interval within each period that carries
    *energy_fraction* of that period's total power.

    Args:
        b_sampled:       Uniformly resampled bandwidth signal.
        prediction:      Prediction object with freq and dominant_freq set.
        energy_fraction: Fraction of period energy the window must contain
                         (default 0.95).

    Returns:
        1-D array of burst widths in seconds, one entry per complete period.
        Empty array if the dominant frequency is invalid or the signal is too
        short.
    """
    f_dom = prediction.get_dominant_freq()
    fs = prediction.freq

    if np.isnan(f_dom) or f_dom <= 0 or fs <= 0:
        return np.array([])

    T_samples = int(round(fs / f_dom))
    if T_samples < 2:
        return np.array([])

    n_complete = len(b_sampled) // T_samples
    if n_complete == 0:
        return np.array([])

    burst_widths = np.empty(n_complete)
    burst_t_starts = np.empty(n_complete)
    t0 = prediction.t_start

    for k in range(n_complete):
        start = k * T_samples
        segment = b_sampled[start : start + T_samples]
        power = segment**2
        width_samples, left_samples = _min_contiguous_window(power, energy_fraction)
        burst_widths[k] = width_samples / fs
        burst_t_starts[k] = t0 + (start + left_samples) / fs

    prediction.burst_t_starts = burst_t_starts
    return burst_widths


def overlay_burst_widths_matplotlib(prediction, alpha: float = 0.18) -> None:
    """Shade estimated burst regions on the *currently active* matplotlib figure.

    Each period's burst is centred on the period's energy-weighted centroid
    (approximated as the centre of the detected contiguous window).  Call this
    immediately after creating the bandwidth figure so the correct axes are
    still active.

    Args:
        prediction: Prediction object with burst_widths, freq, t_start set.
        alpha:      Transparency of the shaded region (default 0.18).
    """
    import matplotlib.pyplot as plt  # lazy import — plotting is optional

    bws = prediction.burst_widths
    if len(bws) == 0:
        return

    f_dom = prediction.get_dominant_freq()
    if np.isnan(f_dom) or f_dom <= 0:
        return

    T = 1.0 / f_dom
    t0 = prediction.t_start

    for k, bw in enumerate(bws):
        period_center = t0 + (k + 0.5) * T
        shade_start = period_center - bw / 2
        shade_end = period_center + bw / 2
        plt.axvspan(
            shade_start,
            shade_end,
            alpha=alpha,
            color="salmon",
            label="Burst width" if k == 0 else "_nolegend_",
        )
