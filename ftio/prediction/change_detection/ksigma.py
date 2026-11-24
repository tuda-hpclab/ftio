"""
State-adaptive k-sigma change-point detector for FTIO online predictor.

Principle
---------
The detector maintains a running mean (μ) and standard deviation (σ) of
all dominant frequencies observed *since the last detected change*.  A new
observation is flagged as a change point when it lies more than k standard
deviations away from the current phase mean:

    |freq_new − μ| > k · σ_eff

where σ_eff = max(σ, sigma_rel_floor · μ) is a noise floor that prevents
over-sensitivity when the signal is very stable (σ → 0).

Why this is robust to small within-phase variations
----------------------------------------------------
Because σ is computed from the *actual* spread of the current phase, the
threshold adapts automatically:

• Noisy phase  (σ large)  → wide threshold → requires a large shift to fire.
• Stable phase (σ small)  → σ_eff falls back to sigma_rel_floor · μ, giving
  a minimum relative threshold of k · sigma_rel_floor (default 6 % of μ).

This self-calibration is the key advantage over fixed-threshold detectors:
a 5 % fluctuation inside a noisy phase will never trigger a transition,
while the same 5 % shift from a perfectly stable phase may — because the
relative noise floor captures genuine measurement uncertainty.

Comparison with CUSUM / Page-Hinkley
-------------------------------------
• CUSUM/PH accumulate deviations over time → sensitive to slow drift but can
  fire on sustained noise.
• k-sigma evaluates each new observation against the full phase history →
  robust to transient spikes but blind to slow, monotone drift.

The two approaches are complementary; combining them (or using period-ratio
as a faster outer trigger and k-sigma as the statistical inner guard) gives
the best coverage.

Parameters
----------
k : float
    Number of sigma deviations required to fire a transition.
    Smaller values increase sensitivity; typical range 2–4.
    Default: 3.0 (classic 3-sigma rule).

min_samples : int
    Minimum number of observations that must be accumulated before the
    detector can fire.  This prevents false positives during the warm-up
    period when σ is unreliable.
    Default: 4.

sigma_rel_floor : float
    Minimum effective sigma, expressed as a fraction of the current mean.
    Prevents over-sensitivity when the signal is very stable.
    Default: 0.02 (2 % of μ).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ksigma_step(
    freq: float,
    timestamp: float,
    state: dict[str, Any],
    k: float = 3.0,
    min_samples: int = 4,
    sigma_rel_floor: float = 0.02,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """
    Perform one step of the state-adaptive k-sigma change-point detector.

    The detector compares each new observation against the distribution
    of all frequencies seen since the last change point.  A transition
    fires when the new value lies more than k effective-sigma away from
    the current phase mean.

    The effective sigma is floored at ``sigma_rel_floor * mu`` so that an
    extremely stable signal (σ → 0) does not fire on tiny measurement noise.
    Observations that are consistent with the current phase are added to
    the running history; outliers that trigger a transition are discarded
    from the old history and start a fresh one.

    Args:
        freq: Current dominant frequency (Hz).  Must be > 0 and not NaN.
        timestamp: Current prediction end-time (seconds).
        state: Mutable detector state dict (pass ``{}`` on the first call).
        k: Sigma multiplier for the detection threshold (default 3.0).
        min_samples: Minimum phase observations before the detector can
            fire; guards against warm-up false positives (default 4).
        sigma_rel_floor: Noise floor as a fraction of the phase mean;
            prevents over-sensitivity on very stable signals (default 0.02).

    Returns:
        A 3-tuple ``(change_detected, change_info, new_state)`` where:

        * ``change_detected`` – True if a phase transition was detected.
        * ``change_info`` – dict with diagnostic fields:
            - ``"timestamp"``   – current timestamp
            - ``"frequency"``   – current frequency
            - ``"mu"``          – phase mean before this observation
            - ``"sigma"``       – phase std before this observation
            - ``"sigma_eff"``   – effective sigma (after applying floor)
            - ``"z_score"``     – standardised distance of freq from mu
            - ``"k"``           – configured threshold multiplier
            - ``"n"``           – number of in-phase observations so far
        * ``new_state`` – updated state dict to pass on the next call.

    Examples
    --------
    Stable phase — no transition expected::

        state = {}
        for f in [0.20, 0.19, 0.21, 0.20, 0.20]:
            detected, info, state = ksigma_step(f, t, state)
        # detected is False throughout

    Genuine phase shift — transition expected::

        for f in [0.20, 0.19, 0.21, 0.20, 0.30]:  # last value is outlier
            detected, info, state = ksigma_step(f, t, state)
        # detected is True on the last observation
    """
    if np.isnan(freq) or freq <= 0:
        return False, {"timestamp": timestamp, "frequency": freq}, {}

    s = state.copy()
    freqs: list[float] = s.get("freqs", [])
    timestamps: list[float] = s.get("timestamps", [])
    n = len(freqs)

    # Warm-up: accumulate without testing
    if n < min_samples:
        freqs = freqs + [freq]
        timestamps = timestamps + [timestamp]
        mu = float(np.mean(freqs))
        new_state = {"freqs": freqs, "timestamps": timestamps}
        change_info = {
            "timestamp": timestamp,
            "frequency": freq,
            "mu": mu,
            "sigma": float(np.std(freqs)) if len(freqs) > 1 else 0.0,
            "sigma_eff": max(
                float(np.std(freqs)) if len(freqs) > 1 else 0.0, sigma_rel_floor * mu
            ),
            "z_score": 0.0,
            "k": k,
            "n": len(freqs),
        }
        return False, change_info, new_state

    # Compute phase statistics from previous observations (exclude new point)
    arr = np.asarray(freqs, dtype=float)
    mu = float(arr.mean())
    sigma = float(arr.std())
    sigma_eff = max(sigma, sigma_rel_floor * mu)
    z = abs(freq - mu) / sigma_eff

    change_info = {
        "timestamp": timestamp,
        "frequency": freq,
        "mu": mu,
        "sigma": sigma,
        "sigma_eff": sigma_eff,
        "z_score": z,
        "k": k,
        "n": n,
    }

    if z > k:
        # Start fresh history with just the new observation
        new_state = {"freqs": [freq], "timestamps": [timestamp]}
        return True, change_info, new_state

    # Consistent with current phase — append and continue
    new_state = {"freqs": freqs + [freq], "timestamps": timestamps + [timestamp]}
    return False, change_info, new_state
