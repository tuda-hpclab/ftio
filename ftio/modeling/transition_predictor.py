"""
TransitionPredictor: predicts when the next phase transition will occur.

Frequency prediction (DFT/wavelet output) and transition prediction are
explicitly separate:
  - Frequency prediction answers: what is the dominant period *right now*?
  - Transition prediction answers: *when* will it change, and to *what*?

The forecast includes uncertainty bounds derived from the dwell-time
distributions in the reference automaton.  On the first run (only one
profiling run in the library, std = 0) timing bounds cannot be computed;
the next-period prediction is still reported.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftio.modeling.automaton_profile import AutomatonProfile
from ftio.modeling.state_tracker import StateTracker


@dataclass
class TransitionForecast:
    """Lookahead prediction for the next phase transition."""

    current_state_idx: int
    n_states: int
    position: float  # 0.0 (first state) → 1.0 (last state)
    eta_seconds: float  # expected time until next transition (nan = unknown or end)
    eta_lower: float  # lower bound −1σ
    eta_upper: float  # upper bound +1σ
    next_period: float  # expected period of the next state (nan = end)
    tracking_quality: float  # 0.0–1.0: how close current period is to reference mean

    @property
    def at_end(self) -> bool:
        """True when the tracker is in the last reference state."""
        return self.current_state_idx >= self.n_states - 1

    @property
    def has_timing(self) -> bool:
        """True when ETA bounds are available (≥2 runs in library)."""
        return not np.isnan(self.eta_seconds)


class TransitionPredictor:
    """
    Computes a TransitionForecast from a reference automaton and a live tracker.

    Usage:
        predictor = TransitionPredictor(reference, tracker)
        forecast = predictor.predict(current_freq, current_ranks)
    """

    def __init__(self, reference: AutomatonProfile, tracker: StateTracker):
        self.reference = reference
        self.tracker = tracker

    def predict(self, current_freq: float, current_ranks: int = 0) -> TransitionForecast:
        """Compute a forecast based on the tracker's current position."""
        idx = self.tracker.current_state_index
        n = self.reference.n_states
        elapsed = self.tracker.elapsed_in_state

        # Tracking quality: relative closeness to the reference mean period
        ref_period = self.reference.state_stats[idx].period_mean
        if current_freq > 0 and not np.isnan(ref_period) and ref_period > 0:
            obs_period = 1.0 / current_freq
            quality = max(0.0, 1.0 - abs(obs_period - ref_period) / ref_period)
        else:
            quality = 0.0

        if idx >= n - 1:
            # Final state — no further transition expected
            return TransitionForecast(
                current_state_idx=idx,
                n_states=n,
                position=self.tracker.position,
                eta_seconds=np.nan,
                eta_lower=np.nan,
                eta_upper=np.nan,
                next_period=np.nan,
                tracking_quality=quality,
            )

        s = self.reference.state_stats[idx]

        if np.isnan(s.dwell_mean):
            eta = eta_lo = eta_hi = np.nan
        else:
            remaining = s.dwell_mean - elapsed
            eta = max(0.0, remaining)
            eta_lo = max(0.0, remaining - s.dwell_std)
            eta_hi = max(0.0, remaining + s.dwell_std)

        next_period = self.reference.state_stats[idx + 1].period_mean

        return TransitionForecast(
            current_state_idx=idx,
            n_states=n,
            position=self.tracker.position,
            eta_seconds=eta,
            eta_lower=eta_lo,
            eta_upper=eta_hi,
            next_period=next_period,
            tracking_quality=quality,
        )
