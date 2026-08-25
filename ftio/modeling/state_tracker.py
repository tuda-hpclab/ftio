"""
StateTracker: tracks position in a reference automaton from live observations.

Three matching strategies, all forward-only (never move backward):
  greedy  — nearest period + rank penalty at each step; O(n) per update
  dtw     — align an observation window against reference suffixes; O(n²) per update
  viterbi — HMM forward pass with Gaussian emission; O(n) per update

Rank count is used as a secondary matching signal in all three strategies,
making the tracker robust for malleable applications.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from ftio.modeling.automaton_profile import AutomatonProfile


class MatchStrategy(StrEnum):
    GREEDY = "greedy"
    DTW = "dtw"
    VITERBI = "viterbi"


class StateTracker:
    """
    Tracks which reference state the live run is currently in.

    All strategies enforce monotone forward progression: once the tracker
    advances to state i it never returns to an earlier state, matching the
    physical reality that the application moves through phases in order.

    Parameters
    ----------
    reference : AutomatonProfile
        Loaded reference to match against.
    strategy : MatchStrategy
        greedy (default), dtw, or viterbi.
    rank_mismatch_weight : float
        Relative penalty added to the distance when observed ranks differ
        from the reference state's ranks.  0 disables rank-aware matching.
    """

    def __init__(
        self,
        reference: AutomatonProfile,
        strategy: MatchStrategy = MatchStrategy.GREEDY,
        rank_mismatch_weight: float = 0.3,
    ):
        self.reference = reference
        self.strategy = MatchStrategy(strategy)
        self.rank_mismatch_weight = rank_mismatch_weight

        self._current_idx: int = 0
        self._obs_window: list[float] = []
        self._state_entry_time: float | None = None
        self._last_timestamp: float = 0.0

        # Viterbi: log-probability vector, one entry per reference state.
        # Initialised to 0 for state 0, -inf for all others.
        n = reference.n_states
        self._viterbi_log_probs: list[float] = [-np.inf] * n
        if n > 0:
            self._viterbi_log_probs[0] = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, freq: float, timestamp: float, ranks: int = 0) -> int:
        """Feed one observation; returns the current estimated state index."""
        if freq <= 0 or np.isnan(freq):
            return self._current_idx

        period = 1.0 / freq
        self._obs_window.append(period)
        self._last_timestamp = timestamp

        if self._state_entry_time is None:
            self._state_entry_time = timestamp

        prev_idx = self._current_idx

        if self.strategy == MatchStrategy.GREEDY:
            new_idx = self._match_greedy(period, ranks)
        elif self.strategy == MatchStrategy.DTW:
            new_idx = self._match_dtw()
        else:
            new_idx = self._match_viterbi(period, ranks)

        if new_idx != prev_idx:
            self._current_idx = new_idx
            self._state_entry_time = timestamp

        return self._current_idx

    @property
    def current_state_index(self) -> int:
        return self._current_idx

    @property
    def position(self) -> float:
        """Normalized lifecycle position: 0.0 = first state, 1.0 = last state."""
        n = self.reference.n_states
        return self._current_idx / (n - 1) if n > 1 else 0.0

    @property
    def elapsed_in_state(self) -> float:
        """Seconds elapsed since entering the current state."""
        if self._state_entry_time is None:
            return 0.0
        return max(0.0, self._last_timestamp - self._state_entry_time)

    @property
    def is_final_state(self) -> bool:
        return self._current_idx >= self.reference.n_states - 1

    # ------------------------------------------------------------------
    # Matching strategies
    # ------------------------------------------------------------------

    def _dist(self, idx: int, period: float, ranks: int) -> float:
        """Combined relative-period + rank-mismatch distance for one reference state."""
        s = self.reference.state_stats[idx]
        if np.isnan(s.period_mean) or s.period_mean <= 0:
            return np.inf
        period_dist = abs(period - s.period_mean) / s.period_mean
        rank_penalty = (
            self.rank_mismatch_weight
            if (s.ranks > 0 and ranks > 0 and s.ranks != ranks)
            else 0.0
        )
        return period_dist + rank_penalty

    def _match_greedy(self, period: float, ranks: int) -> int:
        """Move to the forward state with minimum combined distance."""
        best_idx = self._current_idx
        best_dist = self._dist(self._current_idx, period, ranks)
        for i in range(self._current_idx + 1, self.reference.n_states):
            d = self._dist(i, period, ranks)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _match_dtw(self) -> int:
        """Align the observation window against forward suffixes of the reference."""
        ref_seq = self.reference.period_sequence
        n_ref = len(ref_seq)
        window = (
            self._obs_window[-n_ref:]
            if len(self._obs_window) > n_ref
            else list(self._obs_window)
        )
        if not window:
            return self._current_idx

        best_idx = self._current_idx
        best_cost = np.inf
        for start in range(self._current_idx, self.reference.n_states):
            sub = ref_seq[start : start + len(window)]
            if not sub:
                continue
            cost = _dtw_cost(window, sub)
            if cost < best_cost:
                best_cost = cost
                best_idx = start
        return best_idx

    def _match_viterbi(self, period: float, ranks: int) -> int:
        """HMM forward pass: Gaussian emission on period, left-to-right transitions."""
        n = self.reference.n_states
        prev = self._viterbi_log_probs
        new_log_p = [-np.inf] * n

        for j in range(n):
            s = self.reference.state_stats[j]
            if np.isnan(s.period_mean) or s.period_mean <= 0:
                emit = -1e9
            else:
                sigma = max(
                    s.period_std,
                    0.1 * s.period_mean,
                    1e-6,
                )
                diff = (period - s.period_mean) ** 2 / (2 * sigma**2)
                rank_penalty = (
                    self.rank_mismatch_weight
                    if (s.ranks > 0 and ranks > 0 and s.ranks != ranks)
                    else 0.0
                )
                emit = -diff - rank_penalty

            # Left-to-right: stay or advance by one, never go back
            stay = prev[j] - 0.1  # small cost for staying
            advance = prev[j - 1] if j > 0 else -np.inf
            new_log_p[j] = emit + max(stay, advance)

        self._viterbi_log_probs = new_log_p

        # Forward-only: only consider current index onward
        best = self._current_idx
        best_val = new_log_p[self._current_idx]
        for j in range(self._current_idx + 1, n):
            if new_log_p[j] > best_val:
                best_val = new_log_p[j]
                best = j
        return best


# ------------------------------------------------------------------
# DTW helper (no external dependency)
# ------------------------------------------------------------------


def _dtw_cost(a: list[float], b: list[float]) -> float:
    """O(nm) DTW distance between two period sequences."""
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])
