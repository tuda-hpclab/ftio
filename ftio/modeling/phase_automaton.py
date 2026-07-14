"""
Phase Automaton: models I/O behaviour as a hybrid automaton / state machine.

Each state represents a stable I/O regime characterised by a dominant
frequency (and the derived period). Transitions are triggered by one or
more of the following mechanisms — listed from simplest to most complex:

  1. Rank change      — prediction.ranks differs from the current state.
                        Always an explicit phase boundary (new job config).
  2. Period-ratio     — new_period / current_period (or its reciprocal)
                        exceeds a threshold (e.g. 1.5).  No calibration
                        needed; robust for the I/O domain.
  3. Statistical      — one of the following detectors accumulates the
                        frequency sequence and fires when the distribution
                        shifts significantly:

        cusum   — AV-CUSUM (adaptive-variance CUSUM).  Accumulates signed
                  deviations from a rolling reference; sensitive to
                  sustained drift in either direction.  Good general-purpose
                  choice.

        ph      — Page-Hinkley.  Similar to CUSUM but uses a fixed drift
                  parameter; faster on monotone shifts.

        adwin   — ADWIN (Hoeffding-bound windowing).  Non-parametric;
                  needs many samples per phase (200+) or a very large
                  frequency ratio (>10×) for few samples.

        ksigma  — State-adaptive k-sigma.  Computes the mean (μ) and std
                  (σ) of all frequencies since the last change, then fires
                  when |freq_new − μ| > k · σ_eff, where σ_eff is floored
                  at sigma_rel_floor · μ to prevent over-sensitivity on
                  very stable signals.  Self-calibrating: noisy phases
                  automatically require larger shifts to trigger.
                  Recommended when within-phase frequency fluctuations
                  are expected (e.g. FTIO output over short windows).

Any combination of triggers can be active simultaneously.  Rank changes are
checked first; if that fires, the statistical detector state is reset.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ftio.freq.prediction import Prediction


@dataclass
class PhaseState:
    """A single stable I/O phase in the automaton."""

    state_id: int
    dominant_freq: float  # Hz
    confidence: float
    entry_time: float
    ranks: int = 0
    exit_time: float = np.nan
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def period(self) -> float:
        return 1.0 / self.dominant_freq if self.dominant_freq > 0 else np.inf

    @property
    def n_phases(self) -> int:
        return len(self.predictions)

    @property
    def duration(self) -> float:
        end = (
            self.exit_time
            if not np.isnan(self.exit_time)
            else (self.predictions[-1].t_end if self.predictions else np.nan)
        )
        return end - self.entry_time if not np.isnan(end) else np.nan

    def __repr__(self) -> str:
        return (
            f"State({self.state_id}: freq={self.dominant_freq:.4f} Hz, "
            f"period={self.period:.2f} s, ranks={self.ranks}, "
            f"n_phases={self.n_phases}, duration={self.duration:.1f} s)"
        )


@dataclass
class Transition:
    """A fired edge in the automaton."""

    from_state: int
    to_state: int
    timestamp: float
    prediction_index: int
    old_freq: float
    new_freq: float
    cause: str = "frequency"  # "frequency" | "rank_change" | "period_ratio"

    def __repr__(self) -> str:
        return (
            f"Transition({self.from_state}→{self.to_state} at t={self.timestamp:.2f} s, "
            f"{self.old_freq:.4f}→{self.new_freq:.4f} Hz, "
            f"cause={self.cause!r}, pred #{self.prediction_index})"
        )


class PhaseAutomaton:
    """
    Finite-state machine that models I/O phases from FTIO Predictions.

    Transition triggers (all optional, combinable):

      rank_changes_trigger : bool  (default True)
          A new prediction with a different rank count immediately opens a
          new state.  This is the most explicit trigger — a rank change is
          always a configuration boundary.

      period_ratio_threshold : float | None  (default None)
          Fire when max(new_period/cur_period, cur_period/new_period) >
          threshold.  Recommended value: 1.5 (50% change in period).
          Simpler and more interpretable than statistical detectors; no
          warm-up samples needed.

      method : str | None  (default "cusum")
          Statistical change detector: 'cusum', 'ph', 'adwin', or 'ksigma'.
          Set to None to disable statistical detection entirely and rely
          only on rank changes and/or the period-ratio trigger.

      min_cycles : float  (default 1.0 = off)
          Warm-up guard: ignore a prediction whose analysis window holds fewer
          than this many full periods.  A period cannot be measured from a
          single phase, and early in a run the DFT reports one burst's *width*
          as the "period" (see step()).  Set to 2.0 when the Prediction's
          t_start/t_end span the whole observed run, as they do in the online
          predictor.  Left off by default because a caller may hand in
          Predictions whose window is a short slice rather than the full span.

    Usage (online):
        aut = PhaseAutomaton(method="cusum", rank_changes_trigger=True)
        for pred in stream:
            aut.step(pred)
        aut.print_summary()

    Usage (offline):
        aut = PhaseAutomaton(period_ratio_threshold=1.5, method=None)
        aut.build(predictions)
        aut.plot()
    """

    def __init__(
        self,
        method: str | None = "cusum",
        rank_changes_trigger: bool = True,
        period_ratio_threshold: float | None = None,
        min_cycles: float = 1.0,
    ):
        if method is not None and method not in ("cusum", "ph", "adwin", "ksigma"):
            raise ValueError(
                f"method must be 'cusum', 'ph', 'adwin', 'ksigma', or None, got {method!r}"
            )
        self.method = method
        self.rank_changes_trigger = rank_changes_trigger
        self.period_ratio_threshold = period_ratio_threshold
        self.min_cycles = min_cycles
        self.states: list[PhaseState] = []
        self.transitions: list[Transition] = []
        self._detector_state: dict[str, Any] = {}
        self._current_state: PhaseState | None = None
        self._pred_index: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, prediction: Prediction) -> bool:
        """
        Feed one Prediction; returns True if a transition fired.
        """
        if prediction.is_empty():
            return False
        freq, conf = prediction.get_dominant_freq_and_conf()
        if freq <= 0 or np.isnan(freq):
            return False

        # --- Warm-up guard: a period needs at least two periods to be seen ----
        # A DFT cannot resolve a period longer than the window it ran on, and its
        # analysis window is trimmed to the data seen so far. Early in a run the
        # window therefore holds a single I/O phase, and the transform reports that
        # one burst's *width* as the "period" -- the k=1 bin, one cycle across the
        # whole window. It is not evidence of periodicity, and it is reported with
        # full confidence and stays stable for several rounds, so neither a
        # confidence nor a stability gate can filter it out.
        #
        # Measured on a real LAMMPS run (true period 41.8 s):
        #     window  period  cycles
        #       5.0 s   5.0 s   1.01   <- one burst; the "period" is its width
        #      44.5 s  44.4 s   1.00   <- right answer, still only one cycle
        #      84.8 s  42.4 s   2.00   <- two cycles: now it is measurable
        #
        # Requiring two full cycles in the window costs one extra period before
        # modelling starts, and in exchange the automaton never learns a phase that
        # does not exist. Set min_cycles <= 1 to disable.
        if self.min_cycles > 1:
            window = prediction.t_end - prediction.t_start
            if window > 0 and window * freq < self.min_cycles:
                return False

        ranks = max(0, int(prediction.ranks))
        cause: str | None = None

        # --- Rank-change check (highest priority) --------------------
        if (
            self.rank_changes_trigger
            and self._current_state is not None
            and ranks > 0
            and ranks != self._current_state.ranks
        ):
            cause = "rank_change"
            self._detector_state = {}  # reset statistical detector

        # --- Period-ratio check -------------------------------------
        if (
            cause is None
            and self.period_ratio_threshold is not None
            and self._current_state is not None
        ):
            cur_freq = self._current_state.dominant_freq
            if cur_freq > 0:
                ratio = max(freq / cur_freq, cur_freq / freq)
                if ratio > self.period_ratio_threshold:
                    cause = "period_ratio"

        # --- Statistical detector -----------------------------------
        if cause is None and self.method is not None:
            detected, self._detector_state = self._detect(freq, prediction.t_end)
            if detected:
                cause = "frequency"

        # --- Bootstrap first state ----------------------------------
        if self._current_state is None:
            self._current_state = self._open_state(freq, conf, prediction.t_start, ranks)

        self._current_state.predictions.append(prediction)
        self._pred_index += 1

        # --- Fire transition ----------------------------------------
        if cause is not None:
            self._current_state.exit_time = prediction.t_end
            old = self._current_state
            self._current_state = self._open_state(freq, conf, prediction.t_end, ranks)
            self.transitions.append(
                Transition(
                    from_state=old.state_id,
                    to_state=self._current_state.state_id,
                    timestamp=prediction.t_end,
                    prediction_index=self._pred_index,
                    old_freq=old.dominant_freq,
                    new_freq=freq,
                    cause=cause,
                )
            )
            return True
        return False

    def build(self, predictions: list[Prediction]) -> None:
        """Build automaton offline from a complete list of predictions."""
        for pred in predictions:
            self.step(pred)

    def print_summary(self) -> None:
        print(f"\n{'='*65}")
        print(
            f"PhaseAutomaton  method={self.method!r}  "
            f"rank_sensitive={self.rank_changes_trigger}  "
            f"period_ratio={self.period_ratio_threshold}  "
            f"states={len(self.states)}  transitions={len(self.transitions)}"
        )
        print("─" * 65)
        for s in self.states:
            print(f"  {s}")
        for t in self.transitions:
            print(f"  {t}")
        print("=" * 65)

    def print_graph(self) -> None:
        """Print the automaton as a vertical ASCII state-graph diagram."""
        if not self.states:
            print("  (no states)")
            return

        trans_map = {t.from_state: t for t in self.transitions}

        INNER = 30
        INDENT = "  "
        MID = (INNER + 2) // 2

        def _box(state: PhaseState) -> list[str]:
            dur = state.duration
            dur_str = f"{dur:.1f} s" if not np.isnan(dur) else "ongoing"
            rows = [
                f"S{state.state_id}",
                f"f = {state.dominant_freq:.4f} Hz",
                f"T = {state.period:.2f} s",
                f"ranks = {state.ranks}",
                f"dur   = {dur_str}",
            ]
            top = "┌" + "─" * INNER + "┐"
            bot = "└" + "─" * INNER + "┘"
            body = [f"│ {row:<{INNER - 2}} │" for row in rows]
            return [top] + body + [bot]

        _cause_label = {
            "rank_change": "rank change",
            "period_ratio": "period ratio",
            "frequency": "freq shift (statistical)",
        }

        print(f"\n{'=' * 65}")
        print(
            f"PhaseAutomaton graph  method={self.method!r}  "
            f"states={len(self.states)}  transitions={len(self.transitions)}"
        )
        print("─" * 65)

        for state in self.states:
            for line in _box(state):
                print(INDENT + line)

            tr = trans_map.get(state.state_id)
            if tr is None:
                continue

            pad = " " * MID
            old_p = 1.0 / tr.old_freq if tr.old_freq > 0 else float("inf")
            new_p = 1.0 / tr.new_freq if tr.new_freq > 0 else float("inf")
            cause_str = _cause_label.get(tr.cause, tr.cause)

            print(INDENT + pad + "│")
            print(INDENT + pad + f"├─ {cause_str}  @t={tr.timestamp:.1f} s")
            print(INDENT + pad + f"│  T: {old_p:.2f} s → {new_p:.2f} s")
            print(INDENT + pad + "▼")

        print("=" * 65)

    def to_dict(self) -> dict:
        """Serialise the automaton to a plain, JSON-compatible dict.

        The output contains the full configuration, every state (without the
        raw Prediction objects — only summary statistics), and every
        transition.  NaN / inf values are replaced with ``null`` so the
        result can be passed directly to ``json.dump``.

        Returns:
            dict with keys ``"method"``, ``"rank_changes_trigger"``,
            ``"period_ratio_threshold"``, ``"n_states"``,
            ``"n_transitions"``, ``"states"``, and ``"transitions"``.
        """

        def _float(v):
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            return float(v)

        return {
            "method": self.method,
            "rank_changes_trigger": self.rank_changes_trigger,
            "period_ratio_threshold": self.period_ratio_threshold,
            "n_states": len(self.states),
            "n_transitions": len(self.transitions),
            "states": [
                {
                    "state_id": s.state_id,
                    "dominant_freq": _float(s.dominant_freq),
                    "period": _float(s.period),
                    "confidence": _float(s.confidence),
                    "entry_time": _float(s.entry_time),
                    "exit_time": _float(s.exit_time),
                    "duration": _float(s.duration),
                    "ranks": s.ranks,
                    "n_predictions": s.n_phases,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "timestamp": _float(t.timestamp),
                    "prediction_index": t.prediction_index,
                    "old_freq": _float(t.old_freq),
                    "old_period": _float(1.0 / t.old_freq if t.old_freq > 0 else None),
                    "new_freq": _float(t.new_freq),
                    "new_period": _float(1.0 / t.new_freq if t.new_freq > 0 else None),
                    "cause": t.cause,
                }
                for t in self.transitions
            ],
        }

    def save_json(self, path: str = "./phase_automaton.json") -> None:
        """Export the automaton state to a JSON file.

        Args:
            path: Destination file path (default: ``./phase_automaton.json``).

        The file is human-readable (indented) and can be reloaded with the
        standard ``json`` module.  NaN / inf values are written as ``null``.
        """
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(
            f"[PhaseAutomaton] Saved to {path}  ({len(self.states)} states, {len(self.transitions)} transitions)"
        )

    def plot(self, title: str = "Phase Automaton", show: bool = True):
        """
        Plot the automaton timeline:
          - Colored bands for each state (period on y-axis)
          - Frequency sequence as scatter dots
          - Transitions as vertical dashed lines
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Collect all (t_end, freq) from predictions
        t_pts, f_pts, s_ids = [], [], []
        for state in self.states:
            for pred in state.predictions:
                t_pts.append(pred.t_end)
                f_pts.append(pred.dominant_freq[0] if len(pred.dominant_freq) else np.nan)
                s_ids.append(state.state_id)

        # State bands
        for state in self.states:
            col = colors[state.state_id % len(colors)]
            t0 = state.entry_time
            t1 = (
                state.exit_time
                if not np.isnan(state.exit_time)
                else (state.predictions[-1].t_end if state.predictions else t0)
            )
            ax.axvspan(t0, t1, alpha=0.15, color=col)
            mid = (t0 + t1) / 2
            ax.text(
                mid,
                ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"S{state.state_id}\n{state.period:.2f} s\n{state.ranks} ranks",
                ha="center",
                va="top",
                fontsize=8,
                color=col,
                transform=ax.get_xaxis_transform(),
            )

        # Prediction dots
        scatter_colors = [colors[sid % len(colors)] for sid in s_ids]
        ax.scatter(t_pts, f_pts, c=scatter_colors, zorder=3, s=60, label="predictions")

        # Transitions
        for tr in self.transitions:
            col = (
                "red"
                if tr.cause == "rank_change"
                else ("orange" if tr.cause == "period_ratio" else "black")
            )
            ax.axvline(
                tr.timestamp,
                color=col,
                linestyle="--",
                linewidth=1.2,
                label=f"transition ({tr.cause})",
            )

        ax.set_xlabel("time (s)")
        ax.set_ylabel("dominant frequency (Hz)")
        ax.set_title(title)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, label in zip(handles, labels, strict=False):
            seen.setdefault(label, h)
        ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_state(
        self, freq: float, conf: float, entry_time: float, ranks: int = 0
    ) -> PhaseState:
        state = PhaseState(
            state_id=len(self.states),
            dominant_freq=freq,
            confidence=conf,
            entry_time=entry_time,
            ranks=ranks,
        )
        self.states.append(state)
        return state

    def _detect(self, freq: float, timestamp: float) -> tuple[bool, dict]:
        s = self._detector_state
        if self.method == "cusum":
            from ftio.prediction.change_detection.cusum import cusum_step

            detected, _info, new_s = cusum_step(freq, timestamp, s)
        elif self.method == "ph":
            from ftio.prediction.change_detection.pagehinkley import pagehinkley_step

            detected, _trig, _info, new_s = pagehinkley_step(freq, timestamp, s)
        elif self.method == "adwin":
            from ftio.prediction.change_detection.adwin import adwin_step

            change_idx, _t, new_s = adwin_step(freq, timestamp, s)
            detected = change_idx is not None
        else:  # ksigma
            from ftio.prediction.change_detection.ksigma import ksigma_step

            detected, _info, new_s = ksigma_step(freq, timestamp, s)
        return bool(detected), new_s
