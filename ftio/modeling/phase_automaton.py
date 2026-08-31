"""
Phase Automaton: models I/O behaviour as a hybrid automaton / state machine.

Each state represents a stable I/O regime characterised by a dominant
frequency (and the derived period). Transitions are triggered by one or
more of the following mechanisms — listed from simplest to most complex:

  1. Rank change      — prediction.ranks differs from the current state's
                        ranks.  Fires immediately (waiting for confirmation
                        would mean a real change right before the trace
                        ends never gets confirmed and is silently dropped).
                        The state it replaces stays reachable for
                        `rank_confirm_windows - 1` further predictions: if
                        the count reverts to that state's ranks within that
                        grace period, the transition is undone rather than
                        left standing -- online, `ranks` reflects however
                        many ZMQ messages happened to arrive in that poll
                        cycle, so a lagging straggler rank can make one
                        window look like a rank change when nothing actually
                        changed, and this is what catches that case, without
                        delaying detection of real changes to do it.
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
          A prediction with a different rank count opens a new state
          immediately -- a rank change is always a configuration boundary.
          The replaced state stays reachable for `rank_confirm_windows - 1`
          further predictions; reverting to its rank count within that
          window undoes the transition (see `rank_confirm_windows`).

      rank_confirm_windows : int  (default 2)
          Size of the grace period (in predictions) during which a rank
          change can still be retracted if it reverts. 1 means no grace
          period -- a rank change is permanent the instant it fires; higher
          values are more robust to a single window where not all ranks'
          ZMQ messages had arrived yet, at the cost of briefly tolerating a
          wrong reading before it's corrected. Ignored when
          rank_changes_trigger is False.

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
        rank_confirm_windows: int = 2,
        rank_change_strategy: str = "retract",
        period_ratio_threshold: float | None = None,
        min_cycles: float = 1.0,
    ):
        if method is not None and method not in ("cusum", "ph", "adwin", "ksigma"):
            raise ValueError(
                f"method must be 'cusum', 'ph', 'adwin', 'ksigma', or None, got {method!r}"
            )
        if rank_change_strategy not in ("retract", "confirm"):
            raise ValueError(
                f"rank_change_strategy must be 'retract' or 'confirm', got {rank_change_strategy!r}"
            )
        self.method = method
        self.rank_changes_trigger = rank_changes_trigger
        self.rank_confirm_windows = max(1, int(rank_confirm_windows))
        self.rank_change_strategy = rank_change_strategy
        self.period_ratio_threshold = period_ratio_threshold
        self.min_cycles = min_cycles
        self.states: list[PhaseState] = []
        self.transitions: list[Transition] = []
        self._detector_state: dict[str, Any] = {}
        self._current_state: PhaseState | None = None
        self._pred_index: int = 0
        # "confirm" strategy: wait for rank_confirm_windows consecutive
        # agreeing predictions before trusting a rank change (see
        # _rank_check_confirm).
        self._pending_rank: int | None = None
        self._pending_rank_count: int = 0
        self._pending_predictions: list[Prediction] = []
        # "retract" strategy: a rank-change transition is tentative for
        # `rank_confirm_windows - 1` predictions after it fires (see
        # _rank_check_retract). `_tentative_prior` is the state it replaced
        # (None once the grace period has passed, or there is no tentative
        # transition), `_tentative_deadline` counts down the remaining grace
        # predictions, and `_tentative_detector_backup` is the statistical
        # detector's state from just before it was reset, in case retraction
        # needs to restore it.
        self._tentative_prior: PhaseState | None = None
        self._tentative_deadline: int = 0
        self._tentative_detector_backup: dict[str, Any] = {}

    @classmethod
    def from_args(cls, args: Any) -> PhaseAutomaton:
        """Build a PhaseAutomaton from parsed CLI args (``--pa-*`` flags).

        Shared by the online predictor and offline ``ftio --phase-automaton``
        so both read the same flags the same way.
        """
        method = getattr(args, "pa_method", "ksigma")
        if method == "none":
            method = None
        return cls(
            method=method,
            rank_changes_trigger=getattr(args, "pa_rank_trigger", True),
            rank_confirm_windows=getattr(args, "pa_rank_confirm", 2),
            rank_change_strategy=getattr(args, "pa_rank_strategy", "retract"),
            period_ratio_threshold=getattr(args, "pa_period_ratio", None),
            min_cycles=getattr(args, "pa_min_cycles", 2.0),
        )

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
        # Only a rank-change cause overrides the boundary time (both
        # strategies can place it earlier than this window's own end); every
        # other cause keeps using prediction.t_end, as before. `carryover`
        # (confirm strategy only) lists earlier windows to move to the new
        # state -- see _rank_check_confirm.
        rank_boundary: float | None = None
        carryover: list[Prediction] = []

        rank_differs = (
            self.rank_changes_trigger
            and self._current_state is not None
            and ranks > 0
            and ranks != self._current_state.ranks
        )

        if self.rank_change_strategy == "retract":
            cause, rank_boundary, rank_differs = self._rank_check_retract(
                prediction, ranks, rank_differs
            )
        else:
            cause, rank_boundary, carryover, rank_differs = self._rank_check_confirm(
                prediction, ranks, rank_differs
            )

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

        # --- Corroboration (confirm strategy only) -------------------
        # An unconfirmed rank difference that coincides with an independently
        # detected frequency/period shift is strong evidence the rank change is
        # real (a stray ZMQ blip would not also move the frequency estimate) --
        # attribute the transition to the rank change rather than the shift.
        # The retract strategy never leaves a rank change "unconfirmed" in the
        # first place (it always fires immediately), so this never applies to it.
        if (
            self.rank_change_strategy == "confirm"
            and cause not in (None, "rank_change")
            and rank_differs
        ):
            cause = "rank_change"
            rank_boundary = prediction.t_end
            carryover = []
            self._pending_rank = None
            self._pending_rank_count = 0
            self._pending_predictions = []

        # --- Bootstrap first state ----------------------------------
        if self._current_state is None:
            self._current_state = self._open_state(freq, conf, prediction.t_start, ranks)

        self._pred_index += 1

        # --- Fire transition ----------------------------------------
        if cause is not None:
            old = self._current_state
            if carryover:
                del old.predictions[-len(carryover) :]
            boundary_time = (
                rank_boundary if rank_boundary is not None else prediction.t_end
            )
            old.exit_time = boundary_time
            self._current_state = self._open_state(freq, conf, boundary_time, ranks)
            self._current_state.predictions.extend(carryover)
            self.transitions.append(
                Transition(
                    from_state=old.state_id,
                    to_state=self._current_state.state_id,
                    timestamp=boundary_time,
                    prediction_index=self._pred_index,
                    old_freq=old.dominant_freq,
                    new_freq=freq,
                    cause=cause,
                )
            )
            # This prediction is the evidence that triggered/confirmed the
            # transition -- it already reflects the new regime (that's why
            # it fired), so it belongs to the state just opened, not the one
            # just closed.
            self._current_state.predictions.append(prediction)

            if cause == "rank_change":
                if self.rank_change_strategy == "retract":
                    self._tentative_prior = old
                    self._tentative_deadline = self.rank_confirm_windows - 1
                    self._tentative_detector_backup = self._detector_state
                    self._detector_state = {}
                else:
                    self._pending_rank = None
                    self._pending_rank_count = 0
                    self._pending_predictions = []
            return True

        self._current_state.predictions.append(prediction)
        return False

    def _rank_check_retract(
        self, prediction: Prediction, ranks: int, rank_differs: bool
    ) -> tuple[str | None, float | None, bool]:
        """Fire immediately on a rank difference; retract within the grace period.

        Returns (cause, boundary_time, rank_differs) -- rank_differs comes
        back False if this call just retracted a tentative transition (the
        window is now consistent with the restored state, and should not
        also be treated as differing for this same step).
        """
        if (
            rank_differs
            and self._tentative_prior is not None
            and self._tentative_deadline > 0
            and ranks == self._tentative_prior.ranks
        ):
            # Reverts to the state the tentative one replaced -- undo it.
            reverted = self._current_state
            self.states.pop()
            self.transitions.pop()
            self._current_state = self._tentative_prior
            self._current_state.exit_time = np.nan
            self._current_state.predictions.extend(reverted.predictions)
            self._detector_state = self._tentative_detector_backup
            self._tentative_prior = None
            self._tentative_deadline = 0
            self._tentative_detector_backup = {}
            return None, None, False

        if self._tentative_deadline > 0:
            self._tentative_deadline -= 1
            if self._tentative_deadline == 0:
                self._tentative_prior = None  # survived its grace period; now permanent

        if rank_differs:
            return "rank_change", prediction.t_start, rank_differs
        return None, None, rank_differs

    def _rank_check_confirm(
        self, prediction: Prediction, ranks: int, rank_differs: bool
    ) -> tuple[str | None, float | None, list[Prediction], bool]:
        """Wait for `rank_confirm_windows` consecutive agreeing predictions.

        A single differing window is not trusted on its own: online, `ranks`
        can reflect an incomplete ZMQ poll cycle (not every rank's message had
        arrived yet), which looks identical to a real rank change for one
        window. Costs detecting a real change late if the input ends before
        confirmation completes -- see the "retract" strategy for the
        alternative that doesn't have that cost.

        Returns (cause, boundary_time, carryover, rank_differs). `carryover`
        lists every pending window before this one (e.g. with the default
        rank_confirm_windows=2: just [window N-1], since window N is this
        call) -- they were provisionally appended to the OLD state while
        confirmation was pending, but genuinely belong to the new one, and
        the boundary is where the FIRST of them starts, not where this
        (confirming) window ends -- using this window's end makes the
        transition look `rank_confirm_windows - 1` windows later than the
        evidence supports.
        """
        if rank_differs:
            if ranks == self._pending_rank:
                self._pending_rank_count += 1
                self._pending_predictions.append(prediction)
            else:
                self._pending_rank = ranks
                self._pending_rank_count = 1
                self._pending_predictions = [prediction]
        else:
            self._pending_rank = None
            self._pending_rank_count = 0
            self._pending_predictions = []

        if rank_differs and self._pending_rank_count >= self.rank_confirm_windows:
            self._detector_state = {}  # reset statistical detector
            carryover = self._pending_predictions[:-1]
            boundary = carryover[0].t_start if carryover else prediction.t_end
            return "rank_change", boundary, carryover, rank_differs
        return None, None, [], rank_differs

    def build(self, predictions: list[Prediction]) -> None:
        """Build automaton offline from a complete list of predictions."""
        for pred in predictions:
            self.step(pred)

    def print_summary(self) -> None:
        print(f"\n{'='*65}")
        print(
            f"PhaseAutomaton  method={self.method!r}  "
            f"rank_sensitive={self.rank_changes_trigger} "
            f"({self.rank_change_strategy}, window={self.rank_confirm_windows})  "
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
            ``"rank_confirm_windows"``, ``"period_ratio_threshold"``,
            ``"n_states"``, ``"n_transitions"``, ``"states"``, and
            ``"transitions"``.
        """

        def _float(v):
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            return float(v)

        return {
            "method": self.method,
            "rank_changes_trigger": self.rank_changes_trigger,
            "rank_confirm_windows": self.rank_confirm_windows,
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


def windows_from_stft_prediction(prediction: Prediction) -> list[Prediction]:
    """Split one STFT Prediction into a list of single-window Predictions.

    ``ftio_stft`` (ftio/freq/_stft_workflow.py) already slides a window across
    the whole trace and computes a dominant frequency/confidence/time-range
    per window -- it just packs the result as array-valued fields on a single
    Prediction, with index 0 holding a whole-trace summary and indices 1..N
    the per-window values.  This unpacks indices 1..N into the one-Prediction-
    per-window shape PhaseAutomaton.build() expects, which is how the online
    predictor already feeds it: one Prediction per elapsed time step.

    This is what makes phase-automaton modelling possible offline, from a
    single ``ftio --transformation stft`` run, without a live ZMQ stream.

    Each window's `ranks` comes from `prediction.ranks_per_window` when the
    source trace's rank count actually varied mid-run (malleability -- see
    Simrun.merge_fields and ftio_stft), so a genuine `rank_change` transition
    can fire at the real boundary; otherwise every window falls back to the
    one constant `prediction.ranks` value, as before.

    Returns an empty list if `prediction` does not carry a per-window
    sequence (e.g. it came from DFT/wavelet, or autocorrelation merging
    collapsed the arrays).
    """
    freqs = prediction.dominant_freq
    confs = prediction.conf
    ranges = prediction.ranges
    if len(freqs) < 2 or len(ranges) < 2 or len(freqs) != len(ranges):
        return []

    ranks_per_window = prediction.ranks_per_window
    has_ranks_per_window = len(ranks_per_window) == len(freqs)

    windows = []
    for i in range(1, len(freqs)):  # skip index 0: the whole-trace summary
        t_start, t_end = ranges[i]
        win = Prediction(
            transformation=prediction.source or "stft",
            t_start=float(t_start),
            t_end=float(t_end),
            ranks=int(ranks_per_window[i]) if has_ranks_per_window else prediction.ranks,
        )
        win.dominant_freq = np.array([freqs[i]])
        win.conf = np.array([confs[i] if i < len(confs) else np.nan])
        win.amp = np.array([1.0])
        windows.append(win)
    return windows
