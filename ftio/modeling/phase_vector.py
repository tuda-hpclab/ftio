"""
Phase vectors: describe each phase-automaton state as a vector of descriptors.

A phase is not defined by its I/O mode. Instead, frequency, duration, rank
count and the I/O mode (read/write mix, bytes, intensity) are all features of
one phase, so a checkpoint that writes and reads back stays one phase instead
of separated paths. These vectors are the input for matching phases across runs
and rank counts (see docs/phase_automaton.md, section 5).

Author: Albani Alvarado Mauriello
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: October 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from ftio.modeling.phase_automaton import PhaseAutomaton, PhaseState

# I/O mode (read or write) -> the FTIO -m values summed into it
# ("*_async" loads TMIO's *_async_t data); empty ones contribute nothing
IO_MODES = {
    "read": ("read_sync", "read_async"),
    "write": ("write_sync", "write_async"),
}

# FTIO options for the automaton: it needs STFT windows, no plots are wanted, and
# without the period-ratio trigger short traces rarely leave their first state
FTIO_ARGS = ["-e", "no", "-tr", "stft", "--pa-period-ratio", "1.5"]

# (time, bandwidth); b[i] holds from t[i] until t[i+1]
Signal = tuple[np.ndarray, np.ndarray]


@dataclass
class RunContext:
    """What all phases of one run share."""

    run_id: str = ""  # e.g. the trace file
    duration: float = 0.0
    total_bytes: float = 0.0
    n_phases: int = 1
    min_period: float = 0.0


@dataclass
class IOMode:
    """Data moved by one phase, split by I/O mode (read or write)."""

    read_bytes: float = 0.0
    write_bytes: float = 0.0

    @property
    def total_bytes(self) -> float:
        return self.read_bytes + self.write_bytes

    @property
    def read_fraction(self) -> float:
        """Share of reads in all bytes: 0 = write-only, 1 = read-only, nan = no I/O."""
        return self.read_bytes / self.total_bytes if self.total_bytes > 0 else np.nan


@dataclass
class PhaseVector:
    """Descriptors of one phase (one automaton state) of one run."""

    phase_id: int
    ranks: int
    start: float
    end: float
    period: float  # median over the phase's STFT windows
    n_windows: int  # windows the automaton accepted (the warm-up guard drops some)
    io: IOMode | None = None  # None if the I/O signals were not loaded
    # source automaton state, kept for reference (e.g. its per-window predictions)
    state: PhaseState | None = field(default=None, repr=False)
    run: RunContext | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def frequency(self) -> float:
        return 1.0 / self.period if self.period > 0 else np.nan

    @property
    def intensity(self) -> float:
        """Average bandwidth over the phase (bytes / duration), idle time included."""
        if self.io is None or self.duration <= 0:
            return np.nan
        return self.io.total_bytes / self.duration

    @property
    def bytes_per_rank(self) -> float:
        if self.io is None or self.ranks <= 0:
            return np.nan
        return self.io.total_bytes / self.ranks

    @property
    def period_ratio(self) -> float:
        """Period compared to the shortest period of the run."""
        if self.run is None or self.run.min_period <= 0:
            return np.nan
        return self.period / self.run.min_period

    @property
    def relative_position(self) -> float:
        """Position in the run: 0 = first phase, 1 = last phase."""
        if self.run is None:
            return np.nan
        if self.run.n_phases <= 1:
            return 0.0
        return self.phase_id / (self.run.n_phases - 1)

    @property
    def relative_duration(self) -> float:
        """Fraction of run time spent in this phase"""
        if self.run is None or self.run.duration <= 0:
            return np.nan
        return self.duration / self.run.duration

    @property
    def relative_volume(self) -> float:
        """Portion of total bytes moved in this phase"""
        if self.io is None or self.run is None or self.run.total_bytes <= 0:
            return np.nan
        return self.io.total_bytes / self.run.total_bytes

    def as_dict(self) -> dict:
        """Flat dict of all descriptors (e.g. for matcher)."""
        out = {
            "phase_id": self.phase_id,
            "ranks": self.ranks,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "period": self.period,
            "frequency": self.frequency,
            "n_windows": self.n_windows,
            "period_ratio": self.period_ratio,
            "relative_position": self.relative_position,
            "relative_duration": self.relative_duration,
        }
        if self.io is not None:
            out.update(
                {
                    "read_bytes": self.io.read_bytes,
                    "write_bytes": self.io.write_bytes,
                    "total_bytes": self.io.total_bytes,
                    "read_fraction": self.io.read_fraction,
                    "intensity": self.intensity,
                    "bytes_per_rank": self.bytes_per_rank,
                    "relative_volume": self.relative_volume,
                }
            )
        return out


def bytes_in(signal: Signal, start: float, end: float) -> float:
    """Signal over a time interval [start, end)."""
    t, b = signal
    if len(t) == 0:
        return 0.0
    t_next = np.append(t[1:], t[-1])  # each step ends where the next begins
    lo = np.clip(t, start, end)
    hi = np.clip(t_next, start, end)
    return float(np.sum(b * (hi - lo)))  # steps outside the interval get width 0


def from_automaton(
    aut: PhaseAutomaton,
    t_start: float,
    t_end: float,
    signals: dict[str, list[Signal]] | None = None,
    run_id: str = "",
) -> list[PhaseVector]:
    """One PhaseVector per automaton state, including the I/O of each phase.

    The automaton only provides the timing; the bytes per phase are computed by
    integrating the bandwidth signals over each state's [entry_time, exit_time).

    Args:
        aut: automaton for one run.
        t_start, t_end: start and end of the run's data. The warm-up guard can
            drop windows anywhere in the run, including leading ones, and the
            automaton never sees the last phase end, so the first phase is
            stretched back to t_start and the last to t_end. Dropped windows in
            between stay covered by the neighbouring phases' intervals.
        signals: optional bandwidth signals per I/O mode, {"read": [...],
            "write": [...]} as built by _prediction(); when given, every
            vector gets the bytes moved in its interval, and the run is widened
            to cover all signals (the automaton only saw one FTIO -m value,
            write_sync by default, so e.g. reads after the last write would
            otherwise fall outside).
        run_id: identifier of the run, e.g. the trace path.
    """
    if signals is not None:
        times = [t for sigs in signals.values() for t, _ in sigs if len(t)]
        if times:
            t_start = min(t_start, min(t[0] for t in times))
            t_end = max(t_end, max(t[-1] for t in times))

    vectors = []
    for i, s in enumerate(aut.states):
        start = t_start if i == 0 else s.entry_time
        end = s.exit_time if not np.isnan(s.exit_time) else t_end
        periods = [
            1.0 / w.dominant_freq[0] for w in s.predictions if w.dominant_freq[0] > 0
        ]
        io = None
        if signals is not None:
            io = IOMode(
                read_bytes=sum(
                    bytes_in(sig, start, end) for sig in signals.get("read", [])
                ),
                write_bytes=sum(
                    bytes_in(sig, start, end) for sig in signals.get("write", [])
                ),
            )
        vectors.append(
            PhaseVector(
                phase_id=i,
                ranks=s.ranks,
                start=start,
                end=end,
                period=float(np.median(periods)) if periods else s.period,
                n_windows=len(s.predictions),
                io=io,
                state=s,
            )
        )

    # only known once all phases exist
    run = RunContext(
        run_id=run_id,
        duration=t_end - t_start,
        total_bytes=sum(v.io.total_bytes for v in vectors if v.io is not None),
        n_phases=len(vectors),
        min_period=min(v.period for v in vectors),
    )
    for v in vectors:
        v.run = run
    return vectors


def _load_sims(
    trace: str, modes: dict[str, tuple[str, ...]] = IO_MODES
) -> dict[str, list[dict]]:
    """FTIO's time behavior of a trace per I/O mode (see IO_MODES).

    Each mode is parsed separately, as FTIO's JSON parser only completes the
    selected one. FTIO replaces a mode missing in the trace by another one; such
    a mode is skipped, so no data is counted under the wrong mode.
    """
    from ftio.parse.extract import get_time_behavior_and_args
    from ftio.parse.helper import match_mode

    sims: dict[str, list[dict]] = {}
    for kind, ftio_modes in modes.items():
        sims[kind] = []
        for ftio_mode in ftio_modes:
            data, args = get_time_behavior_and_args(
                ["ftio", trace, "-e", "no", "-m", ftio_mode]
            )
            if match_mode(args.mode) == match_mode(ftio_mode):
                sims[kind] += data[:1]
    return sims


def combine_signals(signals: dict[str, list[Signal]]) -> Signal:
    """Sum all read and write signals into one bandwidth signal. Needed for read/write mix"""
    from ftio.parse.bandwidth import overlap_two_series

    parts = [sig for sigs in signals.values() for sig in sigs]
    if not parts:
        raise ValueError("PhaseVector: no I/O signal to combine")
    t, b = parts[0]
    for t_i, b_i in parts[1:]:
        b, t = overlap_two_series(b, t, b_i, t_i)  # FTIO's order is (b, t)
    return np.asarray(t), np.asarray(b)


def _prediction(trace: str, ftio_args: list[str] | None, mode: str | None = None):
    """FTIO's STFT on one I/O mode, or on the sum of all read and write signals.

    With `mode` (FTIO's -m), only that mode is loaded; if the trace moved no
    bytes in it, all read and write modes are combined instead.
    """
    from ftio.cli.ftio_core import core
    from ftio.parse.args import parse_args
    from ftio.parse.helper import match_mode

    sims = None
    if mode is not None:
        ftio_mode = match_mode(mode)  # e.g. "r" -> "read_sync"
        sims = _load_sims(trace, {ftio_mode.split("_")[0]: (ftio_mode,)})
        if not any(s["total_bytes"] > 0 for ss in sims.values() for s in ss):
            print(  # stderr, as phase_model hides FTIO's stdout
                f"PhaseVector: no {ftio_mode} I/O in {trace}, "
                "combining all read and write modes instead",
                file=sys.stderr,
            )
            sims = None
    if sims is None:
        sims = _load_sims(trace)
    first = next(
        (ss[0] for ss in (sims.get("write", []), sims.get("read", [])) if ss), None
    )
    if first is None:
        raise ValueError(f"PhaseVector: no I/O found in {trace}")
    signals = {
        mode: [(s["time"], s["bandwidth"]) for s in ss] for mode, ss in sims.items()
    }
    t, b = combine_signals(signals)
    total = sum(s["total_bytes"] for ss in sims.values() for s in ss)
    sim = {**first, "time": t, "bandwidth": b, "total_bytes": total}  # keeps ranks
    args = parse_args([*FTIO_ARGS, *(ftio_args or [])], "ftio")
    prediction, _ = core(sim, args)
    return prediction, args, signals


def phase_vectors(
    trace: str,
    ftio_args: list[str] | None = None,
    mode: str | None = None,
) -> list[PhaseVector]:
    """Run FTIO's STFT + phase automaton on one trace and describe its phases.

    By default, phases are detected on the sum of all read and write signals,
    so a phase that writes and reads back (e.g. a checkpoint) stays one phase;
    the read and write bytes of each phase are kept in PhaseVector.io.

    Args:
        trace: path of one trace file.
        ftio_args: extra FTIO options, added after FTIO_ARGS, so they override
            it (e.g. another --pa-period-ratio). Must keep "-tr stft". FTIO's
            -m has no effect here, use `mode` instead.
        mode: analyze only this I/O mode (FTIO's -m values, e.g. "read_sync")
            for both the phases and PhaseVector.io; falls back to all modes
            combined if the trace moved no bytes in it.

    Raises:
        ValueError: FTIO produced no per-window STFT sequence, or the automaton
            accepted none of its windows.
    """
    from ftio.modeling.phase_automaton import windows_from_stft_prediction

    prediction, args, signals = _prediction(trace, ftio_args, mode)
    windows = windows_from_stft_prediction(prediction) if prediction else []
    if not windows:
        raise ValueError(f"PhaseVector: no per-window STFT sequence for {trace}")
    aut = PhaseAutomaton.from_args(args)
    aut.build(windows)
    if not aut.states:
        raise ValueError(
            f"PhaseVector: no phase found for {trace} "
            f"(need at least {aut.min_cycles} periods per STFT window)"
        )
    return from_automaton(
        aut, prediction.t_start, prediction.t_end, signals, run_id=trace
    )
