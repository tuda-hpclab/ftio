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
            "write": [...]} as returned by load_signals(); when given, every
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


def load_signals(trace: str) -> dict[str, list[Signal]]:
    """Bandwidth signals of a trace, grouped by I/O mode (see IO_MODES).
    Parses the trace once per FTIO -m value.
    """
    from ftio.parse.extract import get_time_behavior
    from ftio.parse.scales import Scales

    signals: dict[str, list[Signal]] = {}
    for mode, ftio_modes in IO_MODES.items():
        signals[mode] = []
        for ftio_mode in ftio_modes:
            data = Scales(["ftio", trace, "-e", "no", "-m", ftio_mode])
            sims = get_time_behavior(data.get_io_mode(ftio_mode))
            if sims:  # an FTIO mode may have no I/O at all
                signals[mode].append((sims[0]["time"], sims[0]["bandwidth"]))
    return signals


def phase_vectors(
    trace: str, ftio_args: list[str] | None = None, with_io: bool = True
) -> list[PhaseVector]:
    """Run FTIO's STFT + phase automaton on one trace and describe its phases.

    Args:
        trace: path of one trace file.
        ftio_args: extra FTIO options, added after FTIO_ARGS, so they override
            it (e.g. another --pa-period-ratio). Must keep "-tr stft".
        with_io: also load the read and write signals and fill PhaseVector.io.

    Raises:
        ValueError: FTIO produced no per-window STFT sequence, or the automaton
            accepted none of its windows.
    """
    from ftio.cli.ftio_core import main as ftio_main
    from ftio.modeling.phase_automaton import windows_from_stft_prediction

    predictions, args = ftio_main(["ftio", trace, *FTIO_ARGS, *(ftio_args or [])])
    windows = windows_from_stft_prediction(predictions[0]) if predictions else []
    if not windows:
        raise ValueError(f"PhaseVector: no per-window STFT sequence for {trace}")
    aut = PhaseAutomaton.from_args(args)
    aut.build(windows)
    if not aut.states:
        raise ValueError(
            f"PhaseVector: no phase found for {trace} "
            f"(need at least {aut.min_cycles} periods per STFT window)"
        )
    signals = load_signals(trace) if with_io else None
    return from_automaton(
        aut, predictions[0].t_start, predictions[0].t_end, signals, run_id=trace
    )
