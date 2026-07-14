"""Tests that the accumulated GekkoFS bandwidth trace stays monotonic in time.

`ftio_gekko.run` accumulates each prediction round into b_app/t_app with a plain
`extend`. That is only sound if every round returns the timestamps it actually
observed. The `process-resample` fan-out resamples onto a grid spanning
[0, horizon], so without trimming it hands back a series starting at t=0 on
every round, and the concatenated trace jumps backwards in time.

Regression: WarpX (4 writing ranks) produced a bandwidth.json with 9 backward
jumps in t and 10 stacked copies of the run; the DFT then reported a confident
but bogus 3.0 s period. LAMMPS was unaffected because a single rank writes its
restart, which takes the serial path (`len(files_or_msgs) <= 1`).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import msgpack
import numpy as np
import pytest

from ftio.api.gekkoFs.ftio_gekko import _trim_grid_padding, ingest_app_bandwidth


def _msg(flush_us: int, starts_us: list, ends_us: list, sizes: list) -> bytes:
    """One GekkoFS 8-field message, with flush_t relative in us as the daemon sends it."""
    fields = [
        flush_us,
        "electric",
        1,
        starts_us,
        ends_us,
        sizes,
        len(sizes),
        int(sum(sizes)),
    ]
    return b"".join(msgpack.packb(f) for f in fields)


def _two_ranks_one_burst() -> list:
    """Two ranks each writing 100 MB between t = 1.0 s and t = 2.0 s."""
    msg = _msg(3_000_000, [1_000_000], [2_000_000], [100_000_000])
    return [msg, msg]


# --------------------------------------------------------------------------- #
# _trim_grid_padding
# --------------------------------------------------------------------------- #
def test_trim_drops_the_leading_zeros_the_grid_pads_on():
    grid = np.arange(0.0, 1.0, 0.1)
    b = np.array([0, 0, 0, 0, 0, 5.0, 5.0, 0, 0, 0])
    trimmed_b, trimmed_t = _trim_grid_padding(b, grid)
    # one zero kept on each side so the step still rises from / falls back to zero
    assert trimmed_b == [0.0, 5.0, 5.0, 0.0]
    assert trimmed_t == pytest.approx([0.4, 0.5, 0.6, 0.7])


def test_trim_keeps_interior_zeros_between_two_bursts():
    grid = np.arange(0.0, 0.8, 0.1)
    b = np.array([0, 0, 3.0, 0, 0, 4.0, 0, 0])
    trimmed_b, _ = _trim_grid_padding(b, grid)
    assert trimmed_b == [0.0, 3.0, 0.0, 0.0, 4.0, 0.0]


def test_trim_of_an_all_zero_round_yields_nothing():
    # run() treats an empty b as "no new data" and terminates the prediction.
    assert _trim_grid_padding(np.zeros(5), np.arange(5.0)) == ([], [])


def test_trim_preserves_the_transferred_bytes():
    grid = np.arange(0.0, 1.0, 0.1)
    b = np.zeros(10)
    b[3:6] = 2.0
    before = float(np.sum(b[:-1] * np.diff(grid)))
    tb, tt = _trim_grid_padding(b, grid)
    after = float(np.sum(np.array(tb[:-1]) * np.diff(np.array(tt))))
    assert after == pytest.approx(before)


# --------------------------------------------------------------------------- #
# ingest_app_bandwidth: the fanned-out path must not restart the clock
# --------------------------------------------------------------------------- #
def test_resample_fanout_does_not_restart_the_clock_at_zero():
    """The regression itself: t[0] == 0 every round is what caused the sawtooth."""
    _, t, *_ = ingest_app_bandwidth(
        _two_ranks_one_burst(), "w", 4, "process-resample", 10.0
    )
    assert t, "fan-out returned no samples"
    # The burst starts at 1.0 s. The fan-out keeps one zero sample in front of it;
    # on a single-core box resolve_workers() falls back to the serial path, which
    # starts exactly at the burst. Both are fine -- starting at 0.0 is not.
    assert 0.0 < t[0] <= 1.0


def test_resample_fanout_reports_the_same_bytes_and_window_as_serial():
    msgs = _two_ranks_one_burst()
    _, t_s, bytes_s, *_ = ingest_app_bandwidth(msgs, "w", 1, "thread", 10.0)
    _, t_p, bytes_p, *_ = ingest_app_bandwidth(msgs, "w", 4, "process-resample", 10.0)
    assert bytes_p == bytes_s == 200_000_000
    assert t_p[-1] == pytest.approx(t_s[-1], abs=0.11)
    assert t_p[0] == pytest.approx(t_s[0], abs=0.11)


def test_accumulating_rounds_never_steps_backwards_in_time():
    """Concatenating rounds the way run() does must keep t non-decreasing."""
    b_app, t_app = [], []
    for round_start_s in (1, 4, 7):  # three bursts, each 1 s long, 3 s apart
        us = round_start_s * 1_000_000
        msg = _msg(us + 2_000_000, [us], [us + 1_000_000], [100_000_000])
        b, t, *_ = ingest_app_bandwidth([msg, msg], "w", 4, "process-resample", 10.0)
        b_app.extend(b)
        t_app.extend(t)
    assert len(b_app) == len(t_app)
    assert np.all(np.diff(np.array(t_app)) >= 0), "trace jumps backwards (sawtooth)"


def test_accumulated_rounds_keep_each_burst_separate():
    """Three bursts in, three bursts out -- no stacked copies of the run."""
    b_app, t_app = [], []
    for round_start_s in (1, 4, 7):
        us = round_start_s * 1_000_000
        msg = _msg(us + 2_000_000, [us], [us + 1_000_000], [100_000_000])
        b, t, *_ = ingest_app_bandwidth([msg, msg], "w", 4, "process-resample", 10.0)
        b_app.extend(b)
        t_app.extend(t)
    b_arr = np.array(b_app)
    bursts = np.sum((b_arr[:-1] == 0) & (b_arr[1:] > 0))
    assert bursts == 3
