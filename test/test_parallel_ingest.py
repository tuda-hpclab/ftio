"""
Tests for parallel ingestion: proves that fanning the drained GekkoFS messages
out across workers and folding the per-worker partials reproduces exactly the
single-instance application-level bandwidth, on the real example messages.

Also covers reduce_partials properties and parse hardening (8/9-field layouts,
malformed messages).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import glob
import os

import msgpack
import numpy as np

from ftio.api.gekkoFs.parse_gekko import assign, parse
from ftio.parse.bandwidth import overlap
from ftio.prediction.parallel_ingest import (
    parallel_overlap,
    partition,
    reduce_partials,
    resample_step,
)

MSG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "examples", "API", "gekkoFs", "MSGPACK"
)


def _fresh():
    return {
        "avg_throughput": [],
        "t_end": [],
        "t_start": [],
        "hostname": "",
        "pid": 0,
        "io_type": "",
        "req_size": [],
        "total_bytes": 0,
        "total_iops": 0,
        "t_flush": 0.0,
    }


def _write_msgs():
    return sorted(glob.glob(os.path.join(MSG_DIR, "write_*.msgpack")))


def gekko_map(msgs, io_type="write"):
    """Parse a chunk of messages and overlap them into one (b, t) partial."""
    data = _fresh()
    for m in msgs:
        data, _ = parse(m, data, io_type=io_type)
    if not data["avg_throughput"]:
        return [], []
    return overlap(data["avg_throughput"], data["t_start"], data["t_end"])


def _sample_step(b, t, grid):
    """Evaluate a left-anchored step function (b in effect from t[k]) on grid."""
    b = np.asarray(b, dtype=float)
    t = np.asarray(t, dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.where(idx >= 0, b[idx.clip(min=0)], 0.0)
    return out


def _integral(b, t):
    b = np.asarray(b, dtype=float)
    t = np.asarray(t, dtype=float)
    return float(np.sum(b[:-1] * np.diff(t)))


def _assert_same_function(b1, t1, b2, t2):
    """Two step functions are equal if they agree between all breakpoints.

    Tolerance is scaled to the signal magnitude: the values are sums of many
    ~1e8 B/s terms, so ~1e-4 float rounding at step boundaries is expected and
    must not be mistaken for a real difference.
    """
    breaks = np.unique(np.concatenate([np.asarray(t1), np.asarray(t2)]))
    assert len(breaks) >= 2
    mids = (breaks[:-1] + breaks[1:]) / 2.0
    v1 = _sample_step(b1, t1, mids)
    v2 = _sample_step(b2, t2, mids)
    scale = max(1.0, float(np.max(np.abs(v1))))
    assert np.allclose(v1, v2, rtol=1e-9, atol=1e-6 * scale)
    assert np.isclose(_integral(b1, t1), _integral(b2, t2), rtol=1e-9, atol=1e-6 * scale)


# --------------------------------------------------------------------------- #
# Parse hardening
# --------------------------------------------------------------------------- #
def test_real_message_parses():
    """The shipped 8-field messages must yield throughput (regression: skipped)."""
    msgs = _write_msgs()
    assert msgs, "example messages missing"
    data, ext = parse(msgs[0], _fresh(), io_type="write")
    assert len(data["avg_throughput"]) == 1024
    assert data["hostname"] == "electric"
    assert data["total_bytes"] > 0


def test_nine_field_layout_still_filters_io_type():
    """The legacy 9-field layout keeps io_type filtering (mismatch -> skip)."""
    fields = [111, "host", 7, "read", [1, 2], [3, 4], [8, 8], 2, 16]
    up = msgpack.Unpacker()
    up.feed(msgpack.packb(fields))  # single array item -> len 1 -> skipped safely
    # feed as separate top-level items instead:
    up = msgpack.Unpacker()
    for f in fields:
        up.feed(msgpack.packb(f))
    data = assign(_fresh(), up, io_type="write")  # message says read
    assert len(data["avg_throughput"]) == 0  # filtered out


def test_malformed_message_skipped(capsys):
    """A truncated message is skipped with a log, not silently misaligned."""
    up = msgpack.Unpacker()
    for f in [111, "host", 7, [1, 2, 3]]:  # only 4 fields
        up.feed(msgpack.packb(f))
    data = assign(_fresh(), up, io_type="write")
    assert len(data["avg_throughput"]) == 0
    assert "skipping message" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# reduce_partials properties
# --------------------------------------------------------------------------- #
def test_resample_step():
    """resample_step samples a left-anchored step function onto a grid."""
    # value 5 on [0,2), value 3 on [2,4), 0 after
    b = [5.0, 3.0, 0.0]
    t = [0.0, 2.0, 4.0]
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    out = resample_step(b, t, grid)
    assert list(out) == [5.0, 5.0, 3.0, 3.0, 0.0, 0.0]
    # empty partial -> zeros
    assert list(resample_step([], [], grid)) == [0.0] * len(grid)


def test_reduce_partials_skips_empty():
    b, t = reduce_partials([([], []), ([1.0, 0.0], [0.0, 1.0]), ([], [])])
    assert list(b) == [1.0, 0.0]
    assert list(t) == [0.0, 1.0]


def test_reduce_partials_commutative():
    p1 = overlap([2.0], [0.0], [2.0])
    p2 = overlap([3.0], [1.0], [3.0])
    b_ab, t_ab = reduce_partials([p1, p2])
    b_ba, t_ba = reduce_partials([p2, p1])
    _assert_same_function(b_ab, t_ab, b_ba, t_ba)


def test_reduce_partials_tree_equals_linear():
    """Tree (default) and linear reductions must agree on the real messages."""
    msgs = _write_msgs()
    partials = [gekko_map([m]) for m in msgs]
    b_tree, t_tree = reduce_partials(partials, strategy="tree")
    b_lin, t_lin = reduce_partials(partials, strategy="linear")
    _assert_same_function(b_tree, t_tree, b_lin, t_lin)


# --------------------------------------------------------------------------- #
# Equivalence proof: single-instance == fan-out, on the real messages
# --------------------------------------------------------------------------- #
def test_fanout_equals_single_instance_synthetic():
    """Additive box functions: overlap-all == fold of per-group overlaps."""
    # three overlapping bursts on different grids
    groups = [
        ([5.0], [0.0], [4.0]),
        ([2.0, 7.0], [1.0, 3.0], [2.0, 6.0]),
        ([1.0], [2.5], [5.5]),
    ]
    b_all = [g[0] for g in groups]
    ts_all = [g[1] for g in groups]
    te_all = [g[2] for g in groups]
    b_single, t_single = overlap(sum(b_all, []), sum(ts_all, []), sum(te_all, []))
    partials = [overlap(b, ts, te) for b, ts, te in groups]
    b_fan, t_fan = reduce_partials(partials)
    _assert_same_function(b_single, t_single, b_fan, t_fan)


def test_fanout_equals_single_instance_real_messages():
    """On the shipped GekkoFS messages, fan-out reproduces single-instance."""
    msgs = _write_msgs()
    assert len(msgs) >= 2

    # single instance: all messages in one overlap
    b_single, t_single = gekko_map(msgs)
    assert len(b_single) > 0

    # fan-out: partition -> per-group overlap -> fold
    for n_workers in (2, 3, len(msgs)):
        chunks = partition(msgs, n_workers)
        partials = [gekko_map(c) for c in chunks]
        b_fan, t_fan = reduce_partials(partials)
        _assert_same_function(b_single, t_single, b_fan, t_fan)


def test_parallel_overlap_process_path():
    """parallel_overlap in a real process pool matches inline single-instance."""
    msgs = _write_msgs()
    b_single, t_single = gekko_map(msgs)
    b_par, t_par = parallel_overlap(msgs, gekko_map, n_workers=2)
    _assert_same_function(b_single, t_single, b_par, t_par)


# --------------------------------------------------------------------------- #
# GLASS ingest helper: --ingest-workers must not change the result
# --------------------------------------------------------------------------- #
def test_ingest_app_bandwidth_serial_equals_fanout():
    """ingest_app_bandwidth is invariant to the worker count (b, t, total_bytes)."""
    from ftio.api.gekkoFs.ftio_gekko import ingest_app_bandwidth

    msgs = _write_msgs()
    b1, t1, bytes1, flush1, ext1 = ingest_app_bandwidth(msgs, "write", n_workers=1)
    assert len(b1) > 0 and bytes1 > 0
    for n in (2, 4):
        b, t, total_bytes, t_flush, ext = ingest_app_bandwidth(msgs, "write", n)
        _assert_same_function(b1, t1, b, t)
        assert total_bytes == bytes1  # summed across servers, order-independent
        assert np.isclose(t_flush, flush1)
