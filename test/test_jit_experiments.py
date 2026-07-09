"""
Tests for ftio/api/gekkoFs/jit/experiments/: the synthetic GekkoFS message
generator, the ingest fan-out backends, and the four experiment drivers
(exp_a scaling, exp_b misprediction, exp_c stage-out, exp_d stress) plus the
ingest profiler.

The `*_smoke` tests assert the shape of each driver's result, not its measured
latency, so they run on the smallest grid that still exercises every code path.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import msgpack

from ftio.api.gekkoFs.ftio_gekko import ingest_app_bandwidth
from ftio.api.gekkoFs.jit.experiments.exp_a_scaling import run_sweep
from ftio.api.gekkoFs.jit.experiments.synthetic_messages import (
    make_round,
    make_server_message,
)


def test_synthetic_message_matches_gekkofs_wire_format():
    """Generated messages must match the current 9-field GekkoFS layout.

    Order per msgpack_util.hpp: flush_t, hostname, pid, io_type, start_t,
    end_t, req_size, total_iops, total_bytes.
    """
    msg = make_server_message(server_id=3, n_events=64, io_type="w")
    up = msgpack.Unpacker()
    up.feed(msg)
    items = list(up)
    assert len(items) == 9
    assert isinstance(items[1], str)  # hostname
    assert items[3] == "w"  # io_type
    assert len(items[4]) == 64  # start_t array


def test_synthetic_message_parses_through_ingest():
    """A synthetic round must produce a non-empty application-level bandwidth."""
    msgs = make_round(5, n_events=128, seed=1)
    b, t, total_bytes, _flush, _ext = ingest_app_bandwidth(msgs, "w", n_workers=1)
    assert len(b) > 0
    assert total_bytes == 5 * 128 * 524288  # summed across the 5 servers


def test_gekkofs_io_type_filter():
    """9-field messages are kept for the matching mode and skipped otherwise."""
    write_msgs = make_round(3, n_events=64, io_type="w", seed=2)
    b_w, *_ = ingest_app_bandwidth(write_msgs, "w", n_workers=1)
    b_r, *_ = ingest_app_bandwidth(write_msgs, "r", n_workers=1)
    assert len(b_w) > 0  # write request keeps write messages
    assert len(b_r) == 0  # read request filters them out


def test_run_sweep_smoke():
    """A tiny sweep returns positive single-process latencies for each N."""
    rows = run_sweep(servers=[2, 4], workers=[1], n_events=64, repeat=1)
    assert len(rows) == 2
    for row in rows:
        assert row["w1"] > 0
        assert row["events"] == row["servers"] * 64


def test_misprediction_clean_signal_recovers_frequency():
    """On a clean periodic round FTIO must recover the ground-truth frequency."""
    from ftio.api.gekkoFs.jit.experiments.exp_b_misprediction import evaluate

    r = evaluate(jitter=0.0, period=2.0, n_servers=8, n_events=256)
    assert r["rel_err"] < 0.15  # within 15% of the 0.5 Hz ground truth


def test_misprediction_degrades_with_jitter():
    """Heavy jitter must not predict better than the clean signal."""
    from ftio.api.gekkoFs.jit.experiments.exp_b_misprediction import evaluate

    clean = evaluate(jitter=0.0, period=2.0, n_servers=8, n_events=256)
    noisy = evaluate(jitter=0.5, period=2.0, n_servers=8, n_events=256)
    assert noisy["rel_err"] >= clean["rel_err"]


def test_process_resample_recovers_frequency():
    """The resample fan-out must recover the same frequency as the full path."""
    import numpy as np

    from ftio.api.gekkoFs.ftio_gekko import ingest_app_bandwidth
    from ftio.cli.ftio_core import core
    from ftio.parse.args import parse_args
    from ftio.prediction.helper import get_dominant

    msgs = make_round(4, n_events=128, period=2.0, horizon=20.0, io_type="w", seed=0)

    def freq(backend, workers):
        b, t, tb, _f, _e = ingest_app_bandwidth(
            msgs, "w", n_workers=workers, backend=backend, resample_hz=50.0
        )
        data = {
            "time": np.array(t),
            "bandwidth": np.array(b),
            "total_bytes": tb,
            "ranks": 4,
        }
        pred, _ = core(data, parse_args(["-e", "no", "-f", "50"], "ftio"))
        return get_dominant(pred)

    f_full = freq("thread", 1)  # serial, full resolution
    f_resamp = freq("process-resample", 4)
    assert np.isclose(f_full, f_resamp, rtol=0.1)  # same dominant frequency


def test_profile_ingest_smoke(capsys):
    """The ingest profiler runs and reports the three stages."""
    from ftio.api.gekkoFs.jit.experiments.profile_ingest import profile

    profile(n_servers=2, n_events=16, repeat=1)
    out = capsys.readouterr().out
    assert "msgpack decode" in out
    assert "overlap" in out


def test_backend_matrix_smoke():
    """The 2x2 backend matrix returns a latency for every combination."""
    from ftio.api.gekkoFs.jit.experiments.exp_a_scaling import run_backend_matrix

    res = run_backend_matrix(n_servers=2, n_events=16, workers=2, repeat=1)
    for key in (
        "single",
        "thread_full",
        "process_full",
        "thread_resample",
        "process_resample",
    ):
        assert res[key] > 0


def test_run_grid_smoke():
    """The 2D (events x servers) grid returns a row per pair."""
    from ftio.api.gekkoFs.jit.experiments.exp_a_scaling import run_grid

    # run_grid builds one process pool per cell, so keep the cells tiny: the
    # assertion is on the shape of the result, not on the measured latency.
    rows = run_grid(
        servers=[1, 2], events=[8, 16], repeat=1, n_workers=2, include_thread=False
    )
    assert len(rows) == 4  # one row per (events, servers) pair
    for row in rows:
        assert row["single"] > 0 and row["resample"] > 0
        assert row["thread"] is None  # include_thread=False


def test_stress_sweep_smoke():
    """The stress sweep reports keep-up / backlog per offered rate."""
    from ftio.api.gekkoFs.jit.experiments.exp_d_stress import stress

    rows = stress(n_servers=2, posting_hz=[0.01, 1e6], n_events=16, workers=2)
    assert rows[0]["single_keeps_up"]  # 0.01 Hz is trivially sustainable
    assert not rows[1]["resample_keeps_up"]  # 1e6 Hz cannot be sustained


def test_exp_c_parses_flush_log(tmp_path):
    """Experiment C must parse the real flush-log format and split by trigger."""
    from ftio.api.gekkoFs.jit.experiments.exp_c_stageout import decompose, parse_flush_log

    log = tmp_path / "flush.log"
    log.write_text(
        "2026-07-08 10:00:00 | FTIO-trigger | a -> /lustre/a | copy: 1.200 s | delete: 0.010 s\n"
        "2026-07-08 10:00:01 | FTIO-trigger | b -> /lustre/b | copy: 1.800 s | delete: 0.020 s\n"
        "2026-07-08 10:00:02 | post-app     | c -> /lustre/c | copy: 3.000 s | delete: 0.030 s\n"
        "garbage line that should be ignored\n"
    )
    records = parse_flush_log(str(log))
    assert len(records) == 3
    d = decompose(records)
    assert d["ftio"]["copy"]["n"] == 2
    assert d["post_app"]["copy"]["n"] == 1
    assert abs(d["ftio"]["copy"]["mean"] - 1.5) < 1e-9  # (1.2 + 1.8) / 2
