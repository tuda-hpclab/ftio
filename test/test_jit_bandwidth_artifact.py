"""Tests for the bandwidth.json artifact in the JIT log dir.

FTIO's dump_json writes bandwidth.json into the predictor's cwd, and the
predictor cds into the run's log dir first (see the predictor call in
setup_core), so the trace lands in the log dir directly -- no copy-after. A run
in which FTIO captured no I/O produces no file, and must not inherit an earlier
run's trace.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
from pathlib import Path
from types import SimpleNamespace

from ftio.api.gekkoFs.jit.setup_helper import (
    bandwidth_path,
    clear_bandwidth,
    save_bandwidth,
)


def _settings(tmp_path, exclude_ftio: bool = False) -> SimpleNamespace:
    # The nested per-app layout: logs/<app>/nodes_<n>/rep_<rep>/<mode>
    log_dir = tmp_path / "logs" / "lammps" / "nodes_9" / "rep_1" / "glass"
    log_dir.mkdir(parents=True)
    return SimpleNamespace(log_dir=str(log_dir), exclude_ftio=exclude_ftio)


def test_bandwidth_lives_in_the_log_dir(tmp_path):
    settings = _settings(tmp_path)
    assert bandwidth_path(settings) == os.path.join(settings.log_dir, "bandwidth.json")


def test_clear_removes_a_trace_left_by_an_earlier_run(tmp_path):
    settings = _settings(tmp_path)
    Path(bandwidth_path(settings)).write_text('{"b": [1], "t": [0]}')
    clear_bandwidth(settings)
    assert not os.path.exists(bandwidth_path(settings))


def test_clear_is_a_noop_when_there_is_nothing_to_clear(tmp_path):
    clear_bandwidth(_settings(tmp_path))  # must not raise


def test_clear_keeps_the_trace_when_ftio_is_excluded(tmp_path):
    settings = _settings(tmp_path, exclude_ftio=True)
    Path(bandwidth_path(settings)).write_text("{}")
    clear_bandwidth(settings)
    assert os.path.exists(bandwidth_path(settings))


def test_save_keeps_the_trace_the_predictor_wrote(tmp_path):
    # The predictor already wrote it into the log dir; save_bandwidth just
    # confirms it and must leave it untouched (no copy onto itself).
    settings = _settings(tmp_path)
    Path(bandwidth_path(settings)).write_text('{"b": [2], "t": [1]}')
    save_bandwidth(settings)
    with open(bandwidth_path(settings)) as f:
        assert f.read() == '{"b": [2], "t": [1]}'


def test_save_writes_nothing_when_ftio_captured_no_io(tmp_path):
    # The regression case: Castro wrote outside the mount, so no trace was
    # produced. The log dir must stay empty rather than inherit the last run.
    settings = _settings(tmp_path)
    save_bandwidth(settings)
    assert not os.path.exists(bandwidth_path(settings))
