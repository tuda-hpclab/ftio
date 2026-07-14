"""Tests for the bandwidth.json handoff between FTIO and the JIT log dir.

FTIO's dump_json writes bandwidth.json into the predictor's cwd; jit copies it
into the run's log dir afterwards. A run in which FTIO captured no I/O produces
no file, and must not inherit the previous run's trace.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
from types import SimpleNamespace

from ftio.api.gekkoFs.jit.setup_helper import (
    bandwidth_path,
    clear_bandwidth,
    save_bandwidth,
)


def _settings(tmp_path, exclude_ftio: bool = False) -> SimpleNamespace:
    log_dir = tmp_path / "logs_nodes1_Jobid0_DF_1"
    log_dir.mkdir()
    return SimpleNamespace(log_dir=str(log_dir), exclude_ftio=exclude_ftio)


def test_bandwidth_is_read_from_the_log_dir_parent(tmp_path):
    settings = _settings(tmp_path)
    assert bandwidth_path(settings) == str(tmp_path / "bandwidth.json")


def test_clear_removes_a_trace_left_by_an_earlier_run(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "bandwidth.json").write_text('{"b": [1], "t": [0]}')
    clear_bandwidth(settings)
    assert not os.path.exists(bandwidth_path(settings))


def test_clear_is_a_noop_when_there_is_nothing_to_clear(tmp_path):
    clear_bandwidth(_settings(tmp_path))  # must not raise


def test_clear_keeps_the_trace_when_ftio_is_excluded(tmp_path):
    settings = _settings(tmp_path, exclude_ftio=True)
    (tmp_path / "bandwidth.json").write_text("{}")
    clear_bandwidth(settings)
    assert os.path.exists(bandwidth_path(settings))


def test_save_copies_a_fresh_trace_into_the_log_dir(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "bandwidth.json").write_text('{"b": [2], "t": [1]}')
    save_bandwidth(settings)
    copied = os.path.join(settings.log_dir, "bandwidth.json")
    with open(copied) as f:
        assert f.read() == '{"b": [2], "t": [1]}'


def test_save_writes_nothing_when_ftio_captured_no_io(tmp_path):
    # The regression: Castro wrote outside the mount, so no trace was produced.
    # The log dir must stay empty rather than inherit the last run's LAMMPS data.
    settings = _settings(tmp_path)
    save_bandwidth(settings)
    assert not os.path.exists(os.path.join(settings.log_dir, "bandwidth.json"))
