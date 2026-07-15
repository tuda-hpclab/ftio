"""Tests for the per-app JIT log layout and the jit_plot path resolver.

These pin the structure a multi-app sweep relies on:

1. `mode_label` names the run mode (glass/gekko/pfs) from the exclude flags.
2. `compose_log_dir` lays logs out as logs/<app>/nodes_<n>/rep_<rep>/<mode>,
   deriving the repetition so the three modes of one pass share a rep_<n> and a
   second pass increments it -- without this a multi-app sweep overwrote itself.
3. `resolve_result_json` turns an app name / folder / cwd into its result.json,
   so `jit_plot lammps` and `jit_plot` (from inside logs/lammps) both work.

Portable: no JitSettings() (its branches key off the hostname), no benchmarks,
no GekkoFS. mode_label is a staticmethod; compose_log_dir takes an injectable
existence probe; resolve_result_json is exercised against tmp_path.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import json
import os

from ftio.api.gekkoFs.jit.jit_plot import _mode_label, resolve_result_json, totals_by_node
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.setup_helper import compose_log_dir

# ── mode_label: exclude flags -> the three paper modes ───────────────────────


def test_mode_label_pfs_when_no_daemon():
    # -x excludes everything, so exclude_daemon is set -> plain parallel FS.
    assert JitSettings.mode_label(True, False) == "pfs"
    assert JitSettings.mode_label(True, True) == "pfs"


def test_mode_label_gekko_when_ftio_excluded():
    assert JitSettings.mode_label(False, True) == "gekko"


def test_mode_label_glass_when_daemon_and_ftio_on():
    assert JitSettings.mode_label(False, False) == "glass"


# ── compose_log_dir: nested layout + derived repetition ──────────────────────


def test_compose_log_dir_nests_by_app_nodes_rep_mode():
    log_dir, result_dir = compose_log_dir("lammps", 9, "glass", exists=lambda _: False)
    assert log_dir == os.path.join("lammps", "nodes_9", "rep_1", "glass")
    assert result_dir == "lammps"


def test_compose_log_dir_derives_the_repetition_per_mode():
    seen = {os.path.join("lammps", "nodes_9", "rep_1", "glass")}
    probe = seen.__contains__
    # glass already ran once at this size -> next glass is rep_2 ...
    glass, _ = compose_log_dir("lammps", 9, "glass", exists=probe)
    assert glass.endswith(os.path.join("rep_2", "glass"))
    # ... but a mode that has not run yet stays rep_1 (the modes stay aligned).
    pfs, _ = compose_log_dir("lammps", 9, "pfs", exists=probe)
    assert pfs.endswith(os.path.join("rep_1", "pfs"))


def test_compose_log_dir_falls_back_when_app_or_mode_is_empty():
    log_dir, result_dir = compose_log_dir("", 4, "", exists=lambda _: False)
    assert log_dir == os.path.join("app", "nodes_4", "rep_1", "run")
    assert result_dir == "app"


# ── resolve_result_json: app name / folder / cwd -> result.json ──────────────


def _make_result(tmp_path, app):
    d = tmp_path / "logs" / app
    d.mkdir(parents=True)
    (d / "result.json").write_text("[]")
    return d


def test_resolve_bare_app_name_finds_logs_app_result(tmp_path):
    _make_result(tmp_path, "lammps")
    got = resolve_result_json("lammps", cwd=str(tmp_path))
    assert got == str(tmp_path / "logs" / "lammps" / "result.json")


def test_resolve_prefers_app_dir_over_logs_app(tmp_path):
    # ./<app>/result.json wins over ./logs/<app>/result.json when both exist.
    (tmp_path / "lammps").mkdir()
    (tmp_path / "lammps" / "result.json").write_text("[]")
    _make_result(tmp_path, "lammps")
    got = resolve_result_json("lammps", cwd=str(tmp_path))
    assert got == str(tmp_path / "lammps" / "result.json")


def test_resolve_directory_argument(tmp_path):
    d = _make_result(tmp_path, "warpx")
    got = resolve_result_json(str(d), cwd=str(tmp_path))
    assert got == str(d / "result.json")


def test_resolve_empty_means_cwd_result(tmp_path):
    # `cd logs/lammps && jit_plot` -> ./result.json
    d = _make_result(tmp_path, "lammps")
    assert resolve_result_json("", cwd=str(d)) == str(d / "result.json")
    assert resolve_result_json(".", cwd=str(d)) == str(d / "result.json")


def test_resolve_passes_a_json_path_through(tmp_path):
    got = resolve_result_json("sub/result.json", cwd=str(tmp_path))
    assert got == str(tmp_path / "sub" / "result.json")


def test_resolve_missing_app_returns_a_sensible_guess(tmp_path):
    # Nothing on disk: point at <app>/result.json (the new layout) so the error names it.
    got = resolve_result_json("nope", cwd=str(tmp_path))
    assert got == str(tmp_path / "nope" / "result.json")


# ── the -t/--table console summary (totals per node) ─────────────────────────


def test_mode_label_maps_the_json_letters():
    assert _mode_label("DF") == "glass"
    assert _mode_label("D") == "gekko"
    assert _mode_label("") == "pfs"


def _write_result(tmp_path, entries):
    j = tmp_path / "result.json"
    j.write_text(json.dumps(entries))
    return str(j)


def test_totals_by_node_totals_and_compute_nodes(tmp_path):
    j = _write_result(
        tmp_path,
        [
            {
                "nodes": 9,
                "data": [
                    {"mode": "DF", "app": 24.0, "stage_in": 4.0, "stage_out": 16.0},
                    {"mode": "D", "app": 20.0, "stage_in": 0.0, "stage_out": 16.0},
                    {"mode": "", "app": 22.0, "stage_in": 0.0, "stage_out": 0.0},
                ],
            }
        ],
    )
    rows = totals_by_node(j)
    assert set(rows) == {8}  # 9 minus the dedicated FTIO node
    assert rows[8]["glass"] == (24.0, 44.0)
    assert rows[8]["gekko"] == (20.0, 36.0)
    assert rows[8]["pfs"] == (22.0, 22.0)


def test_totals_by_node_latest_timestamp_wins(tmp_path):
    j = _write_result(
        tmp_path,
        [
            {
                "nodes": 3,
                "data": [
                    {
                        "mode": "DF",
                        "app": 1.0,
                        "stage_in": 0,
                        "stage_out": 0,
                        "timestamp": "2026-01-01 00:00:00",
                    },
                    {
                        "mode": "DF",
                        "app": 9.0,
                        "stage_in": 0,
                        "stage_out": 0,
                        "timestamp": "2026-01-02 00:00:00",
                    },
                ],
            }
        ],
    )
    assert totals_by_node(j)[2]["glass"][0] == 9.0
