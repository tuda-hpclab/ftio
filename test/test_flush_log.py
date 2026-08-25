"""
Tests for the flush log functionality in posix_control.py.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: 0.0.8
Date: May 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse
import os
from unittest.mock import patch

from ftio.api.gekkoFs.posix_control import (
    _write_flush_log,
    copy_file_and_unlink,
    move_files_os,
)

# ---------------------------------------------------------------------------
# _write_flush_log
# ---------------------------------------------------------------------------


def test_write_flush_log_creates_file(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/data/f0.h5", "/stage-out", "ftio", 1.23, 0.05)
    assert os.path.isfile(log)


def test_write_flush_log_ftio_label(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/data/f0.h5", "/stage-out", "ftio", 1.23, 0.05)
    with open(log) as f:
        content = f.read()
    assert "FTIO-trigger" in content
    assert "post-app" not in content


def test_write_flush_log_post_app_label(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/data/f1.h5", "/stage-out", "post_app", 0.9, 0.02)
    with open(log) as f:
        content = f.read()
    assert "post-app" in content
    assert "FTIO-trigger" not in content


def test_write_flush_log_contains_item_and_dst(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/data/f2.h5", "/lustre/stage-out", "ftio", 2.0, 0.1)
    with open(log) as f:
        content = f.read()
    assert "/gkfs/data/f2.h5" in content
    assert "/lustre/stage-out" in content


def test_write_flush_log_contains_timings(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/f.h5", "/out", "ftio", 3.456, 0.789)
    with open(log) as f:
        content = f.read()
    assert "3.456" in content
    assert "0.789" in content


def test_write_flush_log_appends(tmp_path):
    log = str(tmp_path / "flush.log")
    _write_flush_log(log, "/gkfs/a.h5", "/out", "ftio", 1.0, 0.1)
    _write_flush_log(log, "/gkfs/b.h5", "/out", "post_app", 2.0, 0.2)
    with open(log) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert "FTIO-trigger" in lines[0]
    assert "post-app" in lines[1]


def test_write_flush_log_noop_when_empty():
    # must not raise even with no log path
    _write_flush_log("", "/gkfs/f.h5", "/out", "ftio", 1.0, 0.1)


def test_write_flush_log_noop_on_bad_path():
    # unwritable path — must not propagate the exception
    _write_flush_log("/no/such/dir/flush.log", "/f.h5", "/out", "ftio", 1.0, 0.1)


# ---------------------------------------------------------------------------
# copy_file_and_unlink — check that it writes the log
# ---------------------------------------------------------------------------


def _make_args(tmp_path, triggered_by=None):
    args = argparse.Namespace(
        stage_out_path=str(tmp_path / "stage-out"),
        gkfs_mntdir=str(tmp_path / "mnt"),
        ld_preload="",
        host_file="",
        flush_log=str(tmp_path / "flush.log"),
        ignore_mtime=True,
        debug=False,
        regex=None,
        parallel_move_threads=1,
        flush_call="cp",
        node=None,
    )
    os.makedirs(args.stage_out_path, exist_ok=True)
    return args


def test_copy_file_and_unlink_writes_log(tmp_path):
    args = _make_args(tmp_path)
    src = tmp_path / "mnt" / "data.h5"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("hello")

    with patch("ftio.api.gekkoFs.posix_control.preloaded_call") as mock_call:
        # Same count either side of the copy, so the copy verifies as complete.
        mock_call.side_effect = lambda a, c: "1" if c.endswith("| wc -c") else ""
        copy_file_and_unlink(args, str(src), triggered_by="ftio")

    with open(args.flush_log) as f:
        log_content = f.read()
    assert "FTIO-trigger" in log_content
    assert str(src) in log_content


def test_copy_file_and_unlink_post_app_label(tmp_path):
    args = _make_args(tmp_path)
    src = tmp_path / "mnt" / "data.h5"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("hello")

    with patch("ftio.api.gekkoFs.posix_control.preloaded_call") as mock_call:
        # Same count either side of the copy, so the copy verifies as complete.
        mock_call.side_effect = lambda a, c: "1" if c.endswith("| wc -c") else ""
        copy_file_and_unlink(args, str(src), triggered_by="post_app")

    with open(args.flush_log) as f:
        log_content = f.read()
    assert "post-app" in log_content


# ---------------------------------------------------------------------------
# move_files_os — check triggered_by is forwarded
# ---------------------------------------------------------------------------


def test_move_files_os_passes_triggered_by(tmp_path):
    args = _make_args(tmp_path)
    args.flush_call = "cp"

    with (
        patch("ftio.api.gekkoFs.posix_control.get_files", return_value=[]),
        patch("ftio.api.gekkoFs.posix_control.get_items_to_submit", return_value=[]),
        patch("ftio.api.gekkoFs.posix_control.flush_using_cp") as mock_cp,
    ):
        move_files_os(args, triggered_by="post_app")
        mock_cp.assert_called_once()
        _, _, _, tb = mock_cp.call_args[0]
        assert tb == "post_app"


def test_move_files_os_tar_passes_triggered_by(tmp_path):
    args = _make_args(tmp_path)
    args.flush_call = "tar"

    with (
        patch("ftio.api.gekkoFs.posix_control.get_files", return_value=[]),
        patch("ftio.api.gekkoFs.posix_control.get_items_to_submit", return_value=[]),
        patch("ftio.api.gekkoFs.posix_control.flush_using_tar") as mock_tar,
    ):
        move_files_os(args, triggered_by="ftio")
        mock_tar.assert_called_once()
        _, _, tb = mock_tar.call_args[0]
        assert tb == "ftio"
