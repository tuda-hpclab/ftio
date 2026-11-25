"""Tests for item selection in the GekkoFS stage-out / flush path.

Folder mode filters by regex. With no regex it selects nothing, which used to be
indistinguishable from a fast flush in the log. These tests pin the selection
rules that `move_files_os` relies on.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from ftio.api.gekkoFs import posix_control
from ftio.api.gekkoFs.posix_control import (
    get_files,
    get_items_to_submit,
    newest_mtime,
)

MNT = "/mnt/gkfs"
CHECKPOINTS = [f"{MNT}/ckpt.restart.{step}" for step in (10, 20, 30)]

# AMReX (Castro) writes each checkpoint as a directory tree with a dot-free
# name; the correct flush unit is the top-level directory, not Level_0.
CASTRO_RE = r".*/sedov_3d_sph_chk\d+"
CASTRO_CHK = f"{MNT}/sedov_3d_sph_chk00010"
CASTRO_FILES = [
    f"{CASTRO_CHK}/Header",
    f"{CASTRO_CHK}/Level_0/Cell_D_00000",
    f"{CASTRO_CHK}/Level_0/Cell_H",
]
# WarpX writes <mnt>/chk<step>/...
WARPX_RE = r".*/chk\d+"
WARPX_CHK = f"{MNT}/chk000010"
WARPX_FILES = [
    f"{WARPX_CHK}/WarpXHeader",
    f"{WARPX_CHK}/Level_0/Cell_D_00000",
]


def _args(regex: str | None) -> Namespace:
    return Namespace(regex=regex, debug=False)


def test_whole_folder_is_submitted_when_every_file_matches():
    items = get_items_to_submit(CHECKPOINTS, _args(r".*/ckpt\.restart\.\d+$"), "folder")
    assert items == [MNT]


def test_only_matching_files_are_submitted_when_the_folder_is_mixed():
    files = [*CHECKPOINTS, f"{MNT}/log.lammps"]
    items = get_items_to_submit(files, _args(r".*/ckpt\.restart\.\d+$"), "folder")
    assert items == CHECKPOINTS


def test_runtime_flush_never_collapses_to_the_mount_root():
    # LAMMPS writes every checkpoint straight into the mount root. Collapsed to the
    # root, the newest file is always < 5 s old and every runtime flush is skipped.
    args = Namespace(regex=r".*/ckpt\.restart\.\d+$", debug=False, gkfs_mntdir=MNT)
    items = get_items_to_submit(CHECKPOINTS, args, "folder", collapse_root=False)
    assert items == CHECKPOINTS


def test_runtime_flush_still_collapses_subfolders():
    sub = f"{MNT}/epoch1"
    files = [f"{sub}/ckpt.restart.{s}" for s in (1, 2)]
    args = Namespace(regex=r".*/ckpt\.restart\.\d+$", debug=False, gkfs_mntdir=f"{MNT}/")
    assert get_items_to_submit(files, args, "folder", collapse_root=False) == [sub]


def test_move_files_os_disables_root_collapse_only_for_runtime(monkeypatch):
    seen = []

    def fake_items(files, args, mode, collapse_root=True, exclude=None):
        seen.append(collapse_root)
        return []

    monkeypatch.setattr(posix_control, "get_files", lambda args: [])
    monkeypatch.setattr(posix_control, "stat_files", lambda args: {})
    monkeypatch.setattr(posix_control, "get_items_to_submit", fake_items)
    monkeypatch.setattr(posix_control, "flush_using_cp", lambda *a, **k: None)
    args = Namespace(
        ld_preload="",
        stage_out_path="/nonexistent-stage-out",
        parallel_move_threads=0,
        debug=False,
        regex=None,
        gkfs_mntdir=MNT,
        flush_call="cp",
    )
    monkeypatch.setattr(posix_control.os, "makedirs", lambda *a, **k: None)
    posix_control.move_files_os(args, triggered_by="ftio")
    posix_control.move_files_os(args, triggered_by="post_app")
    assert seen == [False, True]


def test_folder_mode_without_a_regex_selects_nothing():
    # This is why an empty regex_flush_match silently disables the async flush.
    assert get_items_to_submit(CHECKPOINTS, _args(None), "folder") == []
    assert get_items_to_submit(CHECKPOINTS, _args(""), "folder") == []


def test_file_mode_ignores_the_regex_and_submits_everything():
    assert get_items_to_submit(CHECKPOINTS, _args(None), "files") == CHECKPOINTS


# ── AMReX directory checkpoints (Defect B) ──────────────────────────────────


def test_amrex_castro_checkpoint_collapses_to_top_level_dir():
    # All three dot-free files live under one checkpoint dir; the flush unit is
    # the top-level directory, never its Level_0 parent.
    items = get_items_to_submit(CASTRO_FILES, _args(CASTRO_RE), "folder")
    assert items == [CASTRO_CHK]


def test_amrex_warpx_checkpoint_collapses_to_top_level_dir():
    items = get_items_to_submit(WARPX_FILES, _args(WARPX_RE), "folder")
    assert items == [WARPX_CHK]


def test_amrex_multiple_checkpoints_each_become_one_dir():
    chk20 = f"{MNT}/sedov_3d_sph_chk00020"
    files = [
        *CASTRO_FILES,
        f"{chk20}/Header",
        f"{chk20}/Level_0/Cell_D_00000",
    ]
    items = get_items_to_submit(files, _args(CASTRO_RE), "folder")
    assert items == [CASTRO_CHK, chk20]


def test_amrex_ignores_non_matching_siblings():
    files = [*CASTRO_FILES, f"{MNT}/inputs", f"{MNT}/run.log"]
    items = get_items_to_submit(files, _args(CASTRO_RE), "folder")
    assert items == [CASTRO_CHK]


# ── existing app patterns keep their selection (regression guard) ────────────


def test_nek_files_in_mount_still_collapse_to_the_mount_dir():
    nek_re = r".*/[a-zA-Z0-9]*turbPipe0\.f\d+"
    files = [f"{MNT}/turbPipe0.f{n:05d}" for n in (1, 2, 3)]
    assert get_items_to_submit(files, _args(nek_re), "folder") == [MNT]


def test_dlio_checkpoint_folder_is_still_collapsed():
    dlio_re = r".*/(checkpoints)/.*"
    epoch = f"{MNT}/checkpoints/epoch1"
    files = [f"{epoch}/model.pt", f"{epoch}/optim.pt"]
    # Every file in the epoch folder matches, so the folder is one flush unit.
    assert get_items_to_submit(files, _args(dlio_re), "folder") == [epoch]


def test_dlio_pattern_ignores_unmatched_files_outside_checkpoints():
    dlio_re = r".*/(checkpoints)/.*"
    epoch = f"{MNT}/checkpoints/epoch1"
    files = [f"{epoch}/model.pt", f"{epoch}/optim.pt", f"{MNT}/train.log"]
    # The two epoch files collapse to their folder; train.log (unmatched,
    # outside checkpoints) is never selected.
    assert get_items_to_submit(files, _args(dlio_re), "folder") == [epoch]


def test_qmcpack_single_config_file_is_the_flush_unit():
    qmc_re = r".*/glass_heg\.s\d+\.config\.h5$"
    config = f"{MNT}/glass_heg.s000.config.h5"
    files = [config, f"{MNT}/qmc.in.xml"]
    assert get_items_to_submit(files, _args(qmc_re), "folder") == [config]


# ── get_files no longer drops dot-free AMReX files (Defect B) ────────────────


def _files_args() -> Namespace:
    return Namespace(
        gkfs_mntdir=MNT, debug=False, host_file=None, node=None, ld_preload=None
    )


def test_get_files_keeps_dot_free_amrex_files(monkeypatch):
    find_output = "\n".join(CASTRO_FILES) + "\n"

    def fake_call(args, call):
        assert call == f"find {MNT} -type f"
        return find_output

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    assert get_files(_files_args()) == CASTRO_FILES


def test_get_files_returns_dotted_files_too(monkeypatch):
    out = "\n".join([*CHECKPOINTS, f"{MNT}/log.lammps"]) + "\n"
    monkeypatch.setattr(posix_control, "preloaded_call", lambda a, c: out)
    assert get_files(_files_args()) == [*CHECKPOINTS, f"{MNT}/log.lammps"]


def test_get_files_falls_back_to_ls_r_without_dot_heuristic(monkeypatch):
    ls_output = f"{MNT}:\n" "Header\n" "Cell_D_00000\n" "\n" "LIBGKFS noise line\n"

    def fake_call(args, call):
        if call.startswith("find"):
            raise RuntimeError("find not available")
        return ls_output

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    # Directory header (":"), blank line and LIBGKFS noise are dropped; the
    # dot-free file entries survive.
    assert get_files(_files_args()) == [f"{MNT}/Header", f"{MNT}/Cell_D_00000"]


# ── directory mtime uses the newest descendant, not the dir's own mtime ──────


def test_newest_mtime_uses_newest_descendant_for_a_directory(monkeypatch):
    def fake_call(args, call):
        # `test -d` must never be used: GekkoFS has no directory inodes, so it
        # answers "not a directory" for every path inside the mount.
        assert not call.startswith("test -d"), "test -d is unreliable under gekko"
        if call.startswith("find"):
            return "100.0\n300.5\n200.0\n"
        raise AssertionError(f"unexpected call: {call}")

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    monkeypatch.setattr(
        posix_control,
        "get_modification_time",
        lambda a, i: pytest.fail("dir mtime should come from descendants"),
    )
    assert newest_mtime(_files_args(), CASTRO_CHK) == 300.5


def test_newest_mtime_uses_file_mtime_for_a_regular_file(monkeypatch):
    # `find <file> -type f` matches the file itself, so one command covers both.
    monkeypatch.setattr(posix_control, "preloaded_call", lambda a, c: "42.0\n")
    monkeypatch.setattr(
        posix_control,
        "get_modification_time",
        lambda a, i: pytest.fail("find already reported the file's mtime"),
    )
    assert newest_mtime(_files_args(), f"{MNT}/ckpt.restart.10") == 42.0


def test_newest_mtime_uses_ctime_when_gekko_reports_no_mtime(monkeypatch):
    # GekkoFS does not maintain mtime: it answers 0 for every file in the mount,
    # so `time.time() - 0` looked like an age of ~1.78e9 s and the "too new to
    # flush" guard never fired. gkfs does keep ctime, and it advances on write.
    def fake_call(args, call):
        assert "%C@" in call, "ctime must be requested, gekko has no mtime"
        return "0.0 1783932771.0\n0.0 1783932775.0\n"

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    assert newest_mtime(_files_args(), CASTRO_CHK) == 1783932775.0


def test_newest_mtime_prefers_mtime_on_a_normal_filesystem(monkeypatch):
    # Outside the mount mtime is the meaningful stamp; taking the max of the two
    # keeps that correct.
    monkeypatch.setattr(
        posix_control, "preloaded_call", lambda a, c: "300.0 250.0\n100.0 90.0\n"
    )
    assert newest_mtime(_files_args(), CASTRO_CHK) == 300.0


def test_newest_mtime_falls_back_for_empty_directory(monkeypatch):
    def fake_call(args, call):
        if call.startswith("find"):
            return "\n"  # no files
        raise AssertionError(f"unexpected call: {call}")

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    monkeypatch.setattr(posix_control, "get_modification_time", lambda a, i: 7.0)
    assert newest_mtime(_files_args(), CASTRO_CHK) == 7.0


def test_emptied_directory_reads_as_brand_new_so_it_is_not_restaged(monkeypatch):
    # After a flush, gekko can still hand back stale directory metadata with a
    # zero timestamp. Treating that as "very old" restaged an already-staged
    # WarpX checkpoint a second time. Report it as new: nothing is left anyway.
    monkeypatch.setattr(posix_control, "preloaded_call", lambda a, c: "\n")
    monkeypatch.setattr(posix_control, "get_modification_time", lambda a, i: 0.0)
    age = time.time() - newest_mtime(_files_args(), CASTRO_CHK)
    assert age < 1.0, "an emptied directory must not look stale"


# ── an already-staged, unchanged item is not staged twice ────────────────────


def test_already_flushed_skips_an_unchanged_item():
    q = posix_control.FileQueue()
    q.mark_flushed(CASTRO_CHK, 100.0)
    # Gekko re-lists the stale metadata with the same (or an older) stamp.
    assert q.already_flushed(CASTRO_CHK, 100.0) is True
    assert q.already_flushed(CASTRO_CHK, 99.0) is True


def test_already_flushed_lets_a_rewritten_item_through():
    q = posix_control.FileQueue()
    q.mark_flushed(CASTRO_CHK, 100.0)
    # The application genuinely wrote the path again: it must be staged.
    assert q.already_flushed(CASTRO_CHK, 101.0) is False


def test_an_item_never_staged_is_not_skipped():
    q = posix_control.FileQueue()
    assert q.already_flushed(CASTRO_CHK, 100.0) is False


def test_move_item_does_not_recopy_an_already_staged_checkpoint(tmp_path, monkeypatch):
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(posix_control, "newest_mtime", lambda a, i: 100.0)
    monkeypatch.setattr(posix_control.time, "time", lambda: 200.0)

    args = _copy_args(tmp_path)
    args.ignore_mtime = False
    posix_control.files_in_progress.mark_flushed(CASTRO_CHK, 100.0)
    try:
        posix_control.move_item(args, CASTRO_CHK)
        assert not calls, "an unchanged, already-staged item must not be copied again"
    finally:
        posix_control.files_in_progress._flushed.pop(CASTRO_CHK, None)


# ── copy/delete commands work for both files and AMReX directories ───────────


def _is_count(call: str) -> bool:
    """The pre/post-copy file count copy_file_and_unlink uses to verify the copy."""
    return call.endswith("| wc -c")


def _capture_calls(monkeypatch) -> list[str]:
    """Record the shell commands copy_file_and_unlink issues."""
    calls: list[str] = []

    def fake_call(args, call):
        calls.append(call)
        # Same count either side of the copy, so the copy verifies as complete.
        return "2" if _is_count(call) else ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    return calls


def _copy_args(tmp_path) -> Namespace:
    return Namespace(
        gkfs_mntdir=MNT,
        stage_out_path=str(tmp_path),
        debug=False,
        host_file=None,
        node=None,
        ld_preload=None,
        flush_log="",
    )


def test_copy_uses_recursive_cp_so_amrex_dirs_are_not_omitted(tmp_path, monkeypatch):
    # `cp -L <dir>` fails with "omitting directory"; -r copies plain files too.
    calls = _capture_calls(monkeypatch)
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)
    assert any(c.startswith(f"cp -rL {CASTRO_CHK} ") for c in calls)
    assert not any(c.startswith(f"cp -L {CASTRO_CHK} ") for c in calls)


def test_whole_mount_flush_copies_contents_into_the_stage_out_path(tmp_path, monkeypatch):
    # When every file matches, the flush unit collapses to the mount itself. Then
    # dst == stage_out_path, so copying into dirname(dst) recreated the mount under
    # the stage dir: LAMMPS' last checkpoint landed in <stage>/tarraf_gkfs_mountdir/
    # while the flush log claimed <stage>/output/.
    calls: list[str] = []

    def fake_call(args, call):
        calls.append(call)
        if "-printf '%P" in call:
            return "ckpt.1\nckpt.2\nsub/ckpt.3\n"
        return "3" if _is_count(call) else ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    args = _copy_args(tmp_path)
    args.parallel_move_threads = 2
    posix_control.copy_file_and_unlink(args, MNT)
    # Each `cp` must be its own preloaded call: a `cd <mount> && find | xargs cp`
    # line only preloads the `cd`, which then fails outside GekkoFS (every
    # whole-mount stage-out 2026-09-29..10-15 failed that way).
    cp = sorted(c for c in calls if c.startswith("cp "))
    assert cp == [
        f"cp -L {MNT}/ckpt.1 {MNT}/ckpt.2 -t {tmp_path}/",
        f"cp -L {MNT}/sub/ckpt.3 -t {tmp_path}/sub",
    ]
    assert (tmp_path / "sub").is_dir()
    assert not any(c.startswith("cd ") for c in calls)
    assert not any(
        c.startswith(f"cp -rL {MNT} ") for c in calls
    ), "copying the mount itself nests it under the stage dir"


def test_a_single_item_still_copies_into_its_parent(tmp_path, monkeypatch):
    # The non-collapsed case must keep intermediate dirs: .../checkpoints/epoch10
    # lands at <stage>/checkpoints/epoch10, not <stage>/epoch10.
    calls = _capture_calls(monkeypatch)
    epoch = f"{MNT}/checkpoints/epoch10"
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), epoch)
    cp = [c for c in calls if c.startswith("cp ")]
    assert cp == [f"cp -rL {epoch} {tmp_path}/checkpoints"]


def test_copy_never_probes_test_d(tmp_path, monkeypatch):
    calls = _capture_calls(monkeypatch)
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)
    assert not any(c.startswith("test -d") for c in calls)


def test_delete_never_forks_per_file(tmp_path, monkeypatch):
    # Every fork re-initialises the gekko client, and that is the entire cost of
    # deleting: on a 54-file AMReX checkpoint `-exec unlink {} \;` took 34.15 s,
    # `-exec rm -f {} +` 1.22 s and `-delete` 0.72 s. `-delete` unlinks from find
    # itself, so it must not degrade into a per-file `-exec`.
    calls = _capture_calls(monkeypatch)
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)
    removes = [
        c
        for c in calls
        if c.startswith("find") and not _is_count(c) and c != f"find {CASTRO_CHK}"
    ]
    assert removes, "no delete command issued"
    assert removes[0] == f"find {CASTRO_CHK} -type f -delete"
    assert r"\;" not in removes[0], "must never fork once per file"


def test_delete_falls_back_to_a_batched_rm(tmp_path, monkeypatch):
    # A find without -delete, or a path a concurrent flush already removed.
    calls: list[str] = []

    def fake_call(args, call):
        calls.append(call)
        if call.endswith("-delete"):
            raise RuntimeError("find: unknown predicate `-delete'")
        return "2" if _is_count(call) else ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    monkeypatch.setattr(
        posix_control.files_in_progress,
        "put_ignore",
        lambda a, i: pytest.fail("a successful fallback must not be ignore-queued"),
    )
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)
    assert calls[-1] == f"find {CASTRO_CHK} -type f -exec rm -f {{}} +"


def test_listing_retries_while_the_mount_is_busy(monkeypatch):
    # A flush in flight makes gekko answer EBUSY. Raising here aborted the whole
    # post-app stage-out and lost every checkpoint still in the mount.
    monkeypatch.setattr(posix_control.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def fake_call(args, call):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Device or resource busy")
        return "\n".join(CASTRO_FILES) + "\n"

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    assert get_files(_files_args()) == CASTRO_FILES
    assert calls["n"] == 3, "the busy listing should have been retried"


def test_listing_gives_up_after_the_retries_are_exhausted(monkeypatch):
    monkeypatch.setattr(posix_control.time, "sleep", lambda _: None)

    def always_busy(args, call):
        raise RuntimeError("Device or resource busy")

    monkeypatch.setattr(posix_control, "preloaded_call", always_busy)
    # Both find and the ls -R fallback exhausting their retries used to raise
    # and kill the whole flush (losing everything still in the mount); it now
    # degrades to "nothing to move this cycle" instead.
    assert get_files(_files_args()) == []


def test_item_is_ignore_queued_only_when_both_deletes_fail(tmp_path, monkeypatch):
    def fake_call(args, call):
        if _is_count(call):
            return "2"
        if call.startswith("find"):
            raise RuntimeError("gekko is wedged")
        return ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    ignored: list[str] = []
    monkeypatch.setattr(
        posix_control.files_in_progress,
        "put_ignore",
        lambda a, i: ignored.append(i),
    )
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)
    assert ignored == [CASTRO_CHK]


def test_renamed_dir_is_copied_from_the_pre_rename_prefix(tmp_path, monkeypatch):
    # GekkoFS renames only the directory entry: after AMReX moves <chk>.temp/ to
    # <chk>/, `cp -rL <chk>` cannot stat a single child, but `find <chk>` still
    # walks the tree and the data reads back under <chk>.temp/.
    calls: list[str] = []

    def fake_call(args, call):
        calls.append(call)
        if _is_count(call):
            # The orphaned source counts 0 through `find -type f`; the fix must
            # take n_src from the recovered listing, not from this.
            return "0" if MNT in call else "3"
        if call.startswith(f"cp -rL {CASTRO_CHK} "):
            raise RuntimeError("cp: cannot stat 'Level_0': No such file or directory")
        if call == f"find {CASTRO_CHK}":
            return "\n".join([CASTRO_CHK, f"{CASTRO_CHK}/Level_0", *CASTRO_FILES])
        return ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)

    dst = f"{tmp_path}/sedov_3d_sph_chk00010"
    copies = [c for c in calls if c.startswith("cp -L ")]
    assert len(copies) == 1
    for rel in ("Header", "Level_0/Cell_D_00000", "Level_0/Cell_H"):
        assert f"cp -L {CASTRO_CHK}.temp/{rel} {dst}/{rel}" in copies[0]
        assert os.path.isdir(os.path.dirname(f"{dst}/{rel}"))
    # Level_0 is a directory in the listing and must not be copied as a file.
    assert f"{CASTRO_CHK}.temp/Level_0 " not in copies[0]
    # `find -delete` cannot reach the orphans; they go by explicit path.
    removes = [c for c in calls if c.startswith("rm -f ")]
    assert removes and f"{CASTRO_CHK}.temp/Header" in removes[0]


def test_a_copy_failure_that_is_not_a_rename_still_raises(tmp_path, monkeypatch):
    def fake_call(args, call):
        if _is_count(call):
            return "2"
        if call.startswith("cp -rL "):
            raise RuntimeError("gekko is wedged")
        if call.startswith("find "):
            return CASTRO_CHK  # listing has no children to recover
        return ""

    monkeypatch.setattr(posix_control, "preloaded_call", fake_call)
    with pytest.raises(RuntimeError, match="wedged"):
        posix_control.copy_file_and_unlink(_copy_args(tmp_path), CASTRO_CHK)


def _flush_args() -> Namespace:
    return Namespace(parallel_move_threads=2, debug=False)


def test_post_app_flush_waits_out_a_slow_item_instead_of_timing_out(monkeypatch):
    # STAGE_OUT_FLUSH_TIMEOUT exists to stop a *live* trigger from hanging against
    # a wedged daemon. Applying it to post_app too made the final, must-complete
    # stage-out give up early and silently report a fast time with checkpoints
    # still missing (caught on WRF 9N: 209/512 files "STAGE-OUT INCOMPLETE").
    monkeypatch.setattr(posix_control, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(posix_control, "STAGE_OUT_FLUSH_TIMEOUT", 0.05)
    monkeypatch.setattr(posix_control, "_daemon_degraded", False)
    monkeypatch.setattr(posix_control, "_consecutive_full_failures", 0)

    done: list[str] = []
    monkeypatch.setattr(
        posix_control, "move_item", lambda a, i, p, t: (time.sleep(0.2), done.append(i))
    )
    critical: list[str] = []
    monkeypatch.setattr(posix_control.TRIGGER_LOGGER, "critical", critical.append)

    posix_control.flush_using_cp(
        _flush_args(), ["/mnt/gkfs/final_ckpt"], 1.0, triggered_by="post_app"
    )

    assert not any("STAGE-OUT TIMEOUT" in m for m in critical)
    assert done == ["/mnt/gkfs/final_ckpt"], "post_app must wait for the slow copy"


def test_live_flush_gives_up_on_a_slow_item_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(posix_control, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(posix_control, "STAGE_OUT_FLUSH_TIMEOUT", 0.05)
    monkeypatch.setattr(posix_control, "_daemon_degraded", False)
    monkeypatch.setattr(posix_control, "_consecutive_full_failures", 0)

    monkeypatch.setattr(posix_control, "move_item", lambda a, i, p, t: time.sleep(0.5))
    critical: list[str] = []
    monkeypatch.setattr(posix_control.TRIGGER_LOGGER, "critical", critical.append)

    posix_control.flush_using_cp(
        _flush_args(), ["/mnt/gkfs/live_ckpt"], 1.0, triggered_by="ftio"
    )

    assert any("STAGE-OUT TIMEOUT" in m for m in critical)


# --- FTIO in-flight registry (jit stages out alongside FTIO's last copies) ---


def test_inflight_registry_adds_and_removes_items(tmp_path):
    registry = posix_control.inflight_registry(str(tmp_path / "flush.log"))
    assert registry.endswith("flush.log.inflight")
    assert posix_control.read_inflight(registry) == []  # missing file = nothing
    posix_control._update_inflight(registry, [CHECKPOINTS[0], CHECKPOINTS[1]], add=True)
    posix_control._update_inflight(registry, [CHECKPOINTS[0]], add=True)  # no dup
    assert posix_control.read_inflight(registry) == CHECKPOINTS[:2]
    posix_control._update_inflight(registry, [CHECKPOINTS[0]], add=False)
    assert posix_control.read_inflight(registry) == [CHECKPOINTS[1]]
    posix_control.reset_inflight(registry)
    assert posix_control.read_inflight(registry) == []
    # written via tmp + rename: no temp file is left next to the registry
    assert not [p for p in os.listdir(tmp_path) if ".tmp." in p]


def test_move_item_unregisters_even_when_the_copy_fails(tmp_path, monkeypatch):
    args = Namespace(flush_log=str(tmp_path / "flush.log"))
    registry = posix_control.inflight_registry(args.flush_log)
    posix_control._update_inflight(registry, [CHECKPOINTS[0]], add=True)

    def boom(*a, **k):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(posix_control, "_move_item", boom)
    with pytest.raises(RuntimeError):
        posix_control.move_item(args, CHECKPOINTS[0], triggered_by="ftio")
    assert posix_control.read_inflight(registry) == []


def test_an_item_queued_past_app_end_goes_back_to_jit(tmp_path, monkeypatch):
    # jit removes the app-start flag at app end and waits for the registry to
    # empty: a queued item must leave it at once instead of being copied late.
    args = Namespace(
        flush_log=str(tmp_path / "flush.log"),
        app_start_file=str(tmp_path / "app_start.flag"),
    )
    registry = posix_control.inflight_registry(args.flush_log)
    posix_control._update_inflight(registry, [CHECKPOINTS[0]], add=True)
    calls = []
    monkeypatch.setattr(posix_control, "_move_item", lambda *a, **k: calls.append(a))
    posix_control.move_item(args, CHECKPOINTS[0], triggered_by="ftio")
    assert not calls
    assert posix_control.read_inflight(registry) == []


def test_an_item_queued_while_the_app_runs_is_copied(tmp_path, monkeypatch):
    flag = tmp_path / "app_start.flag"
    flag.touch()
    args = Namespace(flush_log=str(tmp_path / "flush.log"), app_start_file=str(flag))
    calls = []
    monkeypatch.setattr(posix_control, "_move_item", lambda *a, **k: calls.append(a))
    posix_control.move_item(args, CHECKPOINTS[0], triggered_by="ftio")
    assert len(calls) == 1


def test_post_app_flushes_are_never_registered(tmp_path):
    args = Namespace(flush_log=str(tmp_path / "flush.log"))
    assert posix_control._registry_of(args, "post_app") == ""
    assert posix_control._registry_of(args, "ftio").endswith(".inflight")


def test_exclude_list_keeps_inflight_items_and_blocks_root_collapse():
    # Every file in the root matches, which would collapse to the whole mount and
    # copy+delete the in-flight checkpoint too. With it excluded, only the rest go.
    args = _args(r".*/ckpt\.restart\.\d+")
    items = get_items_to_submit(
        CHECKPOINTS, args, "folder", collapse_root=True, exclude=[CHECKPOINTS[1]]
    )
    assert items == [CHECKPOINTS[0], CHECKPOINTS[2]]


def test_exclude_list_skips_a_checkpoint_dir_holding_an_inflight_item():
    args = _args(CASTRO_RE)
    assert get_items_to_submit(CASTRO_FILES, args, "folder", exclude=[CASTRO_CHK]) == []


# ── --defer_delete: copy during the run, keep the source, skip it afterwards ──


def _is_delete(call: str) -> bool:
    return call.startswith("find") and ("-delete" in call or "rm -f" in call)


def test_defer_delete_copies_but_never_deletes_during_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(posix_control.files_in_progress, "_flushed", {})
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(posix_control, "newest_mtime", lambda a, i: 100.0)
    monkeypatch.setattr(posix_control.time, "time", lambda: 200.0)
    args = _copy_args(tmp_path)
    args.ignore_mtime = False
    args.defer_delete = True
    args.flush_log = str(tmp_path / "flush.log")
    posix_control.move_item(args, CASTRO_CHK)
    assert any(c.startswith(f"cp -rL {CASTRO_CHK} ") for c in calls)
    assert not any(_is_delete(c) for c in calls), "the source must stay in the mount"
    copied = posix_control.read_copied(posix_control.copied_registry(args.flush_log))
    assert copied == {CASTRO_CHK: 100.0}
    assert "delete: 0.000 s" in (tmp_path / "flush.log").read_text()


def test_defer_delete_skips_a_kept_item_unless_it_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(posix_control.files_in_progress, "_flushed", {})
    calls = _capture_calls(monkeypatch)
    stamp = {"t": 100.0}
    monkeypatch.setattr(posix_control, "newest_mtime", lambda a, i: stamp["t"])
    monkeypatch.setattr(posix_control.time, "time", lambda: 200.0)
    args = _copy_args(tmp_path)
    args.ignore_mtime = False
    args.defer_delete = True
    args.flush_log = str(tmp_path / "flush.log")
    reg = posix_control.copied_registry(args.flush_log)
    posix_control._record_copied(reg, CASTRO_CHK, 100.0)
    posix_control.move_item(args, CASTRO_CHK)
    assert not calls, "a kept, unchanged item must not be copied again"
    stamp["t"] = 150.0  # the app rewrote it
    posix_control.move_item(args, CASTRO_CHK)
    assert any(c.startswith(f"cp -rL {CASTRO_CHK} ") for c in calls)


def test_post_app_stage_out_still_deletes_with_defer_delete(tmp_path, monkeypatch):
    # Only FTIO's runtime flushes keep the source; jit's own pass is unchanged.
    monkeypatch.setattr(posix_control.files_in_progress, "_flushed", {})
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(posix_control, "newest_mtime", lambda a, i: 100.0)
    monkeypatch.setattr(posix_control.time, "time", lambda: 200.0)
    args = _copy_args(tmp_path)
    args.ignore_mtime = True
    args.defer_delete = True
    args.flush_log = str(tmp_path / "flush.log")
    posix_control.move_item(args, CASTRO_CHK, triggered_by="post_app")
    assert any(_is_delete(c) for c in calls)
    assert posix_control.read_copied(posix_control.copied_registry(args.flush_log)) == {}


def test_copied_registry_reset_and_missing(tmp_path):
    reg = posix_control.copied_registry(str(tmp_path / "flush.log"))
    assert posix_control.read_copied(reg) == {}
    posix_control._record_copied(reg, "/mnt/a b/ckpt.1", 5.0)
    assert posix_control.read_copied(reg) == {"/mnt/a b/ckpt.1": 5.0}
    posix_control.reset_copied(reg)
    assert posix_control.read_copied(reg) == {}
    assert posix_control.copied_registry("") == ""


# ── --flush_batch: several files per cp call ────────────────────────────────


def _batch_args(tmp_path, **kw) -> Namespace:
    mnt, out = tmp_path / "mnt", tmp_path / "out"
    mnt.mkdir()
    out.mkdir()
    base = {
        "gkfs_mntdir": str(mnt),
        "stage_out_path": str(out),
        "flush_log": str(tmp_path / "flush.log"),
        "defer_delete": False,
        "flush_batch": 4,
        "debug": False,
    }
    base.update(kw)
    return Namespace(**base)


def _shell(monkeypatch):
    """Run the GekkoFS client calls as plain shell commands on the local disk."""
    import subprocess

    calls = []

    def call(args, cmd):
        calls.append(cmd)
        return subprocess.check_output(cmd, shell=True, text=True)

    monkeypatch.setattr(posix_control, "preloaded_call", call)
    return calls


def test_batches_group_files_per_directory_and_spread_over_workers():
    items = [f"/m/a/f{i}" for i in range(6)] + ["/m/b/g0", "/m/b/g1"]
    args = Namespace(flush_batch=4)
    chunks = posix_control._batches(args, items, list(range(8)), workers=4)
    # ceil(8 / 4) = 2 per batch; no batch mixes directories
    assert all(len(c) <= 2 for c in chunks)
    assert sorted(i for c in chunks for i in c) == list(range(8))
    for c in chunks:
        assert len({os.path.dirname(items[i]) for i in c}) == 1


def test_batches_off_keeps_one_item_per_future():
    args = Namespace(flush_batch=0)
    assert posix_control._batches(args, ["/m/a", "/m/b"], [0, 1], workers=8) == [[0], [1]]


def test_move_batch_copies_with_one_cp_and_unlinks_the_sources(tmp_path, monkeypatch):
    args = _batch_args(tmp_path)
    calls = _shell(monkeypatch)
    items = []
    for i in range(3):
        f = tmp_path / "mnt" / f"mpi.1-Part{i:08d}-of-00000003.data"
        f.write_bytes(os.urandom(1000 + i))
        items.append(str(f))
    posix_control.move_batch(args, items, "ftio")
    for i, item in enumerate(items):
        dst = tmp_path / "out" / os.path.basename(item)
        assert dst.stat().st_size == 1000 + i
        assert not os.path.exists(item)
    assert sum(c.startswith("cp ") for c in calls) == 1
    assert sum(c.startswith("rm ") for c in calls) == 1
    log = (tmp_path / "flush.log").read_text().splitlines()
    assert len(log) == 3 and all("FTIO-trigger" in line for line in log)


def test_move_batch_keeps_sources_with_defer_delete_and_skips_a_repeat(
    tmp_path, monkeypatch
):
    args = _batch_args(tmp_path, defer_delete=True)
    calls = _shell(monkeypatch)
    items = []
    for i in range(2):
        f = tmp_path / "mnt" / f"ckpt{i}"
        f.write_bytes(b"x" * 10)
        items.append(str(f))
    posix_control.move_batch(args, items, "ftio")
    assert all(os.path.exists(i) for i in items)
    assert not any(c.startswith("rm ") for c in calls)
    copied = posix_control.read_copied(posix_control.copied_registry(args.flush_log))
    assert set(copied) == set(items)
    n_cp = sum(c.startswith("cp ") for c in calls)
    posix_control.move_batch(args, items, "ftio")  # unchanged -> nothing copied
    assert sum(c.startswith("cp ") for c in calls) == n_cp


def test_move_batch_refuses_a_directory_and_leaves_everything_in_place(
    tmp_path, monkeypatch
):
    args = _batch_args(tmp_path)
    _shell(monkeypatch)
    f = tmp_path / "mnt" / "file"
    f.write_bytes(b"x")
    d = tmp_path / "mnt" / "chk00010"
    d.mkdir()
    with pytest.raises(RuntimeError, match="not regular files"):
        posix_control.move_batch(args, [str(f), str(d)], "ftio")
    assert f.exists() and d.exists()
    assert not (tmp_path / "out" / "file").exists()


def test_final_stage_out_keeps_source_only_when_asked():
    keep = Namespace(keep_source=True, defer_delete=False)
    plain = Namespace(keep_source=False, defer_delete=True)
    assert posix_control._keeps_source(keep, "post_app")
    assert not posix_control._keeps_source(plain, "post_app")
    assert posix_control._keeps_source(plain, "ftio")
    assert not posix_control._keeps_source(keep, "ftio")


def test_all_node_stage_out_splits_files_over_tasks(tmp_path, monkeypatch):
    mnt, out = "/dev/shm/mnt", tmp_path / "out"
    files = [f"{mnt}/ckpt.{i}" for i in range(10)]
    calls = []

    def fake_run(call, shell, check):
        calls.append(call)
        for f in files:  # every task copied its share
            (out / os.path.basename(f)).write_text("x")

    monkeypatch.setattr(posix_control, "preloaded_call", lambda a, c: "\n".join(files))
    monkeypatch.setattr(posix_control.subprocess, "run", fake_run)
    args = Namespace(
        gkfs_mntdir=mnt,
        stage_out_path=str(out),
        flush_log=str(tmp_path / "shm" / "flush.log"),
        all_nodes=2,
        tasks_per_node=2,
        srun_nodes="--jobid=1 --nodelist=a,b",
        host_file=str(tmp_path / "hosts.txt"),
        ld_preload="lib.so",
        keep_source=True,
        defer_delete=False,
    )
    (tmp_path / "shm").mkdir()
    posix_control.stage_out_all_nodes(args, [mnt])
    # the list sits on the shared FS next to the hosts file; a copy goes to the log dir
    lines = (tmp_path / "stage_out_all_nodes.txt").read_text().splitlines()
    assert (tmp_path / "shm" / "stage_out_all_nodes.txt").is_file()
    # 10 files over 4 tasks: batches of 3, every file listed once
    assert len(lines) == 4
    assert sorted(p for line in lines for p in line.split()[1:]) == sorted(files)
    assert "-N 2 --ntasks-per-node=2" in calls[0]
    assert "--nodelist=a,b" in calls[0]


def test_ready_by_size_copies_a_checkpoint_once_it_has_the_constant_size(
    tmp_path, monkeypatch
):
    listings = iter(
        [
            # cfg.0 and cfg.1 copied (copy + delete) after this listing
            {"/m/cfg.0": (1.0, 100), "/m/cfg.1": (2.0, 100), "/m/cfg.2": (3.0, 40)},
            {"/m/cfg.2": (4.0, 100)},  # full size, but just written
            {"/m/cfg.2": (4.0, 100), "/m/cfg.3": (5.0, 100)},  # stable: ready
        ]
    )
    monkeypatch.setattr(posix_control, "stat_files", lambda a: next(listings))
    args = Namespace(
        regex=r".*/cfg\.\d+$", flush_log=str(tmp_path / "flush.log"), defer_delete=False
    )
    state = {}
    assert posix_control.ready_by_size(args, state) == []
    assert posix_control.ready_by_size(args, state) == []
    # cfg.3 has the size too but was not in the previous listing
    assert posix_control.ready_by_size(args, state) == ["/m/cfg.2"]


def test_ready_by_size_waits_when_sizes_differ(tmp_path, monkeypatch):
    listings = iter(
        [
            {"/m/cfg.0": (1.0, 100), "/m/cfg.1": (2.0, 90), "/m/cfg.2": (3.0, 100)},
            {"/m/cfg.2": (3.0, 100)},
            {"/m/cfg.2": (3.0, 100)},
        ]
    )
    monkeypatch.setattr(posix_control, "stat_files", lambda a: next(listings))
    args = Namespace(regex="", flush_log=str(tmp_path / "flush.log"), defer_delete=False)
    state = {}
    for _ in range(3):
        assert posix_control.ready_by_size(args, state) == []


def test_ftio_trigger_leaves_out_files_already_copied(tmp_path, monkeypatch):
    flush_log = str(tmp_path / "flush.log")
    posix_control._record_copied(posix_control.copied_registry(flush_log), "/m/a", 5.0)
    args = Namespace(flush_log=flush_log, defer_delete=True)
    stats = {"/m/a": (5.0, 1), "/m/b": (6.0, 1)}
    assert posix_control._drop_copied(args, ["/m/a", "/m/b"], stats) == ["/m/b"]
    # changed since its copy: copied again
    stats["/m/a"] = (7.0, 1)
    assert posix_control._drop_copied(args, ["/m/a", "/m/b"], stats) == ["/m/a", "/m/b"]
