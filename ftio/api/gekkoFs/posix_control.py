"""
This module provides functionality for managing file operations in a GekkoFS environment.

The module leverages GekkoFS-specific environment variables and commands to manage file operations
efficiently. It also includes mechanisms for monitoring file modification times and ensuring
compatibility with GekkoFS libraries.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse
import math
import mmap
import os
import re
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

from ftio.api.gekkoFs.file_queue import FileQueue
from ftio.api.gekkoFs.gekko_helper import get_modification_time, preloaded_call
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.logger import Logger

TRIGGER_LOGGER = Logger(prefix="trigger", stream=sys.stdout).get()

files_in_progress = FileQueue()


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} PB"


def _write_flush_log(
    log_file: str,
    item: str,
    dst: str,
    triggered_by: str,
    copy_time: float,
    delete_time: float,
) -> None:
    if not log_file:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    label = "FTIO-trigger" if triggered_by == "ftio" else "post-app    "
    line = (
        f"{ts} | {label} | {item} -> {dst}"
        f" | copy: {copy_time:.3f} s | delete: {delete_time:.3f} s\n"
    )
    try:
        with open(log_file, "a") as f:
            f.write(line)
    except Exception:
        pass


def move_files_os(
    args: argparse.Namespace,
    parallel: bool = False,
    period: float = 0,
    triggered_by: str = "ftio",
) -> None:
    """
    Moves files and directories from the source directory to the stage-out path.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        parallel (bool, optional): Whether to use parallel processing for file moving.
            modification time checks. Defaults to False.
        period (float, optional): Time period for file modification checks. Defaults to 0.
        triggered_by (str): Who initiated the flush — "ftio" (predictor) or "post_app".
    """
    TRIGGER_LOGGER.info("Moving files")
    args.ld_preload = (args.ld_preload or "").replace("libc_", "")
    if not os.path.exists(args.stage_out_path):
        os.makedirs(args.stage_out_path)

    # Iterate over all items in the source directory
    files = get_files(args)
    if args.parallel_move_threads > 0:
        TRIGGER_LOGGER.info(
            f"{len(files)} files flagged to move in parallel with {args.parallel_move_threads} threads"
        )
    else:
        TRIGGER_LOGGER.info(f"{len(files)} files flagged to move")

    if args.debug:
        TRIGGER_LOGGER.debug(f"Files are:\n{files}")

    # Ensure the target directory exists
    os.makedirs(args.stage_out_path, exist_ok=True)
    # Check if to submit files or folders
    # items_to_submit = get_items_to_submit(files, args, "files")
    items_to_submit = get_items_to_submit(files, args, "folder")

    try:
        raw = preloaded_call(args, f"find {args.gkfs_mntdir} -type f -printf '%s\\n'")
        total_bytes = sum(int(x) for x in raw.splitlines() if x.strip().isdigit())
        size_str = format_size(total_bytes)
    except Exception:
        size_str = "unknown size"
    TRIGGER_LOGGER.info(
        f"Staging {len(items_to_submit)} item(s) ({size_str})"
        f" from {args.gkfs_mntdir} → {args.stage_out_path}"
    )
    if files and not items_to_submit:
        # Silently staging nothing looks like a fast flush; it is not. Most often
        # the regex for this mode is empty, which matches no file at all.
        TRIGGER_LOGGER.warning(
            f"{len(files)} file(s) present but the {triggered_by} regex "
            f"({args.regex!r}) selected none — nothing will be staged out"
        )

    if "cp" in args.flush_call:
        flush_using_cp(args, items_to_submit, period, triggered_by)
    else:  # flush using tar
        flush_using_tar(args, items_to_submit, triggered_by)


def flush_using_tar(
    args: argparse.Namespace,
    items_to_submit: list[str] = None,
    triggered_by: str = "ftio",
):
    if items_to_submit is None:
        items_to_submit = []
    flush_log = getattr(args, "flush_log", "")
    tar_dst = f"{args.stage_out_path}/data.tar"
    tar_cmd = f"tar -rf {tar_dst} {' '.join(items_to_submit)}"
    TRIGGER_LOGGER.info(f"Taring {len(items_to_submit)} items to {args.stage_out_path}")
    start = time.time()
    preloaded_call(args, tar_cmd)
    tar_time = time.time() - start
    TRIGGER_LOGGER.info(
        f"Finished taring {len(items_to_submit)} items to {args.stage_out_path}"
    )
    start_del = time.time()
    delete_items(args, items_to_submit)
    delete_time = time.time() - start_del
    TRIGGER_LOGGER.info(
        f"Tar {len(items_to_submit)} items: tarred in {tar_time:.3f} s, deleted in {delete_time:.3f} s"
    )
    summary = f"TAR {len(items_to_submit)} items"
    _write_flush_log(flush_log, summary, tar_dst, triggered_by, tar_time, delete_time)


def flush_using_cp(
    args: argparse.Namespace,
    items_to_submit: list[str],
    period: float,
    triggered_by: str = "ftio",
):
    """
    Submits file processing tasks to a ProcessPoolExecutor and tracks progress, ensuring no
    duplicate or ignored items are processed.
    Args:
        args (argparse.Namespace): Command-line arguments namespace.
        items_to_submit (list[str]): List of items (file paths) that need to be processed.
        period (float): Time interval in seconds between task executions.
        triggered_by (str): Who initiated the flush — "ftio" or "post_app".
    Returns:
        None
    """
    # Step 3: Submit tasks to the executor ((only move the files if they are not
    # already in progress
    futures: dict = {}
    num_workers = args.parallel_move_threads
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, item in enumerate(items_to_submit):
            # check that the item is not in progress and not  not in the ignored list
            if not files_in_progress.in_progress(
                item
            ) and not files_in_progress.is_ignored(args, item):
                files_in_progress.put(item)
                futures[executor.submit(move_item, args, item, period, triggered_by)] = (
                    idx
                )
                if args.debug:
                    TRIGGER_LOGGER.debug(f"Moving {item}")
            else:
                if args.debug:
                    TRIGGER_LOGGER.debug(f"Already moving {item}")

        TRIGGER_LOGGER.info(
            f"Submitted {len(futures)} futures using {num_workers} workers"
        )
        if args.debug:
            TRIGGER_LOGGER.debug(f"Files in progress: {files_in_progress}")

        for future in as_completed(futures):
            try:
                future.result()
                files_in_progress.mark_done(items_to_submit[futures[future]])
            except Exception as e:
                index = futures[future]
                TRIGGER_LOGGER.error(f"{items_to_submit[index]} had an error: {e}")
                files_in_progress.mark_done(items_to_submit[index])
            remaining = len(files_in_progress)
            if remaining:
                TRIGGER_LOGGER.info(f"{remaining} item(s) still in the queue")


def get_items_to_submit(files: list, args: argparse.Namespace, mode: str = "files"):
    """
    Determine which items (files or folders) should be submitted based on
    regex filtering and the submission mode.

    If `mode` includes "folder", each file is matched against `args.regex`:
      - If the regex matches only a proper path prefix that names a directory
        containing the file (an AMReX checkpoint such as Castro/WarpX), that
        top-level checkpoint directory is submitted as a single flush unit.
      - Otherwise the file is a flush unit and the folder-collapse optimisation
        applies: if all files in its immediate parent match, the parent folder
        is submitted; else only the matching files are submitted.

    If `mode` does not include "folder", all provided files are returned.

    Args:
        files (list[str]): List of file paths to consider for submission.
        args (Namespace): Parsed arguments containing at least:
            - regex (str | None): Regex pattern to filter files.
            - debug (bool): Whether to print debug information.
        mode (str, optional): Submission mode. If it contains "folder", directory-level
            grouping and filtering logic is applied.

    Returns:
        list[str]: List of items (file paths or folder paths) to be submitted.
    """
    regex = None
    # Compile the regex pattern if provided
    if args.regex:
        TRIGGER_LOGGER.info(f"Using pattern: {args.regex}")
        regex = re.compile(args.regex)

    if "folder" in mode:
        TRIGGER_LOGGER.info(f"Using regex: {regex}")

        # Step 1: Match each file and split into two kinds of flush units.
        #   - directory unit: the regex matched a proper path prefix that names a
        #     directory *containing* the file (the char after the match is "/").
        #     This is how AMReX checkpoints (Castro/WarpX) are handled: many
        #     dot-free files under <mnt>/sedov_3d_sph_chk00010/... collapse to the
        #     single top-level checkpoint directory, not to their Level_0 parent.
        #   - file unit: the regex matched the whole path, so the file itself is
        #     the flush unit (nek, lammps, dlio, s3d, wrf, qmc, ...).
        dir_units: list[str] = []
        file_matches: list[str] = []
        if regex:
            for file_path in files:
                match = regex.match(file_path)
                if not match:
                    continue
                unit = match.group(0)
                if len(unit) < len(file_path) and file_path[len(unit)] == "/":
                    if unit not in dir_units:
                        dir_units.append(unit)
                        if args.debug:
                            TRIGGER_LOGGER.debug(f"Will move checkpoint dir: {unit}")
                else:
                    file_matches.append(file_path)

        # Step 2: Apply the folder-collapse optimisation to the file units: if
        # every file in an immediate parent folder matches, submit the folder;
        # otherwise submit the matching files individually. Files that belong to
        # a directory unit are never in `matching_set`, so their descendant
        # folders (e.g. .../chk00010/Level_0) are skipped here.
        matching_set = set(file_matches)
        folder_to_all_files: dict[str, list[str]] = {}
        for file_path in files:
            folder = file_path.rsplit("/", 1)[0] if "/" in file_path else "."
            folder_to_all_files.setdefault(folder, []).append(file_path)

        items_to_submit: list[str] = list(dir_units)

        for folder, all_files_in_folder in folder_to_all_files.items():
            matching_files_in_folder = [
                f for f in all_files_in_folder if f in matching_set
            ]
            # If all files in the folder match, submit the folder
            if matching_files_in_folder:
                if set(all_files_in_folder) == set(matching_files_in_folder):
                    items_to_submit.append(folder)
                    if args.debug:
                        TRIGGER_LOGGER.debug(f"Will move folder: {folder}")
                else:
                    items_to_submit.extend(matching_files_in_folder)
                    if args.debug:
                        TRIGGER_LOGGER.debug(
                            f"Will move {len(matching_files_in_folder)} files from {folder}"
                        )
        items_to_submit = drop_nested_items(items_to_submit)
    else:
        items_to_submit = files

    return items_to_submit


def drop_nested_items(items: list[str]) -> list[str]:
    """Drop items that live inside another item, keeping the outermost.

    Folders are grouped by immediate parent, so a checkpoint whose entries are
    themselves directories (TensorFlow writes <epoch>/model_states-N_temp/<shards>)
    yields both the epoch dir and every model_states dir. Staging both copies the
    same bytes twice.

    Args:
        items (list[str]): Items selected for staging.

    Returns:
        list[str]: The items with any descendant of another item removed.
    """
    unique = list(dict.fromkeys(items))  # de-duplicate, keeping the caller's order
    return [
        item
        for item in unique
        if not any(item.startswith(f"{other}/") for other in unique if other != item)
    ]


def newest_mtime(args: argparse.Namespace, item: str) -> float:
    """
    Return the newest modification time (epoch seconds) among an item's files.

    A plain stat on a directory reports the directory's own mtime, which only
    advances when entries are added or removed — not while a descendant file is
    being written. For an AMReX checkpoint that is still being filled, that would
    look "old" and be flushed mid-write. So for directories we take the newest
    mtime over all descendant files instead. For regular files this is just the
    file's own mtime.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        item (str): Path of the file or directory to inspect.

    Returns:
        float: Newest modification time in epoch seconds.
    """
    # `test -d` cannot be used to tell a directory apart here: GekkoFS has no
    # real directory inodes, so it reports false for every path inside the mount.
    # `find <item> -type f` covers both cases -- for a regular file it simply
    # prints that file's own timestamps.
    #
    # Ask for mtime *and* ctime and take the newest of the two. GekkoFS does not
    # maintain mtime: it answers 0 (epoch) for every file in the mount, which made
    # `time.time() - mtime` evaluate to the current epoch (~1.78e9 s) and sail past
    # any threshold -- the "too new to flush" guard never fired and WarpX had a
    # checkpoint staged out from under it while it was still being written. gkfs
    # does keep ctime, and it advances on write. On a normal filesystem mtime is
    # the meaningful one and ctime tracks it, so the max is correct on both.
    try:
        out = preloaded_call(args, f"find {item} -type f -printf '%T@ %C@\\n'")
        times = [float(x) for x in out.split() if x.strip()]
        if times:
            return max(times)
    except Exception:
        pass
    # No files under the item. Either it is genuinely empty or a flush already
    # emptied it and gekko is still handing back stale directory metadata -- that
    # is what made an already-staged WarpX checkpoint get copied a second time.
    # Report it as brand new so the caller's freshness check skips it: there is
    # nothing left to stage either way.
    try:
        stamp = get_modification_time(args, item)
    except Exception:
        return time.time()
    return stamp if stamp > 0 else time.time()


def move_item(
    args: argparse.Namespace,
    item: str,
    period: float = 0,
    triggered_by: str = "ftio",
) -> None:
    """
    Stages out a single file if it matches the regex and meets modification time criteria.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        item (str): Name of the file to stage out.
        period (float): Expected I/O period used to derive mtime threshold.
        triggered_by (str): Who initiated the flush — "ftio" or "post_app".
    """
    fast = False  # fast copy still has an error with the preload

    # threshold = period / 2  # the IO took half the time
    threshold = 0  # already considered in calculation of flush time
    threshold = max(threshold, 5)

    item_time = newest_mtime(args, item)
    modification_time = time.time() - item_time

    # Unlinking an item does not immediately retire its metadata in GekkoFS, so a
    # later listing can still report its files and stage the very same checkpoint
    # a second time (WarpX copied one twice every run). Re-copying it is wasted
    # stage-out bandwidth; skip unless the item actually got newer since.
    if files_in_progress.already_flushed(item, item_time):
        TRIGGER_LOGGER.info(f"Skipping {item}: already staged and unchanged since")
        files_in_progress.mark_done(item)
        return

    if args.ignore_mtime or modification_time >= threshold:
        dst = item.replace(args.gkfs_mntdir, args.stage_out_path)
        TRIGGER_LOGGER.info(
            f"Moving {item} → {dst}"
            f" (last modified {modification_time:.3f} s ago > threshold {threshold} s)"
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if fast:
            # Todo: add mode for folder. However this currenlty not used
            fast_chunk_copy_file(args, item, threads=4)
        else:
            copy_file_and_unlink(args, item, triggered_by)
        files_in_progress.mark_flushed(item, item_time)
    else:
        TRIGGER_LOGGER.warning(
            f"Skipping {item}: too new (last modified {modification_time:.3f} s ago < threshold {threshold} s)"
        )
        files_in_progress.mark_failed(item)


def copy_file_and_unlink(
    args: argparse.Namespace, item: str, triggered_by: str = "ftio"
) -> None:
    """
    Copies a file or folder to the stage-out path and removes it from the source.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        item (str): Name of the file or folder to copy and remove.
        triggered_by (str): Who initiated the flush — "ftio" or "post_app".
    """
    flush_log = getattr(args, "flush_log", "")
    # Preserve intermediate directories: copy into the parent of the intended
    # destination so that e.g. .../checkpoints/epoch10 lands at
    # stage_out_path/checkpoints/epoch10, not stage_out_path/epoch10.
    dst = item.replace(args.gkfs_mntdir, args.stage_out_path)
    dst_parent = os.path.dirname(dst)

    # When every file in the mount matches the regex, the flush unit collapses to
    # the mount itself. Then dst *is* stage_out_path and dst_parent is its parent,
    # so `cp -rL <mnt> <dst_parent>` recreated the mount under the stage dir:
    # LAMMPS' last checkpoint landed in <stage>/tarraf_gkfs_mountdir/ while the
    # flush log claimed <stage>/output/. Copy the mount's *contents* instead.
    whole_mount = os.path.normpath(item) == os.path.normpath(args.gkfs_mntdir)
    if whole_mount:
        dst = args.stage_out_path
        src = f"{item}/."
    else:
        src = item
    # One command pair covers files and directories alike. Do not branch on
    # `test -d`: GekkoFS has no real directory inodes, so it answers "not a
    # directory" for every path inside the mount, and an AMReX checkpoint
    # directory would then be copied with `cp -L` -> "omitting directory".
    # `cp -rL` copies a regular file just as well (-L dereferences symlinks so
    # the destination holds real data once the source is unlinked), and
    # `find <item> -type f` matches a regular file by itself.
    #
    # `-delete` unlinks from find itself, so there is no `rm` and no fork per
    # file. Forking is the whole cost here: every fork re-initialises the GekkoFS
    # client. Measured on a 54-file AMReX checkpoint, single-node gkfs:
    #   -exec unlink {} \;   34.15 s  (one fork per file)
    #   -exec rm -f {} +      1.22 s  (one fork in total)
    #   -delete               0.72 s  (no fork at all)
    # `rm -f {} +` remains the fallback: -f tolerates a path a concurrent flush
    # has already removed, and it covers a find without -delete.
    cp_cmd = f"cp -rL {src} {dst if whole_mount else dst_parent}"
    remove_cmd = f"find {item} -type f -delete"
    fallback_remove_cmd = f"find {item} -type f -exec rm -f {{}} +"

    TRIGGER_LOGGER.info(f"Copying {item} to {dst}")
    start = time.time()
    preloaded_call(args, cp_cmd)
    copy_time = time.time() - start
    TRIGGER_LOGGER.info(f"Finished copying {item} ({copy_time:.3f} s)")

    start = time.time()
    TRIGGER_LOGGER.info(f"Removing {item} from source")
    try:
        preloaded_call(args, remove_cmd)
    except Exception as e:
        TRIGGER_LOGGER.error(f"Exception during {remove_cmd}: {e}")
        try:
            preloaded_call(args, fallback_remove_cmd)
        except Exception as e:
            TRIGGER_LOGGER.error(f"Fallback also failed for {fallback_remove_cmd}: {e}")
            # Only now is the item genuinely stuck in the mount. If the fallback
            # cleared it there is nothing left to ignore.
            files_in_progress.put_ignore(args, item)
            TRIGGER_LOGGER.warning(f"Added {item} to ignore queue")

    delete_time = time.time() - start
    TRIGGER_LOGGER.info(
        f"Done {item}: copy {copy_time:.3f} s, delete {delete_time:.3f} s"
    )
    _write_flush_log(flush_log, item, dst, triggered_by, copy_time, delete_time)


def delete_items(args: argparse.Namespace, items: list[str]) -> None:
    """
    Deletes a list of files or folders from the source.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        items (list[str]): List of file or folder paths to delete.
    """
    for item in items:
        start = time.time()
        TRIGGER_LOGGER.info(f"Removing {item}")

        # Choose deletion command based on type
        if "." in item.split("/")[-1]:  # assume file
            remove_cmd = f"unlink {item}"
        else:  # assume folder
            remove_cmd = f"find {item} -type f -exec unlink {{}} \\; && rm -rf {item}"

        try:
            preloaded_call(args, remove_cmd)
            elapsed = time.time() - start
            TRIGGER_LOGGER.info(f"Removed {item} in {elapsed:.2f} s")

        except Exception as e:
            TRIGGER_LOGGER.error(f"Exception during deletion of {item}: {e}")
            try:
                # fallback: try file-only unlink for directories
                remove_cmd = f"find {item} -type f -exec unlink {{}} \\;"
                preloaded_call(args, remove_cmd)
            except Exception as e2:
                TRIGGER_LOGGER.error(f"Fallback deletion also failed for {item}: {e2}")
                files_in_progress.put_ignore(args, item)
                TRIGGER_LOGGER.warning(f"Added {item} to ignore queue")


def _list_with_retry(
    args: argparse.Namespace, call: str, attempts: int = 10, delay: float = 3.0
) -> str:
    """Run a listing command, retrying while GekkoFS reports the mount busy.

    A flush in flight makes the daemon answer EBUSY to a concurrent listing. The
    condition clears on its own once the other client is done, so back off and
    retry rather than letting the caller abort.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        call (str): The listing command to run under the GekkoFS preload.
        attempts (int): How many times to try before giving up.
        delay (float): Seconds to wait between attempts.

    Returns:
        str: Output of the listing command.

    Raises:
        Exception: The last error, if every attempt failed.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return preloaded_call(args, call)
        except Exception as e:
            last = e
            if attempt < attempts:
                TRIGGER_LOGGER.warning(
                    f"Listing the mount failed (attempt {attempt}/{attempts}),"
                    f" retrying in {delay:.0f} s: {e}"
                )
                time.sleep(delay)
    raise last  # type: ignore[misc]


def get_files(args: argparse.Namespace) -> list[str]:
    """
    Retrieves a list of files from the GekkoFS mount directory.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        list[str]: List of file paths.
    """
    try:
        # Ask find for regular files only. The old `"." in item` heuristic
        # dropped every dot-free path, which silently discarded AMReX
        # checkpoints (Castro/WarpX) whose files live under dot-free directory
        # names (e.g. <mnt>/sedov_3d_sph_chk00010/Header, .../Level_0/Cell_D_*).
        #
        # Retry: while another client is mid-flush the daemon can answer EBUSY
        # ("Device or resource busy"). That is transient, but an unhandled raise
        # here aborts the whole post-app stage-out and any checkpoint still in the
        # mount is lost with it -- exactly how WarpX's chk000085 disappeared.
        files = _list_with_retry(args, f"find {args.gkfs_mntdir} -type f")
        if isinstance(files, str):
            files = files.strip().splitlines()
        files = [item for item in files if item.strip()]
        if args.debug:
            TRIGGER_LOGGER.debug(f"Found files: {files}")

    except Exception as e:
        if args.debug:
            TRIGGER_LOGGER.error(f"Error listing files: {e}")

        files = _list_with_retry(args, f"ls -R {args.gkfs_mntdir}")

        if files:
            files = files.splitlines()
            files = [f for f in files if "LIBGKFS" not in f]
            # Drop blank separators and the "directory:" header lines that
            # `ls -R` emits, keeping the file entries (dot-free names included).
            files = [
                f"{args.gkfs_mntdir}/{item}"
                for item in files
                if item.strip() and not item.rstrip().endswith(":")
            ]

    if files:
        return files
    else:
        return []


def write_chunk(
    mmapped_file: mmap.mmap,
    dst: str,
    start: int,
    end: int,
    ld_preload: str = "",
) -> None:
    """
    Writes a chunk of the file to the destination using memory mapping.

    Args:
        mmapped_file (mmap.mmap): Memory-mapped file object.
        dst (str): Destination file path.
        start (int): Start byte of the chunk.
        end (int): End byte of the chunk.
        ld_preload (str, optional): LD_PRELOAD environment variable. Defaults to None.
    """
    # Ensure the LD_PRELOAD is set in the thread environment
    if ld_preload:
        os.environ["LD_PRELOAD"] = str(ld_preload)

    with open(dst, "r+b") as fdst:
        fdst.seek(start)
        fdst.write(mmapped_file[start:end])


def copy_metadata(src: str, dst: str, ld_preload: str = None) -> None:
    """
    Copies metadata (permissions and timestamps) from the source to the destination file.

    Args:
        src (str): Source file path.
        dst (str): Destination file path.
        ld_preload (str, optional): LD_PRELOAD environment variable. Defaults to None.
    """
    # Ensure the LD_PRELOAD is set in the current environment before performing metadata copy
    if ld_preload:
        os.environ["LD_PRELOAD"] = str(ld_preload)

    # Get the current metadata of the source file
    stat_info = os.stat(src)

    # Set the metadata on the destination file (timestamps and permissions)
    os.utime(
        dst, (stat_info.st_atime, stat_info.st_mtime)
    )  # Access and modification times
    os.chmod(dst, stat_info.st_mode)  # File permissions


def fast_chunk_copy_file(
    args: argparse.Namespace, file_name: str, threads: int = 4
) -> None:
    """
    Copies a single file in parallel using multiple threads.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        file_name (str): Name of the file to copy.
        threads (int, optional): Number of threads to use. Defaults to 4.
    """
    src = file_name
    dst = f"{args.stage_out_path}/{file_name}"  # Corrected dst path
    os.environ["LIBGKFS_HOSTS_FILE"] = str(args.host_file)
    os.environ["LD_PRELOAD"] = str(args.ld_preload) if args.ld_preload else ""

    # Get file size using open() to avoid potential issues with os.path.getsize()
    with open(src, "rb") as fsrc:
        fsrc.seek(0, os.SEEK_END)
        file_size = fsrc.tell()  # This retrieves the file size

    # Memory-mapped file for reading (this will use LD_PRELOAD)
    with open(src, "rb") as fsrc:
        mmapped_file = mmap.mmap(fsrc.fileno(), 0, access=mmap.ACCESS_READ)

    # Create an empty file at destination with the same size
    with open(dst, "wb") as fdst:
        fdst.write(b"\0" * file_size)

    # Split file into chunks for each thread
    chunk_size = math.ceil(file_size / threads)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i in range(threads):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, file_size)
            futures.append(
                executor.submit(
                    write_chunk, mmapped_file, dst, start, end, args.ld_preload
                )
            )

        # Wait for all threads to finish
        for future in futures:
            future.result()

    # Copy metadata using os.utime and os.chmod (these system calls will be affected by LD_PRELOAD)
    copy_metadata(src, dst, args.ld_preload)


def jit_move(settings: JitSettings) -> None:
    """
    Moves files based on JitSettings configuration.

    Args:
        settings (JitSettings): JIT settings for file staging.
    """
    # Prepare argument list based on JitSettings
    args = [
        "--stage_out_path",
        str(settings.stage_out_path),
        "--stage_in_path",
        str(settings.stage_in_path),
        "--ld_preload",
        str(settings.gkfs_intercept),
        "--host_file",
        str(settings.gkfs_hostfile),
        "--gkfs_mntdir",
        str(settings.gkfs_mntdir),
        "--ignore_mtime",
        "--flush_call",
        str(settings.flush_call),
    ]
    if settings.regex_match:
        args += ["--regex", f"{str(settings.regex_match)}"]

    if settings.parallel_move_threads > 0:
        args += ["--parallel_move_threads", f"{str(settings.parallel_move_threads)}"]

    if settings.debug_lvl > 0:
        args += ["--debug"]

    if settings.fuse:
        args += ["--node", f"{str(settings.single_node)}"]

    if settings.flush_log:
        args += ["--flush_log", str(settings.flush_log)]

    # Define CLI parser
    parser = argparse.ArgumentParser(
        description="Data staging arguments", prog="file_mover"
    )

    parser.add_argument(
        "--stage_out_path",
        type=str,
        help="Cargo stage out path",
        default="/lustre/project/nhr-admire/tarraf/stage-out",
    )
    parser.add_argument(
        "--stage_in_path",
        type=str,
        help="Cargo stage in path",
        default="/lustre/project/nhr-admire/tarraf/stage-in",
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=None,
        help="Files that match the regex expression are ignored during stage out",
    )
    parser.add_argument(
        "--ld_preload",
        type=str,
        default=None,
        help="LD_PRELOAD call to GekkoFs file.",
    )
    parser.add_argument(
        "--host_file", type=str, default=None, help="Hostfile for GekkoFs."
    )
    parser.add_argument(
        "--gkfs_mntdir",
        type=str,
        default=None,
        help="Mount directory for GekkoFs.",
    )
    parser.add_argument("--ignore_mtime", action="store_true", default=True)
    parser.add_argument("--parallel_move_threads", type=int, default=1)
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug flag",
        default=False,
    )
    parser.add_argument(
        "--handle_new_prediction",
        dest="handle_new_prediction",
        help="Adaptive flag for flushing",
        default="cancel",
        choices={"skip", "cancel", ""},
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["flush", "job_end", "buffer_size"],
        default="flush",
        help="Flushing strategy: 'flush' (immediate), 'job_end' (wait until job finishes), or 'buffer_size' (flush after reaching given size).",
    )
    parser.add_argument(
        "--job_time",
        type=int,
        default=0,
        help="Time in seconds required for strategy 'job_end'.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=0,
        help="Buffer size in bytes required for strategy 'buffer_size'.",
    )
    parser.add_argument(
        "--flush_call",
        type=str,
        choices=["cp", "tar"],
        default="cp",
        help="Flushing method: 'cp' to copy files or 'tar' to compress them.",
    )

    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="single node to flush with srun if fuse is set",
    )
    parser.add_argument(
        "--flush_log",
        type=str,
        default="",
        help="Path to the flush log file.",
    )

    # Parse and call mover
    parsed_args = parser.parse_args(args)

    if not settings.dry_run:
        move_files_os(parsed_args, triggered_by="post_app")
