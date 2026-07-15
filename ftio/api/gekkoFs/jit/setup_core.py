"""
This file provides functions to start and manage various components of the GLASS approach,
including the Gekko daemon, proxy, cargo, and FTIO. It also includes functions for staging
data in and out, and executing pre- and post-application calls.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import math
import os
import re

# import multiprocessing
import subprocess
import time

from ftio.api.gekkoFs.jit.execute_and_wait import (
    end_of_transfer_online,
    execute_background,
    execute_background_and_log,
    execute_background_and_wait_line,
    execute_block,
    execute_block_and_monitor,
    get_time,
    monitor_log_file,
    wait_for_file,
    wait_for_line,
)
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.jittime import JitTime
from ftio.api.gekkoFs.jit.setup_check import check_setup
from ftio.api.gekkoFs.jit.setup_helper import (
    adjust_regex,
    check,
    check_port,
    elapsed_time,
    exit_routine,
    extract_accurate_time,
    flaged_call,
    get_env,
    get_executable_realpath,
    handle_sigint,
    jit_print,
    mpiexec_call,
    relevant_files,
    remove_hostfile,
    shut_down,
)
from ftio.api.gekkoFs.jit.setup_init import init_gekko
from ftio.api.gekkoFs.posix_control import format_size, jit_move

_MAX_RETRIES = 3


def _flush_log_event(log_file: str, event: str, label: str = "") -> None:
    """Write a single-line event marker to the flush log (APP-START / APP-END)."""
    if not log_file:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f" ({label})" if label else ""
    line = f"{ts} | {event}{suffix}\n"
    try:
        with open(log_file, "a") as f:
            f.write(line)
    except Exception:
        pass


_GEKKO_CONN_PATTERNS = (
    "software caused connection abort",
    "connection abort",
    "errno 103",
    "errno 107",
    "transport endpoint is not connected",
    "stale file handle",
    "device or resource busy",  # EBUSY (errno 16) — GekkoFS mount unresponsive
)


def _has_gekko_connection_error(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in _GEKKO_CONN_PATTERNS)


def _cleanup_stale_gekko(settings: JitSettings) -> None:
    """Soft-kills Gekko daemon and FUSE and unmounts stale GekkoFS mount points before a retry."""
    jit_print("[bold yellow]Cleaning up stale Gekko/FUSE state before retry...[/]")
    if not settings.exclude_daemon:
        try:
            shut_down(settings, "GEKKO", settings.gkfs_daemon_pid)
            settings.gkfs_daemon_pid = 0
        except Exception as e:
            jit_print(f"[yellow]Unable to shut down Gekko daemon: {e}[/]")
    if settings.fuse:
        try:
            shut_down(settings, "FUSE", settings.gkfs_fuse_pid)
            settings.gkfs_fuse_pid = 0
        except Exception as e:
            jit_print(f"[yellow]Unable to shut down FUSE: {e}[/]")
    # Unmount stale GekkoFS FUSE mount on all nodes.
    # fusermount -uz handles live mounts; umount -l handles zombie mounts
    # where the FUSE daemon is already dead.
    unmount_call = flaged_call(
        settings,
        f"fusermount -uz {settings.gkfs_mntdir} 2>/dev/null || umount -l {settings.gkfs_mntdir} 2>/dev/null || true",
        nodes=settings.app_nodes,
        procs_per_node=1,
        exclude=["ftio", "demon", "proxy", "cargo"],
    )
    execute_block(unmount_call, raise_exception=False, dry_run=settings.dry_run)
    remove_hostfile(settings)
    time.sleep(5)


#! Start gekko
#!###############################
def start_gekko_daemon(settings: JitSettings) -> None:
    """Starts the Gekko daemon.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.exclude_daemon:
        jit_print(
            f"[bold yellow]############## Skipping Gkfs Demon [/][black][{get_time()}][/]"
        )
        return

    jit_print(
        f"[bold green]############## Starting Gkfs Demon [/][black][{get_time()}][/]"
    )

    if settings.use_mpirun:
        debug_flag = f"-x GKFS_DAEMON_LOG_LEVEL=off -x GKFS_DAEMON_LOG_PATH={settings.gkfs_daemon_log}_intern"
        if settings.debug_lvl > 1:
            debug_flag = debug_flag.replace("=none", "=err")
    else:
        debug_flag = f"--export=ALL,GKFS_DAEMON_LOG_LEVEL=err,GKFS_DAEMON_LOG_PATH={settings.gkfs_daemon_log}_intern"
        if settings.debug_lvl > 1:
            debug_flag = debug_flag.replace("=err", "=trace")

    if settings.cluster:
        if settings.use_mpirun:
            call = (
                f" {settings.gkfs_daemon} "
                f"-r {settings.gkfs_rootdir} -m {settings.gkfs_mntdir} "
                f"-H {settings.gkfs_hostfile}  -c --clean-rootdir -l ib0 -P {settings.gkfs_daemon_protocol}"
            )
            call = mpiexec_call(
                settings,
                call,
                settings.app_nodes,
                additional_arguments=debug_flag,
            )
        else:
            call = (
                f"srun {debug_flag} --jobid={settings.job_id} {settings.app_nodes_command} --disable-status -N {settings.app_nodes} "
                f"--ntasks={settings.app_nodes} --cpus-per-task={settings.procs_daemon} --ntasks-per-node=1 --overcommit --overlap "
                f"--oversubscribe --mem=0 {settings.task_set_0} {settings.gkfs_daemon} -r {settings.gkfs_rootdir} -m {settings.gkfs_mntdir} "
                f"-H {settings.gkfs_hostfile}  -c --clean-rootdir -l ib0 -P {settings.gkfs_daemon_protocol}"
            )
        if not settings.exclude_proxy:
            call += " -p ofi+verbs -L ib0"
    else:  # no cluster mode
        if settings.debug_lvl > 1:
            debug_flag = "GKFS_DAEMON_LOG_LEVEL=trace"
        else:
            debug_flag = "GKFS_DAEMON_LOG_LEVEL=err"
        call = (
            f"{debug_flag} LIBGKFS_LOG=all  GKFS_DAEMON_LOG_PATH={settings.gkfs_daemon_log}_intern {settings.gkfs_daemon} -r {settings.gkfs_rootdir} -m {settings.gkfs_mntdir} "
            f"-H {settings.gkfs_hostfile}  -c --clean-rootdir -l lo -P {settings.gkfs_daemon_protocol}"
        )
        if not settings.exclude_proxy:
            call += " --proxy-listen lo --proxy-protocol {settings.gkfs_daemon_protocol}"

    for attempt in range(_MAX_RETRIES):
        try:
            if attempt > 0:
                jit_print(
                    f"[bold yellow]Retrying Gekko daemon (attempt {attempt + 1}/{_MAX_RETRIES})...[/]"
                )
                _cleanup_stale_gekko(settings)

            init_gekko(settings)

            jit_print("[cyan]Starting Demons[/]", True)
            _ = execute_background_and_log(
                settings,
                call,
                settings.gkfs_daemon_log,
                "daemon",
                settings.gkfs_daemon_err,
            )
            if settings.verbose_error:
                _ = monitor_log_file(settings.gkfs_daemon_err, "Error Demon")

            if wait_for_file(
                settings.gkfs_hostfile,
                dry_run=settings.dry_run,
                error_file=settings.gkfs_daemon_err,
            ) and hostfile_is_sane(settings):
                break

            jit_print(
                f"[bold yellow]Gekko daemon did not create a usable hostfile "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})[/]"
            )
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Gekko daemon failed to start after {_MAX_RETRIES} attempts"
                )
            _cleanup_stale_gekko(settings)

        except RuntimeError:
            raise
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            jit_print(
                f"[bold yellow]Gekko daemon start error (attempt {attempt + 1}/{_MAX_RETRIES}): {e} — retrying...[/]"
            )
            _cleanup_stale_gekko(settings)

    jit_print("[green]############## Gkfs init finished ##############\n\n\n\n ")


#! Start FUSE
#!###############################
def start_fuse(settings: JitSettings) -> None:
    """Starts the Gekko FUSE overlay.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.exclude_daemon or not settings.fuse:
        jit_print(
            f"[bold yellow]############## Skipping FUSE [/][black][{get_time()}][/]"
        )
        return

    jit_print(f"[bold green]############## Starting FUSE [/][black][{get_time()}][/]")
    wait_for_file(settings.gkfs_hostfile, dry_run=settings.dry_run)
    if settings.cluster:
        if settings.use_mpirun:
            # mpiexec
            call = (
                f"{settings.gkfs_fuse}"
                f" -o max_idle_threads={settings.fuse_idle_threads} "
                f"-o direct_io -f -o fifo -o auto_unmount {settings.gkfs_mntdir}"
            )
            call = mpiexec_call(
                settings,
                (
                    f"-x LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile} "
                    f"-x LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio} "
                    f"-x LIBGKFS_ENABLE_METRICS=on -x"
                    f"LIBGKFS_METRICS_FLUSH_INTERVAL=5 "
                    f"{call}"
                ),
                settings.app_nodes,
            )
        else:
            call = (
                f"srun  --jobid={settings.job_id} {settings.app_nodes_command} --disable-status -N {settings.app_nodes} "
                #   f"--ntasks={settings.app_nodes*settings.procs_daemon} --cpus-per-task={settings.procs_daemon} --ntasks-per-node={settings.procs_daemon} --overcommit --overlap "
                f"--export=ALL,LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile},"
                f"LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio},"
                f"LIBGKFS_ENABLE_METRICS=on,LIBGKFS_METRICS_FLUSH_INTERVAL=5 "
                f"--ntasks={settings.app_nodes} --cpus-per-task={settings.procs_daemon} --ntasks-per-node=1 --overcommit --overlap "
                f"--oversubscribe --mem=0 {settings.task_set_0} "
                f"{settings.gkfs_fuse}"
                f" -o max_idle_threads={settings.fuse_idle_threads} "
                f"-o direct_io -f -o fifo -o auto_unmount {settings.gkfs_mntdir}"
            )
    else:
        raise RuntimeError("Not implemented fuse")

    # Disable intercept once — FUSE mode does not use LD_PRELOAD
    settings.gkfs_intercept = ""

    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            jit_print(
                f"[bold yellow]Retrying FUSE (attempt {attempt + 1}/{_MAX_RETRIES})...[/]"
            )
            try:
                shut_down(settings, "FUSE", settings.gkfs_fuse_pid)
                settings.gkfs_fuse_pid = 0
            except Exception as e:
                jit_print(f"[yellow]Unable to shut down FUSE: {e}[/]")
            unmount_call = flaged_call(
                settings,
                f"fusermount -uz {settings.gkfs_mntdir} || true",
                nodes=settings.app_nodes,
                procs_per_node=1,
                exclude=["ftio", "demon", "proxy", "cargo"],
            )
            execute_block(unmount_call, raise_exception=False, dry_run=settings.dry_run)
            time.sleep(5)

        jit_print("[cyan]Starting FUSE Demons[/]", True)
        _ = execute_background_and_log(
            settings,
            call,
            settings.gkfs_fuse_log,
            "fuse",
            settings.gkfs_fuse_err,
        )
        if settings.verbose_error:
            _ = monitor_log_file(settings.gkfs_daemon_err, "Error Fuse")

        timeout = int(20 * settings.app_nodes)
        if wait_for_line(
            settings.gkfs_fuse_log,
            "root node allocated",
            timeout=timeout,
            occurrences=settings.app_nodes,
            error_file=settings.gkfs_fuse_err,
        ):
            # Give the FUSE mount a moment to stabilise after the log confirms startup.
            time.sleep(5)
            break

        jit_print(
            f"[bold yellow]FUSE did not start in time -- {timeout} (attempt {attempt + 1}/{_MAX_RETRIES})[/]"
        )
        if attempt == _MAX_RETRIES - 1:
            raise RuntimeError(f"FUSE failed to start after {_MAX_RETRIES} attempts")

    jit_print("[green]############## Gkfs FUSE init finished ##############\n\n\n\n ")


#! Start Proxy
#!############################################
def start_gekko_proxy(settings: JitSettings) -> None:
    """Starts the Gekko proxy.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.exclude_proxy:
        jit_print(
            f"[bold yellow]############## Skipping Gkfs Proxy [/][black][{get_time()}][/]"
        )
    else:
        jit_print(
            f"[bold green]############## Starting Gkfs Proxy [/][black][{get_time()}][/]"
        )

        if settings.cluster:
            # Proxy call for cluster
            call = (
                f"srun --jobid={settings.job_id} {settings.app_nodes_command} "
                f"--disable-status -N {settings.app_nodes} --ntasks={settings.app_nodes} "
                f"--cpus-per-task={settings.procs_proxy} --ntasks-per-node=1 --overcommit "
                f"--overlap --oversubscribe --mem=0 {settings.task_set_1} {settings.gkfs_proxy} "
                f"-H {settings.gkfs_hostfile} -p ofi+verbs -P {settings.gkfs_proxyfile}"
            )
        else:
            # Proxy call for non-cluster
            call = (
                f"{settings.gkfs_proxy} -H {settings.gkfs_hostfile} "
                f"-p {settings.gkfs_daemon_protocol} -P {settings.gkfs_proxyfile}"
            )
        jit_print("[cyan]Starting Proxy[/]")

        # p = multiprocessing.Process(target=execute_background, args= (call, settings.gkfs_proxy_log, settings.gkfs_proxy_err, settings.dry_run))
        # p.start()
        # if settings.verbose:
        #     _ = monitor_log_file(settings.gkfs_proxy_log,"Proxy")
        # if settings.verbose_error:
        #     _ = monitor_log_file(settings.gkfs_proxy_err,"Error Proxy")

        _ = execute_background_and_log(
            settings,
            call,
            settings.gkfs_proxy_log,
            "proxy",
            settings.gkfs_proxy_err,
        )
        if settings.verbose:
            _ = monitor_log_file(settings.gkfs_proxy_err, "Error Proxy")

    if not settings.exclude_proxy:
        jit_print("[green]############## Proxy init finished ##############\n\n\n\n ")


#! start Cargo
#!###############################
def start_cargo(settings: JitSettings) -> None:
    """Starts the Cargo component.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.exclude_cargo:
        jit_print(
            f"[bold yellow]############## Skipping Cargo [/][black][{get_time()}][/]"
        )
    else:
        jit_print(
            f"[bold green]############## Starting Cargo [/][black][{get_time()}][/]"
        )

        if settings.cluster:
            call = (
                f"{settings.cargo_bin}/cargo "
                f"--listen {settings.gkfs_daemon_protocol}://ib0:{settings.port_cargo} -b 65536"
            )
            call = flaged_call(
                settings,
                call,
                settings.app_nodes,
                settings.procs_cargo,
                exclude=["ftio", "demon", "preload", "proxy"],
            )

        else:
            # if settings.debug_lvl > 2:
            #     debug_flag = "CARGO_LOG_LEVEL=trace"
            # else:
            #     debug_flag = "CARGO_LOG_LEVEL=debug"
            debug_flag = ""

            # Command for non-cluster
            call = f"{debug_flag} {settings.cargo_bin}/cargo --listen {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} -b 65536"
            call = flaged_call(
                settings,
                call,
                1,
                settings.procs_cargo,
                exclude=["ftio", "demon", "preload", "proxy"],
            )

        jit_print("[cyan]Starting Cargo[/]")

        # p_cargo = multiprocessing.Process(target=execute_background,args=(call, settings.cargo_log, settings.cargo_err, settings.dry_run))
        # p_cargo.start()
        # if settings.verbose:
        #     _ = monitor_log_file(settings.cargo_log,"Cargo")
        # if settings.verbose_error:
        #     _ = monitor_log_file(settings.cargo_err,"Error Cargo")

        execute_background_and_log(
            settings, call, settings.cargo_log, "cargo", settings.cargo_err
        )
        if settings.verbose_error:
            _ = monitor_log_file(settings.cargo_err, "Error Cargo")

        # wait for line to appear
        time.sleep(2)
        flag = wait_for_line(
            settings.cargo_log, "Start up successful", dry_run=settings.dry_run
        )
        if not flag:
            handle_sigint(settings)

        time.sleep(4)

    if not settings.exclude_cargo:
        jit_print("[green]############## Cargo init finished ##############\n\n\n\n ")


#! start FTIO
#!##############################
def start_ftio(settings: JitSettings) -> None:
    """Starts the FTIO component.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.exclude_ftio or settings.exclude_all:
        jit_print(
            f"[bold yellow]############## Skipping FTIO [/][black][{get_time()}][/]"
        )

        relevant_files(settings)

        if settings.cluster and not settings.exclude_cargo:
            jit_print(
                "[bold yellow]Executing the calls below used later for staging out[/]"
            )
            call_0 = (
                f"srun --jobid={settings.job_id} {settings.single_node_command} "
                f"--disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 "
                f"--overcommit --overlap --oversubscribe --mem=0 {settings.cargo_bin}/cargo_ftio "
                f"--server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --run"
            )
            call_1 = (
                f"srun --jobid={settings.job_id} {settings.single_node_command} "
                f"--disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 "
                f"--overcommit --overlap --oversubscribe --mem=0 {settings.cargo_bin}/ccp "
                f"--server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --input / --output {settings.stage_out_path} "
                f"--if gekkofs --of {settings.cargo_mode}"
            )

            _ = execute_block(call_0, dry_run=settings.dry_run)
            _ = execute_block(call_1, dry_run=settings.dry_run)
    else:
        jit_print(f"[bold green]############## Starting FTIO [/][black][{get_time()}][/]")

        relevant_files(settings)

        ftio_data_staget_args = ""
        if settings.exclude_cargo:
            ftio_data_staget_args += (
                f"--stage_out_path {settings.stage_out_path} --stage_in_path {settings.stage_in_path} "
                f" --host_file {settings.gkfs_hostfile} --gkfs_mntdir {settings.gkfs_mntdir} "
            )
            if not settings.fuse:
                ftio_data_staget_args += f"--ld_preload {settings.gkfs_intercept} "

            if settings.handle_new_prediction:
                ftio_data_staget_args += (
                    f"--handle_new_prediction '{settings.handle_new_prediction}' "
                )

            if settings.regex_match:
                ftio_data_staget_args += f"--regex '{settings.regex_match}' "

            if settings.ignore_mtime:
                ftio_data_staget_args += "--ignore_mtime "

            if settings.parallel_move_threads > 0:
                ftio_data_staget_args += (
                    f"--parallel_move_threads {settings.parallel_move_threads} "
                )

            if settings.debug_lvl > 0:
                ftio_data_staget_args += "--debug "

            if settings.strategy:
                ftio_data_staget_args += f"--strategy '{settings.strategy}' "

            if settings.job_time:
                ftio_data_staget_args += f"--job_time '{settings.job_time}' "

            if settings.buffer_size:
                ftio_data_staget_args += f"--buffer_size '{settings.buffer_size}' "

            if settings.flush_call:
                ftio_data_staget_args += f"--flush_call '{settings.flush_call}' "
            if settings.fuse:
                ftio_data_staget_args += f"--node '{str(settings.single_node)}'"

        else:
            ftio_data_staget_args += (
                f"--cargo --cargo_bin {settings.cargo_bin} "
                f"--cargo_server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --stage_out_path {settings.stage_out_path} -t 0.2"
            )

        if settings.app_start_file:
            # Remove any stale flag from a previous run before starting predictor_jit
            try:
                os.remove(settings.app_start_file)
            except FileNotFoundError:
                jit_print("[yellow]app_start flag file not found, nothing to remove[/]")
            ftio_data_staget_args += f" --app_start_file {settings.app_start_file}"

        if settings.flush_log:
            ftio_data_staget_args += f" --flush_log {settings.flush_log}"

        if settings.cluster:
            jit_print(
                f"[cyan]FTIO started on node {settings.ftio_node}, remaining nodes for the application: {settings.app_nodes} each with {settings.procs_ftio} processes [/]"
            )
            jit_print(
                f"[cyan]FTIO is listening node is {settings.address_ftio}:{settings.port_ftio} [/]"
            )
            # cd into the run's log dir so FTIO's dump_json writes bandwidth.json
            # straight there (its cwd), instead of into a shared RES_DIR to be
            # copied afterwards -- see save_bandwidth.
            call = (
                f"cd {settings.log_dir} && "
                f"srun --jobid={settings.job_id} {settings.ftio_node_command} "
                f"--disable-status -N 1 --ntasks=1 --cpus-per-task={settings.procs_ftio} "
                f"--ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 "
                f"{settings.ftio_bin_location}/predictor_jit {ftio_data_staget_args} --  --zmq_address {settings.address_ftio} --zmq_port {settings.port_ftio} {settings.ftio_args} "
            )
        else:
            check_port(settings)
            call = f"cd {settings.log_dir} && {settings.ftio_bin_location}/predictor_jit {ftio_data_staget_args} --  --zmq_address {settings.address_ftio} --zmq_port {settings.port_ftio} {settings.ftio_args} "

        jit_print("[cyan]Starting FTIO[/]")

        _ = execute_background_and_log(
            settings, call, settings.ftio_log, "ftio", settings.ftio_err
        )
        if settings.verbose_error:
            _ = monitor_log_file(settings.ftio_err, "Error Ftio")

        # p_ftio = multiprocessing.Process(target=execute_background,args=(call, settings.ftio_log, settings.ftio_err, settings.dry_run))
        # p_ftio.start()

        # if settings.verbose:
        #     _ = monitor_log_file(settings.ftio_log,"Ftio")
        #     _ = monitor_log_file(settings.ftio_err,"Error Ftio")

        if not settings.exclude_cargo:
            _ = wait_for_line(
                settings.ftio_log,
                "FTIO is running on:",
                "Waiting for FTIO startup",
                dry_run=settings.dry_run,
            )
        time.sleep(8)
    if not settings.exclude_ftio:
        jit_print("[green]############## FTIO init finished ##############\n\n\n\n ")


#! Stage in
#!##############################
def stage_in(settings: JitSettings, runtime: JitTime) -> None:
    """Stages data in from the specified path.

    Args:
        settings (JitSettings): jit settings
        runtime (JitTime): runtime object to track elapsed time
    """
    if settings.exclude_all:
        jit_print(
            f"[bold yellow]############## Skipping Stage in [/][black][{get_time()}][/]"
        )
    else:
        jit_print(f"[bold green]############## Staging in [/][black][{get_time()}][/]")
        adjust_regex(settings, "stage_in")

        # remove locks
        jit_print("[cyan]Cleaning locks")
        # make sure the path exists
        os.makedirs(settings.stage_in_path, exist_ok=True)
        os.makedirs(settings.stage_out_path, exist_ok=True)
        # Create an empty file named "test"
        test_file = os.path.join(settings.stage_in_path, "test")
        with open(test_file, "w"):
            pass

        if settings.stage_in_path:
            execute_block(
                f"cd {settings.stage_in_path} && rm -f  $(find .  | grep .lock)"
            )

        jit_print(f"[cyan]Staging in to {settings.stage_in_path}", True)

        if True:  # settings.exclude_cargo:  # settings.exclude_cargo:
            # check that there are actually files
            call = ""
            if (
                len(
                    subprocess.check_output(
                        f"ls -R {settings.stage_in_path}/", shell=True
                    ).decode()
                )
                > 0
            ):
                call = flaged_call(
                    settings,
                    f"cp  -r {settings.stage_in_path}/* {settings.gkfs_mntdir}",
                    exclude=["ftio"],
                )
                # or faster
                # copy_procs = max(10, int(settings.procs_app / 2))
                # call = flaged_call(
                #     settings,
                #     procs_per_node=copy_procs,
                #     call=(
                #         "bash -c "
                #         f'"STAGE_IN_SRC={settings.stage_in_path}; '
                #         f"STAGE_IN_DST={settings.gkfs_mntdir}; "
                #         f'find \\"$STAGE_IN_SRC\\" -type f | xargs -n 1 -P {copy_procs} cp --parents -t \\"$STAGE_IN_DST\\""'
                #     ),
                #     exclude=["ftio"],
                # )

            start = time.time()
            _ = execute_block(call, dry_run=settings.dry_run)
        # TODO: fix cpp cargo
        else:
            if settings.cluster:
                call = flaged_call(
                    settings,
                    f"{settings.cargo_bin}/ccp --server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --output / --input {settings.stage_in_path} --of gekkofs --if {settings.cargo_mode}",
                    exclude=["ftio"],
                )
                # call = f"srun --jobid={settings.job_id} {settings.single_node_command} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 "
            else:
                call = f"mpiexec -x LIBGKFS_ENABLE_METRICS=off -np 1 --oversubscribe {settings.cargo_bin}/ccp --server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --output / --input {settings.stage_in_path} --of gekkofs --if {settings.cargo_mode}"

            start = time.time()

            execute_background_and_wait_line(
                call,
                settings.cargo_log,
                "retval: CARGO_SUCCESS, status: {state: completed",
                settings.dry_run,
            )

        elapsed_time(settings, runtime, "Stage in", time.time() - start)
        check(settings)
        # adjust regex for flushing
        adjust_regex(settings, "flush")
    if not settings.exclude_all:
        jit_print("[green]############## Stage-in finished ##############\n\n\n\n ")


#! Stage out
#!##############################
def hostfile_is_sane(settings: JitSettings) -> bool:
    """Check the GekkoFS hostfile lists exactly the daemons we just started.

    The daemon **appends** to the hostfile (`ios::out | ios::app`) and only
    removes its own line on a *clean* shutdown. A daemon that is SIGKILLed --
    a job timeout, a crash, an OOM, Ctrl-C -- therefore leaves its line behind,
    and the next daemon appends below it.

    The client loads every line as a host and hashes each path onto one of them,
    so a path that lands on the dead entry gets HG_NOENTRY back from Mercury.
    GekkoFS then reports that as EBUSY, i.e. the thoroughly misleading

        cp: target '<mnt>': Device or resource busy

    which is really "the daemon for this path never answered". It is a hash
    coin-flip per path, so it looks random: measured 4 of 6 operations failing
    with one stale line next to one live one.

    Args:
        settings (JitSettings): The JIT settings.

    Returns:
        bool: True when the hostfile holds exactly one entry per daemon node.
    """
    if settings.dry_run:
        return True
    try:
        with open(settings.gkfs_hostfile) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        jit_print(f"[bold red]Cannot read the hostfile: {e}[/]")
        return False

    expected = max(1, settings.app_nodes) if settings.cluster else 1
    if len(lines) <= expected:
        return True

    jit_print(
        f"[bold red]Stale GekkoFS hostfile: {len(lines)} entries for {expected} "
        f"daemon(s). A previous daemon was killed before it could remove its "
        f"line, and the new one appended below it. Clients would hash some paths "
        f"onto the dead daemon and get 'Device or resource busy'. Cleaning up.[/]"
    )
    for line in lines:
        jit_print(f"[yellow]  {' '.join(line.split()[:2])}[/]")
    return False


def flushable_bytes(settings: JitSettings) -> int:
    """Bytes still sitting in the GekkoFS mount, or -1 if the mount cannot be read.

    Args:
        settings (JitSettings): The JIT settings.

    Returns:
        int: Total size of the files left in the mount, -1 when the listing failed.
    """
    try:
        call = flaged_call(
            settings,
            f"find {settings.gkfs_mntdir} -type f -printf '%s\\n'",
            exclude=["ftio"],
        )
        raw = subprocess.check_output(call, shell=True, stderr=subprocess.DEVNULL)
        return sum(int(line) for line in raw.decode().split() if line.strip().isdigit())
    except Exception:
        # A flush in flight makes the daemon answer EBUSY; that is not "empty".
        return -1


def wait_for_flush_to_settle(
    settings: JitSettings, timeout: float = 120.0, poll: float = 2.0
) -> None:
    """Wait for FTIO's in-flight stage-out to finish before FTIO is shut down.

    The application has ended, so nothing new enters the mount; the only thing
    still draining it is a flush FTIO started before the app exited. Killing FTIO
    now tears that flush down mid-copy and leaves the daemon answering EBUSY to
    the post-app sweep, which then aborts and loses whatever was still in the
    mount (this is how WarpX's chk000085 was lost).

    Poll until the mount stops shrinking: either it empties, or two consecutive
    polls report the same number of bytes, meaning no flush is making progress.

    Args:
        settings (JitSettings): The JIT settings.
        timeout (float): Give up waiting after this many seconds.
        poll (float): Seconds between polls.
    """
    deadline = time.time() + timeout
    previous = -1
    while time.time() < deadline:
        current = flushable_bytes(settings)
        if current == 0:
            jit_print("[cyan]FTIO drained the mount, nothing left to stage out")
            return
        if current > 0 and current == previous:
            jit_print(
                f"[cyan]FTIO flush settled with {format_size(current)} left;"
                " staging the remainder"
            )
            return
        previous = current
        time.sleep(poll)

    jit_print(
        "[yellow]Timed out waiting for FTIO to finish flushing;"
        " staging out what is left"
    )


def stage_out(settings: JitSettings, runtime: JitTime) -> None:
    """Stages data out to the specified path.

    Args:
        settings (JitSettings): jit settings
        runtime (JitTime): runtime object to track elapsed time
    """
    if settings.exclude_all:
        jit_print(
            f"[bold yellow]############## Skipping  Stage out [/][black][{get_time()}][/]"
        )
    else:
        jit_print(
            f"[bold green]############## Staging out [/][black][{get_time()}][/]\n",
            f"[cyan]Moving data from {settings.gkfs_mntdir} -> {settings.stage_out_path}[/]",
        )
        adjust_regex(settings, "stage_out")

        if settings.exclude_cargo:
            # call = flaged_call(
            #     settings,
            #     f"cp -r  {settings.gkfs_mntdir} {settings.stage_out_path} || echo 'nothing to stage out'",
            #     exclude=["ftio"],
            # )
            start = time.time()
            # _ = execute_block(call, dry_run=settings.dry_run)

            #  give ftio slightly more time to finish moving
            if not settings.exclude_ftio:
                wait_for_flush_to_settle(settings)
                jit_print("[cyan]Shutting down FTIO as application finished")
                shut_down(settings, "FTIO", settings.ftio_pid)

            jit_move(settings)
            elapsed_time(settings, runtime, "Stage out", time.time() - start)
        else:
            if not settings.dry_run:
                try:
                    call = flaged_call(
                        settings,
                        f"find {settings.gkfs_mntdir} -type f -printf '%s %p\\n'",
                        exclude=["ftio"],
                    )
                    raw = subprocess.check_output(call, shell=True).decode()
                    lines = [line for line in raw.splitlines() if line.strip()]
                    parsed = [
                        (int(line.split()[0]), line.split(None, 1)[1])
                        for line in lines
                        if len(line.split(None, 1)) == 2 and line.split()[0].isdigit()
                    ]
                    total = sum(sz for sz, _ in parsed)
                    file_details = "\n".join(
                        f"  {path} ({format_size(sz)})"
                        f" → {path.replace(settings.gkfs_mntdir, settings.stage_out_path)}"
                        for sz, path in parsed
                    )
                    jit_print(
                        f"[cyan]{len(parsed)} file(s) to stage out"
                        f" ({format_size(total)} total) → {settings.stage_out_path}:[/]",
                        f"[cyan]{file_details or '  (none)'}[/]",
                    )
                except Exception as e:
                    jit_print(f"[red] Error listing files:\n{e}")

            # Reset relevant files
            if settings.cluster:
                call = (
                    f"srun --jobid={settings.job_id} {settings.single_node_command} "
                    f"--disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 "
                    f"--overcommit --overlap --oversubscribe --mem=0 {settings.cargo_bin}/cargo_ftio "
                    f"--server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --run"
                )
            else:
                call = (
                    f"mpiexec -np 1 --oversubscribe {settings.cargo_bin}/cargo_ftio "
                    f"--server {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo} --run"
                )

            # Measure and print elapsed time
            start = time.time()
            execute_background_and_wait_line(
                call,
                settings.cargo_log,
                # "[info] Transfer finished for [",
                "ftio_int RPC was successful!",
                settings.dry_run,
            )
            # end_of_transfer(settings, settings.cargo_log,call)
            end_of_transfer_online(settings, settings.cargo_log, call)
            elapsed_time(settings, runtime, "Stage out", time.time() - start)
            # Set ignored files to default again
            relevant_files(settings)
            time.sleep(5)
    if not settings.exclude_all:
        jit_print("[green]############## Stage-Out finished ##############\n\n\n\n ")


#! App call
#!##############################
def start_application(settings: JitSettings, runtime: JitTime):
    """Starts the application specified in the settings.

    Args:
        settings (JitSettings): jit settings
        runtime (JitTime): runtime object to track elapsed time
    """
    name = (
        settings.app_call.split("/", 1)[1]
        if "/" in settings.app_call
        else settings.app_call
    )
    jit_print(
        f"[green bold]############## Executing Application {name} [/][black][{get_time()}][/]"
    )
    # set up dir
    original_dir = settings.dir
    jit_print(f" Current directory {original_dir}")
    if not settings.dry_run:
        check_setup(settings)
        # pass

    if settings.cluster:
        additional_arguments = ""
        if settings.use_mpirun:
            if not settings.exclude_ftio:
                additional_arguments += f"-x LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio} -x LIBGKFS_ENABLE_METRICS=on -x LIBGKFS_METRICS_FLUSH_INTERVAL=5 "
            if not settings.exclude_proxy:
                additional_arguments += (
                    f"-x LIBGKFS_PROXY_PID_FILE={settings.gkfs_proxyfile} "
                )
            if not settings.exclude_daemon:
                if settings.debug_lvl > 1:
                    log_modules = "all"
                else:
                    log_modules = "errors"
                additional_arguments += (
                    f"-x LIBGKFS_LOG={log_modules} "
                    f"-x LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log} "
                    f"-x LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile} "
                )
                if settings.preload_via_export:
                    additional_arguments += f"-x LD_PRELOAD={settings.gkfs_intercept} "

            if (settings.lock_generator and not settings.exclude_daemon) or (
                settings.lock_consumer and not settings.exclude_cargo
            ):
                additional_arguments += get_env(settings, "mpi")

            app_inner = f"{settings.app_call} {settings.app_flags}"
            if (
                not settings.preload_via_export
                and not settings.exclude_daemon
                and settings.gkfs_intercept
            ):
                app_inner = f'bash -c "LD_PRELOAD={settings.gkfs_intercept} {app_inner}"'
            call = (
                # f" cd {settings.run_dir} && "
                # f"strace -f -e trace=read,write,open,close,stat,fstat,lseek,access -o /gpfs/fs1/home/tarrafah/strace_n{settings.app_nodes}_p{settings.procs_app}.txt mpiexec -np {settings.app_nodes*settings.procs_app} --oversubscribe "
                # f" cd {settings.run_dir} && time -p mpiexec --mca errhandler ftmpi --mca mpi_abort_print_stack 1  -np {settings.app_nodes*settings.procs_app} --oversubscribe "
                # ssh {settings.single_node} 'pwd && cd {settings.run_dir} && pwd  && ls && hostname
                f" cd {settings.run_dir} && time -p  mpiexec -np {settings.app_nodes * settings.procs_app} --oversubscribe "
                f"--hostfile {settings.mpi_hostfile} --map-by node "
                f"{additional_arguments} "
                f"{settings.task_set_1} {app_inner}"
            )
        else:
            if not settings.exclude_proxy:
                additional_arguments += (
                    f"LIBGKFS_PROXY_PID_FILE={settings.gkfs_proxyfile},"
                )
            if not settings.exclude_daemon and (
                not settings.exclude_cargo and settings.lock_generator
            ):
                additional_arguments += get_env(settings, "srun")

            # Only resolve to absolute path when app_call already looks like a
            # path (starts with / or ./).  Bare command names (e.g. "dlio_benchmark")
            # are intentionally left as-is so bash on the compute node resolves
            # them from PATH at runtime — consistent with how flaged_srun_call
            # handles pre_call.  Resolving via shutil.which on the launcher can
            # accidentally pick up executables from the FTIO venv instead of the
            # user's intended installation.
            if os.path.sep in settings.app_call or settings.app_call.startswith("."):
                app_call = get_executable_realpath(settings.app_call, settings.run_dir)
            else:
                app_call = settings.app_call

            if not settings.preload_via_export and not settings.fuse:
                # Default: set all GekkoFS/FTIO vars inline in bash -c to avoid
                # quoting and shell-expansion issues in srun --export.
                gkfs_env = ""
                if not settings.exclude_ftio:
                    gkfs_env += (
                        f"LIBGKFS_ENABLE_METRICS=on "
                        f"LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio} "
                        f"LIBGKFS_METRICS_FLUSH_INTERVAL=5 "
                    )
                if not settings.exclude_daemon:
                    log_modules = "all" if settings.debug_lvl > 1 else "errors"
                    gkfs_env += (
                        f"LIBGKFS_LOG={log_modules} "
                        f"LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log} "
                        f"LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile} "
                    )
                    if settings.gkfs_intercept:
                        gkfs_env = f"LD_PRELOAD={settings.gkfs_intercept} " + gkfs_env
                app_invocation = f'bash -c "{gkfs_env}{app_call} {settings.app_flags}"'
            else:
                # Legacy / FUSE mode: pass GekkoFS vars via srun --export
                if not settings.exclude_ftio:
                    additional_arguments += f"LIBGKFS_ENABLE_METRICS=on,LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio},LIBGKFS_METRICS_FLUSH_INTERVAL=5,"
                if not settings.exclude_daemon:
                    log_modules = "all" if settings.debug_lvl > 1 else "errors"
                    additional_arguments += (
                        f'LIBGKFS_LOG="{log_modules}",'
                        f"LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log},"
                        f"LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile},"
                    )
                    if settings.preload_via_export and settings.gkfs_intercept:
                        additional_arguments += f"LD_PRELOAD={settings.gkfs_intercept},"
                app_invocation = f"{app_call} {settings.app_flags}"
            call = (
                f" cd {settings.run_dir} && time -p srun "
                f"--export=ALL,{additional_arguments}LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')} "
                f"--jobid={settings.job_id} {settings.app_nodes_command} --disable-status "
                f"-N {settings.app_nodes} --ntasks={settings.app_nodes * settings.procs_app} "
                f"--cpus-per-task={settings.procs_app} --ntasks-per-node={settings.procs_app} "
                f"--overcommit --overlap --oversubscribe --mem=0 "
                f"{settings.task_set_1} {app_invocation}"
            )
    else:
        # Define the call for non-cluster environment
        if settings.run_dir:
            os.chdir(settings.run_dir)
            jit_print(f"Changing directory to  {os.getcwd()}")

        additional_arguments = ""
        if not settings.exclude_ftio:
            additional_arguments += f"-x LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio} -x LIBGKFS_ENABLE_METRICS=on "
        if not settings.exclude_proxy:
            additional_arguments += (
                f"-x LIBGKFS_PROXY_PID_FILE={settings.gkfs_proxyfile} "
            )
        if not settings.exclude_daemon:
            log_modules = "all" if settings.debug_lvl > 1 else "errors"
            additional_arguments += (
                f"-x LIBGKFS_LOG={log_modules} "
                f"-x LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log} "
                f"-x LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile} "
            )
            if settings.preload_via_export:
                additional_arguments += f"-x LD_PRELOAD={settings.gkfs_intercept} "
            if (
                not settings.exclude_cargo and settings.lock_generator
            ):  # if gekko and cargo active
                additional_arguments += get_env(settings, "mpi")

        app_inner = f"{settings.app_call} {settings.app_flags}"
        if (
            not settings.preload_via_export
            and not settings.exclude_daemon
            and settings.gkfs_intercept
        ):
            app_inner = f'bash -c "LD_PRELOAD={settings.gkfs_intercept} {app_inner}"'
        call = (
            f" time  mpiexec -np {int(settings.procs_app)} --oversubscribe "
            f"{additional_arguments} {app_inner}"
        )

    # elapsed = execute_block_and_log(call, settings.app_log_dir)
    check(settings)
    _flush_log_event(settings.flush_log, "APP-START", name)
    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            jit_print(
                f"[bold yellow]Retrying application (attempt {attempt + 1}/{_MAX_RETRIES})...[/]"
            )
            time.sleep(5)

        start = time.time()
        process = execute_background(
            call, settings.app_log, settings.app_err, settings.dry_run
        )
        # monitor log
        if settings.verbose:
            _ = monitor_log_file(settings.app_log, "")
        # monitor error
        if settings.verbose_error:
            _ = monitor_log_file(settings.app_err, f"{name} error")

        app_timeout = settings.max_time * 60 if settings.max_time else None
        timed_out = False
        try:
            _, stderr = process.communicate(timeout=app_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            timed_out = True
            jit_print(
                f"[bold yellow]Application timed out after {settings.max_time} min"
                f" (attempt {attempt + 1}/{_MAX_RETRIES}) — stopping and retrying...[/]"
            )

        # The timing result will be in stderr because 'time' outputs to stderr by default
        real_time = time.time() - start
        real_time = extract_accurate_time(settings, real_time)

        if not timed_out and process.returncode == 0:
            elapsed_time(settings, runtime, "App", real_time)
            _flush_log_event(settings.flush_log, "APP-END", name)
            break

        if not timed_out:
            err_content = ""
            if os.path.isfile(settings.app_err):
                with open(settings.app_err) as f:
                    err_content = f.read().strip()
            log_content = ""
            if os.path.isfile(settings.app_log):
                with open(settings.app_log) as f:
                    log_content = f.read().strip()

            jit_print(
                f"[bold yellow]Application failed (attempt {attempt + 1}/{_MAX_RETRIES},"
                f" returncode={process.returncode})[/]"
            )
            jit_print(f"[red]Error executing command: {call}[/]")
            if err_content:
                jit_print(f"[red]Error output:\n{err_content}[/]")

            combined = err_content + "\n" + log_content
            # GekkoFS daemon died mid-run: restart the whole infrastructure before retrying
            if (
                not settings.exclude_daemon
                and _has_gekko_connection_error(combined)
                and attempt < _MAX_RETRIES - 1
            ):
                jit_print(
                    "[bold yellow]GekkoFS I/O error detected — "
                    "restarting daemon, FUSE, re-staging and re-generating data before retry...[/]"
                )
                _cleanup_stale_gekko(settings)
                # Kill FTIO before re-staging so ZMQ messages from stage_in and
                # pre_call are never seen by the predictor.  Restarting FTIO
                # clears all accumulated state (b_app, t_app, hits, trigger)
                # and lets the ZMQ buffer drain while app_start_file is absent.
                if not settings.exclude_ftio:
                    shut_down(settings, "FTIO", settings.ftio_pid)
                start_gekko_daemon(settings)
                if settings.fuse:
                    start_fuse(settings)
                stage_in(settings, runtime)
                # Daemon restart wipes rootdir (--clean-rootdir), so any data
                # generated by pre_call into GekkoFS must be regenerated.
                pre_call(settings)
                # Restart FTIO fresh — start_ftio removes app_start_file
                # internally and waits until the predictor is ready, by which
                # point any buffered ZMQ messages from stage_in / pre_call have
                # been drained and discarded (flag was absent).
                if not settings.exclude_ftio:
                    start_ftio(settings)
                    if settings.app_start_file:
                        open(settings.app_start_file, "w").close()

        if attempt == _MAX_RETRIES - 1:
            elapsed_time(settings, runtime, "App", real_time)
            exit_routine(settings)

    if not settings.exclude_ftio and settings.exclude_cargo:
        jit_print("[cyan]Shuting down FTIO as application finished")
        shut_down(settings, "FTIO", settings.ftio_pid)

    os.chdir(original_dir)
    jit_print(f"Changing directory to {os.getcwd()}")

    jit_print("[green]############## Application finished ##############\n\n\n\n ")


#! Pre app call
#!##############################
def pre_call(settings: JitSettings) -> None:
    """Executes pre-application calls specified in the settings.

    Args:
        settings (JitSettings): jit settings
    """
    name = (
        settings.app_call.split("/", 1)[1]
        if "/" in settings.app_call
        else settings.app_call
    )

    if settings.pre_app_call:
        jit_print(
            f"[green bold]############## Pre-application Call [/][black][{get_time()}][/]"
        )
        for attempt in range(_MAX_RETRIES):
            if attempt > 0:
                jit_print(
                    f"[bold yellow]Retrying pre-application call"
                    f" (attempt {attempt + 1}/{_MAX_RETRIES})...[/]"
                )
                time.sleep(5)

            returncode = 0
            if isinstance(settings.pre_app_call, str):
                call = settings.pre_app_call
                if any(x in call for x in ["mpiex", "mpirun"]):
                    # call = flaged_call(settings, call, exclude=["ftio"])
                    all_procs = re.search(r"-np\s+(\d+)", call)
                    all_procs = int(all_procs.group(1))

                    # use a single node if everything fits on one node
                    if all_procs <= settings.procs_app:
                        app_nodes = 1
                        procs_per_node = all_procs
                    else:
                        # spread processes across available nodes
                        app_nodes = math.ceil(all_procs / settings.procs_app)
                        procs_per_node = math.ceil(all_procs / app_nodes)

                    if procs_per_node == 0:
                        jit_print("[red bold] Procs per node is 0, Adjusting to 4[/]")
                        print(
                            f"all_procs:{all_procs} | procs_per_node:{procs_per_node} | app_nodes:{app_nodes}"
                        )
                        procs_per_node = 4

                    call = flaged_call(
                        settings,
                        call,
                        app_nodes,
                        procs_per_node,
                        exclude=["ftio"],
                    )

                returncode = execute_block_and_monitor(
                    settings.verbose,
                    call,
                    settings.app_log,
                    settings.app_err,
                    settings.dry_run,
                    src=name,
                )
            elif isinstance(settings.pre_app_call, list):
                for call in settings.pre_app_call:
                    if any(x in call for x in ["mpiex", "mpirun"]):
                        call = flaged_call(settings, call, exclude=["ftio"])
                    returncode = execute_block_and_monitor(
                        settings.verbose,
                        call,
                        settings.app_log,
                        settings.app_err,
                        settings.dry_run,
                    )
                    if returncode != 0:
                        break

            if returncode == 0:
                break

            err_content = ""
            if os.path.isfile(settings.app_err):
                with open(settings.app_err) as f:
                    err_content = f.read().strip()

            jit_print(
                f"[bold yellow]Pre-application call failed"
                f" (attempt {attempt + 1}/{_MAX_RETRIES}, returncode={returncode})[/]"
            )
            if err_content:
                jit_print(f"[red]Error output:\n{err_content}[/]")

            if attempt == _MAX_RETRIES - 1:
                exit_routine(settings)

        # _ = execute_block_and_log(
        #     settings.pre_app_call, settings.app_log
        # )
        jit_print(
            "[green]############## Pre-application call finished ##############\n\n\n\n "
        )


#! Post app call
#!##############################
def post_call(settings: JitSettings) -> None:
    """Executes post-application calls specified in the settings.

    Args:
        settings (JitSettings): jit settings
    """
    if settings.post_app_call:
        jit_print(
            f"[green bold]############## Post-application Call [/][black][{get_time()}][/]"
        )
        additional_arguments = ""
        if isinstance(settings.post_app_call, str):
            call = f"{additional_arguments} {settings.post_app_call}"
            execute_block_and_monitor(
                settings.verbose,
                call,
                settings.app_log,
                settings.app_err,
                settings.dry_run,
            )
        elif isinstance(settings.post_app_call, list):
            call = ""
            for s in settings.post_app_call:
                call = f"{additional_arguments} {s}"
                execute_block_and_monitor(
                    settings.verbose,
                    call,
                    settings.app_log,
                    settings.app_err,
                    settings.dry_run,
                )
        jit_print(
            "[green]############## Post-application call finished ##############\n\n\n\n "
        )
