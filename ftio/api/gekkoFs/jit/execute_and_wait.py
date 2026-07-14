"""
This file provides various functions to execute shell commands and monitor their execution.
It includes functions to execute commands in blocking and non-blocking modes, log outputs,
monitor log files, and wait for specific lines or files to appear.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import multiprocessing
import os
import re
import subprocess
import time
from datetime import datetime

from rich.console import Console
from rich.status import Status

from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.logger import Logger
from ftio.api.gekkoFs.jit.setup_helper import (
    check,
    flaged_call,
    get_pid,
    jit_print,
)

console = Console()
TERMINAL_WIDTH = console.size.width
JIT_LOGGER = Logger(prefix="jit").get()
DAEMON_LOGGER = Logger(prefix="daemon").get()
PROXY_LOGGER = Logger(prefix="proxy").get()
FUSE_LOGGER = Logger(prefix="fuse").get()
DLIO_LOGGER = Logger(prefix="dlio").get()
FTIO_LOGGER = Logger(prefix="ftio").get()
ERROR_LOGGER = Logger(prefix="error").get()


def execute_block(call: str, raise_exception: bool = True, dry_run=False) -> str:
    """Executes a call and blocks till it is finished

    Args:
        call (str): bash call to execute
        raise_exception (bool): whether to raise an exception on failure
        dry_run (bool): if True, only print the call without executing

    Returns:
        str: output of the executed call
    """
    jit_print(f"Executing {call}")
    if dry_run:
        print(call)
        return ""

    out = ""
    try:
        # process = subprocess.Popen(call, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # remove capture_output=True to unblock
        out = subprocess.run(
            call,
            shell=True,
            text=True,
            capture_output=True,
            check=True,
            executable="/bin/bash",
            env=os.environ,
        )
        out = out.stdout
    except subprocess.CalledProcessError as e:
        error_message = (
            f"[red]Command failed:[/red]{call}\n"
            f"[red]Exit code:[/red]{e.returncode}\n"
            f"[red]Output:[/red]{e.stdout.strip()}\n"
            f"[red]Error:[/red]{e.stderr.strip()}"
        )
        console.print(f"[red]{error_message}\n[/]")
        if raise_exception:
            raise

    return out


def execute_block_and_log(call: str, log_file: str) -> float:
    """Executes a call and logs its output. This is a blocking call that
    writes the output to the log once finished.

    Args:
        call (str): bash call to execute
        log_file (str): absolute location of the log file

    Returns:
        float: execution time of the call
    """
    log_message = f" Executing command: {call}\n"
    jit_print("[cyan]" + log_message)
    start = time.time()
    end = start
    try:
        out = subprocess.run(
            call,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable="/bin/bash",
            env=os.environ,
        )
        end = time.time()
        log_message += f"Output:\n{out.stdout}\n"
    except subprocess.CalledProcessError as e:
        log_message += f"Error:\n{e.stderr}\n"
        error_message = (
            f"[red]Command failed:[/red]{call}\n"
            f"[red]Exit code:[/red]{e.returncode}\n"
            f"[red]Output:[/red]{e.stdout.strip()}\n"
            f"[red]Error:[/red]{e.stderr.strip()}"
        )
        console.print(f"[red]{error_message}[/]")
        raise
    finally:
        # Write the log message to the file
        with open(log_file, "a") as file:
            file.write(log_message)
    return end - start


def execute_block_and_monitor(
    verbose: bool,
    call: str,
    log_file: str = "",
    log_err_file: str = "",
    dry_run=False,
    src="",
) -> int:
    """Executes a call, monitors its log file, and waits for completion.

    Args:
        verbose (bool): whether to print verbose output
        call (str): bash call to execute
        log_file (str): log file to monitor
        log_err_file (str): error log file to monitor
        dry_run (bool): if True, only print the call without executing
        src (str): source label for colored log output

    Returns:
        int: process return code
    """
    if len(log_err_file) == 0:
        log_err_file = log_file

    process = execute_background(call, log_file, log_err_file, dry_run)
    if verbose:
        out = monitor_log_file(log_file, src)
        if log_err_file != log_file:
            err = monitor_log_file(log_err_file, src)

    process.communicate()
    if verbose:
        out.terminate()
        if log_err_file != log_file:
            err.terminate()

    return process.returncode


def execute_background(
    call: str, log_file: str = "", log_err_file: str = "", dry_run=False
):
    """Executes a call in the background and sets up a log directory.

    Args:
        call (str): call to execute
        log_file (str): log file to write output
        log_err_file (str): error log file to write errors
        dry_run (bool): if True, only print the call without executing

    Returns:
        subprocess.Popen: process object
    """
    jit_print(f"[cyan]Executing {call}")
    with open(log_file, "a") as file:
        file.write(f" Executing {call}")

    if dry_run:
        print(call)
        call = ""
        return subprocess.Popen(call, shell=True, executable="/bin/bash")

    # if log_file and log_err_file:
    #     call = f"{call}  {log_file} 2 {log_err_file}"
    # elif log_file:
    #     call = f"{call}  {log_file}"
    # else:
    #     pass
    # print(call)
    # process = subprocess.Popen(call, shell=True, preexec_fn=os.setpgrp,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # start_new_session=True puts each background component in its own session,
    # so process.pid is a process-group leader. This lets the teardown path
    # (kill_process_tree/shut_down) reap the whole subtree in one call -- notably
    # the predictor_jit worker pool and the bash -c wrapper's child daemon --
    # and it keeps working even after the parent has died (group membership
    # survives reparenting to init).
    if log_file and log_err_file:
        with open(log_file, "a") as log_out, open(log_err_file, "w") as log_err:
            process = subprocess.Popen(
                call,
                shell=True,
                executable="/bin/bash",
                stdout=log_out,
                stderr=log_err,
                env=os.environ,
                start_new_session=True,
            )
    elif log_file:
        with open(log_file, "a") as log_out:
            process = subprocess.Popen(
                call,
                shell=True,
                executable="/bin/bash",
                env=os.environ,
                stdout=log_out,
                stderr=log_out,
                start_new_session=True,
            )
    else:
        process = subprocess.Popen(
            call,
            shell=True,
            executable="/bin/bash",
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    return process


def execute_background_and_log(
    settings: JitSettings,
    call: str,
    log_file: str,
    name="",
    err_file: str = "",
) -> subprocess.Popen:
    """Executes a call in the background and returns the process. The output is displayed using a
    thread that reads the log file.

    Args:
        settings (JitSettings): jit settings
        call (str): bash call to execute
        log_file (str): absolute location of the log file to monitor
        name (str): source of the file for colored output
        err_file (str): error log file

    Returns:
        subprocess.Popen: process object
    """
    process = execute_background(call, log_file, err_file, settings.dry_run)
    get_pid(settings, name, process.pid)

    _ = monitor_log_file(log_file, name)
    return process


def execute_background_and_log_in_process(
    call: str, log_file: str, name="", err_file: str = "", dry_run=False
):
    """Executes a call in the background and returns the process. The output is displayed using a
    thread that reads the log file.

    Args:
        call (str): bash call to execute
        log_file (str): absolute location of the log file to monitor
        name (str): source of the file for colored output
        err_file (str): error log file
        dry_run (bool): if True, only print the call without executing
    """
    if dry_run:
        print(call)
        call = ""

    process = execute_background(call, log_file, err_file, dry_run)
    # get_pid(settings, name, process.pid)
    # daemon is noisy
    _ = monitor_log_file(log_file, src=name)
    _, stderr = process.communicate()
    if process.returncode != 0:
        console.print(f"[red]Error executing command:{call}")
        console.print(f"[red]Error was:\n{stderr}")
    else:
        pass


def execute_block_and_wait_line(call: str, filename: str, target_line: str) -> None:
    """Executes a call and waits for a line to appear in a file.

    Args:
        call (str): bash call to execute
        filename (str): file to monitor for target_line to appear
        target_line (str): target line that needs to appear in filename to stop the execution
    """
    execute_block(call)
    _ = wait_for_line(filename, target_line)


def execute_background_and_wait_line(
    call: str, filename: str, target_line: str, dry_run: bool = False
) -> None:
    """Executes a call in the background and waits for a line to appear in a file.

    Args:
        call (str): bash call to execute
        filename (str): file to monitor for target_line to appear
        target_line (str): target line that needs to appear in filename to stop the execution
        dry_run (bool): if True, only print the call without executing
    """
    process = execute_background(call, filename, dry_run=dry_run)
    _ = monitor_log_file(filename, "")
    if not dry_run:
        _ = wait_for_line(filename, target_line)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            console.print(f"[red]Error executing command:{call}")
            console.print(f"[red]Error was:\n{stderr}")
        else:
            # console.print(stdout, style="bold green")
            pass


def monitor_log_file(file: str, src: str = "") -> multiprocessing.Process:
    """Monitors a file and displays its output on the console. A process is
    in charge of monitoring the file.

    Args:
        file (str): absolute file path
        src (str): source of the file for colored output

    Returns:
        multiprocessing.Process: process object
    """
    monitor_process = multiprocessing.Process(target=print_file, args=(file, src))
    monitor_process.daemon = True
    monitor_process.start()

    return monitor_process


def end_of_transfer(
    settings: JitSettings,
    log_file: str,
    call: str,
    monitored_files: list[str] = None,
) -> None:
    """Monitors the end of a transfer process by checking log files.

    Args:
        settings (JitSettings): jit settings
        log_file (str): log file to monitor
        call (str): bash call to execute if stuck
        monitored_files (list[str]): list of files to monitor
    """
    if monitored_files is None:
        monitored_files = []
    if settings.dry_run:
        return

    online = False
    stuck = False
    hits = 0
    limit = 100
    if len(monitored_files) == 0:
        monitored_files = get_files(settings, True)
        online = True
        stuck = True

    last_lines = read_last_n_lines(log_file)
    n = len(monitored_files)
    if "Transfer finished for []" in last_lines:
        # Nothing to move
        return
    elif n == 0:
        return
    else:
        with open(log_file) as file:
            # Move to the end of the file
            file.seek(0, 2)
            last_pos = file.tell()
            buffer = ""
            with Status(
                f"[cyan]Waiting for end of transfer...monitoring {n} files",
                console=console,
            ) as status:
                while len(monitored_files) > 0:
                    time.sleep(0.1)  # Short sleep interval to quickly catch new lines

                    # Check if the file has grown
                    current_pos = file.tell()
                    if current_pos < last_pos:
                        # Log file was truncated or rotated, reset
                        file.seek(0, 2)
                        last_pos = file.tell()

                    # Read new content
                    file.seek(last_pos)
                    new_data = file.read()
                    last_pos = file.tell()

                    # Process the new data
                    if new_data:
                        buffer += new_data
                        lines = buffer.splitlines()

                        # Check if the last line is incomplete
                        if not new_data.endswith("\n"):
                            buffer = lines.pop()  # Save incomplete line in the buffer
                        else:
                            buffer = ""

                        check(settings)
                        # Process each line
                        for line in lines:
                            content = line.strip()
                            # jit_print(">>[cyan]**{content}")
                            if "Deleting" in content:
                                # Check if any of the monitored files are mentioned in the line
                                for name in monitored_files:
                                    if name in content:
                                        monitored_files.remove(name)
                                        jit_print(
                                            f"[cyan]finished moving '{name}'. Remaining files ({len(monitored_files)})",
                                            True,
                                        )
                                        console.print(
                                            f"Waiting for {len(monitored_files)} more files to be deleted: {monitored_files}"
                                        )
                                        jit_print(
                                            f"[cyan]Files in mount dir {monitored_files}",
                                            True,
                                        )
                                        status.update(
                                            f"Waiting for {len(monitored_files)} more files to be deleted: {monitored_files}"
                                        )
                                        hits = 0
                    if online:
                        # avoids dead loop for disappearing files
                        monitored_files = get_files(settings, False)
                        hits += 1
                        status.update(
                            f"Waiting for {len(monitored_files)} more files to be deleted [yellow]({hits}/{limit})[/]: {monitored_files}"
                        )
                        if hits > 4 and stuck:
                            jit_print("[cyan]Stuck? Triggering cargo again\n")
                            _ = execute_background(
                                call,
                                settings.cargo_log,
                                settings.cargo_err,
                            )
                            stuck = False
                        if hits > limit:
                            jit_print("[cyan]Stopping stage out\n")
                            return
                # All monitored files have been processed
                timestamp = get_time()
                status.update(
                    f"\n[bold green]JIT [cyan]finished moving all files at  [{timestamp}]"
                )
                jit_print(
                    f"\n[cyan]finished moving all files at [{timestamp}]",
                    True,
                )


def end_of_transfer_online(
    settings: JitSettings, log_file: str, call: str, timeout=240
) -> None:
    """Monitors the end of a transfer process by checking log files with a timeout.

    Args:
        settings (JitSettings): jit settings
        log_file (str): log file to monitor
        call (str): bash call to execute if stuck
        timeout (int): timeout in seconds
    """
    if settings.dry_run:
        return

    repeated_trigger = True
    monitored_files = get_files(settings, True)
    stuck_time = 5
    last_lines = read_last_n_lines(log_file)
    stuck_files = 0
    start = time.time()
    n = len(monitored_files)
    if "Transfer finished for []" in last_lines:
        # Nothing to move
        return
    elif n == 0:
        return
    else:
        with Status(
            f"[cyan]Waiting for end of transfer...monitoring {n} files",
            console=console,
        ) as status:
            start_time = time.time()
            last_cargo_time = start_time
            hit = 0
            while len(monitored_files) > 0:
                time.sleep(0.1)  # Short sleep interval to quickly catch new lines

                if n < 10:
                    status.update(
                        f"Waiting for {len(monitored_files)} files: {monitored_files}"
                    )
                else:
                    status.update(
                        f"Waiting for {len(monitored_files)} files ({time.time() - start:.2f} sec)"
                    )

                passed_time = int(time.time() - start_time)
                time_since_last_cargo = int(time.time() - last_cargo_time)

                if passed_time >= timeout:
                    status.update("Timeout reached")
                    jit_print("[bold red]Timeout reached[/]")
                    break
                if repeated_trigger:
                    hit += 1
                    # jit_print(f"{time_since_last_cargo} >= {stuck_time} and {stuck_files} == {len(monitored_files)}")
                    if time_since_last_cargo >= stuck_time and stuck_files == len(
                        monitored_files
                    ):
                        jit_print(
                            f"[cyan]Stucked for {stuck_time} sec. Triggering cargo again\n"
                        )
                        _ = execute_background(
                            call, settings.cargo_log, settings.cargo_err
                        )
                        last_cargo_time = time.time()
                        stuck_time = stuck_time * 2
                        jit_print(f"[cyan]Stucked increased to {stuck_time}\n")
                        if n < 10:
                            status.update(
                                f" Waiting for {len(monitored_files)} more files to be deleted: {monitored_files}"
                            )
                        else:
                            status.update(
                                f" Waiting for {len(monitored_files)} more files to be deleted"
                            )
                        hit = 0
                        if "Transfer finished for []" in last_lines:
                            break
                    if hit == 3:
                        stuck_files = len(monitored_files)
                        hit = 0
                monitored_files = get_files(settings, False)

            timestamp = get_time()
            status.update(
                f"\n[bold green]JIT [cyan]finished moving all files at [{timestamp}] after {time.time() - start:.2f} seconds"
            )
            jit_print(
                f"\n[cyan]finished moving all files at [{timestamp}] after {time.time() - start:.2f} seconds",
                True,
            )
    print("\n")


def get_files(settings: JitSettings, verbose=True):
    """Gets the list of files to be monitored.

    Args:
        settings (JitSettings): jit settings
        verbose (bool): whether to print verbose output

    Returns:
        list[str]: list of files to be monitored
    """
    monitored_files = []
    files = []
    try:
        # try:
        #     command_ls = flaged_call(settings, f" find {settings.gkfs_mntdir}", exclude=["ftio"])
        #     files = subprocess.check_output(command_ls, shell=True, text=True)
        # except subprocess.CalledProcessError as e:
        #     if settings.debug_lvl > 1:
        #         console.print(f"[red] Error using find, trying ls:\n{e}[/]")
        #     else:
        #         console.print("[red] Error using find, trying ls[/]"
        command_ls = flaged_call(
            settings, f" ls -R {settings.gkfs_mntdir}", exclude=["ftio"]
        )
        files = subprocess.check_output(command_ls, shell=True, text=True)
        if files:
            files = files.splitlines()
            files = [f for f in files if "LIBGKFS" not in f]
            files = [
                file.replace(f"{settings.gkfs_mntdir}", "")
                for file in files
                if file.replace(f"{settings.gkfs_mntdir}", "")
            ]
            # remove directories
            files = [item for item in files if "." in item]

        # find monitored files
        monitored_files = files_filtered(files, settings.regex_match, verbose)
        if settings.debug_lvl > 1:
            # try:
            #     cmd = flaged_call(settings, f" du -sh {settings.gkfs_mntdir}", exclude=["ftio"])
            #     # cmd = flaged_call(settings, f" du -sh {settings.gkfs_mntdir} | cut -f1", exclude=["ftio"])
            #     file_size = subprocess.check_output(cmd, shell=True, text=True)
            #     console.print(f"\n[cyan]Files are ({file_size}):\n{files}[/]")
            # except subprocess.CalledProcessError:
            console.print(f"\n[cyan]Files are:\n{files}[/]")
        if verbose or settings.debug_lvl > 2:
            timestamp = get_time()
            console.print(f"[cyan]Files that need to be stage out: [{timestamp}][/]")
            for i, f in enumerate(monitored_files):
                console.print(f"[cyan]{i}. {f}[/]")

    except Exception as e:
        console.print(f"[red] Error listing files script:\n{e}[/]")

    return monitored_files


def get_time():
    """Gets the current time formatted as a string.

    Returns:
        str: current time formatted as 'YYYY-MM-DD HH:MM:SS.sss'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def files_filtered(list_of_files: list[str], regex_pattern, verbose=True) -> list[str]:
    """Filters a list of files based on a regex pattern.

    Args:
        list_of_files (list[str]): list of files to filter
        regex_pattern (str): regex pattern to match files
        verbose (bool): whether to print verbose output

    Returns:
        list[str]: filtered list of files
    """
    monitored_files = []
    if list_of_files:
        if regex_pattern:
            # check if it contains file name without path (less error prone)
            while "/" in regex_pattern:
                regex_pattern = regex_pattern.split("/", 1)[1]
            if verbose:
                jit_print(f"[cyan]Cleaned Regex pattern to: {regex_pattern} ")
        regex = re.compile(regex_pattern)

        for f in list_of_files:
            if regex.match(f):
                # jit_print(f"[yellow]Ignoring: {f} ")
                monitored_files.append(f)

    return monitored_files


#
# def print_file(file, src=""):
#     """Continuously monitors the log file for new lines and prints them.
#
#     Args:
#         file (str): absolute file path
#         src (str): source of the file for colored output
#     """
#     color = ""
#     close = ""
#     newline = True
#     wait_time = 0.05
#     if src:
#         if "daemon" in src.lower():
#             color = "[deep_pink1]"
#             wait_time = 0.1
#         elif "proxy" in src.lower():
#             color = "[purple4]"
#             wait_time = 0.1
#         elif any(keyword in src.lower() for keyword in ["dlio", "lammp"]):
#             color = "[gold3]"
#             wait_time = 0.1
#         elif "ftio" in src.lower():
#             color = "[deep_sky_blue1]"
#             wait_time = 0.1
#         elif "error" in src.lower():
#             color = "[red]"
#             wait_time = 0.1
#
#         if color:
#             close = "[/]"
#             newline = "\n"
#
#     while not os.path.exists(file):
#         if "error" in src.lower():
#             time.sleep(0.1)
#         else:
#             with console.status(
#                 f"[bold green]Waiting for {file} to appear ..."
#             ) as status:
#                 time.sleep(0.1)
#
#     with open(file, "r") as file:
#         # Go to the end of the file
#         file.seek(0, os.SEEK_END)
#         buffer = []
#         last_print_time = time.time()
#
#         while True:
#             line = file.readline()
#             if line:
#                 buffer.append(line.rstrip())
#             else:
#                 # If there's no new line, wait briefly
#                 time.sleep(wait_time)
#
#             # Group and print the buffered lines every 0.1 seconds
#             current_time = time.time()
#             if current_time - last_print_time >= wait_time and buffer:
#                 # Print grouped lines
#                 content = "\n".join(buffer)
#                 buffer.clear()
#                 last_print_time = current_time
#
#                 if not src or "cargo" in src:
#                     print(content)
#                 else:
#                     if newline:
#                         console.print("\n", end="")
#                     # console.print(
#                     #     Panel(
#                     #         color  + escape(content) + close,
#                     #         title= color + src.capitalize() + close,
#                     #         style="white",
#                     #         border_style="white",
#                     #         title_align="left",
#                     #         width=TERMINAL_WIDTH
#                     #     )
#                     # )
#                     console.print(
#                         Panel.fit(
#                             color + escape(content) + close,
#                             title=color + src.upper() + close,
#                             style="white",
#                             border_style="white",
#                             title_align="left",
#                         )
#                     )


def print_file(file, src=""):
    """Continuously monitors the log file for new lines and prints them.

    Args:
        file (str): absolute file path
        src (str): source of the file for colored output
    """
    logger = JIT_LOGGER
    newline = True
    wait_time = 0.0
    if src:
        if "daemon" in src.lower():
            logger = DAEMON_LOGGER
            wait_time = 0.2

        elif "proxy" in src.lower():
            logger = PROXY_LOGGER
            wait_time = 0.2

        if "fuse" in src.lower():
            logger = FUSE_LOGGER
            wait_time = 0.2

        elif any(keyword in src.lower() for keyword in ["dlio", "lammp"]):
            logger = DLIO_LOGGER
            wait_time = 0.2

        elif "ftio" in src.lower():
            logger = FTIO_LOGGER
            wait_time = 0.2

        elif "error" in src.lower():
            logger = ERROR_LOGGER
            wait_time = 0.2

    while not os.path.exists(file):
        if "error" in src.lower():
            time.sleep(0.1)
        else:
            with console.status(f"[bold green]Waiting for {file} to appear ..."):
                time.sleep(0.1)

    with open(file) as file:
        # Go to the end of the file
        file.seek(0, os.SEEK_END)
        buffer = []
        last_print_time = time.time()

        while True:
            line = file.readline()
            if line:
                buffer.append(line.rstrip())
            else:
                # If there's no new line, wait briefly
                time.sleep(wait_time)

            # Group and print the buffered lines every 0.1 seconds
            current_time = time.time()
            if current_time - last_print_time >= wait_time and buffer:
                # Print grouped lines
                # content = "\n".join(buffer)
                content = "\n".join(buffer).rstrip()
                buffer.clear()
                last_print_time = current_time

                if not src or "cargo" in src:
                    print(content)
                else:
                    if "error" in src.lower() and not any(
                        keyword in src.lower() for keyword in ["dlio", "lammp"]
                    ):
                        logger.error("\n" + content)
                    elif (
                        any(keyword in src.lower() for keyword in ["dlio", "lammp"])
                        or newline
                        or "Bandwidth Plot" in content
                    ):
                        logger.info("\n" + content)
                    else:
                        logger.info(content)


_FATAL_PATTERNS = (
    "transport endpoint",
    "address already in use",
    "no such file or directory",
    "permission denied",
    "terminated",
    "software caused connection abort",
)


def _has_fatal_error(text: str) -> bool:
    lower = text.lower()
    return any(pat in lower for pat in _FATAL_PATTERNS)


def wait_for_file(
    filename: str,
    timeout: int = 180,
    dry_run: bool = False,
    error_file: str | None = None,
) -> bool:
    """Waits for a file to be created.

    Args:
        filename (str): absolute file path
        timeout (int): timeout in seconds
        dry_run (bool): if True, only print the call without executing
        error_file (str | None): optional stderr/error log to monitor for fatal
            errors; if new content matching a fatal pattern is detected the
            function returns False immediately without waiting for the timeout

    Returns:
        bool: True if the file was created, False if timeout or fatal error
    """
    if dry_run:
        return True

    start_time = time.time()
    err_offset = (
        os.path.getsize(error_file) if error_file and os.path.isfile(error_file) else 0
    )

    with Status(
        f"[cyan]Waiting for {filename} to be created...\n", console=console
    ) as status:
        while not os.path.isfile(filename):
            passed_time = int(time.time() - start_time)
            if passed_time >= timeout:
                status.update("Timeout reached")
                jit_print("[bold red] Timeout reached[/]")
                return False

            if error_file and os.path.isfile(error_file):
                current_size = os.path.getsize(error_file)
                if current_size > err_offset:
                    with open(error_file) as ef:
                        ef.seek(err_offset)
                        new_content = ef.read()
                    err_offset = current_size
                    if _has_fatal_error(new_content):
                        jit_print(
                            f"[bold red]Fatal error in {error_file}:[/]\n{new_content.strip()}"
                        )
                        return False

            status.update(
                f"[cyan]Waiting for {filename} to be created... ({passed_time}/{timeout}) s"
            )
            time.sleep(0.1)

        # When the file is created, update the status
        status.update(f"{filename} has been created.\n")
        print("\n")
        jit_print(f"{filename} has been created.")
    return True


def wait_for_line(
    filename: str,
    target_line: str,
    msg: str = "",
    timeout: int = 60,
    dry_run: bool = False,
    occurrences: int = 1,
    error_file: str | None = None,
) -> bool:
    """Waits for a specific line to appear in a log file.

    Args:
        filename (str): path to the log file
        target_line (str): line of text to wait for
        msg (str): message to print
        timeout (int): maximal timeout
        dry_run (bool): if True, only print the call without executing
        occurrences (int): number of times the target line must appear
        error_file (str | None): optional stderr/error log to monitor for fatal
            errors; if new content matching a fatal pattern is detected the
            function returns False immediately without waiting for the timeout

    Returns:
        bool: True if the line appeared, False if timeout or fatal error reached
    """
    success = True
    if dry_run:
        return success
    start_time = time.time()
    if not msg:
        msg = "Waiting for line to appear..."

    err_offset = (
        os.path.getsize(error_file) if error_file and os.path.isfile(error_file) else 0
    )

    while not os.path.exists(filename):
        with console.status(
            f"[bold green]Waiting for {filename} to appear ..."
        ) as status:
            time.sleep(0.1)

    with open(filename) as file:
        # Move to the end of the file to start monitoring
        # file.seek(0, 2)  # Go to the end of the file and look at the last 10 entris
        try:
            file.seek(-2, os.SEEK_END)
        except Exception:
            file.seek(0, 0)  # Go to the end of the file and look at the last 10 entris

        count = 0  # NEW: occurrence counter

        with Status(f"[cyan]{msg}", console=console) as status:
            while True:
                line = file.readline()
                if not line:
                    # If no line, wait and check again
                    time.sleep(0.01)
                    passed_time = int(time.time() - start_time)
                    if passed_time >= timeout:
                        status.update("Timeout reached.")
                        jit_print("[bold red] Timeout reached[/]", True)
                        success = False
                        break

                    if error_file and os.path.isfile(error_file):
                        current_size = os.path.getsize(error_file)
                        if current_size > err_offset:
                            with open(error_file) as ef:
                                ef.seek(err_offset)
                                new_content = ef.read()
                            err_offset = current_size
                            if _has_fatal_error(new_content):
                                jit_print(
                                    f"[bold red]Fatal error in {error_file}:[/]\n{new_content.strip()}"
                                )
                                success = False
                                break

                    status.update(f"[cyan]{msg} ({passed_time}/{timeout})")
                    continue

                if target_line in line:
                    count += 1

                    if count >= occurrences:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        status.update(
                            f"Found target line {count}/{occurrences}: '{target_line}'"
                        )
                        jit_print(f"[cyan]Stopped waiting at [{timestamp}]")
                        break
                    else:
                        status.update(
                            f"Found {count}/{occurrences} occurrences of '{target_line}'"
                        )

        return success


def read_last_n_lines(filename, n=3):
    """Reads the last `n` lines of a file.

    Args:
        filename (str): path to the file
        n (int): number of lines to read

    Returns:
        list[str]: list of the last `n` lines
    """
    with open(filename, "rb") as f:
        # Move the pointer to the end of the file
        f.seek(0, 2)
        file_size = f.tell()

        # Initialize variables
        lines = []
        buffer = b""
        pointer = file_size

        while pointer > 0 and len(lines) < n:
            # Move the pointer backwards by one byte and read
            pointer -= 1
            f.seek(pointer)
            buffer = f.read(1) + buffer

            # If a newline is found, add the line to the list
            if buffer.startswith(b"\n") or pointer == 0:
                lines.append(buffer.decode("utf-8"))
                buffer = b""

        return lines[-n:]
