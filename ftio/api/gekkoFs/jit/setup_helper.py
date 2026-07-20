"""
This module contains helper functions for setting up and managing the JIT environment.
It includes functions for checking ports, parsing options, allocating resources,
handling signals, and managing various components like FTIO, GekkoFS, and Cargo.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

from rich.console import Console

from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.jittime import JitTime
from ftio.api.gekkoFs.jit.logger import Logger

console = Console()
jit_logger = Logger(prefix="JIT").get()


def is_port_in_use(port_number: int | str) -> bool:
    """
    Check if a given port is in use.

    Args:
        port_number (int | str): The port number to check.

    Returns:
        bool: True if the port is in use, False otherwise.
    """
    out = False
    try:
        # Run the netstat command and search for the port number
        netstat_output = subprocess.check_output(
            f"netstat -tlpn | grep ':{port_number} '",
            shell=True,
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        # If output is found, the port is in use
        if netstat_output:
            jit_print(
                f"[bold red]Error: Port {port_number} is already in use...[/bold red]"
            )
            out = True  # Port is in use
    except subprocess.CalledProcessError:
        # grep returns a non-zero exit status if it finds no matches
        jit_print(f"[bold blue]Port {port_number} is available.[/bold blue]")
        out = False  # Port is free

    return out


def check_port(settings: JitSettings) -> None:
    """
    Check if a port is available and terminate any existing process using it.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if is_port_in_use(settings.port_ftio):
        jit_print(
            f"[bold red]Error: Port {settings.port_ftio} is already in use on {settings.address_ftio}. Terminating existing process...[/]"
        )

        # Identify the process ID using the settings.port_ftio
        try:
            process_output = (
                subprocess.check_output(
                    f"netstat -nlp | grep :{settings.port_ftio} ", shell=True
                )
                .decode()
                .strip()
            )
            process_id = process_output.split()[6].split("/")[0]

            if process_id:
                jit_print(f"[bold  yellow]Terminating process with PID: {process_id}[/]")
                os.kill(int(process_id), 9)
                jit_print(
                    f"[bold  green]Using port {settings.port_ftio} on {settings.address_ftio}.[/]"
                )
            else:
                jit_print(
                    f"[bold red] Failed to identify process ID for PORT {settings.port_ftio}.[/]"
                )
                exit(1)
        except subprocess.CalledProcessError as e:
            jit_print(f"[bold red] Failed to retrieve process information: {e}[/]")
            exit(1)
    else:
        jit_print(
            f"[bold  green]Using port {settings.port_ftio} on {settings.address_ftio}.[/]"
        )


def parse_options(settings: JitSettings, args: list[str]) -> None:
    """
    Parse command-line options and update the JIT settings.

    Args:
        settings (JitSettings): The JIT settings object.
        args (list[str]): List of command-line arguments.
    """
    parser = argparse.ArgumentParser(description="JIT Setup Script")

    # Define command-line arguments with descriptions
    parser.add_argument(
        "-a",
        "--app",
        type=str,
        help="App to execute. Supported: dlio, lammps, wacom, nek5000, ior, haccio, or s3d.",
        default="",
    )
    parser.add_argument(
        "--app-flags",
        type=str,
        default="",
        help='Application-specific flags as a string (e.g., "600 600 600 6 6 6 0 F .").',
    )

    parser.add_argument(
        "-n", "--nodes", type=int, help="Number of nodes to run the setup."
    )
    parser.add_argument(
        "-t", "--max-time", type=int, help="Max execution time in minutes."
    )
    parser.add_argument(
        "-j",
        "--job-id",
        type=int,
        help="Use existing job ID instead of allocating new resources.",
    )
    parser.add_argument(
        "-l", "--log-name", type=str, help="Directory name for storing logs."
    )
    parser.add_argument(
        "-i",
        "--install_location",
        type=str,
        help="Install everything in the given directory.",
    )

    def _parse_procs(value: str) -> int:
        """Accept plain integers or SLURM N(xM) notation, returning N."""
        m = re.match(r"^(\d+)(?:\(x\d+\))?$", value.strip())
        if not m:
            raise argparse.ArgumentTypeError(f"invalid procs value: '{value}'")
        return int(m.group(1))

    parser.add_argument(
        "-c",
        "--total_procs",
        type=_parse_procs,
        help="Default number of procs if --procs_list is omitted. Accepts plain integers or SLURM N(xM) notation.",
    )
    parser.add_argument(
        "-o", "--omp_threads", type=int, help="Number of OpenMP threads used."
    )
    parser.add_argument(
        "-p",
        "--procs_list",
        type=str,
        help="Comma-separated list specifying procs per node/cpu for app, daemon, proxy, cargo, and ftio.",
    )
    parser.add_argument(
        "-f",
        "--ftio_args",
        type=str,
        help='FTIO arguments as a string (e.g., "--freq 10 -v -e no").',
    )
    parser.add_argument("--address", type=str, help="Address where FTIO is executed.")
    parser.add_argument("--port", type=str, help="Port for FTIO and GekkoFS.")

    parser.add_argument(
        "--gkfs_daemon",
        type=str,
        help="Absolute location to GekkoFS demon",
    )
    parser.add_argument(
        "--gkfs_intercept",
        type=str,
        help="Absolute location to GekkoFS **syscall** intercept. If libc is used, the variable i internally changed",
    )
    parser.add_argument(
        "--gkfs-daemon-protocol",
        type=str,
        choices=["ofi+verbs", "ofi+sockets"],
        help="Protocol for GekkoFS daemon (ofi+verbs or ofi+sockets).",
    )
    parser.add_argument(
        "--handle_new_prediction",
        type=str,
        help="Adaptive flag for flushing",
        default="cancel",
        choices={"skip", "cancel", ""},
    )
    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        help="Exclude specific tools: ftio, daemon, proxy, gkfs (daemon + proxy), cargo, or all.",
    )
    parser.add_argument(
        "-x",
        "--exclude-all",
        action="store_true",
        help="Exclude FTIO, GekkoFs, and Cargo.",
    )
    parser.add_argument(
        "-r",
        "--dry_run",
        action="store_true",
        help="Do not execute tools and app.",
    )
    parser.add_argument(
        "-d", "--debug", type=int, help="Debug level for additional info."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show output of each step.",
    )
    parser.add_argument(
        "-y",
        "--skip_confirm",
        action="store_true",
        help="Automatically cancel running JIT jobs.",
    )
    parser.add_argument(
        "-u",
        "--use_mpirun",
        action="store_true",
        help="Use mpirun instead of srun.",
    )
    parser.add_argument(
        "-s",
        "--use_syscall",
        action="store_true",
        help="GekkoFS uses syscall instead of libc.",
    )
    parser.add_argument(
        "-m",
        "--use_mem",
        action="store_true",
        help="GekkoFS uses nodelocal space (default) or memory (is this flag is passed).",
    )
    parser.add_argument(
        "--ignore_mtime",
        action="store_true",
        help="If set, mtime is ignored during flushing process",
    )
    parser.add_argument(
        "--parallel_move_threads",
        type=int,
        default=0,
        help="If specified, files are moved with multiples threads calls at the same time",
    )
    parser.add_argument(
        "--lock_generator",
        action="store_true",
        help="If set, at open/create we create an extra .lockgekko file with size = number of opens to that file (it is distributed). We decrease and delete the file on close",
    )
    parser.add_argument(
        "--lock_consumer",
        action="store_true",
        help="If set, At Open we wait until (40 seconds~) for the lock file to disappear. No modifications needed on the client, it is transparent.",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="If set, FUSE is used.",
    )
    parser.add_argument(
        "--fuse-idle-threads",
        dest="fuse_idle_threads",
        type=int,
        default=0,
        help=(
            "Override -o max_idle_threads=N passed to fuse_client. "
            "0 = auto (max(16, procs_app * 4)): multiplies tasks-per-node by 4 "
            "to cover internal reader threads from frameworks like TensorFlow/PyTorch. "
            "Lower values reduce concurrent Mercury RPCs and help avoid IB QP depth exhaustion."
        ),
    )
    parser.add_argument(
        "--preload-export",
        dest="preload_via_export",
        action="store_true",
        help=(
            "If set, LD_PRELOAD is passed via --export/mpiexec -x (legacy cluster behaviour). "
            "By default the intercept library is injected inside a 'bash -c' wrapper so that "
            "SLURM does not propagate LD_PRELOAD to unrelated processes."
        ),
    )

    parsed_args = parser.parse_args(args)

    # Assign parsed arguments to settings
    if parsed_args.app:
        settings.app = parsed_args.app
    if parsed_args.app_flags:
        settings.app_flags = parsed_args.app_flags.strip()
    if parsed_args.nodes:
        settings.nodes = parsed_args.nodes
    if parsed_args.max_time:
        settings.max_time = parsed_args.max_time
    if parsed_args.job_id:
        settings.job_id = parsed_args.job_id
        settings.static_allocation = True
    if parsed_args.log_name:
        settings.log_dir = parsed_args.log_name
    if parsed_args.install_location:
        settings.install_location = parsed_args.install_location
        install_all(settings)
    if parsed_args.total_procs:
        settings.procs = parsed_args.total_procs
    if parsed_args.omp_threads:
        settings.omp_threads = parsed_args.omp_threads
    if parsed_args.procs_list:
        try:
            procs_list = [int(proc) for proc in parsed_args.procs_list.split(",")]
        except ValueError:
            jit_logger.error(
                "Invalid --procs value. Must be a comma-separated list of numbers.[/]"
            )
            sys.exit(1)

        if len(procs_list) > 5:
            jit_logger.error("Too many values for --procs. Maximum is 5.[/]")
            sys.exit(1)

        if len(procs_list) > 0:
            settings.procs_app = procs_list[0]
        if len(procs_list) > 1:
            settings.procs_daemon = procs_list[1]
        if len(procs_list) > 2:
            settings.procs_proxy = procs_list[2]
        if len(procs_list) > 3:
            settings.procs_cargo = procs_list[3]
            if settings.procs_cargo < 2:
                jit_print("[bold  yellow]Warning: Not recommended to set Cargo < 2[/]")
        if len(procs_list) > 4:
            settings.procs_ftio = procs_list[4]

    if parsed_args.ftio_args:
        settings.ftio_args = parsed_args.ftio_args.strip()
    if parsed_args.address:
        settings.address_ftio = parsed_args.address
    if parsed_args.port:
        settings.port_ftio = parsed_args.port
    if parsed_args.gkfs_daemon_protocol:
        settings.gkfs_daemon_protocol = parsed_args.gkfs_daemon_protocol
    if parsed_args.handle_new_prediction:
        settings.handle_new_prediction = parsed_args.handle_new_prediction

    if parsed_args.exclude:
        jit_print("[bold  yellow]Excluding: [/]")
        excludes = parsed_args.exclude.split(",")
        for exclude in excludes:
            exclude = exclude.lower()
            if exclude == "ftio":
                settings.exclude_ftio = True
                console.print("[yellow]- ftio[/]")
            elif exclude == "cargo":
                settings.exclude_cargo = True
                console.print("[yellow]- cargo[/]")
            elif exclude in ("gkfs", "daemon", "proxy"):
                if exclude == "gkfs":
                    settings.exclude_daemon = True
                    settings.exclude_proxy = True
                    console.print("[yellow]- gkfs[/]")
                elif exclude == "daemon":
                    settings.exclude_daemon = True
                    console.print("[yellow]- daemon[/]")
                elif exclude == "proxy":
                    settings.exclude_proxy = True
                    console.print("[yellow]- proxy[/]")
            elif exclude == "all":
                settings.exclude_all = True
                console.print("[yellow]- all[/]")
            else:
                jit_print(f"[bold red] Invalid exclude option: {exclude} [/]")
                sys.exit(1)

    if parsed_args.gkfs_daemon:
        settings.parsed_gkfs_daemon = parsed_args.gkfs_daemon
    if parsed_args.gkfs_intercept:
        settings.parsed_gkfs_intercept = parsed_args.gkfs_intercept
    if parsed_args.parallel_move_threads > 0:
        settings.parallel_move_threads = parsed_args.parallel_move_threads
    if parsed_args.exclude_all:
        settings.exclude_all = True
    if parsed_args.dry_run:
        settings.dry_run = True
    if parsed_args.verbose:
        settings.verbose = True
    if parsed_args.debug is not None:
        settings.debug_lvl = parsed_args.debug
    if parsed_args.skip_confirm:
        settings.skip_confirm = True
    if parsed_args.use_mpirun:
        settings.use_mpirun = True
    if parsed_args.use_syscall:
        settings.gkfs_use_syscall = True
    if parsed_args.use_mem:
        settings.node_local = False
    if parsed_args.ignore_mtime:
        settings.ignore_mtime = True
    if parsed_args.lock_generator:
        settings.lock_generator = True
    if parsed_args.lock_consumer:
        settings.lock_consumer = True
    if parsed_args.fuse:
        settings.fuse = True
    if parsed_args.fuse_idle_threads:
        settings.fuse_idle_threads = parsed_args.fuse_idle_threads
    if parsed_args.preload_via_export:
        settings.preload_via_export = True

    # Save the original call as a string
    settings.cmd_call = "jit " + " ".join(args)
    settings.update()


def abort() -> None:
    """
    Abort the installation process.
    """
    jit_print("[bold red]>  Aborting installation[/]")
    exit(1)


def install_all(settings: JitSettings) -> None:
    """
    Install all necessary components for the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    with console.status("[bold green]Starting installation...") as status:
        try:
            # Create directory
            jit_print(">  Creating directory")
            status.update("[bold green]JIT >  Creating directory[/]", speed=30)
            os.makedirs(settings.install_location, exist_ok=True)

            # unload CUDA for Gekko
            command = 'module -t list 2>&1 | grep -q "CUDA" && module unload $(module -t list 2>&1 | grep "CUDA")'
            try:
                # Execute the bash command
                subprocess.run(command, shell=True, check=True)
                jit_print(">  CUDA module unloaded successfully.")
            except subprocess.CalledProcessError:
                jit_print(">  CUDA module not found or failed to unload.")

            # Clone GKFS
            jit_print(">  Installing GKFS")
            if os.path.isdir(os.path.join(settings.install_location, "gekkofs")):
                jit_print(">  GKFS exists, skipping installation")
            else:
                status.update("[bold green]JIT >  Installing GKFS[/]", speed=30)
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--recurse-submodules",
                        "https://storage.bsc.es/gitlab/hpc/gekkofs.git",
                    ],
                    cwd=settings.install_location,
                    check=True,
                )
                os.chdir(os.path.join(settings.install_location, "gekkofs"))
                subprocess.run(["git", "pull", "--recurse-submodules"], check=True)
                os.chdir(settings.install_location)

                # Build GKFS
                jit_print(">  Building GKFS")
                status.update("[bold green]JIT >  Building GKFS[/]", speed=30)
                subprocess.run(
                    [
                        "gekkofs/scripts/gkfs_dep.sh",
                        "-p",
                        "default_zmq",
                        f"{settings.install_location}/iodeps/git",
                        f"{settings.install_location}/iodeps",
                    ],
                    check=True,
                )
                build_dir = os.path.join(settings.install_location, "gekkofs", "build")
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(
                    [
                        "cmake",
                        "-DCMAKE_BUILD_TYPE=Release",
                        f"-DCMAKE_PREFIX_PATH={settings.install_location}/iodeps",
                        "-DGKFS_BUILD_TESTS=OFF",
                        f"-DCMAKE_INSTALL_PREFIX={settings.install_location}/iodeps",
                        "-DGKFS_ENABLE_CLIENT_METRICS=ON",
                        "..",
                    ],
                    cwd=build_dir,
                    check=True,
                )
                subprocess.run(["make", "-j", "4", "install"], cwd=build_dir, check=True)

                jit_print(">  GEKKO installed")
                status.update("[bold green]JIT >  GEKKO installed[/]", speed=30)

            jit_print(">  Installing Boost")
            status.update("[bold green]JIT >  Getting Boost[/]", speed=30)
            if os.path.isfile(
                os.path.join(settings.install_location, "boost_1_84_0.tar.gz")
            ):
                print("File exists.")
                jit_print(">  Boost tar exists, skipping download")
            else:
                try:
                    subprocess.run(
                        [
                            "wget",
                            "https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz",
                        ],
                        cwd=settings.install_location,
                        check=True,
                    )
                except Exception:
                    jit_print(">  Wget failed, copying boost from shared")
                    subprocess.run(
                        [
                            "cp",
                            "/lustre/project/nhr-admire/shared/boost_1_84_0.tar.gz",
                            ".",
                        ],
                        cwd=settings.install_location,
                        check=True,
                    )
            if os.path.isdir(os.path.join(settings.install_location, "boost_1_84_0")):
                jit_print(">  Boost exists, skipping installation")
            else:
                status.update(
                    "[bold green]JIT >  Extracting boost from tar archive [/]",
                    speed=30,
                )
                subprocess.run(
                    ["tar", "-xzf", "boost_1_84_0.tar.gz"],
                    cwd=settings.install_location,
                    check=True,
                )
                status.update(
                    "[bold green]JIT >  Installing Boost: bootstrap[/]",
                    speed=30,
                )
                build_dir = os.path.join(settings.install_location, "boost_1_84_0")
                os.chdir(build_dir)
                subprocess.run(f"{build_dir}/bootstrap.sh", cwd=build_dir, check=True)
                status.update(
                    "[bold green]JIT >  Installing Boost: b2 install[/]",
                    speed=30,
                )
                subprocess.run(
                    [
                        f"{build_dir}/b2",
                        "install",
                        f"--prefix={settings.install_location}/iodeps",
                        "toolset=gcc",
                    ],
                    cwd=build_dir,
                    check=True,
                )
                status.update(
                    "[bold green]JIT >  Installing Boost: b2 headers[/]",
                    speed=30,
                )
                subprocess.run(
                    [
                        f"{build_dir}/b2",
                        "headers",
                        f"--prefix={settings.install_location}/iodeps",
                    ],
                    cwd=build_dir,
                    check=True,
                )
                jit_print(">  Boost installed")
                os.chdir(settings.install_location)
                status.update("[bold green]JIT >  Boost installed[/]", speed=30)

            jit_print(">  Installing Cereal")
            status.update("[bold green]JIT >  Installing Cereal[/]", speed=30)
            subprocess.run(
                ["git", "clone", "https://github.com/USCiLab/cereal"],
                cwd=settings.install_location,
                check=True,
            )
            os.chdir(os.path.join(settings.install_location, "cereal"))
            build_dir = os.path.join(settings.install_location, "cereal", "build")
            os.makedirs(build_dir, exist_ok=True)
            subprocess.run(
                [
                    "cmake",
                    f"-DCMAKE_PREFIX_PATH={settings.install_location}/iodeps",
                    f"-DCMAKE_INSTALL_PREFIX={settings.install_location}/iodeps",
                    "..",
                ],
                cwd=build_dir,
                check=True,
            )
            subprocess.run(["make", "-j", "4", "install"], cwd=build_dir, check=True)
            jit_print(">  Cereal installed")
            status.update("[bold green]JIT >  Cereal installed[/]", speed=30)

            # Install Cargo Dependencies: Thallium
            jit_print(">  Installing Thallium")
            status.update("[bold green]JIT >  Installing Thallium[/]", speed=30)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/mochi-hpc/mochi-thallium",
                ],
                cwd=settings.install_location,
                check=True,
            )
            os.chdir(os.path.join(settings.install_location, "mochi-thallium"))
            build_dir = os.path.join(settings.install_location, "mochi-thallium", "build")
            os.makedirs(build_dir, exist_ok=True)
            subprocess.run(
                [
                    "cmake",
                    f"-DCMAKE_PREFIX_PATH={settings.install_location}/iodeps",
                    f"-DCMAKE_INSTALL_PREFIX={settings.install_location}/iodeps",
                    "..",
                ],
                cwd=build_dir,
                check=True,
            )
            subprocess.run(["make", "-j", "4", "install"], cwd=build_dir, check=True)
            jit_print(">  Mochi-thallium installed")
            status.update("[bold green]JIT >  Mochi-thallium installed[/]", speed=30)

            # Clone and Build Cargo
            jit_print(">  Installing Cargo")
            status.update("[bold green]JIT >  Installing Cargo[/]", speed=30)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://storage.bsc.es/gitlab/hpc/cargo.git",
                ],
                cwd=settings.install_location,
                check=True,
            )
            os.chdir(os.path.join(settings.install_location, "cargo"))
            replace_line_in_file(
                "src/master.cpp",
                332,
                f'  auto patternFile = "{settings.regex_file}";',
            )
            subprocess.run(
                ["git", "checkout", "marc/nek5000"],
                check=True,
            )
            build_dir = os.path.join(settings.install_location, "cargo", "build")
            os.makedirs(build_dir, exist_ok=True)
            subprocess.run(
                [
                    "cmake",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_PREFIX_PATH={settings.install_location}/iodeps",
                    f"-DCMAKE_INSTALL_PREFIX={settings.install_location}/iodeps",
                    "..",
                ],
                cwd=build_dir,
                check=True,
            )
            subprocess.run(["make", "-j", "4", "install"], cwd=build_dir, check=True)

            jit_print(">  Cargo installed")
            status.update("[bold green]JIT >  Cargo installed[/]", speed=30)

            # Build IOR
            jit_print(">  Installing IOR")
            status.update("[bold green]JIT >  Installing IOR[/]", speed=30)
            subprocess.run(
                ["git", "clone", "https://github.com/hpc/ior.git"],
                cwd=settings.install_location,
                check=True,
            )
            os.chdir(os.path.join(settings.install_location, "ior"))
            subprocess.run(["./bootstrap"], check=True)
            subprocess.run(["./configure"], check=True)
            subprocess.run(["make", "-j", "4"], check=True)

            jit_print("  Installation finished")
            status.update("[bold  green]Installation finished[/]", speed=30)
            console.print("\n  Ready to go <<")
            status.update("\n  Ready to go <<", speed=30)
            console.print("Call: ./jit.sh -n NODES -t MAX_TIME")
            status.update("Call: ./jit.sh -n NODES -t MAX_TIME", speed=30)

        except subprocess.CalledProcessError as e:
            jit_print(f"[bold red]>  Error encountered: {e}[/]")
            status.update(
                "[bold green]JIT [bold red]>  Error encountered: {e}[/]",
                speed=30,
            )
            abort()


def replace_line_in_file(file_path: str, line_number: int, new_line_content: str) -> None:
    """
    Replace a specific line in a file with new content.

    Args:
        file_path (str): Path to the file.
        line_number (int): Line number to replace.
        new_line_content (str): New content for the line.
    """
    try:
        # Read the existing file content
        with open(file_path) as file:
            lines = file.readlines()

        # Replace the specific line (line_number - 1 because list indices start at 0)
        if line_number - 1 < len(lines):
            lines[line_number - 1] = new_line_content + "\n"
        else:
            raise IndexError("Line number exceeds the total number of lines in the file.")

        # Write the modified content back to the file
        with open(file_path, "w") as file:
            file.writelines(lines)

        print(f"Successfully replaced line {line_number} in {file_path}.")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except IndexError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def cancel_jit_jobs(settings: JitSettings) -> None:
    """
    Cancel any existing JIT jobs.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.job_id == 0:
        # Check if the hostname contains 'cpu' or 'mogon'
        hostname = subprocess.check_output("hostname", shell=True).decode().strip()
        if "cpu" in hostname or "mogon" in hostname:
            # Get the list of job IDs with the name "JIT"
            try:
                jit_jobs = (
                    subprocess.check_output(
                        "squeue --me --name=JIT --format=%A", shell=True
                    )
                    .decode()
                    .strip()
                    .split("\n")[1:]
                )
            except subprocess.CalledProcessError:
                jit_jobs = []

            if not jit_jobs:
                return

            # Print the list of JIT jobs
            job_list = "\n".join(jit_jobs)
            jit_print(
                "[bold  yellow]The following jobs with the name 'JIT' were found:[/]"
            )
            console.print(job_list)

            if not settings.skip_confirm:
                # Prompt the user to confirm cancellation
                confirmation = (
                    input("Do you want to cancel all 'JIT' jobs? (yes/no): ")
                    .strip()
                    .lower()
                )
            else:
                confirmation = "yes"

            if confirmation in {"yes", "y", "ye"}:
                for job_id in jit_jobs:
                    subprocess.run(f"scancel {job_id}", shell=True, check=True)
                    jit_print(f"[bold  cyan] Cancelled job ID {job_id}[/]")
                jit_print("[bold  green]All 'JIT' jobs have been cancelled[/]")
            else:
                jit_print("[bold  yellow]No jobs were cancelled[/]")


def relevant_files(settings: JitSettings) -> None:
    """
    Set up ignored files based on regex match.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.verbose:  # Mimicking checking for the number of arguments
        jit_print("[cyan]Setting up ignored files[/]")

    # Ensure the directory containing the regex file exists
    os.makedirs(os.path.dirname(settings.regex_file), exist_ok=True)

    # Create or update the regex file with the settings.regex_match
    with open(settings.regex_file, "w") as file:
        file.write(f"{settings.regex_match}\n")

    if settings.verbose:
        jit_print(f"[cyan]Files that match {settings.regex_match} are ignored [/]")

    # Display the contents of the regex file
    with open(settings.regex_file) as file:
        # content = file.read()
        content = file.read().rstrip("\n")  # remove trailing newline

    if settings.verbose:
        jit_print(f"[cyan]content of {settings.regex_file}: {content}[/]")


def adjust_regex(settings: JitSettings, mode: str = "stage_out") -> None:
    """
    Adjust the regex for stage out or flush mode.

    Args:
        settings (JitSettings): The JIT settings object.
        mode (str, optional): Mode for regex adjustment. Defaults to "stage_out".
    """
    # if settings.cluster:
    if "stage_out" in mode:
        settings.regex_match = settings.regex_stage_out_match
    elif "stage_in" in mode:
        settings.regex_match = settings.regex_stage_in_match
    elif "flush" in mode:
        settings.regex_match = settings.regex_flush_match
    else:
        raise ValueError(f"Unsupported mode for regex: {mode} passed")

    jit_print(f"[cyan]Resetting regex for {mode} to {settings.regex_match}[/]")
    os.makedirs(os.path.dirname(settings.regex_file), exist_ok=True)
    with open(settings.regex_file, "w") as file:
        file.write(f"{settings.regex_match}\n")

    # Optionally jit_print the contents of the regex file
    if settings.debug_lvl > 1:
        with open(settings.regex_file) as file:
            content = file.read()

        jit_print(f"[bold  cyan] cat {settings.regex_file}: \n{content} [/]\n")


def total_time(log_dir: str) -> None:
    """
    Calculate and print the total time from the log file.

    Args:
        log_dir (str): Directory containing the log file.
    """
    time_log_file = os.path.join(log_dir, "time.log")

    # Calculate total time from the log file
    try:
        with open(time_log_file) as file:
            lines = file.readlines()
        total_time = sum(float(line.split()[1]) for line in lines if "seconds" in line)
    except FileNotFoundError:
        total_time = 0

    # Print and append total time to the log file
    result = f"\n JIT --> Total Time: {total_time}\n"
    console.print("[bold green]" + result + "[/]")
    with open(time_log_file, "a") as file:
        file.write(result)


def allocate(settings: JitSettings) -> None:
    """
    Allocate resources for the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    settings.app_nodes = 1
    if settings.use_mpirun and settings.job_id == 0:
        jit_print(
            " [bold  yellow]Warning, executing mpiexec on remote. We advice using using fixed allocation with mpirun\n"
        )

    if settings.cluster:
        # allocate if needed
        if settings.job_id == 0:
            # Allocating resources
            jit_print("[green]####### Allocating resources[/]")

            call = f"salloc -N {settings.nodes} -t {settings.max_time} {settings.alloc_call_flags}"
            jit_print(f"[cyan]Executing: {call} [/]")
            # Execute the salloc command
            subprocess.run(call, shell=True, check=True)

            # Get JIT_ID
            try:
                result = subprocess.run(
                    f"squeue -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R' | grep '{settings.job_name}' | sort -k1,1rn | head -n 1 | awk '{{print $1}}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                settings.job_id = int(result.stdout.strip())
            except subprocess.CalledProcessError:
                settings.job_id = 0
        else:
            jit_print(
                f"[bold green]JIT [green]####### Using allocation with id: {settings.job_id}[/]"
            )

        time.sleep(1)
        # Get NODES_ARR
        if settings.job_id:
            settings.set_mpi_host_file(settings.job_id)
            try:
                nodes_result = subprocess.run(
                    f"scontrol show hostname $(squeue -j {settings.job_id} -o '%N' | tail -n +2)",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                nodes_arr = nodes_result.stdout.splitlines()
            except subprocess.CalledProcessError as e:
                jit_print(f"[bold red]Error: {e}[/]")
                nodes_arr = []

            # second try
            if len(nodes_arr) == 0:
                try:
                    nodes_result = subprocess.run(
                        f"scontrol show hostname $(squeue -j {settings.job_id} -h -o '%N' | tr '\n' ',')",
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    nodes_arr = nodes_result.stdout.splitlines()
                except subprocess.CalledProcessError as e:
                    jit_print(f"[bold red]Error: {e}[/]")
                    nodes_arr = []

            print(f"nodes_arr: {nodes_arr}")
            if nodes_arr:
                # Remove specific nodes if present
                nodes_to_remove = []
                # nodes_to_remove = ["cpu0088", "cpu0039"]
                if len(nodes_to_remove) > 0:
                    nodes_arr = [
                        node for node in nodes_arr if node not in nodes_to_remove
                    ]
                    jit_print(
                        f"[bold yellow]Removed nodes {', '.join(nodes_to_remove)} from allocation: {nodes_arr}[/]"
                    )
                    print(f"nodes_arr: {nodes_arr}")

            try:
                nodes_arr = nodes_arr[: int(settings.nodes)]
            except Exception as e:
                print(e)
                jit_print(
                    f"[bold green]JIT [red]Unable to decrease number of nodes. Using {settings.nodes}[/]"
                )

            # Write to mpi_hostfile
            with open(f"{settings.mpi_hostfile}", "w") as file:
                file.write("\n".join(nodes_arr) + "\n")

            if nodes_arr:
                # Get FTIO node
                # its the node where the code is executed
                try:
                    result = subprocess.run(
                        ["hostname"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    ftio_node = result.stdout.strip()

                except subprocess.CalledProcessError:
                    ftio_node = nodes_arr[-1]

                # Get node for single commands
                if len(nodes_arr) > 1:
                    single_node = (
                        nodes_arr[0] if nodes_arr[0] != ftio_node else nodes_arr[1]
                    )
                else:
                    single_node = nodes_arr[0]

                settings.ftio_node = ftio_node
                settings.single_node = single_node

                if len(nodes_arr) > 1:
                    settings.ftio_node_command = f"--nodelist={settings.ftio_node}"
                    settings.app_nodes_command = f"--nodelist={','.join(n for n in nodes_arr if n != settings.ftio_node)}"
                    settings.single_node_command = f"--nodelist={settings.single_node}"
                    settings.app_nodes = len(nodes_arr) - 1
                    # print(nodes_arr)
                    # print(f"{','.join(n for n in nodes_arr if n != settings.ftio_node)}\n")

                    # Remove FTIO node from mpi_hostfile
                    with open(f"{settings.mpi_hostfile}") as file:
                        lines = file.readlines()

                    if any(line.strip() == settings.ftio_node for line in lines):
                        new_lines = [
                            line for line in lines if line.strip() != settings.ftio_node
                        ]
                    else:
                        new_lines = lines[:-1]  # fallback: remove last line

                    with open(f"{settings.mpi_hostfile}", "w") as file:
                        file.writelines(new_lines)

                jit_print(f"[green]JIT Job Id: {settings.job_id} [/]")
                jit_print(f"[green]Allocated Nodes: {len(nodes_arr)} [/]")
                jit_print(f"[green]FTIO Node: {settings.ftio_node} [/]")
                jit_print(f"[green]APP Node command: {settings.app_nodes_command} [/]")
                jit_print(f"[green]FTIO Node command: {settings.ftio_node_command} [/]")

                # Print contents of mpi_hostfile
                with open(f"{settings.mpi_hostfile}") as file:
                    hostfile_content = file.read()
                jit_print(
                    f"[cyan]content of {settings.mpi_hostfile}: \n{hostfile_content} [/]"
                )
            settings.update_geko_files()
        else:
            settings.set_mpi_host_file()
            jit_print("[bold red]JOB ID could not be retrieved[/]")
    settings.update_app_nodes()


def get_pid(settings: JitSettings, name: str, pid: int) -> None:
    """
    Get the process ID for a given component.

    Args:
        settings (JitSettings): The JIT settings object.
        name (str): Name of the component.
        pid (int): Process ID.
    """
    if settings.cluster:
        call = f"ps aux | grep 'srun' | grep '{settings.job_id}' | grep '{name}' | grep -v grep | tail -1 | awk '{{print $2}}'"
        res = subprocess.run(call, shell=True, check=True, capture_output=True, text=True)
        if res.stdout.strip():
            try:
                pid = int(res.stdout.strip())
            except Exception:
                pid = res.stdout.strip()

    if name.lower() in "cargo":
        settings.cargo_pid = pid
        jit_print(f" Cargo startup successful. PID is {pid}")
    elif name.lower() in "gkfs_daemon":
        settings.gkfs_daemon_pid = pid
        jit_print(f" Gekko daemon startup successful. PID is {pid}")
    elif name.lower() in "gkfs_fuse":
        settings.gkfs_fuse_pid = pid
        jit_print(f" Gekko fuse startup successful. PID is {pid}")
    elif name.lower() in "gkfs_proxy":
        settings.gkfs_proxy_pid = pid
        jit_print(f" Gekko proxy startup successful. PID is {pid}")
    elif name.lower() in "ftio" or name.lower() in "predictor_jit":
        settings.ftio_pid = pid
        jit_print(f" FTIO startup successful. PID is {pid}")
    elif name.lower() in "app" or name.lower() in settings.app_call:
        settings.app_pid = pid
        jit_print(f" {settings.app_call} startup successful. PID is {pid}")


def handle_sigint(settings: JitSettings) -> None:
    """
    Handle a termination signal (SIGINT/SIGTERM/SIGHUP) for graceful shutdown.

    Runs the exit routine exactly once (guarded by ``trap_exit``) so that a
    process killed by Ctrl-C, ``timeout``, or a job scheduler still tears down
    the daemon, predictor pool, and other processes it started.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.trap_exit:
        settings.trap_exit = False
        jit_print("[bold blue]Termination signal detected. Exiting script.[/]")
        exit_routine(settings)


def install_signal_handlers(settings: JitSettings) -> None:
    """
    Register the shutdown handler for every signal that must trigger cleanup.

    SIGINT (Ctrl-C), SIGTERM (sent by ``timeout`` and most job schedulers), and
    SIGHUP all route to :func:`handle_sigint`. Without a SIGTERM handler Python's
    default disposition kills the interpreter before the ``finally`` cleanup
    runs, orphaning the gkfs daemon (which holds a tmpfs rootdir) and the whole
    ``predictor_jit`` process pool.

    Args:
        settings (JitSettings): The JIT settings object.
    """

    def handler(signum, frame):
        handle_sigint(settings)

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    for sig in signals:
        signal.signal(sig, handler)


def exit_routine(settings: JitSettings) -> None:
    """
    Perform the exit routine for the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    info = f"{settings.app_call} with {settings.nodes} nodes {settings.log_suffix} (./{os.path.relpath(settings.log_dir, os.getcwd())})"
    jit_print(f"[bold blue]Killing Job: {info} [/]")
    log_failed_jobs(settings, info)
    soft_kill(settings)
    hard_kill(settings)
    jit_print(" Exciting\n")
    sys.exit(0)


def _list_child_pids(pid: int) -> list[int]:
    """Return the direct child PIDs of ``pid`` via ``ps`` (empty on any error)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=", "--ppid", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    children = []
    for token in out.stdout.split():
        try:
            children.append(int(token))
        except ValueError:
            continue
    return children


def _descendant_pids(pid: int) -> list[int]:
    """Return all descendants of ``pid``, deepest first (children before parents)."""
    result: list[int] = []
    for child in _list_child_pids(pid):
        result.extend(_descendant_pids(child))
        result.append(child)
    return result


def _safe_kill(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid``, ignoring processes that are already gone."""
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception as e:
        jit_logger.debug(f"Unable to signal PID {pid}: {e}")


def kill_process_tree(pid: int | str, sig: int = signal.SIGTERM) -> None:
    """
    Terminate ``pid`` and its whole subtree without touching unrelated processes.

    Background components are launched in their own session (see
    ``execute_background``), so when ``pid`` leads a process group the entire
    group is signalled in a single call -- this reaps the ``predictor_jit``
    worker pool and any ``bash -c`` wrapper. Group membership survives
    reparenting to init, so this still works after the parent has died. If
    ``pid`` is not a group leader, its descendants are walked via ``ps`` and
    signalled individually. Only PIDs derived from ``pid`` are touched (no
    blanket, name-based matching), the JIT's own group is never signalled, and
    the call is idempotent -- an already-dead PID never raises.

    Args:
        pid (int | str): PID of the group leader / process to terminate.
        sig (int): Signal to send (default ``signal.SIGTERM``).
    """
    if not pid:
        return
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if pid <= 1:
        # Never signal init (pid 1) or an invalid/negative PID.
        return

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return  # already gone
    except Exception:
        pgid = None

    # Prefer a single process-group kill, but only when pid actually leads its
    # own group and that group is not the JIT's own (guards against self-kill).
    if pgid is not None and pgid == pid and pgid != os.getpgrp():
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        except Exception as e:
            jit_logger.debug(f"killpg failed for PID {pid}: {e}")

    # Fallback: signal descendants (deepest first), then the process itself.
    for child in _descendant_pids(pid):
        _safe_kill(child, sig)
    _safe_kill(pid, sig)


def soft_kill(settings: JitSettings) -> None:
    """
    Perform a soft kill of the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.soft_kill:
        jit_print("[bold green]####### Soft kill [/]", True)

        if not settings.exclude_ftio:
            try:
                shut_down(settings, "FTIO", settings.ftio_pid)
                jit_print("[bold  cyan]killed FTIO [/]")
            except Exception:
                jit_print("[bold  cyan]Unable to soft kill FTIO [/]")

    if not settings.exclude_daemon:
        try:
            shut_down(settings, "GEKKO", settings.gkfs_daemon_pid)
            jit_print("[bold  cyan]killed GEKKO DEMON [/]")
        except Exception:
            jit_print("[bold  cyan]Unable to soft kill GEKKO DEMON [/]")

    if settings.fuse:
        try:
            shut_down(settings, "FUSE", settings.gkfs_fuse_pid)

            subprocess.run(
                f"fusermount -u {settings.gkfs_mntdir}",
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )

            subprocess.run(
                f"fusermount -uz {settings.gkfs_mntdir}",
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )

            jit_print("[bold cyan]Killed GEKKO FUSE[/]")

        except Exception as e:
            jit_print(f"[bold yellow]FUSE shutdown warning:[/] {e}")

    if not settings.exclude_proxy:
        try:
            shut_down(settings, "GEKKO", settings.gkfs_proxy_pid)
            jit_print("[bold  cyan]killed GEKKO PROXY [/]")
        except Exception:
            jit_print("[bold  cyan]Unable to soft  kill GEKKO PROXY [/]")

    if not settings.exclude_cargo:
        try:
            shut_down(settings, "CARGO", settings.cargo_pid)
            jit_print("[bold  cyan]killed CARGO [/]")
        except Exception:
            jit_print("[bold  cyan]Unable to soft kill CARGO [/]")

    if not settings.dry_run:
        try:
            shut_down(settings, "CARGO", settings.app_pid)
            jit_print("[bold  cyan]killed App [/]")
        except Exception:
            jit_print("[bold  cyan]Unable to soft kill App [/]")

    if settings.app_start_file:
        try:
            os.remove(settings.app_start_file)
            jit_print("[bold cyan]Removed app_start flag file[/]")
        except FileNotFoundError:
            jit_print("[yellow]app_start flag file already absent[/]")

    jit_print("Soft kill finished")


def hard_kill(settings: JitSettings) -> None:
    """
    Perform a hard kill of the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.hard_kill:
        jit_print("[bold green]####### Hard kill[/]", True)

        if settings.cluster and not settings.static_allocation:
            # Cluster environment: use `scancel` to cancel the job
            _ = subprocess.run(
                f"scancel {settings.job_id}",
                shell=True,
                text=True,
                capture_output=True,
                check=True,
                executable="/bin/bash",
            )
        else:
            # Non-cluster environment: SIGKILL only the process trees this jit
            # instance started. We use the tracked PIDs (never a blanket, name
            # based `ps | grep <binary>`), so another user's or another run's
            # daemon using the same binary is left untouched.
            tracked_pids = [
                settings.gkfs_daemon_pid,
                settings.gkfs_proxy_pid,
                settings.gkfs_fuse_pid,
                settings.ftio_pid,
                settings.cargo_pid,
                settings.app_pid,
            ]
            for pid in tracked_pids:
                kill_process_tree(pid, signal.SIGKILL)

        # A SIGKILLed daemon never reaches destroy_hosts_file, so its line stays
        # in the hostfile -- and the daemon *appends*, so the next run's daemon
        # lands underneath the corpse. Clients then hash some paths onto the dead
        # entry and get HG_NOENTRY, which GekkoFS reports as the misleading
        # "Device or resource busy". One failed run would poison every run after
        # it. Since we killed the daemon, we clean up on its behalf.
        remove_hostfile(settings)

        jit_print("Hard kill finished")


def shut_down(settings: JitSettings, name: str, pid: int) -> None:
    """
    Shut down a component and its entire process subtree by PID.

    The recorded PID is the group leader started by ``execute_background``, so
    killing the whole group also reaps children such as the ``predictor_jit``
    worker pool. Already-dead PIDs are ignored (idempotent).

    Args:
        settings (JitSettings): The JIT settings object (kept for API symmetry).
        name (str): Name of the component.
        pid (int): Process ID of the group leader to terminate.
    """
    jit_logger.debug(f"Shutting down {name} with PID {pid}")
    kill_process_tree(pid, signal.SIGTERM)


def compose_log_dir(
    app: str, nodes: int, mode: str, exists=os.path.exists
) -> tuple[str, str]:
    """Build the per-app log layout: ``<app>/nodes_<n>/rep_<rep>/<mode>``.

    Grouping the logs by app (then node count, repetition, and run mode) keeps
    a multi-app sweep from clobbering itself: the old name held only nodes and
    the job id, so two apps at the same size landed in the same directory. The
    repetition is derived, not passed in -- ``rep`` is one past however many
    times this exact (app, nodes, mode) has already run, so the three modes of
    one sweep pass line up under the same ``rep_<n>``.

    Args:
        app: The application name (``-a``); falls back to ``app`` if empty.
        nodes: The node count for the run.
        mode: The run mode folder (``glass``/``gekko``/``pfs``).
        exists: Existence probe, injectable for testing.

    Returns:
        (log_dir, result_dir): the leaf to run in, and ``<app>`` where the
        per-app ``result.json`` and plot live.
    """
    app = app or "app"
    mode = mode or "run"
    base = os.path.join(app, f"nodes_{nodes}")
    rep = 1
    while exists(os.path.join(base, f"rep_{rep}", mode)):
        rep += 1
    return os.path.join(base, f"rep_{rep}", mode), app


def log_dir(settings: JitSettings) -> None:
    """
    Create and set up the log directory.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if not settings.log_dir:
        settings.log_dir, settings.result_dir = compose_log_dir(
            settings.app, settings.nodes, settings.mode_name
        )

    counter = 0
    name = settings.log_dir
    while os.path.exists(settings.log_dir):
        counter += 1
        settings.log_dir = f"{name}_{counter}"

    # Resolve and return the absolute path of LOG_DIR
    settings.log_dir = os.path.abspath(settings.log_dir)
    if settings.result_dir:
        settings.result_dir = os.path.abspath(settings.result_dir)

    # Create directory if it does not exist
    os.makedirs(settings.log_dir, exist_ok=True)

    settings.set_log_dirs()
    # Auto-compute unless overridden via --fuse-idle-threads.
    # procs_app is tasks-per-node; multiply by 4 to cover the internal reader
    # threads that frameworks like TensorFlow/PyTorch spawn per rank.
    if settings.fuse_idle_threads == 0:
        settings.fuse_idle_threads = max(16, settings.procs_app * 4)


def get_address_ftio(settings: JitSettings) -> None:
    """
    Get the address for FTIO.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    # Get Address and port
    jit_print(" Getting FTIO ADDRESS")
    if settings.cluster:
        # call = f"srun --jobid={settings.job_id} {settings.ftio_node_command} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ip addr | grep ib0 | awk '{{print $2}}' | cut -d'/' -f1 | tail -1"
        call = "ip addr | grep ib0 | awk '{{print $2}}' | cut -d'/' -f1 | tail -1"
        call = generate_cluster_command(
            settings, call, 1, 1, node_list=settings.ftio_node
        )
        jit_print(f"[bold  cyan]Executing: {call}")
        try:
            result = subprocess.run(
                call, shell=True, capture_output=True, text=True, check=True
            )
            settings.address_ftio = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            jit_print(f"[bold red]Error occurred: {e}")
            settings.address_ftio = ""

    jit_print(f" Address FTIO: {settings.address_ftio}")


def get_address_cargo(settings: JitSettings) -> None:
    """
    Get the address for Cargo.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    jit_print(" Getting Cargo ADDRESS")
    if settings.cluster:
        # call = f"srun --jobid={settings.job_id} {settings.single_node_command} --disable-status -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --overcommit --overlap --oversubscribe --mem=0 ip addr | grep ib0 | awk '{{print $2}}' | cut -d'/' -f1 | tail -1"
        call = "ip addr | grep ib0 | awk '{print $2}' | cut -d'/' -f1 | tail -1"
        call = generate_cluster_command(
            settings, call, 1, 1, node_list=settings.single_node
        )
        jit_print(f"[bold  cyan]Executing: {call}")
        try:
            result = subprocess.run(
                call, shell=True, capture_output=True, text=True, check=True
            )
            settings.address_cargo = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            jit_print(f"[bold red]Error occurred: {e}")
            settings.address_cargo = ""

    jit_print(f" Address CARGO: {settings.address_cargo}")
    jit_print(f" Port CARGO:  {settings.port_cargo}")
    jit_print(
        f" Server CARGO:  {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo}"
    )


def set_dir_gekko(settings: JitSettings) -> None:
    """
    Set the directory for GekkoFS.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.node_local and settings.cluster:
        jit_print(" Node local flag set")
        # old_gkfs_rootdir = settings.gkfs_rootdir
        old_gkfs_mntdir = settings.gkfs_mntdir

        # decide where node local is
        hostname = socket.gethostname()
        if "gs" in hostname or "bsc" in os.path.expanduser("~"):
            nodelocal = "/scratch/tmp"
        elif "cpu" in hostname:
            nodelocal = "/localscratch"
        else:
            raise RuntimeError("Unsupported location to Nodelocal. Add it here.")

        print(f"nodelocal is {nodelocal}")
        settings.gkfs_rootdir = (
            f"{nodelocal}/{settings.job_id}/{os.path.basename(settings.gkfs_rootdir)}"
        )
        settings.gkfs_mntdir = (
            f"{nodelocal}/{settings.job_id}/{os.path.basename(settings.gkfs_mntdir)}"
        )
        jit_print(f" |-> Gekko root dir updated to: {settings.gkfs_rootdir}")
        jit_print(f" |-> Gekko mnt dir updated to: {settings.gkfs_mntdir}")

        for attr in ["run_dir", "app_flags", "pre_app_call", "post_app_call", "app_call"]:
            value = getattr(settings, attr)
            # pre_app_call may be a list of calls (e.g. dlio's mkdir + mpirun
            # steps) instead of a single string.
            if isinstance(value, list):
                setattr(
                    settings,
                    attr,
                    [
                        call.replace(old_gkfs_mntdir, settings.gkfs_mntdir)
                        for call in value
                    ],
                )
            elif old_gkfs_mntdir in value:
                setattr(
                    settings, attr, value.replace(old_gkfs_mntdir, settings.gkfs_mntdir)
                )
                jit_print(
                    f" |-> {attr.replace('_', ' ').capitalize()} updated to: {getattr(settings, attr)}"
                )

    if settings.update_files_with_gkfs_mntdir:
        for file_path in settings.update_files_with_gkfs_mntdir:
            with open(file_path) as file:
                content = file.read()

            updated_content = repoint_mntdir_in_text(content, settings.gkfs_mntdir)

            with open(file_path, "w") as file:
                file.write(updated_content)

                jit_print(
                    f" |-> File updated: {file_path}",
                )


def repoint_mntdir_in_text(content: str, new_mntdir: str) -> str:
    """Rewrite every ``<old>_gkfs_mountdir/<rest>`` path to ``<new_mntdir>/<rest>``.

    Re-points files (QMCPACK glass.xml, WRF namelist, wacom json) that were written
    with the default mount dir before :func:`set_dir_gekko` finalized it to
    node-local scratch. The character class stops at quotes and whitespace, so it
    is safe for double-quoted (XML/JSON) and single-quoted (Fortran namelist)
    values alike -- a greedy ``[^"]*`` would swallow a namelist's closing ``'`` and
    the rest of the line.

    Args:
        content: The file text to rewrite.
        new_mntdir: The finalized GekkoFS mount directory.

    Returns:
        The text with every mount-dir path re-pointed at ``new_mntdir``.
    """
    return re.sub(
        r"(/[^\"'\s]*_gkfs_mountdir)(/[^\"'\s]*)",
        lambda match: f"{new_mntdir}{match.group(2)}",
        content,
    )


def save_hosts_file(settings: JitSettings):
    """
    Saves the hostfiles in log dir

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if not settings.dry_run:
        for hostfile in [settings.gkfs_hostfile, settings.mpi_hostfile]:
            if hostfile:
                try:
                    shutil.move(hostfile, settings.log_dir)
                except Exception as e:
                    print(f"Could not move {hostfile} to {settings.log_dir}: {e}")


def snapshot_directory(settings: JitSettings):
    """
    Runs 'ls -lahrt' in the current directory and writes the result to a file.
    If no output file is given, a timestamped file is created.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{settings.log_dir}/snapshot_{timestamp}.txt"

    try:
        result = subprocess.run(
            ["ls", "-lahrt"], check=True, capture_output=True, text=True
        )
        with open(output_file, "w") as f:
            f.write(result.stdout)
        jit_print(f"Snapshot saved to {output_file}")
    except subprocess.CalledProcessError as e:
        jit_print(f"Error running ls: {e}")


def print_settings(settings: JitSettings) -> None:
    """
    Print the current JIT settings.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    # Default values
    ftio_status = "[bold green]ON[/]"
    gkfs_daemon_status = "[bold green]ON[/]"
    gkfs_fuse_status = "[bold green]ON[/]"
    gkfs_proxy_status = "[bold green]ON[/]"
    cargo_status = "[bold green]ON[/]"

    task_daemon = f"{settings.app_nodes}"
    cpu_daemon = f"{settings.procs_daemon}"
    task_proxy = f"{settings.app_nodes}"
    cpu_proxy = f"{settings.procs_proxy}"
    task_cargo = f"{settings.app_nodes * settings.procs_cargo}"
    cpu_cargo = f"{settings.procs_cargo}"
    task_ftio = "1"
    cpu_ftio = f"{settings.procs_ftio}"

    # Default settings text
    ftio_text = f"""
├─ ftio location  : {settings.ftio_bin_location}
├─ ftio args      : {settings.ftio_args}
├─ address ftio   : {settings.address_ftio}
├─ port ftio      : {settings.port_ftio}
├─ node count     : 1
└─ ftio node      : {settings.ftio_node_command.replace("--nodelist=", "")}"""

    gkfs_daemon_text = f"""
├─ gkfs daemon    : {settings.gkfs_daemon}
├─ gkfs fuse      : {settings.gkfs_fuse}
├─ gkfs intercept : {settings.gkfs_intercept}
├─ gkfs mntdir    : {settings.gkfs_mntdir}
├─ gkfs rootdir   : {settings.gkfs_rootdir}
├─ gkfs protocol  : {settings.gkfs_daemon_protocol}
├─ gkfs hostfile  : {settings.gkfs_hostfile}"""

    gkfs_proxy_text = f"""
├─ gkfs proxy     : {settings.gkfs_proxy}
└─ gkfs proxyfile : {settings.gkfs_proxyfile}"""

    cargo_text = f"""
├─ cargo bin      : {settings.cargo_bin}
├─ cargo mode     : {settings.cargo_mode}
├─ stage in path  : {settings.stage_in_path}
├─ stage out path : {settings.stage_out_path}
├─ address cargo  : {settings.address_cargo}
├─ port cargo     : {settings.port_cargo}
└─ server cargo   : {settings.gkfs_daemon_protocol}://{settings.address_cargo}:{settings.port_cargo}"""

    if settings.exclude_ftio:
        ftio_text = """
├─ ftio activate  : [yellow]none[/]
├─ address ftio   : [yellow]none[/]
├─ port ftio      : [yellow]none[/]
├─ node count     : [yellow]none[/]
└─ ftio node      : [yellow]none[/]"""
        ftio_status = "[bold yellow]off[/]"
        task_ftio = "[bold yellow]-[/]"
        cpu_ftio = "[bold yellow]-[/]"

    if settings.exclude_daemon:
        gkfs_daemon_text = """
├─ gkfs daemon    : [yellow]none[/]
├─ gkfs intercept : [yellow]none[/]
├─ gkfs fuse      : [yellow]none[/]
├─ gkfs mntdir    : [yellow]none[/]
├─ gkfs rootdir   : [yellow]none[/]
├─ gkfs protocol  : [yellow]none[/]
├─ gkfs hostfile  : [yellow]none[/]"""
        gkfs_daemon_status = "[bold yellow]off[/]"
        gkfs_fuse_status = "[bold yellow]off[/]"
        task_daemon = "[bold yellow]-[/]"
        cpu_daemon = "[bold yellow]-[/]"

    if not settings.fuse:
        gkfs_fuse_status = "[bold yellow]off[/]"

    if settings.exclude_proxy:
        gkfs_proxy_text = """
├─ gkfs proxy     : [yellow]none[/]
└─ gkfs proxyfile : [yellow]none[/]"""
        gkfs_proxy_status = "[bold yellow]off[/]"
        task_proxy = "[bold yellow]-[/]"
        cpu_proxy = "[bold yellow]-[/]"

    if settings.exclude_cargo:
        cargo_text = f"""
├─ cargo bin      : [yellow]none[/]
├─ cargo mode     : [yellow]none[/]
├─ stage in path  : {settings.stage_in_path}
├─ stage out path : {settings.stage_out_path}
├─ address cargo  : [yellow]none[/]
├─ port cargo     : [yellow]none[/]
└─ server cargo   : [yellow]none[/]"""
        cargo_status = "[bold yellow]off[/]"
        task_cargo = "[bold yellow]-[/]"
        cpu_cargo = "[bold yellow]-[/]"

    app_flags = ""
    if len(settings.app_flags) > 0:
        flags_list = [flag for flag in settings.app_flags.split(" ") if flag.strip()]
        # Indent each flag after the colon
        app_flags = "\n                  : ".join(flags_list)

    # print settings
    text = f"""
[bold green]Settings
##################[/]
[bold green]setup[/]
├─ logs dir       : {settings.log_dir}
├─ pwd            : {settings.dir}
├─ stage in       : {settings.stage_in_path}
├─ stage out      : {settings.stage_out_path}
├─ ftio           : {ftio_status}
├─ gkfs daemon    : {gkfs_daemon_status}
├─ gkfs proxy     : {gkfs_proxy_status}
├─ gkfs fuse      : {gkfs_fuse_status}
├─ cargo          : {cargo_status}
├─ cluster        : {settings.cluster}
├─ total nodes    : {settings.nodes}
|   ├─ app        : {settings.app_nodes}
|   └─ ftio       : 1
├─ tasks per node :
|   ├─ app        : {settings.procs_app}
|   ├─ daemon     : {task_daemon}
|   ├─ proxy      : {task_proxy}
|   ├─ cargo      : {task_cargo}
|   ├─ ftio       : {task_ftio}
|   └─ fuse idle threads: {settings.fuse_idle_threads}
├─ cpus per task  : {settings.procs}
|   ├─ app        : 1
|   ├─ daemon     : {cpu_daemon}
|   ├─ proxy      : {cpu_proxy}
|   ├─ cargo      : {cpu_cargo}
|   └─ ftio       : {cpu_ftio}
├─ OMP threads    : {settings.omp_threads}
├─ max time       : {settings.max_time}
├─ tasks affinity : {settings.set_tasks_affinity}
|   ├─ set 0      : {settings.task_set_0 if settings.set_tasks_affinity else "[yellow]none[/]"}
|   └─ set 1      : {settings.task_set_1 if settings.set_tasks_affinity else "[yellow]none[/]"}
├─ debug level    : {settings.debug_lvl}
├─ use mpirun     : {settings.use_mpirun}
├─ ignore mtime   : {settings.ignore_mtime}
├─ lock system    : {settings.lock_generator}
├─ parallel move  : {settings.parallel_move_threads}
└─ job id         : {settings.job_id}

[bold green]ftio[/]{ftio_text}

[bold green]gekko[/]{gkfs_daemon_text}{gkfs_proxy_text}

[bold  green]cargo[/]{cargo_text}

[bold green]app[/]
├─ run dir        : {settings.run_dir}
├─ realpath       : {os.path.realpath(settings.run_dir)}
├─ app nodes      : {settings.app_nodes}
├─ app nodes list : {settings.app_nodes_command.replace("--nodelist=", "")}
├─ app            : {settings.app}
├─ app call       : {settings.app_call}
└─ app flags      : {app_flags}
[bold green]##################[/]
"""
    console.print(text)
    print_to_file(text, os.path.join(settings.log_dir, "settings.log"))


def print_to_file(text: str, file: str) -> None:
    """
    Print text to a file.

    Args:
        text (str): Text to print.
        file (str): Path to the file.
    """
    remove = ["bold", "green", "yellow", "red", "cyan", "[/]", "[ ]", "[]"]
    for r in remove:
        text = text.replace(r, "")

    with open(file, "a") as file:
        file.write(text)


def jit_print(s: str, new_line: bool = False) -> None:
    """
    Print a message with JIT prefix.

    Args:
        s (str): Message to print.
        new_line (bool, optional): Whether to print a new line before the message. Defaults to False.
    """
    COLOR_LEVEL_MAP = {
        "cyan": "debug",
        "green": "info",
        "yellow": "warning",
        "red": "error",
        "bold_red": "critical",
    }
    # if new_line:
    #     s = "\n" + s
    # console.print(f"[bold green]JIT[/][green]{s}[/]")
    # LOGGER
    # Find color tags like [bold cyan], [green], etc.
    match = re.search(r"\[([^\]]+)\]", s)
    if match:
        color_tag = match.group(1).lower().replace("bold ", "")
        level_name = COLOR_LEVEL_MAP.get(color_tag, "info")
    else:
        level_name = "info"

    # Remove all [..] tags
    clean_msg = re.sub(r"\[/?[^\]]+\]", "", s)
    # jit_logger.info(clean_msg)
    getattr(jit_logger, level_name)(clean_msg)


def remove_hostfile(settings: JitSettings) -> None:
    """Delete the GekkoFS hostfile.

    The daemon appends to this file and only removes its own line on a clean
    shutdown, so any line left by a killed daemon must be cleared by us -- see
    setup_core.hostfile_is_sane for what a stale line does to the next run.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if not settings.gkfs_hostfile:
        return
    try:
        os.remove(settings.gkfs_hostfile)
        jit_print("[yellow]Hostfile removed[/]")
    except FileNotFoundError:
        pass
    except OSError as e:
        jit_print(f"[bold red]Error removing hostfile:[/bold red]{e}", True)


def create_hostfile(settings: JitSettings) -> None:
    """
    Create a hostfile for the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    jit_print(f"[cyan]Cleaning Hostfile: {settings.gkfs_hostfile}")
    remove_hostfile(settings)

    # # Optionally, recreate the hostfile and populate it if needed
    # try:
    #     # Create the hostfile (if needed)
    #     # with open(settings.gkfs_hostfile, 'w') as file:
    #     #     for node in settings.nodes_arr[:-1]:  # Exclude the last element
    #     #         file.write(f"cpu{node}\n")
    #     # console.print(f"[cyan] Creating Hostfile: {settings.gkfs_hostfile}")
    #     pass
    # except Exception as e:
    #     console.print(f"[bold red]Error creating hostfile:[/bold red]{e}")


def format_time(elapsed: float) -> str:
    """
    Format the elapsed time in a readable way.

    Args:
        elapsed (float): Elapsed time in seconds.

    Returns:
        str: Formatted elapsed time.
    """
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


def elapsed_time(
    settings: JitSettings, runtime: JitTime, name: str, elapsed: float
) -> None:
    """
    Calculate and print the elapsed time for a specific component.

    Args:
        settings (JitSettings): The JIT settings object.
        runtime (JitTime): The JIT runtime object.
        name (str): Name of the component.
        elapsed (float): Elapsed time in seconds.
    """
    elapsed_formatted = format_time(elapsed)
    log_message = (
        f"\n[cyan]############[JIT]##############\n"
        f"# {name}\n"
        f"# time: [yellow]{elapsed_formatted} [cyan]\n"
        f"# [yellow]{elapsed} [cyan]seconds\n"
        f"##############################"
    )
    # console.print(log_message)
    jit_print(log_message)

    # Write to log file. print_to_file appends verbatim, and log_message ends
    # without a newline -- consecutive blocks only looked right because the next
    # one starts with "\n". The last block is followed by the run summary, which
    # then ran onto the same line ("##############################App time ...").
    # Write a self-terminating block instead: drop the leading newline, add a
    # trailing one. Console output is unchanged.
    print_to_file(
        log_message.lstrip("\n") + "\n",
        os.path.join(settings.log_dir, "time.log"),
    )

    if "stage" in name.lower():
        if "in" in name.lower():
            runtime.stage_in = elapsed
        elif "out" in name.lower():
            runtime.stage_out = elapsed
    elif "app" in name.lower():
        runtime.app = elapsed


def check(settings: JitSettings) -> None:
    """
    Check the files in the GekkoFS mount directory.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.dry_run or settings.exclude_daemon:
        return

    call = flaged_call(settings, f"ls -lahrt {settings.gkfs_mntdir}", exclude=["ftio"])
    try:
        files = subprocess.check_output(
            f"{call}",
            shell=True,
            text=True,
        )
        jit_print(f"[cyan] {call}: [/]\n{files}")
    except subprocess.CalledProcessError as e:
        jit_print(f"[red] Failed to list files in {settings.gkfs_mntdir}: {e}[/]")


def generate_cluster_command(
    settings: JitSettings,
    call: str,
    nodes: int = 1,
    procs_per_node: int = 1,
    additional_arguments: str = "",
    node_list: str = "",
    force_srun=False,
) -> str:
    """
    Generate a command to execute on the cluster, using either `mpiexec` or `srun`.

    Args:
        settings (JitSettings): The JIT settings object containing configuration details.
        call (str): The command to execute.
        nodes (int, optional): Number of nodes to use. Defaults to 1.
        procs_per_node (int, optional): Number of processes per node. Defaults to 1.
        additional_arguments (str, optional): Additional arguments to include in the command. Defaults to "".
        node_list (str, optional): Specific node list to use. Defaults to "".

    Returns:
        str: The generated command string ready for execution.
    """
    procs = nodes * procs_per_node

    if not settings.cluster:
        call = f"mpiexec -np {procs} --oversubscribe {additional_arguments} {call}"
    else:
        if not node_list:
            node_list = f"--nodelist={settings.single_node}"

        if settings.use_mpirun and not force_srun:
            call, procs = clean_call(call, procs)
            call = mpiexec_call(settings, call, procs, additional_arguments, node_list)
        else:
            call = srun_call(
                settings,
                call,
                nodes,
                procs_per_node,
                additional_arguments,
                node_list,
            )

    return call


def flaged_call(
    settings: JitSettings,
    call: str,
    nodes: int = 1,
    procs_per_node: int = 1,
    exclude: list = None,
    special_flags: dict = None,
) -> str:
    """
    Generate a command with appropriate flags for execution.

    Args:
        settings (JitSettings): The JIT settings object.
        call (str): Command to execute.
        nodes (int, optional): Number of nodes. Defaults to 1.
        procs_per_node (int, optional): Number of processes per node. Defaults to 1.
        exclude (list, optional): List of components to exclude. Defaults to [].
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        str: Command with appropriate flags.
    """
    if special_flags is None:
        special_flags = {}
    if exclude is None:
        exclude = []
    if settings.use_mpirun:
        call = flaged_mpiexec_call(
            settings, call, nodes * procs_per_node, exclude, special_flags
        )
    else:
        call = flaged_srun_call(
            settings, call, nodes, procs_per_node, exclude, special_flags
        )

    return call


def flaged_mpiexec_call(
    settings: JitSettings,
    call: str,
    procs: int = 1,
    exclude: list = None,
    special_flags: dict = None,
) -> str:
    """
    Generate an mpiexec command with appropriate flags.

    Args:
        settings (JitSettings): The JIT settings object.
        call (str): Command to execute.
        procs (int, optional): Number of processes. Defaults to 1.
        exclude (list, optional): List of components to exclude. Defaults to [].
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        str: mpiexec command with appropriate flags.
    """
    if special_flags is None:
        special_flags = {}
    if exclude is None:
        exclude = []
    additional_arguments = load_flags_mpiexec(
        settings, exclude=exclude, special_flags=special_flags
    )
    call, procs = clean_call(call, procs)
    # In new mode wrap the command so LD_PRELOAD is set inside bash,
    # not via mpiexec -x (avoids SLURM/MPI environment propagation issues).
    # Mirror the original condition: only when daemon flags are not excluded.
    if (
        not settings.preload_via_export
        and not settings.fuse
        and not settings.exclude_daemon
        and "demon" not in exclude
        and "preload" not in exclude
        and settings.gkfs_intercept
    ):
        call = (
            f'bash -c "LD_PRELOAD={settings.gkfs_intercept}'
            f" LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile}"
            f' {call}"'
        )
    if settings.cluster:
        call = mpiexec_call(settings, call, procs, additional_arguments)
    else:
        call = (
            f"mpiexec -np {procs} --oversubscribe "
            # f"-x LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')} "
            f"{additional_arguments} {call}"
        )

    return call


def flaged_srun_call(
    settings: JitSettings,
    call: str,
    nodes: int = 1,
    procs: int = 1,
    exclude: list = None,
    special_flags: dict = None,
) -> str:
    """
    Generate an srun command with appropriate flags.

    Args:
        settings (JitSettings): The JIT settings object.
        call (str): Command to execute.
        nodes (int, optional): Number of nodes. Defaults to 1.
        procs (int, optional): Number of processes. Defaults to 1.
        exclude (list, optional): List of components to exclude. Defaults to [].
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        str: srun command with appropriate flags.
    """
    if special_flags is None:
        special_flags = {}
    if exclude is None:
        exclude = []
    if settings.cluster:
        additional_arguments = load_flags_srun(
            settings, exclude=exclude, special_flags=special_flags
        )
        # In new mode wrap the command so LD_PRELOAD is set inside bash,
        # not via srun --export (avoids SLURM environment propagation issues).
        # Mirror the original condition: only when daemon flags are not excluded.
        if (
            not settings.preload_via_export
            and not settings.fuse
            and not settings.exclude_daemon
            and "demon" not in exclude
            and "preload" not in exclude
            and settings.gkfs_intercept
        ):
            call = (
                f'bash -c "LD_PRELOAD={settings.gkfs_intercept}'
                f" LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile}"
                f' {call}"'
            )
        call = srun_call(settings, call, nodes, procs, additional_arguments)
    else:
        call = flaged_mpiexec_call(
            settings, call, procs, exclude=exclude, special_flags=special_flags
        )

    return call


def load_flags_mpiexec(
    settings: JitSettings, exclude: list = None, special_flags: dict = None
) -> str:
    """
    Load flags for mpiexec command.

    Args:
        settings (JitSettings): The JIT settings object.
        exclude (list, optional): List of components to exclude. Defaults to [].
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        str: Flags for mpiexec command.
    """
    if special_flags is None:
        special_flags = {}
    if exclude is None:
        exclude = []
    default = load_defauts(settings, special_flags)
    additional_arguments = ""
    if not settings.exclude_ftio and "ftio" not in exclude:
        additional_arguments += (
            f"-x LIBGKFS_METRICS_IP_PORT={default['LIBGKFS_METRICS_IP_PORT']} "
            f"-x LIBGKFS_ENABLE_METRICS={default['LIBGKFS_ENABLE_METRICS']} "
        )
    if not settings.exclude_proxy and "proxy" not in exclude:
        additional_arguments += (
            f"-x LIBGKFS_PROXY_PID_FILE={default['LIBGKFS_PROXY_PID_FILE']} "
        )
    if not settings.exclude_daemon and "demon" not in exclude:
        if "demon_log" not in exclude:
            additional_arguments += (
                f"-x LIBGKFS_LOG={default['LIBGKFS_LOG']} "
                f"-x LIBGKFS_LOG_OUTPUT={default['LIBGKFS_LOG_OUTPUT']} "
            )
        if "hostfile" not in exclude:
            additional_arguments += (
                f"-x LIBGKFS_HOSTS_FILE={default['LIBGKFS_HOSTS_FILE']} "
            )
        if "preload" not in exclude and not settings.fuse and settings.preload_via_export:
            additional_arguments += f"-x LD_PRELOAD={default['LD_PRELOAD']} "

    additional_arguments += get_env(settings, "mpi")

    return additional_arguments


def load_flags_srun(
    settings: JitSettings, exclude: list = None, special_flags: dict = None
) -> str:
    """
    Load flags for srun command.

    Args:
        settings (JitSettings): The JIT settings object.
        exclude (list, optional): List of components to exclude. Defaults to [].
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        str: Flags for srun command.
    """
    if special_flags is None:
        special_flags = {}
    if exclude is None:
        exclude = []
    default = load_defauts(settings, special_flags)
    additional_arguments = ""
    if not settings.exclude_ftio and "ftio" not in exclude:
        additional_arguments += (
            f"LIBGKFS_ENABLE_METRICS={default['LIBGKFS_ENABLE_METRICS']},"
            f"LIBGKFS_METRICS_IP_PORT={default['LIBGKFS_METRICS_IP_PORT']},"
        )
    if not settings.exclude_proxy and "proxy" not in exclude:
        additional_arguments += (
            f"LIBGKFS_PROXY_PID_FILE={default['LIBGKFS_PROXY_PID_FILE']},"
        )
    if not settings.exclude_daemon and "demon" not in exclude:
        if "demon_log" not in exclude:
            additional_arguments += (
                f"LIBGKFS_LOG=\"{default['LIBGKFS_LOG']}\","
                f"LIBGKFS_LOG_OUTPUT={default['LIBGKFS_LOG_OUTPUT']},"
            )
        if "hostfile" not in exclude:
            additional_arguments += f"LIBGKFS_HOSTS_FILE={default['LIBGKFS_HOSTS_FILE']},"
        if "preload" not in exclude and not settings.fuse and settings.preload_via_export:
            additional_arguments += f"LD_PRELOAD={default['LD_PRELOAD']},"
    # env if needed
    additional_arguments += get_env(settings, "srun")

    return additional_arguments


def load_defauts(settings: JitSettings, special_flags: dict = None):
    """
    Load default flags for a command.

    Args:
        settings (JitSettings): The JIT settings object.
        special_flags (dict, optional): Special flags for the command. Defaults to {}.

    Returns:
        dict: Default flags for the command.
    """
    if special_flags is None:
        special_flags = {}
    default = {
        "LIBGKFS_METRICS_IP_PORT": f"{settings.address_ftio}:{settings.port_ftio}",
        "LIBGKFS_ENABLE_METRICS": "on",
        "LIBGKFS_PROXY_PID_FILE": f"{settings.gkfs_proxyfile}",
        "LIBGKFS_LOG": "errors" if settings.debug_lvl <= 1 else "all",
        "LIBGKFS_LOG_OUTPUT": f"{settings.gkfs_client_log}",
        "LIBGKFS_HOSTS_FILE": f"{settings.gkfs_hostfile}",
        "LD_PRELOAD": f"{settings.gkfs_intercept}",
    }
    if special_flags:
        for key, value in special_flags.items():
            if key in default:
                if value:
                    default[key] = value
                else:
                    default.pop(key)
    return default


def mpiexec_call(
    settings: JitSettings,
    command: str,
    procs: int = 1,
    additional_arguments: str = "",
    node_list: str = "",
) -> str:
    """
    Generate an mpiexec command.

    Args:
        settings (JitSettings): The JIT settings object.
        command (str): Command to execute.
        procs (int, optional): Number of processes. Defaults to 1.
        additional_arguments (str, optional): Additional arguments for the command. Defaults to "".
        nodelist (str, optional): List of nodes. Defaults to "".

    Returns:
        str: mpiexec command.
    """
    if not node_list:
        node_list = f"--hostfile {settings.mpi_hostfile}"
    else:
        if "--host" not in node_list:
            node_list = f"--host {node_list}"

    call = (
        f" mpiexec -np {procs} --oversubscribe "
        f"{node_list} -map-by node "
        f"{additional_arguments} "
        f"{settings.task_set_0} {command}"
    )
    return call


def srun_call(
    settings: JitSettings,
    command: str,
    nodes: int = 1,
    procs: int = 1,
    additional_arguments: str = "",
    node_list: str = "",
) -> str:
    """
    Generate an srun command.

    Args:
        settings (JitSettings): The JIT settings object.
        command (str): Command to execute.
        nodes (int, optional): Number of nodes. Defaults to 1.
        procs (int, optional): Number of processes. Defaults to 1.
        additional_arguments (str, optional): Additional arguments for the command. Defaults to "".
        node_list (str, optional): List of nodes. Defaults to "".

    Returns:
        str: srun command.
    """
    if not node_list:
        node_list = (
            settings.single_node_command if nodes == 1 else settings.app_nodes_command
        )
    else:
        if "--nodelist" not in node_list:
            node_list = f"--nodelist={node_list}"

    # MPI handling
    if "mpirun" in command or "mpiexec" in command:
        nprocs = None
        command, nprocs = clean_mpi_command(command)
        if nprocs is not None:
            nodes = settings.app_nodes
            procs = max(1, int(nprocs / nodes))

    call = (
        f"srun "
        f"--export=ALL,{additional_arguments} "
        f"--jobid={settings.job_id} {node_list} --disable-status "
        f"-N {nodes} --ntasks={nodes * procs} "
        f"--cpus-per-task={procs} --ntasks-per-node={procs} "
        f"--overcommit --overlap --oversubscribe --mem=0 "
        f"{settings.task_set_0} {command}"
    )

    return call


def clean_call(call: str, procs: int) -> tuple:
    """
    Clean a command by removing mpiexec or mpirun.

    Args:
        call (str): Command to clean.
        procs (int): Number of processes.

    Returns:
        tuple: Cleaned command and number of processes.
    """
    if "mpiexec" in call or "mpirun" in call:
        call = call.replace("mpiexec", "").replace("mpirun", "").strip()
        parts = call.split()
        if "-np" in parts:
            procs = int(parts[parts.index("-np") + 1])
            parts = [part for part in parts if part != "-np" and part != str(procs)]
            call = " ".join(parts)
    else:
        pass

    return call, procs


def clean_mpi_command(command: str):
    """
    Convert an MPI command (mpirun/mpiexec) into a clean command
    and extract the number of processes (-np) if present.

    Args:
        command (str): MPI command string.

    Returns:
        tuple:
            - cleaned_command (str)
            - nprocs (int | None)
    """

    nprocs = None
    parts = command.split()
    cleaned = []

    i = 0
    while i < len(parts):
        if parts[i] in ("mpirun", "mpiexec"):
            i += 1
            continue

        if parts[i] == "-np" and i + 1 < len(parts):
            nprocs = int(parts[i + 1])
            i += 2
            continue

        cleaned.append(parts[i])
        i += 1

    return " ".join(cleaned), nprocs


def get_executable_realpath(executable_name: str, search_location: str = None) -> str:
    """
    Get the real path of an executable.

    Args:
        executable_name (str): Name of the executable.
        search_location (str, optional): Location to search for the executable. Defaults to None.

    Returns:
        str: Real path of the executable.
    """
    if search_location:
        potential_path = os.path.join(search_location, executable_name)
        if os.path.isfile(potential_path) and os.access(potential_path, os.X_OK):
            try:
                return os.path.realpath(potential_path)
            except Exception as e:
                print(f"Warning: Could not resolve real path for {potential_path}: {e}")
                return executable_name

    # Fall back to searching in the system PATH
    executable_path = shutil.which(executable_name)
    if executable_path:
        try:
            return os.path.realpath(executable_path)
        except Exception as e:
            print(f"Warning: Could not resolve real path for {executable_name}: {e}")

    # Fallback: return the name if not found
    jit_print(f" Application: {executable_name}")
    return executable_name


def update_mpi_hostfile(settings: JitSettings) -> None:
    """
    Update the hostfile for MPI.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    # Command to get the list of hostnames for the job
    squeue_command = f"squeue -j {settings.job_id} -o '%N' | tail -n +2"
    scontrol_command = f"scontrol show hostname $({squeue_command})"

    # Execute the command and capture the hostnames
    hostnames = (
        subprocess.check_output(scontrol_command, shell=True).decode().splitlines()
    )

    # Write the hostnames to the file, excluding the Ftio node
    with open(f"{settings.mpi_hostfile}", "w") as hostfile:
        for hostname in hostnames:
            if hostname.strip() != settings.ftio_node:
                hostfile.write(hostname + "\n")


def log_failed_jobs(settings: JitSettings, info: str) -> None:
    """
    Log failed jobs to a file.

    Args:
        settings (JitSettings): The JIT settings object.
        info (str): Information about the failed job.
    """
    parent = os.path.dirname(settings.log_dir)
    execution_path = os.path.join(parent, "failed_execution.txt")
    if settings.cmd_call:
        try:
            jit_print(
                f"[yellow] Adding execution to list of failed jobs in {execution_path}.[/]"
            )
            with open(execution_path, "a") as file:
                file.write(f"- {info}\n\t{settings.cmd_call}\n")
            settings.cmd_call = ""
        except Exception as e:
            jit_print(
                f"[Red] Killing Job: {info}. No entry added to failed_execution due to {e}.\n Exiting script.[/]"
            )


def log_execution(settings: JitSettings) -> None:
    """
    Log job executions with their status. Update the status to 'complete' if the job call matches.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    parent = os.path.dirname(settings.log_dir)
    execution_path = os.path.join(parent, "executions.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = f"(Log dir: {os.path.abspath(settings.log_dir)})"

    if settings.cmd_call:
        try:
            # Ensure the file exists before reading
            if not os.path.exists(execution_path):
                # If the file doesn't exist, create it and add an initial comment
                with open(execution_path, "w") as file:
                    file.write("# Execution log file\n")

            # Check if job is already in the log file
            job_exists = False
            updated_lines = []
            with open(execution_path) as file:
                lines = file.readlines()
                for line in lines:
                    if info in line and settings.cmd_call in line:
                        # If the job info and cmd_call match, replace status with complete and timestamp
                        updated_lines.append(
                            f"- [{timestamp}] Completed:  {settings.cmd_call} {info}\n"
                        )
                        job_exists = True
                        jit_print(
                            f"[yellow] Job marked as completed in {execution_path}.[/]"
                        )
                    else:
                        updated_lines.append(line)

            # If the job does not exist, add it with the 'pending' status and timestamp
            if not job_exists:
                jit_print(
                    f"[yellow] Adding execution to list of pending jobs in {execution_path}.[/]"
                )
                updated_lines.append(
                    f"- [{timestamp}] Pending: {settings.cmd_call} {info}\n"
                )

            # Write the updated lines back to the file
            with open(execution_path, "w") as file:
                file.writelines(updated_lines)

        except Exception as e:
            jit_print(f"[red] Error logging execution: {e}[/]")


def set_env(settings: JitSettings) -> None:
    """
    Set environment variables for the JIT environment.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    for _, key in enumerate(settings.env_var):
        jit_print(f"[green] Setting {key} to {settings.env_var[key]}[/]")
        os.environ[str(key)] = str(settings.env_var[key])
        jit_print(f"[cyan] {key} set to {os.getenv(str(key))}[/]")
        if os.getenv(str(key)) is None:
            jit_print(
                f"[red bold] {key} not set, do this manually:\nexport {key}={settings.env_var[key]}[/]"
            )


def get_env(settings: JitSettings, mode: str = "srun") -> str:
    """
    Get environment variables for a command.

    Args:
        settings (JitSettings): The JIT settings object.
        mode (str, optional): Mode for the command. Defaults to "srun".

    Returns:
        str: Environment variables for the command.
    """
    env = ""
    if "mpi" in mode:
        env = " ".join(f"-x {key}={value}" for key, value in settings.env_var.items())
    elif "srun":  # srun
        env = ",".join(f"{key}={value}" for key, value in settings.env_var.items())
        env = env + ","
    else:
        pass
    return env


def bandwidth_path(settings: JitSettings) -> str:
    """Where FTIO's dump_json writes bandwidth.json: the predictor cds into the
    run's log dir first (see the predictor call in setup_core), so it lands here."""
    return os.path.join(settings.log_dir, "bandwidth.json")


def clear_bandwidth(settings: JitSettings) -> None:
    """
    Drop any stale bandwidth.json in this run's log dir before FTIO starts.

    FTIO writes bandwidth.json into the log dir directly. The dir is fresh per
    run so there is normally nothing to remove, but if FTIO sees no I/O at all
    (e.g. the app never wrote into the gkfs mount) it produces no file -- this
    guarantees a leftover can never be mistaken for this run's trace.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.exclude_ftio:
        return
    with contextlib.suppress(OSError):
        os.remove(bandwidth_path(settings))


def save_bandwidth(settings: JitSettings) -> None:
    """
    Verify FTIO produced a bandwidth trace in this run's log dir.

    No copy is needed: the predictor cds into the log dir so dump_json writes
    bandwidth.json there directly. This only warns when the file is absent.

    Args:
        settings (JitSettings): The JIT settings object.
    """
    if settings.exclude_ftio:
        return
    if not os.path.exists(bandwidth_path(settings)):
        jit_print(
            "[yellow]No bandwidth.json was produced: FTIO captured no I/O. "
            "Does the application write into the gkfs mountdir?"
        )


def parse_time(line: str) -> float | None:
    """
    Parses a line starting with 'real' to extract a duration in seconds.

    Supports:
    - Compound formats like '1d2h3m4.56s'
    - Simple float seconds like '123.456'
    - European decimal format with commas (e.g., '1m1,129s')

    Args:
        line: A string that should start with 'real' followed by a duration.

    Returns:
        The parsed time in seconds as a float, or None if parsing fails.
    """
    try:
        # Remove 'real' and strip whitespace
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] != "real":
            return None
        duration_str = parts[1].replace(",", ".").strip()

        # Try parsing compact duration format (e.g., 1d2h3m4.56s)
        pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?$"
        match = re.fullmatch(pattern, duration_str)
        if match:
            days = int(match.group(1)) if match.group(1) else 0
            hours = int(match.group(2)) if match.group(2) else 0
            minutes = int(match.group(3)) if match.group(3) else 0
            seconds = float(match.group(4)) if match.group(4) else 0.0
            return days * 86400 + hours * 3600 + minutes * 60 + seconds

        # Try parsing as a raw float (e.g., "123.456")
        return float(duration_str)
    except Exception as parse_err:
        jit_print(
            f"[yellow]Warning: Failed to parse real time line: '{line.strip()}' — {parse_err}[/]"
        )
        return None


def extract_accurate_time(settings: JitSettings, real_time: float) -> float:
    """
    Attempts to extract a more accurate 'real' time from the application error log.

    Args:
        settings: A JitSettings object containing the path to the error log file.
        real_time: A fallback real time value (e.g., measured with time.time()).

    Returns:
        The smaller of the fallback time and the extracted time from the error log.
    """
    try:
        with open(settings.app_err) as file:
            for line in file:
                if line.startswith("real"):
                    accurate_real_time = parse_time(line)
                    if accurate_real_time is not None:
                        real_time = min(accurate_real_time, real_time)
                    break
    except Exception as e:
        jit_print(f"[red]Could not extract real time due to\n{e}[/]\n")

    return real_time
