"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Okt 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse
import subprocess


def get_modification_time(args: argparse.Namespace, file_name: str) -> float:
    """
    Retrieves the last modification time of a file.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        file_name (str): Name of the file.

    Returns:
        float: Last modification time of the file.
    """
    # Ask for mtime (%Y) and ctime (%Z) and keep the newer. GekkoFS does not
    # maintain mtime and answers 0 for everything in the mount, which turned
    # `time.time() - mtime` into the current epoch and defeated every "is this
    # file still being written?" check. It does keep ctime. On a normal
    # filesystem mtime leads and ctime tracks it, so the max is right in both.
    output = preloaded_call(args, f"stat --format='%Y %Z' {file_name}")
    stamps = [float(x) for x in output.split() if x.strip()]
    return max(stamps) if stamps else 0.0


def preloaded_call(args: argparse.Namespace, call: str) -> str:
    """
    Executes a shell command with GekkoFS preloaded environment variables.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        call (str): Shell command to execute.

    Returns:
        str: Output of the shell command.
    """
    hostfile = f"LIBGKFS_HOSTS_FILE={args.host_file}"
    if args.node:  # fuse requires srun, no need for preload
        call = (
            f"srun --nodelist={args.node} --export=ALL,{hostfile} "
            f"-N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 "
            f"--overcommit --overlap --oversubscribe --mem=0 "
            f"{call}"
        )
    else:
        call = f"{hostfile} LD_PRELOAD={args.ld_preload} {call}"
    return subprocess.check_output(call, shell=True, text=True)
