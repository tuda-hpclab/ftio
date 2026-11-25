"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import fcntl
import os
import signal
from time import sleep

from rich.console import Console

from ftio.multiprocessing.async_process import join_procs

CONSOLE = Console()


def monitor(name: str, _cached_stamp: str, procs: list = None) -> tuple[str, list]:
    """Monitors a file for change and can optionally join processes in the mean time

    Args:
        name (str): filename
        _cached_stamp (str): time stamp since last time
        procs (list, optional): List of process to join. Defaults to [].

    Returns:
        str: _description_
    """
    if procs is None:
        procs = []
    return monitor_stat(name, _cached_stamp, procs)
    # try:
    #     return monitor_stat(name, _cached_stamp, procs)
    # finally:
    #     return monitor_fcntl(name, _cached_stamp, procs)


#! Method 1
def monitor_stat(name: str, _cached_stamp: str, procs: list) -> tuple[str, list]:
    """Monitors a file for changes

    Args:
        name (str): _description_
        _cached_stamp (str): change time stamp
        procs: list or process

    Returns:
        str: _description_
    """
    if _cached_stamp == "":
        stream = os.popen(f"stat -c %z {name} 2>&1")
        stamp = stream.read()
        CONSOLE.print(f"[purple][PREDICTOR][/] Monitoring file {name}")
        CONSOLE.print(f"[purple][PREDICTOR][/] Stamp is {stamp}")
        return stamp, procs
    else:
        sleep(1)
        while True:
            procs = join_procs(procs)
            stream = os.popen(f"stat -c %z {name} 2>&1")
            stamp = stream.read()
            if stamp != _cached_stamp:
                CONSOLE.print(
                    f"[purple][PREDICTOR][/][green bold] Stamp changed[/] to {stamp}"
                )
                sleep(0.2)
                return stamp, procs


def monitor_list(
    name: list, n_buffers, _cached_stamp: dict = None, procs: list = None
) -> tuple[dict, list]:
    """Monitors a file for changes

    Args:
        name (str): _description_
        _cached_stamp (str): change time stamp
        procs: list or process

    Returns:
        str: _description_
    """
    if procs is None:
        procs = []
    if _cached_stamp is None:
        _cached_stamp = {}
    if not _cached_stamp:
        stamp = {}
        for i in name:
            stream = os.popen(f"stat -c %z {i} 2>&1")
            file_stamp = stream.read()
            stamp[i] = file_stamp
            CONSOLE.print(
                f"[purple][PREDICTOR][/] Monitoring file {name.index(i)}/{n_buffers} {i}"
                f"[purple][PREDICTOR][/] Stamp is {stamp[i]}"
            )
        return stamp, procs
    else:
        sleep(1)
        seen = []
        counter = n_buffers
        while True:
            procs = join_procs(procs)
            for _, i in enumerate(_cached_stamp):
                stream = os.popen(f"stat -c %z {i} 2>&1")
                file_stamp = stream.read()
                if file_stamp != _cached_stamp[i] and _cached_stamp[i] not in seen:
                    counter = counter - 1
                    seen.append(_cached_stamp[i])
                    CONSOLE.print(
                        f"[purple][PREDICTOR][/][green bold] Stamp changed[/] to {file_stamp}"
                        f"[purple][PREDICTOR][/] {n_buffers - counter}/{n_buffers} files changed"
                    )

            if counter == 0:
                for i in name:
                    stream = os.popen(f"stat -c %z {i} 2>&1")
                    file_stamp = stream.read()
                    _cached_stamp[i] = file_stamp
                return _cached_stamp, procs

            sleep(0.2)


#! Method 2
def monitor_fcntl(name: str, _cached_stamp: str, procs: list) -> tuple[str, list]:
    if _cached_stamp == 0:
        return "", procs

    w = watcher(name)
    signal.signal(signal.SIGIO, w.handler)
    fd = os.open(w.path, os.O_RDONLY)
    fcntl.fcntl(fd, fcntl.F_SETSIG, 0)
    fcntl.fcntl(
        fd,
        fcntl.F_NOTIFY,
        fcntl.DN_ACCESS
        | fcntl.DN_MODIFY
        | fcntl.DN_CREATE
        | fcntl.DN_DELETE
        | fcntl.DN_RENAME
        | fcntl.DN_ATTRIB
        | fcntl.DN_MULTISHOT,
    )
    while w.flag:
        sleep(0.1)
        procs = join_procs(procs)

    return str(w.stamp), procs


class watcher:
    def __init__(self, name):
        self.flag = True
        if "/" in name:
            index = name.rfind("/")
            self.path = name[:index]
            self.name = name[index + 1 :]
        else:
            self.path = os.getcwd()
            self.name = name
        self.absolute_name = f"{self.path}/{self.name}"
        print(f"Monitoring file {self.absolute_name} in path:{self.path}")
        self.stamp = os.stat(self.absolute_name).st_mtime
        CONSOLE.print(f"[purple][PREDICTOR][/] Stamp is {self.stamp}")

    def handler(self, signum, path):
        CONSOLE.print(f"[purple][PREDICTOR][/] Folder {path} modified ")
        stamp = os.stat(self.absolute_name).st_mtime
        if self.stamp != stamp:
            self.flag = False
            CONSOLE.print(
                f"[purple][PREDICTOR][/][green bold] Stamp changed[/] to {stamp}"
            )
