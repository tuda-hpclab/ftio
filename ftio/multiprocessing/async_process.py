"""
Performs action async to current process

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Process


def handle_in_process(function: Callable, args) -> Process:
    """Handle function in a dedicated process

    Args:
        function (Callable): function name
        args (argparse): arguments passed to function

    Returns:
        None
    """
    process = Process(target=function, args=args)
    # print(f'Process {process.name} (PID {os.getpid()}) started to execute {function}')
    process.start()
    # print(f'Process {process.name} (PID {os.getpid()}) ended')
    # print(f"Process {process} created")
    return process


def join_procs(procs: list, blocking: bool = True) -> list:
    """
    Joins finished processes safely. Optionally non-blocking.

    Args:
        procs (list): list of multiprocessing.Process objects
        blocking (bool): if True, join finished processes immediately
    Returns:
        list: updated list of alive processes
    """
    alive_procs = []
    for p in procs:
        if p.is_alive():
            alive_procs.append(p)
        else:
            if blocking:
                p.join()  # join if requested
    return alive_procs


def enforce_limit(procs: list, max_concurrent: int) -> list:
    """Cap the number of concurrent processes (bounded pool).

    First reaps finished processes (non-blocking). Then, if ``max_concurrent``
    is a positive number, blocks on the OLDEST process until fewer than
    ``max_concurrent`` remain alive — so draining/listening continues right up
    to the spawn point and only the oldest in-flight prediction is waited on.
    ``max_concurrent <= 0`` means unlimited (original behaviour). ``= 1`` is the
    debounce case (one prediction at a time).

    Args:
        procs (list): list of multiprocessing.Process objects (oldest first)
        max_concurrent (int): concurrency cap; <= 0 disables the cap
    Returns:
        list: alive processes, with room for one more when a cap is set
    """
    procs = join_procs(procs, blocking=False)
    if max_concurrent and max_concurrent > 0:
        while len(procs) >= max_concurrent:
            procs[0].join()  # wait for the oldest to finish, then re-reap
            procs = join_procs(procs, blocking=False)
    return procs
