"""
Tests for the bounded prediction-process pool (enforce_limit).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import time

from ftio.multiprocessing.async_process import enforce_limit, handle_in_process


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def test_enforce_limit_caps_concurrency():
    """With cap K, never more than K prediction processes run at once."""
    procs = []
    for _ in range(6):
        procs = enforce_limit(procs, 2)
        procs.append(handle_in_process(_sleep, (0.3,)))
        assert sum(p.is_alive() for p in procs) <= 2
    for p in procs:
        p.join()


def test_enforce_limit_unlimited_when_zero():
    """Cap <= 0 disables the limit (original behaviour)."""
    procs = []
    for _ in range(5):
        procs = enforce_limit(procs, 0)
        procs.append(handle_in_process(_sleep, (0.3,)))
    # all spawned without waiting -> several alive concurrently
    assert sum(p.is_alive() for p in procs) >= 2
    for p in procs:
        p.join()


def test_enforce_limit_reaps_finished():
    """Finished processes are reaped even under a generous cap."""
    procs = [handle_in_process(_sleep, (0.01,))]
    time.sleep(0.2)  # let it finish
    procs = enforce_limit(procs, 10)
    assert procs == []
