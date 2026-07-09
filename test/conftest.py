"""
Shared pytest configuration.

Two process-related settings, both applied before any test module is imported.

1. Native thread pools are pinned to one thread. scikit-learn (via
   ftio.prediction.group) starts an OpenMP pool on first use, leaving ~38 OS
   threads alive for the rest of the session. Those threads are invisible to
   threading.enumerate() but not to os.fork(), and forking a process with a
   live OpenMP runtime is a known deadlock source: the child inherits locked
   OpenMP mutexes with no threads to release them. Python 3.14 rightly warns
   about it. Pinning the pools removes the hazard instead of hiding the
   warning, and makes the suite faster.

2. The "fork" start method is restored. Python 3.14 defaults to "forkserver",
   where every pool worker re-imports numpy and a single Pool() costs ~4.4 s
   instead of ~0.03 s. Production forks (see ftio/api/gekkoFs/file_queue.py),
   so the suite matches it. Guarded on availability: Windows has no "fork" and
   the suite then runs with the platform default.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import contextlib
import os

# Must happen before numpy / scikit-learn are imported for the first time.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402

if "fork" in mp.get_all_start_methods():
    # RuntimeError if the start method was already fixed by an earlier import.
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("fork")
