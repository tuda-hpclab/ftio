"""
Parallel ingestion for ZMQ predictors: partition the drained messages across
worker processes, compute a per-worker application-level bandwidth partial, and
fold the partials back into a single series.

The reduction is exact because the application-level bandwidth is an additive
sum of per-server box functions. Summing is associative and commutative, so
splitting the messages, overlapping each subset, and merging the resulting step
functions with ``overlap_two_series`` yields the same series as overlapping all
messages at once (see test_parallel_ingest.py).

Default is a single worker (``n_workers=1``), i.e. the original behavior.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import os
from collections.abc import Callable

from ftio.parse.bandwidth import overlap_two_series


def partition(items: list, n_workers: int) -> list[list]:
    """Split items into up to n_workers near-equal contiguous chunks."""
    n_workers = max(1, min(n_workers, len(items))) if items else 1
    k, r = divmod(len(items), n_workers)
    chunks = []
    start = 0
    for i in range(n_workers):
        end = start + k + (1 if i < r else 0)
        chunks.append(items[start:end])
        start = end
    return [c for c in chunks if c]


def _reduce_linear(level: list[tuple]) -> tuple[list, list]:
    """Fold left-to-right into one accumulator: O(L W) merge work."""
    b_acc, t_acc = level[0]
    for b, t in level[1:]:
        b_arr, t_arr = overlap_two_series(b_acc, t_acc, b, t)
        b_acc, t_acc = list(b_arr), list(t_arr)
    return b_acc, t_acc


def _reduce_tree(level: list[tuple]) -> tuple[list, list]:
    """Balanced pairwise (binary-tree) fold: O(L log W) merge work.

    Level-wise pairwise merges are independent, so this also parallelizes.
    """
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                b1, t1 = level[i]
                b2, t2 = level[i + 1]
                b_arr, t_arr = overlap_two_series(b1, t1, b2, t2)
                nxt.append((list(b_arr), list(t_arr)))
            else:
                nxt.append(level[i])  # odd one out carries to next level
        level = nxt
    return level[0]


def reduce_partials(partials: list[tuple], strategy: str = "tree") -> tuple[list, list]:
    """Fold per-worker (b, t) step-function partials into one (b, t) series.

    The merge (overlap_two_series) is associative and commutative, so both
    strategies give the same result (up to float rounding). ``tree`` (default)
    does a balanced pairwise reduction, O(L log W); ``linear`` folds into one
    accumulator, O(L W) — kept for reference/debugging. Empty partials skipped.
    """
    level = [(list(b), list(t)) for b, t in partials if len(b) > 0]
    if not level:
        return [], []
    if strategy == "linear":
        return _reduce_linear(level)
    return _reduce_tree(level)


def resample_step(b, t, grid):
    """Sample a left-anchored step function (b in effect from t[k]) onto grid.

    Turns a variable-length (b, t) partial into a fixed-length vector on a
    common grid, so partials can be summed element-wise (and shipped cheaply
    across a process boundary). Exact w.r.t. the sampled representation the DFT
    uses; the full-resolution step function is not recoverable from it.
    """
    import numpy as np

    grid = np.asarray(grid)
    if len(b) == 0:
        return np.zeros(len(grid))
    b = np.asarray(b)
    t = np.asarray(t)
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, b[idx.clip(min=0)], 0.0)


def resolve_workers(n_workers: int) -> int:
    """Clamp the requested worker count to the available CPU budget.

    Avoids oversubscription: never more workers than usable cores.
    """
    try:
        budget = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        budget = os.cpu_count() or 1
    return max(1, min(n_workers, budget))


def parallel_overlap(
    items: list,
    map_fn: Callable[[list], tuple],
    n_workers: int = 1,
    reduce_fn: Callable[[list[tuple]], tuple] = reduce_partials,
    backend: str = "thread",
) -> tuple[list, list]:
    """Map map_fn over partitions of items, then reduce the partials.

    ``map_fn`` takes a chunk of items and returns a ``(b, t)`` partial. With
    ``n_workers == 1`` this runs inline. ``backend`` selects how chunks are
    mapped: ``thread`` (default) shares memory so there is no IPC — it only
    speeds things up for GIL-releasing work (the numba overlap is compiled
    ``nogil``); ``process`` avoids the GIL but pays pickling/IPC per round
    (measured IPC-bound for this workload). The map/reduce math is identical
    for all modes, so correctness does not depend on the execution mode.
    """
    if not items:
        return [], []

    n = resolve_workers(n_workers)
    chunks = partition(items, n)
    if len(chunks) == 1:
        return reduce_fn([map_fn(chunks[0])])

    if backend == "process":
        from multiprocessing import Pool

        with Pool(processes=len(chunks)) as pool:
            partials = pool.map(map_fn, chunks)
    else:
        from multiprocessing.pool import ThreadPool

        with ThreadPool(len(chunks)) as pool:
            partials = pool.map(map_fn, chunks)
    return reduce_fn(partials)
