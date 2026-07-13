"""
Experiment A — FTIO ingestion scaling / overhead micro-benchmark.

Answers reviewers R2/R3: "a single FTIO instance becomes a bottleneck at
hundreds of GekkoFS servers." We measure the wall-clock cost of turning one
flush round of N server messages into the application-level bandwidth
(parse + overlap), which is the only stage that scales with the server count
(the DFT runs on the reduced series, whose length is set by the time window,
not by N).

We sweep N and, for each N, compare the single-process ingest against the
fan-out (--ingest-workers > 1). The reported latency is compared against a
typical flush interval to show the ingest cost stays well below the I/O period.

Run:
    python -m ftio.api.gekkoFs.jit.experiments.exp_a_scaling
    python -m ftio.api.gekkoFs.jit.experiments.exp_a_scaling --servers 8 64 512 --workers 1 4

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse
import time

from ftio.api.gekkoFs.ftio_gekko import _parse_overlap_chunk, ingest_app_bandwidth
from ftio.api.gekkoFs.jit.experiments.synthetic_messages import make_round
from ftio.prediction.parallel_ingest import partition, reduce_partials


def measure_ingest(
    msgs: list[bytes], n_workers: int, repeat: int = 3, backend: str = "thread"
) -> float:
    """Return the best-of-`repeat` ingest latency (s) for the given workers.

    backend "thread" (nogil overlap, no IPC) or "process" (IPC-bound).
    """
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        b, _t, _bytes, _flush, _ext = ingest_app_bandwidth(
            msgs, "w", n_workers, backend=backend
        )
        dt = time.perf_counter() - t0
        assert b, "ingest produced no bandwidth"
        best = min(best, dt)
    return best


def measure_ingest_warmpool(msgs: list[bytes], n_workers: int, repeat: int = 3) -> float:
    """Fan-out latency with a persistent (warm) pool, isolating compute cost.

    A deployed predictor would keep the pool alive across flush rounds; this
    reuses one pool so the per-round cost excludes process startup.
    """
    from multiprocessing import Pool

    chunks = partition(msgs, n_workers)
    best = float("inf")
    with Pool(processes=len(chunks)) as pool:
        pool.map(_parse_overlap_chunk, [([], "w")])  # warm up workers
        for _ in range(repeat):
            t0 = time.perf_counter()
            results = pool.map(_parse_overlap_chunk, [(c, "w") for c in chunks])
            b, _t = reduce_partials([(r[0], r[1]) for r in results])
            dt = time.perf_counter() - t0
            assert b, "ingest produced no bandwidth"
            best = min(best, dt)
    return best


def measure_ingest_resample_warm(
    msgs: list[bytes], n_workers: int, repeat: int = 3, hz: float = 100.0
) -> float:
    """Warm process pool where workers resample to a small shared grid.

    Only a fixed-length vector crosses the process boundary (no big-array IPC).
    """
    from multiprocessing import Pool

    import numpy as np

    from ftio.api.gekkoFs.ftio_gekko import _parse_overlap_resample_chunk, _peek_horizon

    chunks = partition(msgs, n_workers)
    grid = np.arange(0.0, _peek_horizon(msgs) + 1.0 / hz, 1.0 / hz)
    best = float("inf")
    with Pool(processes=len(chunks)) as pool:
        pool.map(_parse_overlap_resample_chunk, [([], "w", grid)])  # warm up
        for _ in range(repeat):
            t0 = time.perf_counter()
            results = pool.map(
                _parse_overlap_resample_chunk, [(c, "w", grid) for c in chunks]
            )
            rb = np.sum([r[0] for r in results], axis=0)
            dt = time.perf_counter() - t0
            assert rb.any(), "ingest produced no bandwidth"
            best = min(best, dt)
    return best


def measure_ingest_resample_thread(
    msgs: list[bytes], n_workers: int, repeat: int = 3, hz: float = 100.0
) -> float:
    """Thread pool where workers resample to a small shared grid.

    Threads share memory (no IPC) but the parse+overlap is GIL-bound, so
    resampling does not rescue the thread backend — kept to complete the
    {thread, process} x {full, resample} matrix.
    """
    from multiprocessing.pool import ThreadPool

    import numpy as np

    from ftio.api.gekkoFs.ftio_gekko import _parse_overlap_resample_chunk, _peek_horizon

    chunks = partition(msgs, n_workers)
    grid = np.arange(0.0, _peek_horizon(msgs) + 1.0 / hz, 1.0 / hz)
    best = float("inf")
    with ThreadPool(len(chunks)) as pool:
        for _ in range(repeat):
            t0 = time.perf_counter()
            results = pool.map(
                _parse_overlap_resample_chunk, [(c, "w", grid) for c in chunks]
            )
            rb = np.sum([r[0] for r in results], axis=0)
            dt = time.perf_counter() - t0
            assert rb.any(), "ingest produced no bandwidth"
            best = min(best, dt)
    return best


def run_backend_matrix(
    n_servers: int, n_events: int, workers: int = 4, repeat: int = 3
) -> dict:
    """{thread, process} x {full, resample} latencies vs single-process."""
    msgs = make_round(n_servers, n_events=n_events, seed=0)
    res = {
        "single": measure_ingest(msgs, 1, repeat),
        "thread_full": measure_ingest(msgs, workers, repeat, backend="thread"),
        "process_full": measure_ingest(msgs, workers, repeat, backend="process"),
        "thread_resample": measure_ingest_resample_thread(msgs, workers, repeat),
        "process_resample": measure_ingest_resample_warm(msgs, workers, repeat),
    }
    print(f"servers={n_servers} events/msg={n_events} workers={workers}")
    base = res["single"]
    for k, v in res.items():
        tag = "" if k == "single" else f"  ({base/v:.1f}x)"
        print(f"  {k:18s} {v*1e3:8.1f} ms{tag}")
    return res


def run_sweep(
    servers: list[int],
    workers: list[int],
    n_events: int = 1024,
    flush_interval: float = 1.0,
    repeat: int = 3,
) -> list[dict]:
    """Sweep server counts x worker counts; return a list of result rows."""
    rows = []
    for n in servers:
        msgs = make_round(n, n_events=n_events, seed=0)
        row = {
            "servers": n,
            "events": n * n_events,
            "w1": measure_ingest(msgs, 1, repeat),
        }
        for w in workers:
            if w > 1:
                row[f"thread{w}"] = measure_ingest(msgs, w, repeat, backend="thread")
                row[f"resamp{w}"] = measure_ingest_resample_warm(msgs, w, repeat)
        rows.append(row)
        _print_row(row, workers, flush_interval)
    return rows


def _print_row(row: dict, workers: list[int], flush_interval: float) -> None:
    parts = [f"servers={row['servers']:>5}", f"events={row['events']:>8}"]
    lat1 = row["w1"]
    parts.append(f"w1={lat1*1e3:7.1f} ms ({lat1/flush_interval*100:5.1f}%)")
    for w in workers:
        if w > 1:
            tlat = row[f"thread{w}"]
            rlat = row[f"resamp{w}"]
            parts.append(f"thread{w}={tlat*1e3:7.1f} ms ({lat1/tlat:.1f}x)")
            parts.append(f"resample{w}={rlat*1e3:7.1f} ms ({lat1/rlat:.1f}x)")
    print("  ".join(parts))


def run_grid(
    servers: list[int],
    events: list[int],
    repeat: int = 2,
    n_workers: int = 4,
    include_thread: bool = True,
    flush_interval: float = 1.0,
) -> list[dict]:
    """Full 2D grid: for every (events/server, server count) report the latency
    of each backend — single process, thread fan-out, and process-resample.

    The headline is not the speed-up but whether a round still fits inside the
    flush interval: once ingest outlives the interval that triggered it, rounds
    pile up. Latency is therefore also reported as a fraction of
    *flush_interval*, and `*_keeps_up` says whether that backend is still viable.
    """
    rows = []
    hdr = f"{'events/srv':>10} {'servers':>8} {'single':>16}"
    if include_thread:
        hdr += f" {'thread' + str(n_workers):>10}"
    hdr += f" {'resample' + str(n_workers):>17} {'speedup':>8}"
    print(hdr)
    for ev in events:
        for n in servers:
            msgs = make_round(n, n_events=ev, seed=0)
            single = measure_ingest(msgs, 1, repeat)
            thread = (
                measure_ingest(msgs, n_workers, repeat, backend="thread")
                if include_thread
                else None
            )
            resamp = measure_ingest_resample_warm(msgs, n_workers, repeat)
            row = {
                "events": ev,
                "servers": n,
                "single": single,
                "thread": thread,
                "resample": resamp,
                "single_keeps_up": single <= flush_interval,
                "resample_keeps_up": resamp <= flush_interval,
            }
            rows.append(row)
            # "!" marks a backend that can no longer keep up at this size.
            s_pct = f"{single / flush_interval * 100:.0f}%"
            r_pct = f"{resamp / flush_interval * 100:.0f}%"
            s_flag = " " if row["single_keeps_up"] else "!"
            r_flag = " " if row["resample_keeps_up"] else "!"
            line = f"{ev:>10} {n:>8} {single * 1e3:>8.1f}ms {s_pct:>5}{s_flag}"
            if include_thread:
                line += f" {thread * 1e3:>8.1f}ms"
            line += f" {resamp * 1e3:>9.1f}ms {r_pct:>5}{r_flag} {single / resamp:>7.1f}x"
            print(line)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTIO ingestion scaling benchmark")
    parser.add_argument("--servers", type=int, nargs="+", default=[8, 32, 128, 512, 1024])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--events", type=int, nargs="+", default=[1024])
    parser.add_argument("--flush-interval", type=float, default=1.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--grid", action="store_true", help="2D sweep over every (events x servers) pair"
    )
    args = parser.parse_args(argv)

    if args.grid or len(args.events) > 1:
        print(
            f"Ingest scaling grid (flush_interval={args.flush_interval}s, "
            f"best of {args.repeat}; '!' = cannot keep up)"
        )
        run_grid(
            args.servers,
            args.events,
            args.repeat,
            flush_interval=args.flush_interval,
        )
    else:
        print(
            f"Ingest scaling (events/server={args.events[0]}, "
            f"flush_interval={args.flush_interval}s, best of {args.repeat})"
        )
        run_sweep(
            args.servers, args.workers, args.events[0], args.flush_interval, args.repeat
        )


if __name__ == "__main__":
    main()
