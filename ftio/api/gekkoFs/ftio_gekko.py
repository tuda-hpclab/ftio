"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

# import os
import argparse
import glob

import numpy as np
import plotly.graph_objects as go

from ftio.api.gekkoFs.parse_gekko import parse
from ftio.cli.ftio_core import core
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.multiprocessing.async_process import handle_in_process
from ftio.parse.args import parse_args
from ftio.parse.bandwidth import overlap
from ftio.plot.helper import format_plot
from ftio.plot.units import set_unit
from ftio.prediction.helper import dump_json
from ftio.prediction.parallel_ingest import (
    partition,
    reduce_partials,
    resample_step,
    resolve_workers,
)
from ftio.processing.print_output import display_prediction

CONSOLE = MyConsole()
CONSOLE.set(True)


def _fresh_data_rank() -> dict:
    return {
        "avg_throughput": [],
        "t_end": [],
        "t_start": [],
        "hostname": "",
        "pid": 0,
        "io_type": "",
        "req_size": [],
        "total_bytes": 0,
        "total_iops": 0,
        "t_flush": 0.0,
    }


def _parse_overlap_chunk(chunk_and_io: tuple) -> tuple:
    """Worker body: parse a chunk of messages and overlap it into one partial.

    Returns (b, t, total_bytes, t_flush, ext). Module-level so it is picklable
    for the process pool.
    """
    chunk, io_type = chunk_and_io
    data = _fresh_data_rank()
    ext = ""
    for msg in chunk:
        data, ext = parse(msg, data, io_type=io_type, debug_level=0)
    if data["avg_throughput"]:
        b, t = overlap(data["avg_throughput"], data["t_start"], data["t_end"])
    else:
        b, t = [], []
    return b, t, data["total_bytes"], data["t_flush"], ext


def _parse_overlap_resample_chunk(args: tuple) -> tuple:
    """Like _parse_overlap_chunk but resamples the partial onto a shared grid.

    Returns (resampled_b, total_bytes, t_flush, ext). The resampled vector is
    fixed-length, so it is cheap to ship out of a worker process — this is what
    makes the process backend viable (no big-array IPC).
    """
    chunk, io_type, grid = args
    data = _fresh_data_rank()
    ext = ""
    for msg in chunk:
        data, ext = parse(msg, data, io_type=io_type, debug_level=0)
    if data["avg_throughput"]:
        b, t = overlap(data["avg_throughput"], data["t_start"], data["t_end"])
        rb = resample_step(b, t, grid)
    else:
        rb = np.zeros(len(grid))
    return rb, data["total_bytes"], data["t_flush"], ext


def _trim_grid_padding(b: np.ndarray, grid: np.ndarray) -> tuple[list, list]:
    """Drop the all-zero samples the shared grid pads a round's I/O with.

    The resample grid always spans [0, horizon], but a round only carries the
    I/O that happened inside it, so everything before the first burst is zero.
    Returning the whole grid makes every round start at t=0. The caller
    accumulates rounds by concatenation (``b_app.extend``), so the trace would
    jump backwards in time and the DFT would run on non-monotonic timestamps.
    The serial path returns only the breakpoints it saw; match that.

    One zero sample is kept on each side so the step function still rises from
    and falls back to zero.
    """
    nonzero = np.flatnonzero(b)
    if nonzero.size == 0:
        return [], []
    lo = max(int(nonzero[0]) - 1, 0)
    hi = min(int(nonzero[-1]) + 2, len(b))
    return list(b[lo:hi]), list(grid[lo:hi])


def _peek_horizon(msgs: list) -> float:
    """Cheap upper bound on the time window from each message's flush_t (field 0)."""
    import msgpack

    hi = 0.0
    for m in msgs:
        if isinstance(m, bytes):
            up = msgpack.Unpacker()
            up.feed(m)
            hi = max(hi, float(next(iter(up))) * 1e-6)  # first object = flush_t (us)
    return hi


def ingest_app_bandwidth(
    files_or_msgs: list,
    io_type: str,
    n_workers: int = 1,
    backend: str = "thread",
    resample_hz: float = 100.0,
) -> tuple:
    """Parse messages and overlap them into the application-level bandwidth.

    n_workers == 1 is the original serial path. n_workers > 1 fans the messages
    out and folds the per-worker partials with the same (associative) overlap,
    so the (b, t) result is identical. ``backend`` chooses how: ``thread``
    (default) shares memory — no IPC — and the numba overlap is compiled
    ``nogil`` so it runs truly parallel; ``process`` avoids the GIL but is
    IPC-bound for this workload. total_bytes is summed across servers and
    t_flush is the max. Returns (b, t, total_bytes, t_flush, ext).
    """
    n = resolve_workers(n_workers) if n_workers > 1 else 1
    if n == 1 or len(files_or_msgs) <= 1:
        return _parse_overlap_chunk((files_or_msgs, io_type))

    chunks = partition(files_or_msgs, n)

    # process-resample: each worker returns a small fixed-length resampled
    # vector on a shared grid, so nothing large crosses the process boundary.
    # _peek_horizon only reads the flush_t header of *packed* messages; for file
    # inputs it yields 0 and the grid would collapse to a single point, silently
    # returning an empty series. Fall back to the exact path in that case.
    horizon = _peek_horizon(files_or_msgs) if backend == "process-resample" else 0.0
    if backend == "process-resample" and horizon > 0.0:
        from multiprocessing import Pool

        grid = np.arange(0.0, horizon + 1.0 / resample_hz, 1.0 / resample_hz)
        tasks = [(c, io_type, grid) for c in chunks]
        with Pool(processes=len(chunks)) as pool:
            results = pool.map(_parse_overlap_resample_chunk, tasks)
        rb = np.sum([r[0] for r in results], axis=0)
        total_bytes = sum(r[1] for r in results)
        t_flush = max((r[2] for r in results), default=0.0)
        ext = next((r[3] for r in results if r[3]), "")
        b, t = _trim_grid_padding(rb, grid)
        return b, t, total_bytes, t_flush, ext

    tasks = [(c, io_type) for c in chunks]
    if backend == "process":
        from multiprocessing import Pool

        with Pool(processes=len(chunks)) as pool:
            results = pool.map(_parse_overlap_chunk, tasks)
    else:
        from multiprocessing.pool import ThreadPool

        with ThreadPool(len(chunks)) as pool:
            results = pool.map(_parse_overlap_chunk, tasks)
    b, t = reduce_partials([(r[0], r[1]) for r in results])
    total_bytes = sum(r[2] for r in results)
    t_flush = max((r[3] for r in results), default=0.0)
    ext = next((r[4] for r in results if r[4]), "")
    return b, t, total_bytes, t_flush, ext


def run(
    files_or_msgs: list, argv=None, b_app=None, t_app=None
) -> tuple[Prediction, argparse.Namespace, float]:  # "0.01"] ):
    """Executes ftio on a list of files_or_msgs.

    Args:
        files_or_msgs (list): list with msgpack msg or json files
        argv: command line arguments for ftio
        b_app: app level bandwidth
        t_app: app level timestamps
    """

    # parse args
    if t_app is None:
        t_app = []
    if b_app is None:
        b_app = []
    if argv is None:
        argv = ["-e", "plotly", "-f", "100"]
    args = parse_args(argv, "ftio")
    ranks = len(files_or_msgs)

    # Set up data
    data_rank = _fresh_data_rank()

    # 1) parse the messages and overlap them into the app-level bandwidth. With
    #    --ingest-workers > 1 this is fanned out across processes and the
    #    per-worker partials are folded back (identical result).
    n_workers = getattr(args, "ingest_workers", 1)
    backend = getattr(args, "ingest_backend", "process-resample")
    resample_hz = getattr(args, "freq", 100.0) or 100.0
    b, t, total_bytes, t_flush, ext = ingest_app_bandwidth(
        files_or_msgs, args.mode[0], n_workers, backend, resample_hz
    )
    data_rank["total_bytes"] = total_bytes
    data_rank["t_flush"] = t_flush

    # 2) exit if no new data (retry as read to flag "read data ignored").
    #    "r" matches GekkoFS's io_type; args.mode[0] is likewise "w"/"r".
    if not b:
        rb, *_ = ingest_app_bandwidth(files_or_msgs, "r", 1)
        if rb:
            CONSOLE.print("[red]Read data passed -- ignoring [/]")
        else:
            CONSOLE.print("[red]Terminating prediction (no data passed) [/]")
        exit(0)

    # Debug
    dt = np.diff(t)  # time intervals
    bytes_total = np.sum(b[:-1] * dt)  # total bytes
    print(
        f"Total transferred in this burst: {bytes_total:.0f} bytes ({bytes_total/1e9:.3f} GB)"
    )

    # # 5) Extend for ZMQ
    if "ZMQ" in ext.upper():
        # extend data
        b_app.extend(b)
        t_app.extend(t)
        b = np.array(b_app[:])
        t = np.array(t_app[:])
        # print(f"App Bandwidth: {b_app}")
        # print(f"App Time: {t_app}")
        # 5) overlap with app bandwdith so far
        # b, t = overlap_two_series(b_app[:], t_app[:], b, t)
        # t_app = t.tolist()
        # b_app = b.tolist()
        # print(f"App Bandwidth: {b_app}")
        # print(f"App Time: {t_app}")

    else:
        b = np.array(list(b))
        t = np.array(list(t))

    # save the bandwidth
    process = handle_in_process(
        dump_json,
        args=(b, t),
    )

    # 6) plot to check:
    if any(x in args.engine for x in ["mat", "plot"]):
        fig = go.Figure()
        unit, order = set_unit(b)
        # fig.add_trace(go.Scatter(x=t, y=b * order, name="App Bandwidth",mode='lines+markers'))
        fig.add_trace(
            go.Scatter(x=t, y=b * order, name="App Bandwidth", line={"shape": "hv"})
        )
        fig.update_layout(xaxis_title="Time (s)", yaxis_title=f"Bandwidth ({unit})")
        fig = format_plot(fig)
        fig.show()

    # 7) set up data
    data = {
        "time": t,
        "bandwidth": b,
        "total_bytes": data_rank["total_bytes"],
        "ranks": ranks,
    }

    # 8) perform prediction
    prediction, analysis_figures = core(data, args)

    # 9) plot and print info
    # if args.verbose:
    display_prediction(args, prediction)

    analysis_figures.show()
    process.join()

    return prediction, args, data_rank["t_flush"]


if __name__ == "__main__":
    # absolute path to search all text files_or_msgs inside a specific folder
    # path=r'/d/github/FTIO/examples/API/gekkoFs/JSON/*.json'
    # path = r"/d/github/FTIO/examples/API/gekkoFs/MSGPACK/write*.msgpack"
    path = r"/d/Downloads/metrics/metrics/write_*.msgpack"
    matched_files_or_msgs = glob.glob(path)
    run(matched_files_or_msgs)
