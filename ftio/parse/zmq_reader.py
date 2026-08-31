"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import time

import msgpack
from rich.console import Console

from ftio.parse.input_template import init_data


def extract(msgs, args: list) -> tuple[dict, int]:
    # init
    start = time.time()
    mode, io_data, io_time = init_data(args)

    # unpack data
    unpacked_data = msgpack.unpackb(msgs)

    # Wire shape by key presence: {b, ts, te?, ranks?} rank-level (overlap runs
    # downstream), or {b, t, ranks?} application-level (already overlapped).
    ranks = unpacked_data.get("ranks", 0)
    bw = io_data["bandwidth"]

    if "b" in unpacked_data and "ts" in unpacked_data:
        ts = unpacked_data["ts"]
        te = unpacked_data.get("te")
        if te is None:  # each interval ends where the next starts
            te = list(ts[1:]) + [ts[-1]] if ts else []
        bw["b_rank_avr"] = unpacked_data["b"]
        bw["t_rank_s"] = ts
        bw["t_rank_e"] = te
    elif "b" in unpacked_data and "t" in unpacked_data:
        # set both overlap keys so Bandwidth doesn't overlap([], [], []) the empty
        # rank arrays and wipe t_overlap; keep the rank keys so mixed streams merge
        bw["b_overlap_avr"] = bw["b_overlap_sum"] = unpacked_data["b"]
        bw["t_overlap"] = unpacked_data["t"]
    else:
        raise ValueError(
            f"unrecognised ZMQ message shape; got keys {sorted(unpacked_data)}"
        )

    console = Console()
    console.print(f"[cyan]Elapsed time:[/] {time.time()-start:.3f} s")
    # io_time[f"delta_t_{kind}"] = 0

    # pack everything
    data = {
        f"{mode}": io_data,
        "io_time": io_time,
    }

    return data, ranks
