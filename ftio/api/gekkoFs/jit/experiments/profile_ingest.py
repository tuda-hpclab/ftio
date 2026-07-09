"""
Profile the FTIO ingest to decide whether nogil threads could help.

Breaks one flush round into stages (msgpack decode, field/bandwidth work,
overlap) and separately measures how each stage scales in a *thread* pool. A
stage only benefits from threads if it releases the GIL: if thread-scaling gives
~Nx it is GIL-free (thread-friendly), if ~1x it is GIL-bound.

Run:
    python -m ftio.api.gekkoFs.jit.experiments.profile_ingest --servers 512

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
from multiprocessing.pool import ThreadPool

import msgpack
import numpy as np

from ftio.api.gekkoFs.jit.experiments.synthetic_messages import make_round
from ftio.parse.bandwidth import overlap


def _decode(msg: bytes) -> list:
    up = msgpack.Unpacker()
    up.feed(msg)
    return list(up)


def _bandwidth(items: list) -> tuple:
    """Field extraction + per-message bandwidth (the numpy part of parse)."""
    start = np.array(items[4], dtype=float) * 1e-6
    end = np.array(items[5], dtype=float) * 1e-6
    req = np.array(items[6], dtype=float)
    dur = end - start
    dur[dur == 0] = 1e-6
    b = req / dur
    return b, start, end


def _best(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def profile(n_servers: int, n_events: int = 1024, repeat: int = 5) -> None:
    msgs = make_round(n_servers, n_events=n_events, io_type="w", seed=0)

    # --- stage breakdown (serial) ---
    t_decode = _best(lambda: [_decode(m) for m in msgs], repeat)
    decoded = [_decode(m) for m in msgs]
    t_bw = _best(lambda: [_bandwidth(it) for it in decoded], repeat)
    bws = [_bandwidth(it) for it in decoded]
    t_overlap = _best(lambda: [overlap(b, s, e) for (b, s, e) in bws], repeat)
    total = t_decode + t_bw + t_overlap

    print(f"\n=== {n_servers} servers x {n_events} events ===")
    print(f"  msgpack decode : {t_decode*1e3:8.1f} ms  ({t_decode/total*100:4.1f}%)")
    print(f"  field+bandwidth: {t_bw*1e3:8.1f} ms  ({t_bw/total*100:4.1f}%)")
    print(f"  overlap        : {t_overlap*1e3:8.1f} ms  ({t_overlap/total*100:4.1f}%)")
    print(f"  total (serial) : {total*1e3:8.1f} ms")

    # --- thread-scaling per stage (GIL test): 4 threads vs serial ---
    n_thr = 4
    with ThreadPool(n_thr) as pool:
        t_decode_thr = _best(lambda: pool.map(_decode, msgs), repeat)
        t_overlap_thr = _best(lambda: pool.map(lambda x: overlap(*x), bws), repeat)
    print(
        f"  [threads x{n_thr}] decode : {t_decode_thr*1e3:7.1f} ms "
        f"-> {t_decode/t_decode_thr:.2f}x  ({'GIL-free' if t_decode/t_decode_thr>1.8 else 'GIL-bound'})"
    )
    print(
        f"  [threads x{n_thr}] overlap: {t_overlap_thr*1e3:7.1f} ms "
        f"-> {t_overlap/t_overlap_thr:.2f}x  ({'GIL-free' if t_overlap/t_overlap_thr>1.8 else 'GIL-bound'})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile FTIO ingest stages")
    parser.add_argument("--servers", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    for n in args.servers:
        profile(n, args.events, args.repeat)


if __name__ == "__main__":
    main()
