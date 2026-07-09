"""
Stress test — what happens when servers push faster than FTIO can ingest.

One flush round takes L seconds to turn into the application-level bandwidth,
where L depends on both the server count and the events per message. If GekkoFS
posts (flushes) at frequency f = 1 / flush_interval and f > 1/L, a single
ingester cannot keep up and a backlog grows. This reports the maximum
sustainable GekkoFS posting frequency (1/L) for the single-process path and the
process-resample fan-out. GLASS currently posts every ~5 s (0.2 Hz).

Run:
    python -m ftio.api.gekkoFs.jit.experiments.exp_d_stress --servers 512
    python -m ftio.api.gekkoFs.jit.experiments.exp_d_stress --servers 1024 --posting-hz 0.2 0.5 1 2

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
import os

from ftio.api.gekkoFs.jit.experiments.exp_a_scaling import (
    measure_ingest,
    measure_ingest_resample_warm,
)
from ftio.api.gekkoFs.jit.experiments.synthetic_messages import make_round


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def measure_latencies(
    n_servers: int, n_events: int, workers: int, repeat: int = 5
) -> tuple:
    """Per-prediction ingest latency (s): single-process vs process-resample."""
    msgs = make_round(n_servers, n_events=n_events, io_type="w", seed=0)
    single_lat = measure_ingest(msgs, 1, repeat)
    resamp_lat = measure_ingest_resample_warm(msgs, workers, repeat)
    return single_lat, resamp_lat


def stress(
    n_servers: int,
    posting_hz: list[float],
    n_events: int = 1024,
    workers: int = 4,
    cores: int | None = None,
) -> list[dict]:
    """Comprehensive sweep of the GekkoFS posting frequency vs core occupancy.

    Posting frequency f = 1 / flush_interval (GLASS default ~0.2 Hz / 5 s). Each
    prediction runs as its OWN process(es) and several run concurrently, so the
    limit is core occupancy, not a serial queue:

      * single-process-per-prediction: 1 core for L1 s -> in-flight cores = f*L1,
        throughput ceiling C/L1 predictions/s.
      * process-resample fan-out: `workers` cores for Lr s -> in-flight cores =
        f*Lr*workers, throughput ceiling C/(workers*Lr).

    Resample has the LOWER latency (fresher prediction) but uses more cores per
    prediction, so at high f the single-process path sustains a HIGHER frequency.
    'keeps up' means in-flight cores <= C; beyond that, predictions pile up
    unboundedly (the case task #7's bounded pool caps).
    """
    cores = cores or _cores()
    single_l, resamp_l = measure_latencies(n_servers, n_events, workers)
    ceil_single = cores / single_l
    ceil_resamp = cores / (workers * resamp_l)
    print(
        f"servers={n_servers} events/msg={n_events} cores={cores} "
        f"workers/pred={workers}"
    )
    print(
        f"  latency/pred: single={single_l*1e3:.0f} ms, resample={resamp_l*1e3:.0f} ms "
        f"({single_l/resamp_l:.1f}x fresher)"
    )
    print(
        f"  max posting freq: single={ceil_single:.2f} Hz, "
        f"resample={ceil_resamp:.2f} Hz"
    )
    print(
        f"  {'posting':>8} {'flush':>7} {'single: cores(keep)':>22} "
        f"{'resample: cores(keep)':>24}"
    )
    rows = []
    for f in posting_hz:
        c_single = f * single_l
        c_resamp = f * resamp_l * workers
        row = {
            "posting_hz": f,
            "interval_s": 1.0 / f if f > 0 else float("inf"),
            "single_cores": c_single,
            "single_keeps_up": c_single <= cores,
            "resample_cores": c_resamp,
            "resample_keeps_up": c_resamp <= cores,
        }
        rows.append(row)
        sk = "ok" if row["single_keeps_up"] else "PILEUP"
        rk = "ok" if row["resample_keeps_up"] else "PILEUP"
        print(
            f"  {f:6.2f}Hz {row['interval_s']:6.1f}s "
            f"{c_single:14.1f} ({sk:6s}) {c_resamp:16.1f} ({rk:6s})"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTIO ingestion stress test")
    parser.add_argument("--servers", type=int, default=512)
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--posting-hz",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 1, 2, 5],
        help="GekkoFS posting frequencies to test (Hz); "
        "GLASS default is ~0.2 Hz (5 s flush)",
    )
    args = parser.parse_args(argv)
    stress(args.servers, args.posting_hz, args.events, args.workers)


if __name__ == "__main__":
    main()
