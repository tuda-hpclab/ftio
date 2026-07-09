"""
Experiment B — FTIO misprediction / irregular-workload sensitivity.

Answers the reviewers' robustness concern: how does FTIO's prediction degrade
as the workload becomes less regular? We drive a synthetic multi-server round
with a known ground-truth period and sweep a timing-jitter knob that perturbs
the burst centers. For each jitter level we run the real FTIO prediction and
report the detected frequency error and the confidence.

At jitter 0 the aggregate is cleanly periodic (fundamental 1/period); as jitter
grows the confidence should fall and the frequency error should widen — a
graceful degradation rather than a cliff.

Run:
    python -m ftio.api.gekkoFs.jit.experiments.exp_b_misprediction
    python -m ftio.api.gekkoFs.jit.experiments.exp_b_misprediction --jitter 0 0.1 0.3 0.5

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

import numpy as np

from ftio.api.gekkoFs.ftio_gekko import ingest_app_bandwidth
from ftio.api.gekkoFs.jit.experiments.synthetic_messages import make_round
from ftio.cli.ftio_core import core
from ftio.parse.args import parse_args
from ftio.prediction.helper import get_dominant


def evaluate(
    jitter: float,
    period: float = 2.0,
    n_servers: int = 16,
    n_events: int = 512,
    horizon: float = 20.0,
    sampling_hz: float = 50.0,
    seed: int = 0,
) -> dict:
    """Run one prediction at a given jitter and return detected freq / conf."""
    msgs = make_round(
        n_servers,
        n_events=n_events,
        period=period,
        horizon=horizon,
        jitter=jitter,
        io_type="w",
        seed=seed,
    )
    b, t, total_bytes, _flush, _ext = ingest_app_bandwidth(msgs, "w", n_workers=1)
    data = {
        "time": np.array(t),
        "bandwidth": np.array(b),
        "total_bytes": total_bytes,
        "ranks": n_servers,
    }
    args = parse_args(["-e", "no", "-f", str(sampling_hz)], "ftio")
    prediction, _ = core(data, args)
    freq = get_dominant(prediction)
    conf = float(np.max(prediction.conf)) if len(prediction.conf) else 0.0
    gt = 1.0 / period
    rel_err = abs(freq - gt) / gt if np.isfinite(freq) and gt > 0 else float("nan")
    return {"jitter": jitter, "gt": gt, "freq": freq, "conf": conf, "rel_err": rel_err}


def run_sweep(jitters: list[float], **kwargs) -> list[dict]:
    rows = [evaluate(j, **kwargs) for j in jitters]
    for r in rows:
        print(
            f"jitter={r['jitter']:.2f}  gt={r['gt']:.3f}Hz  detected={r['freq']:.3f}Hz  "
            f"rel_err={r['rel_err']*100:5.1f}%  conf={r['conf']:.2f}"
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTIO misprediction sensitivity")
    parser.add_argument(
        "--jitter", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--servers", type=int, default=16)
    parser.add_argument("--events", type=int, default=512)
    args = parser.parse_args(argv)

    print(
        f"Misprediction sweep (period={args.period}s -> gt={1/args.period:.3f}Hz, "
        f"servers={args.servers})"
    )
    run_sweep(
        args.jitter, period=args.period, n_servers=args.servers, n_events=args.events
    )


if __name__ == "__main__":
    main()
