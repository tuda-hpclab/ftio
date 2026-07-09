"""
Synthetic GekkoFS server messages for the reviewer-response experiments.

Each message is the 8-field msgpack frame a GekkoFS server pushes over ZMQ:
[flush_t, hostname, pid, start_t[us], end_t[us], req_size, total_iops,
total_bytes]. A periodic burst pattern is embedded so FTIO detects a known
frequency, which lets the misprediction study (Experiment B) inject controlled
irregularity against a known ground truth.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import msgpack
import numpy as np


def make_server_message(
    server_id: int,
    n_events: int = 1024,
    period: float = 1.0,
    duty: float = 0.5,
    req_bytes: int = 524288,
    horizon: float = 20.0,
    jitter: float = 0.0,
    io_type: str = "w",
    rng: np.random.Generator | None = None,
) -> bytes:
    """Build one msgpack message matching the current GekkoFS wire format.

    The current GekkoFS packs 9 top-level fields (msgpack_util.hpp):
    flush_t, hostname, pid, io_type, start_t[us], end_t[us], req_size,
    total_iops, total_bytes. io_type is "w"/"r".

    Args:
        server_id: used to derive the pid and a small phase offset.
        n_events: number of I/O intervals in the message.
        period: burst period in seconds (ground-truth 1/period frequency).
        duty: fraction of each period that is active (I/O happening).
        req_bytes: request size per event.
        horizon: total time span the events cover, in seconds.
        jitter: relative timing jitter in [0, 1] to perturb periodicity
            (0 = perfectly periodic; used by the misprediction study).
        rng: optional numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(server_id)

    # one tight burst per period so the fundamental (1/period) dominates the
    # aggregate signal. Events are clustered around each burst center; jitter
    # perturbs the centers to degrade periodicity for the misprediction study.
    n_bursts = max(1, int(horizon / period))
    centers = np.arange(n_bursts) * period + 0.5 * duty * period
    if jitter > 0:
        centers = centers + rng.normal(0.0, jitter * period, size=n_bursts)
    burst_of = rng.integers(0, n_bursts, size=n_events)
    spread = 0.25 * duty * period  # cluster width inside the active window
    starts = centers[burst_of] + rng.normal(0.0, spread, size=n_events)
    starts = np.clip(np.sort(starts), 0.0, None)

    durations = np.full(n_events, duty * period / max(1, n_events // 8))
    ends = starts + durations

    start_us = (starts * 1e6).astype(np.int64).tolist()
    end_us = (ends * 1e6).astype(np.int64).tolist()
    req_size = [int(req_bytes)] * n_events

    flush_t = int((horizon + server_id * 1e-3) * 1e6)
    # order matches GekkoFS msgpack_util.hpp pack(): flush_t, hostname, pid,
    # io_type, start_t, end_t, req_size, total_iops, total_bytes
    fields = [
        flush_t,
        "electric",
        100000 + server_id,
        io_type,
        start_us,
        end_us,
        req_size,
        n_events,
        req_bytes * n_events,
    ]
    # GekkoFS packs each field as a separate top-level object (not one array)
    return b"".join(msgpack.packb(f) for f in fields)


def make_round(n_servers: int, **kwargs) -> list[bytes]:
    """Build one flush round: one message per server."""
    rng = np.random.default_rng(kwargs.pop("seed", 0))
    return [
        make_server_message(
            sid, rng=np.random.default_rng(rng.integers(1 << 30)), **kwargs
        )
        for sid in range(n_servers)
    ]
