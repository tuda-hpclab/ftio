"""
Decode a single FlexMPI-monitor ZMQ message into the internal FTIO

Wire format (``--zmq_format flexmpi``)
-------------------------------------
One msgpack-encoded map per rank per monitored iteration, e.g.::

    {
        "rank":   0,        # MPI rank id of the sender
        "size":   64,       # MPI communicator size == number of ranks
        "iter":   1234,     # iteration counter
        "flops":  1.2e9,    # floating point ops in this iteration
        "mflops": 1200.0,   # MFLOP/s in this iteration
        "rtime":  53.7,     # cumulative wall-clock run time [s] at this point
        "ptime":  0.9,      # compute time of this iteration [s]
        "ctime":  0.3,      # communication time of this iteration [s]
        "iotime": 0.4,      # I/O time of this iteration [s]
    }
    
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Sep 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import time

import msgpack
from rich.console import Console

from ftio.parse.input_template import init_data

# Fields we read from a FlexMPI message; every one is optional and defaults to
# 0.0 so a monitor that omits some of them still parses.
_FIELDS = (
    "rank",
    "size",
    "iter",
    "flops",
    "mflops",
    "rtime",
    "ptime",
    "ctime",
    "iotime",
)


def extract(msg: bytes, args) -> tuple[dict, int]:
    """Decode FlexMPI message into ``(data, ranks)``.

    Args:
        msg: raw msgpack bytes from the ZMQ socket.
        args: parsed CLI args (passed straight through to ``init_data``).

    Returns:
        tuple ``(data, ranks)`` -- ``ranks`` comes from the ``size`` field
        (MPI communicator size), or ``0`` when absent.
    """
    start = time.time()
    mode, io_data, io_time = init_data(args)

    fields = _unpack(msg)
    ranks = int(fields.get("size", 0) or 0)

    b = _io_signal(fields)
    ts, te = _interval(fields)

    io_data["number_of_ranks"] = ranks
    # No bytes field on the wire 
    io_data["total_bytes"] = int(fields.get("bytes", 0) or 0)

    bw = io_data["bandwidth"]
    bw["b_rank_avr"] = [b]
    bw["t_rank_s"] = [ts]
    bw["t_rank_e"] = [te]

    Console().print(
        f"[cyan]FlexMPI msg[/] rank={fields.get('rank')} iter={fields.get('iter')} "
        f"-> b={b:.3g} [{ts:.3f}, {te:.3f}] s  "
        f"([cyan]parsed in[/] {time.time() - start:.4f} s)"
    )

    data = {
        f"{mode}": io_data,
        "io_time": io_time,
    }
    return data, ranks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unpack(msg: bytes) -> dict:
    """msgpack-unpack a FlexMPI message and keep only the known fields."""
    raw = msgpack.unpackb(msg, raw=False)
    if not isinstance(raw, dict):
        raise ValueError(
            f"FlexMPI message must be a msgpack map, got {type(raw).__name__}"
        )
    return {k: raw[k] for k in _FIELDS if k in raw}


def _io_signal(fields: dict) -> float:
    """Return the scalar the frequency analysis should treat as "bandwidth".

    -------------------------------------------------------------------------
    PLACEHOLDER -- adapt to the real FlexMPI semantics.
    -------------------------------------------------------------------------
    """
    iotime = float(fields.get("iotime", 0.0) or 0.0)
    return iotime

    # --- option 2 (uncomment to use I/O fraction instead) -------------------
    # ptime = float(fields.get("ptime", 0.0) or 0.0)
    # ctime = float(fields.get("ctime", 0.0) or 0.0)
    # total = ptime + ctime + iotime
    # return iotime / total if total > 0 else 0.0


def _interval(fields: dict) -> tuple[float, float]:
    """Return ``(t_start, t_end)`` in seconds for this iteration's I/O sample.

    -------------------------------------------------------------------------
    PLACEHOLDER -- adapt to the real FlexMPI semantics.
    -------------------------------------------------------------------------
    """
    rtime = float(fields.get("rtime", 0.0) or 0.0)
    iotime = float(fields.get("iotime", 0.0) or 0.0)
    ts = max(rtime - iotime, 0.0)
    te = rtime if rtime > ts else ts  # tolerate iotime == 0 (zero-width sample)
    return ts, te
