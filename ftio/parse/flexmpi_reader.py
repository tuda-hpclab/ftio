"""
Decode a single FlexMPI-monitor ZMQ message into the internal FTIO data
structure (the same shape ``ftio.parse.zmq_reader.extract`` produces for the
``direct`` wire format).

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

The reference listener that motivated this format:
https://github.com/tuda-parallel/FTIO  (see the FlexMPI monitor SUB example)

.. important::

   FlexMPI messages carry **no bandwidth / bytes field**, so unlike the
   ``direct`` format there is nothing to feed straight into the frequency
   analysis.  :func:`_io_signal` and :func:`_interval` below are deliberate
   **placeholders** -- they build a plausible periodic signal out of the
   timing fields so the pipeline runs end to end.  Adapt them once the exact
   semantics of the FlexMPI fields for your setup are known (in particular
   whether any field encodes I/O volume in bytes).

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
    """Decode one FlexMPI message into ``(data, ranks)``.

    Mirrors :func:`ftio.parse.zmq_reader.extract`: the returned ``data`` is a
    ``{mode: io_data, "io_time": io_time}`` dict shaped like one JSONL part, so
    ``ParseZmq.to_simrun`` can hand a batch of them to ``Simrun`` via the
    existing multi-part merge path.

    Each message is one rank / one iteration, so it contributes a single
    ``(b, ts, te)`` sample.  A drain cycle delivers many such messages; the
    merge step concatenates them into the per-rank interval arrays that the
    overlap step downstream turns into an application-level bandwidth.

    .. note::

       With more than one rank, samples from different ranks are simply
       concatenated (and the shared timeline is then non-monotonic).  The
       overlap step tolerates that, but the cleaner fix -- once you adapt
       this module -- is to group the batch by ``iter`` and emit one
       aggregated sample per iteration (sum ``iotime`` across ranks, or take
       the max, depending on whether I/O is collective).

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
    # No bytes field on the wire -- leave total_bytes at 0 unless a future
    # revision of the FlexMPI monitor adds one (then read it here).
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
    The transform only needs a signal whose *amplitude rises and falls with
    the I/O phases*; the physical unit is irrelevant to periodicity detection.

    Current choice: ``iotime`` -- seconds this rank spent in I/O during the
    iteration.  It is 0 on compute-only iterations and jumps up on the
    periodic checkpoint/output iterations, which is exactly the square-ish
    wave FTIO looks for.

    Better options once the fields are pinned down, roughly in order of
    preference:

      1. **True bandwidth** ``bytes_written / iotime`` -- if the monitor is
         extended to report I/O volume (add a ``"bytes"`` field and use it
         here).  This is the only option that yields a physically meaningful
         MiB/s figure downstream.

      2. **I/O fraction** ``iotime / (ptime + ctime + iotime)`` -- normalises
         out variable iteration length; robust when iterations are not
         uniform in wall-clock duration.

      3. **iotime** (current) -- simplest; fine when iterations are roughly
         equal length.

    Do NOT use ``flops`` / ``mflops`` here -- those track the *compute* phase,
    which is the inverse of the I/O phase and would make FTIO report the
    compute periodicity instead.
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
    Assumes ``rtime`` is the *cumulative* wall-clock run time at the end of
    the iteration, so the I/O of this iteration occupies the last ``iotime``
    seconds before it:  ``ts = rtime - iotime``, ``te = rtime``.

    This reader is stateless (one call per message), so it cannot accumulate
    its own clock -- it relies on ``rtime`` being monotonic across a rank's
    messages.  If ``rtime`` turns out to be *per-iteration* rather than
    cumulative, replace this with a running sum kept by the caller, or switch
    the time base to ``iter`` (uniform spacing: ``ts = iter``, ``te = iter+1``)
    and let the resampler handle the rest.
    """
    rtime = float(fields.get("rtime", 0.0) or 0.0)
    iotime = float(fields.get("iotime", 0.0) or 0.0)
    ts = max(rtime - iotime, 0.0)
    te = rtime if rtime > ts else ts  # tolerate iotime == 0 (zero-width sample)
    return ts, te
