"""
Decode a FlexMPI-monitor ZMQ message (``--zmq_format flexmpi``) into FTIO's
internal structure.

Each msgpack map is one rank / one iteration. Fields used: ``size`` (bytes
sent during this I/O), ``iotime`` (interval over which they were sent, s) and
``rtime`` (current time instance, s). The sample is ``size / iotime`` over
``[rtime - iotime, rtime]``. Other fields (``rank``, ``iter``, ``flops``,
``mflops``, ``ptime``, ``ctime``) are ignored.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Sep 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import msgpack

from ftio.parse.input_template import init_data


def extract(msg: bytes, args) -> tuple[dict, int]:
    """Decode one FlexMPI message into ``(data, ranks)``."""
    mode, io_data, io_time = init_data(args)

    raw = msgpack.unpackb(msg, raw=False)
    if not isinstance(raw, dict):
        raise ValueError(
            f"FlexMPI message must be a msgpack map, got {type(raw).__name__}"
        )

    size = float(raw.get("size", 0.0) or 0.0)
    iotime = float(raw.get("iotime", 0.0) or 0.0)
    rtime = float(raw.get("rtime", 0.0) or 0.0)

    b = size / iotime if iotime > 0 else 0.0
    te = rtime
    ts = rtime - iotime

    io_data["total_bytes"] = int(size)
    io_data["bandwidth"]["b_rank_avr"] = [b]
    io_data["bandwidth"]["t_rank_s"] = [ts]
    io_data["bandwidth"]["t_rank_e"] = [te]

    return {mode: io_data, "io_time": io_time}, 0
