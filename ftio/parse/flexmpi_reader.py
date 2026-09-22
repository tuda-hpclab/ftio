"""
Decode a FlexMPI-monitor ZMQ message (``--zmq_format flexmpi``) into FTIO's
internal structure.

Each msgpack map is one rank / one iteration. Fields used: ``iobytes`` (bytes
sent during this I/O), ``iotime`` (interval over which they were sent, s) and
``rtime`` (current time instance, s). The sample is ``iobytes / iotime`` over
``[rtime - iotime, rtime]``. Other fields (``rank``, ``iter``, ``size``,
``flops``, ``mflops``, ``ptime``, ``ctime``) are ignored.

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

    # avialble fields
    # print(
    #     f"rank={data['rank']} "
    #     f"size={data['size']} "
    #     f"iter={data['iter']} "
    #     f"flops={data['flops']} "
    #     f"mflops={data['mflops']} "
    #     f"rtime={data['rtime']} "
    #     f"ptime={data['ptime']} "
    #     f"ctime={data['ctime']} "
    #     f"iotime={data['iotime']}"
    #     f"itertimeapp={data['itertimeapp']} "
    #     f"itertimewall={data['itertimewall']} "
    #     f"iobytes={data['iobytes']} "
    #     f"iobandwidth={data['iobandwidth']} "
    #     )

    ranks = int(raw.get("rank", 0) or 0)
    iobytes = float(raw.get("iobytes", 0.0) or 0.0)

    io_time_s = float(raw.get("iotime", 0.0) or 0.0)
    r_time = float(raw.get("rtime", 0.0) or 0.0)

    b = iobytes / io_time_s if io_time_s > 0 else 0.0
    te = r_time
    ts = r_time - io_time_s

    io_data["total_bytes"] = iobytes
    io_data["number_of_ranks"] = ranks
    io_data["bandwidth"]["b_rank_avr"] = [b]
    io_data["bandwidth"]["t_rank_s"] = [ts]
    io_data["bandwidth"]["t_rank_e"] = [te]

    io_time["delta_t_agg_io"] = io_time_s
    io_time["delta_t_agg"] = io_time_s

    return {mode: io_data, "io_time": io_time}, 0
