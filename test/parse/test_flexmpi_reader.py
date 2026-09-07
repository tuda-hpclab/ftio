"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Sep 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import msgpack
import pytest

from ftio.parse.flexmpi_reader import extract

MODE = "write_sync"


def _bandwidth(data: dict) -> dict:
    return data[MODE]["bandwidth"]


def _msg(**overrides) -> bytes:
    fields = {
        "rank": 0,
        "size": 8,
        "iter": 1,
        "flops": 1.0e9,
        "mflops": 1000.0,
        "rtime": 10.0,
        "ptime": 0.8,
        "ctime": 0.1,
        "iotime": 0.4,
    }
    fields.update(overrides)
    return msgpack.packb(fields)


def test_single_message_is_one_rank_level_sample():
    data, ranks = extract(_msg(rtime=10.0, iotime=0.4), [])

    assert ranks == 8  # from the "size" field
    bw = _bandwidth(data)
    assert bw["b_rank_avr"] == [0.4]  # placeholder signal == iotime
    assert bw["t_rank_s"] == [pytest.approx(9.6)]  # rtime - iotime
    assert bw["t_rank_e"] == [pytest.approx(10.0)]  # rtime


def test_size_absent_gives_zero_ranks():
    raw = {"rank": 0, "iter": 1, "rtime": 5.0, "iotime": 0.0}
    _, ranks = extract(msgpack.packb(raw), [])
    assert ranks == 0


def test_zero_iotime_is_a_zero_width_sample():
    data, _ = extract(_msg(rtime=7.0, iotime=0.0), [])
    bw = _bandwidth(data)
    assert bw["b_rank_avr"] == [0.0]
    assert bw["t_rank_s"] == [pytest.approx(7.0)]
    assert bw["t_rank_e"] == [pytest.approx(7.0)]


def test_non_map_payload_raises_valueerror():
    with pytest.raises(ValueError):
        extract(msgpack.packb([1, 2, 3]), [])


def test_end_to_end_prediction():
    """A batch of FlexMPI messages survives Scales -> Simrun -> Bandwidth."""
    from ftio.cli.ftio_core import main

    # square wave: I/O every other iteration, iterations ~1 s apart
    msgs = []
    for i in range(40):
        iotime = 0.5 if i % 2 == 0 else 0.0
        msgs.append(_msg(rank=0, size=4, iter=i, rtime=float(i) + iotime, iotime=iotime))

    preds, _ = main(
        ["ftio", "--zmq", "--zmq_format", "flexmpi", "-e", "no", "-f", "10"], msgs
    )
    assert preds and preds[0].t_end > preds[0].t_start
    assert preds[0].ranks == 4
