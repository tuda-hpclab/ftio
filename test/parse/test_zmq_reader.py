"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import msgpack
import pytest

from ftio.parse.zmq_reader import extract

MODE = "write_sync"


def _bandwidth(data: dict) -> dict:
    return data[MODE]["bandwidth"]


def test_rank_level_with_end_times():
    """The original wire shape: per-rank intervals, overlap runs downstream."""
    msg = msgpack.packb({"ranks": 4, "b": [3.0, 0.0], "ts": [1.0, 2.0], "te": [5.0, 6.0]})
    data, ranks = extract(msg, [])

    assert ranks == 4
    bw = _bandwidth(data)
    assert bw["b_rank_avr"] == [3.0, 0.0]
    assert bw["t_rank_s"] == [1.0, 2.0]
    assert bw["t_rank_e"] == [5.0, 6.0]


def test_rank_level_without_end_times():
    """te omitted: each interval ends where the next starts (last holds)."""
    msg = msgpack.packb({"ranks": 4, "b": [3.0, 0.0, 3.0], "ts": [1.0, 2.0, 4.0]})
    data, ranks = extract(msg, [])

    assert ranks == 4
    bw = _bandwidth(data)
    assert bw["b_rank_avr"] == [3.0, 0.0, 3.0]
    assert bw["t_rank_s"] == [1.0, 2.0, 4.0]
    assert bw["t_rank_e"] == [2.0, 4.0, 4.0]


def test_application_level():
    """{b, t}: an already-overlapped signal, no overlap() step."""
    msg = msgpack.packb({"b": [10.0, 0.0, 10.0], "t": [0.0, 1.0, 2.0]})
    data, ranks = extract(msg, [])

    assert ranks == 0
    bw = _bandwidth(data)
    assert bw["b_overlap_avr"] == [10.0, 0.0, 10.0]
    assert bw["t_overlap"] == [0.0, 1.0, 2.0]


def test_ranks_optional_for_rank_level():
    msg = msgpack.packb({"b": [3.0], "ts": [1.0], "te": [2.0]})
    _, ranks = extract(msg, [])
    assert ranks == 0


def test_unrecognised_shape_raises_valueerror():
    msg = msgpack.packb({"ranks": 4, "foo": [1, 2, 3]})
    with pytest.raises(ValueError):
        extract(msg, [])


@pytest.mark.parametrize(
    "payload",
    [
        {"b": [1e6, 0.0] * 20, "t": [float(i) for i in range(40)]},  # app-level
        {
            "ranks": 4,
            "b": [1e6, 0.0] * 10,
            "ts": [float(5 * i) for i in range(20)],
        },  # rank-level, no te
    ],
    ids=["app-level", "rank-level-no-te"],
)
def test_end_to_end_prediction(payload):
    """Both new wire shapes survive Scales -> Simrun -> Bandwidth and predict.

    Guards the Bandwidth.__init__ path where a leftover empty t_rank_s would
    make overlap([], [], []) wipe the app-level t_overlap.
    """
    from ftio.cli.ftio_core import main

    preds, _ = main(["ftio", "--zmq", "-e", "no", "-f", "10"], [msgpack.packb(payload)])
    assert preds and preds[0].t_end > preds[0].t_start
