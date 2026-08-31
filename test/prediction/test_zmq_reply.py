"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import math
import struct

import msgpack
import numpy as np
import pytest

from ftio.freq.prediction import Prediction
from ftio.prediction.helper import build_reply


def _record(freq=2.0, conf=0.9, burst_widths=None):
    """Shape produced by save_data(): {phase, **prediction.to_dict()}."""
    p = Prediction()
    p.dominant_freq = np.array([freq])
    p.conf = np.array([conf])
    p.amp = np.array([1.0])
    p.source = "dft"
    p.t_start, p.t_end = 1.5, 41.5
    p.total_bytes, p.ranks, p.n_samples = 4096, 8, 400
    p.freq = 10.0
    if burst_widths is not None:
        p.burst_widths = np.array(burst_widths, dtype=float)
    return {"phase": 3, **p.to_dict()}


def test_struct_reply_is_16_bytes_freq_conf():
    assert build_reply(_record(2.0, 0.9), "struct") == struct.pack("dd", 2.0, 0.9)


def test_raw_reply_is_struct_bytes():
    assert build_reply(_record(2.0, 0.9), "raw") == struct.pack("dd", 2.0, 0.9)


def test_msgpack_reply_carries_the_full_prediction():
    payload = msgpack.unpackb(build_reply(_record(2.0, 0.9), "msgpack"))
    # the single dominant value, not the array
    assert payload["dominant_freq"] == pytest.approx(2.0)
    assert payload["conf"] == pytest.approx(0.9)
    assert payload["period"] == pytest.approx(0.5)
    # rich fields from to_dict() come through
    assert payload["source"] == "dft"
    assert payload["n_samples"] == 400
    assert payload["t_start"] == pytest.approx(1.5)
    assert payload["total_bytes"] == 4096
    assert payload["ranks"] == 8
    # numpy arrays are serialised as lists
    assert payload["amp"] == [1.0]


def test_msgpack_reply_nan_freq_gives_zero_period():
    payload = msgpack.unpackb(build_reply(_record(np.nan, 0.0), "msgpack"))
    assert payload["period"] == 0.0


def test_msgpack_reply_has_duty_cycle_and_burst_widths():
    payload = msgpack.unpackb(
        build_reply(_record(2.0, 0.9, burst_widths=[0.1, 0.1, 0.1]), "msgpack")
    )
    assert payload["burst_widths"] == [0.1, 0.1, 0.1]
    assert payload["duty_cycle"] == pytest.approx(0.1 * 2.0)


def test_msgpack_reply_without_burst_data():
    payload = msgpack.unpackb(build_reply(_record(2.0, 0.9), "msgpack"))
    assert payload["burst_widths"] == []
    assert math.isnan(payload["duty_cycle"])
