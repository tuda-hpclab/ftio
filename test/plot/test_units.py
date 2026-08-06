"""
Author: lucasch03
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: 0.0.8
Date: Jan 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import numpy as np
import pytest

from ftio.plot.units import set_unit

"""
Tests for class ftio/plot/units.py
"""


def test_units_empty():
    arr = np.array([])
    unit, order = set_unit(arr)
    assert unit == "B/s"
    assert order == 1


def test_units_gigabyte():
    arr = np.array([10000000000.0, 20000000000.0])
    unit, order = set_unit(arr)
    assert unit == "GB/s"
    assert order == 1e-09


def test_units_megabyte():
    arr = np.array([10000000.0, 20000000.0])
    unit, order = set_unit(arr)
    assert unit == "MB/s"
    assert order == 1e-06


def test_units_kilobyte():
    arr = np.array([10000.0, 20000.0])
    unit, order = set_unit(arr)
    assert unit == "KB/s"
    assert order == 0.001


def test_units_byte():
    arr = np.array([100, 200])
    unit, order = set_unit(arr)
    assert unit == "B/s"
    assert order == 1


def test_units_suffix():
    arr = np.array([10000000000.0])
    unit, order = set_unit(arr, suffix="tuda")
    assert unit == "Gtuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_units_python_int_beyond_int64():
    # window_adaptation passes a plain Python int; when a near-zero frequency
    # makes n_phases tiny, total_bytes/n_phases exceeds int64 and numpy stores
    # it as an object array, where np.log10 raises TypeError. That crash took
    # down the FTIO process and with it every glass run (BSC 44288164).
    unit, order = set_unit(2**70, "B")
    assert unit == "GB"
    assert order == 1e-9


def test_units_plain_int_still_scales():
    assert set_unit(5_000_000, "B") == ("MB", 1e-6)
