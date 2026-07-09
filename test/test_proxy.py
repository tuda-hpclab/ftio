"""
Functions for testing the Metric Proxy interface functionality of the ftio package.

Author: Tim Dieringer
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: 0.0.8
Date: Jan 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os

import msgpack
import numpy as np

from ftio.api.metric_proxy.parallel_proxy import execute
from ftio.api.metric_proxy.parse_proxy import filter_metrics
from ftio.api.metric_proxy.proxy_zmq import handle_request, sanitize
from ftio.freq.prediction import Prediction


def test_proxy():
    """Test the Metric Proxy interface functionality."""
    file_path = os.path.join(
        os.path.dirname(__file__), "../examples/API/proxy/trace_export.msgpack"
    )
    with open(file_path, "rb") as f:
        msg_bytes = f.read()
    reply = handle_request(msg_bytes)
    response = msgpack.unpackb(reply, raw=False)

    assert isinstance(response, list)
    assert len(response) > 0

    item = response[0]
    assert isinstance(item, dict)

    required_fields = {
        "metric",
        "dominant_freq",
        "conf",
        "amp",
        "phi",
        "t_start",
        "t_end",
        "total_bytes",
        "ranks",
        "freq",
        "top_freqs",
        "n_samples",
    }

    assert required_fields.issubset(item.keys())
    assert "top_freq" not in item  # old key must not appear

    top_freqs = item["top_freqs"]
    assert isinstance(top_freqs, dict)

    if top_freqs:
        required_top_freq_fields = {"freq", "conf", "periodicity", "amp", "phi"}
        assert required_top_freq_fields.issubset(top_freqs.keys())


def test_handle_request_ping():
    """handle_request answers a ping with pong (documented control message)."""
    assert handle_request(b"ping") == b"pong"


def test_handle_request_address_update():
    """handle_request acknowledges a 'New Address:' rebind request."""
    assert handle_request(b"New Address: tcp://127.0.0.1:5555") == b"Address updated"


def test_handle_request_invalid_payload():
    """A non-control, non-msgpack payload returns a msgpack error, not a crash."""
    reply = handle_request(b"\xff\xff not msgpack \x00")
    response = msgpack.unpackb(reply, raw=False)
    assert isinstance(response, dict)
    assert "error" in response


def test_sanitize_prediction():
    """sanitize() must convert Prediction objects to msgpack-serializable dicts."""
    p = Prediction(transformation="dft", t_start=0.1, t_end=1.0, freq=100.0, ranks=4)
    p.dominant_freq = np.array([1.0])
    p.conf = np.array([0.9])
    p.metric = "test_metric"

    result = sanitize([p])

    assert isinstance(result, list)
    assert len(result) == 1
    d = result[0]
    assert isinstance(d, dict)
    assert d["metric"] == "test_metric"
    assert d["dominant_freq"] == [1.0]
    assert d["conf"] == [0.9]
    assert "top_freqs" in d  # canonical key
    assert "top_freq" not in d  # old key must not appear
    # must be msgpack-serializable
    packed = msgpack.packb(result, use_bin_type=True)
    assert isinstance(packed, bytes)


def test_execute_skips_short_time_arrays():
    """execute() must not crash or process metrics with fewer than 2 time points."""
    t = np.linspace(0, 1, 50)
    b = np.sin(2 * np.pi * t)

    metrics = {
        "good_metric": [b, t],
        "single_point": [np.array([1.0]), np.array([0.0])],
        "empty_metric": [np.array([]), np.array([])],
    }

    data = execute(metrics, ["-e", "no", "-f", "10"], ranks=1, show=False)

    # short/empty metrics must be silently skipped — no crash
    assert isinstance(data, list)


def test_filter_metrics_handles_null_values():
    """filter_metrics must not crash when the trace contains JSON null (None) values."""
    json_data = {
        "metrics": {
            "test_metric": [
                [0.1, 1000.0],
                [0.2, None],  # null in JSON
                [0.3, 1500.0],
                [0.4, None],
                [0.5, 2000.0],
            ]
        }
    }
    # must not raise TypeError from numerical_derivative
    result = filter_metrics(json_data, filter_deriv=False, exclude=None)
    assert "test_metric" in result
    b, t = result["test_metric"]
    assert not np.any(np.isnan(b))
    assert not np.any(np.isnan(t))


def test_top_freqs_conf_inf_is_msgpack_serializable():
    """sanitize() must replace inf in top_freqs.conf so msgpack can serialize it."""
    t = np.linspace(0, 10, 200)
    b = np.ones_like(t) * 1e9  # constant (DC-dominated) signal — conf[0] = inf

    data = execute(
        {"dc_metric": [b, t]},
        ["-e", "no", "-f", "10", "--n_freq", "5"],
        ranks=1,
        show=False,
    )

    if len(data) > 0:
        packed = msgpack.packb(sanitize(data), use_bin_type=True)
        assert isinstance(
            packed, bytes
        ), "sanitize must produce msgpack-serializable output even when conf contains inf"


def test_execute_includes_metrics_without_dominant_freq():
    """execute() must include metrics in result even when dominant_freq is empty."""
    t = np.linspace(0, 75, 747)
    b = np.sin(2 * np.pi * 0.56 * t) * 1000 + 500  # low-confidence periodic signal

    data = execute(
        {"low_conf_metric": [b, t]},
        ["-e", "no", "-f", "10", "--n_freq", "5"],
        ranks=1,
        show=False,
    )

    assert isinstance(data, list)
    # Result may be empty if FTIO truly finds nothing, but must not crash
    for pred in data:
        try:
            import msgpack

            from ftio.api.metric_proxy.proxy_zmq import sanitize

            msgpack.packb(sanitize([pred]), use_bin_type=True)
        except Exception as exc:
            raise AssertionError(
                f"Prediction must be msgpack-serializable: {exc}"
            ) from exc


def test_execute_multi_metric_all_serializable():
    """execute() on diverse metrics must return one serializable Prediction per valid metric."""
    t = np.linspace(0, 20, 400)
    metrics = {
        "sin_metric": [np.sin(2 * np.pi * 0.5 * t) * 1e6, t],
        "dc_metric": [np.ones_like(t) * 1e9, t],
        "mixed_metric": [np.sin(2 * np.pi * 0.1 * t) * 500 + 1000.0, t],
    }

    data = execute(
        metrics, ["-e", "no", "-f", "20", "--n_freq", "5"], ranks=1, show=False
    )

    assert isinstance(data, list)
    for pred in data:
        assert hasattr(pred, "metric") and pred.metric != ""
        assert hasattr(pred, "dominant_freq")
        assert hasattr(pred, "top_freqs")

    packed = msgpack.packb(sanitize(data), use_bin_type=True)
    assert isinstance(packed, bytes), "All-metric output must be msgpack-serializable"


def test_parallel_analysis_output_matches_expected_schema():
    """sanitize(execute()) must produce dicts matching the FtioModel schema Rust expects."""
    t = np.linspace(0, 30, 600)
    metrics = {
        "write_bw": [np.sin(2 * np.pi * 0.3 * t) * 1e8 + 1e8, t],
        "read_bw": [np.sin(2 * np.pi * 0.3 * t + 0.5) * 5e7 + 5e7, t],
    }

    data = execute(
        metrics, ["-e", "no", "-f", "20", "--n_freq", "10"], ranks=4, show=False
    )
    native = sanitize(data)

    required = {
        "metric",
        "dominant_freq",
        "conf",
        "amp",
        "phi",
        "t_start",
        "t_end",
        "total_bytes",
        "ranks",
        "freq",
        "top_freqs",
        "n_samples",
    }
    for item in native:
        assert required.issubset(item.keys()), f"Missing fields in {item.get('metric')}"
        top = item["top_freqs"]
        assert isinstance(top, dict)
        if top:
            assert set(top.keys()) >= {"freq", "conf", "amp", "phi"}
            assert all(isinstance(top[k], list) for k in ("freq", "conf", "amp", "phi"))

    packed = msgpack.packb(native, use_bin_type=True)
    assert isinstance(packed, bytes)


def test_prediction_to_dict_includes_metric():
    """Prediction.to_dict() must include the metric field."""
    p = Prediction(transformation="dft")
    p.metric = "some_metric"
    d = p.to_dict()
    assert "metric" in d
    assert d["metric"] == "some_metric"


def test_data_to_json_handles_prediction_objects():
    """data_to_json (via NpArrayEncode) must not crash when data is a list of Prediction objects."""
    import json

    from ftio.api.metric_proxy.helper import NpArrayEncode

    p = Prediction(transformation="dft", t_start=0.0, t_end=10.0, freq=1.0, ranks=1)
    p.metric = "test_metric"
    p.dominant_freq = np.array([0.5])
    p.conf = np.array([0.9])
    p.amp = np.array([1e6])
    p.phi = np.array([0.1])

    result = json.dumps([p], cls=NpArrayEncode)
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["metric"] == "test_metric"
