"""
Helper functions

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import tempfile

import msgpack
import numpy as np

from ftio.freq.prediction import Prediction


def get_dominant(prediction: Prediction) -> float:
    """Gets the dominant frequency based on the confidence

    Args:
        prediction (dict|predicition): prediction contacting the dominant frequencies and their confidence

    Returns:
        float: dominant frequency (only one value!)
    """
    if isinstance(prediction, Prediction):
        return prediction.get_dominant_freq()
    elif isinstance(prediction, dict):
        tmp = Prediction()
        tmp.set_from_dict(
            {
                "dominant_freq": prediction["dominant_freq"],
                "conf": prediction["conf"],
                "amp": prediction["amp"],
            }
        )
        return tmp.get_dominant_freq()
    else:
        raise TypeError("prediction must be a Prediction or dict")


def get_dominant_and_conf(prediction: Prediction | dict) -> tuple[float, float]:
    """Gets the dominant frequency and its confidence based on the confidence

    Args:
        prediction (Prediction|dict): prediction contacting the dominant frequencies and their confidence

    Returns:
        tuple[float, float]: dominant frequency (only one value!) and corresponding confidence
    """

    if isinstance(prediction, Prediction):
        return prediction.get_dominant_freq_and_conf()
    elif isinstance(prediction, dict):
        tmp = Prediction()
        tmp.set_from_dict(
            {
                "dominant_freq": prediction["dominant_freq"],
                "conf": prediction["conf"],
                "amp": prediction["amp"],
            }
        )
        return tmp.get_dominant_freq_and_conf()
    else:
        raise TypeError("prediction must be a Prediction or dict")


def print_data(data: list[dict]) -> None:
    """Prints the predictions in a nice format

    Args:
        data [dict]: contacting the predictions
    """
    print("Data collected is:")
    for pred in data:
        string = ""
        for keys, values in pred.items():
            if isinstance(values, float):
                string += f"'{keys}': {values:.2f}, "
            else:
                string += f"'{keys}': {values}, "
        print("{" + string[:-2] + "}")


def export_extrap(data: list[dict], name: str = "./freq.jsonl"):
    """Generates measurement points for Extra-p out of the frequency
    collected at different phases

    Args:
        data (list[dict]): List of predictions
        name (str, optional): Name of the output file. Defaults to "./freq.jsonl".
    """
    extrap_string, ranks = format_jsonl(data)

    if not np.isnan(ranks):
        name = f"./freq_{ranks}.jsonl"
    with open(name, "w", encoding="utf-8") as file:
        file.write(extrap_string)


def format_jsonl(data: list[dict]) -> tuple[str, float]:
    """Formats the metric as in the JSONL format for Extra-P

    Args:
        data (list[dict]): List of predictions

    Returns:
        tuple[str, float]: formatted string and number of ranks

    """
    string = ""
    out_ranks = np.nan
    for pred in data:
        ranks = np.nan
        call_path = "main"
        dominant_freq = np.nan
        for keys, values in pred.items():
            if "dominant_freq" in keys:
                dominant_freq, _ = get_dominant_and_conf(pred)
            if "ranks" in keys:
                ranks = values
            if not np.isnan(dominant_freq) and not np.isnan(ranks):
                string += f'{{"params":{{"Processes":{ranks}}},"callpath":"{call_path}","metric":"Frequency (Hz)","value":{dominant_freq:e} }}\n'
                out_ranks = ranks
                if dominant_freq > 0:
                    string += f'{{"params":{{"Processes":{ranks}}},"callpath":"{call_path}","metric":"Period (s)","value":{1 / dominant_freq:e} }}\n'
                break

    return string, out_ranks


def dump_json(b: np.ndarray, t: np.ndarray, filename: str = "bandwidth.json") -> None:
    data = {"b": b.tolist(), "t": t.tolist()}
    json_file_path = os.path.join(os.getcwd(), filename)
    # Atomic write: write to a temp file first so a mid-write kill leaves
    # the previous complete bandwidth.json intact rather than a truncated one.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.getcwd(), suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, json_file_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def build_reply(prediction: dict, reply_format: str = "msgpack") -> bytes:
    """Serialise a prediction record for the ZMQ reply socket.

    ``struct``/``raw`` give ``pack("dd", freq, conf)`` (the legacy TMIO bytes).
    ``msgpack`` gives the whole record (every ``Prediction.to_dict()`` field,
    numpy converted to lists) with ``dominant_freq``/``conf`` set to the single
    dominant value and ``period``/``duty_cycle`` derived.
    """
    freq, conf = get_dominant_and_conf(prediction)
    freq = 0.0 if np.isnan(freq) else float(freq)
    conf = 0.0 if np.isnan(conf) else float(conf)

    if reply_format in ("struct", "raw"):
        return struct.pack("dd", freq, conf)

    payload = Prediction().convert_json(prediction)
    payload["dominant_freq"] = freq  # to_dict() stores the array; send the strongest
    payload["conf"] = conf
    payload["period"] = 1.0 / freq if freq > 0 else 0.0

    result = msgpack.packb(payload)
    return result if isinstance(result, bytes) else b""
