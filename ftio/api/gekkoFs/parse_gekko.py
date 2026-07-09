"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import json
from pathlib import Path

import msgpack
import numpy as np


def parse(file_path_or_msg, data, io_type="w", debug_level: int = 0) -> tuple[dict, str]:
    """Parses data from gekko

    Args:
        file_path_or_msg (list): list of files or messages (ZMQ)
        data (dict): data to append to
        io_type (str, optional): Can be w for write or r for read. Defaults to "w".
        debug_level (int, optional): Debug flag for printing fields. Defaults to False.

    Raises:
        RuntimeError: _description_

    Returns:
        tuple[dict, str]: _description_
    """
    if isinstance(file_path_or_msg, bytes):
        extension = "ZMQ"
    else:
        extension = Path(file_path_or_msg).suffix

    # ZMQ
    if "ZMQ" in extension.upper():
        # if data is no struct:
        # unpacked_data = msgpack.unpackb(file_path_or_msg)
        # else:
        unpacker = msgpack.Unpacker()
        unpacker.feed(file_path_or_msg)
        data = assign(data, unpacker, io_type, debug_level)

    # MsgPack
    elif "MSG" in extension.upper():
        # Read the binary data
        with open(file_path_or_msg, "rb") as in_file:
            binary_data = in_file.read()

        # Deserialize the MessagePack data
        unpacker = msgpack.Unpacker()
        unpacker.feed(binary_data)
        data = assign(data, unpacker, io_type, debug_level)

    # JSON
    elif "JSON" in extension.upper():
        with open(file_path_or_msg) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            if "avg_throughput" in key:
                data["avg_throughput"].extend(value)
            elif "start_t_micro" in key:
                data["start_t_micro"].extend(value)
            elif "end_t_micro" in key:
                data["end_t_micro"].extend(value)
            elif "req_size" in key:
                data["req_size"].extend(value)
            elif "hostname" in key:
                data["hostname"] = value
            elif "flush_t_micro" in key:
                data["flush_t_micro"] = value
            elif "pid" in key:
                data["pid"] = value
            elif "total_bytes" in key:
                data["total_bytes"] += value
            elif "total_iops" in key:
                data["total_iops"] += value
            elif "io_type" in key:
                data["io_type"] += value

        scale = [1.07 * 1e6, 1e-3, 1e-3]
        if len(data["avg_throughput"]) > 0:
            data["avg_throughput"] = np.array(data["avg_throughput"]) * scale[0]
            data["t_start"] = np.array(data["start_t_micro"]) * scale[1]
            data["t_end"] = np.array(data["end_t_micro"]) * scale[2]
            if "flush_t" in data:
                data["t_flush"] = data["flush_t_micro"] * scale[2]

    else:
        raise RuntimeError("Unsupported file format specified")

    return data, extension


def assign(data: dict, unpacker, io_type="w", debug_level: int = 0) -> dict:
    # Two wire layouts are supported. The 8-field one is what GekkoFS emits
    # (io_type comes from the filename / -m mode, not the message); the
    # 9-field one is the older format that carried io_type at index 3.
    fields_8 = [
        "flush_t",
        "hostname",
        "pid",
        "start_t_micro",
        "end_t_micro",
        "req_size",
        "total_iops",
        "total_bytes",
    ]
    fields_9 = fields_8[:3] + ["io_type"] + fields_8[3:]

    # materialize once so we can validate the layout before mapping fields
    items = list(unpacker)
    if len(items) == 9:
        data_fields = fields_9
    elif len(items) == 8:
        data_fields = fields_8
    else:
        # malformed / truncated: skip with a log instead of misaligning fields
        print(f"[parse_gekko] skipping message: expected 8 or 9 fields, got {len(items)}")
        return data

    # a field may arrive dict-wrapped ({name: value}); unwrap by its known name
    record = {
        name: (item[name] if isinstance(item, dict) else item)
        for name, item in zip(data_fields, items, strict=True)
    }
    t_flush = max(data["t_flush"], record["flush_t"] * 1e-6)

    # message-carried io_type (9-field layout) is filtered against the request
    skip = "io_type" in record and record["io_type"] != io_type
    if not skip:
        data["t_flush"] = t_flush
        data["hostname"] = record["hostname"]
        data["pid"] = record["pid"]
        # summed across servers (matches the JSON path and the fan-out reduce)
        data["total_iops"] += record["total_iops"]
        data["total_bytes"] += record["total_bytes"]

        # bandwidth is computed per message (µs -> s) so avg_throughput stays
        # aligned 1:1 with this message's events and is grouping-independent
        t_start = np.array(record["start_t_micro"], dtype=float) * 1e-6
        t_end = np.array(record["end_t_micro"], dtype=float) * 1e-6
        req_size = np.array(record["req_size"], dtype=float)
        duration = t_end - t_start
        duration[duration == 0] = 1e-6
        b = req_size / duration  # in B/s

        if np.isnan(b).any():
            print(f"b_rank : {b} \nt_rank_s : {t_start} \nt_rank_e : {t_end} \n")
        b[np.isnan(b)] = 0
        b[np.isinf(b)] = 0

        data["t_start"].extend(t_start)
        data["t_end"].extend(t_end)
        data["req_size"].extend(req_size)
        data["avg_throughput"].extend(b)
        if debug_level > 0:
            total_req = np.sum(data["req_size"])
            print(f"Total request size: {total_req} bytes ({total_req/1e9:.3f} GB)")
            if debug_level > 1:
                # averaged throughput
                print(
                    f"Transfer speed: {np.sum(np.array(data['req_size']))/(max(np.array(data['t_end'])) - min(np.array(data['t_start']))) *1e-9} GB/s"
                )
                print(f"Start time: {np.array(data['t_start'])} sec")
                print(f"End time: {np.array(data['t_end'])} sec")
                print(f"Request size: {np.array(data['req_size'])} bytes")
                # Actual Throughput
                print(f"Bandwidth: {b} b/s")
                if debug_level > 2:
                    print(f"Total bytes: {data['total_bytes']} bytes")
                    print(f"Total IOPS: {data['total_iops']} bytes")

    return data
