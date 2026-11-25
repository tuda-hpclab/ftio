"""
Parses messagpack object to a format similar to jsonl
This function can be also executed as a standalone. Just call:
> python3 msgpack_reader.py FILE

Returns:
    list[dict]: _description_

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import sys

import msgpack


def get_type(arr: list[str]) -> str:
    io_type = ""
    if arr:
        io_type = arr[0]
    return io_type


def convert_to_class(arr: list, io_type: str) -> list:
    if arr:
        if "write" in io_type or "read" in io_type:
            return convert_to_iosample(arr, io_type)
        elif "time" in io_type:
            return convert_to_iotime(arr)
        else:
            raise ValueError(f"Parse error: type is {io_type}")


def convert_to_iosample(arr: list, io_type: str) -> list:
    """Brings the data into an structure similar to Sample/Time
    depending on the type of the I/O (read/write + sync/async | time)

    Args:
        arr (list): metrics
        io_type (str): type of the I/O (read/write + sync/async | time)

    Returns:
        list: list
    """
    if "async" in io_type:
        bandwidth = assign_bandwidth(arr, io_type)
        return [
            {f"{io_type}_t": assign_metrics(arr, bandwidth[0])},
            {f"{io_type}_b": assign_metrics(arr, bandwidth[1])},
        ]
    else:
        bandwidth = assign_bandwidth(arr, io_type)
        return [{io_type: assign_metrics(arr, bandwidth[0])}]


def assign_metrics(arr: list, bandwidth: dict) -> dict:
    return {
        "total_bytes": arr[3],  # Bytes
        "max_bytes_per_rank": arr[4],
        "max_bytes_per_phase": arr[5],
        "max_io_phases_per_rank": arr[6],
        "total_io_phases": arr[7],
        "max_io_ops_in_phase": arr[8],
        "max_io_ops_per_rank": arr[9],
        "total_io_ops": arr[10],
        "number_of_ranks": arr[11],
        "total_number_of_ranks": arr[12],
        "bandwidth": bandwidth,
    }


def assign_bandwidth(arr: list, io_type: str) -> list[dict]:
    # data = []
    t_start = []
    t_end_act = []
    t_end_req = []
    T_sum = []
    T_avr = []
    B_sum = []
    B_avr = []
    if arr[13]:
        for i in arr[13]:
            # data.append(i[0])
            t_start.append(i[1])
            t_end_act.append(i[2])
            T_sum.append(i[4])  # Bytes/s
            T_avr.append(i[5])  # Bytes/s
            if "async" in io_type:
                t_end_req.append(i[3])
                B_sum.append(i[6])  # Bytes/s
                B_avr.append(i[7])  # Bytes/s
            # n_op.append(i[8])
    if "async" in io_type:
        return [
            {
                "b_rank_sum": T_sum,
                "b_rank_avr": T_avr,
                "t_rank_s": t_start,
                "t_rank_e": t_end_act,
            },
            {
                "b_rank_sum": B_sum,
                "b_rank_avr": B_avr,
                "t_rank_s": t_start,
                "t_rank_e": t_end_req,
            },
        ]
    else:
        return [
            {
                "b_rank_sum": T_sum,
                "b_rank_avr": T_avr,
                "t_rank_s": t_start,
                "t_rank_e": t_end_act,
            }
        ]


def convert_to_iotime(arr: list) -> list:
    tmp = {
        "delta_t_agg": arr[1],
        "delta_t_sr": arr[2],
        "delta_t_sw": arr[3],
        "delta_t_ara": arr[4],
        "delta_t_arr": arr[5],
        "delta_t_ar_lost": arr[6],
        "delta_t_awa": arr[7],
        "delta_t_awr": arr[8],
        "delta_t_aw_lost": arr[9],
        "delta_t_agg_io": arr[10],
        "delta_t_rank0": arr[11],
        "delta_t_rank0_app": arr[12],
        "delta_t_rank0_overhead_post_runtime": arr[13],
        "delta_t_rank0_overhead_peri_runtime": arr[14],
        "delta_t_overhead": arr[15],
        "delta_t_overhead_post_runtime": arr[16],
        "delta_t_overhead_peri_runtime": arr[17],
        "delta_t_overhead_dft": arr[18],
    }
    return [{"io_time": tmp}]


def extract(file: str) -> list[dict]:
    """Extracts the data stored in Darshan file

    Args:
        file (str): file name

    Returns:
        list[dict]: file content
    """
    data = []
    # Open the MessagePack binary file for reading
    with open(file, "rb") as in_file:
        # Read the binary data
        binary_data = in_file.read()

    data = extract_data(binary_data, data)
    return data


def extract_data(binary_data, data):
    # Deserialize the MessagePack data into the class instances
    unpacker = msgpack.Unpacker()
    unpacker.feed(binary_data)
    for item in unpacker:
        io_type = get_type(item)
        data.extend(convert_to_class(item, io_type))
    return data


def main(args) -> None:
    """Pass variables and call main_core. The extraction of the traces
    and the parsing of the arguments is done in this function.
    """
    file = args[1]
    extract(file)


if __name__ == "__main__":
    main(sys.argv)
