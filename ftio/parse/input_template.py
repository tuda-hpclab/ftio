"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""


def init_data(args: list) -> tuple[str, dict[str, int], dict[str, int]]:
    """init data set

    Returns:
        tuple[dict[str, int], dict[str, int]]: empty data
    """

    io_data = {
        "number_of_ranks": 0,
        "total_bytes": 0,
        "max_bytes_per_rank": 0,
        "max_bytes_per_phase": 0,
        "max_io_phases_per_rank": 0,
        "total_io_phases": 0,
        "bandwidth": {
            "b_rank_sum": [],
            "b_rank_avr": [],
            "t_rank_s": [],
            "t_rank_e": [],
        },
    }

    io_time = {
        "delta_t_sw": 0,
        "delta_t_sr": 0,
        "delta_t_awa": 0,
        "delta_t_awr": 0,
        "delta_t_aw_lost": 0,
        "delta_t_ara": 0,
        "delta_t_arr": 0,
        "delta_t_ar_lost": 0,
        "delta_t_overhead": 0,
        "delta_t_agg_io": 0,
        "delta_t_agg": 0,
    }

    if isinstance(args, list):
        mode = "write_sync"
    else:
        if "w" in args.mode:
            mode = "write"
        else:
            mode = "read"

        if "async" in args.mode:
            mode = mode + "_async_t"
        else:
            mode = mode + "_sync"

    return mode, io_data, io_time
