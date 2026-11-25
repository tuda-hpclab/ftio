"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""


def convert() -> tuple[dict[str, str], dict[str, tuple[str, float]]]:
    """Converts input according to pattern into and ftio supported file format.
    The translation dictionary species the matching fields from ftio/parse/sample.py
    Returns:
        tuple[dict[str,str],dict[str,tuple[str,float]]]: pattern and translation
            (1) pattern (dict[str,str]): dictionary containing name and a regex
                        expression to find the custom pattern.
            (2) translate (dict[str,tuple[str,(optional)float]]): dictionary
                        containing matching filed from sample.py and the matching
                        the name from the pattern. The unit can be optionally specified
    """
    pattern = {
        "avg_throughput_mib": r"avg_throughput_mib:\s*\[([\d.\d,\s*]+)\]",
        "end_t_micro": r"end_t_micro:\s*\[([\d,\s*]+)\]",
        "start_t_micro": r"start_t_micro:\s*\[([\d,\s*]+)\]",
        # "req_size": r"req_size:\s+\[([\d,\s]+)\]",
        "total_bytes": r"total_bytes:\s*+(\d+)",
        "total_iops": r"total_iops:\s*+(\d+)",
    }

    # Define the map according to sample.py class, along with the scale if any:
    # ftio_field: ("custom_name", scale)
    # ftio units are default in bytes, b/s, ...
    # scale applies ftio_field = custom_name*scale
    translate = {
        "bandwidth": {
            "b_rank_avr": ("avg_throughput_mib", 1.07 * 1e6),
            "t_rank_e": ("end_t_micro", 1e-3),
            "t_rank_s": ("start_t_micro", 1e-3),
        },
        # "max_bytes_per_phase": "req_size",
        "total_bytes": "total_bytes",
        "max_io_ops_per_rank": "total_iops",
    }
    return pattern, translate
