"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jan 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import json

import numpy as np


def parse_options():
    parser = argparse.ArgumentParser(
        description="Converts old traces by scaling them 10^6."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="The paths and name to the JSON file to convert",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        nargs="?",
        default="",
        help="The path and name to the output JSON file",
    )
    args = parser.parse_args()

    return args


def main(args=parse_options()):
    """converts a old trace to a new one by scaling

    Args:
        args (argparse):see @parse_options

    """
    if not args.outfile:
        args.outfile = args.filename.replace(".json", "_new.json")

    fields_metrics = [
        "total_bytes",
        "max_bytes_per_rank",
        "max_transfersize_over_ranks",
    ]
    fields_bandwidth = [
        "arithmetic_mean",
        "median",
        "max",
        "min",
        "b_rank_avr",
        "b_rank_sum",
        "b_ind",
    ]
    with open(args.filename) as current_file:
        data = json.load(current_file)
        for mode in data:
            if ("read" in mode or "write" in mode) and data[mode]:
                for metric in fields_metrics:
                    if metric in data[mode]:
                        scale(data[mode], metric)
                for metric in fields_bandwidth:
                    if "bandwidth" in data[mode] and metric in data[mode]["bandwidth"]:
                        scale(data[mode]["bandwidth"], metric)

    # json.dump(data,out_file)
    with open(args.outfile, "w") as out_file:
        out_file.write(
            "{" + ",\n".join(f'"{i}":' + json.dumps(data[i]) for i in data) + "}\n"
        )


def scale(data_dict: dict, field: str, value: int = 1000000):
    if isinstance(data_dict[field], (float, int)):
        data_dict[field] * value
    elif isinstance(data_dict[field], list):
        data_dict[field] = list(np.array(data_dict[field]) * value)
    else:
        raise TypeError("unsupported type passed")


if __name__ == "__main__":
    main(parse_options())
