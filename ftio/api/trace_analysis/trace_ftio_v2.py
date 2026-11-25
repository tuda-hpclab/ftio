"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jul 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
from rich.console import Console

from ftio.parse.csv_reader import read_csv_file

# from ftio.api.trace_analysis.helper import quick_plot
from ftio.processing.compact_operations import quick_ftio

console = Console()


def main(argv=sys.argv[1:], verbose=True, json_flag=True):
    if json_flag:
        # Time step is provided in the data for the json
        return extract_arrays_from_json(argv, verbose)
    else:
        # CSV uses a implicit time step
        return extract_arrays_from_csv(argv, verbose)


def extract_arrays_from_json(argv=sys.argv[1:], verbose=True):
    full_path = get_path(argv, verbose)
    with open(full_path) as file:
        arrays = json.load(file)

    b_r = np.array([])
    t_r = np.array([])
    b_w = np.array([])
    t_w = np.array([])

    if "read" in arrays:
        b_r = np.array(arrays["read"]["bandwidth"]["b_overlap_avr"]).astype(float)
        t_r = np.array(arrays["read"]["bandwidth"]["t_overlap"]).astype(float)
    if "write" in arrays:
        b_w = np.array(arrays["write"]["bandwidth"]["b_overlap_avr"]).astype(float)
        t_w = np.array(arrays["write"]["bandwidth"]["t_overlap"]).astype(float)

    # adapt for FTIO
    # command line arguments
    argv = [x for x in argv if ".py" not in x and ".json" not in x]
    if "-e" not in argv:
        argv.extend(["-e", "no"])

    if verbose:
        console.print(f"Args: {argv}")

    res = run_ftio_on_group(argv, verbose, arrays, b_r, t_r, b_w, t_w)
    return res


def extract_arrays_from_csv(argv=sys.argv[1:], verbose=True):
    full_path = get_path(argv, verbose)
    arrays = read_csv_file(full_path)
    b_r = np.array([])
    b_w = np.array([])
    b_b = np.array([])

    # for key, array in arrays.items():
    #     print(f"{key}: {array}")

    if "read" in arrays:
        b_r = np.array(arrays["read"]).astype(float)
    if "write" in arrays:
        b_w = np.array(arrays["write"]).astype(float)
    if "both" in arrays:
        b_b = np.array(arrays["both"]).astype(float)

    if "timestamp" in arrays:
        entries = [
            datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in arrays["timestamp"]
        ]
        t_s = entries[0]
        time_diffs_in_seconds = [(dt - t_s).total_seconds() for dt in entries]
        t = np.array(time_diffs_in_seconds)
    else:
        t_step = np.nan
        if "--time-step" not in argv:
            t_step = 1
            if verbose:
                console.print(f"[bold]Sampling rate set to {t_step}[/]")
        else:
            flag_index = argv.index("--time-step")
            t_step = float(argv[flag_index + 1])
            argv[flag_index : flag_index + 2] = []
            if verbose:
                console.print(f"[bold green]Sampling rate set to {t_step} sec[/]")

        t = np.arange(0, len(b_w) * t_step, t_step).astype(float)

    # set sampling frequency if not set
    if "-f" not in argv:
        argv.extend(["-f", f"{2/t_step}"])
        if verbose:
            console.print(
                f"[bold green]Sampling rate set to {t_step} sec ({2/t_step:.3f}) Hz[/]"
            )
    # plot
    # quick_plot(t,b_w)

    # adapt for FTIO
    # command line arguments
    argv = [x for x in argv if ".py" not in x and ".csv" not in x]
    if "-e" not in argv:
        argv.extend(["-e", "no"])

    if verbose:
        console.print(f"Args: {argv}")
    # argv = ['-e', 'mat']

    res = run_ftio_on_group(argv, verbose, arrays, b_r, t, b_w, t, b_b, t)
    return res


def run_ftio_on_group(argv, verbose, arrays, b_r, t_r, b_w, t_w, b_b=None, t_b=None):
    if t_b is None:
        t_b = []
    if b_b is None:
        b_b = []
    total_bytes_r = 0  # np.sum(np.repeat(t_s,len(b_r))*len(b_r))
    total_bytes_w = 0  # np.sum(np.repeat(t_s,len(b_w))*len(b_w))
    total_bytes_b = 0  # np.sum(np.repeat(t_s,len(b_b))*len(b_b))
    res_r = {}
    res_w = {}
    res_b = {}

    if "read" in arrays and len(b_r) > 1:
        res_r = quick_ftio(argv, b_r, t_r, total_bytes_r, 10, "read", verbose)
    if "write" in arrays and len(b_w) > 1:
        res_w = quick_ftio(argv, b_w, t_w, total_bytes_w, 10, "write", verbose)
    if "both" in arrays and len(b_b) > 1:
        res_b = quick_ftio(argv, b_b, t_b, total_bytes_b, 10, "both", verbose)

    return {"read": res_r, "write": res_w, "both": res_b}


def get_path(argv, verbose=True):

    full_path = ""
    # Example usage

    # Check if argv is a string
    if argv and isinstance(argv, str):
        argv = [argv]  # Convert string to a list with one element

    for i in argv:
        if "csv" in i or "json" in i:
            path = i
            if os.path.isabs(path):
                # full path
                full_path = path
            else:
                # relative path
                full_path = f"{os.getcwd()}/{path}"
            break

    if not full_path:
        full_path = f"{os.getcwd()}/data_2.csv"

    if verbose:
        console.print(f"[green]current file: {full_path}[/]")

    return full_path


if __name__ == "__main__":
    _ = main(sys.argv)
