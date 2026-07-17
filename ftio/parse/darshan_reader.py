"""
Parses Darshan object to simrun
This function can be also executed as a standalone. Just call:
> python3 darshan_reader.py FILE

Returns:
    list[dict]: _description_

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import importlib.util
import sys
import time

import pandas as pd
from rich.console import Console

# darshan is optional: this module is imported unconditionally via
# ftio.parse.scales -> parse_darshan, so a hard import made every FTIO command
# unusable wherever darshan is missing or broken, just to read Darshan traces.
DARSHAN_AVAILABLE: bool = importlib.util.find_spec("darshan") is not None
if DARSHAN_AVAILABLE:
    import darshan

DARSHAN_MISSING_MSG = (
    "Reading a Darshan trace needs the optional 'darshan' package, which is not "
    "installed.\nInstall FTIO with Darshan support:\n"
    "    pip install 'ftio-hpc[darshan-libs]'\n"
    "or, from a clone:\n"
    "    pip install '.[darshan-libs]'\n"
    "The package also needs the darshan-util C library: pip installs the Python "
    "bindings but not the library, so on Python 3.13/3.14 build darshan-util from "
    "source as well (see the README, Installation)."
)


def extract(path, args) -> tuple[dict, int]:
    """extracts Darshan file and generates dictionary with relevant keys

    Args:
        path (str): filename
        args (Argparse): optional arguments

    Returns:
        tuple[dict,int]:
            1. dictionary with relevant files
            2. number of ranks
    """
    dataframe, ranks, time_total = extract_data(path, args)
    write, read, time_io = extract_darshan(dataframe)
    data = {
        "read_sync": read,
        "write_sync": write,
        "io_time": {**time_total, **time_io},
    }
    return data, ranks


def extract_data(path: str, args) -> tuple[list, int, dict]:
    """Extracts module from Darshan

    Args:
        path (str): file location
        args (Argparse): optional arguments

    Raises:
        RuntimeError: Not implemented

    Returns:
        _type_: tuple[dataframe,int]
    """
    if not DARSHAN_AVAILABLE:
        raise ImportError(DARSHAN_MISSING_MSG)

    start = time.time()
    total_time = {}
    with darshan.DarshanReport(path, read_all=True) as report:
        ranks = int(report.metadata["job"]["nprocs"])
        console = Console()
        console.print(f"[cyan]Elapsed time:[/] {time.time()-start:.3f} s")
        start = time.time()
        # dataframe = pd.DataFrame()
        dataframe = []
        # get modules captured
        modules = list(report.modules.keys())
        if isinstance(args, list) or "MPI" in args.dxt_mode.upper():
            if "DXT_MPIIO" in modules:
                dataframe = report.records["DXT_MPIIO"].to_df()
            if "LUSTRE" in modules:
                dataframe = report.records["LUSTRE"].to_df()
            else:
                console.print("[red]No DXT Module[/]\n[cyan]Trying heatmap[/]")
                dataframe, freq, total_time = extract_heatmap(report, "MPIIO")
                if isinstance(args, list):
                    pass
                elif freq > 0 and "ftio" in args.files[0]:
                    args.freq = freq
                    console.print(f"[cyan]Adjusting sampling freq:[/] {freq:.3e}")

        elif "POSIX" in args.dxt_mode.upper():
            if "DXT_POSIX" in modules:
                dataframe = report.records["DXT_POSIX"].to_df()
            else:
                console.print("[red]No DXT Module[/]\n[cyan]Trying heatmap[/]")
                dataframe, freq, total_time = extract_heatmap(report, "POSIX")
                if freq > 0:
                    args.freq = freq
                    console.print(f"[cyan]Adjusting sampling freq:[/] {freq:.3e}")

        console.print(f"[cyan]Done:[/] {time.time()-start:.3f} s\n")

    return dataframe, ranks, total_time


def extract_heatmap(report, kind: str) -> tuple[list, float, dict]:
    """Extract heatmap to support types

    Args:
        report: darshan report
        kind (str): either MPIIO, POSIX, or STDIO

    Returns:
        list: data
        float: adjusted sampling frequency
        float: total time
    """
    dataframe = []
    total_time = {}
    freq = 0
    for mode in ["read", "write"]:
        heatmap = report.heatmaps[kind].to_df([mode])
        # bins = heatmap._nbins
        bin_width = report.heatmaps[kind]._bin_width_seconds
        freq = 5 / bin_width
        col_name = heatmap.columns.values
        t_sum_rank = 0  # Measures total time
        for rank, row in heatmap.iterrows():
            length = []
            start_time = []
            end_time = []
            for n_bin, value in enumerate(row):
                if value != 0:
                    length.append(value)
                    start_time.append(col_name[n_bin].left)
                    end_time.append(col_name[n_bin].right)
            if length:
                dataframe.append(
                    {
                        "rank": rank,
                        f"{mode}_segments": pd.DataFrame(
                            {
                                "length": length,
                                "start_time": start_time,
                                "end_time": end_time,
                            }
                        ),
                    }
                )
            else:
                dataframe.append({"rank": rank, f"{mode}_segments": pd.DataFrame()})
            t_sum_rank += col_name[-1].right

        total_time["delta_t_agg"] = t_sum_rank

    return dataframe, freq, total_time


def extract_darshan(dataframe: list) -> tuple[dict, dict, dict]:
    """_summary_

    Args:
        df (pd.Dataframe): dataframe contating darshan module

    Returns:
        tuple[dict, dict]:
            1. write dict
            2. read dict
            3. time dict
    """
    write = {
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
    read = {
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
    time_sr = 0
    time_sw = 0

    for rank, _ in enumerate(dataframe):
        if (
            "write_segments" in dataframe[rank]
            and not dataframe[rank]["write_segments"].empty
        ):
            bandwidth = dataframe[rank]["write_segments"]["length"] / (
                dataframe[rank]["write_segments"]["end_time"]
                - dataframe[rank]["write_segments"]["start_time"]
            )
            write["bandwidth"]["b_rank_avr"].extend(bandwidth.to_list())
            write["bandwidth"]["b_rank_sum"].extend(bandwidth.to_list())
            write["bandwidth"]["t_rank_e"].extend(
                dataframe[rank]["write_segments"]["end_time"].to_list()
            )
            write["bandwidth"]["t_rank_s"].extend(
                dataframe[rank]["write_segments"]["start_time"].to_list()
            )
            write["number_of_ranks"] = max(
                write["number_of_ranks"], dataframe[rank]["rank"] + 1
            )
            time_sw += sum(
                (
                    dataframe[rank]["write_segments"]["end_time"]
                    - dataframe[rank]["write_segments"]["start_time"]
                ).to_list()
            )
            write["total_bytes"] += sum(dataframe[rank]["write_segments"]["length"])
            write["max_bytes_per_rank"] = max(
                write["max_bytes_per_rank"],
                sum(dataframe[rank]["write_segments"]["length"]),
            )
            write["max_bytes_per_phase"] = max(
                write["max_bytes_per_phase"],
                max(dataframe[rank]["write_segments"]["length"]),
            )
            write["max_io_phases_per_rank"] = max(
                write["max_io_phases_per_rank"],
                len(dataframe[rank]["write_segments"]["length"]),
            )
            write["total_io_phases"] += len(dataframe[rank]["write_segments"]["length"])

        if (
            "read_segments" in dataframe[rank]
            and not dataframe[rank]["read_segments"].empty
        ):
            bandwidth = dataframe[rank]["read_segments"]["length"] / (
                dataframe[rank]["read_segments"]["end_time"]
                - dataframe[rank]["read_segments"]["start_time"]
            )
            read["bandwidth"]["b_rank_avr"].extend(bandwidth.to_list())
            read["bandwidth"]["b_rank_sum"].extend(bandwidth.to_list())
            read["bandwidth"]["t_rank_e"].extend(
                dataframe[rank]["read_segments"]["end_time"].to_list()
            )
            read["bandwidth"]["t_rank_s"].extend(
                dataframe[rank]["read_segments"]["start_time"].to_list()
            )
            read["number_of_ranks"] = max(
                read["number_of_ranks"], dataframe[rank]["rank"] + 1
            )
            time_sr += sum(
                (
                    dataframe[rank]["read_segments"]["end_time"]
                    - dataframe[rank]["read_segments"]["start_time"]
                ).to_list()
            )
            read["total_bytes"] += sum(dataframe[rank]["read_segments"]["length"])
            read["max_bytes_per_rank"] = max(
                read["max_bytes_per_rank"],
                sum(dataframe[rank]["read_segments"]["length"]),
            )
            read["max_bytes_per_phase"] = max(
                read["max_bytes_per_phase"],
                max(dataframe[rank]["read_segments"]["length"]),
            )
            read["max_io_phases_per_rank"] = max(
                read["max_io_phases_per_rank"],
                len(dataframe[rank]["read_segments"]["length"]),
            )
            read["total_io_phases"] += len(dataframe[rank]["read_segments"]["length"])

    # total time
    time = {
        "delta_t_sw": time_sw,
        "delta_t_sr": time_sr,
        "delta_t_awa": 0,
        "delta_t_awr": 0,
        "delta_t_aw_lost": 0,
        "delta_t_ara": 0,
        "delta_t_arr": 0,
        "delta_t_ar_lost": 0,
        "delta_t_overhead": 0,
        "delta_t_agg_io": time_sw + time_sr,
    }

    return write, read, time


def main(args) -> None:
    """Pass varibales and call main_core. The extraction of the traces
    and the parsing of the arguments is done in this function.
    """
    file = args[1]
    data, _ = extract(file, args[1:])
    print(data)


if __name__ == "__main__":
    main(sys.argv)
