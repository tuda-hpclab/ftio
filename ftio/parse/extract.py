"""
Extracts time behavior form parsed data

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import numpy as np
import pandas as pd

from ftio.parse.scales import Scales

# from ftio.freq.helper import get_mode


def get_time_behavior(df) -> list[dict]:
    """Get the time behavior

    Args:
        df (dataframe): obtained from scales.py
    """
    out = []
    files = [int(i) for i in pd.unique(df[0]["number_of_ranks"])]
    for i in files:
        ranks = df[1]["number_of_ranks"].isin([i])
        reported_ranks = _resolve_ranks(df[0], i)
        if len(df[1]["file_index"][ranks]) != 0:
            for j in range(0, int(df[1]["file_index"][ranks].max() + 1)):
                # print(f"  \033[1;32mRanks {i}\033[1;0m")
                file_index = df[1]["file_index"][ranks].isin([j])
                time = df[1]["t_overlap"][ranks][file_index].to_numpy()
                bandwidth = df[1]["b_overlap_avr"][ranks][file_index].to_numpy()
                try:
                    total_bytes = df[0]["total_bytes"].to_numpy()
                    total_bytes = int(float(total_bytes[-1]))
                except ValueError:
                    total_bytes = 0
                    # expe.center()np.sum(bandwidth * (np.concatenate([time[1:], time[-1:]]) - time)
                tmp = {
                    "time": time,
                    "bandwidth": bandwidth,
                    "total_bytes": total_bytes,
                    "ranks": reported_ranks,
                }
                out.append(tmp)
    return out


def _resolve_ranks(df0, number_of_ranks: int) -> int:
    """Prefer the rank count the trace reports directly (``total_number_of_ranks``,
    the size of the run's own communicator) over ``number_of_ranks``, which is
    only a grouping key -- online, it reflects however many ranks' messages had
    arrived by the time this window was assembled, so it can look like a rank
    change when a straggler rank was just running behind schedule.
    """
    if "total_number_of_ranks" not in df0:
        return number_of_ranks
    reported = df0.loc[df0["number_of_ranks"] == number_of_ranks, "total_number_of_ranks"]
    reported = reported.dropna()
    if reported.empty:
        return number_of_ranks
    value = int(reported.iloc[0])
    return value if value > 0 else number_of_ranks


def get_time_behavior_and_args(cmd_input: list[str], msgs=None):
    """
    Parses the input command and messages to extract time behavior and arguments.
    Args:
        cmd_input (list[str]): The input command to be parsed.
        msgs (optional): Additional messages or data to be parsed. Default is None.
    Returns:
        tuple: A tuple containing:
            - data: The extracted time behavior data.
            - args: The extracted arguments.
    """
    #! Parse the data
    data = Scales(cmd_input, msgs)
    #! extract the arguments
    args = data.args

    #! Assign all fields and extract relevant mode
    # # assign the different fields in data (read/write sync/async and io time)
    # data.assign_data()
    # # extract mode of interest
    # df = get_mode(data, args.mode)
    # extract relevant data in one step without unnecessary assigning other fields
    df = data.get_io_mode(args.mode)

    #! extract the fields bandwidth, time, total_bytes, and ranks from the file/msg
    data = get_time_behavior(df)

    return data, args


def extract_fields(data_list):
    """
    Extracts specific fields from a list or dictionary of data.
    Parameters:
    data_list (list or dict): A list containing a single dictionary or a dictionary itself
                            with keys 'bandwidth', 'time', 'total_bytes', and 'ranks'.
    Returns:
    tuple: A tuple containing:
        - bandwidth (np.array): The bandwidth data if present, otherwise an empty numpy array.
        - time_b (np.array): The time data if present, otherwise an empty numpy array.
        - total_bytes (int): The total bytes if present, otherwise 0.
        - ranks (int): The ranks if present, otherwise 0.
    """
    if isinstance(data_list, list):
        data = data_list[0]
    else:
        data = data_list

    bandwidth = data["bandwidth"] if "bandwidth" in data else np.array([])
    time_b = data["time"] if "time" in data else np.array([])
    total_bytes = data.get("total_bytes", 0)
    ranks = data.get("ranks", 0)

    return bandwidth, time_b, total_bytes, ranks
