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

    Groups by source file (``file_index``), not by rank count: a single file's
    own rank count can vary mid-run under malleability (see
    Simrun.merge_fields), so filtering rows by "number_of_ranks == <value>"
    would silently drop whichever rank count wasn't picked as the file's
    label. Grouping by file_index instead keeps every row from a file
    together regardless of how many distinct rank counts it contains.
    """
    out = []
    file_indices = [int(i) for i in pd.unique(df[0]["file_index"])]
    for j in file_indices:
        file_mask = df[1]["file_index"].isin([j])
        if len(df[1]["file_index"][file_mask]) == 0:
            continue
        time = df[1]["t_overlap"][file_mask].to_numpy()
        bandwidth = df[1]["b_overlap_avr"][file_mask].to_numpy()
        try:
            total_bytes = df[0]["total_bytes"].to_numpy()
            total_bytes = int(float(total_bytes[-1]))
        except ValueError:
            total_bytes = 0
            # expe.center()np.sum(bandwidth * (np.concatenate([time[1:], time[-1:]]) - time)
        rank_sequence, rank_sequence_time = _rank_sequence(df[2], j)
        tmp = {
            "time": time,
            "bandwidth": bandwidth,
            "total_bytes": total_bytes,
            "ranks": _resolve_ranks(df[0], j),
            "rank_sequence": rank_sequence,
            "rank_sequence_time": rank_sequence_time,
        }
        out.append(tmp)
    return out


def _rank_sequence(df2, file_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-burst (rank-level) rank count + its own time basis, for one file.

    This deliberately reads df2 (the rank-level "t_rank_s"/"t_rank_e"
    grouping), not df1 (the overlap-merged "t_overlap" signal FTIO actually
    analyses): Sample.number_of_ranks_sequence is built burst-aligned to
    whichever raw array Simrun.merge_fields saw per JSONL/msgpack part
    (t_rank_s/t_rank_e), which is *not* the same length as the overlap
    algorithm's output in df1 -- overlap merges concurrent per-rank
    intervals into fewer, non-1:1-corresponding points. Pairing the sequence
    with its own (df2) time basis and doing a time-range lookup later (see
    ftio_stft) sidesteps needing that alignment at all.

    Returns two empty arrays if this file's rank count never varied (the
    common case) -- callers should fall back to the scalar ``ranks`` value.
    """
    if "number_of_ranks" not in df2 or "t_rank_e" not in df2:
        return np.array([]), np.array([])
    mask = df2["file_index"].isin([file_index])
    ranks_col = df2["number_of_ranks"][mask]
    if ranks_col.empty or ranks_col.nunique() <= 1:
        return np.array([]), np.array([])
    return ranks_col.to_numpy(), df2["t_rank_e"][mask].to_numpy()


def _resolve_ranks(df0, file_index: int) -> int:
    """Prefer the rank count the trace reports directly (``total_number_of_ranks``,
    the size of the run's own communicator) over ``number_of_ranks``, which is
    only a grouping label -- online, it reflects however many ranks' messages
    had arrived by the time this window was assembled, so it can look like a
    rank change when a straggler rank was just running behind schedule.

    This is a single scalar summary for the whole file (its peak rank count,
    since number_of_ranks/total_number_of_ranks collapse to max() across a
    malleable run -- see Simrun.merge_fields). For a per-window rank count use
    the ``rank_sequence`` array on the returned dict instead.

    df0 mixes a string column ("type") in with the rest, which forces the
    whole frame to string dtype (see Scales.assign_data_io) -- values are
    coerced through pd.to_numeric rather than compared/cast directly.
    """
    file_index_col = pd.to_numeric(df0["file_index"], errors="coerce")
    row = df0.loc[file_index_col == file_index]
    if row.empty:
        return 0
    number_of_ranks = pd.to_numeric(row["number_of_ranks"], errors="coerce").iloc[0]
    if pd.isna(number_of_ranks):
        return 0
    number_of_ranks = int(number_of_ranks)
    if "total_number_of_ranks" not in df0:
        return number_of_ranks
    reported = pd.to_numeric(row["total_number_of_ranks"], errors="coerce").dropna()
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
