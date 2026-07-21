"""
# Time Window functions
This file contains function that allow modifying and setting the data according the the time window:
- data_in_time_window: cuts the data according the start and end time specified by the arguments.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import numpy as np


def data_in_time_window(
    args,
    bandwidth: np.ndarray,
    time_b: np.ndarray,
    total_bytes: int,
    ranks: int = 0,
    rank_sequence: np.ndarray | None = None,
    rank_sequence_time: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None, np.ndarray | None]:
    """Cuts the data according the start and end time specified by the arguments.

    Args:
        args (_type_): argparse
        bandwidth (np.ndarray)
        time_b (np.ndarray)
        total_bytes (int)
        ranks (int, optional). Defaults to 0.
        rank_sequence (np.ndarray, optional): per-burst rank count (see
            extract.get_time_behavior) -- only present when the trace's rank
            count actually varied mid-run (malleability). Has its OWN time
            basis (`rank_sequence_time`), not `time_b`'s, since it comes from
            a different granularity of the source data (see
            extract._rank_sequence) -- cut against its own times, not
            time_b's indices. Defaults to None.
        rank_sequence_time (np.ndarray, optional): time basis for
            `rank_sequence`, same length as it. Defaults to None.

    Returns:
        tuple[np.ndarray,np.ndarray,str,np.ndarray|None,np.ndarray|None]: cut
        bandwidth, time, text, rank_sequence, rank_sequence_time (None if not
        provided)
    """
    text = f"Ranks: [cyan]{ranks}[/]\n"
    ignored_bytes = total_bytes
    have_rank_sequence = rank_sequence is not None and rank_sequence_time is not None
    # shorten data according to start time
    if args.ts:
        indices = np.where(time_b >= args.ts)
        time_b = time_b[indices]
        bandwidth = bandwidth[indices]
        if have_rank_sequence:
            r_indices = np.where(rank_sequence_time >= args.ts)
            rank_sequence = rank_sequence[r_indices]
            rank_sequence_time = rank_sequence_time[r_indices]
        total_bytes = int(
            np.sum(bandwidth * (np.concatenate([time_b[1:], time_b[-1:]]) - time_b))
        )
        text += f"[green]Start time set to {args.ts:.2f}[/] s\n"
    else:
        text += f"Start time: [cyan]{time_b[0]:.2f}[/] s \n"

    # shorten data according to end time
    if args.te:
        indices = np.where(time_b <= args.te)
        time_b = time_b[indices]
        bandwidth = bandwidth[indices]
        if have_rank_sequence:
            r_indices = np.where(rank_sequence_time <= args.te)
            rank_sequence = rank_sequence[r_indices]
            rank_sequence_time = rank_sequence_time[r_indices]
        total_bytes = int(
            np.sum(bandwidth * (np.concatenate([time_b[1:], time_b[-1:]]) - time_b))
        )
        text += f"[green]End time set to {args.te:.2f}[/] s\n"
    else:
        text += f"End time: [cyan]{time_b[-1]:.2f}[/] s\n"

    # ignored bytes
    ignored_bytes = ignored_bytes - total_bytes
    if ignored_bytes < 0:
        ignored_bytes = 0
    text += f"Total bytes: [cyan]{total_bytes:.2e} bytes[/]\n"
    text += f"Ignored bytes: [cyan]{ignored_bytes:.2e} bytes[/]\n"

    return bandwidth, time_b, text, rank_sequence, rank_sequence_time
