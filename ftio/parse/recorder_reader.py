"""
Parses recorder object to simrun
This function can be also executed as a standalone. Just call:
> python3 darshan_reader.py FILE

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

import os
import sys


def extract(path, args) -> tuple[dict, int]:
    """exrtracts Darshan file and generates dictionary with relevent keys

    Args:
        path (str): filename
        args (Argparse): optional arguments

    Returns:
        tuple[dict,int]:
            1. dictionary with relevant files
            2. number of ranks
    """
    data, ranks = extract_data(path, args)
    write, read = extract_recorder(data, ranks)
    data = {"read_sync": read, "write_sync": write}
    return data, ranks


def extract_data(path: str, args) -> tuple[list, int]:
    """Extract relevent fields from all recorder files

    Args:
        path (str): path of file
        args (Argparse): _description_

    Returns:
        tuple[list, int]:
        1. list of data contating read and write information
        2. number of ranks
    """
    data = []
    rank = 0
    for root, _, files in os.walk(path):
        for file in sorted(files, key=len):
            file = os.path.join(root, file)
            with open(file) as f:
                current_file = f.readlines()
            # data.extend([k for k in f if 'MPI_File_w' in k or 'MPI_File_r' in k])
            data.extend([k for k in current_file if " write " in k or " read " in k])
        rank = max([int(x.replace(".txt", "")) for x in files]) + 1
        break  # no recursive walk
    return data, rank


def extract_recorder(data: list, ranks: int) -> tuple[dict, dict]:
    """Extract recorder traces

    Args:
        data (array): simulation data

    Returns:
        w,r: returns two dicts containing the data
    """
    # FIXME: This needs to ne b_ind t_ind_s and t_ind_e, however required ind overlaping alorithm in bandiwdth.py
    write = {
        "number_of_ranks": ranks,
        "bandwidth": {
            "b_rank_sum": [],
            "b_rank_avr": [],
            "t_rank_s": [],
            "t_rank_e": [],
        },
    }
    read = {
        "number_of_ranks": ranks,
        "bandwidth": {
            "b_rank_sum": [],
            "b_rank_avr": [],
            "t_rank_s": [],
            "t_rank_e": [],
        },
    }
    for line in data:
        if "write" in line or "read" in line:
            s_line = line.find(" ")
            t_start = float(line[:s_line])
            t_end = float(line[s_line + 1 : s_line + line[s_line + 1 :].find(" ") + 1])
            # ? For MPI oly
            # b  = line.rfind('%p')
            # b = int(line[b+3: b+3+line[b+3:].find(' ')])
            # ? write
            b_part = line.rfind(")")
            b_part = int(line[line[: b_part - 1].rfind(" ") + 1 : b_part - 1])

            b_part = b_part / (t_end - t_start) if t_end - t_start != 0 else 0  # B/s
            # ? Assign
            if "write" in line:
                write["bandwidth"]["t_rank_s"].append(t_start)
                write["bandwidth"]["t_rank_e"].append(t_end)
                write["bandwidth"]["b_rank_avr"].append(b_part)
                write["bandwidth"]["b_rank_sum"].append(b_part)
            elif "read" in line:
                read["bandwidth"]["t_rank_s"].append(t_start)
                read["bandwidth"]["t_rank_e"].append(t_end)
                read["bandwidth"]["b_rank_avr"].append(b_part)
                read["bandwidth"]["b_rank_sum"].append(b_part)

    if not read["bandwidth"]["t_rank_s"]:
        read = {"bandwidth": []}
    if not write["bandwidth"]["t_rank_s"]:
        write = {"bandwidth": []}
    return write, read


def main(args) -> None:
    """Pass varibales and call main_core. The extraction of the traces
    and the parsing of the arguments is done in this function.
    """
    file = args[1]
    dataframe, _ = extract(file, args[1:])
    print(dataframe)


if __name__ == "__main__":
    main(sys.argv)
