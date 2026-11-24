"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import re
import sys
import time

from rich.console import Console

from ftio.parse.custom_patterns import convert
from ftio.parse.input_template import init_data


def extract(path: str, args: list, custom: bool = False) -> tuple[dict, int]:
    # init
    start = time.time()
    ranks = 0
    mode, io_data, io_time = init_data(args)

    # Define your pattern and translate it
    if not custom:
        pattern, translate = convert()
    else:
        data = {}
        with open(f"{args.custom_file}") as f:
            exec(f.read(), globals(), data)
        pattern = data["pattern"]
        translate = data["translate"]

    # read data
    extracted_data = read(path, pattern)

    # Display the extracted data
    for key, value in translate.items():
        if "bandwidth" in key:
            for key2, value2 in translate["bandwidth"].items():
                unit, value2 = find_scale(value2)
                io_data["bandwidth"][key2] = extracted_data[value2]
                io_data["bandwidth"][key2] = scale(io_data["bandwidth"][key2], unit)
        else:
            unit, value = find_scale(value)
            io_data[key] = extracted_data[value]
            io_data[key] = scale(io_data[key], unit)

    console = Console()
    console.print(f"[cyan]Elapsed time:[/] {time.time()-start:.3f} s")

    # fill time
    kind = mode[0] + mode.split("_")[1][0]
    if "a" in mode[0]:
        kind = kind + "a"
    io_time[f"delta_t_{kind}"] = 0

    # pack everything
    data = {
        f"{mode}": io_data,
        "io_time": io_time,
    }

    return data, ranks


def read(file_path, patterns):
    results = {}
    with open(file_path) as file:
        data = file.read()
        for key, pattern in patterns.items():
            match = re.search(pattern, data)
            if match:
                if "," in match.group(1):
                    # If the matched group contains commas, split and convert to integers
                    tmp = match.group(1).replace(", ", ",")
                    results[key] = [
                        float(val) if "." in val else int(val) for val in tmp.split(",")
                    ]

                else:
                    results[key] = int(match.group(1))
            else:
                print(f"Unable to extract data for {key} from the file.")

    return results


def find_scale(value: dict):
    unit = 1
    if isinstance(value, tuple):
        unit = value[1]
        value = value[0]
    return unit, value


def scale(mylist: list, unit: float) -> list:
    """scales lists with unit

    Args:
        mylist (list): input list
        unit (float): unit

    Returns:
        list: scale
    """
    if unit != 1:
        return [unit * i for i in mylist]
    else:
        return mylist


def main(args) -> None:
    """Pass varibales and call main_core. The extraction of the traces
    and the parsing of the arguments is done in this function.
    """
    file = args[1]
    data, _ = extract(file, args[1:])
    print(data)


if __name__ == "__main__":
    main(sys.argv)
