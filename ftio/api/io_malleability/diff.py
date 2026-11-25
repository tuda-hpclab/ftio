"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rich.console import Console


def parse_args():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert an XLSX file to JSON.")
    # Set default values for the file
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="IOTraces_2.xlsx",
        help="Path to the Excel file (default: IOTraces.xlsx)",
    )
    parser.add_argument(
        "--out", "-o", type=str, default="diff.json", help="Output file name"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_true",
        help="If passed, allows to select the relevant field for the bandwith",
    )
    parser.add_argument(
        "-t",
        "--time_column",
        default=-1,
        dest="t",
        type=int,
        help="specifies the column directly for time (see -i for help)",
    )

    return parser.parse_args()


def parse_txt(args, file_path):
    console = Console()
    ranks = 0
    total_bytes = 0
    t = []
    # Prepare to store the data columns (for interaction)
    data = []

    # Read the file and prepare columns
    with open(file_path) as file:
        for line in file:
            parts = line.split()
            # if len(parts) >= 13:
            data.append(parts)  # Store the full line (split into parts)

    # If interactive mode is enabled, display a preview of the data
    if args.interactive:
        num_rows = len(data)
        preview_rows = min(10, num_rows)  # Display up to 10 rows for preview

        # Print a preview of the first few lines
        console.print(f"[green]Preview of the first {preview_rows} rows of data:[/]")
        for i in range(preview_rows):
            console.print(f'{i}: {" | ".join(data[i])}')

        console.print("\nPlease select the column to map to time:")
        value_t = int(input("> "))
        console.print(f"[green]> Time set to column {value_t}: {data[0][value_t]}[/]")
        for i in range(num_rows):
            t.append(float(data[i][value_t]))  # Append time data

    elif args.t >= 0:
        value_t = int(args.t)
        console.print(f"[green]> Selected Time column: {value_t}")
        for i in range(len(data)):
            t.append(float(data[i][value_t]))  # Append time data

    t = np.diff(np.asarray(t)).tolist()
    samples = np.arange(len(t)).tolist()

    return ranks, total_bytes, t, samples


def main(args=parse_args()):
    console = Console()

    # Resolve the file path
    file_path = os.path.join(os.path.dirname(__file__), args.file)

    ranks, total_bytes, t, samples = parse_txt(args, file_path)

    # Remove matching NaNs
    valid_indices = [i for i in range(len(t)) if not pd.isna(t[i])]
    t = [t[i] for i in valid_indices]
    samples = [samples[i] for i in valid_indices]

    # Construct JSON format
    json_data = {
        "write_sync": {
            "total_bytes": total_bytes,
            "number_of_ranks": ranks,
            "bandwidth": {"b_overlap_avr": t, "t_overlap": samples},
        }
    }

    # Save JSON to file
    with open(args.out, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    # Print JSON for verification
    # print(json.dumps(json_data, indent=4))
    console.print("[green]--- done ---[/]")


# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
