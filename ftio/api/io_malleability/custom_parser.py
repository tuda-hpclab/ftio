"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mär 2025

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

from ftio.parse.bandwidth import overlap


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
        "--out", "-o", type=str, default="output.json", help="Output file name"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_true",
        help="If passed, allows to select the relevant field for the bandwith",
    )
    parser.add_argument(
        "-b",
        "--bandwidth_column",
        default=-1,
        dest="b",
        type=int,
        help="specifies the column directly for bandwidth (see -i for help)",
    )
    parser.add_argument(
        "-t",
        "--time_column",
        default=-1,
        dest="t",
        type=int,
        help="specifies the column directly for time (see -i for help)",
    )
    parser.add_argument(
        "-io_time",
        "--io_time_column",
        default=None,
        dest="io_time",
        type=int,
        help="specifies the column directly for io_time (see -i for help)",
    )

    return parser.parse_args()


def parse_txt(args, file_path):
    console = Console()
    ranks = 0
    total_bytes = 0
    b = []
    t_s = []
    io_time = []

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
            console.print(f"{i}: {' | '.join(data[i])}")

        # Ask the user to map the appropriate columns
        console.print("\nPlease select the column to map to bandwidth:")
        value_b = int(input("> "))
        console.print(
            f"[green]> Bandwidth set to column {value_b}: {data[0][value_b]}[/]"
        )
        for i in range(num_rows):
            b.append(float(data[i][value_b]))  # Append bandwidth data

        console.print("\nPlease select the column to map to time:")
        value_t = int(input("> "))
        console.print(f"[green]> Time set to column {value_t}: {data[0][value_t]}[/]")
        for i in range(num_rows):
            t_s.append(float(data[i][value_t]))  # Append time data

        console.print("\nPlease select the column to map to I/O time:")
        value_io = input("> ").strip()
        # Only append io_time if a column is selected
        if value_io:
            value_io = int(value_io)
            console.print(
                f"[green]> I/O Time set to column {value_io}: {data[0][value_io]}[/]"
            )
            for i in range(num_rows):
                io_time.append(float(data[i][value_io]))  # Append I/O time data
        else:
            console.print("[yellow]> I/O Time not selected, skipping assignment.[/]")

    elif args.b >= 0 and args.t >= 0:
        # Use args.b and args.t for the columns directly
        value_b = int(args.b)
        console.print(f"[green]> Selected Bandwidth column: {value_b}")
        for i in range(len(data)):
            b.append(float(data[i][value_b]))  # Append bandwidth data

        value_t = int(args.t)
        console.print(f"[green]> Selected Time column: {value_t}")
        for i in range(len(data)):
            t_s.append(float(data[i][value_t]))  # Append time data

        if args.io_time:
            value_io_time = int(args.io_time)
            console.print(f"[green]> Selected IO time column: {value_io_time}")
            for i in range(len(data)):
                io_time.append(float(data[i][value_io_time]))  # Append time data

    else:
        # If not interactive, just load the data based on fixed indices
        for line in data:
            t_s.append(float(line[1]))  # IOTimeStamp
            ranks = int(line[3])  # NumRanks (assuming constant)
            b.append(int(line[7]))  # IOPerf
            total_bytes = int(line[9])  # Total bytes (assuming constant)
            io_time.append(float(line[10]))  # IO time

    return ranks, total_bytes, b, t_s, io_time


def parse_excel(args, file_path):
    console = Console()
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    ranks = 0
    total_bytes = 0
    b = []
    t_s = []
    io_time = []

    names = df.columns.to_list()
    for name in names:
        if isinstance(name, str):
            name_lower = name.lower()
            if "i/o bandwidth" in name_lower:
                b.extend(df[name].tolist())
            elif "time stamp" in name_lower:
                t_s.extend(df[name].tolist())
            elif "i/o time" in name_lower:
                io_time.extend(df[name].tolist())
            elif "rank" in name_lower:
                ranks = int(df[name].tolist()[0])
            elif "total bytes" in name_lower:
                total_bytes = int(df[name].tolist()[0])

    if args.interactive:
        for i, value in enumerate(names):
            console.print(f"[{i}]: {value}")
        b = []
        t_s = []
        value = input("\nPlease select the column to map to bandwidth:\n> ")
        value = int(value)
        console.print(f"[green]> bandwidth set to [{value}]: {names[value]}[/]")
        b.extend(df[names[value]].tolist())
        value = input("\nPlease select the column to map to time:\n> ")
        value = int(value)
        console.print(f"[green]> time set to [{value}]: {names[value]}[/]")
        t_s.extend(df[names[value]].tolist())
        value = input("\nPlease select the column to map to I/O time:\n> ")
        value = int(value)
        console.print(f"[green]> time set to [{value}]: {names[value]}[/]")
        io_time.extend(df[names[value]].tolist())

    if args.b >= 0:
        value = int(args.b)
        console.print(f"[green]> Selected [{value}]: {names[value]}[/]")
        b.extend(df[names[value]].tolist())

    if args.t >= 0:
        t_s = []
        value = int(args.t)
        console.print(f"[green]> Selected [{value}]: {names[value]}[/]")
        t_s.extend(df[names[value]].tolist())

    if args.io_time:
        io_time = []
        value = int(args.t)
        console.print(f"[green]> Selected [{value}]: {names[value]}[/]")
        io_time.extend(df[names[value]].tolist())

    return ranks, total_bytes, b, t_s, io_time


def main(args=parse_args()):
    console = Console()

    # Resolve the file path
    file_path = os.path.join(os.path.dirname(__file__), args.file)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Error: File "{file_path}" not found.')

    if file_path.endswith(".xlsx"):
        ranks, total_bytes, b, t_s, io_time = parse_excel(args, file_path)
    elif file_path.endswith(".txt"):
        ranks, total_bytes, b, t_s, io_time = parse_txt(args, file_path)

    # Remove matching NaNs
    valid_indices = [
        i for i in range(len(b)) if not pd.isna(b[i]) and not pd.isna(t_s[i])
    ]
    # Filter b and t using the valid indices
    b = [b[i] for i in valid_indices]
    t_s = [t_s[i] for i in valid_indices]
    if io_time:
        console.print("[green]> Calculating overlap metrics[/]")
        io_time = [io_time[i] for i in valid_indices]
        t_e = np.array(t_s) + np.array(io_time)
        #  sort:
        # sorted_indices = np.argsort(t_s)
        # b = b[sorted_indices]
        # t_s = t_s[sorted_indices]
        # t_e = t_e[sorted_indices]
        b, t = overlap(b, t_s, t_e)
    else:
        t = t_s

    # Construct JSON format
    json_data = {
        "write_sync": {
            "total_bytes": total_bytes,
            "number_of_ranks": ranks,
            "bandwidth": {"b_overlap_avr": b, "t_overlap": t},
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
