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


def process_log_files(input_files, output_file, timestamp_column):
    last_timestamp = 0
    with open(output_file, "w") as outfile:
        for input_file in input_files:
            with open(input_file) as infile:
                for line in infile:
                    # Split line by spaces and extract the relevant columns
                    columns = line.split()

                    # Extract the timestamp from the specified column and offset it
                    timestamp = float(columns[timestamp_column])
                    timestamp += last_timestamp

                    # Update the timestamp in the line
                    columns[timestamp_column] = f"{timestamp:.6f}"

                    # Join the columns back into a string and write to the output file
                    outfile.write(" ".join(columns) + "\n")

            # Update the last timestamp to the current line's timestamp
            last_timestamp = timestamp


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process log files and offset timestamps."
    )

    # Add arguments for input files and output file
    parser.add_argument(
        "input_files", nargs="+", help="List of input log files to process"
    )
    parser.add_argument("output_file", help="Output file to store the processed logs")
    parser.add_argument(
        "-t",
        "--timestamp-column",
        type=int,
        default=2,
        help="Column index (zero-based) for timestamps (default: 2)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Process the log files
    process_log_files(args.input_files, args.output_file, args.timestamp_column)
    print(f"Processed log files and saved the result to {args.output_file}")


if __name__ == "__main__":
    main()
