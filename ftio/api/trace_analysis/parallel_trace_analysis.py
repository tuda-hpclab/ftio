"""
Parallel Trace Analysis Module

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ftio.api.trace_analysis.trace_analysis import (
    convert_dict,
    flatten_dict,
    statistics,
)
from ftio.api.trace_analysis.trace_ftio_v2 import main as trace_ftio

# Initialize the console for printing
console = Console()


def process_file(file_path: str, argv: list, settings: dict, index: int = 0) -> tuple:
    """
    Process a single file using the trace_ftio function and save the results.

    Args:
        file_path (str): Path to the file to be processed.
        argv (list): List of additional arguments.
        settings (dict): Dictionary containing settings for processing.
        index (int): Index of the file in the list of files.

    Returns:
        tuple: A tuple containing the flattened result (dict or None), index (int), file path (str), and error message (str).
    """
    error = ""
    try:
        # Call your trace_ftio function (adjust the import and call as necessary)
        res = trace_ftio([file_path] + argv, settings["verbose"], settings["json"])

        # Create the new file name by replacing the pattern
        base_name = os.path.basename(file_path)

        # if input file is not a json, save ftio result
        if not settings["json"] and settings["save"]:
            json_file = base_name.replace(
                f"_signal_{settings['name']}.csv",
                f"_freq_{settings['name']}.json",
            )
            json_path = os.path.join(os.path.dirname(file_path), json_file)
            # Convert NumPy arrays to lists and save the ftio results
            data_converted = convert_dict(res)
            if json_file.endswith("json"):
                with open(json_path, "w") as file:
                    json.dump(data_converted, file, indent=4)
            else:
                console.print(f"[bold red]Cannot dump Json file in {json_path}[/]")

        flat_res = flatten_dict(res)
        try:
            if settings["json"]:
                flat_res["job_id"] = base_name.removesuffix(".json")
            else:
                flat_res["job_id"] = base_name.split("_")[0]
            flat_res["file"] = file_path
        except Exception:
            flat_res["job_id"] = "??"
            flat_res["file"] = "??"
            console.print("[bold red]Unable to extract job id[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error processing file {file_path}: {e}[/]")
        error = str(e)
        flat_res = None

    return flat_res, index, file_path, error


def main(argv: list = sys.argv[1:]) -> None:
    """
    Main function to process multiple files in parallel using multiprocessing.

    Args:
        argv (list): List of command-line arguments.
    """
    settings = {
        "verbose": False,
        "json": False,
        "save": False,
        "name": "plafrim",
        "num_procs": -1,
        "res_path": ".",
        "freq": 10,
        "folder_path": "",
    }

    # Specify the name with -n
    if "-p" in argv:
        index = argv.index("-p")
        settings["num_procs"] = int(argv[index + 1])
        argv.pop(index)
        argv.pop(index)
    if "-n" in argv:
        index = argv.index("-n")
        settings["name"] = str(argv[index + 1])
        argv.pop(index)
        argv.pop(index)
    if "-v" in argv:
        index = argv.index("-v")
        argv.pop(index)
        settings["verbose"] = True
    if "-j" in argv:
        index = argv.index("-j")
        argv.pop(index)
        settings["json"] = True
    if "-s" in argv:
        index = argv.index("-s")
        argv.pop(index)
        settings["save"] = True
    if "-o" in argv:
        index = argv.index("-o")
        settings["res_path"] = str(argv[index + 1])
        argv.pop(index)
        argv.pop(index)
    if "-h" in argv:
        console.print(
            "Usage:  parallel_trace_analysis  <dir>\n\n"
            "-n <str>: filter according if they contain the name\n"
            "--time-step <float>: specifies the value of the implicit time steps between the samples\n"
            "-o <str>: Output dir location\n"
            "-j <bool>: Enables JSON search\n"
            "-v <bool>: verbose\n"
            "-s <bool>: save the FTIO result for each file in a file that contains the name _freq_\n"
            "All ftio options (see ftio -h)\n\n"
        )
        sys.exit()
    if "-f" in argv:  # save the freq
        index = argv.index("-f")
        settings["freq"] = str(argv[index + 1])

    start_time = time.time()
    pattern = f"_signal_{settings['name']}.csv"
    df = pd.DataFrame()
    if settings["json"]:
        pattern = ".json"

    # Handle command-line arguments properly
    if len(argv) > 0:
        folder_path = argv[0]
        argv.pop(0)
    else:
        folder_path = "/d/github/FTIO/ftio/api/trace_analysis/plafrim"

    folder_path = os.path.abspath(folder_path)
    settings["folder_path"] = folder_path  # log this
    console.print(f"[bold green]Path is: {folder_path}[/]")
    # Create a list to store all matching csv files
    trace_files = []
    # Walk through all subdirectories and find matching files
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(pattern) and "_freq_" not in filename:
                trace_files.append(os.path.join(dirpath, filename))

    if not trace_files:
        console.print(f"[bold red]No files matched the pattern: {pattern}![/]")
        return

    total_files = len(trace_files)
    # Create a progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description} ({task.completed}/{task.total})"
        ),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        "[yellow]-- runtime",
        TimeElapsedColumn(),
    )

    try:
        with progress:
            # Use multiprocessing Pool
            if settings["num_procs"] == -1:
                settings["num_procs"] = int(cpu_count() / 2)

            console.print(f"[bold green]Using {settings['num_procs']} processes[/]\n")
            task = progress.add_task("[green]Processing files", total=total_files)

            # List to store failed file details
            failed_files = []

            # use future or apply_async
            future = True  # managing tasks without blocking others due to slow processes

            if future:
                counter = 0
                with ProcessPoolExecutor(max_workers=settings["num_procs"]) as executor:
                    # Submit tasks to the executor
                    futures = {
                        executor.submit(process_file, file_path, argv, settings, i): i
                        for i, file_path in enumerate(trace_files)
                    }

                    # Process results as they complete
                    for future in as_completed(futures):
                        # index = futures[future]
                        try:
                            flat_res, index, file_path, error = future.result(timeout=120)
                        except TimeoutError:
                            index = futures[future]
                            flat_res = None
                            error = f"[red bold] {trace_files[index]} reached timeout"
                        if flat_res:  # Process result
                            df = pd.concat(
                                [df, pd.DataFrame([flat_res])],
                                ignore_index=True,
                            )
                        if error:  # Log any failed files
                            failed_files.append((file_path, error))
                        # Update progress
                        progress.console.print(
                            f"Processed ({index + 1}/{len(trace_files)}): {trace_files[index]}"
                        )
                        counter += 1
                        progress.update(task, completed=counter)
            else:
                with Pool(processes=settings["num_procs"]) as pool:
                    # Pass the index and total files to process_file
                    results = [
                        pool.apply_async(process_file, (file_path, argv, settings, index))
                        for i, file_path in enumerate(trace_files)
                    ]

                    for result in results:
                        flat_res, index, file_path, error = result.get()
                        if flat_res:  # Process result
                            df = pd.concat(
                                [df, pd.DataFrame([flat_res])],
                                ignore_index=True,
                            )
                        if error:  # Log any failed files
                            failed_files.append((file_path, error))

                        # Update progress
                        progress.console.print(
                            f"Processed ({index + 1}/{total_files}): {trace_files[index]}"
                        )
                        progress.update(task, completed=index + 1)

        # After processing, log the files that failed
        if failed_files:
            progress.console.print(
                f"\n[bold]{total_files} files processed: [/]\n"
                f" - [bold green]{total_files - len(failed_files)} files processed successfully![/]\n"
                f" - [bold red]{len(failed_files)} files failed[/]\n"
            )
            console.print("\n[bold yellow]The following files failed to process:[/]")
            with open(f"{settings['res_path']}/log.err", "w") as log:
                for file_path, error in failed_files:
                    console.print(f"[bold red]{file_path}[/]")
                    log.write(f"{file_path}: {error}\n")
        else:
            progress.console.print("\n[bold green]All files processed successfully![/]\n")

        console.print(
            f"[blue]FTIO total time:[/] {time.time() - start_time:.4f} seconds\n"
            f"[blue]Location:[/] {folder_path}\n"
            f"[blue]Pattern:[/] {pattern}\n"
        )
        elapsed_time = f"Execution time {time.time() - start_time:.4f} seconds"
        statistics(df, elapsed_time, settings)
    except KeyboardInterrupt:
        progress.console.print("[bold red]Keyboard interrupt![/]\n")
        statistics(df, "", settings)
        sys.exit()
    console.print(f"[blue]Execution time:[/] {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    """
    Entry point for the script. Processes command-line arguments and calls the main function.
    """
    main(sys.argv[1:])
