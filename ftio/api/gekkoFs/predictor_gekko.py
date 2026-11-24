"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import glob

from rich.console import Console

import ftio.prediction.monitor as pm
from ftio.api.gekkoFs.ftio_gekko import run
from ftio.multiprocessing.async_process import handle_in_process
from ftio.prediction.helper import export_extrap, get_dominant, print_data
from ftio.prediction.online_analysis import (
    display_result,
    save_data,
    window_adaptation,
)
from ftio.prediction.probability_analysis import find_probability
from ftio.prediction.shared_resources import SharedResources


def main(args: list[str] = None) -> None:
    """
    Main function to monitor files and launch prediction processes.

    Args:
        args (list[str], optional): List of arguments for the prediction process. Defaults to an empty list.

    Initializes shared resources and monitors specified files for changes. When changes are detected,
    it launches a prediction process with the provided arguments. The function runs indefinitely until
    interrupted by a KeyboardInterrupt, at which point it prints and exports the collected data.

    Raises:
        KeyboardInterrupt: When the function is interrupted by the user.
    """

    if args is None:
        args = []
    n_buffers = 4
    args = ["-e", "plotly", "-f", "0.01"]
    # path=r'/d/github/FTIO/examples/API/gekkoFs/JSON/*.json'
    path = r"/d/github/FTIO/examples/API/gekkoFs/MSGPACK/write*.msgpack"
    # path = r"/tmp/gkfs_client_metrics/write*.msgpack"
    matched_files = glob.glob(path)

    # Init
    shared_resources = SharedResources()
    procs = []

    # Init: Monitor a file
    stamp, _ = pm.monitor_list(matched_files, n_buffers)

    # Loop and predict if changes occur
    try:
        while True:
            # monitor
            stamp, procs = pm.monitor_list(matched_files, n_buffers, stamp, procs)

            # launch prediction_process
            procs.append(
                handle_in_process(
                    prediction_process,
                    args=(
                        shared_resources,
                        args,
                        matched_files,
                        n_buffers,
                    ),
                )
            )
    except KeyboardInterrupt:
        print_data(shared_resources.data)
        export_extrap(shared_resources.data)
        print("-- done -- ")


def prediction_process(
    shared_resources: SharedResources,
    args: list[str],
    matched_files: list[str],
    n_buffers: int,
) -> None:
    """
    Executes the prediction process using the provided shared resources, arguments, and matched files.

    Args:
        shared_resources (SharedResources): An instance containing shared resources and data.
        args (list[str]): A list of arguments to be modified and used in the prediction process.
        matched_files (list[str]): A list of file paths that match the criteria for prediction.
        n_buffers (int): The number of buffers expected to match the number of files.

    Raises:
        RuntimeError: If the number of buffers does not match the number of files.

    Returns:
        None
    """
    console = Console()
    console.print(f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/]  Started")
    # Modify the arguments
    args.extend(["-e", "no"])
    args.extend(["-ts", f"{shared_resources.start_time.value:.2f}"])

    # set up data
    if len(matched_files) != n_buffers:
        raise RuntimeError("Error, number of buffers does not match number of files")

    # Perform prediction
    prediction, parsed_args, _ = run(matched_files, args)

    # get data
    freq = get_dominant(prediction)  # just get a single dominant value
    # save prediction results
    save_data(prediction, shared_resources)
    # display results
    text = display_result(freq, prediction, shared_resources=shared_resources)
    # data analysis to decrease window
    adaptation_text, _, _ = window_adaptation(
        parsed_args, prediction, freq, shared_resources
    )
    text += adaptation_text
    console.print(text)
    while not shared_resources.queue.empty():
        shared_resources.data.append(shared_resources.queue.get())

    _ = find_probability(shared_resources.data, counter=shared_resources.count.value)
    shared_resources.count.value += 1


if __name__ == "__main__":
    main()
