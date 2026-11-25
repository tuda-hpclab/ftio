"""
Performs prediction with Pools (ProcessPoolExecutor) and a callback mechanism

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import ftio.prediction.monitor as pm
from ftio.prediction.helper import print_data
from ftio.prediction.online_analysis import ftio_process
from ftio.prediction.probability_analysis import find_probability

# from ftio.prediction.async_process import handle_in_process


def predictor_with_pools(shared_resources, args):
    """performs prediction in ProcessPoolExecuter. FTIO is a submitted future and probability is calculated as a callback

    Args:
        filename (str): name of file
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to ftio
    """
    # Init: Monitor a file
    filename = args[1]
    stamp, _ = pm.monitor(filename, "")

    # Loop and predict if changes occur
    try:
        while True:
            with ProcessPoolExecutor(max_workers=1) as executor:
                # monitor
                stamp, _ = pm.monitor(filename, stamp)
                future = executor.submit(ftio_future, shared_resources, args)
                future.add_done_callback(probability_callback)
                shared_resources.count.value += 1
    except KeyboardInterrupt:
        print_data(shared_resources.data)
        print("-- done -- ")


def ftio_future(shared_resources, args: list[str]) -> list[dict]:
    """Performs prediction made up of two part: (1) Executes FTIO and (2) appends to data the value

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to FTIO
    """
    ftio_process(shared_resources, args)
    while not shared_resources.queue.empty():
        shared_resources.data.append(shared_resources.queue.get())
    return shared_resources.data


def probability_callback(future):
    """executes the probability calculation in a callback

    Args:
        future (Future): future containing found frequency in prediction
    """
    _ = find_probability(future.result())
