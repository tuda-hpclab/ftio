"""
This module provides functionality for generating online FTIO predictions and triggering
Cargo for data staging. It includes processes for handling ZMQ messages, performing
predictions, and managing shared resources.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Nov 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
import sys
import time

import numpy as np

# import numpy as np
import zmq
from rich.console import Console

from ftio.api.gekkoFs.ftio_gekko import run
from ftio.api.gekkoFs.stage_data import (
    parse_args_data_stager,
    setup_cargo,
    trigger_flush,
)
from ftio.freq.helper import MyConsole
from ftio.multiprocessing.async_process import (
    enforce_limit,
    handle_in_process,
    join_procs,
)
from ftio.plot.plot_bandwidth import plot_bar_with_rich
from ftio.prediction.helper import (
    get_dominant_and_conf,
    print_data,  # , export_extrap
)
from ftio.prediction.online_analysis import (
    display_result,
    save_data,
    window_adaptation,
)
from ftio.prediction.probability_analysis import find_probability
from ftio.prediction.processes_zmq import receive_messages, setup_socket
from ftio.prediction.shared_resources import SharedResources

# T_S = time.time()
CONSOLE = MyConsole()
CONSOLE.set(True)


def main(args: list[str] = sys.argv[1:]) -> None:
    """generates online FTIO predictions and triggers cargo

    Args:
        args (_type_, optional): FTIO arguments, see 'ftio -h'.
    """
    # parse arguments
    data_stager_args, ftio_args = parse_args_data_stager(args, False)
    ranks = 0
    procs = []
    debounce = "--debounce" in ftio_args
    # ftio_args is a raw list here; read the bounded-pool cap from it
    max_predictions = 0
    if "--max-predictions" in ftio_args:
        idx = ftio_args.index("--max-predictions")
        if idx + 1 < len(ftio_args):
            max_predictions = int(ftio_args[idx + 1])

    ftio_args.extend(["-e", "no"])
    if "-zmq" not in ftio_args:
        ftio_args.extend(["--zmq"])
    # args.extend(['-e', 'no', '-f', '10', '-m', 'write','-v'])

    # start cargo
    setup_cargo(data_stager_args)

    # bind to socket
    socket = setup_socket(data_stager_args.zmq_address, data_stager_args.zmq_port)
    # can be extended to listen to multiple sockets
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    # # Init
    shared_resources = SharedResources()

    # for Cargo trigger process:
    trigger = handle_in_process(
        trigger_flush,
        args=(shared_resources.sync_trigger, data_stager_args),
    )

    # Loop and predict if changes occur
    try:
        with CONSOLE.status("[green]started\n", spinner="arrow3") as status:
            while True:
                procs = join_procs(procs, False)

                # get all messages
                msgs, ranks = receive_messages(socket, poller)

                if not msgs:
                    # CONSOLE.print('[red]No messages[/]')
                    status.update("[cyan]Waiting for messages\n", spinner="dots")
                    continue

                # Hold predictions until the application has started
                if data_stager_args.app_start_file and not os.path.isfile(
                    data_stager_args.app_start_file
                ):
                    status.update(
                        "[yellow]Holding predictions — waiting for application to start...\n",
                        spinner="dots",
                    )
                    continue

                CONSOLE.print(f"[cyan]Got message from {ranks}:[/]")
                status.update("")

                # launch prediction_process
                if debounce:
                    if procs:
                        procs[-1].join()
                        procs.pop()
                    procs.append(
                        handle_in_process(
                            prediction_zmq_process,
                            args=(shared_resources, ftio_args, msgs),
                        )
                    )
                else:
                    # bounded pool: wait for the oldest if the cap is reached
                    procs = enforce_limit(procs, max_predictions)
                    procs.append(
                        handle_in_process(
                            prediction_zmq_process,
                            args=(shared_resources, ftio_args, msgs),
                        )
                    )

    except KeyboardInterrupt:
        trigger.join()
        print_data(shared_resources.data)
        # export_extrap(data=data)
        print("-- done -- ")


def prediction_zmq_process(
    shared_resources: SharedResources, args: list[str], msg
) -> None:
    """performs prediction

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to ftio.py
        msg: zmq message
    """
    t_prediction = time.time()
    console = Console()
    console.print(f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/]  Started")

    # Modify the arguments
    args.extend(["-ts", f"{shared_resources.start_time.value:.2f}"])

    # Perform prediction
    prediction, parsed_args, t_flush = run(
        msg, args, shared_resources.b_app, shared_resources.t_app
    )
    shared_resources.t_flush.append(t_flush)

    # plot
    plot_bar_with_rich(
        shared_resources.t_app, shared_resources.b_app, width_percentage=0.8
    )

    # get data
    freq, conf = get_dominant_and_conf(prediction)  # just get a single dominant value
    # save prediction results
    save_data(prediction, shared_resources)
    # display results
    text = display_result(freq, prediction, shared_resources)
    # data analysis to decrease window thus change start_time
    adaptation_text, _, _ = window_adaptation(
        parsed_args, prediction, freq, shared_resources
    )
    text += adaptation_text
    # print text
    console.print(text)

    # save data to queue
    while not shared_resources.queue.empty():
        shared_resources.data.append(shared_resources.queue.get())

    # calculate probability
    prob = find_probability(shared_resources.data, counter=shared_resources.count.value)

    probability = -1
    for p in prob:
        if p.get_freq_prob(freq):
            probability = p.p_freq_given_periodic
            break

    # total bytes and average bytes per phase
    if not np.isnan(freq):
        total_bytes = prediction.total_bytes
    else:
        total_bytes = 0

    # send data to trigger proc
    console.print(
        f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Added data to trigger queue"
    )
    shared_resources.sync_trigger.put(
        {
            "t_wait": time.time(),
            "t_end": prediction.t_end,
            "t_start": prediction.t_start,
            "t_flush": t_flush + (t_prediction - time.time()),
            "freq": freq,
            "conf": conf,
            "probability": probability,
            "total_bytes": total_bytes,
            "source": f"#{shared_resources.count.value}",
        }
    )

    console.print(f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Ended")
    shared_resources.count.value += 1  # proc-safe, as manager already handles this


if __name__ == "__main__":
    main(sys.argv)
