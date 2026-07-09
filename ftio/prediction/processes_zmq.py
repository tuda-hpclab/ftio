"""
Performs prediction with Pools (ProcessPoolExecutor) and a callback mechanism

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import struct
import subprocess

import zmq

from ftio.freq.helper import MyConsole
from ftio.multiprocessing.async_process import (
    enforce_limit,
    handle_in_process,
    join_procs,
)
from ftio.parse.args import parse_args
from ftio.prediction.helper import export_extrap, get_dominant_and_conf, print_data
from ftio.prediction.processes import prediction_process

CONSOLE = MyConsole()
CONSOLE.set(True)


def predictor_with_processes_zmq(
    shared_resources,
    args,
    return_data: bool = False,
) -> None:
    """Monitors a ZMQ socket and runs predictions whenever messages arrive.

    Two modes are supported, selected by the ``--debounce`` flag in *args*:

    * **Default (parallel)** — original behaviour: a new prediction process is
      spawned for every batch of received messages regardless of whether a
      previous prediction is still running.

    * **Debounce (serial, --debounce)** — only one prediction runs at a time.
      After a prediction finishes, another poll is performed immediately so
      that messages that accumulated during the prediction are not silently
      dropped — they trigger a follow-up prediction right away.

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to ftio
    """
    procs = []
    # parse arguments
    tmp_args = parse_args(args)
    addr = tmp_args.zmq_address
    port_in = tmp_args.zmq_port
    debounce = getattr(tmp_args, "debounce", False)
    max_predictions = getattr(tmp_args, "max_predictions", 0)

    # bind the socket
    socket_in = setup_socket(addr, port_in, zmq.PULL)
    socket_out = None

    if return_data:
        port_out = tmp_args.zmq_port_reply
        socket_out = setup_socket(addr, port_out, zmq.PUSH, False)

    # can be extended to listen to multiple sockets
    poller = zmq.Poller()
    poller.register(socket_in, zmq.POLLIN)

    if "-zmq" not in args:
        args.extend(["--zmq"])

    try:
        with CONSOLE.status("[green]started\n", spinner="arrow3") as status:
            while True:
                pre_num_procs = len(procs)
                if not debounce:
                    # Original behaviour: reap finished procs, then wait for msgs.
                    procs = join_procs(procs)

                if return_data and socket_out and pre_num_procs > len(procs):
                    CONSOLE.print("[cyan]Returning Results[/]")
                    data = get_dominant_and_conf(shared_resources.data[-1])
                    CONSOLE.print(f"[cyan]Sending Frequency:{data[0]}[/]")
                    CONSOLE.print(f"[cyan]Sending Confidence:{data[1]}[/]")
                    packet = struct.pack("dd", data[0], data[1])
                    socket_out.send(packet)

                # get messages
                msgs, ranks = receive_messages(socket_in, poller)

                if not msgs:
                    CONSOLE.print("[red]No messages[/]")
                    continue
                CONSOLE.print(f"[cyan]Got message from {ranks}:[/]")
                status.update("")

                if debounce:
                    # Serial: run one prediction and wait before the next poll.
                    proc = handle_in_process(
                        prediction_process, args=(shared_resources, args, msgs)
                    )
                    proc.join()
                else:
                    # bounded pool: wait for the oldest if the cap is reached
                    procs = enforce_limit(procs, max_predictions)
                    procs.append(
                        handle_in_process(
                            prediction_process, args=(shared_resources, args, msgs)
                        )
                    )
    except KeyboardInterrupt:
        print_data(shared_resources.data)
        export_extrap(shared_resources.data)
        print("-- done -- ")


def setup_socket(addr: str, port: str, socket_type=zmq.PULL, bind: bool = True):
    """Bind the ZMQ socket, retrying with a corrected IP if necessary."""
    context = zmq.Context()
    socket = context.socket(socket_type)
    if not bind and addr == "*":
        addr = "127.0.0.1"
    try:
        if bind:
            socket.bind(f"tcp://{addr}:{port}")
        else:
            socket.connect(f"tcp://{addr}:{port}")
    except zmq.error.ZMQError as e:
        CONSOLE.print(f"[yellow]Error encountered:\n{e}[/]")
        CONSOLE.print("[yellow]Wrong IP address. Attempting to correct...[/]")
        # addr = str(
        #     subprocess.check_output(
        #         "ip addr | grep 'inet 10' | awk  '{print $2}'", shell=True
        #     )
        # )
        # end = addr.rfind("/")
        # start = addr.find("'")
        # addr = addr[start + 1 : end]
        # CONSOLE.print("[bold green]Corrected IP address:[/]", addr)
        # socket.bind(f"tcp://{addr}:{port}")
        output = subprocess.check_output(
            "ip addr | grep 'inet 10' | awk '{print $2}'",
            shell=True,
            text=True,  # returns str instead of bytes
        )

        # Take first matching address
        addr = output.splitlines()[0].split("/")[0]

        CONSOLE.print("[bold green]Corrected IP address:[/]", addr)
        if bind:
            socket.bind(f"tcp://{addr}:{port}")
        else:
            socket.connect(f"tcp://{addr}:{port}")

    CONSOLE.print(f"[green]FTIO is running on: {addr}:{port}[/]")

    return socket


def receive_messages(socket, poller):
    """Polls for and receives messages from the socket, returning a list of messages and count."""
    msgs = []
    ranks = 0
    # start = time.time()
    socks = dict(poller.poll(1000))
    # 1) just a single msg
    # msg = socket.recv(zmq.NOBLOCK)

    # 2) Loop and accept messages from both channels, acting accordingly
    # if socks:
    #     if socks.get(socket) == zmq.POLLIN:
    #         print(f"got message ",{socket.recv(zmq.NOBLOCK)})
    # else:
    #     print("No message received")
    #     continue

    # 3) Loop and accept messages from both channels, acting accordingly
    while socks:
        if socks.get(socket) == zmq.POLLIN:
            msgs.append(socket.recv(zmq.NOBLOCK))
            # CONSOLE.print(f"[cyan]Got message {ranks}:[/] {msg}")
            ranks += 1
        # if time.time() - start > 0.5:
        #     break
        socks = dict(poller.poll(1000))

    return msgs, ranks


#
# def receive_messages(socket, poller, timeout=1000):
#     """Receive all pending messages from a ZMQ socket safely, including large messages."""
#     msgs = []
#     ranks = 0
#
#     while True:
#         socks = dict(poller.poll(timeout))
#         if socket not in socks or socks[socket] != zmq.POLLIN:
#             break  # no more messages ready
#
#         try:
#             msg = socket.recv(zmq.NOBLOCK)  # or recv_multipart() if needed
#             msgs.append(msg)
#             ranks += 1
#         except zmq.Again:
#             break  # no more messages currently available
#
#         # After first poll, switch to non-blocking poll to empty the queue
#         timeout = 0
#
#     return msgs, ranks
#
