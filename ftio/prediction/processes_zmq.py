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

import subprocess
import time

import zmq

from ftio.freq.helper import MyConsole
from ftio.multiprocessing.async_process import (
    enforce_limit,
    handle_in_process,
    join_procs,
)
from ftio.parse.args import parse_args
from ftio.prediction.helper import build_reply, export_extrap, print_data
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

    # bind the incoming socket; the reply socket (if any) is opened once here
    socket_in = setup_socket(addr, port_in, zmq.PULL)
    socket_out = None
    if return_data:
        socket_out = setup_socket(addr, tmp_args.zmq_port_reply, zmq.PUSH, False)
    reply_format = getattr(tmp_args, "zmq_reply_format", "msgpack")
    sent = 0  # predictions already replied to

    # can be extended to listen to multiple sockets
    poller = zmq.Poller()
    poller.register(socket_in, zmq.POLLIN)

    if "-zmq" not in args:
        args.extend(["--zmq"])

    try:
        with CONSOLE.status("[green]started\n", spinner="arrow3") as status:
            while True:
                if not debounce:
                    # Original behaviour: reap finished procs, then wait for msgs.
                    procs = join_procs(procs)

                # a prediction finished since the last poll -> PUSH it back
                if return_data and socket_out and len(shared_resources.data) > sent:
                    socket_out.send(build_reply(shared_resources.data[-1], reply_format))
                    sent = len(shared_resources.data)

                # get messages
                msgs, ranks, recv_time = receive_messages(socket_in, poller)

                if not msgs:
                    CONSOLE.print("[red]No messages[/]")
                    continue
                CONSOLE.print(f"[cyan]Got message from {ranks}:[/]")
                # LATENCY line is greppable from job.log: batch size (ranks) vs.
                # wall-clock FTIO spent draining that batch as a single ZMQ PULL
                # sink -- the direct signal for whether rank density degrades
                # FTIO's own response time, independent of prediction cost.
                CONSOLE.print(
                    f"[magenta]LATENCY ranks={ranks} recv_ms={recv_time * 1000:.3f}[/]"
                )
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
            # The gekko clients were already launched with the *old* address, so
            # they now push metrics into a socket nobody drains: FTIO sees a
            # partial stream and the app can stall on the send path.
            CONSOLE.print(
                f"[bold red]WARNING: FTIO moved to {addr}:{port}; already-launched "
                f"GekkoFS clients still target the previous address.[/]"
            )
            socket.bind(f"tcp://{addr}:{port}")
        else:
            socket.connect(f"tcp://{addr}:{port}")

    CONSOLE.print(f"[green]FTIO is running on: {addr}:{port}[/]")

    return socket


def receive_messages(socket, poller):
    """Polls for and receives messages from the socket, returning a list of messages and count.

    Also returns the wall-clock time actually spent receiving messages
    (recv_time) -- FTIO is a single sink for every sender's ZMQ PUSH, so this
    is the direct measure of whether growing rank density degrades FTIO's
    own response time, independent of prediction/processing cost downstream.

    Batching behavior is unchanged from before: each poll(1000) gives a
    straggler rank up to 1s to arrive before the batch is considered done,
    and the loop naturally bounds itself (it only keeps going while messages
    keep arriving within that window). What changed is only what gets
    measured -- recv_time now stops at the last message actually received,
    not at the final poll(1000) that times out with nothing left to receive.
    A prior version measured up to that trailing timeout, so every recv_time
    was inflated by a guaranteed ~1000ms of dead time whenever nothing more
    arrived (the common case) -- it was measuring the poll timeout, not
    receive latency.
    """
    msgs = []
    ranks = 0
    start = time.time()
    socks = dict(poller.poll(1000))
    recv_time = time.time() - start

    while socks:
        if socks.get(socket) == zmq.POLLIN:
            msgs.append(socket.recv(zmq.NOBLOCK))
            ranks += 1
            recv_time = time.time() - start
        socks = dict(poller.poll(1000))

    return msgs, ranks, recv_time


def unbatch_messages(msgs: list[bytes]) -> list[bytes]:
    """Auto-detect and flatten GekkoFS LIBGKFS_METRICS_AGGREGATOR batches.

    A batched message (the aggregator daemon buffers what local ranks sent
    it and forwards one combined message per window) is a single top-level
    msgpack array of raw per-rank payloads. A direct, non-aggregated
    message is a flat sequence of 8-9 top-level scalar/array fields
    starting with flush_t (an int) -- see parse_gekko.assign(). Peeking at
    just the first top-level object's type tells them apart, so this needs
    no flag and works per-message even if a run somehow mixed both (it
    shouldn't, but nothing here assumes it can't).
    """
    import msgpack

    out: list[bytes] = []
    for m in msgs:
        unpacker = msgpack.Unpacker(raw=True)
        unpacker.feed(m)
        first = next(unpacker, None)
        if isinstance(first, list):
            out.extend(first)
        else:
            out.append(m)
    return out


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
