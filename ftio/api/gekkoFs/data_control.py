"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import os
from multiprocessing import Event, Pool, Process
from time import sleep

import zmq
from rich.console import Console

from ftio.api.gekkoFs.jit.execute_and_wait import execute_block, get_files
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.setup_helper import flaged_call


class DataControl:
    """
    A class to control the processing of file data using ZeroMQ for communication.

    This class listens for the start signal from the main process, finds all files in the directory,
    and prints their modification timestamps to the console.
    """

    def __init__(self, settings: JitSettings) -> None:
        """
        Initialize the DataControl object with a specific address, port, and directory for file searching.

        Args:
            address (str): The address to listen on.
            port (int): The port to listen on.
            settings (str): Jit settings
        """
        self.address = settings.address_cargo
        self.port = settings.port_cargo
        self.settings = settings
        self.console = Console()  # Rich Console for fancy printing
        # Start the worker process
        self.stop_event = Event()
        self.worker_proc = Process(target=self.start)
        self.worker_proc.start()

    def start(self) -> None:
        """
        Start the ZeroMQ server to listen for incoming signals. Once a signal is received,
        the worker will find files in the specified directory and print their modification timestamps.
        """
        context = zmq.Context()

        # Create a PUSH socket (no reply expected)
        socket = context.socket(zmq.PUSH)
        socket.bind(f"tcp://{self.address}:{self.port}")
        self.console.print(
            f"[DataControl] Listening on tcp://{self.address}:{self.port}...",
            style="bold green",
        )

        # Create a poller and register the socket to watch for incoming messages
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            # Wait for events (like messages) on the socket
            events = dict(poller.poll(timeout=100))  # Timeout in milliseconds (100ms)

            # Check if there is an event on the socket
            if socket in events and events[socket] == zmq.POLLIN:
                # If there is an event (message), receive it
                signal = socket.recv_string()
                self.console.print(
                    f"[DataControl] Received signal: {signal}",
                    style="bold yellow",
                )

                if signal == "START":
                    # If the signal is "START", process files in the directory
                    self.process_files()

            sleep(
                0.1
            )  # Add sleep to avoid tight loop, can be adjusted or omitted based on your needs

    def move_file(self, file: str, counter: int, monitored_files: list) -> None:
        """
        Move a single file and print its progress.

        Args:
            file (str): The file to move.
            counter (int): The current counter of processed files.
            monitored_files (list): List of all files being processed.
        """
        try:
            full_path = os.path.join(self.settings.gkfs_mntdir, file)
            # Prepare the command for moving the file
            # 1) get time now:
            now = gkfs_call(self.settings, "date +%s")

            # 2) check the time:
            out_time = gkfs_call(self.settings, f"stat -c %Y {full_path}")

            # 3) move the file
            if int(out_time) - int(now) > 5:
                _ = gkfs_call(
                    self.settings,
                    f"mv  {full_path} {self.settings.stage_out_path}/file",
                )

            counter += 1
            self.console.print(
                f"[bold green]Finished moving ({counter}/{len(monitored_files)}): {file}[/]"
            )

        except Exception as e:
            self.console.print(f"[Error] Error moving file {file}: {e}", style="bold red")

    def process_files(self) -> None:
        """
        Find files in the directory using 'ls -R', and move them in parallel using multiprocessing.
        """
        # Use 'ls -R' to get all files in the directory recursively
        monitored_files = get_files(self.settings, False)
        counter = 0

        try:
            # Create a pool of workers to process the files in parallel
            with Pool(processes=2) as pool:  # Use 2 processes
                # Distribute the files and pass the counter and monitored files for progress tracking
                results = [
                    pool.apply_async(self.move_file, (file, counter, monitored_files))
                    for file in monitored_files
                ]

                # Wait for all processes to finish
                for result in results:
                    result.wait()

        except Exception as e:
            self.console.print(f"[Error] Error processing files: {e}", style="bold red")

    def __del__(self) -> None:
        """
        Stop the worker process by setting the stop event and terminating the process.
        """
        self.stop_event.set()  # Set the stop event to signal the worker to stop
        self.worker_proc.join()  # Wait for the worker process to terminate
        self.console.print("[DataControl] Worker process stopped.", style="bold green")


def trigger_data_controller(address: str = "127.0.0.1", port: str = "65432") -> None:
    """
    Trigger the worker process to start processing files. The main process sends a 'START' signal
    to the worker, and the worker will scan the directory for files, print their modification timestamps,
    and then continue.

    Args:
        address (str): The address of the worker.
        port (str): The port to connect to.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # Use REQ (Request) for sending requests
    socket.connect(f"tcp://{address}:{port}")
    socket.send_string("START")
    socket.close()  # Close the socket after use


# data_control = DataControl(address, port, settings)


def gkfs_call(settings: JitSettings, call: str):
    call = flaged_call(
        settings,
        call,
        exclude=["ftio", "cargo"],
    )
    out = execute_block(call, dry_run=settings.dry_run)

    return out
