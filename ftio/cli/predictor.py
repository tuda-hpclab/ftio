"""
This functions performs the prediction with the the help of ftio.py

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import sys

from ftio.gui.socket_logger import init_socket_logger
from ftio.parse.helper import print_info
from ftio.prediction.pools import predictor_with_pools
from ftio.prediction.processes import predictor_with_processes
from ftio.prediction.processes_zmq import predictor_with_processes_zmq
from ftio.prediction.shared_resources import SharedResources


def main(args: list[str] = sys.argv) -> None:
    """runs the prediction and launches new threads whenever data is available

    Args:
        args (list[str]): arguments passed from command line
    """
    # Init
    print_info("Predictor", False)
    shared_resources = SharedResources()
    mode = "procs"  # "procs" or "pool"

    # Initialize GUI socket logger if --gui flag is present

    if "gui" in args:
        init_socket_logger(True)
        print("[INFO] GUI mode enabled - forwarding predictions to ftio-gui dashboard")

    if "pool" in mode.lower():
        # prediction with a Pool of process and a callback mechanism
        predictor_with_pools(shared_resources, args)
    else:
        if any("zmq" in x for x in args):
            # prediction with Processes of process and a callback mechanism + zmq
            predictor_with_processes_zmq(
                shared_resources,
                args,
                any("zmq_port_reply" in x.replace("-", "_") for x in args),
            )
        else:
            # prediction with processes and a callback mechanism
            predictor_with_processes(shared_resources, args)


if __name__ == "__main__":
    main(sys.argv)
