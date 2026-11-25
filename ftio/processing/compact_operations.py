"""
This module contains operations for processing and plotting data using ftio.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from rich.console import Console

from ftio.cli.ftio_core import core
from ftio.freq.prediction import Prediction
from ftio.parse.args import parse_args
from ftio.processing.print_output import display_prediction

console = Console()


def quick_ftio(
    argv: list,
    b: list,
    t: list,
    total_bytes: int,
    ranks: int,
    msg: str = "",
    verbose: bool = True,
) -> Prediction:
    """Quickly process and plot data using ftio.

    Args:
        argv (list): Command line arguments.
        b (list): Bandwidth data.
        t (list): Time data.
        total_bytes (int): Total bytes data.
        ranks (int): MPI Ranks.
        msg (str): Message to display.
        verbose (bool, optional): Whether to display verbose output. Defaults to True.

    Returns:
        dict: Prediction results.
    """
    # set up data
    data = {
        "time": t,
        "bandwidth": b,
        "total_bytes": total_bytes,
        "ranks": ranks,
    }

    # parse args
    args = parse_args(argv, "ftio")

    # perform prediction
    prediction, analysis_figures = core(data, args)

    # plot and print info
    analysis_figures.show()
    if verbose and len(msg) > 0:
        console.print(f"[green]Prediction for {msg}[/]")
        display_prediction(args, prediction)

    return prediction
