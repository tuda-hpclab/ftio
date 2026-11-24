"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jul 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np

from ftio.cli.ftio_core import core

# from ftio.prediction.helper import get_dominant
# from ftio.plot.freq_plot import convert_and_plot
from ftio.freq.helper import MyConsole
from ftio.parse.args import parse_args
from ftio.processing.print_output import display_prediction

CONSOLE = MyConsole()
CONSOLE.set(True)

# TODO: extend this similar to ftio/prediction/analysis.py


def ftio_metric_task(
    metric: str,
    arrays: np.ndarray,
    argv: list,
    ranks: int = 0,
    show: bool = True,
) -> dict:
    """generate FTIO prediction from 2d np array

    Args:
        metric (str): Name of metric
        arrays (np.ndarray): 2D np array containing the bandwidth and time
        argv (list): List of args for ftio
        ranks (int): Number of ranks
        show (bool): show the prediction result

    Returns:
        dict: prediction from FTIO (see core from ftio.cli.ftio_core)
    """
    if len(arrays[0]) > 1:
        # set up data
        data = {
            "bandwidth": arrays[0],
            "time": arrays[1],
            "total_bytes": 0,
            "ranks": ranks,
        }

        # parse args
        args = parse_args(argv, "ftio")

        # perform prediction
        prediction, analysis_figures = core(data, args)

        # # plot and print info
        # convert_and_plot(args, dfs, len(data))
        if show:
            CONSOLE.info(f"\n[green underline]Metric: {metric}[/]")
            display_prediction(args, prediction)

        if any(x in args.engine for x in ["mat", "plot"]):
            analysis_figures.show()

        return prediction


def ftio_metric_task_save(
    data,
    metric: str,
    arrays: np.ndarray,
    argv: list,
    ranks: int,
    show: bool = False,
) -> None:
    prediction = ftio_metric_task(metric, arrays, argv, ranks, show)
    if prediction is None:
        return
    has_data = (
        len(prediction.dominant_freq) > 0 or len(prediction.top_freqs.get("freq", [])) > 0
    )
    if has_data:
        prediction.metric = metric
        data.append(prediction)
    else:
        CONSOLE.info(f"\n[yellow underline]Warning: {metric} has no periodicity data[/]")
