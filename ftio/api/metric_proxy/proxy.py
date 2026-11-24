"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.api.metric_proxy.parse_proxy import parse
from ftio.cli.ftio_core import core
from ftio.freq.helper import MyConsole
from ftio.parse.args import parse_args
from ftio.processing.post_processing import label_phases
from ftio.processing.print_output import display_prediction

CONSOLE = MyConsole()
CONSOLE.set(True)

# ---------------------------------
# Modification Area
# ---------------------------------
b, t = parse("/d/sim/metric_proxy/traces/Mixed_1x8_5.json", "total___mpi___size_total")
# b, t = parse("/d/sim/metric_proxy/traces/Mixed_1x8_5.json", "mpi___size___mpi_allgather")
ranks = 32

# command line arguments
argv = ["-e", "plotly"]  # ["-e", "no"] to disable the plot
argv.extend(
    ["-n", "2"]
)  # finds up to n frequencies. Comment this out to go back to the default version
# ---------------------------------

# set up data
data = {"time": t, "bandwidth": b, "total_bytes": 0, "ranks": ranks}

# parse args
args = parse_args(argv, "ftio")

# perform prediction
prediction, analysis_figures = core(data, args)

# plot and print info
display_prediction(args, prediction)
analysis_figures.show()

# ------------------

# Post processing
if prediction and len(prediction["dominant_freq"]) != 0:
    phases, time = label_phases(prediction, args, b, t)
    CONSOLE.print(
        f"[cyan]Phases[/]: {phases}\n[cyan]Start time[/]: {time['t_s']}\n[cyan]End time[/]: {time['t_e']}\n"
    )
