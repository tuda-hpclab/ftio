"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import datetime

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ftio import __version__


def scale_metric(metric: str, number: float) -> tuple[str, float]:
    """set unit for the plots

    Args:
        number (np.ndarray): array

    Returns:
        unit (string): unit in GB/s, MB/s, KB/s or B/s
        number: scaled array according to the unit
    """
    unit = metric
    order = 1e-0
    prefix = ""
    if number > 0:
        order = np.log10(number)

        if any(x in metric.lower() for x in ["bytes", "b", "bandwidth", "transfer"]):
            if order > 9:
                order = 1e-9
                prefix = "G"
            elif order > 6:
                order = 1e-6
                prefix = "M"
            elif order > 3:
                order = 1e-3
                prefix = "K"
            else:
                order = 1e-0
                prefix = ""
        elif any(x in metric.lower() for x in ["time", "(s)"]):
            if order > 0:
                order = 1e-0
                prefix = ""
            elif order > -3:
                order = 1e-3
                prefix = "μ"
            else:
                order = np.nan
        else:
            return metric, 1e-0

        if not np.isnan(order):
            if any(
                x in metric.lower()
                for x in [
                    "(b/s)",
                    "(bytes/s)",
                    "(bytes/second)",
                    "(b/second)",
                    "bandwidth",
                ]
            ):
                unit = f"Bandwidth ({prefix}B/s)"
            elif any(x in metric.lower() for x in ["(bytes)", "(b)"]):
                unit = f"Size ({prefix}B)"
            elif any(x in metric.lower() for x in ["(s)", "(second)", "time"]):
                unit = f"Time ({prefix}s)"
            else:
                unit = "UNKOWN"
        else:
            unit = "UNKOWN"
            order = 1e-0

    return unit, order


def match_modes(mode):
    mode = mode.lower()
    if isinstance(mode, list):
        for i in range(0, len(mode)):
            mode[i] = match_mode(mode[i])
    else:
        mode = [match_mode(mode)]
    return mode


def match_mode(mode: str) -> str:
    out = ""
    if "w" in mode:
        out = "write"
    else:
        out = "read"

    if "async" in mode:
        out = out + "_async"
    else:
        out = out + "_sync"

    return out


def detect_source(data: dict, args) -> str:
    if "tmio" in args.source.lower() or "custom" in args.source.lower():
        return args.source.lower()
    else:  # autodetect
        tmio_fields = [
            "read_sync",
            "read_async_t",
            "read_async_b",
            "write_async_t",
            "io_time",
        ]
        if all(fields in data for fields in tmio_fields):
            return "tmio"
        else:
            return "unspecified"


def print_info(prog_name: str, flag=True) -> None:
    console = Console()
    if "ftio" in prog_name.lower():
        color = "cyan"
    elif "plot" in prog_name.lower():
        color = "yellow"
    elif "parse" in prog_name.lower():
        color = "green"
    elif "predictor" in prog_name.lower():
        if flag:
            return
        else:
            color = "dark_violet"
    else:
        color = "black"

    title = Panel(
        Text(prog_name, justify="center"),
        style=f"bold white on {color}",
        border_style="white",
        title_align="left",
    )

    # color = cyan
    text = f"\n[{color}]Author:[/]  Ahmad Tarraf\n"
    text += f"[{color}]Date:[/]    {str(datetime.date.today())}\n"
    text += f"[{color}]Version:[/]  {__version__}\n"
    text += f"[{color}]License:[/]  BSD\n"
    console.print(title)
    console.print(text)
