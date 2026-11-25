"""
Helper function for frequency techniques

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
from rich.console import Console

from ftio.prediction.shared_resources import SharedResources


def get_mode(data, mode):
    """used after get_data() to extract df. The df
    contains all sims group by mode

    Args:
        data (Scales): simulation data
        mode (str): "read_sync", "write_sync", "read_async" or "write_async"

    Raises:
        Exception: unsupported mode

    Returns:
        df: pandas dataframe containing data group by mode
    """
    mode = mode.lower()
    if "read" in mode:
        if "async" in mode:
            return data.df_rat
        elif "sync" in mode:
            return data.df_rst
    if "write" in mode:
        if "async" in mode:
            return data.df_wat
        elif "sync" in mode:
            return data.df_wst
    raise Exception("undefined mode set")


def get_sim(data, mode):
    mode = mode.lower()
    if "read" in mode:
        if "async" in mode:
            return data.read_async_t
        elif "sync" in mode:
            return data.read_sync
    if "write" in mode:
        if "async" in mode:
            return data.write_async_t
        elif "sync" in mode:
            return data.write_sync
    raise Exception("undefined mode set")


class MyConsole(Console):
    """Console child class that overwrites
    the print method for silent version

    Args:
        Console (_type_): _description_
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def set(self, flag):
        if flag:
            self.verbose = True
        else:
            self.verbose = False

    def print(self, *args, **kwargs):
        if self.verbose:
            super().print(*args, **kwargs)

    def info(self, s):
        Console.print(self, s)


def append_messages(data: dict, shared_resources: SharedResources) -> dict:
    for sim in data:
        if "bandwidth" in sim:
            b = sim["bandwidth"] if "bandwidth" in sim else np.array([])
            t = sim["time"] if "time" in sim else np.array([])
            shared_resources.b_app.extend(b)
            shared_resources.t_app.extend(t)
            sim["bandwidth"] = np.array(shared_resources.b_app[:])
            sim["time"] = np.array(shared_resources.t_app[:])
    return data
