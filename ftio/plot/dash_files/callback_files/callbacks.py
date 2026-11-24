"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.plot.dash_files.callback_files.io_mode_callbacks import (
    get_io_mode_specific_callbacks,
)
from ftio.plot.dash_files.data_source import get_data_source


def get_callbacks(app, plot_core, io_modes: list[str]) -> None:
    """Is responsible for defining all callbacks for the dash app.

    Args:
        app (DashProxy): The dash app
        plot_core (plot_core): The core component for plotting.
        io_modes (list[str]): All possible io_modes (according to the app).
    """
    for io_mode in io_modes:
        data_source = get_data_source(plot_core, io_mode)
        get_io_mode_specific_callbacks(app, data_source)
