"""
IDs for dash app components

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import ftio.plot.dash_files.constants.io_mode as io_mode

# Button
BUTTON_SHOW: str = "button-show"

# Checklist
CHECKLIST_IO_MODE: str = "checklist-io-mode"

# Div
DIV_SYNC_READ: str = "div-sync-read"
DIV_SYNC_WRITE: str = "div-sync-write"
DIV_ASNYC_READ: str = "div-async-read"
DIV_ASYNC_WRITE: str = "div-async-write"
DIV_IO_TIME: str = "div-io-time"

DIV_IO_BY_IO_MODE: dict[str, str] = {
    io_mode.SYNC_READ: DIV_SYNC_READ,
    io_mode.SYNC_WRITE: DIV_SYNC_WRITE,
    io_mode.ASYNC_READ: DIV_ASNYC_READ,
    io_mode.ASYNC_WRITE: DIV_ASYNC_WRITE,
    io_mode.TIME: DIV_IO_TIME,
}

# Dropdown
DROPDOWN_FILE: str = "dropdown-file"

# Store
STORE_FIGURES_ASYNC_READ: str = "store-figures-" + io_mode.ASYNC_READ
STORE_FIGURES_ASYNC_WRITE: str = "store-figures-" + io_mode.ASYNC_WRITE
STORE_FIGURES_SYNC_READ: str = "store-figures-" + io_mode.SYNC_READ
STORE_FIGURES_SYNC_WRITE: str = "store-figures-" + io_mode.SYNC_WRITE
STORE_FIGURES_TIME: str = "store-figures-" + io_mode.TIME

STORE_FIGURES_BY_IO_MODE: dict[str, str] = {
    io_mode.ASYNC_READ: STORE_FIGURES_ASYNC_READ,
    io_mode.ASYNC_WRITE: STORE_FIGURES_ASYNC_WRITE,
    io_mode.SYNC_READ: STORE_FIGURES_SYNC_READ,
    io_mode.SYNC_WRITE: STORE_FIGURES_SYNC_WRITE,
    io_mode.TIME: STORE_FIGURES_TIME,
}

# Type
TYPE_DYNAMIC_GRAPH: str = "dynamic-graph"

TYPE_DYNAMIC_UPDATER_ASYNC_READ: str = "dynamic-updater-" + io_mode.ASYNC_READ
TYPE_DYNAMIC_UPDATER_ASYNC_WRITE: str = "dynamic-updater-" + io_mode.ASYNC_WRITE
TYPE_DYNAMIC_UPDATER_SYNC_READ: str = "dynamic-updater-" + io_mode.SYNC_READ
TYPE_DYNAMIC_UPDATER_SYNC_WRITE: str = "dynamic-updater-" + io_mode.SYNC_WRITE
TYPE_DYNAMIC_UPDATER_TIME: str = "dynamic-updater-" + io_mode.TIME

TYPE_DYNAMIC_UPDATER_BY_IO_MODE: dict[str, str] = {
    io_mode.ASYNC_READ: TYPE_DYNAMIC_UPDATER_ASYNC_READ,
    io_mode.ASYNC_WRITE: TYPE_DYNAMIC_UPDATER_ASYNC_WRITE,
    io_mode.SYNC_READ: TYPE_DYNAMIC_UPDATER_SYNC_READ,
    io_mode.SYNC_WRITE: TYPE_DYNAMIC_UPDATER_SYNC_WRITE,
    io_mode.TIME: TYPE_DYNAMIC_UPDATER_TIME,
}
