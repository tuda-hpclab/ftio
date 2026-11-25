"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import importlib.util

import ftio.plot.dash_files.constants.id as id
import ftio.plot.dash_files.constants.io_mode as io_mode
from ftio.plot.dash_files.callback_files.callbacks import get_callbacks

DASH_AVAILABLE = importlib.util.find_spec("dash") is not None
if not DASH_AVAILABLE:
    raise RuntimeError(
        "Dash module not found. Please install it using 'make full' or 'pip install dash dash-extensions'."
    )
else:
    from dash import dcc, html
    from dash_extensions.enrich import (
        DashProxy,
        ServersideOutputTransform,
        TriggerTransform,
    )


class IOAnalysisApp(DashProxy):
    def __init__(self, plot_core) -> None:
        super().__init__(
            __name__,
            transforms=[ServersideOutputTransform(), TriggerTransform()],
        )
        self.title = "IO Analysis"

        self._plot_core = plot_core
        self._io_modes = self._collect_io_modes()

        self.layout = self._div_layout()

        get_callbacks(self, self._plot_core, self._io_modes)

    def _div_layout(self) -> html.Div:
        return html.Div(
            children=[
                dcc.Loading(
                    [
                        dcc.Store(
                            id=id.STORE_FIGURES_SYNC_READ,
                            data={},
                            storage_type="memory",
                        ),
                        dcc.Store(
                            id=id.STORE_FIGURES_SYNC_WRITE,
                            data={},
                            storage_type="memory",
                        ),
                        dcc.Store(
                            id=id.STORE_FIGURES_ASYNC_READ,
                            data={},
                            storage_type="memory",
                        ),
                        dcc.Store(
                            id=id.STORE_FIGURES_ASYNC_WRITE,
                            data={},
                            storage_type="memory",
                        ),
                        # dcc.Store(id=id.STORE_FIGURES_TIME, data={}, storage_type="memory"),
                    ],
                    fullscreen=True,
                    type="dot",
                ),
                html.H1(self.title),
                html.Hr(),
                html.Div(
                    children="Number of files: " + str(len(self._plot_core.data.paths))
                ),
                html.Hr(),
                dcc.Dropdown(
                    options=self._plot_core.data.paths,
                    value=self._plot_core.data.paths,
                    id=id.DROPDOWN_FILE,
                    multi=True,
                    style={"marginTop": 10},
                    disabled=(bool(self._plot_core.data.args.merge_plots)),
                ),
                dcc.Checklist(
                    options=self._io_modes,
                    value=self._io_modes,
                    inline=True,
                    style={"marginBottom": 10, "marginTop": 10},
                    id=id.CHECKLIST_IO_MODE,
                ),
                html.Button("Show", id=id.BUTTON_SHOW, n_clicks=0),
                html.Hr(),
                html.Div(id=id.DIV_SYNC_READ, children=[]),
                html.Div(id=id.DIV_SYNC_WRITE, children=[]),
                html.Div(id=id.DIV_ASNYC_READ, children=[]),
                html.Div(id=id.DIV_ASYNC_WRITE, children=[]),
                # html.Div(id=id.DIV_IO_TIME, children=[]),
            ],
        )

    def _collect_io_modes(self) -> list[str]:
        data = self._plot_core.data
        io_modes = []
        if not (data.df_rat is None or data.df_rat[1].empty) or not (
            data.df_rab is None or data.df_rab[1].empty
        ):
            io_modes.append(io_mode.ASYNC_READ)
        if not (data.df_wat is None or data.df_wat[1].empty) or not (
            data.df_wab is None or data.df_wab[1].empty
        ):
            io_modes.append(io_mode.ASYNC_WRITE)
        if not (data.df_rst is None or data.df_rst[1].empty):
            io_modes.append(io_mode.SYNC_READ)
        if not (data.df_wst is None or data.df_wst[1].empty):
            io_modes.append(io_mode.SYNC_WRITE)

        return io_modes
