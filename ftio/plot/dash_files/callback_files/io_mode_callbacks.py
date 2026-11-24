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
import sys

import numpy as np
import plotly.graph_objects as go

import ftio.plot.dash_files.constants.id as id
import ftio.plot.dash_files.constants.io_mode as io_mode
import ftio.plot.dash_files.constants.legend_group as legend_group
from ftio.plot.dash_files.data_source import DataSource, FileData

DASH_AVAILABLE = importlib.util.find_spec("dash") is not None
if not DASH_AVAILABLE:
    sys.exit(
        "Dash module not found. Please install it using 'make full' or 'pip install dash dash-extensions plotly_resampler'."
    )
else:
    import dash
    from dash import MATCH, Input, Output, State, dcc, html
    from dash_extensions.enrich import DashProxy, Serverside
    from plotly_resampler import FigureResampler
    from plotly_resampler.aggregation import (
        MinMaxAggregator,
        NoGapHandler,
    )

    # -------
    # TODO: Check if this problem is still not fixed or feature is now implemented
    # Solves a problem for a still open issue
    # Issue: line plot ends cutoff after last visible point #257
    # https://github.com/predict-idlab/plotly-resampler/issues/257
    from plotly_resampler.aggregation.plotly_aggregator_parser import (
        PlotlyAggregatorParser,
    )
    from trace_updater import TraceUpdater

    class patched_parser(PlotlyAggregatorParser):
        @staticmethod
        def get_start_end_indices(hf_trace_data, axis_type, start, end):
            start_idx, end_idx = PlotlyAggregatorParser.get_start_end_indices(
                hf_trace_data, axis_type, start, end
            )
            start_idx = min(max(0, start_idx - 1), max(0, start_idx - 2))
            length_x = len(hf_trace_data["x"])
            end_idx = max(min(length_x, end_idx + 1), min(length_x, end_idx + 2))
            return start_idx, end_idx

    from plotly_resampler.figure_resampler import figure_resampler_interface

    figure_resampler_interface.PlotlyAggregatorParser = patched_parser
    # end of problem solving issue #257
    # -------


def _create_id_figure(data: DataSource, filename: str = ""):
    return f"{data.io_mode}-{filename}"


def _find_x_min_and_max(file_data: FileData, data: DataSource) -> tuple[float, float]:
    x_min = np.inf
    x_max = -np.inf
    if file_data.data_actual_is_not_empty:
        x_min = min(x_min, min(file_data.actual_time_overlap, default=0.0))
        x_max = max(x_max, max(file_data.actual_time_overlap, default=0.0))
        if data.individual_is_selected:
            x_min = min(
                x_min,
                min(file_data.actual_time_overlap_individual, default=0.0),
            )
            x_max = max(
                x_max,
                max(file_data.actual_time_overlap_individual, default=0.0),
            )
    if file_data.data_required_is_not_empty:
        x_min = min(x_min, min(file_data.required_time_overlap, default=0.0))
        x_max = max(x_max, max(file_data.required_time_overlap, default=0.0))
        if data.individual_is_selected:
            x_min = min(
                x_min,
                min(file_data.required_time_overlap_individual, default=0.0),
            )
            x_max = max(
                x_max,
                max(file_data.required_time_overlap_individual, default=0.0),
            )
    return (x_min, x_max)


def _add_trace_average(fig: FigureResampler, file_data: FileData) -> None:
    if file_data.data_actual_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.ACTUAL_AVERAGE,
                legendgrouptitle_text=legend_group.ACTUAL_AVERAGE_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.actual_time_overlap,
            hf_y=file_data.actual_bandwidth_overlap_average,
        )
    if file_data.data_required_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.REQUIRED_AVERAGE,
                legendgrouptitle_text=legend_group.REQUIRED_AVERAGE_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.required_time_overlap,
            hf_y=file_data.required_bandwidth_overlap_average,
        )


def _add_trace_sum(fig: FigureResampler, file_data: FileData) -> None:
    if file_data.data_actual_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.ACTUAL_SUM,
                legendgrouptitle_text=legend_group.ACTUAL_SUM_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.actual_time_overlap,
            hf_y=file_data.actual_bandwidth_overlap_sum,
        )
    if file_data.data_required_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.REQUIRED_SUM,
                legendgrouptitle_text=legend_group.REQUIRED_SUM_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.required_time_overlap,
            hf_y=file_data.required_bandwidth_overlap_sum,
        )


def _add_trace_indiviual(fig: FigureResampler, file_data: FileData) -> None:
    if file_data.data_actual_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.ACTUAL_INDIVIDUAL,
                legendgrouptitle_text=legend_group.ACTUAL_INDIVIDUAL_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.actual_time_overlap_individual,
            hf_y=file_data.actual_bandwidth_overlap_individual,
        )
    if file_data.data_required_is_not_empty:
        fig.add_trace(
            go.Scatter(
                mode="lines",
                line={"shape": "hv"},
                fill="tozeroy",
                legendgroup=legend_group.REQUIRED_INDIVIDUAL,
                legendgrouptitle_text=legend_group.REQUIRED_INDIVIDUAL_TITLE,
                name=file_data.name,
            ),
            hf_x=file_data.required_time_overlap_individual,
            hf_y=file_data.required_bandwidth_overlap_individual,
        )


def _add_trace_invisible_for_complete_presentation(
    fig: FigureResampler, file_data: FileData, data: DataSource
) -> None:
    x_min, x_max = _find_x_min_and_max(file_data, data)
    fig.add_trace(
        go.Scatter(
            showlegend=False,
            hoverinfo="skip",
            opacity=0.0,
        ),
        hf_x=[x_min, x_max],
        hf_y=[0.0, 0.0],
    )


def _add_traces(fig: FigureResampler, file_data: FileData, data: DataSource) -> None:
    _add_trace_average(fig, file_data)
    _add_trace_sum(fig, file_data)
    if data.individual_is_selected:
        _add_trace_indiviual(fig, file_data)
    _add_trace_invisible_for_complete_presentation(fig, file_data, data)


def _update_layout(fig: FigureResampler, file_data: FileData, data: DataSource) -> None:
    if data.merge_plots_is_selected:
        title = f"{io_mode.MODE_STRING_BY_MODE[data.io_mode]} collected"
    else:
        title = f"{file_data.rank} Ranks (Run {file_data.run}: {file_data.name})"

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 50, "b": 70},
        yaxis_rangemode="nonnegative",
        barmode="stack",
        xaxis_title="Time (s)",
        yaxis_title="Transfer Rate (MB/s)",
        font={"family": data.fontfamily, "size": data.fontsize},
        width=data.width_figure,
        height=data.height_figure,
        title=title,
    )


def _update_axes(fig: FigureResampler) -> None:
    fig.update_xaxes(
        ticks="outside",
        tickcolor="black",
        ticklen=6,
        minor={
            "ticklen": 3,
            "tickcolor": "black",
            "tickmode": "auto",
            "nticks": 10,
            "showgrid": True,
        },
    )
    fig.update_yaxes(
        ticks="outside",
        tickcolor="black",
        ticklen=6,
        minor={
            "ticklen": 3,
            "tickcolor": "black",
            "tickmode": "auto",
            "nticks": 10,
            "showgrid": True,
        },
    )


def _create_separate_figures(filenames: list[str], data: DataSource) -> dict:
    figure_by_id_figure = {}

    for filename in filenames:
        file_data = data.file_data_by_filename[filename]
        fig = FigureResampler(
            default_n_shown_samples=data.n_shown_samples,
            default_downsampler=MinMaxAggregator(),
            default_gap_handler=NoGapHandler(),
        )
        _add_traces(fig, file_data, data)

        _update_layout(fig, file_data, data)
        _update_axes(fig)

        id_figure = _create_id_figure(data, filename)
        figure_by_id_figure[id_figure] = fig
    return figure_by_id_figure


def _create_one_common_figure(filenames: list[str], data: DataSource) -> dict:
    figure_by_id_figure = {}

    fig = FigureResampler(
        default_n_shown_samples=data.n_shown_samples,
        default_downsampler=MinMaxAggregator(),
        default_gap_handler=NoGapHandler(),
    )

    for filename in filenames:
        file_data = data.file_data_by_filename[filename]
        _add_traces(fig, file_data, data)

    _update_layout(fig, file_data, data)
    _update_axes(fig)
    id_figure = _create_id_figure(data)
    figure_by_id_figure[id_figure] = fig
    return figure_by_id_figure


def _append_merged_plot(
    div_children: list[html.Div],
    figure_by_id_figure: dict[str, FigureResampler],
    data: DataSource,
) -> html.Div:
    id_figure = _create_id_figure(data)
    new_child = html.Div(
        children=[
            dcc.Graph(
                id={"type": id.TYPE_DYNAMIC_GRAPH, "index": id_figure},
                figure=figure_by_id_figure[id_figure],
                mathjax=True,
                responsive=True,
            ),
            TraceUpdater(
                id={
                    "type": id.TYPE_DYNAMIC_UPDATER_BY_IO_MODE[data.io_mode],
                    "index": id_figure,
                },
                gdID=f"{id_figure}",
            ),
        ],
    )
    div_children.append(new_child)
    return div_children


def _append_each_figure_separately(
    div_children: list[html.Div],
    figure_by_id_figure: dict[str, FigureResampler],
    filenames: list[str],
    data: DataSource,
) -> html.Div:
    for rank in data.ranks_unique:
        for filename in filenames:
            if data.file_data_by_filename[filename].rank != rank:
                continue
            id_figure = _create_id_figure(data, filename)
            new_child = html.Div(
                children=[
                    dcc.Graph(
                        id={"type": id.TYPE_DYNAMIC_GRAPH, "index": id_figure},
                        figure=figure_by_id_figure[id_figure],
                        mathjax=True,
                        responsive=True,
                    ),
                    TraceUpdater(
                        id={
                            "type": id.TYPE_DYNAMIC_UPDATER_BY_IO_MODE[data.io_mode],
                            "index": id_figure,
                        },
                        gdID=f"{id_figure}",
                    ),
                ],
            )
            div_children.append(new_child)
    return div_children


def get_io_mode_specific_callbacks(app: DashProxy, data: DataSource) -> None:
    """Defines the callbacks to a specific io_mode which is indirectly given by the data.

    Args:
        app (DashProxy): The dash app
        data (DataSource): data_source belonging to the io_mode
    """

    @app.callback(
        Output(id.STORE_FIGURES_BY_IO_MODE[data.io_mode], "data"),
        Input(id.DROPDOWN_FILE, "options"),
    )
    def create_figures(filenames: list[str]) -> dict[str, FigureResampler]:
        figure_by_id_figure = {}

        if data.merge_plots_is_selected:
            figure_by_id_figure = _create_one_common_figure(filenames, data)
        else:
            figure_by_id_figure = _create_separate_figures(filenames, data)

        return Serverside(figure_by_id_figure)

    @app.callback(
        Output(id.DIV_IO_BY_IO_MODE[data.io_mode], "children"),
        State(id.CHECKLIST_IO_MODE, "value"),
        State(id.DROPDOWN_FILE, "value"),
        State(id.STORE_FIGURES_BY_IO_MODE[data.io_mode], "data"),
        Input(id.BUTTON_SHOW, "n_clicks"),
        prevent_initial_call=True,
    )
    def fill_div_graph(
        io_modes: list[str],
        filenames: list[str],
        figure_by_id_figure: dict[str, FigureResampler],
        n_clicks: int,
    ):
        if data.io_mode not in io_modes:
            return []

        div_children = [
            html.Div(
                dcc.Markdown(
                    f"### {io_mode.MODE_STRING_BY_MODE[data.io_mode]}",
                    mathjax=True,
                )
            )
        ]

        if data.merge_plots_is_selected:
            div_children = _append_merged_plot(div_children, figure_by_id_figure, data)
        else:
            div_children = _append_each_figure_separately(
                div_children, figure_by_id_figure, filenames, data
            )

        return div_children

    @app.callback(
        Output(
            {
                "type": id.TYPE_DYNAMIC_UPDATER_BY_IO_MODE[data.io_mode],
                "index": MATCH,
            },
            "updateData",
        ),
        Input({"type": id.TYPE_DYNAMIC_GRAPH, "index": MATCH}, "relayoutData"),
        State({"type": id.TYPE_DYNAMIC_GRAPH, "index": MATCH}, "id"),
        State(id.STORE_FIGURES_BY_IO_MODE[data.io_mode], "data"),
        prevent_initial_call=True,
        memoize=True,
    )
    def update_figure(
        relayoutdata: dict,
        id: dict[str, str],
        figure_by_id_figure: dict[str, FigureResampler],
    ):
        fig = figure_by_id_figure[id["index"]]
        if fig is None:
            return dash.no_update
        return fig._construct_update_data(relayoutdata)
