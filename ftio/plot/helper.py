"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def legend_fix(data, c: int) -> bool:
    """Determines if the legend should be fixed for the current plot.

    Args:
        data: Data object containing plot information.
        c (int): Current index of the plot.

    Returns:
        bool: True if the legend should be fixed, False otherwise.
    """
    return c == data.nRun - 1


def format_plot_and_ticks(
    fig: go.Figure,
    legend: bool = True,
    font: bool = True,
    x_minor: bool = True,
    y_minor: bool = True,
    x_ticks: bool = True,
    y_ticks: bool = True,
    n_ticks: int = 5,
    font_size: int = 17,
) -> go.Figure:
    """Formats the plot with specified settings for legend, font, and axis ticks.

    Args:
        fig (go.Figure): Plotly figure to format.
        legend (bool, optional): Whether to format the legend. Defaults to True.
        font (bool, optional): Whether to format the font. Defaults to True.
        x_minor (bool, optional): Whether to show minor ticks on the x-axis. Defaults to True.
        y_minor (bool, optional): Whether to show minor ticks on the y-axis. Defaults to True.
        x_ticks (bool, optional): Whether to show major ticks on the x-axis. Defaults to True.
        y_ticks (bool, optional): Whether to show major ticks on the y-axis. Defaults to True.
        n_ticks (int, optional): Number of minor ticks. Defaults to 5.

    Returns:
        go.Figure: Formatted Plotly figure.
    """
    if legend:
        fig.update_layout(
            legend={
                "bgcolor": "rgba(255,255,255,.99)",
                "bordercolor": "Black",
                "borderwidth": 1,
            }
        )
    if font:
        fig.update_layout(
            font={"family": "Courier New, monospace", "size": font_size, "color": "black"}
        )

    fig.update_layout(
        plot_bgcolor="white",
        margin={"r": 5},
    )

    x_settings = {
        "mirror": True,
        "showgrid": True,
        "showline": True,
        "linecolor": "black",
        "gridcolor": "lightgrey",
    }
    if x_ticks:
        x_settings["ticks"] = "outside"
        x_settings["ticklen"] = 10

    if x_minor:
        x_settings["minor_ticks"] = "outside"
        x_settings["minor"] = {
            "ticklen": 2,
            "tickcolor": "black",
            "tickmode": "auto",
            "nticks": n_ticks,
            "showgrid": True,
        }

    fig.update_xaxes(**x_settings)

    y_settings = {
        "mirror": True,
        "showgrid": True,
        "showline": True,
        "linecolor": "black",
        "gridcolor": "lightgrey",
    }
    if y_ticks:
        y_settings["ticks"] = "outside"
        y_settings["ticklen"] = 10

    if y_minor:
        y_settings["minor_ticks"] = "outside"
        y_settings["minor"] = {
            "ticklen": 2,
            "tickcolor": "black",
            "tickmode": "auto",
            "nticks": n_ticks,
            "showgrid": True,
        }

    fig.update_yaxes(**y_settings)

    return fig


def format_plot(fig: go.Figure, font_size: int = 22) -> go.Figure:
    """Applies uniform formatting to the plot.

    Args:
        fig (go.Figure): Plotly figure to format.
        font_size (int, optional): Font size for the plot. Defaults to 24.

    Returns:
        go.Figure: Formatted Plotly figure.
    """
    fig.update_layout(
        plot_bgcolor="white",
        legend={
            "bgcolor": "rgba(255,255,255,.8 )",
            "bordercolor": "Black",
            "borderwidth": 1,
        },
        font={"family": "Courier New, monospace", "size": font_size, "color": "black"},
        # margin=dict(l=5, r=5, t=5, b=5) #IEEE
        # margin=dict(t=top_margin),
    )

    fig.update_xaxes(
        ticks="outside",
        # tickwidth=1,
        ticklen=10,
        showgrid=True,
        # gridwidth=1,
        mirror=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        minor_ticks="outside",
        minor={"ticklen": 2},
    )

    fig.update_yaxes(
        ticks="outside",
        # tickwidth=1,
        ticklen=10,
        showgrid=True,
        # gridwidth=1,
        mirror=True,
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        minor_ticks="outside",
        minor={"ticklen": 2},
    )

    return fig


def save_fig(fig: go.Figure, f: list[go.Figure], path: str, name: str) -> None:
    """Saves the given figure to a file.

    Args:
        fig (go.Figure): Plotly figure to save.
        f (list): List of figures.
        path (str): Directory path to save the figure.
        name (str): Base name for the saved file.

    Returns:
        None
    """
    length = len(f)
    index = f.index(fig)
    try:
        print(f"   -> Working on {index}/{length-1}")
        fig.write_image(f"{path}/{name}_{index}.pdf")
    except Exception as error:
        print(f"An exception occurred: {error}")
        fig.write_image(f"{index}.pdf")
    print(f"   -> Finished {index}/{length-1}")


def add_fig_row(
    nRun: int, onefig: bool = False, specs=None, subplot_titles=None
) -> go.Figure:
    """Creates a figure with rows of subplots.

    Args:
        nRun (int): Number of rows.
        onefig (bool, optional): Whether to create a single figure. Defaults to False.
        specs (list, optional): Specifications for the subplots. Defaults to None.
        subplot_titles (list, optional): Titles for the subplots. Defaults to None.

    Returns:
        go.Figure: Plotly figure with rows of subplots.
    """
    if nRun == 1 or onefig:
        # self.f_t.append(go.Figure())
        f = make_subplots(rows=1, cols=1, specs=specs)
    else:
        f = make_subplots(rows=nRun, cols=1, specs=specs, subplot_titles=subplot_titles)

    return f


def add_fig_col(
    nRun: int,
    onefig: bool = False,
    specs=None,
    subplot_titles=None,
    horizontal_spacing: float = 0.01,
) -> go.Figure:
    """Creates a figure with columns of subplots.

    Args:
        nRun (int): Number of columns.
        onefig (bool, optional): Whether to create a single figure. Defaults to False.
        specs (list, optional): Specifications for the subplots. Defaults to None.
        subplot_titles (list, optional): Titles for the subplots. Defaults to None.
        horizontal_spacing (float, optional): Spacing between columns. Defaults to 0.01.

    Returns:
        go.Figure: Plotly figure with columns of subplots.
    """
    if nRun == 1 or onefig:
        # self.f_t.append(go.Figure())
        f = make_subplots(rows=1, cols=1, specs=specs)
    else:
        subplot_titles = np.arange(0, nRun).astype(str).tolist()
        subplot_titles = ["Run " + x for x in subplot_titles]
        f = make_subplots(
            rows=1,
            cols=nRun,
            specs=specs,
            shared_yaxes=True,
            subplot_titles=subplot_titles,
            horizontal_spacing=horizontal_spacing,
        )

    return f
