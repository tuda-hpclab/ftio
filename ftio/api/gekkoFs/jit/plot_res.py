"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import plotly.graph_objects as go


def format(fig: go.Figure, title: str = "") -> None:
    """
    Formats the given Plotly figure with specific layout and text settings.

    Args:
        fig (go.Figure): The Plotly figure to format.
        title (str, optional): The title of the figure. Defaults to an empty string.
    """
    # Update text formatting
    fig.update_traces(
        textposition="inside",
        texttemplate="%{text:.2f}",
        textfont_size=16,  # Increased font size
        textangle=0,
        textfont={"color": "white"},
    )

    fig.update_layout(barmode="group")

    # Update layout with larger font sizes
    fig.update_layout(
        yaxis_title="Time (s)",
        # xaxis_title=f"Experimental Runs with # Nodes",
        xaxis_title="",
        showlegend=True,
        title=title,
        title_font_size=24,  # Increased title font size
        width=1000,
        height=700,
        xaxis={"title_font": {"size": 24}},  # Increased x-axis title font size
        yaxis={"title_font": {"size": 24}},  # Increased y-axis title font size
        legend={
            "font": {"size": 20},  # Increased legend font size
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.9,
            "xanchor": "right",
            "x": 0.995,
        },
    )

    format_plot_and_ticks(fig, x_minor=False, font_size=20)
    # Display the plot
    fig.show()

    # Comment out to see all text
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")


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
    """
    Formats the plot with specified settings for legend, font, and axis ticks.

    Args:
        fig (go.Figure): Plotly figure to format.
        legend (bool, optional): Whether to format the legend. Defaults to True.
        font (bool, optional): Whether to format the font. Defaults to True.
        x_minor (bool, optional): Whether to show minor ticks on the x-axis. Defaults to True.
        y_minor (bool, optional): Whether to show minor ticks on the y-axis. Defaults to True.
        x_ticks (bool, optional): Whether to show major ticks on the x-axis. Defaults to True.
        y_ticks (bool, optional): Whether to show major ticks on the y-axis. Defaults to True.
        n_ticks (int, optional): Number of minor ticks. Defaults to 5.
        font_size (int, optional): Font size for the plot. Defaults to 17.

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
                legend={
                    "font": {
                        "family": "Courier New, monospace",
                        "size": font_size - 1,
                        "color": "black",
                    }
                }  # set your desired font size here
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


def your_function() -> None:
    """
    function to create and format a bar chart using Plotly.
    """
    fig = go.Figure()
    nodes = [2, 4, 6, 8]
    y1 = [311.23, 122.7, 80.62, 65.3]
    y2 = [367.5, 160.12, 159.12, 158.74]
    y3 = [370.12, 162.14, 161.21, 157.97]

    fig.add_bar(x=nodes, y=y1, text=y1, name="Gekko+Lustre HSM")
    fig.add_bar(x=nodes, y=y2, text=y2, name="LPCC")
    fig.add_bar(x=nodes, y=y3, text=y3, name="Lustre")
    format(fig)


def main() -> None:
    your_function()
    # add more


if __name__ == "__main__":
    main()
