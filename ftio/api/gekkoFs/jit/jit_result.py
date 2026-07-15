"""
This file provides the `JitResult` class for managing and visualizing experimental data
related to application times, stage out times, and stage in times. It includes methods
for adding experiments, processing data from dictionaries, sorting data based on timestamps,
and generating plots for analysis.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mar 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os

import numpy as np
import plotly.graph_objects as go
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ftio.plot.helper import format_plot_and_ticks

CONSOLE = Console()


class JitResult:
    def __init__(self) -> None:
        """
        Initialize a JitResult object to store experimental data.
        """
        self.app = []
        self.stage_out = []
        self.stage_in = []
        self.node = []
        self.app_stats = {}
        self.stage_out_stats = {}
        self.stage_in_stats = {}

    def add_experiment(
        self,
        tmp_app: list[float],
        tmp_stage_out: list[float],
        tmp_stage_in: list[float],
        run: str,
    ):
        """
        Add an experiment's data to the JitResult object.

        Args:
            tmp_app (list[float]): Application times.
            tmp_stage_out (list[float]): Stage out times.
            tmp_stage_in (list[float]): Stage in times.
            run (str): Identifier for the experiment run.
        """
        self.app = add_mode(self.app, tmp_app)
        self.stage_out = add_mode(self.stage_out, tmp_stage_out)
        self.stage_in = add_mode(self.stage_in, tmp_stage_in)
        self.node.append(run)

    def add_experiment_with_stats(
        self,
        tmp_app: list[float],
        tmp_stage_out: list[float],
        tmp_stage_in: list[float],
        run: str,
        app_stats: dict,
        stage_out_stats: dict,
        stage_in_stats: dict,
    ):
        """
        Add an experiment's data to the JitResult object, including stats (min, max, mean).

        Args:
            tmp_app (list[float]): Application times.
            tmp_stage_out (list[float]): Stage out times.
            tmp_stage_in (list[float]): Stage in times.
            run (str): Identifier for the experiment run.
            app_stats (dict): Dictionary containing min, max, and mean stats for 'app'.
            stage_out_stats (dict): Dictionary containing min, max, and mean stats for 'stage_out'.
            stage_in_stats (dict): Dictionary containing min, max, and mean stats for 'stage_in'.
        """
        # First, use add_experiment to handle the raw data
        self.add_experiment(tmp_app, tmp_stage_out, tmp_stage_in, run)

        # Now, store the statistics
        self.app_stats[run] = app_stats
        self.stage_out_stats[run] = stage_out_stats
        self.stage_in_stats[run] = stage_in_stats

    def add_dict(self, data_list: list[dict], get_latest_timestamp: bool = True):
        """
        Add data from a dictionary to the JitResult object, optionally using the latest data based on the timestamp.

        Args:
            data_list (list[dict]): List of dictionaries containing experimental data.
            get_latest_timestamp (bool): Whether to use the latest entry based on the timestamp (default is False).
        """
        tmp_app = [0.0, 0.0, 0.0]
        tmp_stage_out = [0.0, 0.0, 0.0]
        tmp_stage_in = [0.0, 0.0, 0.0]

        sorted_data = sort_data(data_list, get_latest_timestamp)

        # Now loop through the (possibly sorted) data
        for i in sorted_data:
            # Apply logic based on the mode
            if "F" in i["mode"]:
                index = 0
            elif "D" in i["mode"]:
                index = 1
            else:
                index = 2

            tmp_app[index] = i["app"]
            tmp_stage_out[index] = i["stage_out"]
            tmp_stage_in[index] = i["stage_in"]
            # Jit nodes are off by 1, as FTIO runs on a dedicated on
            nodes = int(data_list["nodes"]) - 1
        # self.add_experiment(tmp_app, tmp_stage_out, tmp_stage_in, f"# {nodes}")
        self.add_experiment(tmp_app, tmp_stage_out, tmp_stage_in, f"{nodes}")

    def plot(self, title="", save_dir=None):
        """
        Plot the experimental data stored in the JitResult object.

        Args:
            title (str): Title for the plot.
            save_dir (str | None): If given, also write the figure there as
                plot.html (used to drop the plot into the per-app logs/<app> folder).
        """
        # Sample data for the stacked plot
        categories = ["GLASS", "GekkoFs & Lustre", "Lustre"]
        repeated_strings = [s for s in categories for _ in self.node]
        repeated_numbers = self.node * len(categories)
        categories = [repeated_strings, repeated_numbers]

        fig = go.Figure()

        fig.add_bar(x=categories, y=self.app, text=self.app, name="App")
        fig.add_bar(
            x=categories,
            y=self.stage_out,
            text=self.stage_out,
            name="Stage out",
        )
        fig.add_bar(x=categories, y=self.stage_in, text=self.stage_in, name="Stage in")

        # Update text formatting
        fig.update_traces(
            textposition="inside",
            texttemplate="%{text:.1f}",
            textfont={"color": "white"},
            insidetextanchor="middle",
            textangle=-90 if len(self.node) > 3 else 0,
        )

        # Sum total
        total = list(
            np.round(
                np.array(self.app) + np.array(self.stage_out) + np.array(self.stage_in),
                2,
            )
        )

        fig.add_bar(
            x=categories,
            y=np.zeros_like(total),
            text=total,
            textangle=-90 if len(self.node) > 3 else 0,
            name="Total",
            marker_color="rgba(0,0,0,0)",  # Invisible bars
            showlegend=False,
            textposition="outside",
            constraintext="none",
        )

        stat(self.app, self.stage_out, self.stage_in, self.node)
        self.format_and_show(fig, title, False, save_dir=save_dir)

    def format_and_show(
        self,
        fig: go.Figure,
        title: str = "",
        text: bool = True,
        barmode: str = "relative",
        save_dir: str | None = None,
    ):

        if text:
            # Update text formatting
            fig.update_traces(
                textposition="inside",
                texttemplate="%{text:.2f}",
                textfont={"color": "white"},
            )

        font_size = 18 - len(self.node)
        if len(self.node) > 3:
            font_size = 14
        fig.update_traces(
            textfont={"size": font_size},
            texttemplate="%{text:.1f}",
        )
        fig.update_annotations(font={"sie": font_size - 1})
        fig.update_layout(uniformtext_minsize=font_size - 1, uniformtext_mode="hide")

        # Update layout with larger font sizes
        fig.update_layout(
            yaxis_title="Time (s)",
            # xaxis_title=f"Experimental Runs with # Nodes",
            xaxis_title="",
            showlegend=True,
            title=title,
            barmode=barmode,
            title_font_size=24,  # Increased title font size
            width=(
                200 + 100 * len(self.node)
                if len(self.node) > 3
                else 500 + 100 * len(self.node)
            ),
            height=500,
            xaxis={"title_font": {"size": 20}},  # Increased x-axis title font size
            yaxis={"title_font": {"size": 20}},  # Increased y-axis title font size
            legend={
                "font": {"size": 22},  # Increased legend font size
            },
        )
        format_plot_and_ticks(fig, x_minor=False, font_size=16)
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 0.85,
                "xanchor": "left",
                "x": 0.005,
                # xanchor="right",
                # x=0.997,
            }
        )
        # fig.update_traces(
        #     textposition="inside",
        #     texttemplate="%{text:.2f}",
        #     textfont_size=20,  # Increased font size
        #     textangle=0,
        #     textfont=dict(color="white"),
        # )

        # fig.update_layout(uniformtext_minsize=13, uniformtext_mode="hide")

        # # Update layout with larger font sizes
        # fig.update_layout(
        #     yaxis_title="Time (s)",
        #     xaxis_title=f"Experimental Runs with # Nodes",
        #     showlegend=True,
        #     title=title,
        #     barmode=barmode,
        #     title_font_size=24,  # Increased title font size
        #     width=1000 + 100 * len(self.node),
        #     height=550,
        #     xaxis=dict(title_font=dict(size=24)),  # Increased x-axis title font size
        #     yaxis=dict(title_font=dict(size=24)),  # Increased y-axis title font size
        #     legend=dict(
        #         font=dict(size=20),  # Increased legend font size
        #     ),
        # )
        # format_plot_and_ticks(fig, x_minor=False, font_size=20)
        # fig.update_layout(
        #     legend=dict(
        #         orientation="h",
        #         yanchor="bottom",
        #         y=0.87,
        #         xanchor="left",
        #         x=0.005,
        #         # xanchor="right",
        #         # x=0.997,
        #     )
        # )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, "plot.html")
            fig.write_html(out)
            CONSOLE.print(f"[green]Plot written to {out}[/]")
        fig.show()

    ######################
    # Add and plot all
    #####################

    def add_all(self, data_list: list[dict]):
        """
        Add data from a dictionary to the JitResult object, optionally using the latest data based on the timestamp.

        Args:
            data_list (list[dict]): List of dictionaries containing experimental data.
        """
        sorted_data = sort_data(data_list, False)

        # Add data per node configuration
        tmp_app = [0.0, 0.0, 0.0]
        tmp_stage_out = [0.0, 0.0, 0.0]
        tmp_stage_in = [0.0, 0.0, 0.0]

        # Track the entries for calculation
        app_values = {0: [], 1: [], 2: []}
        stage_out_values = {0: [], 1: [], 2: []}
        stage_in_values = {0: [], 1: [], 2: []}

        # Now loop through the (possibly sorted) data
        for i in sorted_data:
            # Apply logic based on the mode
            if "F" in i["mode"]:
                index = 0
            elif "D" in i["mode"]:
                index = 1
            else:
                index = 2

            # Add the **latest** data to the temporary lists
            tmp_app[index] = i["app"]
            tmp_stage_out[index] = i["stage_out"]
            tmp_stage_in[index] = i["stage_in"]

            # Collect all data points for min, max, mean calculations
            app_values[index].append(i["app"])
            stage_out_values[index].append(i["stage_out"])
            stage_in_values[index].append(i["stage_in"])

        # Calculate min, max, mean for each field and mode
        stats = {"app": {}, "stage_out": {}, "stage_in": {}}

        for index in range(3):
            stats["app"][index] = {
                "min": min(app_values[index]) if app_values[index] else 0,
                "max": max(app_values[index]) if app_values[index] else 0,
                "mean": np.mean(app_values[index]) if app_values[index] else 0,
            }
            stats["stage_out"][index] = {
                "min": (min(stage_out_values[index]) if stage_out_values[index] else 0),
                "max": (max(stage_out_values[index]) if stage_out_values[index] else 0),
                "mean": (
                    np.mean(stage_out_values[index]) if stage_out_values[index] else 0
                ),
            }
            stats["stage_in"][index] = {
                "min": (min(stage_in_values[index]) if stage_in_values[index] else 0),
                "max": (max(stage_in_values[index]) if stage_in_values[index] else 0),
                "mean": (
                    np.mean(stage_in_values[index]) if stage_in_values[index] else 0
                ),
            }

        # Jit nodes are off by 1, as FTIO runs on a dedicated on
        nodes = int(data_list["nodes"]) - 1

        # Now call add_experiment, including stats
        self.add_experiment_with_stats(
            tmp_app,
            tmp_stage_out,
            tmp_stage_in,
            # f"# {nodes}",
            f"{nodes}",
            stats["app"],
            stats["stage_out"],
            stats["stage_in"],
        )

    def plot_all(self, title: str = "", save_dir: str | None = None):
        """
        Generate and plot all experimental data in a bar diagram with deviations (error bars).

        Args:
            title (str): Title for the plot.
            save_dir (str | None): If given, also write the figure there as plot.html.
        """
        categories = ["GLASS", "Lustre & Gekko", "Lustre"]
        repeated_strings = [s for s in categories for _ in self.node]
        repeated_numbers = self.node * len(categories)
        categories = [repeated_strings, repeated_numbers]

        fig = go.Figure()
        app_values = []
        stage_out_values = []
        stage_in_values = []
        app_error_diff = []
        stage_out_error_diff = []
        stage_in_error_diff = []

        # Loop through all runs and plot the mean, min, and max for each mode (app, stage_out, stage_in)
        for index in range(3):
            for run in self.node:
                # Get the statistics, defaulting to None if not available
                app_stats = self.app_stats.get(run, None)
                stage_out_stats = self.stage_out_stats.get(run, None)
                stage_in_stats = self.stage_in_stats.get(run, None)

                # Check if the statistics are available (otherwise skip this run or use default values)
                if app_stats is None or stage_out_stats is None or stage_in_stats is None:
                    continue  # Skip this run if any of the stats are missing

                # Extract the mean, min, and max values for each mode
                app_mean, app_min, app_max = (
                    app_stats.get(index).get("mean", 0.0),
                    app_stats.get(index).get("min", 0.0),
                    app_stats.get(index).get("max", 0.0),
                )
                stage_out_mean, stage_out_min, stage_out_max = (
                    stage_out_stats.get(index).get("mean", 0.0),
                    stage_out_stats.get(index).get("min", 0.0),
                    stage_out_stats.get(index).get("max", 0.0),
                )
                stage_in_mean, stage_in_min, stage_in_max = (
                    stage_in_stats.get(index).get("mean", 0.0),
                    stage_in_stats.get(index).get("min", 0.0),
                    stage_in_stats.get(index).get("max", 0.0),
                )

                # Add these values to the lists
                app_values.append(app_mean)
                stage_out_values.append(stage_out_mean)
                stage_in_values.append(stage_in_mean)

                # Calculate error bars (min-max) and add them to the lists
                app_error_diff.append((app_max - app_min) / 2)
                stage_out_error_diff.append((stage_out_max - stage_out_min) / 2)
                stage_in_error_diff.append((stage_in_max - stage_in_min) / 2)

        # Plot app mean with min-max error bars
        fig.add_trace(
            go.Bar(
                x=[categories[0], categories[1]],
                y=app_values,
                text=app_values,
                name="App",
                error_y={"type": "data", "array": app_error_diff},  # Min-max error bars
            )
        )

        # Plot stage out mean with min-max error bars
        fig.add_trace(
            go.Bar(
                x=[categories[0], categories[1]],
                y=stage_out_values,
                text=stage_out_values,
                name="Stage out",
                error_y={
                    "type": "data",
                    "array": stage_out_error_diff,
                },  # Min-max error bars
            )
        )

        # Plot stage in mean with min-max error bars
        fig.add_trace(
            go.Bar(
                x=[categories[0], categories[1]],
                y=stage_in_values,
                text=stage_in_values,
                name="Stage in",
                error_y={
                    "type": "data",
                    "array": stage_in_error_diff,
                },  # Min-max error bars
            )
        )

        stat_with_min_max(
            app_values,
            stage_out_values,
            stage_in_values,
            self.node,
            app_error_diff,
            stage_out_error_diff,
            stage_in_error_diff,
        )
        self.format_and_show(fig, barmode="group", title=title, save_dir=save_dir)


def add_mode(list1, list2):
    """
    Combine two lists by interleaving elements from the first list with
    elements from the second list.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        list: The combined list.
    """
    # Create a new list to store the result
    result = []
    step = int(len(list1) / 3)

    # Iterate over the range of the lists' lengths
    for i in range(len(list2)):
        # Add the current element from each list
        for j in range(step):
            result.append(list1[i * step + j])
        result.append(list2[i])

    return result


def sort_data(data_list: list[dict], get_latest_timestamp: bool = False):
    """
    Sorts a list of dictionaries based on timestamps and modes.
        data_list (list[dict]): A list of dictionaries containing data entries.
                                Each dictionary should have a "data" key with a list of entries.
        get_latest_timestamp (bool): If True, sorts the data by timestamp in descending order
                                    and keeps only the latest entry per unique combination of
                                    "nodes" and "mode". If False, retains the original order.
    Returns:
        list[dict]: A sorted list of dictionaries. If `get_latest_timestamp` is True, the list
                    contains the latest entries for each unique combination of "nodes" and "mode",
                    along with entries without timestamps. Otherwise, the original order is preserved.
    """
    if get_latest_timestamp:
        # Separate entries with and without timestamps
        with_timestamp = [i for i in data_list["data"] if "timestamp" in i]
        without_timestamp = [i for i in data_list["data"] if "timestamp" not in i]

        # Sort the entries that have timestamps in descending order
        with_timestamp.sort(
            key=lambda x: x.get("timestamp", "0000-00-00 00:00:00"),
            reverse=True,
        )

        # Now loop through the sorted data and update only the latest entry per mode
        nodes_data = {}  # A dictionary to track the latest data for each mode and nodes
        for i in with_timestamp:
            nodes = i["nodes"]
            mode = i["mode"]

            # If no entry exists for the current mode and nodes, add the entry
            if (nodes, mode) not in nodes_data:
                nodes_data[(nodes, mode)] = i
            else:
                # If an entry exists, check if the new entry is more recent
                existing_entry = nodes_data[(nodes, mode)]
                if i["timestamp"] > existing_entry["timestamp"]:
                    nodes_data[(nodes, mode)] = i

        # Combine the latest entries and those without timestamps
        sorted_data = list(nodes_data.values()) + without_timestamp
    else:
        # If not using timestamps, just use the original order
        sorted_data = data_list["data"]

    return sorted_data


def speed_up(arr, nodes):
    # Ensure the array length is divisible by the number of nodes
    assert (
        len(arr) % len(nodes) == 0
    ), "Array length must be divisible by the number of nodes"

    # Reshape the array into chunks based on the number of nodes
    chunks = arr.reshape(-1, len(nodes))
    first_chunk = chunks[0]
    second_chunk = chunks[1]
    last_chunk = chunks[-1]

    # Calculate the speedups

    glass_speedup = np.where(last_chunk != 0, last_chunk / first_chunk, 0)
    gekko_lustre_speedup = np.where(last_chunk != 0, last_chunk / second_chunk, 0)
    glass_vs_gekko_speedup = np.where(second_chunk != 0, second_chunk / first_chunk, 0)

    # #this is improvment not sspeedup
    # glass_speedup = np.where(last_chunk != 0, (last_chunk - first_chunk) / last_chunk, 0)*100
    # gekko_lustre_speedup = np.where(last_chunk != 0, (last_chunk - second_chunk) / last_chunk, 0)*100
    # glass_vs_gekko_speedup = np.where(second_chunk != 0, (second_chunk - first_chunk) / second_chunk, 0)*100

    return (
        glass_speedup,
        gekko_lustre_speedup,
        glass_vs_gekko_speedup,
        first_chunk,
        second_chunk,
        last_chunk,
    )


def apply_color_scale(value, min_val, max_val):
    """Map the value to a color scale between red and green safely."""
    # Return gray if value is NaN or range is zero/invalid
    if value is None or min_val is None or max_val is None:
        return "rgb(128,128,128)"

    if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
        return "rgb(128,128,128)"  # gray

    # Normalize and clip to [0,1] to prevent NaN/infinity
    normalized_value = (value - min_val) / (max_val - min_val)
    normalized_value = (
        0.0 if np.isnan(normalized_value) else np.clip(normalized_value, 0, 1)
    )

    red = int((1 - normalized_value) * 255)
    green = int(normalized_value * 255)
    return f"rgb({red},{green},0)"


def highlight_row(values, chunks, min_value=None, max_value=None):
    """Returns a list of strings with a color scale for the values."""
    result = []
    for i, val in enumerate(values):
        # Skip color scale if value is NaN
        if np.isnan(val):
            formatted = "[gray]NaN[/]"
            result.append(formatted)
            continue

        # Apply color scale from red to green
        color = apply_color_scale(val, min_value, max_value)

        # Format the value with the calculated color
        formatted = f"[{color}]{val:.2f}x[/]"

        # Skip division info if any chunk at index i is NaN
        if any(np.isnan(chunk) for chunk in chunks):
            result.append(formatted)
            continue

        # Add the division information
        if i == 0:
            formatted += f"[cyan]({chunks[2]:.1f}/{chunks[0]:.1f} -- [{color}]{100*(chunks[2]-chunks[0])/chunks[2] if chunks[2] > 0 else 0:.2f}%[/])"
        elif i == 1:
            formatted += f"[cyan]({chunks[2]:.1f}/{chunks[1]:.1f} -- [{color}]{100*(chunks[2]-chunks[1])/chunks[2] if chunks[2] > 0 else 0:.2f}%[/])"
        elif i == 2:
            formatted += f"[cyan]({chunks[1]:.1f}/{chunks[0]:.1f} -- [{color}]{100*(chunks[1]-chunks[0])/chunks[1] if chunks[1] > 0 else 0:.2f}%[/])"

        result.append(formatted)
    return result


def print_table(
    title,
    lustre_vs_glass,
    lustre_vs_gekko,
    gekkos_vs_glass,
    nodes,
    first_chunk,
    second_chunk,
    last_chunk,
    panel=None,
):
    table = Table(title=title)
    table.add_column("Node", justify="right")
    table.add_column("Lustre / GLASS", style="green")
    table.add_column("Lustre / GekkoFS", style="cyan")
    table.add_column("GekkoFS / GLASS", style="magenta")

    # concat list and find max
    min_value = min(np.concatenate([lustre_vs_glass, lustre_vs_gekko, gekkos_vs_glass]))
    max_value = max(np.concatenate([lustre_vs_glass, lustre_vs_gekko, gekkos_vs_glass]))

    for i in range(
        len(nodes)
    ):  # Use len(nodes) to iterate through the list of node names
        row_values = [
            lustre_vs_glass[i],
            lustre_vs_gekko[i],
            gekkos_vs_glass[i],
        ]
        chunks = [first_chunk[i], second_chunk[i], last_chunk[i]]
        highlighted = highlight_row(row_values, chunks, min_value, max_value)
        table.add_row(nodes[i], *highlighted)  # Use the node name directly

    if panel:
        color = "red"
        if "max" in panel.lower():
            color = "green"
        CONSOLE.print(Panel.fit(table, title=panel, border_style=color))
    else:
        CONSOLE.print(table)


def stat(app, stage_out, stage_in, nodes, panel=""):
    # Convert lists to numpy arrays
    app = np.array(app)
    stage_out = np.array(stage_out)
    stage_in = np.array(stage_in)

    # Individual speedups
    (
        glass_app_speedup,
        gekko_lustre_app_speedup,
        glass_vs_gekko_app_speedup,
        first_chunk_app,
        second_chunk_app,
        last_chunk_app,
    ) = speed_up(app, nodes)
    (
        glass_out_speedup,
        gekko_lustre_out_speedup,
        glass_vs_gekko_out_speedup,
        first_chunk_out,
        second_chunk_out,
        last_chunk_out,
    ) = speed_up(stage_out, nodes)
    (
        glass_in_speedup,
        gekko_lustre_in_speedup,
        glass_vs_gekko_in_speedup,
        first_chunk_in,
        second_chunk_in,
        last_chunk_in,
    ) = speed_up(stage_in, nodes)

    # Combined array and speedups
    combined = app + stage_out + stage_in
    (
        glass_comb_speedup,
        gekko_lustre_comb_speedup,
        glass_vs_gekko_comb_speedup,
        first_chunk_comb,
        second_chunk_comb,
        last_chunk_comb,
    ) = speed_up(combined, nodes)

    # Print all tables with the adjusted node list
    print_table(
        "App Speedup",
        glass_app_speedup,
        gekko_lustre_app_speedup,
        glass_vs_gekko_app_speedup,
        nodes,
        first_chunk_app,
        second_chunk_app,
        last_chunk_app,
        panel,
    )
    print_table(
        "Stage Out Speedup",
        glass_out_speedup,
        gekko_lustre_out_speedup,
        glass_vs_gekko_out_speedup,
        nodes,
        first_chunk_out,
        second_chunk_out,
        last_chunk_out,
        panel,
    )
    print_table(
        "Stage In Speedup",
        glass_in_speedup,
        gekko_lustre_in_speedup,
        glass_vs_gekko_in_speedup,
        nodes,
        first_chunk_in,
        second_chunk_in,
        last_chunk_in,
        panel,
    )
    print_table(
        "Combined Speedup (App + Out + In)",
        glass_comb_speedup,
        gekko_lustre_comb_speedup,
        glass_vs_gekko_comb_speedup,
        nodes,
        first_chunk_comb,
        second_chunk_comb,
        last_chunk_comb,
        panel,
    )


def stat_with_min_max(
    app,
    stage_out,
    stage_in,
    node,
    app_error_diff,
    stage_out_error,
    stage_in_error,
):
    console = Console()
    console.print("[bold blue]--- Average  Speed-up  ----[/]")
    stat(app, stage_out, stage_in, node)

    if (
        any(x != 0 for x in app_error_diff)
        or any(x != 0 for x in stage_in_error)
        or any(x != 0 for x in stage_out_error)
    ):
        app = np.array(app)
        stage_out = np.array(stage_out)
        stage_in = np.array(stage_in)
        app_error_diff = np.array(app_error_diff)
        stage_out_error = np.array(stage_out_error)
        stage_in_error = np.array(stage_in_error)
        console.print(
            Panel.fit(
                "[bold blue]--- Max Speed-up  ----[/]",
                style="white",
                border_style="green",
            )
        )
        stat(
            app + app_error_diff,
            stage_out + stage_out_error,
            stage_in + stage_in_error,
            node,
            "Max",
        )
        console.print(
            Panel.fit(
                "[bold blue]--- Min Speed-up  ----[/]",
                style="white",
                border_style="red",
            )
        )
        stat(
            app - app_error_diff,
            stage_out - stage_out_error,
            stage_in - stage_in_error,
            node,
            "Min",
        )
