"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import os  # To handle file path operations
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def main(argv=sys.argv[1:]):
    # Parse command-line arguments for file paths
    if not argv:
        argv = ["/d/traces/traces_from_plafrim/ftio_flat.csv"]

    parser = argparse.ArgumentParser(description="Process some CSV files.")
    parser.add_argument(
        "file_paths",
        metavar="F",
        type=str,
        nargs="+",
        help="Paths to the CSV files",
    )
    args = parser.parse_args(argv)

    # List to store the DataFrames and filenames
    dataframes = []
    file_basenames = []

    # Read the CSV files into DataFrames and store their basenames
    for file_path in args.file_paths:
        df = pd.read_csv(file_path)
        basename = os.path.basename(file_path)  # Extract basename from file path
        basename = os.path.splitext(basename)[0]  # Remove extension
        df["Source File"] = basename  # Add a new column with the basename
        dataframes.append(df)
        file_basenames.append(basename)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Reshape the DataFrame to long format
    # Use "Source File" for the color coding and melt the DataFrame
    combined_df_long = pd.melt(
        combined_df,
        id_vars=["Source File"],
        var_name="Metric",
        value_name="Value",
    )

    # plot
    # single_plot(combined_df_long)
    sub_plots(combined_df_long)


def single_plot(df):
    # Use Plotly Express to create a box plot with color by filename
    fig = px.box(
        df,
        x="Metric",  # Column name for x-axis
        y="Value",  # Column name for y-axis
        color="Source File",  # Use the filename column for color
        title="Combined Box Plots by Metric",
        labels={"Metric": "Metric", "Value": "Values"},
    )  # Labels for the axes

    # Update layout
    fig.update_layout(
        boxmode="group",  # Group the boxes for better comparison
        height=600,
        width=1200,
        xaxis_title="Metric",
        yaxis_title="Values",
    )

    # Show the plot
    fig.show()


def sub_plots(df):
    # Get unique metrics
    metrics = df["Metric"].unique()
    n = len(df["Source File"].unique())
    metrics = metrics[:-2]  # remove jobid and file name
    num_metrics = len(metrics)

    order = []
    subplot_titles = []
    fields = ["read", "write", "both"]
    for i, field in enumerate(fields):
        row = 0
        for _j, metric in enumerate(metrics):
            if field in metric:
                row += 1
                subplot_titles.append(metric)
                order.append([i + 1, row])

    suffixes = []
    for title in subplot_titles:
        suffix = title.split("_", 1)[1]  # Split only at the first underscore
        if suffix not in suffixes:
            suffixes.append(suffix)

    # Reorder the list based on the prefix order and original suffix order
    subplot_titles = [
        f"{prefix}_{suffix}"
        for suffix in suffixes
        for prefix in fields
        if f"{prefix}_{suffix}" in subplot_titles
    ]

    col = 0
    for i in order:
        col = max(i[0], col)

    row = int(np.ceil(num_metrics / (col)))

    # Create subplots with each row representing a metric
    fig = sp.make_subplots(rows=row, cols=col, subplot_titles=subplot_titles)

    # Loop through each metric and add a box plot to the corresponding subplot
    for i, metric in enumerate(metrics):
        # Filter data for the current metric
        metric_df = df[df["Metric"] == metric]

        # Create a box plot for the current metric
        box_trace = go.Box(
            x=metric_df["Source File"],
            y=metric_df["Value"],
            name=metric,
            # hide points
            boxpoints=False,
            # std and mean
            boxmean="sd",
        )
        # Clean the 'Value' column: Convert to numeric, coerce errors to NaN
        # metric_df['Value'] = pd.to_numeric(metric_df['Value'], errors='coerce')
        metric_df.loc[:, "Value"] = pd.to_numeric(metric_df["Value"], errors="coerce")

        # Drop rows where 'Value' could not be converted to numeric (i.e., NaN)
        metric_df = metric_df.dropna(subset=["Value"])

        # Add the box plot to the appropriate subplot
        fig.add_trace(box_trace, row=order[i][1], col=order[i][0])
        for s in metric_df["Source File"].unique():
            fig.add_annotation(
                x=s,  # Use specific source file as x coordinate
                y=metric_df[metric_df["Source File"] == s]["Value"].max(),
                text=str(len(metric_df[metric_df["Source File"] == s]["Value"])),
                yshift=10,
                showarrow=False,
                col=order[i][0],
                row=order[i][1],
            )

    # Update layout for the entire figure
    fig.update_layout(
        title="Box Plots by Metric",
        height=400 * row,  # Adjust height for number of metrics
        width=300 + 80 * col * n,  # Adjust width for visibility
        xaxis_title="Source File",
        yaxis_title="Values",
        showlegend=False,  # Hide legend to reduce clutter
    )

    # Show the plot
    fig.show()
    print(f"Saving HTML filer out.html in {os.getcwd()}")
    fig.write_html(f"{os.getcwd()}/out.html")
    # print(
    #     f"Saving online HTML filer out_online.html in {os.getcwd()}."
    #     f"This file needs an internet connection to be viewed for plotly js"
    #     )
    # fig.write_html(f"{os.getcwd()}/out_online.html",include_plotlyjs=False)


if __name__ == "__main__":
    main(sys.argv[1:])
