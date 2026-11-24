"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, OPTICS

from ftio.plot.helper import format_plot


def optics(df: pd.DataFrame, s1: str, s2: str) -> pd.DataFrame:
    # Combine x and y into a 2D array
    data = np.column_stack((df[s1], df[s2]))

    # Apply OPTICS clustering
    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
    optics.fit(data)

    # Extract cluster labels
    labels = optics.labels_

    plot_cluster(df, s1, s2, labels)
    # plot_cluster_simple(df,s1,s2,labels)

    return pd.concat([df, pd.DataFrame({"label": labels})], axis=1)


def dbscan(df: pd.DataFrame, s1: str, s2: str, eps=0.5) -> pd.DataFrame:
    # Combine s1 and s2 columns into a 2D array for clustering
    data = np.column_stack((df[s1], df[s2]))

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(data)

    # Extract cluster labels
    labels = dbscan.labels_

    plot_cluster(df, s1, s2, labels, "DBscan")

    return pd.concat([df, pd.DataFrame({"label": labels})], axis=1)


def plot_cluster_simple(
    df: pd.DataFrame, s1: str, s2: str, labels: np.ndarray, method="Optics"
) -> None:
    # Plot the clusters
    fig = px.scatter(
        x=df[s1],
        y=df[s2],
        color=labels.astype(str),
        title=f"{method} Clustering",
        labels={"color": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.show()


def plot_cluster(
    df: pd.DataFrame,
    s1: str,
    s2: str,
    labels: np.ndarray,
    method="Optics",
    unit_s1="rad",
    unit_s2="Hz",
) -> None:
    x = df[s1]
    y = df[s2]

    # Define unique labels and their corresponding colors and symbols
    labels = labels.astype(str)
    for i in range(len(labels)):
        if "-1" in labels[i]:
            labels[i] = "Outliers"
        else:
            labels[i] = f"Cluster {labels[i]}"

    unique_labels = set(labels)
    colors = px.colors.qualitative.Dark24[: len(unique_labels)]
    symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
    ] * (len(unique_labels) // 7 + 1)

    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    symbol_map = {
        label: symbols[i % len(symbols)] for i, label in enumerate(unique_labels)
    }

    # Map each point to its corresponding color and symbol
    point_colors = [color_map[label] for label in labels]
    point_symbols = [symbol_map[label] for label in labels]

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(
        {
            f"{s1} ({unit_s1})": x,
            f"{s2} ({unit_s2})": y,
            "label": labels.astype(str),
            "color": point_colors,
            "symbol": point_symbols,
            "metric": df["Metric"],
        }
    )

    # Plot the clusters using Plotly
    fig = px.scatter(
        data_frame=df_plot,
        x=f"{s1} ({unit_s1})",
        y=f"{s2} ({unit_s2})",
        title=f"{method} Clustering",
        labels={"label": "Clusters"},
        color="label",
        color_discrete_map=color_map,
        symbol="label",
        symbol_map=symbol_map,
        size_max=15,
    )  # Set maximum size for markers

    # Customize the marker size and add hover information
    fig.update_traces(
        marker={"size": 10},
        hovertemplate=(
            "x: %{x}<br>"
            "y: %{y}<br>"
            "Metric: %{customdata}<br>"
            "Label: %{marker.color}<br>"
        ),
    )
    fig.update_layout(uniformtext_mode="hide")
    fig.update_traces(customdata=df["Metric"])
    fig = format_plot(fig)
    fig.show()
