"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import linkage

from ftio.plot.helper import format_plot_and_ticks


def heatmap(data):
    # Prepare the data for the heatmap
    metrics = []
    dominant_freqs = []
    confs = []

    if data:
        # nbins = round(data[0]['freq']*(data[0]['t_end'] - data[0]['t_start']))
        nbins = round(data[0].t_end - data[0].t_start)
    for d in data:
        metric = d.metric
        if len(d.dominant_freq) > 0 and len(d.conf) > 0:
            max_conf_index = np.argmax(d.conf)
            dominant_freq = d.dominant_freq[max_conf_index]
            conf = d.conf[max_conf_index] * 100
        else:
            continue
            dominant_freq = 0
            conf = 0
        metrics.append(metric)
        dominant_freqs.append(dominant_freq)
        confs.append(conf)

    # Convert to DataFrame
    heatmap_data = pd.DataFrame(
        {
            "Metric": metrics,
            "Dominant Frequency": dominant_freqs,
            "Confidence": confs,
        }
    )

    # Ensure data is not empty and contains valid ranges
    if heatmap_data.empty:
        raise ValueError("The heatmap data is empty. Cannot generate heatmap.")

    # Define equally spaced bins for Dominant Frequency
    freq_min, freq_max = (
        heatmap_data["Dominant Frequency"].min(),
        heatmap_data["Dominant Frequency"].max(),
    )
    if freq_min == freq_max:
        raise ValueError(
            "The range for Dominant Frequency binning is invalid. Minimum and maximum values are the same."
        )
    freq_bins = np.linspace(freq_min, freq_max, num=nbins)  # Example: 4 bins

    # Bin the Dominant Frequency data
    heatmap_data["Dominant Frequency Binned"] = pd.cut(
        heatmap_data["Dominant Frequency"], bins=freq_bins, include_lowest=True
    )

    # Convert Interval bins to strings for Plotly
    heatmap_data["Dominant Frequency Binned"] = heatmap_data[
        "Dominant Frequency Binned"
    ].astype(str)
    heatmap_data.sort_values(by="Dominant Frequency", inplace=True, ignore_index=True)

    # Pivot the DataFrame to switch x and y axes
    heatmap_pivot = heatmap_data.pivot_table(
        index="Dominant Frequency Binned",
        columns="Metric",
        values="Confidence",
        aggfunc="mean",
        sort=False,
    )
    # or without sorting
    # heatmap_pivot = heatmap_data.pivot_table(index='Dominant Frequency Binned', columns='Metric', values='Confidence', aggfunc='mean')

    # Create the heatmap with switched axes
    fig = px.imshow(
        heatmap_pivot,
        labels={"x": "Metric", "y": "Dominant Frequency", "color": "Confidence"},
        # text_auto=True,
        origin="lower",
        color_continuous_scale="Viridis",
        aspect="auto",  #'auto' or equal
    )
    fig.update_layout(
        # height=1500,
        xaxis_tickangle=-45,
        margin={"l": 100, "r": 100, "t": 50, "b": 150},
        coloraxis_colorbar={
            "yanchor": "top",
            "y": 1,
            "ticks": "outside",
            "ticksuffix": " %",
            # lenmode="pixels",
            # len=200,
        },
    )
    fig = format_plot_and_ticks(fig, False, True, False, False)
    fig.show()


def scatter(df, x, y, color, symbol) -> None:
    # Create the scatter plot
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        color_continuous_scale="Viridis",
    )
    # Display the plot
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        xaxis_tickangle=-45,
        margin={"l": 100, "r": 100, "t": 50, "b": 150},
        coloraxis_colorbar={
            "orientation": "h",
            "ticks": "outside",
            "ticksuffix": " %",
            "title": "",
        },
    )
    fig.show()


def scatter2D(df) -> None:
    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Metric",
        y="Dominant Frequency",
        color="Confidence",
        labels={
            "Metric": "Metric",
            "Dominant Frequency": "Dominant Frequency",
            "Confidence": "Confidence",
        },
        color_continuous_scale="Viridis",
    )
    # Display the plot
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Dominant Frequency",
        xaxis_tickangle=-45,
        margin={"l": 100, "r": 100, "t": 50, "b": 150},
        coloraxis_colorbar={
            "yanchor": "top",
            "y": 1,
            "ticks": "outside",
            "ticksuffix": " %",
        },
    )
    fig = format_plot_and_ticks(fig, False, True, False, False)
    fig.show()


def heatmap_2(data):
    # Prepare the data for the heatmap
    zscore = True
    metrics = []
    dominant_freqs = []
    confs = []
    counter = 0

    for prediction in data:
        metric = prediction.metric
        if len(prediction.dominant_freq) > 0 and len(prediction.conf) > 0:
            max_conf_index = np.argmax(prediction.conf)
            dominant_freq = prediction.dominant_freq[max_conf_index]
            conf = prediction.conf[max_conf_index]
        else:
            continue
            dominant_freq = 0
            conf = 0
        metrics.append(metric)
        # metrics.append(f"{counter}")
        dominant_freqs.append(dominant_freq)
        confs.append(conf)
        counter += 1

    # Convert to DataFrame
    heatmap_data = pd.DataFrame(
        {
            "Metric": metrics,
            "Dominant Frequency": dominant_freqs,
            "Confidence": confs,
        }
    )

    # Ensure data is not empty and contains valid ranges
    if heatmap_data.empty:
        raise ValueError("The heatmap data is empty. Cannot generate heatmap.")

    # Select the dominant frequency with the highest confidence for each Metric
    dominant_freq_per_metric = heatmap_data.loc[
        heatmap_data.groupby("Metric")["Confidence"].idxmax()
    ]
    dominant_freq_per_metric.sort_values(
        "Dominant Frequency", ignore_index=True, inplace=True
    )

    # Create a DataFrame for the differences in Dominant Frequency
    # metrics_unique = sorted(dominant_freq_per_metric['Metric'].unique())
    metrics_unique = dominant_freq_per_metric["Metric"].tolist()
    heatmap_diff = pd.DataFrame(index=metrics_unique, columns=metrics_unique)

    if zscore:
        mu = np.mean(dominant_freqs)
        sigma = np.std(dominant_freqs)

    for i, metric_i in enumerate(metrics_unique):
        for j, metric_j in enumerate(metrics_unique):
            if i <= j:  # Only calculate for one half of the matrix
                freq_i = dominant_freq_per_metric.loc[
                    dominant_freq_per_metric["Metric"] == metric_i,
                    "Dominant Frequency",
                ].values
                freq_j = dominant_freq_per_metric.loc[
                    dominant_freq_per_metric["Metric"] == metric_j,
                    "Dominant Frequency",
                ].values
                if len(freq_i) > 0 and len(freq_j) > 0:
                    if zscore:
                        Z_i = (freq_i[0] - mu) / sigma
                        Z_j = (freq_j[0] - mu) / sigma
                        diff = abs(Z_i - Z_j) * 100
                    else:
                        diff = (
                            abs(freq_i[0] - freq_j[0])
                            / ((freq_i[0] + freq_j[0]) / 2)
                            * 100
                        )
                        # diff = abs(freq_i[0] - freq_j[0]) * 100
                else:
                    diff = -1
                heatmap_diff.at[metric_i, metric_j] = diff
                heatmap_diff.at[metric_j, metric_i] = diff  # Mirror the value

    # # Fill the diagonal with zeros (or any other value if necessary)
    # for metric in metrics_unique:
    #     heatmap_diff.at[metric, metric] = 0

    # Convert to numeric type
    # heatmap_diff = heatmap_diff.astype(float)
    # top = float(max(heatmap_diff.max(),100))

    # Ensure the index and columns are correctly set to metrics_unique
    heatmap_diff.index = metrics_unique
    heatmap_diff.columns = metrics_unique
    plot_heatmap(heatmap_diff)

    heatmap_diff = heatmap_diff.fillna(0)  # Replace NaN with 0 or another strategy
    # Apply hierarchical clustering using fastcluster
    # linkage_matrix = fastcluster.linkage(
    #     heatmap_diff, method="average", metric="euclidean"
    # )
    linkage_matrix = linkage(heatmap_diff, method="average", metric="euclidean")

    sns.clustermap(
        heatmap_diff,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="viridis",
        annot=False,
        xticklabels=metrics_unique,
        yticklabels=metrics_unique,
        figsize=(12, 10),
    )
    plt.show()


def density_heatmap(data) -> None:
    # Prepare the data for the plot
    data_points = []

    # Calculate number of bins based on data range
    if data:
        t_start = data[0].t_start
        t_end = data[0].t_end
        nbins = round((t_end - t_start) / 10)
    else:
        nbins = 30  # Default value if no data is provided

    for prediction in data:
        if len(prediction.dominant_freq) > 0 and len(prediction.conf) > 0:
            max_conf_index = np.argmax(prediction.conf)
            dominant_freq = prediction.dominant_freq[max_conf_index]
            conf = prediction.conf[max_conf_index] * 100
            data_points.append((prediction.metric, dominant_freq, conf))
        else:
            continue

    # Create a DataFrame for the plot
    df = pd.DataFrame(data_points, columns=["Metric", "Dominant Frequency", "Confidence"])

    # Create the density heatmap
    fig = px.density_heatmap(
        df,
        x="Metric",
        y="Dominant Frequency",
        z="Confidence",
        labels={
            "Metric": "Metric",
            "Dominant Frequency": "Dominant Frequency",
            "Confidence": "Confidence",
        },
        # color_continuous_scale='Viridis',
        nbinsx=nbins,  # Number of bins along the x-axis
        nbinsy=nbins,  # Number of bins along the y-axis
        marginal_y="histogram",
    )

    # Display the plot
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Dominant Frequency",
        xaxis_tickangle=-45,
        coloraxis_colorbar={
            "yanchor": "top",
            "y": 1,
            "ticks": "outside",
            "ticksuffix": " %",
            "title": "Confidence (%)",
        },
        margin={"l": 100, "r": 100, "t": 50, "b": 150},
    )
    fig.show()


def plot_heatmap(heatmap_diff):

    # Create the heatmap with switched axes
    fig = px.imshow(
        heatmap_diff,
        labels={"x": "", "y": "", "color": "Difference in Dominant Frequency"},
        # text_auto=True,
        origin="lower",
        # width=1200,  # Adjust width as needed
        # height=1000   # Adjust height as needed
        aspect="auto",
        # color_continuous_scale='RdBu'
        color_continuous_scale=[
            (0, "black"),
            (0.1, "red"),
            (0.2, "orange"),
            (0.4, "yellow"),
            (0.6, "white"),
            (0.8, "blue"),
            (1, "purple"),
        ],
    )

    # Update layout to adjust margins and spacing
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Metric",
        coloraxis_colorbar={
            "title": "Relative deviation (%)",
            "yanchor": "top",
            "y": 1,
            "ticks": "outside",
            "ticksuffix": " %",
            # tickvals=[0, critical/2, critical, 2*critical, 1/critical],
        },
        xaxis={
            "tickmode": "linear",  # Ensure tick labels are spaced out
            "tickangle": 90,  # Rotate tick labels if they overlap
        },
        margin={
            "l": 100,
            "r": 100,
            "t": 50,
            "b": 150,
        },  # Adjust margins to give more space
    )
    fig = format_plot_and_ticks(fig, False, True, False, False)
    fig.show()


def plot_timeseries_metrics(metrics, width=None, height=None):
    fig = go.Figure()
    for metric, arrays in metrics.items():
        if len(arrays[0]) > 1:
            fig.add_trace(
                go.Scatter(x=arrays[1], y=arrays[0], mode="lines+markers", name=metric)
            )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Metrics",
    )
    if width and height:
        fig.update_layout(autosize=False, width=width, height=height)
    fig.show()
    return fig
