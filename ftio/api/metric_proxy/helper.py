"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import json

import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def extract_data(data):
    """Extracts relevant data that is not NaN

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Prepare the data for the plot
    data_points = []

    for prediction in data:
        if len(prediction.dominant_freq) > 0 and len(prediction.conf) > 0:
            max_conf_index = np.argmax(prediction.conf)
            dominant_freq = prediction.dominant_freq[max_conf_index]
            conf = prediction.conf[max_conf_index] * 100
            phi = prediction.phi[max_conf_index]  # np.degrees(d['phi'][max_conf_index])
            amp = prediction.amp[max_conf_index]
            t_s = prediction.t_start
            t_e = prediction.t_end
            data_points.append(
                (prediction.metric, dominant_freq, conf, amp, phi, t_s, t_e)
            )
        else:
            continue

    # Create a DataFrame for the plot
    df = pd.DataFrame(
        data_points,
        columns=[
            "Metric",
            "Dominant Frequency",
            "Confidence",
            "Amp",
            "Phi",
            "time start",
            "time end",
        ],
    )
    df.sort_values(by="Dominant Frequency", inplace=True)

    return df


class NpArrayEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            np.nan_to_num(x=obj, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return obj.tolist()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


def data_to_json(data: list[dict]) -> None:
    print(json.dumps(data, cls=NpArrayEncode))


def create_process_bar(total_files):

    # Create a progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description} ({task.completed}/{task.total})"
        ),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        "[yellow]-- runtime",
        TimeElapsedColumn(),
    )
    return progress
