"""
ARIMA and SARIMA models for time series prediction.

This module provides training and prediction functions using ARIMA
(AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA)
models for I/O bandwidth prediction.

Author: Robert Alles
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import importlib.util
import os
from math import inf

import numpy as np
import pandas as pd

from ftio.ml.hybrid_training import extract

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss
else:
    raise RuntimeError(
        "Torch module not found. Please install it using 'make full' or 'pip install ftio[ml-libs]'."
    )


def train_arima(
    file_or_directory, max_depth=3, model_architecture="SARIMA", additional_ftio_args=None
):
    """The entry point for training the ARIMA and SARIMA models

    Args:
        file_or_directory: the file or directory to train the ARIMA model
        max_depth: the maximum depth of the ARIMA model
        model_architecture: choose between SARIMA or ARIMA architecture
        additional_ftio_args: additional supported ftio arguments that aren't the initial ftio call and files

    """
    # checks if file_or_path is either just a singular file OR a path to a directory with files
    # and then saves the individual files paths
    files = []
    if os.path.isfile(file_or_directory):
        files.append(file_or_directory)
    elif os.path.isdir(file_or_directory):
        files = [
            os.path.join(file_or_directory, f)
            for f in os.listdir(file_or_directory)
            if os.path.isfile(os.path.join(file_or_directory, f))
        ]

    # in the case of no supplied files or filepath
    if not files:
        print("No files found. Aborting prediction.")
        return []

    # use extract method for extraction of data
    predictions = []
    sequences = []
    for file in files:
        cmd_input = ["ftio", file]
        if additional_ftio_args is not None:
            cmd_input += additional_ftio_args
        n, bandwidth_sequence = extract(cmd_input=cmd_input)
        sequences.append([n, bandwidth_sequence])

    for sequence in sequences:
        # initial min-max normalization
        min_val_initial, max_val_initial = inf, -inf
        for value in sequence[1]:
            if value[0] < min_val_initial:
                min_val_initial = value[0]
            if value[0] > max_val_initial:
                max_val_initial = value[0]

        for value in sequence[1]:
            value[0] = (value[0] - min_val_initial) / (
                max_val_initial - min_val_initial + 1e-8
            )

        # split data into trainings and comparison data
        split_index = int(len(sequence[1]) / sequence[0]) * (sequence[0] - 1)
        trainings_part = sequence[1][:split_index]
        future_part = sequence[1][split_index:]
        d = 0
        # main loop to apply KPSS and ADF to find depth d
        for y in range(1):
            partial_sequence = pd.Series(
                [x[y] for x in trainings_part],
                pd.Index(np.arange(len([x[y] for x in trainings_part]), dtype="int64")),
            )

            try:
                kps = kpss(partial_sequence.dropna())[1]
            except Exception:
                kps = 0.0

            try:
                adf = adfuller(partial_sequence.dropna())[1]
            except Exception:
                adf = 1.0

            while d is not max_depth and kps > 0.05 > adf:
                partial_sequence = partial_sequence.diff()
                d = d + 1

                try:
                    kps = kpss(partial_sequence.dropna())[1]
                except Exception:
                    kps = 0.0

                try:
                    adf = adfuller(partial_sequence.dropna())[1]
                except Exception:
                    adf = 1.0
            # in case NA, INF and -INF values are introduced by differencing
            partial_sequence = partial_sequence.fillna(0.0)
            partial_sequence = partial_sequence.replace([np.inf, -np.inf], 0)

        final_model = None
        best_aic = inf
        # search grid based on AIC, limited to max p = 5 and q = 8, going deeper requires longer computing
        for p in range(5):
            for q in range(8):
                try:
                    if model_architecture == "SARIMA":
                        model = SARIMAX(
                            partial_sequence,
                            order=(p, d, q),
                            seasonal_order=(0, 0, 0, len(future_part)),
                        )
                    else:
                        model = ARIMA(partial_sequence, order=(p, d, q))
                    model = model.fit()

                    # application of the AIC
                    aic = model.aic
                    if aic < best_aic:
                        final_model = model
                        best_aic = aic
                        print("p : d : q", (p, d, q))
                # some variations will throw exceptions and warnings, this will exclude them
                except Exception:
                    continue
        # the forecast
        prediction = final_model.forecast(len(future_part))
        # min-max normalization for uniform displaying of results to actual
        min_val, max_val = prediction.min(), prediction.max()
        prediction = (prediction - min_val) / (max_val - min_val + 1e-8)
        predictions.append(prediction)
    return predictions
