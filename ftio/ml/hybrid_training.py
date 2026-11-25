"""
Training and prediction functions for the hybrid model.

This module provides functions for training and prediction using the hybrid
Transformer-LSTM model for I/O bandwidth prediction.

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

import numpy as np
import pandas

from ftio.cli import ftio_core
from ftio.ml.hybrid_model import BandwidthDataset, HybridModel, spectral_loss

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
else:
    raise RuntimeError(
        "Torch module not found. Please install it using 'make full' or 'pip install ftio[ml-libs]'."
    )


def train_hybrid_model(
    file_or_directory,
    bandwidth_cutoff=-1,
    load_state_dict_and_optimizer_state=None,
    emb_dim=128,
    n_heads=4,
    ff_dim=128,
    num_layers=3,
    epochs=10,
    lr=1e-3,
    save=False,
    additional_ftio_args=None,
) -> HybridModel:
    """Trains the hybrid model contained in this file and saves the trained model as a .pth file containing
        the model's state dict and optimizer state.


    Args:
        file_or_directory: Path to either a singular file or a directory of files to train the model with.
        bandwidth_cutoff: Any partial bandwidth sequence below this thresh hold will be cut off from the training data.
        load_state_dict_and_optimizer_state: .pth file checkpoint to continue training the model from
        emb_dim: embedded dimensions of the model
        n_heads: heads of the model
        ff_dim: feedforward dimension of the model
        num_layers: number of layers of the model
        epochs: number of epochs used to train the model
        lr: intensity of the learn rate
        save: boolean representing if the model is supposed to be saved. Doesn't affect the return value.
        additional_ftio_args: additional supported ftio arguments that aren't the initial ftio call and files

    Returns:
          HybridModel trained with provided data

    """

    # checks if file_or_directory is either just a singular file OR a path to a directory with files
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

    if not files:
        raise ValueError("No file(s) found.")

    # the initialised model
    model = HybridModel(
        emb_dim=emb_dim, n_heads=n_heads, ff_dim=ff_dim, num_layers=num_layers
    )
    load_optimizer = None

    # in case the function caller wants to train a previously trained model
    if load_state_dict_and_optimizer_state is not None:
        loaded_data = torch.load(load_state_dict_and_optimizer_state)
        model.load_state_dict(loaded_data["model_state_dict"])
        load_optimizer = loaded_data["optimizer_state_dict"]

    for x in files:
        set = []
        frequency = []
        # extraction of data through frequency analysis for darshan & json
        if x.endswith(".darshan") | x.endswith(".jsonl"):
            cmd_input = ["ftio", x]
            if additional_ftio_args is not None:
                cmd_input += additional_ftio_args
            frequency, set = extract(cmd_input=cmd_input)

        # extraction from .csv files of the sdumont traces ; not a general method to extract from any .csv file
        if x.endswith(".csv"):
            csv_file = pandas.read_csv(x)
            n = 0
            for y in csv_file["both"]:
                set.append([y])
                n = n + 1

                # max size of positional encoding of the model, limitation due to the system used for computation
                if n == 5555:
                    break

        dataset = BandwidthDataset([set], num_parts=frequency)
        load_optimizer = train_model(
            model, dataset, epochs=epochs, lr=lr, load_optimizer=load_optimizer
        )
        predict_next_sequence(model, "", only_data=(set, frequency))

    # save the model
    # currently saving it in the currently active archive
    if save:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": load_optimizer,
            },
            "model_and_optimizer.pth",
        )

    return model


def train_model(model, dataset, epochs=3, lr=1e-3, load_optimizer=None, optimizer=None):
    """Training method for the hybrid model.

    Args:
        model: The hybrid model to be trained.
        dataset: The bandwidth dataset contains all datapoints to train the model.
        epochs: The amount of epochs used to train the model.
        lr: The learn rate of the model.
        load_optimizer: Optimizer dic in case the model was previously trained and not lose the optimizer state.
        optimizer: The optimizer to be used in the training process.

    Returns:
        The state dict of the optimizer for saving/reusing purposes.

    """
    # check if customizer needs to be loaded or initialized
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if load_optimizer is not None:
        optimizer.load_state_dict(load_optimizer)

    # beginning of the training loop
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for past, future in dataset:
            past = past.unsqueeze(0)
            future = future.unsqueeze(0)
            # zero gradient
            optimizer.zero_grad()
            pred_future = model(past, prediction_length=future.size(1))
            # loss & backwards propagation
            loss = (0.3 * torch.nn.functional.huber_loss(pred_future, future)) + (
                0.7 * spectral_loss(pred_future, future)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        # just in case to not divide by zero, it shouldn't happen but making sure it doesn't
        if len(dataset) == 0:
            a = 1
        else:
            a = len(dataset)
        scheduler.step(total_loss / a)
        # provides the total loss for some insight during the epochs IF a human is overseeing the training
        if epoch % 2 == 0:
            print(f"Last Loss: {total_loss:.4f}")
    return optimizer.state_dict()


def predict_next_sequence(
    model_or_pthfile,
    file_or_directory,
    bandwidth_cutoff=0,
    additional_ftio_args=None,
    only_data=None,
):
    """Entry point of the prediction process.

    Args:
        model_or_pthfile: Accepts either a HybridModel or a .pth file (to initialize a model) to use for the prediction.
        file_or_directory: Path to either a singular file or a directory of files to train the model with.
        bandwidth_cutoff: Any bandwidth below this thresh hold will be cut off from the data used for prediction.
        additional_ftio_args: additional supported ftio arguments that aren't the initial ftio call and files
        only_data: in the case of directly using extracted data instead of files or a path to files
    Returns:
        A list of the following form [[ranks, bandwidths], ...] where inner lists represent singular predictions.
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

    # in case no files nor data was supplied
    if (not files) and (only_data is None):
        print("No files found. Aborting prediction.")
        return []

    # either use the given model or instantiate with .pth file or test with untrained model
    model = None
    if isinstance(model_or_pthfile, HybridModel):
        model = model_or_pthfile
    elif model_or_pthfile.endswith(".pth"):
        model = HybridModel(emb_dim=128, n_heads=4, ff_dim=128, num_layers=3)
        model.load_state_dict((torch.load(model_or_pthfile))["model_state_dict"])

    # prediction with an untrained model... questionable but maybe someone desires this as a test?
    if model is None:
        model = HybridModel(emb_dim=128, n_heads=4, ff_dim=128, num_layers=3)

    predictions = []

    if only_data is not None:
        set, frequency = only_data
        dataset = BandwidthDataset([set], num_parts=frequency)

        # min-max normalization
        min1, max1 = min(set)[0], max(set)[0]
        for u in range(len(set)):
            set[u][0] = (set[u][0] - min1) / (max1 - min1 + 1e-8)

        # prediction process
        prediction_current = _predict_next_trace(model, dataset)
        predictions.append(prediction_current)

        actual = []
        for c in dataset.data[0][1]:
            actual.append([c[0]])
        return predictions

    for x in files:
        set = []
        frequency = []
        # extraction of data through frequency analysis for darshan & json
        if x.endswith(".darshan") | x.endswith(".jsonl"):
            cmd_input = ["ftio", x]
            if additional_ftio_args is not None:
                cmd_input += additional_ftio_args
            frequency, set = extract(cmd_input=cmd_input)
        # extract data from traces obtained from sdumont dataset, most certainly not an approach for any .csv file
        if x.endswith(".csv"):
            csv_file = pandas.read_csv(x)
            n = 0
            for y in csv_file["both"]:
                set.append([y])
                n = n + 1

                # size limit on local machine, can be changed IF positional encoding in hybrid model is also changed
                if n == 5555:
                    break
        # min-max normalization
        min1, max1 = min(set)[0], max(set)[0]
        for u in range(len(set)):
            set[u][0] = (set[u][0] - min1) / (max1 - min1 + 1e-8)

        # prediction
        dataset = BandwidthDataset([set], num_parts=frequency)
        prediction_current = _predict_next_trace(model, dataset)
        predictions.append(prediction_current)

    return predictions


def _predict_next_trace(model, dataset):
    """Uses the provided hybrid model to predict the next bandwidth sequence.

    Args:
        model: the hybrid model of this file
        dataset: dataset containing the bandwidth sequences used for the prediction

    """

    model.eval()
    with torch.no_grad():
        prediction = []
        for past, future in dataset:
            # prediction
            squeezed_past = past.unsqueeze(0)
            pred_future = model(squeezed_past, prediction_length=future.size(0))
            pred_sequence = pred_future.squeeze(0).tolist()
            # only positive values are interesting
            pred_sequence = np.abs(pred_sequence)

            min_val, max_val = pred_sequence.min(), pred_sequence.max()
            pred_sequence = (pred_sequence - min_val) / (max_val - min_val + 1e-8)

            prediction.append(pred_sequence)

            # currently assumes that past and future have the same length
            # calculate MAE, MSE and RMSE
            mae = 0
            mse = 0
            for actual, predicted in zip(future, pred_sequence, strict=False):
                inner = actual.item() - predicted[0]
                mae = mae + abs(inner)
                mse = mse + (inner * inner)

            n = future.size(dim=0)
            if n == 0:
                n = 1
            mae = mae / n
            mse = mse / n

            print("Predicted Bandwidth Trace:")
            print(pred_sequence)
    return prediction


def extract(cmd_input, msgs=None) -> list:
    """Extraction method leveraging frequency analysis for the initial data in dataframe form. And calculates the amount
        of expected partial patterns

    Args:
        cmd_input: The ftio arguments in the form of a list of strings.
        msgs: ZMQ message (not used / no use intended yet)

    Returns:
        list[ n, list[bandwidth], ...]
    """
    result = []
    n = 3
    cmd_input.append("--machine_learning")
    prediction_list, args = ftio_core.main(cmd_input)
    # take only the latest for now
    if prediction_list:
        prediction = prediction_list[-1]
        b_sampled = prediction._b_sampled
        for x in b_sampled:
            result.append([x])
        # calculates the amount of partial patterns using the predicted dominant frequency of FTIO
        if (prediction.dominant_freq.size != 0) and (prediction.dominant_freq[0] != 0):
            n = int(prediction.t_end / (1 / prediction.dominant_freq[0]))
    return [n, result]
