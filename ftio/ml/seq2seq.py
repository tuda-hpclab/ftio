import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ftio.cli import ftio_core
from ftio.ml import seq2seq_GRU
from ftio.ml.Dataloaders import TimeSeriesDataset


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )

    def forward(self, input):
        output, (hidden, cell) = self.lstm(input)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.prob = torch.nn.Linear(hidden_size, 1)
        self.mag = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # prediction = self.fc(output)
        return self.prob(output), self.mag(output), hidden, cell


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        if trg is None:
            raise ValueError("No target provided")
        trg_len = trg.size(1)
        output_dim = self.decoder.output_size

        hidden, cell = self.encoder(src)

        # Constancy is baseline assumption. Predicting deviation
        input = torch.zeros(batch_size, 1, self.decoder.input_size, device=src.device)

        # outputs = torch.zeros(batch_size, trg_len, output_dim, device=src.device)
        prob = torch.zeros(batch_size, trg.size(1), 1, device=src.device)
        mag = torch.zeros(batch_size, trg.size(1), 1, device=src.device)

        for i in range(trg_len):
            output_prob, output_mag, hidden, cell = self.decoder(input, hidden, cell)
            prob[:, i : i + 1], mag[:, i : i + 1] = output_prob, output_mag
            if not self.training:
                teacher_forcing_ratio = 0.0
            teacher_force = trg is not None and random.random() < teacher_forcing_ratio
            input = (
                trg[:, i : i + 1, :]
                if teacher_force
                else (torch.sigmoid(output_prob) > 0.5).float() * output_mag
            )
        return prob, mag

    def predict(self, src, pred_len):
        batch_size = src.size(0)
        output_dim = self.decoder.output_size

        outputs_prob_total = torch.zeros(
            batch_size, pred_len, output_dim, device=src.device
        )
        outputs_mag_total = torch.zeros(
            batch_size, pred_len, output_dim, device=src.device
        )

        hidden, cell = self.encoder(src)

        input = torch.zeros(batch_size, 1, self.decoder.input_size, device=src.device)

        for i in range(pred_len):
            output_prob, output_mag, hidden, cell = self.decoder(input, hidden, cell)
            outputs_prob_total[:, i : i + 1], outputs_mag_total[:, i : i + 1] = (
                output_prob,
                output_mag,
            )
            input = (torch.sigmoid(output_prob) > 0.5).float() * output_mag
        return outputs_prob_total, outputs_mag_total


def train(device, model, dataloader, crit, optimizer, num_epochs):
    """
    Trains model for number of epochs. Returns losses per epoch.

    Args:
        device: Device that is running the training
        model: Model to be trained
        dataloader: Provided dataset
        crit: Criterion for loss
        optimizer: Optimizer used for trainng
        num_epochs: Number of full iterations over dataloader

    Returns:
        losses: List of [sigma(loss)/len(dataset)]
    """
    losses = []
    for epoch in range(num_epochs):
        delts = []
        model.train()
        epoch_loss = 0
        for series in dataloader:
            series = series.to(device)
            src, trg = series[:, : series.size(1) // 2], series[:, series.size(1) // 2 :]
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            trg_delta = trg[:, :, :1] - src[:, -1:, :1]
            output_prob, output_mag = model(src, trg_delta)

            change = (trg_delta.abs() > 1e-6).float()
            delts.append(trg_delta[change.bool()].detach().cpu())
            loss_prob = torch.nn.functional.binary_cross_entropy_with_logits(
                output_prob, change, pos_weight=torch.tensor([20.0], device=device)
            )
            loss_mag = (
                ((output_mag - trg_delta) ** 2) * change
            ).sum() / change.sum().clamp(min=1.0)
            loss = loss_prob + 1.0 * loss_mag
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * src.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
    return losses


@torch.no_grad()
def evaluate(device, model, valid_dataloader, threshold):
    """
    Use prediction to evaluate the model. Report summarizes false / true positives.

    Args:
            device: Device that is running the evaluation
            model: Model used for evaluation
            valid_dataloader: Provided dataset
            threshold: Thresholding for probability of change

        Returns:
            baseline: List of differences between assumption of constancy and real values
            evaluation: List of differences between predicted values and real values
            res_error: Report including false / true positives, missed events and total length
    """
    model.eval()

    evaluation = []
    baseline = []
    for series in valid_dataloader:
        series = series.to(device)
        src, trg = series[:, : series.size(1) // 2], series[:, series.size(1) // 2 :]
        src = src.to(device)
        trg = trg.to(device)

        output_prob, output_mag = model.predict(src, trg.size(1))

        prediction = src[:, -1:, :1] + torch.where(
            torch.sigmoid(output_prob) > threshold,
            output_mag,
            torch.zeros_like(output_mag),
        )

        log_loss = (prediction - trg[:, :, :1]).reshape(-1).cpu()
        constant_loss = (src[:, -1:, :1] - trg[:, :, :1]).reshape(-1).cpu()

        evaluation.append(log_loss)
        baseline.append(constant_loss)

    base = torch.cat(baseline).numpy()
    evalu = torch.cat(evaluation).numpy()

    res_error = {
        "length": len(base),
        "events": int((np.abs(base) > 1e-9).sum()),
        "ratio": 1 - np.abs(evalu).sum() / np.abs(base).sum(),
        "true_pos": int(((np.abs((evalu - base)) > 1e-9) & (np.abs(base) > 1e-9)).sum()),
        "false_pos": int(
            ((np.abs((evalu - base)) > 1e-9) & ~(np.abs(base) > 1e-9)).sum()
        ),
        "missed": int((~(np.abs((evalu - base)) > 1e-9) & (np.abs(base) > 1e-9)).sum()),
    }

    return baseline, evaluation, res_error


def log_z_score(pred_list, keys, stats=None):
    """
    Applies logarithmic scaling and z-scores the data. stats can be provided to score based on pre-existing statistics.

        Args:
            pred_list: List of dictionaries containing data
            keys: Keys that scoring should be applied to
            stats: Mean and standard deviation used for scoring

        Returns:
            pred_list: Normalized data
            stat_value: Means and standard deviations of the given dataset. Always returns the true mean and standard deviation, regardless of provided stats.
    """

    cumalitive = {}
    for k in keys:
        cumalitive[k] = []

    for pred in pred_list:
        for k in keys:
            pred[k] = np.log(np.clip(pred[k], 1e-8, None))
            pred[k][np.isnan(pred[k])] = 0.0
            cumalitive[k].append(pred[k])

    stat_value = {}
    for k in keys:
        stat_value[k] = {
            "mean": np.array(np.concatenate(cumalitive[k])).mean(),
            "std": np.array(np.concatenate(cumalitive[k])).std(),
        }

    for pred in pred_list:
        for k in keys:
            k_np = np.array(pred[k])
            if stats == None:
                if stat_value[k]["std"] == 0.0:
                    raise ValueError("Divide by zero")
                pred[k] = (k_np - stat_value[k]["mean"]) / stat_value[k]["std"]
            else:
                if stats[k]["std"] == 0.0:
                    raise ValueError("Divide by zero")
                pred[k] = (k_np - stats[k]["mean"]) / stats[k]["std"]
    return pred_list, stat_value


if __name__ == "__main__":

    list_of_freqs = []
    keys = ["dominant_freq", "conf", "amp", "phi"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    releveant_keys = ["dominant_freq", "conf", "amp"]
    input_size = 3
    output_size = 1
    hidden_size = 64
    num_layers = 1
    dropout = 0.1
    input_target_base = 1
    batch_size = 1

    list_of_freqs = [{y: x[y] for y in releveant_keys} for x in list_of_freqs]

    # Initialize model and send to device
    encoder = Encoder(input_size, hidden_size, num_layers, dropout)
    decoder = Decoder(1, output_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)

    random.Random(0).shuffle(list_of_freqs)
    df_tr = list_of_freqs[: 9 * (len(list_of_freqs) // 10)]
    df_ev = list_of_freqs[9 * (len(list_of_freqs) // 10) :]

    df_tr, stat_values = log_z_score(df_tr, ["dominant_freq", "amp"])
    df_ev, _ = log_z_score(df_ev, ["dominant_freq", "amp"], stats=stat_values)

    df_tr = [
        np.stack([np.asarray(item[k], dtype=np.float32) for k in releveant_keys], axis=1)
        for item in df_tr
    ]
    df_ev = [
        np.stack([np.asarray(item[k], dtype=np.float32) for k in releveant_keys], axis=1)
        for item in df_ev
    ]

    dataloader_tr = DataLoader(TimeSeriesDataset(df_tr), batch_size=batch_size)
    dataloader_ev = DataLoader(TimeSeriesDataset(df_ev), batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    tr_start = time.time()
    res = train(device, model, dataloader_tr, criterion, optimizer, num_epochs)
    tr_end = time.time()

    threshold = 0.7
    baseline, evaluation, res_error = evaluate(device, model, dataloader_ev, threshold)
    ev_end = time.time()
