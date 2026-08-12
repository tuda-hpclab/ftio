import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            output_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


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
        output_dim = trg.size(2)

        hidden, cell = self.encoder(src)

        # Constancy is baseline assumption. Predicting deviation
        input = torch.zeros_like(src[:, -1:, :])

        outputs = torch.zeros(batch_size, trg_len, output_dim, device=self.device)

        for i in range(trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, i : i + 1, :] = output
            if not self.training:
                teacher_forcing_ratio = 0.0
            teacher_force = trg is not None and random.random() < teacher_forcing_ratio
            input = trg[:, i : i + 1, :] if teacher_force else output
        return outputs

    def predict(self, src, pred_len):
        batch_size = src.size(0)
        output_dim = src.size(2)

        outputs = torch.zeros(batch_size, pred_len, output_dim, device=self.device)
        hidden, cell = self.encoder(src)

        input = torch.zeros_like(src[:, -1:, :])

        for i in range(pred_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, i : i + 1, :] = output
            input = output
        return outputs


def train(device, model, dataloader, crit, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for series in dataloader:
            series = series.to(device)
            src, trg = series[:, : series.size(1) // 2], series[:, series.size(1) // 2 :]
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            trg_delta = trg - src[:, -1:, :]
            output = model(src, trg_delta)

            loss = crit(output, trg_delta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * src.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_loss)
    return losses


@torch.no_grad()
def evaluate(device, model, valid_dataloader):
    model.eval()

    evaluation = []
    baseline = []
    for series in valid_dataloader:
        series = series.to(device)
        src, trg = series[:, : series.size(1) // 2], series[:, series.size(1) // 2 :]
        src = src.to(device)
        trg = trg.to(device)

        output = model.predict(src, trg.size(1)) + src[:, -1:, :]

        log_loss = output - trg
        constant_loss = src[:, -1:, :] - trg

        evaluation.append(log_loss)
        baseline.append(constant_loss)

    return baseline, evaluation


if __name__ == "__main__":

    path = ""
    list_of_freqs = []
    keys = ["dominant_freq", "conf", "amp", "phi"]
    for root, dirs, files in os.walk(path):
        for file in files:

            path_to = root + "/" + file
            path_to = path_to.replace("//", "/")

            content = None
            with open(path_to) as f:
                content = json.load(f)
            if (not content) or (len(content[0]["dominant_freq"]) < 3):
                continue
            for k in keys:
                content[0][k].pop(0)

            list_of_freqs.append(content[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    releveant_keys = ["X"]
    input_size = 1
    output_size = 1
    hidden_size = 64 * 2
    num_layers = 2
    dropout = 0.1
    input_target_base = 1
    batch_size = 50

    # Initialize model and send to device
    encoder = Encoder(input_size, hidden_size, num_layers, dropout)
    decoder = Decoder(output_size, hidden_size, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)

    x = np.linspace(0, 5, 100)
    df_test = pd.DataFrame({"x": x, "sine": np.random.normal(0, 0.5, 100) + np.sin(x)})

    df_tr = [
        np.log(x["dominant_freq"])
        for x in list_of_freqs[: 9 * (len(list_of_freqs) // 10)]
    ]
    df_ev = [
        np.log(x["dominant_freq"])
        for x in list_of_freqs[9 * (len(list_of_freqs) // 10) :]
    ]

    dataloader_tr = DataLoader(TimeSeriesDataset(df_tr))
    dataloader_ev = DataLoader(TimeSeriesDataset(df_ev))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    res = train(device, model, dataloader_tr, criterion, optimizer, num_epochs)

    baseline, evaluation = evaluate(device, model, dataloader_ev)

    baseline = [x.cpu().abs().mean() for x in baseline]
    evaluation = [x.cpu().abs().mean() for x in evaluation]
