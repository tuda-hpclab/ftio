import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )

    def forward(self, input):
        output, hidden = self.rnn(input)
        return hidden, None


class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = torch.nn.GRU(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.prob = torch.nn.Linear(hidden_size, 1)
        self.mag = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, cell):
        output, hidden = self.rnn(input, hidden)
        # prediction = self.fc(output)
        return self.prob(output), self.mag(output), hidden, None


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
