import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# In, Out, Hidden
K, H, D = 16, 4, 32


class Seq2Seq(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.enc = nn.GRU(1, d, batch_first=True)
        self.dec = nn.GRUCell(1, d)
        self.out = nn.Linear(d, 1)

    def forward(self, x, y=None):

        anchor = x[:, -1:]
        _, h = self.enc((x - anchor).unsqueeze(-1))
        h = h[0]
        inp, preds = torch.zeros_like(anchor), []

        for t in range(H):
            h = self.dec(inp, h)
            step = self.out(h)
            preds.append(step)
            inp = (y[:, t : t + 1] - anchor) if y is not None else step
        return anchor + torch.cat(preds, 1)

    @torch.no_grad()
    def predict(self, recent_freq):
        x = torch.log2(torch.tensor(recent_freq[-K:], dtype=torch.float32))[None]
        return (2 ** self(x))[0].numpy()


def windows(job):
    # 1D input series of frequincies in Hz. freqs are logged.

    if not (np.asarray(job, dtype=np.float32) > 0).all():
        raise RuntimeError("Jobs contain zero or negative values")

    s = np.log2(np.asarray(job, dtype=np.float32))
    return [(s[i : i + K], s[i + K : i + K + H]) for i in range(len(s) - K - H + 1)]


def training(model, X, Y, n=200):
    optim = torch.optim.Adam(model.parameters(), 3e-3)
    loss = nn.L1Loss()

    for epoch in range(n):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i : i + 128]
            optim.zero_grad()
            loss(model(X[b], Y[b]), Y[b]).backward()
            optim.step()


if __name__ == "__main__":

    model = Seq2Seq()
    train_jobs = None
    X, Y = zip(*[w for job in train_jobs for w in windows(job)])
    X, Y = torch.tensor(np.array(X)), torch.tensor(np.array(Y))
    training(model, X, Y, 200)
