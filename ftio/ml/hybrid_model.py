"""
Neural network model classes for the hybrid model.

This module contains the neural network components used for I/O bandwidth prediction:
- PositionalEncoding: Position encoding for sequential data
- BandwidthDataset: Custom dataset for bandwidth sequences
- HybridModel: Transformer-LSTM hybrid model for time series prediction
- spectral_loss: Loss function based on Fourier transform

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

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
else:
    raise RuntimeError(
        "Torch module not found. Please install it using 'make full' or 'pip install ftio[ml-libs]'."
    )


class PositionalEncoding(nn.Module):
    """Position encoding used in the hybrid model to increase the likelihood of understanding sequential data."""

    def __init__(self, emb_dim, max_len=5555):
        """Constructor of the positional encoding.

        Args:
            emb_dim: Size of the embedding dimension
            max_len: Maximum length of sequence length that can be handled.
        """
        super().__init__()
        # Learned Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, emb_dim))

    def forward(self, x):
        """Forward method of the positional encoding.

        Args:
            x: The embedded sequence.

        Returns: The embedded sequence added to the vector of the sliced positional encoding.

        """
        # add embedded sequence + sliced positional encoding (added to device to prevent RuntimeError)
        return x + self.positional_encoding[:, : x.size(1)].to(x.device)


class BandwidthDataset(Dataset):
    """This Dataset accepts lists containing multiple partial sequences of list[ranks, bandwidth, duration, start_time]
    that combine into one complete sequence.
    The conversion of the list into a new representation in torch.tensors allows for direct training or prediction.

    """

    def __init__(self, data_list, num_parts=3):
        """Constructor of the Dataset. Convertion of data into torch.tensors.

        Args:
            data_list: Lists containing the bandwidth values of the sequence.
            num_parts: Int deciding the number of slices to be made from the sequence.
        """
        self.data = []
        self.num_parts = num_parts

        if not self.num_parts:
            self.num_parts = 3

        for seq in data_list:
            seq = torch.tensor(seq, dtype=torch.float32)
            bandwidth = seq[:, 0].unsqueeze(-1)

            # normalize per-sequence
            min_val, max_val = bandwidth.min(), bandwidth.max()
            bandwidth = (bandwidth - min_val) / (max_val - min_val + 1e-8)

            L = len(bandwidth)
            part_len = L // self.num_parts

            # split into num_parts equal parts
            parts = [
                bandwidth[i * part_len : (i + 1) * part_len]
                for i in range(self.num_parts - 1)
            ]
            parts.append(bandwidth[(self.num_parts - 1) * part_len :])
            # training input
            past = torch.cat(parts[:-1], dim=0)

            # target = last part
            future = parts[-1]
            self.data.append((past, future))

    def __len__(self):
        """Method providing the amount of datapoints inside the dataset.

        Returns: Amount of datapoints inside the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """Method providing datapoint from specified index.

        Args:
            idx: Index of the specific datapoint.

        Returns: The datapoint from the specified index.

        """
        return self.data[idx]


class HybridModel(nn.Module):
    """A hybrid model leveraging transformer and long short-term memory."""

    def __init__(self, emb_dim=128, n_heads=4, ff_dim=128, num_layers=6, lstm_hidden=128):
        """Constructor of the hybrid model. Currently only supports the most important parameters.

        Args:
            emb_dim: Size of the embedding dimension
            n_heads: Amount of attention heads of transformer approach
            ff_dim: Size of the Feedforward network dimension
            num_layers: Amount of Transformer encoder layers
            lstm_hidden: Amount of hidden units in the LSTM part after transformer
        """
        super().__init__()
        self.bandwidth_embedding = nn.Linear(1, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_bandwidth = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, past_seq, prediction_length=None):
        x = self.bandwidth_embedding(past_seq)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)

        # shrink output to target length if needed
        if prediction_length is not None:
            x = x[:, -prediction_length:, :]  # take last part only

        return self.fc_bandwidth(x)


def spectral_loss(pred, target):
    """Basic spectral loss function based on one-dimensional fourier transform

    Args:
        pred: the predicted values
        target: the actual values
    """
    # pred, target: [batch, seq_len]
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    return F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
