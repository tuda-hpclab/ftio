import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataloader excludes series too short.
    """

    def __init__(self, series_list, col=None):
        filtered_series = []

        for s in series_list:
            s = torch.as_tensor(s, dtype=torch.float32)
            if len(s) >= 2:
                filtered_series.append(s)
        self.series = filtered_series
        self.col = col
        self.lengths = [len(s) for s in filtered_series]

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        # validate index
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return (
            self.series[idx]
            if self.series[idx].ndim == 2
            else self.series[idx].unsqueeze(-1)
        )
