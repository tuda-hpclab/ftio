"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np


def sliding_correlation(x, y, window_size):
    """
    Compute Pearson correlation in a sliding window.

    Parameters:
        x, y: Input 1D signals (must be same length)
        window_size: Size of window in samples

    Returns:
        corrs: Array of local correlations
    """
    n = len(x)
    w = window_size
    corrs = np.zeros(n - w + 1)
    for i in range(len(corrs)):
        x_win = x[i : i + w]
        y_win = y[i : i + w]
        if np.std(x_win) == 0 or np.std(y_win) == 0:
            corrs[i] = 0
        else:
            corrs[i] = np.corrcoef(x_win, y_win)[0, 1]
    return corrs
