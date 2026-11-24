"""
Module concerned with probability calculation

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np
from rich.console import Console

CONSOLE = Console()


class Probability:
    """Class that stores the conditional probability according to frequency intervals"""

    def __init__(
        self,
        freq_min,
        freq_max,
        p_periodic=0,
        p_freq=0,
        p_freq_given_periodic=0,
        p_periodic_given_freq=1,
    ):
        """init function

        Args:
            freq_min (float): minimum frequencies in group
            freq_max (float): max frequencies in group
            p_periodic (float, optional): probability of being periodic. Defaults to 0.
            p_freq (float, optional): Probability that the signal has the frequency in the range [freq_min, freq_max]. Defaults to 0.
            p_freq_given_periodic (float, optional): conditional probability that expresses how probable the frequency is in the interval [freq_min, freq_max] ,
            given that the signal is periodic. Defaults to 0.
            p_periodic_given_freq (float, optional): Probability that the signal is periodic given that it has a frequency A. Defaults to 1.
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.p_periodic = p_periodic
        self.p_freq = p_freq
        self.p_freq_given_periodic = p_freq_given_periodic
        self.p_periodic_given_freq = p_periodic_given_freq

    def set(
        self,
        p_periodic=np.nan,
        p_freq=np.nan,
        p_freq_given_periodic=np.nan,
        p_periodic_given_freq=np.nan,
    ):
        self.p_periodic = p_periodic if not np.isnan(p_periodic) else self.p_periodic
        self.p_freq = p_freq if not np.isnan(p_freq) else self.p_freq
        self.p_freq_given_periodic = (
            p_freq_given_periodic
            if not np.isnan(p_freq_given_periodic)
            else self.p_freq_given_periodic
        )
        self.p_periodic_given_freq = (
            p_periodic_given_freq
            if not np.isnan(p_periodic_given_freq)
            else self.p_periodic_given_freq
        )

    def display(self, prefix=""):
        CONSOLE.print(
            f"{prefix} P([{self.freq_min:.3f},{self.freq_max:.3f}] Hz) = {self.p_periodic*100:.2f}%\n"
            f"{prefix} |-> [{self.freq_min:.3f},{self.freq_max:.3f}] Hz = [{1/self.freq_max if self.freq_max != 0 else np.nan:.3f},{1/self.freq_min if self.freq_min != 0 else np.nan:.3f}] sec\n"
            f"{prefix} '-> P([{self.freq_min:.3f},{self.freq_max:.3f}] Hz | periodic) = {self.p_freq_given_periodic*100:.2f}%"
        )

    def get_freq_prob(self, freq):
        return bool(freq >= self.freq_min and freq <= self.freq_max)
