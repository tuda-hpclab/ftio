"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Okt 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np

from ftio.freq.prediction import Prediction


class PhaseMode:
    def __init__(self, name: str, matches: list[str]) -> None:
        self.name = name
        self.matches = matches
        self.data = []
        self.wave = np.nan
        self.t = np.array([])

    def get(self, attribute: str):
        if hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            raise AttributeError(f"'MyClass' object has no attribute '{attribute}'")

    def match(self, d: dict):
        return bool(any(n in d.metric for n in self.matches))

    def add(self, d: dict) -> None:
        self.data.append(d)

    def set_time(self, t: np.ndarray) -> None:
        self.t = t

    def aggregates_waves(self, t, normed=True):
        self.wave = np.zeros(len(t))
        if self.t.size != 0:
            for i in self.data:
                self.wave = self.wave + calculate_wave(i, self.t)[0]

            if normed:
                self.wave = norm(self.wave)

    def get_wave(self, metric: str):
        wave = np.array([])
        name = ""
        if self.t.size != 0:
            for prediction in self.data:
                if metric in prediction.metric:
                    wave, name = calculate_wave(prediction, self.t)

        return (wave, name)


def calculate_wave(
    prediction: Prediction, t: np.ndarray = np.array([])
):  # -> tuple[NDArray[Any] | Any, str | LiteralString]:# -> tuple[NDArray[Any] | Any, str | LiteralString]:# -> tuple[NDArray[Any] | Any, str | LiteralString]:
    wave = np.array([])
    name = ""
    if prediction:
        if t.size == 0:
            t = np.arange(prediction.t_start, prediction.t_end, 1 / prediction.freq)

        if len(prediction.top_freqs) > 0:
            wave = np.zeros_like(t)
            for j in range(0, len(prediction.top_freqs["freq"])):
                amp = prediction.top_freqs["amp"][j]
                freq = prediction.top_freqs["freq"][j]
                phi = prediction.top_freqs["phi"][j]
                cosine_wave, cosine_name = prediction.get_wave_and_name(freq, amp, phi, t)
                wave = wave + cosine_wave
                if j < 3:
                    name += cosine_name
                elif j == 4:
                    name += "..."
        else:
            freq, amp, phi = prediction.get_dominant_freq_amp_phi()
            wave, cosine_name = prediction.get_wave_and_name(freq, amp, phi, t)
            name += cosine_name
    return (wave, name)


def norm(wave: np.ndarray) -> np.ndarray:
    max_value = np.max(np.abs(wave))
    normed_wave = wave / max_value if max_value > 0 else wave
    return normed_wave
