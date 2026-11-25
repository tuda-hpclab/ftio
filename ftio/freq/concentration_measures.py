"""
file description

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
from scipy.fftpack import fftshift
from scipy.signal import stft
from scipy.signal.windows import boxcar

# TODO:
# - window function CM?
# - find an CM implementation to verify approach
# - adapt min_win to f_s

# Pei, S. C., & Huang, S. G. (2012).
# STFT with adaptive window width based on the chirp rate.
# IEEE Transactions on Signal Processing, 60(8), 4065-4080.

min_win = 80
max_win = 500

alpha = 0.1
beta = 5
p = 4
L = 105


# https://de.mathworks.com/matlabcentral/answers/356692-how-to-normalize-a-fft-to-plot-in-frequency-domain
def normalize_fft(yf):
    yf_norm = fftshift(yf / len(yf))
    # yf_norm = yf / np.max(yf)
    return yf_norm


def cm3(x, fs):
    win_len = min_win

    max_win = len(x)
    p = int(len(x) * 0.005)
    if p == 0:
        p = 1
    L = (max_win - min_win) // p

    sum = np.zeros((L,), dtype=np.complex128)

    for j in range(0, L):
        win = boxcar(win_len)
        f, t, Zxx = stft(x, fs=fs, window=win, nperseg=win_len, noverlap=win_len // 2)
        Zxx = Zxx.transpose()

        for yf in Zxx:
            yf_norm = normalize_fft(yf) ** alpha
            sum[j] = sum[j] + np.sum(yf_norm)

        sum[j] = 1 / sum[j]
        win_len = win_len + p

    peak = np.argmax(sum)
    final_win_len = min_win + p * peak

    return final_win_len


def cm4(x, fs):
    win_len = min_win

    max_win = len(x)
    p = int(len(x) * 0.005)
    if p == 0:
        p = 1
    L = (max_win - min_win) // p

    sum = np.zeros((L,), dtype=np.complex128)

    for j in range(0, L):
        win = boxcar(win_len)
        # win = hann(win_len)
        f, t, Zxx = stft(x, fs=fs, window=win, nperseg=win_len, noverlap=win_len // 2)
        Zxx = Zxx.transpose()

        for yf in Zxx:
            yf_norm = normalize_fft(yf) ** beta
            sum[j] = sum[j] + np.sum(yf_norm)

        win_len = win_len + p

    peak = np.argmax(sum)
    final_win_len = min_win + p * peak

    return final_win_len


def cm5(x, fs):
    win_len = min_win

    max_win = len(x)
    p = int(len(x) * 0.005)
    if p == 0:
        p = 1
    L = (max_win - min_win) // p

    sum = np.zeros((L,), dtype=np.complex128)

    for j in range(0, L):
        win = boxcar(win_len)
        f, t, Zxx = stft(x, fs=fs, window=win, nperseg=win_len, noverlap=win_len // 2)
        Zxx = Zxx.transpose()

        sum1 = 0.0
        sum2 = 0.0

        for yf in Zxx:
            yf_norm = normalize_fft(yf)
            sum1 = sum1 + np.sum(yf_norm) ** beta
            sum2 = sum2 + np.sum(yf_norm) ** alpha

        sum[j] = (sum1 ** (1 / beta)) / (sum2 ** (1 / alpha))
        win_len = win_len + p

    peak = np.argmax(sum)
    final_win_len = min_win + p * peak

    return final_win_len
