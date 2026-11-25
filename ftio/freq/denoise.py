"""
This file provides functions for time–frequency peak filtering (TFPF) based on the
Pseudo Wigner–Ville Distribution (PWVD) to estimate instantaneous frequency and
attenuate noise in 1D signals. The implementation follows phase encoding and
time–frequency ridge extraction as described in the signal processing literature
and includes utilities for signal scaling, reconstruction, and visualization.

The current implementation is intended for experimental and exploratory use.
Several parameters (e.g., scaling factor, lag window length, and peak frequency
range) are heuristic and should be made adaptive for robust applications.

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hamming
from tftb.processing import PseudoWignerVilleDistribution as pwvd

# TODO:
# - scaling parameter mu
# - set f_peak (highest relevant frequency), lag window too large
# - normalization of max?

# Boashash, B., & Mesbah, M. (2004).
# Signal enhancement by time-frequency peak filtering.
# IEEE Transactions on signal processing, 52(4), 929-937.


def tfpf_wvd(x, fs, time):

    # lag window length
    f_peak = 4
    tau = (0.634 * fs) / (np.pi * f_peak)
    tau = int(tau) + 1 - (tau % 2)

    # TODO: make lag window length adaptive
    tau = 21

    # endpoint extension
    """
    Lin, T., Zhang, Y., Yi, X., Fan, T., & Wan, L. (2018).
    Time-frequency peak filtering for random noise attenuation
    of magnetic resonance sounding signal.
    Geophysical Journal International, 213(2), 727-738.
    """
    p = tau // 2 + 1
    x_ext = np.pad(x, (p, p), "constant", constant_values=(x[0], x[-1]))

    # scaling
    x_scaled = scale_signal(x_ext)

    # encode
    mu = 0.5  # TODO: guessed

    z_x = np.zeros((len(x_scaled),), dtype=np.complex128)
    sum = 0.0
    for m in range(0, len(x_scaled)):
        sum += x_scaled[m]
        z_x[m] = np.exp(1j * 2 * np.pi * mu * sum)

    # time sample
    ts = np.arange(time[0], time[-1], (time[-1] - time[0]) / len(x))

    # add endpoints
    step = (time[-1] - time[0]) / len(x)
    ts_begin = np.arange(time[0] - p * step, time[0], step)
    ts_end = np.arange(time[-1], time[-1] + p * step, step)
    ts_padded = np.concatenate((ts_begin, ts, ts_end))

    # PWVD
    tfr_wvd = pwvd(z_x, timestamps=ts_padded)
    tfr_wvd.fwindow = hamming(tau)
    tfr_wvd.run()
    tfr_wvd.plot(show_tf=True, kind="contour")

    # estimate peak
    x_c_hat = np.argmax(tfr_wvd.tfr, axis=0) / mu

    # conversion: "Time–frequency peak filtering for random noise attenuation [...]"
    # nfreqbin = len(tfr_wvd.fwindow)
    # x_c_hat = (x_c_hat - 1) * fs/ nfreqbin

    x_c_hat /= len(x_ext) * 2

    # recover signal
    x_hat = recover_signal(x_ext, x_c_hat)

    # endpoint removal
    x_rem = x_hat[p:-p]

    tfpf_wvd_plot(x, x_rem, ts)

    return x_rem


def scale_signal(x):
    # [0,0.5], arb chosen in "IF estimation for multicomponent signals [..]"
    a = 0.4
    b = 0.1

    x_min = np.min(x)
    x_max = np.max(x)

    scaled = np.zeros((len(x),), dtype=np.complex128)
    for m in range(0, len(x)):
        scaled[m] = (a - b) * (x[m] - x_min) * 1 / (x_max - x_min) + b

    return scaled


def recover_signal(x, x_c_hat):
    # [0,0.5], arb chosen in "IF estimation for multicomponent signals [..]"
    a = 0.4
    b = 0.1

    x_min = np.min(x)
    x_max = np.max(x)

    recovered = np.zeros((len(x),), dtype=np.complex128)
    for m in range(0, len(x)):
        recovered[m] = (x_c_hat[m] - b) * (x_max - x_min) * 1 / (a - b) + x_min

    return recovered


def tfpf_wvd_plot(x, x_hat, t):
    fig, ax = plt.subplots(2)

    if len(x) < len(x_hat):
        end = len(x)
    else:
        end = len(x_hat)

    ax[0].plot(t[:end], x[:end])
    ax[1].plot(t[:end], x_hat[:end])

    plt.show()
