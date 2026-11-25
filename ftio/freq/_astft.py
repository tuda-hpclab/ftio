"""
This module implements the Adaptive Short-Time Fourier Transform (ASTFT)
processing pipeline used in FTIO. It provides functionality for:

- Resampling bandwidth time-series to an evenly spaced time grid
- Applying optional denoising via TFPF-WVD filtering
- Performing ASTFT analysis in a three-step procedure
- Identifying and refining periodic components in the time-frequency domain
- Generating simple plots of the extracted components and time series

The module integrates with FTIO’s component selection and linking utilities
to identify meaningful frequency components and refine their properties
such as amplitude, frequency, and phase.

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from collections import namedtuple

import numpy as np
from scipy.fft import fft
from scipy.signal import stft
from scipy.signal.windows import boxcar

from ftio.freq.concentration_measures import cm5
from ftio.freq.denoise import tfpf_wvd
from ftio.freq.if_comp_separation import (
    binary_image_zscore,
    component_linking,
)
from ftio.plot.plot_amd import plot_amd_components


def astft(b_sampled, freq, bandwidth, time_b, args):
    t_start = time_b[0]
    t_end = time_b[-1]
    N = len(b_sampled)
    t = np.linspace(t_start, t_end, N, dtype=float)

    if args.tfpf:
        signal_hat = b_sampled
        for _ in range(0, args.tfpf):
            signal_hat = tfpf_wvd(signal_hat, freq, t)
        return astft_3step(
            b_sampled, freq, t, args, filtered=signal_hat, b_orig=bandwidth, t_orig=time_b
        )
    else:
        return astft_3step(b_sampled, freq, t, args, b_orig=bandwidth, t_orig=time_b)


def astft_3step(signal, freq, time_b, args, filtered=None, b_orig=None, t_orig=None):
    stft_window = str(args.stft_window)
    if stft_window.endswith("s"):
        win_len = int(float(stft_window[:-1]) * freq)
    else:
        win_len = int(float(stft_window))

    if win_len > 0:
        pass
    elif filtered is None:
        win_len = cm5(signal, freq)
    else:
        win_len = cm5(filtered, freq)

    if filtered is None:
        signal_tfr, f, t = ptfr(signal, freq, win_len)
    else:
        signal_tfr, f, t = ptfr(filtered, freq, win_len)

    image = binary_image_zscore(signal_tfr, freq, args)
    components = component_linking(image, freq, win_len)

    return simple_astft(
        components, signal, filtered, freq, time_b, args, b_orig=b_orig, t_orig=t_orig
    )


def ptfr(x, fs, win_len):
    win = boxcar(win_len)
    f, t, Zxx = stft(x, fs=fs, window=win, nperseg=win_len, noverlap=(win_len - 1))
    Zxx = Zxx.transpose()
    return Zxx, f, t


def check_3_periods(signal, fs, exp_freq, est_period, start, end):
    win_len = int(3 * est_period)
    bonus = est_period.astype(int)
    _start = max(0, start - bonus)
    _end = min(len(signal), end + bonus)

    subsignal = signal[_start:_end]
    win = boxcar(win_len)
    hop = 1
    f, t, Zxx = stft(
        subsignal, fs=fs, window=win, nperseg=win_len, noverlap=(win_len - hop)
    )

    exp_frq_bin = np.round((exp_freq * win_len) / fs).astype(int)
    comp = []
    flag = False
    start_idx = 0

    for i in range(0, len(t)):
        yf = Zxx[: exp_frq_bin + 3, i].transpose()
        yf = np.abs(yf)
        if (
            (
                exp_frq_bin > 1
                and yf[exp_frq_bin] > yf[exp_frq_bin - 1]
                and yf[exp_frq_bin] > yf[exp_frq_bin + 1]
            )
            or (
                len(yf) > exp_frq_bin + 2
                and yf[exp_frq_bin] > yf[exp_frq_bin - 1]
                and yf[exp_frq_bin] > yf[exp_frq_bin + 2]
            )
            or (
                yf[exp_frq_bin] > yf[exp_frq_bin + 1]
                and yf[exp_frq_bin] > yf[exp_frq_bin - 2]
            )
        ):
            if not flag:
                start_idx = i
                flag = True
        else:
            if flag:
                comp.append((start_idx, i))
                flag = False

    if flag:
        comp.append((start_idx, len(t) - 1))

    final_comp = []
    for c in comp:
        if (c[1] - c[0]) * hop >= win_len:
            final_comp.append((c[0] + _start, c[1] + _start))

    return final_comp


def frq_refinement(signal, start, end, frq_est, fs, duration):
    est_period_time = 1 / frq_est
    est_period = len(signal) * (est_period_time / duration)

    end_per = start + 3 * est_period
    while end_per + est_period < end:
        end_per += est_period
    end_per = end_per.astype(int)

    over = end - end_per
    start_per = start + over // 2
    end_per += over // 2

    yf = fft(signal[start_per:end_per])
    frqs = np.abs(yf[1 : len(yf) // 2])
    exp_frq_bin = int((frq_est * (end_per - start_per)) / fs)

    ind = np.argmax(frqs[exp_frq_bin - 2 : exp_frq_bin + 2])
    ind = ind + (exp_frq_bin - 2) + 1

    frq_est_ref = ind * fs / len(yf)
    magn = np.abs(yf[ind])

    if np.abs(1 - (frq_est / frq_est_ref)) < 0.005:
        return yf, ind, start_per, end_per

    maxiter = 3
    for _ in range(0, maxiter):
        yf_old, ind_old, start_per_old, end_per_old, magn_old = (
            yf,
            ind,
            start_per,
            end_per,
            magn,
        )
        frq_est = frq_est_ref
        est_period_time = 1 / frq_est
        est_period = len(signal) * (est_period_time / duration)

        end_per = start + 3 * est_period
        if end_per > end:
            return yf_old, ind_old, start_per_old, end_per_old

        while end_per + est_period < end:
            end_per += est_period
        end_per = end_per.astype(int)

        over = end - end_per
        start_per = start + over // 2
        end_per += over // 2

        yf = fft(signal[start_per:end_per])
        frqs = np.abs(yf[1 : len(yf) // 2])
        exp_frq_bin = int((frq_est * (end_per - start_per)) / fs)

        ind = np.argmax(frqs[exp_frq_bin - 2 : exp_frq_bin + 2])
        ind = ind + (exp_frq_bin - 2) + 1

        frq_est_ref = ind * fs / len(yf)
        magn = np.abs(yf[ind])

        if magn < magn_old:
            return yf_old, ind_old, start_per_old, end_per_old

        if (
            ind > 1
            and np.abs(yf[ind - 1]) > np.abs(yf[ind - 2])
            and np.abs(yf[ind - 1]) > np.abs(yf[ind + 1])
        ):
            frq_est_ref = (
                (ind * np.abs(yf[ind]) + (ind - 1) * np.abs(yf[ind - 1]))
                * fs
                / (len(yf) * (np.abs(yf[ind]) + np.abs(yf[ind - 1])))
            )
        elif (
            ind < len(yf) - 2
            and np.abs(yf[ind + 1]) > np.abs(yf[ind + 2])
            and np.abs(yf[ind + 1]) > np.abs(yf[ind - 1])
        ):
            frq_est_ref = (
                (ind * np.abs(yf[ind]) + (ind + 1) * np.abs(yf[ind + 1]))
                * fs
                / (len(yf) * (np.abs(yf[ind]) + np.abs(yf[ind + 1])))
            )

    return yf, ind, start_per, end_per


def simple_astft(
    components,
    signal,
    filtered,
    fs,
    time_b,
    args,
    merge=True,
    imfs=None,
    b_orig=None,
    t_orig=None,
):
    signal_plot = signal
    t_start, t_end = time_b[0], time_b[-1]
    N = len(signal)
    t = np.arange(t_start, t_end, (t_end - t_start) / N, dtype=float)
    duration = t_end - t_start

    if filtered is not None:
        signal = filtered

    if merge:
        i = 0
        length = len(components)
        while i < length - 1:
            if components[i][1] == components[i + 1][1]:
                est_period = len(signal) * ((1 / components[i][1]) / duration)
                if (components[i + 1][0][0] - components[i][0][1]) < est_period:
                    components[i] = (
                        (components[i][0][0], components[i + 1][0][1]),
                        components[i][1],
                        components[i][2],
                    )
                    del components[i + 1]
                    length -= 1
            i += 1

    Component = namedtuple("Component", ["start", "end", "amp", "freq", "phase"])
    per_comp = []

    for i in components:
        start, end = i[0][0], i[0][1] + 1
        comp_length = end - start
        est_period = len(signal) * ((1 / i[1]) / duration)

        if comp_length < 3 * est_period * 0.9:
            continue

        final_components = check_3_periods(signal, fs, i[1], est_period, start, end)

        for fc in final_components:
            source_sig = imfs[i[3]] if imfs is not None else signal
            yf, peak, start_frq, stop_frq = frq_refinement(
                source_sig, fc[0], fc[1], i[1], fs, duration
            )

            amp_final = np.abs(yf[peak]) / (stop_frq - start_frq)
            frq_final = peak * fs / (stop_frq - start_frq)
            phi_shift = -2 * np.pi * frq_final * (t[start_frq] - t[fc[0]])
            phase_final = np.angle(yf[peak] * np.exp(1j * phi_shift))

            merged = False
            for c in per_comp:
                if c.freq == frq_final and c.end > fc[0]:
                    phi_shift_comp = -2 * np.pi * frq_final * (t[start_frq] - t[c.start])
                    phase_comp = np.angle(yf[peak] * np.exp(1j * phi_shift_comp))
                    if np.abs(phase_final - phase_comp) < 0.001:
                        # Update existing component end
                        idx = per_comp.index(c)
                        per_comp[idx] = c._replace(end=fc[1])
                        merged = True
                        break

            if not merged:
                per_comp.append(
                    Component(fc[0], fc[1], amp_final, frq_final, phase_final)
                )

    figs = []
    if any(x in args.engine for x in ["mat", "plot"]):
        fig = plot_amd_components(
            args,
            signal_plot,
            t,
            per_comp,
            title="ASTFT components",
            b_orig=b_orig,
            t_orig=t_orig,
        )
        if fig:
            figs.append(fig)

    return per_comp, figs
