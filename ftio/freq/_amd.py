"""
AMD processing pipeline for FTIO.

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from collections import namedtuple

import numpy as np
from pysdkit import EFD
from scipy.fft import fft, ifft
from scipy.signal import hilbert
from vmdpy import VMD

from ftio.analysis._correlation import correlation
from ftio.analysis.anomaly_detection import z_score
from ftio.freq._astft import simple_astft
from ftio.freq.denoise import tfpf_wvd
from ftio.plot.plot_amd import plot_amd_components, plot_imfs


def amd(b_sampled, freq, time_b, args, b_orig=None, t_orig=None):
    """
    Identify periodic time windows with adaptive mode decomposition.
    """
    t_start, t_end = time_b[0], time_b[-1]
    N = len(b_sampled)
    t = np.linspace(t_start, t_end, N)

    if "efd" in args.transformation:
        return efd(b_sampled, t, freq, args, b_orig=b_orig, t_orig=t_orig)

    if "vmd" in args.transformation:
        return vmd(b_sampled, t, freq, args, b_orig=b_orig, t_orig=t_orig)

    return [], []


def vmd(signal, t, fs, args, b_orig=None, t_orig=None):
    """
    Variational Mode Decomposition (VMD) based periodicity detection.
    """
    # fixed parameters
    tau, DC, init, tol = 0.0, 0, 0, 1e-7
    # signal dependent parameters
    alpha, K = 5000, 8

    figs = []
    if args.tfpf:
        signal_hat = signal
        for _ in range(0, args.tfpf):
            signal_hat = tfpf_wvd(signal_hat, fs, t)
        u, u_hat, omega = VMD(signal_hat, alpha, tau, K, DC, init, tol)

        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_imfs(args, signal, t, u, title="VMD modes")
            if fig:
                figs.append(fig)

        center_freqs = omega[-1] * fs
        u_periodic, cen_freq_per = rm_nonperiodic(u, center_freqs, t)
        components, astft_figs = imf_selection(
            signal_hat, u_periodic, t, cen_freq_per, fs, args
        )
        figs.extend(astft_figs)

        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_amd_components(
                args,
                signal_hat,
                t,
                components,
                title="VMD components",
                b_orig=b_orig,
                t_orig=t_orig,
            )
            if fig:
                figs.append(fig)
    else:
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_imfs(args, signal, t, u, title="VMD modes")
            if fig:
                figs.append(fig)

        center_freqs = omega[-1] * fs
        u_periodic, cen_freq_per = rm_nonperiodic(u, center_freqs, t)
        components, astft_figs = imf_selection(
            signal, u_periodic, t, cen_freq_per, fs, args
        )
        figs.extend(astft_figs)

        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_amd_components(
                args,
                signal,
                t,
                components,
                title="VMD components",
                b_orig=b_orig,
                t_orig=t_orig,
            )
            if fig:
                figs.append(fig)

    return components, figs


Component = namedtuple("Component", ["start", "end", "amp", "freq", "phase"])


def efd(signal, t, fs, args, b_orig=None, t_orig=None):
    """
    Empirical Fourier decomposition (EFD) based periodicity detection.
    """
    numIMFs = 8
    efd_model = EFD(max_imfs=numIMFs)
    figs = []

    signal_hat = None
    if args.tfpf:
        signal_hat = signal
        for _ in range(0, args.tfpf):
            signal_hat = tfpf_wvd(signal_hat, fs, t)
        imfs, cerf, _bounds = efd_model.fit_transform(signal_hat, return_all=True)
        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_imfs(args, signal, t, imfs, title="EFD modes")
            if fig:
                figs.append(fig)
    else:
        imfs, cerf, _bounds = efd_model.fit_transform(signal, return_all=True)
        if any(x in args.engine for x in ["mat", "plot"]):
            fig = plot_imfs(args, signal, t, imfs, title="EFD modes")
            if fig:
                figs.append(fig)

    cerf2 = np.empty(numIMFs)
    for i in range(0, numIMFs):
        yf = fft(imfs[i])
        ind = np.argmax(np.abs(yf[1:])) + 1
        cerf2[i] = (ind * fs) / len(imfs[i])

    per_segments = energy_windowed(t, imfs, cerf2, fs)
    pot_comp = []
    duration = t[-1] - t[0]

    for p in per_segments:
        est_frq = cerf2[p.index]
        est_period = len(imfs[0]) * ((1 / cerf2[p.index]) / duration)

        if len(imfs[p.index][p.start : p.end]) > est_period * 3 * 0.9:
            source = signal_hat if args.tfpf else signal
            yf = fft(source[p.start : p.end])
            N = p.end - p.start
            freq_arr = fs * np.arange(0, N) / N
            indices = z_score(yf, freq_arr, args)[0]
            exp_frq_bin = np.round((cerf2[p.index] * N) / fs).astype(int)

            for idx in indices:
                if exp_frq_bin in [idx, idx - 1, idx + 1]:
                    pot_comp.append(((p.start, p.end), est_frq, est_frq, p.index))
                    break

    res, astft_figs = simple_astft(
        pot_comp,
        signal,
        signal_hat,
        fs,
        t,
        args,
        merge=False,
        imfs=imfs,
        b_orig=b_orig,
        t_orig=t_orig,
    )
    figs.extend(astft_figs)
    return res, figs


def rm_nonperiodic(u, center_freqs, t):
    duration = t[-1] - t[0]
    min_period = duration / 2
    min_frq = 1 / min_period
    i = 0
    while i < len(center_freqs) and center_freqs[i] < min_frq:
        i += 1
    return u[i:], center_freqs[i:]


def det_imf_frq(imf, center_frq, fs, args):
    N = len(imf)
    n_pad = max(3, N // 10)
    imf_padded = np.pad(imf, (n_pad, n_pad), "reflect")
    analytic_signal = hilbert(imf_padded)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    ifreq = instantaneous_frequency[n_pad:-n_pad]

    if np.std(ifreq) > 0.04:
        return -1

    yf = fft(imf)
    ind = np.argmax(np.abs(yf)[1:]) + 1
    frq_arr = np.zeros(len(imf), dtype="complex")
    frq_arr[ind] = yf[ind]
    frq_est = ind * fs / len(imf)

    # Simplified refinement logic
    amp = ifft(frq_arr)
    min_len = min(len(imf), len(amp))
    corr = correlation(
        imf[:min_len].astype(float), amp[:min_len].real.astype(float), method="spearman"
    )
    if corr > 0.8:
        return frq_est

    yf_env = fft(amplitude_envelope[n_pad:-n_pad])
    freq_arr = fs * np.arange(0, len(yf_env)) / len(yf_env)
    indices = z_score(yf_env, freq_arr, args)[0]
    for idx in indices:
        env_frq_arr = np.zeros(len(yf_env), dtype="complex")
        env_frq_arr[idx] = yf_env[idx]
        env_cos = ifft(env_frq_arr)
        corr = correlation(
            amplitude_envelope[n_pad:-n_pad].astype(float),
            env_cos.real.astype(float),
            method="spearman",
        )
        if corr > 0.8:
            return fs * idx / len(yf_env)
    return -1


component = namedtuple("component", ["index", "start", "end"])


def imf_select_windowed(signal, t, u_per, fs, overlap=0.5):
    duration = t[-1] - t[0]
    per_segments = []
    for j in range(0, u_per.shape[0]):
        imf = u_per[j]
        yf = fft(imf)
        peak = np.argmax(np.abs(yf)[1:]) + 1
        if peak <= 1:
            continue
        est_frq = peak * fs / len(imf)
        est_per = int(len(imf) * ((1 / est_frq) / duration))
        win_size = int(3 * est_per)
        ind, start, flag = 0, 0, False
        skip = int(est_per * overlap)
        max_len = min(len(imf), len(signal))
        while ind + win_size < max_len:
            sig_win = signal[ind : ind + win_size].astype(float)
            imf_win = imf[ind : ind + win_size].astype(float)
            corr = correlation(sig_win, imf_win, method="pearson")
            if corr > 0.65 and not flag:
                start, flag = ind, True
            elif corr <= 0.65 and flag:
                per_segments.append(component(j, start, ind + win_size - skip))
                start, flag = 0, False
            ind += skip
        if flag:
            per_segments.append(component(j, start, max_len))
    return per_segments


def energy_windowed(t, imfs, cerf, fs, overlap=0.5):
    duration = t[-1] - t[0]
    rel_segments = []
    for i in range(0, imfs.shape[0]):
        energy = np.sum(np.abs(imfs[i]) ** 2)
        est_period = int(len(imfs[i]) * ((1 / cerf[i]) / duration))
        win_size = int(3 * est_period)
        ind, start, flag = 0, 0, False
        skip = int(est_period * overlap)
        while ind + win_size < len(imfs[i]):
            energy_win = np.sum(np.abs(imfs[i][ind : ind + win_size]) ** 2)
            energy_scaled = energy_win * (len(imfs[i]) / win_size)
            if energy_scaled >= energy * 0.95 and not flag:
                start, flag = ind, True
            elif energy_scaled < energy * 0.95 and flag:
                rel_segments.append(component(i, start, ind + win_size - skip))
                start, flag = False, False
            ind += skip
        if flag:
            rel_segments.append(component(i, start, len(imfs[i])))
    return rel_segments


def imf_selection(signal, u_per, t, center_freqs, fs, args):
    signal_float = signal.astype(float)
    confirmed_win = []
    figs = []
    for j in range(0, u_per.shape[0]):
        imf = u_per[j]
        min_len = min(len(signal_float), len(imf))
        corr = correlation(signal_float[:min_len], imf[:min_len], method="pearson")
        if corr > 0.65:
            start = remove_zero_single_mode(imf)
            est_frq = det_imf_frq(imf[start:], center_freqs[j], fs, args)

            if est_frq != -1:
                # Perform sinusoidal reconstruction
                imf_trimmed = imf[start:]
                analytic_signal = hilbert(imf_trimmed)
                amplitude_envelope = np.abs(analytic_signal)
                amp_val = np.mean(amplitude_envelope)
                # phi = -2 * pi * f * t_shift
                phi_val = np.angle(
                    np.mean(
                        analytic_signal * np.exp(-1j * 2 * np.pi * est_frq * t[start:])
                    )
                )

                if (t[-1] - t[0]) > (3 * (1 / est_frq) * 0.9):
                    return [Component(start, len(imf), amp_val, est_frq, phi_val)], []

    per_segments = imf_select_windowed(signal, t, u_per, fs)
    for comp in per_segments:
        imf = u_per[comp.index][comp.start : comp.end]
        est_frq = det_imf_frq(imf, center_freqs[comp.index], fs, args)
        if est_frq == -1:
            frq = center_freqs[comp.index]
            add_comp = [
                ((comp.start, comp.end), frq, frq),
                ((comp.start, comp.end), frq / 2, frq / 2),
                ((comp.start, comp.end), frq / 3, frq / 3),
            ]
            res, astft_figs = simple_astft(add_comp, signal, signal, fs, t, args)
            for r in res:
                confirmed_win.append(r)
            figs.extend(astft_figs)
        elif (t[comp.end - 1] - t[comp.start]) > (3 * (1 / est_frq) * 0.9):
            # Estimate amplitude and phase for the segment
            analytic_signal = hilbert(imf)
            amp_val = np.mean(np.abs(analytic_signal))
            phi_val = np.angle(
                np.mean(
                    analytic_signal
                    * np.exp(-1j * 2 * np.pi * est_frq * t[comp.start : comp.end])
                )
            )
            confirmed_win.append(
                Component(comp.start, comp.end, amp_val, est_frq, phi_val)
            )

    return confirmed_win, figs


def remove_zero_single_mode(imf):
    N = len(imf)
    n_pad = N // 10
    imf_padded = np.pad(imf, (n_pad, n_pad), "reflect")
    analytic_signal = hilbert(imf_padded)
    amp_env = np.abs(analytic_signal)[n_pad:-n_pad]
    median_amp = np.median(amp_env)
    rel_ind = np.nonzero(amp_env > median_amp * 0.3)[0]
    return rel_ind[0] if len(rel_ind) > 0 and rel_ind[0] < len(imf) else 0
