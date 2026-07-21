"""
STFT workflow for FTIO. This module implements the
Short-Time Fourier Transform (STFT) processing pipeline used
by FTIO. It handles resampling of bandwidth time-series data, executes
the STFT analysis, and prepares prediction and analysis outputs in a
format compatible with the FTIO framework.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import copy
import time
from argparse import Namespace

import numpy as np
from rich.console import Group
from rich.panel import Panel

from ftio.analysis.anomaly_detection import outlier_detection
from ftio.analysis.periodicity_analysis import new_periodicity_scores
from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq._dft_workflow import ftio_dft
from ftio.freq._stft import compute_stft
from ftio.freq.discretize import sample_data
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.plot.plot_stft import plot_stft


def ftio_stft(
    args: Namespace,
    bandwidth: np.ndarray,
    time_stamps: np.ndarray,
    total_bytes: int = 0,
    ranks: int = 1,
    text: str = "",
    rank_sequence: np.ndarray | None = None,
    rank_sequence_time: np.ndarray | None = None,
) -> tuple[Prediction, AnalysisFigures]:
    """
    Performs a Short-Time Fourier Transform (STFT) on the sampled bandwidth data.

    Args:
        args (Namespace): The arguments passed to the function.
        bandwidth (np.ndarray): Bandwidth values.
        time_stamps (np.ndarray): Time points corresponding to the bandwidth values.
        total_bytes (int, optional): Total number of bytes transferred.
        ranks (int, optional): The number of ranks.
        text (str, optional): Additional text for output.
        rank_sequence (np.ndarray, optional): per-burst rank count -- only
            non-None when the source trace's rank count actually changed
            mid-run (malleability; see extract.get_time_behavior). Has its
            OWN time basis, `rank_sequence_time` -- it comes from a different
            granularity of the source data than `time_stamps`/`bandwidth`
            (see extract._rank_sequence), so it is matched to each STFT
            window by time range, not by index. Used to label each window
            with the rank count actually active during it, instead of the
            one constant `ranks` value for windows that don't match it.
        rank_sequence_time (np.ndarray, optional): time basis for
            `rank_sequence`, same length as it.

    Returns:
        tuple: (prediction, analysis_figures)
    """
    prediction = Prediction(args.transformation)
    analysis_figures = AnalysisFigures(args)
    console = MyConsole(verbose=args.verbose)

    #! Sample the bandwidth evenly spaced in time
    tik = time.time()
    console.print("[cyan]Executing:[/] Discretization\n")
    b_sampled, args.freq = sample_data(bandwidth, time_stamps, args)
    console.print(f"\n[cyan]Discretization finished:[/] {time.time() - tik:.3f} s")

    #! Perform STFT
    tik = time.time()

    # Determine window length (nperseg) in samples.
    stft_window = str(args.stft_window)
    if stft_window.endswith("s"):
        nperseg = int(float(stft_window[:-1]) * args.freq)
    else:
        nperseg = int(float(stft_window))

    if nperseg <= 0:
        # Automatically determine window size:
        # We aim for ~4 periods of the dominant frequency found via DFT
        # to get a balanced time-frequency representation.
        temp_args = copy.deepcopy(args)
        temp_args.engine = "no"
        temp_args.verbose = False
        temp_args.stft_window = "0"  # Avoid recursion
        temp_args.periodicity_detection = None  # Silence print in dft
        temp_prediction, _ = ftio_dft(
            temp_args, bandwidth, time_stamps, total_bytes, ranks
        )

        dominant_freq = temp_prediction.get_dominant_freq()
        if not np.isnan(dominant_freq) and dominant_freq > 0:
            # Calculate samples needed for 4 periods: 4 * (fs / freq)
            nperseg = int(4 * args.freq / dominant_freq)
            # If window is too small (e.g., less than 5 samples),
            # fallback to a reasonable default to avoid over-segmentation.
            if nperseg < 5:
                nperseg = max(len(b_sampled) // 4, 2)
                console.info(
                    f"[yellow]Window size too small ({int(4 * args.freq / dominant_freq)}) "
                    f"falling back to default ({nperseg} samples).[/]"
                )
        else:
            # If no dominant frequency found, use 1/4 of the trace
            nperseg = max(len(b_sampled) // 4, 2)

    # Ensure nperseg is valid (at least 2 samples, no more than total length)
    nperseg = min(max(nperseg, 2), len(b_sampled))

    # Adjust noverlap to fit windows exactly within trace without padding
    # This ensures that windows have the same size and cover the whole trace
    L = len(b_sampled)
    if nperseg < L:
        # Target roughly 1/10 overlap (nstep = 9/10 * nperseg)
        n_windows = int(np.ceil((L - nperseg) / (nperseg * 0.9))) + 1
        if n_windows > 1:
            nstep = (L - nperseg) // (n_windows - 1)
            if nstep <= 0:
                nstep = 1
            noverlap = nperseg - nstep
        else:
            noverlap = 0
    else:
        noverlap = 0

    console.print(
        f"[cyan]Executing:[/] {args.transformation.upper()} (window size: {nperseg}, overlap: {noverlap})\n"
    )

    # Center data to remove DC artifacts
    b_centered = b_sampled - np.mean(b_sampled)

    f, t_stft, Zxx = compute_stft(
        b_centered,
        args.freq,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    # Shift time to be absolute relative to trace start
    t_stft += time_stamps[0]
    t_start_trace = time_stamps[0]
    t_end_trace = time_stamps[-1]

    # Calculate magnitude
    amp_stft = np.abs(Zxx)
    amp_mean = np.mean(amp_stft, axis=1)

    #! Find the dominant frequency in each time window using outlier detection
    dominant_freqs = []
    conf_vals = []
    amplitudes = []
    phases = []
    ranges = []

    # Disable plotting for per-window outlier detection to avoid overhead/multiple plots
    original_engine = args.engine
    args.engine = "no"

    # Calculate window width
    half_width = nperseg / (2 * args.freq)

    for i in range(amp_stft.shape[1]):
        # Perform outlier detection on each window
        d_idx, d_conf, _ = outlier_detection(amp_stft[:, i], f, args)

        # Exclude DC component (index 0)
        d_idx_no_dc = [idx for idx in d_idx if idx > 0]
        if d_idx_no_dc:
            # Pick the one with the highest amplitude among the detected outliers
            best_idx = d_idx_no_dc[np.argmax(amp_stft[d_idx_no_dc, i])]
            try:
                c = d_conf[best_idx - 1]
            except IndexError:
                c = 0
        else:
            # Fallback to the maximum amplitude (excluding DC) if no outlier is found
            best_idx = np.argmax(amp_stft[1:, i]) + 1
            c = 0  # Low confidence

        dominant_freqs.append(f[best_idx])
        conf_vals.append(c)
        amplitudes.append(amp_stft[best_idx, i])
        phases.append(np.angle(Zxx[best_idx, i]))

        # Calculate window range and clip to trace bounds
        win_t_start = np.clip(t_stft[i] - half_width, t_start_trace, t_end_trace)
        win_t_end = np.clip(t_stft[i] + half_width, t_start_trace, t_end_trace)
        ranges.append([win_t_start, win_t_end])

    # Restore original engine
    args.engine = original_engine

    #! Prepend the overall dominant frequency from amp_mean as the first entry
    best_idx_global = np.argmax(amp_mean[1:]) + 1
    # Global summary entry
    dominant_freqs.insert(0, f[best_idx_global])
    conf_vals.insert(0, 1.0)  # Use 100% confidence for the primary global result
    amplitudes.insert(0, amp_mean[best_idx_global])
    phases.insert(0, np.angle(np.mean(Zxx[best_idx_global, :], axis=0)))
    ranges.insert(0, [t_start_trace, t_end_trace])

    #! Per-window rank count, only when the trace's rank count actually
    #! varies (malleability) -- otherwise leave it empty and let consumers
    #! fall back to the constant `ranks` scalar. rank_sequence has its own
    #! time basis (rank_sequence_time), not time_stamps -- see docstring.
    ranks_per_window: list[int] = []
    if (
        rank_sequence is not None
        and rank_sequence_time is not None
        and len(rank_sequence) == len(rank_sequence_time)
        and len(rank_sequence) > 0
        and len(np.unique(rank_sequence)) > 1
    ):
        rank_sequence = np.asarray(rank_sequence)
        rank_sequence_time = np.asarray(rank_sequence_time)
        for win_t_start, win_t_end in ranges:
            mask = (rank_sequence_time >= win_t_start) & (rank_sequence_time <= win_t_end)
            if np.any(mask):
                values, counts = np.unique(rank_sequence[mask], return_counts=True)
                ranks_per_window.append(int(values[np.argmax(counts)]))
            else:
                ranks_per_window.append(ranks)

    #! Assign data to prediction (global summary + windows)
    prediction.dominant_freq = np.array(dominant_freqs)
    prediction.conf = np.array(conf_vals)
    prediction.amp = np.array(amplitudes)
    prediction.phi = np.array(phases)
    prediction.t_start = t_start_trace
    prediction.t_end = t_end_trace
    prediction.freq = args.freq
    prediction.ranks = ranks
    prediction.ranks_per_window = np.array(ranks_per_window)
    prediction.total_bytes = total_bytes
    prediction.n_samples = len(b_sampled)
    prediction.ranges = np.array(ranges)

    # Save up to n_freq from the top candidates based on amp_mean
    if args.n_freq > 0:
        n = len(amp_mean)
        arr = amp_mean[0 : int(np.ceil(n / 2))]
        top_candidates = np.argsort(-arr)  # from max to min
        n_freq = int(min(len(arr), args.n_freq))

        # Approximate phase for global frequencies from the mean complex spectrum
        phi_global = np.angle(np.mean(Zxx, axis=1))

        prediction.top_freqs = {
            "freq": f[top_candidates[0:n_freq]],
            "conf": np.zeros(n_freq),  # Global confidence is not available
            "periodicity": np.zeros(n_freq),
            "amp": amp_mean[top_candidates[0:n_freq]],
            "phi": phi_global[top_candidates[0:n_freq]],
        }

    periodicity_score = new_periodicity_scores(amp_mean, b_sampled, prediction, args)

    if getattr(args, "burst_width", False):
        from ftio.freq.duty_cycle import estimate_burst_widths
        from ftio.plot.plot_burst_width import plot_burst_width

        prediction.burst_widths = estimate_burst_widths(
            b_sampled, prediction, getattr(args, "burst_energy_fraction", 0.95)
        )
        analysis_figures.add_figure(
            [
                plot_burst_width(
                    prediction, b_sampled, getattr(args, "burst_energy_fraction", 0.95)
                )
            ],
            "burst_width",
        )

    #! Prepare for plotting
    if any(x in args.engine for x in ["mat", "plot"]):
        console.print(f"Generating {args.transformation.upper()} Plot\n")
        # STFT Spectrogram and Reconstruction Plots
        figs = plot_stft(
            args, prediction, b_sampled, args.freq, stft_data=(f, t_stft, Zxx)
        )
        if figs:
            analysis_figures.add_figure(figs, "stft")

        # Reuse AnalysisFigures for frequency spectrum plot
        t_sampled = t_start_trace + np.arange(len(b_sampled)) / args.freq
        analysis_figures.set_bulk(
            args,
            bandwidth,
            time_stamps,
            b_sampled,
            t_sampled,
            f,
            amp_mean,
            None,
            None,  # No global confidence available
            ranks,
        )
        if any(x in args.engine for x in ["mat", "plot"]) and args.runtime_plots:
            analysis_figures.show()
        console.print(" --- Done --- \n")

    text = Group(text, periodicity_score)

    console.print(
        Panel.fit(
            text,
            style="white",
            border_style="cyan",
            title=args.transformation.upper(),
            title_align="left",
        )
    )
    console.print(
        f"\n[cyan]{args.transformation.upper()} finished:[/] {time.time() - tik:.3f} s"
    )

    return prediction, analysis_figures
