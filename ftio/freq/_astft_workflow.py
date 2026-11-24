"""
ASTFT workflow for FTIO. This module implements the
Adaptive Short-Time Fourier Transform (STFT) processing pipeline used
by FTIO. It handles resampling of bandwidth time-series data, executes
the ASTFT analysis, and prepares prediction and analysis outputs in a
format compatible with the FTIO framework.

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import time
from argparse import Namespace

import numpy as np

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq._astft import astft
from ftio.freq.discretize import sample_data
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction


def ftio_astft(
    args: Namespace,
    bandwidth: np.ndarray,
    time_samples: np.ndarray,
    total_bytes: int,
    ranks: int,
    text: str = "",
) -> tuple[Prediction, AnalysisFigures]:

    # Default values for variables
    prediction = Prediction(args.transformation)
    analysis_figures = AnalysisFigures(args)
    console = MyConsole(verbose=args.verbose)

    #  Extract time series: Sample the bandwidth evenly spaced in time
    tik = time.time()
    console.print("[cyan]Executing:[/] Resampling\n")

    b_sampled, freq = sample_data(bandwidth, time_samples, args)

    console.print(f"\n[cyan]Resampling finished:[/] {time.time() - tik:.3f} s")

    # Perform ASTFT
    per_comp, figs = astft(b_sampled, freq, bandwidth, time_samples, args)

    # Fill prediction object
    dominant_freqs = []
    conf_vals = []
    amplitudes = []
    phases = []
    ranges = []

    if per_comp:
        # Duration for mapping indices to time
        duration = time_samples[-1] - time_samples[0]
        N = len(b_sampled)

        for c in per_comp:
            dominant_freqs.append(c.freq)
            # ASTFT components are often quite clear if linked
            conf_vals.append(0.9)
            amplitudes.append(c.amp)
            phases.append(c.phase)
            # Map indices back to time
            t_start = time_samples[0] + (c.start / N) * duration
            t_end = time_samples[0] + (c.end / N) * duration
            ranges.append([t_start, t_end])

        # Prepend the "best" component as the global summary to avoid recalculating FFT
        # We pick the one with the highest "energy" (amplitude * duration)
        best_comp_idx = 0
        max_energy = -1
        for i, c in enumerate(per_comp):
            energy = c.amp * (c.end - c.start)
            if energy > max_energy:
                max_energy = energy
                best_comp_idx = i

        dominant_freqs.insert(0, per_comp[best_comp_idx].freq)
        conf_vals.insert(0, 1.0)
        amplitudes.insert(0, per_comp[best_comp_idx].amp)
        phases.insert(0, per_comp[best_comp_idx].phase)
        ranges.insert(0, [time_samples[0], time_samples[-1]])

    prediction.dominant_freq = np.array(dominant_freqs)
    prediction.conf = np.array(conf_vals)
    prediction.amp = np.array(amplitudes)
    prediction.phi = np.array(phases)
    prediction.ranges = np.array(ranges)

    prediction.t_start = time_samples[0]
    prediction.t_end = time_samples[-1]
    prediction.freq = freq
    prediction.ranks = ranks
    prediction.total_bytes = total_bytes
    prediction.n_samples = len(b_sampled)

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

    # Add figures if any were generated
    if figs:
        analysis_figures.add_figure(figs, "astft")

    # Display figures if needed
    if any(x in args.engine for x in ["mat", "plot"]) and args.runtime_plots:
        analysis_figures.show()

    return prediction, analysis_figures
