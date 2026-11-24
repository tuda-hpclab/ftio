"""
AMD workflow for FTIO. This module implements Adaptive Mode Decomposition (AMD)
based frequency analysis for FTIO. It provides functionality to identify periodic
time windows in bandwidth time-series using Variational Mode Decomposition
(VMD) and Empirical Fourier Decomposition (EFD). The results are integrated
with FTIO’s prediction and reporting framework.

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

from ftio.freq._amd import amd
from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.discretize import sample_data
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction


def ftio_amd(
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
    console.print("[cyan]Executing:[/] Discretization\n")
    b_sampled, freq = sample_data(bandwidth, time_samples, args)
    console.print(f"\n[cyan]Discretization finished:[/] {time.time() - tik:.3f} s")

    # Perform AMD
    components, figs = amd(
        b_sampled, freq, time_samples, args, b_orig=bandwidth, t_orig=time_samples
    )

    # Fill prediction object
    dominant_freqs = []
    conf_vals = []
    amplitudes = []
    phases = []
    ranges = []

    if components:
        # Duration for mapping indices to time
        duration = time_samples[-1] - time_samples[0]
        N = len(b_sampled)

        # Track the best component for global summary
        best_comp_idx = 0
        max_energy = -1

        for i, c in enumerate(components):
            # Handle different formats (VMD returns list of tuples, EFD returns Component namedtuples)
            if hasattr(c, "freq"):  # Namedtuple (EFD/simple_astft)
                f_val = c.freq
                a_val = c.amp
                p_val = c.phase
                s_idx = c.start
                e_idx = c.end
            else:  # Tuple (VMD imf_selection: (time, index, freq))
                f_val = c[2]
                a_val = 0  # Not directly available in VMD tuple
                p_val = 0  # Not directly available in VMD tuple
                s_idx = c[0][0]
                e_idx = c[0][1]

            dominant_freqs.append(f_val)
            conf_vals.append(0.85)  # High confidence for linked components
            amplitudes.append(a_val)
            phases.append(p_val)

            # Map indices back to time
            t_start = time_samples[0] + (s_idx / N) * duration
            t_end = time_samples[0] + (e_idx / N) * duration
            ranges.append([t_start, t_end])

            # Simple energy-based heuristic for global summary
            energy = a_val * (e_idx - s_idx)
            if energy > max_energy:
                max_energy = energy
                best_comp_idx = i

        # Prepend the "best" component as the global summary
        dominant_freqs.insert(0, dominant_freqs[best_comp_idx])
        conf_vals.insert(0, 1.0)
        amplitudes.insert(0, amplitudes[best_comp_idx])
        phases.insert(0, phases[best_comp_idx])
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

    # Add figures if any were generated
    if figs:
        analysis_figures.add_figure(figs, "amd")

    # Display figures if needed
    if any(x in args.engine for x in ["mat", "plot"]) and args.runtime_plots:
        analysis_figures.show()

    return prediction, analysis_figures
