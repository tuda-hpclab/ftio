"""
Contains functions that execute workflow using the continuous Wavelet Transform.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import time
from argparse import Namespace

import numpy as np
from pywt import frequency2scale
from scipy.signal import find_peaks

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq._dft_workflow import ftio_dft
from ftio.freq._wavelet import wavelet_cont
from ftio.freq._wavelet_helpers import get_scales
from ftio.freq.discretize import sample_data
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.plot.plot_wavelet_cont import (  # plot_spectrum, plot_wave_cont
    plot_scales,
    plot_scales_all_in_one,
    plot_wave_cont_and_spectrum,
)
from ftio.prediction.helper import get_dominant_and_conf

# from ftio.analysis._logicize import logicize


def ftio_wavelet_cont(
    args: Namespace,
    bandwidth: np.ndarray,
    time_stamps: np.ndarray,
    ranks: int = 0,
):
    """
    Executes continuous wavelet transformation on the provided bandwidth data.

    Args:
        args: The Namespace object containing configuration options.
        bandwidth (np.ndarray): The bandwidth data to process.
        time_stamps (np.ndarray): The corresponding time points for the bandwidth data.
        ranks (int): The rank value (default is 0).
    """
    #! Default values for variables
    prediction = Prediction(args.transformation)
    console = MyConsole(verbose=args.verbose)

    dominant_freq = np.nan
    t_sampled = np.array([])
    mw_central_frequency = 1

    #! Sample the bandwidth evenly spaced in time
    tik = time.time()
    console.print("[cyan]Executing:[/] Discretization\n")
    # bandwidth = logicize(bandwidth)
    b_sampled, args.freq = sample_data(bandwidth, time_stamps, args)
    console.print(f"\n[cyan]Discretization finished:[/] {time.time() - tik:.3f} s")

    tik = time.time()
    console.print("[cyan]Finding Scales:[/] \n")

    #! Continuous Wavelet transform
    # TODO: use DFT to select the scales (see tmp/test.py)
    # FIXME: determine how to specify the step of the scales

    # args.wavelet = "morl"
    # args.wavelet = "haar"
    # args.wavelet = 'cmor' #Phase is not rally needed, no need for complex
    # args.wavelet = 'mexh'
    # args.wavelet = 'morl'
    # args.wavelet = "cmor1.5-1.0"

    # NOTE: By making the signal alternate, the wavelet achieves better results
    # b_sampled = b_sampled - np.mean(b_sampled)
    # method = "dft"
    method = "scale"
    args.wavelet = "mexh"
    if "scale" in method:
        scales = get_scales(args, b_sampled)
    elif "dft" in method:  # use dft
        args.n_freq = 5
        use_dominant_only = False
        scales = []
        t_sampled = time_stamps[0] + np.arange(0, len(b_sampled)) * 1 / args.freq
        prediction, analysis_figures = ftio_dft(args, b_sampled, t_sampled)
        dominant_freq, _ = get_dominant_and_conf(prediction)

        # Adjust wavelet
        mw_bandwidth = 1 / (dominant_freq)
        mw_central_frequency = dominant_freq  # /args.freq
        use_custom_cmor = False
        if not np.isnan(dominant_freq) and use_custom_cmor:
            args.wavelet = f"cmor{mw_bandwidth:f}-{mw_central_frequency:f}"
            # args.wavelet = f"cmor{args.freq/dominant_freq}-{dominant_freq}"
        else:
            # This works better
            args.wavelet = "mexh"
            mw_central_frequency = 1

        # Use only the dominant frequency
        if use_dominant_only and not np.isnan(dominant_freq):
            scales = frequency2scale(args.wavelet, np.array([dominant_freq])) * args.freq
        # use the top frequencies
        elif not use_dominant_only:
            top_freqs = prediction.top_freqs
            top_freqs = np.delete(top_freqs, np.where(top_freqs == 0))
            if len(top_freqs) > 0:
                if mw_central_frequency == 1:
                    scales = (
                        frequency2scale(args.wavelet, np.array(top_freqs)) * args.freq
                    )
                else:
                    scales = mw_central_frequency / np.array(top_freqs)

                console.print(f"Wavelet: {args.wavelet}")
                console.print(f"Scales: {scales}")
                console.print(f"Dominant freq: {dominant_freq}")
                console.print(f"Top freq: {top_freqs}")

        if len(scales) == 0:
            console.info(
                "[red] No dominant freq found with DFT -> Falling back to default method [/]"
            )
            scales = get_scales(args, b_sampled)

    tik = time.time()
    console.print(f"\n[cyan]Scales found:[/] {time.time() - tik:.3f} s")
    console.print(f"[cyan]Executing:[/] {args.transformation.upper()} + {args.outlier}\n")

    # b_sampled = b_sampled - np.mean(b_sampled) #downshift by average
    coefficients, frequencies = wavelet_cont(b_sampled, args.wavelet, scales, args.freq)
    power_spectrum = np.abs(coefficients) ** 2  # Power of the wavelet transform

    # FIXME: Rather than averaging the power, find the dominant frequency by examining the power spectrum
    # NOTE: The power spectrum specifies how much a examined frequency is presented in a signal contributes to the signal
    # NOTE: Don't rely on the hight of the power spectrum to find the dominant frequency
    # Find the dominant scale by averaging the power across all time points
    average_power_per_scale = power_spectrum.mean(axis=1)
    dominant_scale_idx = np.argmax(average_power_per_scale)
    dominant_scale = scales[dominant_scale_idx]
    # Norm the power spectrum
    # power_spectrum = power_spectrum/np.max(power_spectrum[dominant_scale_idx, :])

    # Extract the power at the dominant scale
    power_at_scale = power_spectrum[dominant_scale_idx, :]
    # Extract dominant frequency
    dominant_frequency = frequencies[dominant_scale_idx]
    console.print(
        f"dominant_scale: {dominant_scale:.3f}, dominant_frequency: {dominant_frequency:.3f}"
    )

    # Find local peaks in the power spectrum (use a sliding window approach)
    # samples per time step
    distance = int(args.freq / dominant_frequency)
    peaks, _ = find_peaks(
        power_at_scale, distance=distance
    )  # 'distance' avoids detecting too close peaks
    # use cwt and find peaks
    # peaks = find_peaks_cwt(power_at_scale, np.arange(distance,5*distance))
    if len(t_sampled) == 0:
        t_sampled = time_stamps[0] + np.arange(0, len(b_sampled)) * 1 / args.freq

    peak_times = t_sampled[peaks]

    # find all peaks
    all_peaks = [find_peaks(p)[0] for p in power_spectrum]

    # plot functions
    if any(x in args.engine for x in ["mat", "plot"]):
        console.print(f"Generating {args.transformation.upper()} Plot\n")
        analysis_figures = AnalysisFigures(
            args,
            bandwidth,
            time_stamps,
            b_sampled,
            t_sampled,
            frequencies,
            scales=scales,
            coefficients=coefficients,
        )
        # plot_spectrum(args, t, power_at_scale, dominant_scale, peaks)
        # _ = plot_wave_cont(b_sampled, frequencies, args.freq, time_b, coefficients)
        f = [
            plot_wave_cont_and_spectrum(
                args,
                t_sampled,
                frequencies,
                power_spectrum,
                power_at_scale,
                dominant_scale,
                peaks,
            ),
            plot_scales(args, t_sampled, b_sampled, power_spectrum, frequencies, scales),
            plot_scales_all_in_one(
                args,
                t_sampled,
                b_sampled,
                power_spectrum / np.max(b_sampled),
                frequencies,
                scales,
            ),
        ]
        if "ploty" in args.engine:
            f.append(
                plot_scales_all_in_one(
                    args,
                    t_sampled,
                    b_sampled,
                    coefficients,
                    frequencies,
                    scales,
                    all_peaks,
                )
            )

        for i, fig in enumerate(f):
            analysis_figures.add_figure([fig], f"wavelet_cont_{i}")
        console.print(" --- Done --- \n")
    else:
        analysis_figures = AnalysisFigures()

    # Calculate the period (time difference between consecutive peaks)
    if len(peak_times) > 1:
        np.diff(peak_times)  # Differences between consecutive peak times (periods)
    else:
        pass

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

    console.print(
        f"\n[cyan]{args.transformation.upper()} + {args.outlier} finished:[/] {time.time() - tik:.3f} s"
    )

    return prediction, analysis_figures
