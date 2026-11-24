"""
This module contains functions to find the period using autocorrelation
and to filter outliers from the detected peaks.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import time
from argparse import Namespace

import numpy as np
from rich.panel import Panel
from scipy.signal import find_peaks

# from rich.padding import Padding
import ftio.freq.discretize as dis
from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.plot.plot_autocorrelation import plot_autocorr_results


def find_autocorrelation(
    args: Namespace,
    data: dict,
    analysis_figures: AnalysisFigures,
    prediction_freq_analysis: Prediction,
) -> Prediction:
    """Finds the period using autocorreleation

    Args:
        args (argparse.Namespace): Command line arguments
        data (dict): Sampled data
        analysis_figures (AnalysisFigures): Data and plot figures.
        prediction_freq_analysis (Prediction): Frequency analysis prediction contains bandwidth and time stamps.
    Returns:
        tuple:
            - prediction (Prediction): Contains prediction results including dominant frequency, confidence, amplitude, etc.
    """
    prediction = Prediction("")
    candidates = np.array([])
    console = MyConsole()
    console.set(args.verbose)
    tik = time.time()
    if args.autocorrelation:
        console.print("[cyan]Executing:[/] Autocorrelation\n")
        prediction.set_from_dict(
            {
                "source": "autocorrelation",
                "dominant_freq": np.array([]),
                "conf": np.array([]),
                "t_start": 0,
                "t_end": 0,
                "total_bytes": 0,
            }
        )

        # Take data if already aviable from previous step
        if len(prediction_freq_analysis.b_sampled) > 0:
            b_sampled = prediction_freq_analysis.b_sampled
            freq = prediction_freq_analysis.freq
            t_s = prediction_freq_analysis.t_start
            t_e = prediction_freq_analysis.t_end
            total_bytes = prediction_freq_analysis.total_bytes
        else:
            total_bytes = 0
            bandwidth = data["bandwidth"] if "bandwidth" in data else np.array([])
            time_stamps = data["time"] if "time" in data else np.array([])
            t_s = time_stamps[0]
            t_e = time_stamps[-1]
            if args.ts:  # shorten data
                time_stamps = time_stamps[time_stamps >= args.ts]
                t_s = time_stamps[0]
                bandwidth = bandwidth[len(bandwidth) - len(time_stamps) :]
                total_bytes = np.sum(
                    bandwidth
                    * (np.concatenate([time_stamps[1:], time_stamps[-1:]]) - time_stamps)
                )
                console.print(f"[purple]Start time set to {args.ts:.2f}")
            else:
                console.print(f"[purple]Start time: {time_stamps[0]:.2f}")

            if args.te:  # shorten data
                time_stamps = time_stamps[time_stamps <= args.te]
                t_s = time_stamps[-1]
                bandwidth = bandwidth[len(bandwidth) - len(time_stamps) :]
                total_bytes = np.sum(
                    bandwidth
                    * (np.concatenate([time_stamps[1:], time_stamps[-1:]]) - time_stamps)
                )
                console.print(f"[purple]End time set to {args.te:.2f}")
            else:
                console.print(f"[purple]End time: {time_stamps[-1]:.2f}")

            # sample the bandwidth
            b_sampled, freq = dis.sample_data(
                bandwidth, time_stamps, args.freq, args.verbose
            )

        res = find_fd_autocorrelation(args, b_sampled, freq, analysis_figures)

        # save the results
        prediction.dominant_freq = (
            1 / res["periodicity"] if res["periodicity"] > 0 else np.nan
        )
        prediction.conf = res["conf"]
        prediction.t_start = t_s
        prediction.t_end = t_e
        prediction.total_bytes = total_bytes
        prediction.freq = freq
        prediction.candidates = candidates
        console.print(f"\n[cyan]Autocorrelation finished:[/] {time.time() - tik:.3f} s")

    return prediction


def filter_outliers(
    freq: float, candidates: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, str]:
    """removes outliers using either qunatil method or Z-score

    Args:
        candidates (np.ndarray): peaks
        weights (np.ndarray): weights of the peaks

    Returns:
        np.ndarray: outliers
        str: string text
    """
    text = ""
    outliers = np.array([])
    # remove outliers:
    if len(candidates) > 0 and any(candidates > 0):
        ind = np.where(candidates > 1 / freq)
        candidates = candidates[ind]  # remove everythin above 10 Hz
        if len(weights > 0):
            weights = weights[ind]
        method = "z"
        # With quantil and weights
        if "q" in method:
            text += "Filtering method: [purple]quantil[/]\n"
            # candidates = candidates*weights/sum(weights)
            q1 = np.percentile(candidates, 25)
            q3 = np.percentile(candidates, 75)
            iqr = q3 - q1
            threshold = 1.5 * iqr
            outliers = np.where(
                (candidates < q1 - threshold) | (candidates > q3 + threshold)
            )
        elif "z" in method:
            text += "Filtering method: [purple]Z-score with weighteed mean[/]\n"
            # With Zscore:
            mean = np.average(candidates, weights=weights) if len(weights) > 0 else 0
            # std = np.std(candidates)
            std = (
                np.sqrt(np.abs(np.average((candidates - mean) ** 2, weights=weights)))
                if len(weights) > 0
                else 0
            )
            text += f"Wighted mean is [purple]{mean:.3f}[/] and weighted std. is [purple]{std:.3f}[/]\n"
            z_score = np.abs((candidates - mean) / std) if std != 0 else np.array([])
            outliers = np.where(z_score > 1)
            text += f"Z-score is [purple]{print_array(z_score)}[/]\n"

        text += (
            f"[purple]{len(candidates)}[/] period candidates found:\n"
            f"[purple]{print_array(candidates)}[/]\n\n"
            f"{len(candidates[outliers])} outliers found:\n[purple]{print_array(candidates[outliers])}[/]\n\n"
        )
    else:
        text += "[purple]Empty[/]\n"

    return outliers, text


def find_fd_autocorrelation(
    args: Namespace,
    b_sampled: np.ndarray,
    freq: float,
    analysis_figures: AnalysisFigures,
) -> dict:
    """
    Computes the autocorrelation of a sampled signal, detects peaks, and calculates periodicity and confidence
    based on the detected periods.

    Args:
        args (Namespace): Command line arguments.
        b_sampled (np.ndarray): The sampled input signal.
        freq (float): The frequency at which the signal is sampled.
        analysis_figures (AnalysisFigures): Data and plot figures.

    Returns:
        dict: A dictionary containing the autocorrelation, peak locations, weights,
                calculated periodicity, and confidence level.
    """
    # Compute autocorrelation of the sampled signal
    acorr = autocorrelation(b_sampled)

    # Finding peak locations and calculating the average time differences between them
    peaks, prop = find_peaks(acorr, height=0.15)
    candidates = np.diff(peaks) / freq
    weights = np.diff(prop["peak_heights"])

    # Removing outliers
    outliers, text = filter_outliers(freq, candidates, weights)
    if len(outliers) > 0:
        candidates = np.delete(candidates, outliers)
        weights = np.delete(weights, outliers)

    # Calculating period and its statistics
    if candidates.size > 0:
        mean = np.average(candidates, weights=weights)
        std = np.sqrt(np.abs(np.average((candidates - mean) ** 2, weights=weights)))
        tmp = [f"{1 / i:.4f}" for i in candidates]  # Formatting frequencies
        periodicity = mean
        coef_var = np.abs(std / mean)
        conf = 1 - coef_var
    else:
        mean = np.nan
        std = np.nan
        tmp = np.nan
        periodicity = np.nan
        conf = np.nan

    # Building output text
    text += (
        f"Found periods are [purple]{candidates}[/]\n"
        f"Matching frequencies are [purple]{tmp}[/]\n"
        f"Average period is [purple]{periodicity:.2f} [/]sec\n"
        f"Average frequency is [purple]{1 / periodicity if periodicity > 0 else np.nan:.4f} [/]Hz\n"
        f"Confidence is [purple]{conf * 100:.2f} [/]%\n"
    )
    console = MyConsole()
    console.set(args.verbose)
    console.print(
        Panel.fit(
            text[:-1],
            style="white",
            border_style="purple",
            title="Autocorrelation",
            title_align="left",
        )
    )

    # plot
    if any(x in args.engine for x in ["mat", "plot"]):
        console.print("Generating Autocorrelation Plot\n")
        fig = plot_autocorr_results(args, acorr, peaks, outliers, len(candidates) > 0)
        analysis_figures.add_figure([fig], "Autocorrelation")
        console.print(" --- Done --- \n")

    return {
        "autocorrelation": acorr,
        "candidates": candidates,
        "peaks": peaks,
        "outliers": outliers,
        "weights": weights,
        "periodicity": periodicity,
        "conf": conf,
    }


def autocorrelation(arr: np.ndarray) -> np.ndarray:
    """
    Computes the autocorrelation of a given array.

    Args:
        arr (np.ndarray): Input array for which to compute the autocorrelation.

    Returns:
        np.ndarray: Autocorrelation values.
    """
    # Scipy autocorrelation
    # lags = range(int(freq*len(arr)))
    # acorr = sm.tsa.acf(arr, nlags = len(lags)-1)
    #! numpy autocorrelation
    # # Mean
    mean = np.mean(arr)
    # Variance
    var = np.var(arr)
    # Normalized data
    ndata = arr - mean
    # Calculate autocorrelation:
    acorr = np.correlate(ndata, ndata, "full")[len(ndata) - 1 :]
    acorr = acorr / var / len(ndata)

    return acorr


def print_array(array: np.ndarray) -> str:
    out = ""
    if len(array) == 0:
        out = " "
    for i in array:
        out += f" {i:.2f}"

    return "[" + out[1:] + "]"
