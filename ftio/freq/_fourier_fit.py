"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from rich.panel import Panel
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.plot.helper import format_plot
from ftio.plot.units import set_unit


def fourier_sum(t: np.ndarray, *params) -> np.ndarray:
    """
    Computes the sum of multiple cosine functions (Fourier components) with given amplitudes, frequencies, and phases.

    Args:
        t (np.ndarray): Input array representing time or the independent variable.
        *params : Variable length argument list. Each group of three values corresponds to the amplitude, frequency,
        and phase of a cosine component, in the order: [amp1, freq1, phi1, amp2, freq2, phi2, ...].

    Returns:
        out (np.ndarray) : ndarray
            Array of the same shape as `t`, containing the sum of the cosine components evaluated at each value of `t`.
    """
    out = np.zeros_like(t)
    n = len(params) // 3
    for i in range(n):
        amp = params[3 * i]
        freq = params[3 * i + 1]
        phi = params[3 * i + 2]
        out += amp * np.cos(2 * np.pi * freq * t + phi)
    return out


def fourier_fit(
    args: Namespace,
    prediction: Prediction,
    analysis_figures: AnalysisFigures,
    b_sampled: np.ndarray,
    t: np.ndarray,
    maxfev: int = 50000,
) -> None:
    """
    Fits a sum of Fourier components to sampled data using non-linear least squares.
    Parameters. Finds the best frequencies (and other params) by minimizing error, starting from initial guess
    (usually DFT peaks). It:
    - Finds frequencies off the DFT grid (non-integer multiples)
    - Refines amplitude and phase estimates
    - Handles noise better by fitting parameters globally
    - Uses initial guesses often derived from the DFT maxima

    The filed top_freq is changed in Prediction is overwritten by the new results

    Args:
        args (Namespace): Arguments containing configuration, must include 'n_freq' (number of frequencies).
        prediction (Prediction): Prediction class containing initial predictions for amplitudes ('amp'), frequencies ('freq'), and phases ('phi')
            under the field 'top_freq'.
        analysis_figures (AnalysisFigures): Data and plot figures.
        b_sampled (np.ndarray): Sampled signal data to fit.
        t (np.ndarray): Time points corresponding to the sampled data.
        maxfev (int, optional): Maximum number of function evaluations for the optimizer (default is 10000).
    Returns:
        None
    """
    p0 = []
    n = len(b_sampled)
    console = MyConsole()
    console.set(args.verbose)

    if args.reconstruction and max(args.reconstruction) < args.n_freq:
        args.reconstruction.append(args.n_freq)

    for i in range(args.n_freq):
        if prediction.top_freqs["freq"][i] == 0:
            scale = 1 / n
        else:
            scale = 2 / n

        p0 += [
            scale * prediction.top_freqs["amp"][i],
            prediction.top_freqs["freq"][i],
            prediction.top_freqs["phi"][i],
        ]

    f_nyquist = 1 / (2 * (t[1] - t[0]))
    b_max = max(b_sampled)
    lower_bounds = []
    upper_bounds = []
    for _ in range(args.n_freq):
        lower_bounds += [-b_max, 0, -np.pi]
        upper_bounds += [b_max, 2 * f_nyquist, np.pi]

    # Fit Fourier sum model
    text = f"maxfev:{maxfev}\n"
    params_opt, _ = curve_fit(
        fourier_sum,
        t,
        b_sampled,
        p0=p0,
        maxfev=maxfev,
        # bounds=(lower_bounds, upper_bounds),
        method="dogbox",
    )

    opt_mag = params_opt[0::3]  # Magnitude is 2*amp/n
    opt_amp = np.zeros_like(opt_mag)
    opt_freq = params_opt[1::3]
    opt_phi = params_opt[2::3]
    for i, freq in enumerate(opt_freq):
        if freq == 0:
            opt_amp[i] = n * opt_mag[i]
        else:
            opt_amp[i] = n * opt_mag[i] / 2

    #  Adapt negative freq, as these require flipping phi:
    opt_phi = np.where(opt_freq < 0, -opt_phi, opt_phi)
    opt_freq = np.abs(opt_freq)

    prediction.set(
        "top_freqs",
        {
            "conf": np.repeat(1, args.n_freq),
            "amp": opt_amp,
            "freq": opt_freq,
            "phi": opt_phi,
        },
    )

    dft_res = fourier_sum(t, *p0)
    fitted = fourier_sum(t, *params_opt)
    if any(x in args.engine for x in ["mat", "plot"]):
        console.print("Generating Fourier Fit Plot\n")
        fig = plot_fourier_fit(args, t, b_sampled, prediction, fitted, dft_res)
        analysis_figures.add_figure_and_show([fig], "fourier_fit")
        console.print(" --- Done --- \n")

    error_before = mean_squared_error(b_sampled, dft_res)
    error_after = mean_squared_error(b_sampled, fitted)
    improvement = error_before - error_after
    improvement_pct = (
        100 * improvement / error_before if error_before != 0 else float("inf")
    )
    text += (
        f"Fourier fit improvement: \nMSE reduced from {error_before:.3e} to {error_after:.3e} \n"
        f"--> {improvement_pct:.2f}% improvement"
    )
    console.print(
        Panel.fit(
            text,
            style="white",
            border_style="green",
            title="Fourier Fit",
            title_align="left",
        )
    )


def plot_fourier_fit(
    args: Namespace,
    t: np.ndarray,
    b_sampled: np.ndarray,
    prediction: Prediction,
    fourier_fit,
    dft_res,
    show_top=False,
):
    """
    Plot sampled signal and Fourier sum using either matplotlib or plotly
    depending on args.engine content.

    Args:
        args: Object with attribute 'engine' specifying plotting backend
        t (array-like): Time vector
        b_sampled (array-like): Sampled signal data
        prediction : Prediction from FTIO
        fourier_fit (array-like): Fourier sum data
        dft_res (array-like): DFT data
        show_top (bool, optional): If True, show the top frequencies

    Returns:
        matplotlib.figure.Figure or plotly.graph_objs.Figure:
            Figure object for further manipulation or display.
    """

    components = []
    unit, order = set_unit(b_sampled)
    fourier_fit *= order
    dft_res *= order
    b_sampled *= order
    N = len(prediction.top_freqs["freq"])
    if show_top:
        for k in range(N):
            amp = order * prediction.top_freqs["amp"][k] / N
            if k != 0 and not (N % 2 != 0 and k == N - 1):
                freq = prediction.top_freqs["freq"][k]
                amp *= 2
                phi = prediction.top_freqs["phi"][k]
                trace, label = prediction.get_wave_and_name(freq, amp, phi)
                components.append((trace, label))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if "mat" in args.engine.lower():
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            t,
            b_sampled,
            linestyle="--",
            color=colors[0],
            linewidth=2,
            label="Sampled Signal",
        )
        ax.plot(
            t,
            fourier_fit,
            color=colors[1],
            linewidth=2.5,
            label=f"Fourier Fit {N}-Components Sum",
        )
        ax.plot(t, dft_res, color=colors[2], linewidth=2, label=f"DFT {N}-Component Sum")

        if show_top:
            for i, (wave, label) in enumerate(components):
                ax.plot(
                    t,
                    wave,
                    linestyle="-",
                    linewidth=1.5,
                    color=colors[(3 + i) % 10],
                    label=label,
                )

        ax.set_xlabel("Time (s)", fontsize=17)
        ax.set_ylabel(f"Bandwidth ({unit})", fontsize=17)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, 3))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-5, 3))
        ax.tick_params(axis="both", labelsize=12)
        plt.xlim(t[0], t[-1])
        ax.legend(frameon=True, shadow=True, fontsize=13, loc="upper left", ncol=2)
        fig.tight_layout()

    else:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=t,
                y=b_sampled,
                mode="lines",
                name="Sampled Signal",
                line={"dash": "dash", "width": 3, "color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=fourier_fit,
                mode="lines",
                name=f"Fourier Fit {N}-Components Sum",
                line={"width": 4, "color": "orange"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dft_res,
                mode="lines",
                name=f"DFT {N}-Component Sum",
                line={"width": 2, "color": "green"},
            )
        )

        if show_top:
            plotly_colors = ["red", "purple", "brown", "magenta", "cyan"]
            for i, (wave, label) in enumerate(components):
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=wave,
                        mode="lines",
                        name=label,
                        line={"width": 2, "color": plotly_colors[i % len(plotly_colors)]},
                    )
                )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=f"Bandwidth ({unit})",
        )
        format_plot(fig)

    return fig
