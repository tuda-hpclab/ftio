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

import copy
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq._dft_workflow import ftio_dft
from ftio.freq._dft_x_dwt import analyze_correlation
from ftio.freq._wavelet import wavelet_disc
from ftio.freq._wavelet_helpers import (
    decomposition_level,
    upsample_coefficients,
    wavelet_freq_bands,
)
from ftio.freq.autocorrelation import find_fd_autocorrelation
from ftio.freq.discretize import sample_data
from ftio.freq.helper import MyConsole
from ftio.plot.plot_wavelet_disc import (
    plot_coeffs_reconst_signal,
    plot_wavelet_disc_spectrum,
)
from ftio.processing.print_output import display_prediction

# from ftio.prediction.helper import[] get_dominant_and_conf


def ftio_wavelet_disc(
    args: Namespace,
    bandwidth: np.ndarray,
    time_stamps: np.ndarray,
    ranks: int = 0,
    total_bytes: int = 0,
):
    """
    Executes discrete wavelet transformation on the provided bandwidth data.

    Args:
        args: The Namespace object containing configuration options.
        bandwidth (np.ndarray): The bandwidth data to process.
        time_stamps (np.ndarray): The corresponding time points for the bandwidth data.
        ranks (int): The rank value (default is 0).
        total_bytes (int): total transferred bytes (default is 0).
    """
    # Default values for variables
    prediction = {
        "source": {args.transformation},
        "dominant_freq": [],
        "conf": [],
        "t_start": 0,
        "t_end": 0,
        "total_bytes": 0,
        "freq": 0,
        "ranks": 0,
    }
    console = MyConsole(verbose=args.verbose)

    # ! Sample the bandwidth evenly spaced in time
    tik = time.time()
    console.print("[cyan]Executing:[/] Discretization\n")
    b_sampled, args.freq = sample_data(bandwidth, time_stamps, args)
    console.print(f"\n[cyan]Discretization finished:[/] {time.time() - tik:.3f} s")

    # b_sampled = logicize(b_sampled)

    tik = time.time()
    console.print(f"[cyan]Executing:[/] {args.transformation.upper()} + {args.outlier}\n")

    # Ensure wavelet is set early
    if not args.wavelet:
        args.wavelet = "db1"

    # ! Find the level for the discrete wavelet
    # https://edisciplinas.usp.br/pluginfile.php/4452162/mod_resource/content/1/V1
    # -Parte%20de%20Slides%20de%20p%C3%B3sgrad%20PSI5880_PDF4%20em%20Wavelets%20
    # -%202010%20-%20Rede_AIASYB2.pdf
    # https://www.youtube.com/watch?v=hAQQwvKsWCY&ab_channel=NathanKutz
    console.print("[green]Performing discrete wavelet decomposition[/]")
    # Max level or use DFT
    method = "dft"
    if args.level == 0:
        if method == "dft":
            args.transformation = "dft"
            prediction, _ = ftio_dft(args, bandwidth, time_stamps, total_bytes, ranks)
            if len(prediction.dominant_freq) > 0:
                args.level = int(1 / (5 * prediction.get_dominant_freq()))
                console.print(f"[green]Decomposition level adjusted to {args.level}[/]")
                args.transformation = "wave_disc"
            else:
                args.level = decomposition_level(args, len(b_sampled))
        else:
            args.level = decomposition_level(args, len(b_sampled))

    # ! calculate the coefficients using the discrete wavelet
    # coefficients ->  [cA_n, cD_n, cD_n-1, …, cD2, cD1]
    coefficients = wavelet_disc(b_sampled, args.wavelet, args.level)
    # compute the frequency ranges
    freq_ranges = wavelet_freq_bands(args.freq, args.level)
    # Upsample coefficients for equal length
    t_sampled = time_stamps[0] + np.arange(0, len(b_sampled)) * 1 / args.freq
    coefficients_upsampled = upsample_coefficients(
        coefficients, args.wavelet, len(b_sampled)
    )
    # plot functions
    if any(x in args.engine for x in ["mat", "plot"]):
        console.print(f"Generating {args.transformation.upper()} Plot\n")
        analysis_figures_wavelet = AnalysisFigures(
            args, coefficients=coefficients_upsampled
        )
        f1 = plot_coeffs_reconst_signal(
            args,
            time_stamps,
            bandwidth,
            t_sampled,
            b_sampled,
            coefficients_upsampled,
            freq_ranges,
        )
        f2 = plot_wavelet_disc_spectrum(
            args, t_sampled, coefficients_upsampled, freq_ranges
        )

        analysis_figures_wavelet.add_figure(f1, "wavelet_disc")
        analysis_figures_wavelet.add_figure(f2, "wavelet_disc_spectrum")

        if args.runtime_plots:
            analysis_figures_wavelet.show()
        console.print(" --- Done --- \n")
    else:
        analysis_figures_wavelet = AnalysisFigures(args)

    # ! Perform analysis on the result from the DWT
    # analysis = "dft_on_approx_coeff"
    # analysis = "dft_on_all"
    analysis = "dft_x_dwt"
    # analysis = "dwt_x_autocorrelation"
    # if len(coefficients) <= 2:
    #     analysis = "dft_on_all"
    #     console.print(f"[green]Setting analysis to {analysis}")

    # ? Option 1 ("dft_on_approx_coeff"): Execute DFT on approx. coefficients from DWT
    # cont = input("\nContinue with the DFT? [y/n]")
    # if len(cont) == 0 or "y" in cont.lower():
    if "dft_on_approx_coeff" in analysis:
        args.transformation = "dft"
        # Option 1: Filter using wavelet and call DFT on lowest last coefficient
        prediction, analysis_figures_dft = ftio_dft(
            args, coefficients_upsampled[0], t_sampled, total_bytes, ranks
        )
        analysis_figures_wavelet += analysis_figures_dft

    # ? Option 2: Find intersection between DWT and DFT
    elif "dft_on_all" in analysis:
        args.transformation = "dft"
        # TODO: For this to be parallel, the generated HTML files need different names as
        #  they are overwritten
        with ProcessPoolExecutor(max_workers=1) as executor:
            # submit futures
            futures = {}
            for i, coeffs in enumerate(coefficients_upsampled):
                tmp_args = copy.deepcopy(args)
                tmp_args.plot_name = f"ftio_dwt{i}_result"
                future = executor.submit(
                    ftio_dft, tmp_args, coeffs, t_sampled, total_bytes, ranks
                )
                futures[future] = i

            # Process futures as they complete
            for future in as_completed(futures):
                prediction, analysis_figures_dft = future.result()
                display_prediction(["ftio"], prediction)
                index = futures[future]
                analyze_correlation(
                    args,
                    prediction,
                    coefficients_upsampled[index],
                    t_sampled,
                    analysis_figures_dft,
                    bandwidth,
                    time_stamps,
                )
                console.print(f"[green]{index} completed[/]")
            exit()

    # ? Option 3: Find intersection between DWT and DFT
    elif "dft_x_dwt" in analysis:
        args.transformation = "dft"
        # 1): compute DFT on lowest last coefficient
        prediction, analysis_figures_dft = ftio_dft(
            args, coefficients_upsampled[0], t_sampled, total_bytes, ranks
        )
        # 2) compare the results
        analyze_correlation(
            args,
            prediction,
            coefficients_upsampled[0],
            t_sampled,
            analysis_figures_dft,
            bandwidth,
            time_stamps,
        )
        if any(x in args.engine for x in ["mat", "plot"]):
            analysis_figures_wavelet += analysis_figures_dft

    # ? Option 4: Apply autocorrelation on low
    elif "dwt_x_autocorrelation" in analysis:
        find_fd_autocorrelation(
            args,
            coefficients_upsampled[0],
            args.freq,
            analysis_figures_wavelet,
        )
        if any(x in args.engine for x in ["mat", "plot"]) and args.runtime_plots:
            analysis_figures_wavelet.show()
        exit()

    if any(x in args.engine for x in ["mat", "plot"]) and args.runtime_plots:
        analysis_figures_wavelet.show()

    if getattr(args, "burst_width", False):
        from ftio.freq.duty_cycle import estimate_burst_widths
        from ftio.plot.plot_burst_width import plot_burst_width

        prediction.burst_widths = estimate_burst_widths(
            b_sampled, prediction, getattr(args, "burst_energy_fraction", 0.95)
        )
        analysis_figures_wavelet.add_figure(
            [
                plot_burst_width(
                    prediction, b_sampled, getattr(args, "burst_energy_fraction", 0.95)
                )
            ],
            "burst_width",
        )

    return prediction, analysis_figures_wavelet
