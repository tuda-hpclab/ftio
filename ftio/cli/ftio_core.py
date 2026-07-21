"""
This file provides functions to process input data, perform frequency analysis, and generate predictions
using the FTIO framework. It includes core functionalities for data transformation, frequency analysis,
autocorrelation, and result visualization.
This functions calculates the frequency based on the input data provided.
Currently Darshan, recorder, and traces generated with our internal tool TMIO and the metric proxy are supported.

call ftio -h to see list of supported arguments.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace

import numpy as np

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq._dft_workflow import ftio_dft
from ftio.freq._stft_workflow import ftio_stft
from ftio.freq._wavelet_cont_workflow import ftio_wavelet_cont
from ftio.freq._wavelet_disc_workflow import ftio_wavelet_disc
from ftio.freq.autocorrelation import find_autocorrelation
from ftio.freq.helper import MyConsole, append_messages
from ftio.freq.prediction import Prediction
from ftio.freq.time_window import data_in_time_window
from ftio.parse.extract import get_time_behavior_and_args
from ftio.prediction.unify_predictions import merge_predictions
from ftio.processing.print_output import display_prediction


def main(
    cmd_input: list[str], msgs=None, shared_resource=None
) -> tuple[list[Prediction], Namespace]:  # -> dict[Any, Any]:
    """
    Main entry point to process input data, perform frequency analysis, and generate predictions.

    This function handles the following:
    - Parsing input arguments and reading trace data.
    - Performing frequency analysis based on the chosen transformation and mode.
    - Extracting time behavior and performing further analysis.
    - Generating predictions using the core processing logic.
    - Plotting and displaying results.

    Args:
        cmd_input (list[str]): A list of input strings containing the command-line arguments or data.
        msgs (optional): ZMQ message, in case the input is not provided via a file.
        shared_resource (optional): Shared resources among processes, in case of predictor analysis.

    Returns:
        tuple[list[Prediction], Namespace]:
            - A list of Prediction containing the prediction results.
            - A Namespace object with the parsed arguments used in the analysis.
    """

    start = time.time()
    # extract the data
    data, args = get_time_behavior_and_args(cmd_input, msgs)
    console = MyConsole(args.verbose)
    console.print(f"\n[cyan]Data imported in:[/] {time.time() - start:.2f} s")
    console.print(f"[cyan]Frequency Analysis:[/] {args.transformation.upper()}")
    console.print(f"[cyan]Mode:[/] {args.mode}")

    # Handle shared resource
    if shared_resource is not None:
        data = append_messages(data, shared_resource)

    list_analysis_figures = []
    list_predictions = []

    for sim in data:
        # get prediction
        prediction, analysis_figures = core(sim, args)
        # merge results from several files into one
        list_predictions.append(prediction)
        if any(x in args.engine for x in ["mat", "plot"]):
            list_analysis_figures.append(analysis_figures)

    # show merge results
    display_prediction(args, list_predictions)
    for analysis_figures in list_analysis_figures:
        analysis_figures.show()
    console.print(f"[cyan]Total elapsed time:[/] {time.time() - start:.3f} s\n")

    # Offline phase automaton: only for direct (non-predictor) STFT runs --
    # the online predictor already steps its own automaton once per elapsed-
    # time invocation (see ftio.prediction.online_analysis._automaton_step).
    # STFT already slides a window across the whole trace and reports a
    # dominant frequency per window, so that sequence can be fed to the
    # automaton directly, without a live ZMQ stream.
    if (
        shared_resource is None
        and getattr(args, "phase_automaton", False)
        and "stft" in args.transformation
        and "astft" not in args.transformation
    ):
        _run_offline_phase_automaton(args, list_predictions, console)

    return list_predictions, args


def _run_offline_phase_automaton(
    args: Namespace, predictions: list[Prediction], console: MyConsole
) -> None:
    """Build and export a PhaseAutomaton from an STFT run's per-window sequence.

    Each Prediction in `predictions` (one per input file/rank-group) already
    holds a per-window dominant-frequency sequence from `ftio_stft`; unpack it
    with `windows_from_stft_prediction` and feed the automaton one window at a
    time, exactly as the online predictor feeds it one Prediction per step.
    """
    from ftio.modeling.phase_automaton import (
        PhaseAutomaton,
        windows_from_stft_prediction,
    )

    for prediction in predictions:
        windows = windows_from_stft_prediction(prediction)
        if not windows:
            console.print(
                "[yellow]Phase automaton: no per-window STFT sequence found "
                "(need --transformation stft with more than one window); "
                "skipping.[/]"
            )
            continue

        aut = PhaseAutomaton.from_args(args)
        aut.build(windows)
        aut.print_summary()
        aut.print_graph()
        aut.save_json(getattr(args, "pa_export", "./phase_automaton.json"))


def core(sim: dict, args: Namespace) -> tuple[Prediction, AnalysisFigures]:
    """
    FTIO core function that processes data and arguments to generate predictions and results.

    This function processes input data containing time, bandwidth, among others and
    performs operations based on the parsed arguments.

    Args:
        sim (dict): dictionary where each dictionary contains:
            - "time" (np.ndarray): Array of time points.
            - "bandwidth" (np.ndarray): Array of bandwidth values.
            - "total_bytes" (int): Total number of bytes.
            - "ranks" (int): Number of ranks in the system.
        args (Namespace): Parsed arguments to customize the FTIO's behavior.

    Returns:
        tuple[dict, list]: A tuple where:
            - The first element is a dictionary with the obtained predictions (key-value pairs).
            - The second element is used for storing dataframes for plotting.
    """
    if not sim:
        return Prediction(), AnalysisFigures()

    # Perform frequency analysis (dft/wavelet)
    prediction_freq_analysis, analysis_figures = freq_analysis(args, sim)
    # Perform autocorrelation if args.autocorrelation is true + Merge the results into a single prediction
    prediction_auto = find_autocorrelation(
        args, sim, analysis_figures, prediction_freq_analysis
    )
    # Merge results
    prediction = merge_predictions(
        args, prediction_freq_analysis, prediction_auto, analysis_figures
    )

    return prediction, analysis_figures


def freq_analysis(args: Namespace, data: dict) -> tuple[Prediction, AnalysisFigures]:
    """
    Performs frequency analysis (DFT, continuous wavelet, or discrete wavelet) and prepares data for plotting.

    This function analyzes the provided data using the specified frequency technique
    (DFT, continuous wavelet, or discrete wavelet) and then creates a dataframe
    with relevant information for visualization and further analysis.

    Args:
        args (Namespace): Parsed arguments to customize the FTIO's behavior (see io_args.py).
        data (dict[str, np.ndarray]): A dictionary containing the following fields:
            - "bandwidth" (np.ndarray): An array representing the bandwidth values.
            - "time" (np.ndarray): An array representing the time points when the bandwidth changed.
            - "total_bytes" (int): The total number of transferred bytes.
            - "ranks" (int): The number of ranks involved in the I/O operation.

    Raises:
        Exception: If an unsupported method is passed in the `args.transformation`.

    Returns:
        tuple: A tuple containing:
            - Prediction: Contains the prediction results, including
                - "dominant_freq" (list): The identified dominant frequencies.
                - "conf" (np.ndarray): Confidence values corresponding to the dominant frequencies.
                - "t_start" (int): Start time of the analysis.
                - "t_end" (int): End time of the analysis.
                - "total_bytes" (int): Total bytes involved in the analysis.
            - AnalysisFigures
    """

    #! Init
    bandwidth = data["bandwidth"] if "bandwidth" in data else np.array([])
    time_b = data["time"] if "time" in data else np.array([])
    total_bytes = data.get("total_bytes", 0)
    ranks = data.get("ranks", 0)
    rank_sequence = data.get("rank_sequence")
    rank_sequence_time = data.get("rank_sequence_time")

    #! Extract relevant data
    bandwidth, time_b, text, rank_sequence, rank_sequence_time = data_in_time_window(
        args, bandwidth, time_b, total_bytes, ranks, rank_sequence, rank_sequence_time
    )

    #! Perform transformation
    if "dft" in args.transformation:
        prediction, analysis_figures = ftio_dft(
            args, bandwidth, time_b, total_bytes, ranks, text
        )

    elif "wave_disc" in args.transformation or "dwt" in args.transformation:
        prediction, analysis_figures = ftio_wavelet_disc(
            args, bandwidth, time_b, ranks, total_bytes
        )

    elif "wave_cont" in args.transformation or "cwt" in args.transformation:
        prediction, analysis_figures = ftio_wavelet_cont(args, bandwidth, time_b, ranks)

    elif "stft" in args.transformation and "astft" not in args.transformation:
        prediction, analysis_figures = ftio_stft(
            args,
            bandwidth,
            time_b,
            total_bytes,
            ranks,
            text,
            rank_sequence,
            rank_sequence_time,
        )

    elif "astft" in args.transformation:
        # Check dependencies for AMD-based methods
        try:
            import vmdpy  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "ASTFT transformation is disabled.\n"
                'Install with: pip install "ftio[amd-libs]" or pip install -e ".[amd-libs]"'
            ) from None

        from ftio.freq._astft_workflow import ftio_astft

        prediction, analysis_figures = ftio_astft(
            args, bandwidth, time_b, total_bytes, ranks, text
        )

    elif any(t in args.transformation for t in ("efd", "vmd")):
        # Check dependencies for AMD-based methods
        try:
            import vmdpy  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"{args.transformation.upper()} transformation is disabled.\n"
                'Install with: pip install "ftio[amd-libs]" or pip install -e ".[amd-libs]"'
            ) from None

        from ftio.freq._amd_workflow import ftio_amd

        prediction, analysis_figures = ftio_amd(
            args, bandwidth, time_b, total_bytes, ranks, text
        )

    else:
        raise Exception("Unsupported decomposition specified")

    return prediction, analysis_figures


def run():
    _ = main(sys.argv)


if __name__ == "__main__":
    _ = main(sys.argv)
