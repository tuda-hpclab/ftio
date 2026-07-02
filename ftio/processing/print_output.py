"""
Module for displaying prediction results from the ftio package.

This module provides functionality to process and display prediction results on the console.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
from rich.table import Table

from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction


def display_prediction(
    argv: str | Namespace = "ftio",
    prediction: Prediction | list[Prediction] = Prediction(),
) -> None:
    """
    Displays the result of the prediction from ftio.

    Args:
        argv (str | Namespace): Command-line arguments or parsed arguments.
        prediction (Prediction | list[Prediction]): The result from ftio.
    """

    # Handle multiple predictions
    if isinstance(prediction, list):
        for pred in prediction:
            display_prediction(argv, pred)
        return

    console = MyConsole()
    # Display results if a prediction is available
    if any(x in prediction.source.lower() for x in ["stft", "astft", "vmd", "efd"]):
        # show results in a table
        console.info(prediction.disp_dominant_freq_and_conf())
        console.info(prediction.display_frequencies_in_ranges())
    else:
        console.info(prediction.disp_dominant_freq_and_conf() + prediction.disp_ranges())

    # If -n is provided, print the top frequencies with their confidence and amplitude in a table
    if isinstance(argv, Namespace) and argv.n_freq > 0:
        console.info(f"[cyan underline]Top {int(argv.n_freq)} Frequencies:[/]")
        if prediction.top_freqs:
            freq_array = prediction.top_freqs["freq"]
            conf_array = np.round(
                np.where(
                    np.isinf(prediction.top_freqs["conf"]),
                    1,
                    prediction.top_freqs["conf"],
                )
                * 100,
                2,
            )
            if prediction.periodicity is not None and prediction.periodicity.size > 0:
                periodicity_array = np.round(
                    np.where(
                        np.isinf(prediction.top_freqs["periodicity"]),
                        1,
                        prediction.top_freqs["periodicity"],
                    )
                    * 100,
                    2,
                )
            else:
                periodicity_array = np.array([])
            amp_array = prediction.top_freqs["amp"]
            phi_array = prediction.top_freqs["phi"]

            table = Table(show_header=True, header_style="bold cyan")

            columns = [
                ("Freq (Hz)", "right"),
                ("Conf. (%)", "right"),
            ]

            if len(periodicity_array) > 0:
                columns.append(("Periodicity (%)", "right"))

            columns.extend(
                [
                    ("Amplitude", "right"),
                    ("Phi", "right"),
                    ("Cosine Wave", "right"),
                ]
            )

            for col_name, justify in columns:
                table.add_column(col_name, justify=justify, style="white", no_wrap=True)

            description = ""
            # Add frequency data
            for i, freq in enumerate(freq_array):
                wave_name = prediction.get_wave_name(freq, amp_array[i], phi_array[i])
                row = [
                    f"{freq:.3e}",
                    f"{conf_array[i]:.2f}",
                ]

                if len(periodicity_array) > 0:
                    row.append(f"{periodicity_array[i]:.2f}")

                row.extend(
                    [
                        f"{amp_array[i]:.3e}",
                        f"{phi_array[i]:.3e}",
                        wave_name,
                    ]
                )
                table.add_row(*row)
                if i == 0:
                    description = wave_name
                else:
                    description += " + " + wave_name

            console.info(table)
            console.info(
                f"Carterization up to {argv.n_freq} frequencies: \n{description}\n"
            )
        else:
            console.info("[red]No top frequency data found[/]")


def display_frequencies_in_ranges(
    argv: str | Namespace = "ftio", prediction: Prediction = Prediction()
) -> None:
    """
    Displays the result of the prediction from ftio per window.

    Args:
        argv (str | Namespace): Command-line arguments or parsed arguments.
        prediction (Prediction): The result from ftio.
    """
    console = MyConsole()
    # Use standard display for overall result
    console.info(prediction.disp_dominant_freq_and_conf())
    # show stft results
    console.info(prediction.display_frequencies_in_ranges())
