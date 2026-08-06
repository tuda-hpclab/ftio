"""
Merge Predections

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import time
from argparse import Namespace

import numpy as np
from rich.panel import Panel

from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.helper import MyConsole
from ftio.freq.prediction import Prediction
from ftio.plot.plot_dft import plot_dft

CONSOLE = MyConsole()


def merge_predictions(
    args: Namespace,
    pred_dft: Prediction,
    pred_auto: Prediction,
    analysis_figures: AnalysisFigures,
) -> Prediction:
    """Merge prediction if both autocorrelation and freq. techniques (dft/wavelet) are executed

    Args:
        args (_type_): _description_
        pred_dft (Prediction): prediction frequency technique
        pred_auto (Prediction): prediction autocorrelation

    Returns:
        dict: merge prediction
    """
    CONSOLE.set(args.verbose)
    pred_merged = pred_dft
    if args.autocorrelation:
        tik = time.time()
        CONSOLE.print("[cyan]Merging Started:[/]\n")
        text = f"Merging Autocorrelation and {args.transformation.upper()}\n"
        if not pred_auto.is_empty() and not np.isnan(pred_auto.dominant_freq):
            pred_merged, text = merge_core(pred_dft, pred_auto, args.freq, text)
            if len(pred_merged.dominant_freq) > 0:
                text += f"Dominant frequency: [blue bold]{np.round(pred_merged.dominant_freq[-1],4)}[/] Hz -> [blue bold]{np.round(1/pred_merged.dominant_freq[-1],4)}[/] sec\n"
                text += f"Confidence: [bold]{pred_merged.conf[-1]}{np.round(pred_merged.conf[-1]*100,2)}[/] %\n"
            pred_auto.candidates = []
        else:
            text += "[yellow]Autrocorrelation prediction is empty[/]\n"
        CONSOLE.print(
            Panel.fit(
                text[:-1],
                style="white",
                border_style="blue",
                title="Merging Predictions",
                title_align="left",
            )
        )
        CONSOLE.print(f"\n[cyan]Merging finished:[/] {time.time() - tik:.3f} s")

        if any(x in args.engine for x in ["mat", "plot"]):
            analysis_figures.args = args
            plot_dft(args, pred_merged, analysis_figures)

    return pred_merged


def merge_core(
    pred_dft: Prediction, pred_auto: Prediction, freq: float, text: str
) -> tuple[Prediction, str]:
    """Merge the predictions

    Args:
        pred_dft (Prediction): prediction from dft/wavelet techniques
        pred_auto (Prediction): prediction from autocorreleation
        freq: sampling frequency
        text (str): display string

    Returns:
        tuple[Prediction,str]: merged prediction and display string
    """
    dominant_freq = -1
    conf = 0
    out_freq, out_conf = [], []
    method = "hits"
    # method = "cov"  # ratio or cov for method hits
    dominant_index_auto = pred_auto.get_dominant_index()
    if len(pred_dft.dominant_freq) >= 1:
        if "alike" in method:
            alike = (
                pred_auto.dominant_freq[dominant_index_auto]
                - abs(
                    pred_dft.dominant_freq - pred_auto.dominant_freq[dominant_index_auto]
                )
            ) / pred_auto.dominant_freq[dominant_index_auto]
            text += f"Frequencies [yellow]{np.round(pred_dft.dominant_freq,4)}[/] Hz match [yellow]{np.round(pred_auto.dominant_freq,4)}[/] Hz by:\n[yellow]{np.round(alike,4)}[/]%\n\n"
            dominant_index = np.argmax(alike)
            dominant_freq = pred_dft.dominant_freq[dominant_index]
            conf = (
                pred_dft.conf[dominant_index] + pred_auto.conf[dominant_index_auto]
            ) / 2
        elif "hits" in method:
            tol = 2 / freq  # 2 frequency steps
            hits = np.zeros(len(pred_dft.dominant_freq))
            if "ratio" in method:
                alike = np.zeros((len(pred_dft.dominant_freq), len(pred_auto.candidates)))
            else:  # cov method
                alike = np.zeros(len(pred_dft.dominant_freq))

            # Find perfect hits and caclualte distance of the candiidates to the freq predictions
            for i, f_d in enumerate(pred_dft.dominant_freq):
                t_d = 1 / f_d
                hits[i] = sum(
                    (t_d - tol < pred_auto.candidates)
                    & (pred_auto.candidates < t_d + tol)
                )
                if "ratio" in method:
                    alike[i, :] = np.min(
                        [
                            pred_auto.candidates / t_d,
                            t_d / pred_auto.candidates,
                        ],
                        axis=0,
                    )  # 1 - abs(pred_auto.candidates - t_d)/t_d
                else:  # cov method
                    alike[i] = 1 - np.abs(
                        np.std(np.append(pred_auto.candidates, t_d))
                        / np.mean(np.append(pred_auto.candidates, t_d))
                    )

            # # check if there is a perfect prediction
            # text += f"Perfect hits of [blue]{pred_dft.dominant_freq}[/] are [blue]{hits}[/]  \n"
            # for i, value in enumerate(hits):
            #     if value == len(pred_auto.candidates):
            #         text += "[green bold]Perfect Prediction found! [/]\n"
            #         dominant_freq = pred_dft.dominant_freq[i]
            #         conf = 1
            #         break

            # No perfect hit, see which better fits
            if "ratio" in method:
                dominant_index = np.argmax(np.sum(alike, axis=1))
            else:  # cov method
                dominant_index = np.argmax(alike)

            dominant_freq = pred_dft.dominant_freq[dominant_index]
            # calculate conf:
            agreed_predictions = np.argmax(alike, axis=0)
            if "ratio" in method:
                text += f"Agreed prediction ratio  [blue]{np.sum(alike, axis=1)/len(agreed_predictions)*100}[/] %\n"
                conf = (
                    1
                    - np.std(
                        [
                            pred_dft.conf[dominant_index],
                            pred_auto.conf[dominant_index_auto],
                        ]
                    )
                    / np.mean(
                        [
                            pred_dft.conf[dominant_index],
                            pred_auto.conf[dominant_index_auto],
                        ]
                    )
                    + pred_dft.conf[dominant_index]
                    + pred_auto.conf[dominant_index_auto]
                ) / 3
            else:
                text += (
                    "Agreed prediction [blue]"
                    + ", ".join(f"{i * 100:.2f}" for i in alike)
                    + " %[/]\n"
                )
                conf = (
                    alike[dominant_index]
                    + pred_dft.conf[dominant_index]
                    + pred_auto.conf[dominant_index_auto]
                ) / 3
                # text += f"[red bold]{alike[dominant_index]} + {pred_dft.conf[dominant_index]} + {pred_auto.conf}[/]\n"
            if conf >= 0.2:
                out_freq, out_conf = [dominant_freq], [conf]
                # amp/phi/periodicity are parallel arrays to pred_dft.dominant_freq
                # (see _dft_workflow.py: all four are indexed by the same
                # dominant_index). dominant_freq/conf get narrowed to the single
                # merge winner above -- narrow these the same way, to the same
                # index, or get_dominant_index() later argmaxes over the stale,
                # longer amp array and indexes the now-shorter dominant_freq with
                # an out-of-range position.
                out_amp = (
                    [pred_dft.amp[dominant_index]]
                    if len(pred_dft.amp) > dominant_index
                    else []
                )
                out_phi = (
                    [pred_dft.phi[dominant_index]]
                    if len(pred_dft.phi) > dominant_index
                    else []
                )
                out_periodicity = (
                    [pred_dft.periodicity[dominant_index]]
                    if len(pred_dft.periodicity) > dominant_index
                    else []
                )
            else:
                out_freq, out_conf = [], []
                out_amp, out_phi, out_periodicity = [], [], []
                text += "[red bold]Too low confidence, no dominant freq[/]\n"

    else:  # no dft result
        dominant_freq = pred_auto.dominant_freq
        conf = pred_auto.conf / 3
        text += "Confidence: [red] Warning! Low confidence! [/]\n"
        out_freq, out_conf = dominant_freq, conf
        out_amp, out_phi, out_periodicity = (
            pred_dft.amp,
            pred_dft.phi,
            pred_dft.periodicity,
        )

    pred_merged = pred_dft
    pred_merged.dominant_freq = out_freq
    pred_merged.conf = out_conf
    pred_merged.amp = out_amp
    pred_merged.phi = out_phi
    pred_merged.periodicity = out_periodicity
    return pred_merged, text
