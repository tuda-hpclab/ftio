"""
Performs the analysis for prediction. This includes the calculation of ftio and parsing of the data into a queue

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import time
from argparse import Namespace

import numpy as np
from rich.console import Console

from ftio.cli import ftio_core
from ftio.freq.prediction import Prediction
from ftio.gui.socket_logger import get_socket_logger, log_to_gui_and_console
from ftio.plot.units import set_unit
from ftio.prediction.change_detection.adwin import (
    detect_pattern_change_adwin,
)
from ftio.prediction.change_detection.cusum import (
    detect_pattern_change_cusum,
)
from ftio.prediction.change_detection.helper import (
    change_post_processing,
    create_change_point_info,
)
from ftio.prediction.change_detection.pagehinkley import (
    detect_pattern_change_pagehinkley,
)
from ftio.prediction.helper import get_dominant
from ftio.prediction.shared_resources import SharedResources


def _automaton_step(
    prediction,
    args: Namespace,
    shared_resources: SharedResources,
) -> str:
    """Step the phase automaton with one new Prediction.

    Retrieves (or creates) the automaton from ``shared_resources``, feeds
    the prediction, and stores the updated automaton back.  Returns a
    Rich-formatted status string for the predictor log.

    The automaton survives across processes because it is stored in the
    multiprocessing Manager dict ``shared_resources.online_detection``.
    """
    from ftio.prediction.phase_automaton import PhaseAutomaton

    det = shared_resources.online_detection
    aut = det.get("pa_automaton", None)

    if aut is None:
        # Online, a Prediction's t_start/t_end span the whole run observed so
        # far, so the min_cycles warm-up guard applies: do not model a period
        # the window has not held twice. Without it the automaton learns the
        # very first prediction, which is a single I/O phase and therefore
        # reports that burst's width, not a period.
        aut = PhaseAutomaton.from_args(args)
        # Store config and export path once (for later retrieval at exit)
        det["pa_export"] = getattr(args, "pa_export", "./phase_automaton.json")

    transitioned = aut.step(prediction)

    # Write back — required because Manager proxies don't track in-place mutations
    det["pa_automaton"] = aut

    state = aut._current_state
    if state is None:
        return ""

    pred_count = shared_resources.count.value
    text = (
        f"[purple][PREDICTOR] (#{pred_count}):[/] "
        f"[cyan]Phase automaton:[/] "
        f"State {state.state_id} — "
        f"freq={state.dominant_freq:.4f} Hz, period={state.period:.2f} s, "
        f"ranks={state.ranks}, n_preds={state.n_phases}"
    )
    if transitioned:
        tr = aut.transitions[-1]
        text += (
            f"\n[purple][PREDICTOR] (#{pred_count}):[/] "
            f"[bold yellow]  → TRANSITION: "
            f"State {tr.from_state} → {tr.to_state}  "
            f"({tr.old_freq:.4f} → {tr.new_freq:.4f} Hz, cause={tr.cause!r})[/]"
        )
    text += (
        f"\n[purple][PREDICTOR] (#{pred_count}):[/] "
        f"[cyan]  Automaton summary: "
        f"{len(aut.states)} state(s), {len(aut.transitions)} transition(s)[/]"
    )
    return text


def _model_manager_step(
    prediction,
    args: Namespace,
    shared_resources: SharedResources,
) -> str:
    """Step the model manager with one new Prediction.

    Retrieves (or creates) the ModelManager from shared_resources, feeds the
    prediction, and writes the updated manager back.  Returns a Rich-formatted
    status string for the predictor log.

    Only the ModelManager object itself is stored in the Manager dict; its
    read-only reference automaton and compact tracker state live on the object.
    """
    from ftio.modeling.model_manager import ModelManager

    det = shared_resources.online_detection
    mgr: ModelManager | None = det.get("pa_model_manager", None)

    if mgr is None:
        library_dir = getattr(args, "pa_library", None) or "./ftio_models"
        app_name = getattr(args, "pa_app_name", None) or _derive_app_name(args)
        strategy = getattr(args, "pa_match", "greedy")
        mgr = ModelManager(library_dir, app_name, strategy)

    forecast = mgr.step(prediction)
    det["pa_model_manager"] = mgr  # write back — Manager proxies don't track mutations

    pred_count = shared_resources.count.value

    if forecast is None:
        return (
            f"[purple][PREDICTOR] (#{pred_count}):[/] "
            f"[dim]Reference ({mgr.app_name}): cold start — "
            f"no library match. Learning from this run.[/]"
        )

    pos_pct = forecast.position * 100
    state_str = f"{forecast.current_state_idx + 1}/{forecast.n_states}"
    text = (
        f"[purple][PREDICTOR] (#{pred_count}):[/] "
        f"[green]Reference ({mgr._strategy}): "
        f"state {state_str}, pos={pos_pct:.0f}%, "
        f"tracking={forecast.tracking_quality:.2f}[/]"
    )

    if forecast.at_end:
        text += (
            f"\n[purple][PREDICTOR] (#{pred_count}):[/] "
            f"[bold yellow]  → APPLICATION IN FINAL REFERENCE STATE[/]"
        )
    elif forecast.has_timing:
        text += (
            f"\n[purple][PREDICTOR] (#{pred_count}):[/] "
            f"[green]  Transition in ~{forecast.eta_seconds:.1f}s "
            f"[{forecast.eta_lower:.1f}s–{forecast.eta_upper:.1f}s] "
            f"→ next period ≈ {forecast.next_period:.2f}s[/]"
        )
    elif not np.isnan(forecast.next_period):
        text += (
            f"\n[purple][PREDICTOR] (#{pred_count}):[/] "
            f"[green]  Next period ≈ {forecast.next_period:.2f}s "
            f"(timing improves after ≥2 runs in library)[/]"
        )

    return text


def _derive_app_name(args: Namespace) -> str:
    """Derive an application name from the monitored filename."""
    import os

    files = getattr(args, "files", [])
    if files:
        return os.path.splitext(os.path.basename(files[0]))[0] or "unknown"
    return "unknown"


def ftio_process(shared_resources: SharedResources, args: Namespace, msgs=None) -> None:
    """Perform a single prediction with FTIO

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (Namespace): arguments passed to ftio
        msgs: ZMQ messages (optional)
    """
    console = Console()
    pred_id = shared_resources.count.value
    start_msg = f"[purple][PREDICTOR] (#{pred_id}):[/] Started"
    gui_enabled = "gui" in args
    log_to_gui_and_console(
        gui_enabled, console, start_msg, "predictor_start", {"count": pred_id}
    )

    # Modify the arguments
    args.extend(["-e", "no"])
    args.extend(["-ts", f"{shared_resources.start_time.value:.2f}"])
    # perform prediction
    prediction_list, parsed_args = ftio_core.main(args, msgs, shared_resources)
    if not prediction_list:
        log_to_gui_and_console(
            gui_enabled,
            console,
            "[yellow]Terminating prediction (no data passed)[/]",
            "termination",
            {"reason": "no_data"},
        )
        return

    # get latest prediction
    prediction = prediction_list[-1]
    # plot_bar_with_rich(shared_resources.t_app,shared_resources.b_app, width_percentage=0.9)

    # get dominant frequency
    freq = get_dominant(prediction)

    # save prediction results
    save_data(prediction, shared_resources)

    # --pa-library implies --phase-automaton
    if getattr(parsed_args, "pa_library", None):
        parsed_args.phase_automaton = True

    # Phase automaton: step if enabled (stored in shared_resources across processes)
    automaton_text = ""
    if getattr(parsed_args, "phase_automaton", False):
        automaton_text = _automaton_step(prediction, parsed_args, shared_resources)

    # Reference library: position tracking + transition forecast
    model_text = ""
    if getattr(parsed_args, "pa_library", None):
        model_text = _model_manager_step(prediction, parsed_args, shared_resources)

    # display results
    text = display_result(freq, prediction, shared_resources)
    if automaton_text:
        text += "\n" + automaton_text
    if model_text:
        text += "\n" + model_text

    # data analysis to decrease window by changing start_time
    adaptation_text, change_detected, change_point_info = window_adaptation(
        parsed_args, prediction, freq, shared_resources
    )
    text += adaptation_text
    candidates = [
        {"frequency": f, "confidence": c}
        for f, c in zip(prediction.dominant_freq, prediction.conf, strict=True)
    ]
    if candidates:
        best = max(candidates, key=lambda c: c["confidence"])
        dominant_freq = best["frequency"]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
        confidence = best["confidence"]
    else:
        dominant_freq = dominant_period = confidence = 0.0

    if gui_enabled:
        logger = get_socket_logger()
        if logger is not None:
            structured_prediction = {
                "prediction_id": pred_id,
                "timestamp": str(time.time()),
                "dominant_freq": dominant_freq,
                "dominant_period": dominant_period,
                "confidence": confidence,
                "candidates": candidates,
                "time_window": (float(prediction.t_start), float(prediction.t_end)),
                "total_bytes": str(prediction.total_bytes),
                "bytes_transferred": str(prediction.total_bytes),
                "current_hits": int(shared_resources.hits.value),
                "periodic_probability": 0.0,
                "frequency_range": (0.0, 0.0),
                "period_range": (0.0, 0.0),
                "change_detected": change_detected,
                "change_point": change_point_info,
            }
            logger.send_log(
                "prediction", "FTIO structured prediction", structured_prediction
            )

    # print text
    log_to_gui_and_console(
        gui_enabled,
        console,
        text,
        "prediction_log",
        {"count": pred_id, "freq": dominant_freq},
    )
    # shared_resources.count.value += 1


def window_adaptation(
    args: Namespace,
    prediction: Prediction,
    freq: float,
    shared_resources: SharedResources,
) -> tuple[str, bool, dict]:
    """Modifies the start time if conditions are true. Also performs change point detection.

    Args:
        args (argparse): command line arguments
        prediction (Prediction): result from FTIO
        freq (float|NaN): dominant frequency
        shared_resources (SharedResources): shared resources among processes

    Returns:
        tuple: (text, change_detected, change_point_info)
    """
    # Initialize output and prediction info
    text = ""
    t_s, t_e = prediction.t_start, prediction.t_end
    total_bytes = prediction.total_bytes
    prediction_count = shared_resources.count.value

    # Hits
    text += hits(args, prediction, shared_resources)

    # ---------------------- Intermediate info ----------------------
    if not np.isnan(freq) and freq > 0:
        time_window = t_e - t_s
        n_phases = avr_bytes = 0
        unit = "B"

        if time_window > 0:
            n_phases = time_window * freq
            if n_phases > 0:
                avr_bytes = int(total_bytes / n_phases)
                unit_factor, order = set_unit(avr_bytes, "B")
                avr_bytes = avr_bytes * order

        # FIXME: this needs to compensate for smaller windows
        if not args.window_adaptation:
            text += (
                f"[purple][PREDICTOR] (#{prediction_count}):[/] "
                f"Estimated phases {n_phases:.2f}\n"
                f"[purple][PREDICTOR] (#{prediction_count}):[/] "
                f"Average transferred {avr_bytes:.0f} {unit}\n"
            )

    # ---------------------- Change point detection ----------------------
    change_detected = False
    change_point_info = None
    if prediction_count == 0:
        shared_resources.online_detection["change_count"] = 0
        shared_resources.online_detection["last_change_time"] = t_s
        shared_resources.online_detection["state"] = {}

    if args.window_adaptation is not None:
        # Select and run the appropriate detector
        if args.window_adaptation in {"cusum", "ph", "adwin"}:
            if args.window_adaptation == "cusum":
                change_detected, log, detected_t_s, old_freq, new_freq = (
                    detect_pattern_change_cusum(
                        args, shared_resources, prediction, prediction_count
                    )
                )
            elif args.window_adaptation == "ph":
                change_detected, log, detected_t_s, old_freq, new_freq = (
                    detect_pattern_change_pagehinkley(
                        args, shared_resources, prediction, prediction_count
                    )
                )
            elif args.window_adaptation == "adwin":
                change_detected, log, detected_t_s, old_freq, new_freq = (
                    detect_pattern_change_adwin(
                        args, shared_resources, prediction, prediction_count
                    )
                )
            else:
                raise ValueError(
                    f"Unknown online adaptation algorithm: {args.window_adaptation}"
                )
            frequency_count = len(shared_resources.data)
            text += f"{log}\n" if log else ""
            # create info for logger
            change_point_info = create_change_point_info(
                prediction_count,
                t_e,
                old_freq,
                new_freq,
                detected_t_s,
                frequency_count,
            )
            t_s, text = change_post_processing(
                args, prediction_count, t_e, t_s, detected_t_s, text
            )

        # Frequency hits adaptation
        elif "frequency_hits" in args.window_adaptation:
            if shared_resources.hits.value > args.hits:
                tmp = max(t_e - 3 / freq, 0)
                t_s = tmp
                text += f"[bold purple][PREDICTOR] (#{prediction_count}):[/][green] Adjusting start time to {t_s} sec\n[/]"
                shared_resources.online_detection["change_count"] += 1
                shared_resources.online_detection["last_change_time"] = t_s
            else:
                t_s = 0
                if shared_resources.hits.value == 0:
                    text += f"[purple][PREDICTOR] (#{prediction_count}):[/][red bold] Resetting start time to {t_s} sec\n[/]"

        # Data-based adaptation
        elif "data" in args.window_adaptation:
            if len(shared_resources.data) > 0:
                test = (
                    np.floor(prediction_count / args.hits) if prediction_count != 0 else 0
                )
                text += f"[purple][PREDICTOR] (#{prediction_count}):[/][green] Trying time window adaptation: {test:.0f} =? {shared_resources.hits.value:.0f}\n[/]"
                if (
                    test == shared_resources.hits.value
                    and shared_resources.hits.value > 0
                ):
                    # t_s = shared_resources.data[-shared_resources.count.value]['t_start']
                    # text += f'[bold purple][PREDICTOR] (#{prediction_count}):[/][green] Adjusting start time to t_start {t_s} sec\n[/]'
                    index = int(prediction_count - 1)
                    shared_resources.hits.value = 0
                    if len(shared_resources.t_flush) > 0:
                        t_s = shared_resources.t_flush[index]
                        text += f"[bold purple][PREDICTOR] (#{prediction_count}):[/][green] Adjusting start time to t_flush[{index}] {t_s} sec\n[/]"
                    else:
                        t_s = shared_resources.data[index]["t_end"]
                        text += f"[bold purple][PREDICTOR] (#{prediction_count}):[/][green] Adjusting start time to t_end[{index}] {t_s} sec\n[/]"

                    shared_resources.online_detection["change_count"] += 1
                    shared_resources.online_detection["last_change_time"] = t_s

        else:
            raise ValueError(
                f"Unknown online adaptation algorithm: {args.window_adaptation}"
            )

        # ---------------------- Summary ----------------------
        if not np.isnan(freq):
            # Get all captured data
            all_data = list(shared_resources.data)

            # Count how many valid frequencies we have in history
            history_freqs = [get_dominant(d) for d in all_data]
            valid_history_count = len([f for f in history_freqs if not np.isnan(f)])

            # Total valid samples (history + current)
            samples = valid_history_count + 1

            # Total attempts (history length + current).
            # We use the max with prediction_count to handle race conditions gracefully.
            total_attempts = max(len(all_data) + 1, prediction_count + 1)

            changes = shared_resources.online_detection.get("change_count", 0)
            success_rate = (samples / total_attempts * 100) if total_attempts > 0 else 0

            text += (
                f"\n[purple][PREDICTOR] (#{prediction_count}): [bold cyan]{'=' * 50}[/]\n"
            )
            text += f"[purple][PREDICTOR] (#{prediction_count}): [bold cyan]{args.window_adaptation.upper()} ANALYSIS (Prediction #{prediction_count})[/]\n"
            text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]Frequency detections: {samples}/{total_attempts} ({success_rate:.1f}% coverage)[/]\n"
            text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]Pattern changes detected: {changes}[/]\n"
            text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]Current frequency: {freq:.3f} Hz ({1 / freq:.2f}s period)[/]\n"
            text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]Time window: [{t_s:.2f} -- {t_e:.2f}] s[/]\n"

            if samples > 1 and args.verbose:
                recent_freqs = [f for f in history_freqs if not np.isnan(f)][-4:] + [freq]
                text += f"[cyan][purple][PREDICTOR] (#{prediction_count}): Recent freq history: {[f'{f:.3f}Hz' for f in recent_freqs]}[/]\n"
                if len(recent_freqs) >= 2:
                    trend = (
                        "increasing"
                        if recent_freqs[-1] > recent_freqs[-2]
                        else (
                            "decreasing"
                            if recent_freqs[-1] < recent_freqs[-2]
                            else "stable"
                        )
                    )
                    text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]Frequency trend: {trend}[/]\n"

            text += f"[cyan][purple][PREDICTOR] (#{prediction_count}): {args.window_adaptation.upper()} window size: {samples} samples[/]\n"
            text += f"[purple][PREDICTOR] (#{prediction_count}): [cyan]{args.window_adaptation.upper()} changes detected: {changes}[/]\n"
            text += (
                f"[purple][PREDICTOR] (#{prediction_count}): [bold cyan]{'=' * 50}[/]\n\n"
            )

    # TODO 1: Make sanity check -- see if the same number of bytes was transferred
    # TODO 2: Train a model to validate the predictions?

    text += f"[purple][PREDICTOR] (#{prediction_count}):[/] Ended"

    # set change time
    shared_resources.start_time.value = t_s
    return text, change_detected, change_point_info


def save_data(prediction: Prediction, shared_resources) -> None:
    """Put all data from `prediction` in a `queue`. The total bytes are as well saved here.

    Args:
        prediction (dict): result from FTIO
        shared_resources (SharedResources): shared resources among processes
    """
    # safe total transferred bytes
    shared_resources.aggregated_bytes.value += prediction.total_bytes

    # save data
    shared_resources.queue.put(
        {
            "phase": shared_resources.count.value,
            "dominant_freq": prediction.dominant_freq,
            "conf": prediction.conf,
            "amp": prediction.amp,
            "phi": prediction.phi,
            "t_start": prediction.t_start,
            "t_end": prediction.t_end,
            "total_bytes": prediction.total_bytes,
            "ranks": prediction.ranks,
            "freq": prediction.freq,
            # 'hits': shared_resources.hits.value,
        }
    )


def display_result(
    freq: float, prediction: Prediction, shared_resources: SharedResources
) -> str:
    """Displays the results from FTIO

    Args:
        freq (float): dominant frequency
        prediction (Prediction): prediction setting from FTIO
        shared_resources (SharedResources): shared resources among processes

    Returns:
        str: text to print to console
    """
    # Dominant frequency
    if not np.isnan(freq):
        text = f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Dominant freq {freq:.3f} Hz ({1 / freq if freq != 0 else 0:.2f} sec)\n"
    else:
        text = f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] No dominant frequency found\n"

    # Candidates
    if len(prediction.dominant_freq) > 0:
        text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Freq candidates ({len(prediction.dominant_freq)} found): \n"
        for i, f_d in enumerate(prediction.dominant_freq):
            text += (
                f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/]    {i}) "
                f"{f_d:.2f} Hz -- conf {prediction.conf[i]:.2f}\n"
            )
    else:
        text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] No frequency candidates detected\n"

    # time window
    text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Time window {prediction.t_end - prediction.t_start:.3f} sec ([{prediction.t_start:.3f},{prediction.t_end:.3f}] sec)\n"

    # total bytes
    total_bytes = shared_resources.aggregated_bytes.value
    # total_bytes =  prediction.total_bytes
    unit, order = set_unit(total_bytes, "B")
    total_bytes = order * total_bytes
    text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Total bytes {total_bytes:.0f} {unit}\n"

    # Bytes since last time
    # tmp = abs(prediction.total_bytes -shared_resources.aggregated_bytes.value)
    tmp = abs(shared_resources.aggregated_bytes.value)
    unit, order = set_unit(tmp, "B")
    tmp = order * tmp
    text += (
        f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Bytes transferred since last "
        f"time {tmp:.0f} {unit}\n"
    )

    return text


def hits(args, prediction, shared_resources):
    text = ""
    if args.window_adaptation is not None:
        if "frequency_hits" in args.window_adaptation:
            if len(prediction.dominant_freq) == 1:
                shared_resources.hits.value += 1
                text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Current hits {shared_resources.hits.value}\n"
            else:
                shared_resources.hits.value = 0
                text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/][red bold] Resetting hits {shared_resources.hits.value}[/]\n"
        elif "data" in args.window_adaptation:
            shared_resources.hits.value = shared_resources.count.value // args.hits
            text += f"[purple][PREDICTOR] (#{shared_resources.count.value}):[/] Current hits {shared_resources.hits.value}\n"

    return text
