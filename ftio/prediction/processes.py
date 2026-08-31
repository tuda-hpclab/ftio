"""
Performs prediction with Pools (ProcessPoolExecutor) and a callback mechanism

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from argparse import Namespace

import ftio.prediction.monitor as pm
from ftio.multiprocessing.async_process import handle_in_process
from ftio.prediction.helper import export_extrap, print_data
from ftio.prediction.online_analysis import ftio_process
from ftio.prediction.probability_analysis import find_probability

# from ftio.prediction.async_process import handle_in_process


def predictor_with_processes(shared_resources, args):
    """Monitors a file and runs predictions whenever it changes.

    Two modes are supported, selected by the ``--debounce`` flag in *args*:

    * **Default (parallel)** — the original behaviour: a new prediction
      process is spawned on every file-change event regardless of whether a
      previous prediction is still running.  Suitable when predictions are
      fast relative to the I/O period.

    * **Debounce (serial, --debounce)** — only one prediction runs at a time.
      After a prediction finishes, the monitor stamp is re-checked; if the
      file changed again during the prediction the next iteration returns
      immediately and triggers a follow-up prediction — no event is permanently
      lost.  This also eliminates concurrent writes to the unprotected Value
      fields in SharedResources (count, hits, aggregated_bytes).

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to ftio
    """
    filename = args[1]
    debounce = "--debounce" in args

    procs = []
    # Init: capture the initial stamp
    stamp, _ = pm.monitor(filename, "")

    try:
        if debounce:
            while True:
                # Block until the file changes (returns immediately when the
                # stamp is already stale — the debounce re-trigger path).
                stamp, _ = pm.monitor(filename, stamp)
                proc = handle_in_process(
                    prediction_process, args=(shared_resources, args)
                )
                # Wait for the prediction to finish before accepting the next
                # trigger, keeping shared state access strictly serial.
                proc.join()
        else:
            while True:
                # Original behaviour: spawn a prediction on every trigger
                # without waiting for previous ones to finish.
                stamp, procs = pm.monitor(filename, stamp, procs)
                procs.append(
                    handle_in_process(prediction_process, args=(shared_resources, args))
                )
    except KeyboardInterrupt:
        print_data(shared_resources.data)
        export_extrap(shared_resources.data)
        _export_phase_automaton(shared_resources)
        print("-- done -- ")


def _export_phase_automaton(shared_resources) -> None:
    """Export the phase automaton to JSON and save to the library if configured."""
    aut = shared_resources.online_detection.get("pa_automaton", None)
    if aut is None:
        return
    path = shared_resources.online_detection.get("pa_export", "./phase_automaton.json")
    try:
        aut.save_json(path)
    except Exception as exc:
        print(f"[PhaseAutomaton] Could not export to {path}: {exc}")

    mgr = shared_resources.online_detection.get("pa_model_manager", None)
    if mgr is not None:
        try:
            mgr.save_run(aut)
        except Exception as exc:
            print(f"[ModelManager] Could not save to library: {exc}")


def prediction_process(shared_resources, args: Namespace, msgs=None) -> None:
    """Performs prediction made up of two parts: (1) Executes FTIO and (2) appends to data the value

    Args:
        shared_resources (SharedResources): shared resources among processes
        args (list[str]): additional arguments passed to ftio.py
        msg: zmq message
    """
    ftio_process(shared_resources, args, msgs)
    while not shared_resources.queue.empty():
        shared_resources.data.append(shared_resources.queue.get())

    _ = find_probability(shared_resources.data, counter=shared_resources.count.value)
    shared_resources.count.value += 1
