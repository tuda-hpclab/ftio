"""
Fit one Extra-P model per phase family from several runs of an application.

Author: Albani Alvarado Mauriello
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: October 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

from rich.console import Console

from ftio.modeling.phase_matcher import PhaseMatcher
from ftio.modeling.phase_vector import phase_vectors
from ftio.parse.ftio_to_extrap import build_experiment, fit_models, print_models
from ftio.parse.helper import print_info
from ftio.parse.record import Record

# same folder rules as Scales.load_setup, which cannot be used here as it merges
# all traces of a folder into one signal
TRACE_EXTENSIONS = (".json", ".jsonl", ".msgpack", ".darshan", ".txt")
SKIP_FILES = ("scale.jsonl", ".call.txt")
SKIP_FOLDERS = ("io_results", "exported_images")

MIN_FIT_POINTS = 5  # below this, Extra-P's fit is not reliable

# descriptor -> (Extra-P metric, unit for printing, value of a PhaseVector)
DESCRIPTORS = {
    "period": ("Time (s)", "s", lambda v: v.period),
    "frequency": ("Frequency (Hz)", "Hz", lambda v: v.frequency),
}

console = Console()


def find_traces(paths: list[str]) -> list[str]:
    """Expand folders into their trace files; every trace is one run."""
    traces = []
    for path in paths:
        if os.path.isdir(path) and not path.rstrip("/").endswith("_text"):
            for root, _, files in os.walk(path):
                if any(folder in root for folder in SKIP_FOLDERS):
                    continue
                traces += [
                    os.path.join(root, file)
                    for file in sorted(files)
                    if file.endswith(TRACE_EXTENSIONS) and file not in SKIP_FILES
                ]
        else:  # a file, or a Recorder folder
            traces.append(path)
    return traces


def main(args: list[str] = sys.argv) -> None:
    """Match the phases of several runs and model each family's period.

    Args:
        args (list[str]): arguments passed from command line
    """
    parser = argparse.ArgumentParser(
        prog="phase_model",
        description="Fit one Extra-P model per phase family from several runs.",
        epilog="All other options are passed to FTIO and must come after the "
        "files, e.g.: phase_model traces/ --pa-period-ratio 2.5",
    )
    parser.add_argument("files", nargs="+", help="traces or folders, one trace per run")
    parser.add_argument(
        "--allow-negative-exponents",
        "--allow_negative_exponents",
        dest="allow_negative_exponents",
        action="store_true",
        help="allow p^(-x) terms, needed for strong scaling",
    )
    parser.add_argument(
        "--predict", type=int, nargs="+", default=[], help="rank counts to predict"
    )
    phase_model_args, ftio_args = parser.parse_known_args(
        args[1:]
    )  # the rest goes to FTIO

    print_info("Phase_model", False)

    missing = [path for path in phase_model_args.files if not os.path.exists(path)]
    if missing:  # checked here, as FTIO's own message is hidden below
        console.print(f"[red]--- Error --- [/]\n[red]-> Could not open {missing}.[/]")
        sys.exit(1)

    traces = find_traces(phase_model_args.files)
    if not traces:
        console.print(
            f"[red]--- Error --- [/]\n[red]-> No traces found in {phase_model_args.files}.[/]"
        )
        sys.exit(1)

    runs = {}
    for trace in traces:
        with contextlib.redirect_stdout(io.StringIO()):  # hide FTIO's output
            # with_io set to false for baseline approach
            runs[trace] = phase_vectors(trace, ftio_args, with_io=False)

    # matching by position only works if every run has the same phases
    counts = [(os.path.basename(trace), len(run)) for trace, run in runs.items()]
    if len({n for _, n in counts}) != 1:
        console.print(
            f"[red]--- Error --- [/]\n[red]-> Different number of phases per run: {counts}.[/]"
        )
        sys.exit(1)

    matcher = PhaseMatcher()
    for run in sorted(runs.values(), key=lambda run: run[0].ranks):
        matcher.match_run(run)

    for fam in matcher.families.values():
        family = f"family_{fam.family_id}"
        records = [
            Record(v.ranks, f"{family}->{name}", metric, value(v))
            for v in fam.members
            for name, (metric, _, value) in DESCRIPTORS.items()
        ]

        if len(fam.ranks) < MIN_FIT_POINTS:
            console.print(
                f"[yellow]{family}: fitting on only {len(fam.ranks)} rank counts, "
                f"{MIN_FIT_POINTS} are recommended[/]"
            )

        models = fit_models(
            build_experiment(records), phase_model_args.allow_negative_exponents
        )
        print_models(models)
        for (callpath, _), model in models.items():
            name = callpath.name.split("->")[1]
            unit = DESCRIPTORS[name][1]
            for p in phase_model_args.predict:
                value = model.hypothesis.function.evaluate(p)
                print(f"  {name} at p={p}: {value:.4g} {unit}")


if __name__ == "__main__":
    main(sys.argv)
