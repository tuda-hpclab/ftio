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

from ftio.modeling.alignment_matcher import AlignmentMatcher
from ftio.modeling.assignment_matcher import AssignmentMatcher
from ftio.modeling.phase_matcher import (
    DESCRIPTOR_KINDS,
    FEATURES,
    NEW_FAMILY,
    NOT_CLEAN,
    PhaseMatcher,
    parse_features,
)
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
    parser.add_argument(
        "-m",
        "--mode",
        default=None,
        help="I/O mode to analyze, as in FTIO (e.g. write_sync, read_sync); "
        "default: all read and write modes combined",
    )
    parser.add_argument(
        "--matcher",
        choices=("position", "alignment", "assignment"),
        default="position",
        help="match phases by index (all runs need the same phases), by alignment "
        "(same order in every run) or by assignment (any order, at most one phase "
        "per family and run)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        metavar="NAME[:WEIGHT]",
        default=None,
        help="descriptors the alignment or assignment matcher compares, each with "
        "an optional weight (default 1.0), e.g. --features read_fraction "
        f"bytes_per_rank:0.5. Known: {', '.join(DESCRIPTOR_KINDS)}. "
        "Default: " + " ".join(f"{name}:{w:g}" for name, (_, w) in FEATURES.items()),
    )
    parser.add_argument(
        "--recurrences",
        action="store_true",
        help="with --matcher assignment: a phase that repeats an earlier phase of "
        "its run (same rank count, similar period) joins that phase's family "
        "instead of being assigned on its own",
    )
    phase_model_args, ftio_args = parser.parse_known_args(
        args[1:]
    )  # the rest goes to FTIO

    print_info("Phase_model", False)

    if phase_model_args.recurrences and phase_model_args.matcher != "assignment":
        console.print(
            "[red]--- Error --- [/]\n[red]-> --recurrences needs --matcher assignment.[/]"
        )
        sys.exit(1)

    features = FEATURES
    if phase_model_args.features is not None:
        if phase_model_args.matcher == "position":
            console.print(
                "[red]--- Error --- [/]\n[red]-> --features needs --matcher "
                "alignment or assignment, the position matcher compares no descriptors.[/]"
            )
            sys.exit(1)
        try:
            features = parse_features(phase_model_args.features)
        except ValueError as e:
            console.print(f"[red]--- Error --- [/]\n[red]-> --features: {e}.[/]")
            sys.exit(1)
    if phase_model_args.mode is not None and phase_model_args.matcher != "position":
        single = [f for f in ("read_fraction", "relative_volume") if f in features]
        if single:  # with one I/O mode, they no longer tell phases apart
            console.print(
                f"[yellow]-m {phase_model_args.mode} analyzes one I/O mode only, "
                f"so {', '.join(single)} carry little information[/]"
            )
    not_clean = [f for f in features if f in NOT_CLEAN]
    if not_clean and phase_model_args.matcher != "position":
        reasons = "; ".join(f"{f} {NOT_CLEAN[f]}" for f in not_clean)
        console.print(
            f"[yellow]Not clean: {reasons}. Use this run to compare matching only, "
            "not to judge the fitted models[/]"
        )

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
            runs[trace] = phase_vectors(trace, ftio_args, mode=phase_model_args.mode)

    # matching by position only works if every run has the same phases
    counts = [(os.path.basename(trace), len(run)) for trace, run in runs.items()]
    if phase_model_args.matcher == "position" and len({n for _, n in counts}) != 1:
        console.print(
            f"[red]--- Error --- [/]\n[red]-> Different number of phases per run: {counts}.[/]"
        )
        sys.exit(1)

    matchers = {
        "position": PhaseMatcher,
        "alignment": AlignmentMatcher,
        "assignment": AssignmentMatcher,
    }
    if phase_model_args.matcher == "position":
        matcher = PhaseMatcher()
        print("matcher: position")
    elif phase_model_args.matcher == "assignment":
        matcher = AssignmentMatcher(
            features=features, recurrences=phase_model_args.recurrences
        )
        used = ", ".join(f"{name}={w:g}" for name, (_, w) in features.items())
        rec = ", recurrences" if phase_model_args.recurrences else ""
        print(f"matcher: assignment{rec}, features: {used}")
    else:
        matcher = matchers[phase_model_args.matcher](features=features)
        used = ", ".join(f"{name}={w:g}" for name, (_, w) in features.items())
        print(f"matcher: {phase_model_args.matcher}, features: {used}")
    for run in sorted(runs.values(), key=lambda run: run[0].ranks):
        for m in matcher.match_run(run):
            # a cost exists only where a matcher compared the phase with a family
            matched = type(matcher) is not PhaseMatcher and m.case != NEW_FAMILY
            cost = f", cost {m.cost:.3f}" if matched else ""
            repeat = (
                f", repeats phase {m.phase.same_as}"
                if m.phase.same_as is not None
                else ""
            )
            print(
                f"p={m.phase.ranks} phase {m.phase.phase_id} -> family_{m.family_id} "
                f"({m.case}{cost}{repeat})"
            )

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
