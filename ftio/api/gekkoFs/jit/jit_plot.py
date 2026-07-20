"""
This file provides functionality to plot results from JSON files
containing experimental data for JIT, JIT no FTIO, and Pure modes. It includes
functions to extract data, process it, and generate visualizations.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Dec 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse
import glob
import json
import os

from rich.console import Console
from rich.table import Table

from ftio.api.gekkoFs.jit.jit_result import JitResult

CONSOLE = Console()


def _mode_label(mode: str) -> str:
    """Map the JSON mode letters to a run label (matches add_dict's indexing)."""
    if "F" in mode:
        return "glass"
    if "D" in mode:
        return "gekko"
    return "pfs"


def totals_by_node(json_path: str) -> dict:
    """Read a result.json into {compute_nodes: {label: {app,stage_in,stage_out,total}}}.

    total = app + stage_in + stage_out. For each (nodes, mode) the latest entry
    by timestamp wins, so reruns show the most recent number.
    """
    with open(json_path) as f:
        data = json.load(f)
    rows: dict = {}
    for grp in data:
        # jit reserves one node for FTIO, so compute nodes = nodes - 1
        n = int(grp["nodes"]) - 1
        latest: dict = {}
        for e in grp.get("data", []):
            label = _mode_label(e.get("mode", ""))
            ts = e.get("timestamp", "")
            if label not in latest or ts >= latest[label][0]:
                latest[label] = (ts, e)
        rows[n] = {
            label: {
                "app": e["app"],
                "stage_in": e["stage_in"],
                "stage_out": e["stage_out"],
                "total": e["app"] + e["stage_in"] + e["stage_out"],
            }
            for label, (ts, e) in latest.items()
        }
    return rows


def started_runs(app_dir: str) -> set:
    """(compute_nodes, mode) pairs that have a log dir under app_dir.

    A run whose dir exists but which has no result.json entry started but did
    not finish (e.g. the stage-out hung) -- the table flags those as FAIL.
    """
    out = set()
    for d in glob.glob(os.path.join(app_dir, "nodes_*", "rep_*", "*")):
        mode = os.path.basename(d)
        if mode not in ("glass", "gekko", "pfs") or not os.path.isdir(d):
            continue
        try:
            n = int(os.path.basename(os.path.dirname(os.path.dirname(d))).split("_")[1])
        except (IndexError, ValueError):
            continue
        out.add((n - 1, mode))  # compute nodes = nodes - 1 (FTIO node)
    return out


def print_totals_table(json_path: str) -> None:
    """Print a per-node app/total table (glass | gekko | pfs) to the console.

    Cells: numbers when the run finished, red FAIL when it started but never
    recorded a result (stage-out hang / crash), "-" when never attempted.

    A missing result.json means no mode ever recorded a result -- every attempted
    run is then a FAIL. Read it only if it exists so the table still prints (from
    the on-disk log dirs) instead of crashing with FileNotFoundError. A half-written
    file (a run is appending to it right now) is treated the same way.
    """
    app_dir = os.path.dirname(os.path.abspath(json_path))
    rows = {}
    if os.path.exists(json_path):
        try:
            rows = totals_by_node(json_path)
        except json.JSONDecodeError:
            CONSOLE.print(
                f"[yellow]{json_path} is being written right now; showing the "
                f"on-disk runs only.[/yellow]"
            )
    started = started_runs(app_dir)
    if not rows and not started:
        CONSOLE.print(f"[yellow]No results found under {app_dir}[/yellow]")
        return
    app = os.path.basename(app_dir)
    table = Table(
        title=f"{app}  —  time in s (total = app+in+out; "
        f"green = smallest, yellow = largest per row)"
    )
    table.add_column("nodes", justify="right")
    for label in ("glass", "gekko"):
        table.add_column(f"{label} app", justify="right")
        table.add_column(f"{label} in", justify="right")
        table.add_column(f"{label} out", justify="right")
        table.add_column(f"{label} total", justify="right", style="bold")
    table.add_column("pfs app", justify="right")
    table.add_column("pfs total", justify="right", style="bold")

    for n in sorted(set(rows) | {node for node, _ in started}):
        row = rows.get(n, {})
        # smallest/largest app and smallest/largest total across modes for this
        # row get highlighted, kept as separate groups since total >= app always
        # and would otherwise dominate.
        app_vals = {label: row[label]["app"] for label in row}
        total_vals = {label: row[label]["total"] for label in row}
        smallest_app = min(app_vals, key=app_vals.get) if app_vals else None
        biggest_app = max(app_vals, key=app_vals.get) if app_vals else None
        smallest_total = min(total_vals, key=total_vals.get) if total_vals else None
        biggest_total = max(total_vals, key=total_vals.get) if total_vals else None

        def _cell(
            row: dict, label: str, key: str, smallest: str | None, biggest: str | None
        ) -> str:
            value = f"{row[label][key]:.1f}"
            if smallest != biggest and label == smallest:
                return f"[green]{value}[/green]"
            if smallest != biggest and label == biggest:
                return f"[yellow]{value}[/yellow]"
            return value

        cells = [str(n)]
        for label in ("glass", "gekko"):
            if label in row:
                cells.append(_cell(row, label, "app", smallest_app, biggest_app))
                cells.append(f"{row[label]['stage_in']:.1f}")
                cells.append(f"{row[label]['stage_out']:.1f}")
                cells.append(_cell(row, label, "total", smallest_total, biggest_total))
            elif (n, label) in started:
                cells.extend(["[red]FAIL[/red]"] * 4)
            else:
                cells.extend(["-"] * 4)
        if "pfs" in row:
            cells.append(_cell(row, "pfs", "app", smallest_app, biggest_app))
            cells.append(_cell(row, "pfs", "total", smallest_total, biggest_total))
        elif (n, "pfs") in started:
            cells.extend(["[red]FAIL[/red]"] * 2)
        else:
            cells.extend(["-"] * 2)

        # GLASS goal for the row: glass total < gekko total < pfs total. Flag it
        # on the "nodes" cell only -- coloring the whole row would swamp the
        # per-cell smallest/largest highlighting done above.
        goal_hit = {"glass", "gekko", "pfs"} <= set(total_vals) and total_vals[
            "glass"
        ] < total_vals["gekko"] < total_vals["pfs"]
        if goal_hit:
            cells[0] = f"[bold green]{cells[0]}[/bold green]"
        table.add_row(*cells)
    CONSOLE.print(table)


def resolve_result_json(arg: str, cwd: str | None = None) -> str:
    """Turn a jit_plot argument into a path to a result.json.

    Accepts, in order: a result.json file, a directory holding one, or a bare
    app name (looked up under ./<app> then ./logs/<app>, the per-app layout jit
    now writes). "" or "." means the current directory -- so `cd logs/lammps &&
    jit_plot` just works.

    Args:
        arg: The user-supplied argument.
        cwd: Base directory, injectable for testing (defaults to os.getcwd()).

    Returns:
        The resolved path to a result.json (may not exist, so the caller's open
        error names a sensible path).
    """
    cwd = cwd or os.getcwd()
    if not arg or arg == ".":
        return os.path.join(cwd, "result.json")
    p = arg if os.path.isabs(arg) else os.path.join(cwd, arg)
    if os.path.isdir(p):
        return os.path.join(p, "result.json")
    if p.endswith(".json"):
        return p
    # bare app name: ./<app>/result.json (logs/<app> kept as a fallback for old runs)
    for cand in (
        os.path.join(cwd, arg, "result.json"),
        os.path.join(cwd, "logs", arg, "result.json"),
    ):
        if os.path.exists(cand):
            return cand
    return os.path.join(cwd, arg, "result.json")


def app_result_jsons(job_dir: str) -> list[str]:
    """Return every <job_dir>/<app>/result.json, sorted by app name.

    An app whose every mode failed has log dirs but no result.json; its path is
    returned anyway so the table shows it as FAIL rather than omitting it.

    Args:
        job_dir: A jit job folder (e.g. ~/jit/<jobid>).

    Returns:
        The result.json path of every app in the job (existing or not). Empty if
        this is not a job folder.
    """
    if not os.path.isdir(job_dir):
        return []
    found = []
    for entry in sorted(os.listdir(job_dir)):
        app_dir = os.path.join(job_dir, entry)
        if not os.path.isdir(app_dir):
            continue
        if os.path.exists(os.path.join(app_dir, "result.json")) or started_runs(app_dir):
            found.append(os.path.join(app_dir, "result.json"))
    return found


def resolve_result_jsons(arg: str, cwd: str | None = None) -> list[str]:
    """Resolve an argument into every result.json it names.

    An app name or per-app folder names one; a job folder expands to one per app.

    Args:
        arg: The user-supplied argument.
        cwd: Base directory, injectable for testing (defaults to os.getcwd()).

    Returns:
        One or more result.json paths. Falls back to the single resolved path
        (which may not exist) so the caller still reports a sensible name.
    """
    single = resolve_result_json(arg, cwd)
    if os.path.exists(single):
        return [single]
    found = app_result_jsons(os.path.dirname(single))
    return found or [single]


def plot_results(args):
    """
    Plot results from the given JSON files. If no filenames are provided,
    default data is used.

    Args:
        filenames (list): List of JSON file paths.
    """
    # No argument: use ./result.json if we are sitting inside a per-app folder.
    if args and not args.filenames:
        here = os.path.join(os.getcwd(), "result.json")
        if os.path.exists(here):
            if getattr(args, "table", False):
                print_totals_table(here)
            else:
                extract_and_plot(JitResult(), here, here, no_diff=args.no_diff)
            return
        # Standing in a job folder instead: table every app it holds.
        if getattr(args, "table", False):
            found = app_result_jsons(os.getcwd())
            if found:
                for path in found:
                    print_totals_table(path)
                return

    if not args:
        results = JitResult()
        # # run with x nodes  128 procs [jit | jit_no_ftio | pure] (now in old folder)
        # ################ Edit area ############################
        # title = "Nek5000 with 128 procs checkpointing every 10 steps with a total of 100 steps"
        # tmp_app = [207.3, 181.44, 181.56]
        # tmp_stage_out = [3.95,  17.68, 0]
        # tmp_stage_in =  [0.72,  0.75, 0]
        # results.add_experiment(tmp_app,tmp_stage_out,tmp_stage_in,"# 2")

        # # run with 3 nodes
        # tmp_app = [90.11, 84.81, 103]
        # tmp_stage_out = [2.26, 61.0, 0]
        # tmp_stage_in = [1.11, 1.11, 0]
        # results.add_experiment(tmp_app,tmp_stage_out,tmp_stage_in,"# 3")

        # tmp_app = [70.53, 72.71, 80.21]
        # tmp_stage_out = [1.891,9.430, 0]
        # tmp_stage_in = [1.149, 1.145, 0]
        # results.add_experiment(tmp_app,tmp_stage_out,tmp_stage_in,"# 4")

        # run with x nodes 32 procs [jit | jit_no_ftio | pure] (now in old folder)
        ################ Edit area ############################
        # # run with 3 nodes
        # tmp_app       = [ 157.8 , 156.51 , 160 ]
        # tmp_stage_out = [ 1.041 , 1.04 ,  0]
        # tmp_stage_in  = [ 1.099 ,  1.09,  0]
        # add_experiment(data,tmp_app,tmp_stage_out,tmp_stage_in,"# 3")

        # tmp_app       = [ 114.72,122.07  ,  130.95]
        # tmp_stage_out = [ 1.04, 1.03 ,  0]
        # tmp_stage_in  = [ 1.11, 1.83  ,  0]
        # add_experiment(data,tmp_app,tmp_stage_out,tmp_stage_in,"# 4")

        # tmp_app       = [ 97.11,106.31, 98.11]
        # tmp_stage_out = [ 1.05,1.12,0]
        # tmp_stage_in  = [ 1.13,1.12,0]
        # add_experiment(data,tmp_app,tmp_stage_out,tmp_stage_in,"# 5")

        # tmp_app       = [ 126.93,119.06, 93.27]
        # tmp_stage_out = [ 1.12,1.159,0]
        # tmp_stage_in  = [ 1.59,1.96,0]
        # add_experiment(data,tmp_app,tmp_stage_out,tmp_stage_in,"# 10")

        # tmp_app       = [ 182.58,174.67,90.64 ]
        # tmp_stage_out = [ 1.08,1.08,0]
        # tmp_stage_in  = [ 3.57,2.84,0]
        # add_experiment(data,tmp_app,tmp_stage_out,tmp_stage_in,"# 20")

        # title = "Nek5000 with 16 procs checkpointing every 10 steps with a total of 50 steps"
        # filename = "results_mogon/procs16_steps50_writeinterval10.json "

        title = (
            "Nek5000 with 16 procs checkpointing every 5 steps with a total of 50 steps"
        )
        filename = "results_mogon/wacom++_app_proc_1_OMPthreads_64_12500000.json"
        current_directory = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_directory, filename)

        extract_and_plot(results, json_file_path, title)
    else:
        for filename in args.filenames:
            if getattr(args, "table", False):
                # A job folder tables every app in it; an app names just itself.
                for json_file_path in resolve_result_jsons(filename):
                    print_totals_table(json_file_path)
                continue
            # Resolve an app name / folder / result.json into the actual file.
            json_file_path = resolve_result_json(filename)
            print(f"Processing file: {json_file_path}")
            extract_and_plot(
                JitResult(), json_file_path, json_file_path, no_diff=args.no_diff
            )


def extract_and_plot(
    results: JitResult, json_file_path: str, title: str, no_diff: bool = True
):
    """
    Extract data from a JSON file and plot the results.

    Args:
        results (JitResult): The JitResult object to store the extracted data.
        json_file_path (str): Path to the JSON file.
        title (str): Title for the plot.
        all (bool): Flag to control whether to call add_dict or add_all (default is False).
    """
    # result.json only appears once a run finishes (it is appended after
    # stage-out), so a job that is still in flight legitimately has none -- and a
    # job writing it right now can be caught mid-append. Neither should end in a
    # traceback.
    if not os.path.exists(json_file_path):
        CONSOLE.print(
            f"[yellow]No result.json at {json_file_path} -- nothing to plot yet "
            f"(is the run still in progress?). Use -t for the per-run table.[/yellow]"
        )
        return

    # Open the file and load the JSON data
    try:
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError as e:
        CONSOLE.print(
            f"[yellow]{json_file_path} is not valid JSON yet ({e}); a run is "
            f"probably still writing it. Try again once it finishes.[/yellow]"
        )
        return

    # Sort the data by 'nodes'
    data = sorted(data, key=lambda x: x["nodes"])

    # Drop the plot next to the result.json (the per-app logs/<app> folder).
    save_dir = os.path.dirname(os.path.abspath(json_file_path))

    # Depending on the 'all' flag, process the data differently
    if no_diff:
        for d in data:
            results.add_dict(d)
        results.plot(title, save_dir=save_dir)
    else:
        for d in data:
            results.add_all(d)
        results.plot_all(title, save_dir=save_dir)


def main():
    """
    Main function to parse command-line arguments and plot results.
    """
    parser = argparse.ArgumentParser(description="Load JSON data from files and plot.")
    parser.add_argument(
        "filenames",
        type=str,
        nargs="*",  # '*' allows zero or more filenames
        default=[],
        help="The paths to the JSON file(s) to plot.",
    )
    # Boolean argument to determine whether to use the diff data
    parser.add_argument(
        "-n",
        "--no_diff",
        action="store_true",  # This stores True if the argument is provided, False otherwise
        help="Use the latest data based on the timestamp. Otherwise all data are plotted with error bars",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--table",
        action="store_true",
        help="Print a per-node app/total table to the console; no browser plot.",
        default=False,
    )
    args = parser.parse_args()
    plot_results(args)


if __name__ == "__main__":
    main()
