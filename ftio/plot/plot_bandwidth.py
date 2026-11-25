"""
Function to plot the bandwidth from the JIT

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jan 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel

from ftio.plot.helper import format_plot_and_ticks
from ftio.plot.units import set_unit


def load_json_and_plot(filenames):
    """Loads a JSON file and plots it

    Args:
        filenames (str): absolute path with filename
    """
    # Get the current directory
    current_directory = os.getcwd()

    # Create a list to hold traces
    traces = []
    unit = "MB"
    order = 1e-6

    for filename in filenames:
        # Define the path to the JSON file
        json_file_path = os.path.join(current_directory, filename)

        # Load the JSON file, recovering partial data if the file was truncated
        with open(json_file_path) as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] {filename} is truncated at offset {exc.pos} "
                    "(FTIO killed mid-write?), recovering partial data..."
                )
                json_file.seek(0)
                raw = json_file.read()[: exc.pos]
                raw = raw.rstrip().rstrip(",")
                open_brackets = raw.count("[") - raw.count("]")
                raw += "]" * open_brackets + "}"
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[error] Could not recover {filename}, skipping.")
                    continue

        # Extract arrays from the JSON data
        b = np.array(data.get("b", []))
        t = np.array(data.get("t", []))

        # If both b and t are empty, proceed to extract from sync types
        if b.size == 0 and t.size == 0:
            # Search for the 'bandwidth' key in all sync types (write_sync, read_sync, etc.)
            for sync_type, sync_data in data.items():
                if "bandwidth" in sync_data:
                    print(f"found type:{sync_type}")
                    b = np.array(sync_data["bandwidth"].get("b_overlap_avr", []))
                    t = np.array(sync_data["bandwidth"].get("t_overlap", []))
                    break

        if filenames.index(filename) == 0:
            unit, order = set_unit(b)

        if t.size == 0:
            print(f"[warn] {filename}: time array is empty, skipping terminal plot.")
        plot_bar_with_rich(t, b)

        # Create a scatter plot trace of b against t
        trace = go.Scatter(
            x=t,
            y=b * order,
            mode="lines+markers",
            line={"shape": "hv"},
            fill="tozeroy",
            name=filename,  # Set the legend name to the filename
        )

        # Add the trace to the list
        traces.append(trace)

    # Create a figure and add all traces
    fig = go.Figure(data=traces)

    # Update layout for better visualization
    fig.update_layout(
        title="Stage-out Data",
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        showlegend=True,
    )
    fig = format_plot_and_ticks(fig, font_size=27, n_ticks=10)

    # Show the plot
    pio.show(fig)


def parse_flush_log(path: str) -> tuple[list[dict], list[dict]]:
    """Parse a flush.log file produced by the JIT staging layer.

    Returns:
        entries: dicts with ts, start, label, src, dst, copy_time, delete_time.
            A line is written once copy AND delete are done, so ``ts`` is the END
            of the flush; ``start`` = ts - copy_time - delete_time.
        events:  dicts with ts, event (APP-START / APP-END / FTIO-SHUTDOWN strings)
    """
    entries: list[dict] = []
    events: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" | ", maxsplit=4)
                try:
                    ts = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                second = parts[1].strip() if len(parts) > 1 else ""
                if second.startswith(("APP-", "FTIO-SHUTDOWN")):
                    events.append({"ts": ts, "event": second})
                elif len(parts) >= 5:
                    src_dst = parts[2].strip()
                    src = src_dst.split(" -> ")[0].strip()
                    dst = src_dst.split(" -> ")[1].strip() if " -> " in src_dst else ""
                    copy_time = float(
                        parts[3].replace("copy:", "").replace("s", "").strip()
                    )
                    delete_time = float(
                        parts[4].replace("delete:", "").replace("s", "").strip()
                    )
                    entries.append(
                        {
                            "ts": ts,
                            "start": ts - timedelta(seconds=copy_time + delete_time),
                            "label": second,
                            "src": src,
                            "dst": dst,
                            "copy_time": copy_time,
                            "delete_time": delete_time,
                        }
                    )
    except FileNotFoundError:
        print(f"flush log not found: {path}")
        print("Hint: flush logs are written into the job log directory, e.g.")
        print("  logs_nodes5_Jobid12345_DF/flush.log")
    except OSError as e:
        print(f"Could not read {path}: {e}")
    return entries, events


def plot_flush_log(flush_log_path: str, show: bool = True) -> go.Figure:
    """Gantt-style timeline of flush log entries.

    Each flushed file is shown as two stacked horizontal segments: copy
    time (solid) and delete time (semi-transparent), coloured by trigger
    source (FTIO-trigger = blue, post-app = orange).  APP-START / APP-END
    appear as vertical dashed lines.

    Args:
        flush_log_path: path to the flush.log file written by the JIT layer.
        show: when True the figure is displayed in the browser.

    Returns:
        The Plotly Figure object (useful for embedding or saving).
    """
    entries, events = parse_flush_log(flush_log_path)
    if not entries and not events:
        if os.path.isfile(flush_log_path):
            print(f"No parseable flush data in {flush_log_path}")
        return go.Figure()

    app_start_events = [ev for ev in events if "START" in ev["event"]]
    if app_start_events:
        t0 = app_start_events[0]["ts"]
    else:
        all_ts = [e["start"] for e in entries] + [ev["ts"] for ev in events]
        t0 = min(all_ts)

    def to_sec(ts: datetime) -> float:
        return (ts - t0).total_seconds()

    fig = go.Figure()

    # Two segment colours per group: copy (solid), delete (lighter)
    group_cfg = [
        ("FTIO-trigger", "#1f77b4", "rgba(31,119,180,0.35)"),
        ("post-app", "#e07b00", "rgba(224,123,0,0.35)"),
    ]
    first_in_legend: set[str] = set()

    for label, copy_color, del_color in group_cfg:
        group = [e for e in entries if label in e["label"]]
        if not group:
            continue
        names = [os.path.basename(e["src"]) for e in group]

        # copy segment
        fig.add_trace(
            go.Bar(
                x=[e["copy_time"] for e in group],
                y=names,
                orientation="h",
                base=[to_sec(e["start"]) for e in group],
                name=label,
                legendgroup=label,
                showlegend=label not in first_in_legend,
                marker_color=copy_color,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Trigger: {label}<br>"
                    "Copy: %{x:.3f} s<extra></extra>"
                ),
            )
        )
        first_in_legend.add(label)

        # delete segment (stacked on top of copy)
        fig.add_trace(
            go.Bar(
                x=[e["delete_time"] for e in group],
                y=names,
                orientation="h",
                base=[to_sec(e["start"]) + e["copy_time"] for e in group],
                name=f"{label} (delete)",
                legendgroup=label,
                showlegend=False,
                marker_color=del_color,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Trigger: {label}<br>"
                    "Delete: %{x:.3f} s<extra></extra>"
                ),
            )
        )

    for ev in events:
        t = to_sec(ev["ts"])
        is_start = "START" in ev["event"]
        fig.add_vline(
            x=t,
            line_dash="dash",
            line_width=1.5,
            line_color="green" if is_start else "red",
            annotation_text=ev["event"],
            annotation_font_size=12,
            annotation_position="top right" if is_start else "top left",
        )

    # Height: ~35 px per unique file row, min 200 px
    n_rows = max(len({os.path.basename(e["src"]) for e in entries}), 1)
    height = max(200, 35 * n_rows + 120)  # +120 for title + legend + x-axis

    fig.update_layout(
        title={
            "text": f"Flush Timeline — {os.path.basename(flush_log_path)}",
            "font": {"size": 16},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Time from APP-START (s)",
        yaxis_title=None,
        barmode="overlay",
        bargap=0.5,
        height=height,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
        },
        margin={"l": 10, "r": 20, "t": 60, "b": 40},
    )
    fig = format_plot_and_ticks(fig, font_size=13, n_ticks=5, y_minor=False)

    if show:
        pio.show(fig)
    return fig


def non_zero_mean(arr: np.ndarray):
    return np.mean(arr[np.nonzero(arr)]) if len(arr[np.nonzero(arr)]) > 0 else 0


def plot_bar_with_rich(
    x,
    y,
    max_height=10,
    terminal_width=None,
    width_percentage=0.95,
    func=non_zero_mean,
    flush_log=None,
):
    """
    Plots a bar chart using Rich library with dynamic width and properly scaled axis labels.

    Args:
        x (list or np.array): X-axis values.
        y (list or np.array): Y-axis values.
        max_height (int): Maximum height of the plot in characters.
        terminal_width (int, optional): Width of the terminal. If None, it will be auto-detected.
        width_percentage (float): Percentage of terminal width to use for the plot.
        func (function): for interpolating y in the interpolated x ranges (i.e., how to calculate a single
        representative y inside an x range)
    """

    console = Console()
    terminal_width = terminal_width or int(
        0.95 * console.size.width
    )  # Detect terminal width if not provided
    plot_width = int(
        terminal_width * (width_percentage - 0.05)
    )  # Width for the plot area

    x = np.array(x[:])
    y = np.array(y[:])

    if x.size == 0 or y.size == 0:
        return

    # Normalize y to fit within the height of the plot and determine y-axis units
    y_unit, y_scale = set_unit(y)
    y = y * y_scale
    y_min = min(y)
    y_max = max(y)
    y_scaled = np.interp(y, (y_min, y_max), (0, max_height)).astype(int)
    # Offset adjustment for the axis label indentation
    label_offset = len(f"{y_max:.2f}") + 1
    # print(f'y:{y}\n\nx:{x}\n\ny_scaled:{y_scaled}')

    # Normalize x to fit within the plot width
    x_min = min(x)
    x_max = max(x)
    x_scaled = np.interp(x, (x_min, x_max), (0, plot_width - label_offset - 1)).astype(
        int
    )

    # Create the plot grid ensuring no row exceeds plot width

    # Create the plot grid
    grid = []
    for level in range(max_height, -1, -1):
        row = []
        for i in range(plot_width - label_offset):
            idx = np.where(x_scaled == i)[0]
            # if i in x_scaled and y_scaled[idx[0]] >= level:
            if i in x_scaled and func(y_scaled[idx]) >= level:
                row.append("█")  # Bar character
            else:
                row.append(" ")
        grid.append("".join(row))

    # Create labels for start, middle, and end
    start_label = f"{x_min:.1f} sec"
    middle_label = f"{x[len(x) // 2]:.1f} sec"
    end_label = f"{x_max:.1f} sec"

    # Calculate label positions dynamically
    new_label_offset = len(f"{y_max:.2f} {y_unit}") + 2
    start_pos = 0
    middle_pos = (plot_width - new_label_offset) // 2 - len(middle_label) // 2
    end_pos = (plot_width - len(y_unit)) - len(end_label) - 1

    # Construct the label line
    label_line = [" "] * plot_width
    label_line[start_pos : start_pos + len(start_label)] = list(start_label)
    label_line[middle_pos : middle_pos + len(middle_label)] = list(middle_label)
    label_line[end_pos : end_pos + len(end_label)] = list(end_label)

    # Combine into final plot with 'Bytes' on y-axis
    plot_str = "\n".join(
        f"{y_min + (y_max - y_min) * (max_height - i) / max_height:{label_offset}.2f} {y_unit} │ {row}"
        for i, row in enumerate(grid)
    )

    # Adjust the alignment of the x-axis line and labels to account for the label_offset

    plot_str += f"\n{' ' * new_label_offset}└{'─' * (plot_width - len('└')-len(y_unit) -1)}>"  # Adjusted x-axis line
    plot_str += (
        f"\n{' ' * new_label_offset}{''.join(label_line)}"  # Adjusted x-axis label
    )
    plot_str = " " * (label_offset + 2 + len(y_unit)) + "^\n" + plot_str

    # Create a panel with the plot
    panel = Panel(
        plot_str,
        title="Bandwidth Plot",
        border_style="bold cyan",
        width=terminal_width,
    )

    # Display the plot
    console.print(panel)

    if flush_log and os.path.isfile(flush_log):
        entries, _ = parse_flush_log(flush_log)
        if entries:
            n_ftio = sum(1 for e in entries if "ftio" in e["label"].lower())
            n_post = len(entries) - n_ftio
            t_ftio = sum(
                e["copy_time"] + e["delete_time"]
                for e in entries
                if "ftio" in e["label"].lower()
            )
            t_post = sum(
                e["copy_time"] + e["delete_time"]
                for e in entries
                if "ftio" not in e["label"].lower()
            )
            summary = (
                f"Flushes: {n_ftio} FTIO-triggered ({t_ftio:.1f} s total), "
                f"{n_post} post-app ({t_post:.1f} s total)"
            )
            console.print(f"[dim]{summary}[/]")


def _load_bandwidth_json(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (t, b) arrays from a bandwidth JSON file."""
    with open(filename) as f:
        data = json.load(f)
    b = np.array(data.get("b", []))
    t = np.array(data.get("t", []))
    if b.size == 0 and t.size == 0:
        for sync_data in data.values():
            if isinstance(sync_data, dict) and "bandwidth" in sync_data:
                b = np.array(sync_data["bandwidth"].get("b_overlap_avr", []))
                t = np.array(sync_data["bandwidth"].get("t_overlap", []))
                break
    return t, b


def plot_flush_and_bandwidth(
    flush_log_path: str,
    t: np.ndarray,
    b: np.ndarray,
    show: bool = True,
    t_offset: float = 0.0,
) -> go.Figure:
    """Combined figure: Gantt flush timeline (top) + bandwidth (bottom), shared x-axis.

    Both axes use seconds from APP-START as the common reference.  If the
    bandwidth ``t`` array appears shifted relative to the flush events, pass
    ``t_offset`` (in seconds) to translate it: the plotted x will be
    ``t + t_offset``.

    Args:
        flush_log_path: path to the JIT flush.log file.
        t: time array from the bandwidth JSON (seconds from app start).
        b: bandwidth array (bytes/s) from the bandwidth JSON.
        show: open in browser when True.
        t_offset: seconds to add to ``t`` so it aligns with the flush log
            timeline (positive = shift bandwidth right).

    Returns:
        The combined Plotly Figure.
    """
    entries, events = parse_flush_log(flush_log_path)
    if not entries and not events:
        if os.path.isfile(flush_log_path):
            print(f"No parseable flush data in {flush_log_path}")
        return go.Figure()

    # Anchor t0 to APP-START so both axes share the same zero.
    app_start_events = [ev for ev in events if "START" in ev["event"]]
    if app_start_events:
        t0 = app_start_events[0]["ts"]
    else:
        all_ts = [e["start"] for e in entries] + [ev["ts"] for ev in events]
        t0 = min(all_ts)

    def to_sec(ts: datetime) -> float:
        return (ts - t0).total_seconds()

    n_gantt_rows = max(len({os.path.basename(e["src"]) for e in entries}), 1)
    gantt_h = max(80, 35 * n_gantt_rows + 50)
    bw_h = 220
    total_h = gantt_h + bw_h + 110  # title + legend + spacing

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[gantt_h / (gantt_h + bw_h), bw_h / (gantt_h + bw_h)],
        vertical_spacing=0.06,
    )

    # ── Gantt traces (row 1) ──────────────────────────────────────────────
    group_cfg = [
        ("FTIO-trigger", "#1f77b4", "rgba(31,119,180,0.35)"),
        ("post-app", "#e07b00", "rgba(224,123,0,0.35)"),
    ]
    first_in_legend: set[str] = set()
    for label, copy_color, del_color in group_cfg:
        group = [e for e in entries if label in e["label"]]
        if not group:
            continue
        names = [os.path.basename(e["src"]) for e in group]
        fig.add_trace(
            go.Bar(
                x=[e["copy_time"] for e in group],
                y=names,
                orientation="h",
                base=[to_sec(e["start"]) for e in group],
                name=label,
                legendgroup=label,
                showlegend=label not in first_in_legend,
                marker_color=copy_color,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Trigger: {label}<br>"
                    "Copy: %{x:.3f} s<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        first_in_legend.add(label)
        fig.add_trace(
            go.Bar(
                x=[e["delete_time"] for e in group],
                y=names,
                orientation="h",
                base=[to_sec(e["start"]) + e["copy_time"] for e in group],
                name=f"{label} (delete)",
                legendgroup=label,
                showlegend=False,
                marker_color=del_color,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Trigger: {label}<br>"
                    "Delete: %{x:.3f} s<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ── Bandwidth trace (row 2) ───────────────────────────────────────────
    y_unit, y_scale = set_unit(b)
    fig.add_trace(
        go.Scatter(
            x=np.array(t) + t_offset,
            y=np.array(b) * y_scale,
            mode="lines",
            line={"shape": "hv", "color": "#2ca02c", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.15)",
            name="Bandwidth",
        ),
        row=2,
        col=1,
    )

    # ── APP-START / APP-END lines across both subplots ────────────────────
    for ev in events:
        t_ev = to_sec(ev["ts"])
        is_start = "START" in ev["event"]
        fig.add_vline(
            x=t_ev,
            line_dash="dash",
            line_width=1.5,
            line_color="green" if is_start else "red",
            annotation_text=ev["event"],
            annotation_font_size=11,
            annotation_position="top right" if is_start else "top left",
        )

    fig.update_xaxes(title_text="Time from APP-START (s)", row=2, col=1)
    fig.update_yaxes(title_text=None, row=1, col=1)
    fig.update_yaxes(title_text=f"Bandwidth ({y_unit}/s)", row=2, col=1)

    fig.update_layout(
        title={
            "text": (f"Flush Timeline + Bandwidth — {os.path.basename(flush_log_path)}"),
            "font": {"size": 15},
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="overlay",
        bargap=0.5,
        height=total_h,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
        },
        margin={"l": 10, "r": 20, "t": 70, "b": 40},
    )
    fig = format_plot_and_ticks(fig, font_size=13, n_ticks=5, y_minor=False)

    if show:
        pio.show(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Load JSON data from multiple files and plot bandwidth vs. time."
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        default=["bandwidth.json"],
        help="The paths to the JSON files to plot. Multiple files can be provided.",
    )
    parser.add_argument(
        "-f",
        "--flush_log",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional path to a JIT flush.log file. "
        "When provided, a Gantt-style flush timeline is shown alongside the bandwidth plot.",
    )
    parser.add_argument(
        "-s",
        "--same_figure",
        action="store_true",
        default=False,
        help="Combine the flush timeline and bandwidth into a single figure with a "
        "shared x-axis (requires -f). Without -s the two plots open separately.",
    )
    parser.add_argument(
        "-o",
        "--t_offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Shift the bandwidth time axis by SECONDS to align it with the flush log "
        "(positive = shift right). Only used with -s/--same_figure.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save the figure to FILE (e.g. timeline.html) instead of opening a browser.",
    )
    args = parser.parse_args()

    if args.same_figure and args.flush_log:
        t, b = _load_bandwidth_json(args.filenames[0])
        fig = plot_flush_and_bandwidth(
            args.flush_log, t, b, show=args.save is None, t_offset=args.t_offset
        )
        if args.save and fig is not None:
            pio.write_html(fig, args.save)
            print(f"Saved combined figure to {args.save}")
    else:
        if args.same_figure and not args.flush_log:
            print("Warning: -s/--same_figure requires -f/--flush_log; ignored.")
        load_json_and_plot(args.filenames)
        if args.flush_log:
            fig = plot_flush_log(args.flush_log, show=args.save is None)
            if args.save and fig is not None:
                pio.write_html(fig, args.save)
                print(f"Saved flush timeline to {args.save}")


def flush_main():
    """Standalone entry point: plot_flush_log <path> [--save FILE]."""
    parser = argparse.ArgumentParser(
        description="Plot a JIT flush.log as a Gantt-style timeline."
    )
    parser.add_argument(
        "flush_log",
        type=str,
        help="Path to the flush.log file produced by the JIT staging layer.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save the figure to FILE instead of opening a browser.",
    )
    args = parser.parse_args()
    fig = plot_flush_log(args.flush_log, show=args.save is None)
    if args.save and fig is not None:
        pio.write_html(fig, args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
