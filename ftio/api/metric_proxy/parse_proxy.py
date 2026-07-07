"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mär 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import json
import re
import sys
from time import process_time

import numpy as np

from ftio.api.metric_proxy.req import MetricProxy
from ftio.freq.helper import MyConsole

CONSOLE = MyConsole()
CONSOLE.set(True)


def parse(
    file_path, match="proxy_component_critical_temperature_celcius"
) -> tuple[np.ndarray, np.ndarray]:
    b_out = np.array([])
    t_out = np.array([])
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return b_out, t_out
    except json.JSONDecodeError:
        print(
            f"Error: Unable to decode JSON from file '{file_path}'. Check if the file is valid JSON."
        )
        return b_out, t_out

    b_out, t_out = extract(json_data, match)

    if len(b_out) == 0:
        print("No match found. Exciting\n")
        exit(0)
    return b_out, t_out


def extract(json_data, match, verbose=False):
    b_out = np.array([])
    t_out = np.array([])
    for key, value in json_data.items():
        if isinstance(value, dict):
            b_out, t_out = extract(value, match, verbose)
            if len(b_out) > 0:
                break
        else:
            if match == key:
                if verbose:
                    print(f"matched {key}")
                x = np.array(value, dtype=float)
                x = np.nan_to_num(x=x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                t_out = x[:, 0]
                b_out = x[:, 1]
                # reduce to derivative
                if "deriv" not in key:
                    if verbose:
                        print("removing aggregation")
                    b_shifted = b_out[:-1]
                    b_shifted = np.insert(b_shifted, 0, 0)
                    # b_out = b_out - b_shifted
                    b_out = numerical_derivative(t_out, b_out)
                    break
    return b_out, t_out


def dewrap_bandwidth(size_xy, time_xy, ranks_xy=None):
    """Reconstruct the wall-clock bandwidth signal from cumulative
    ___size___/___time___ counter pairs.

    The proxy's MPI wrapper accounts a call's bytes and duration only when the
    call *returns*, so a burst appears as a spike at the completion sample.
    Here each burst is spread back over its estimated wall-clock span
    (delta time / concurrent ranks), so the burst start point is right and the
    amplitude is the aggregate bytes/s during the burst.

    Args:
        size_xy: cumulative size counter as [[ts, value], ...]
        time_xy: cumulative in-call time counter as [[ts, value], ...]
        ranks_xy: optional proxy_mpi_ranks series for the concurrency estimate

    Returns:
        (bandwidth, timestamps) as numpy arrays on the size counter's grid
    """
    s = np.nan_to_num(np.array(size_xy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(np.array(time_xy, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ts = s[:, 0]
    sv = s[:, 1]
    if len(ts) < 2:
        return np.zeros(len(ts)), ts

    def value_at(arr, x):
        """last value at or before x"""
        idx = int(np.searchsorted(arr[:, 0], x, side="right")) - 1
        return arr[idx, 1] if idx >= 0 else None

    ranks = None
    if ranks_xy is not None and len(ranks_xy) > 0:
        ranks = np.nan_to_num(np.array(ranks_xy, dtype=float), nan=0.0)

    ts0 = ts[0]
    # bytes landing in bin k, which spans (ts[k-1] .. ts[k]]
    bytes_bin = np.zeros(len(ts))
    for k in range(1, len(ts)):
        ds = sv[k] - sv[k - 1]
        if ds <= 0:
            continue
        cur = value_at(t, ts[k])
        prev = value_at(t, ts[k - 1])
        dt = (cur - prev) if cur is not None and prev is not None else 0.0
        n = 1.0
        if ranks is not None:
            r = value_at(ranks, ts[k])
            if r:
                n = max(1.0, round(r))
        start = max(ts[k] - dt / n, ts0)
        span = ts[k] - start
        if dt <= 1e-9 or span <= 1e-9:
            # no duration info: keep completion attribution
            bytes_bin[k] += ds
            continue
        # spread ds over the bins overlapping [start, ts[k]]
        j = max(int(np.searchsorted(ts, start, side="right")), 1)
        while j <= k:
            lo = max(ts[j - 1], start)
            hi = min(ts[j], ts[k])
            if hi > lo:
                bytes_bin[j] += ds * (hi - lo) / span
            j += 1

    # each value is the rate of the interval STARTING at its timestamp
    # (sample-and-hold until the next point); the last point closes at 0 so
    # bursts sit at their reconstructed start instead of one bin late
    bw = np.zeros(len(ts))
    widths = np.diff(ts)
    bw[:-1] = np.where(widths > 1e-9, bytes_bin[1:] / np.maximum(widths, 1e-9), 0.0)
    return bw, ts


def dewrap_metrics(json_data, out: dict, scale_t: float = 1) -> dict:
    """For each ___size___ metric with a matching ___time___ counter, add the
    dewrapped bandwidth reconstruction (see dewrap_bandwidth) on top of the
    existing metrics, under the ___bandwidth_dewrap___ name. That name matches
    the virtual metric the proxy trace UI offers, so the FTIO model lands on
    the metric the user plots. All other metrics are analyzed unchanged."""
    raw = json_data["metrics"]
    # prefer the job's own rank count (exact, follows malleable jobs)
    ranks_xy = raw.get("job_mpi_ranks") or raw.get("proxy_mpi_ranks")
    n_dw = 0
    for name in list(out.keys()):
        if "___size___" in name and not name.startswith("deriv"):
            tname = name.replace("___size___", "___time___")
            if name in raw and tname in raw:
                bw, ts = dewrap_bandwidth(raw[name], raw[tname], ranks_xy)
                out[name.replace("___size___", "___bandwidth_dewrap___")] = [
                    bw,
                    ts * scale_t,
                ]
                n_dw += 1
    CONSOLE.info(f"[blue]Dewrap: added {n_dw} reconstructed size/time metric pairs[/]")
    return out


def filter_metrics(
    json_data,
    filter_deriv: bool = True,
    exclude=None,
    scale_t: float = 1,
    rename: dict = None,
    dewrap: bool = False,
):
    if rename is None:
        rename = {}
    out = {}
    t = process_time()
    metrics = json_data["metrics"].keys()
    # extract either derive or all
    if filter_deriv:
        metrics = clean_metrics(metrics)

    if exclude:
        old_length = len(metrics)
        metrics = [metric for metric in metrics if all(n not in metric for n in exclude)]
        text = ", ".join([str(item) for item in exclude])
        CONSOLE.info(
            f"[blue]\nExcluded matches for: \\[{text}]\nMetrics reduced further from {old_length} to {len(metrics)}[/]"
        )

    for metric in metrics:
        b_out, t_out = extract(json_data, metric, False)
        out[metric] = [b_out, t_out * scale_t]

    if dewrap:
        out = dewrap_metrics(json_data, out, scale_t)

    # rename keys if only one metric passed
    if exclude:
        if "func" not in exclude:
            keys_to_rename = []
            for metric in out:
                if "func" in metric:
                    keys_to_rename.append(
                        (
                            metric,
                            "f_" + re.findall(r"func__(.*?)__", metric)[0],
                        )
                    )
            for old_key, new_key in keys_to_rename:
                original_new_key = new_key
                suffix = 1
                while new_key in out:
                    new_key = f"{original_new_key}_{suffix}"
                    suffix += 1
                out[new_key] = out.pop(old_key)

        if len({"time", "hits", "size"} & set(exclude)) == 2:
            keys_to_rename = [(metric, metric.rsplit("__", 1)[-1]) for metric in out]
            # Perform renaming after collecting all keys
            if keys_to_rename:
                CONSOLE.info(
                    f"[blue]\nRenaming Metrics: Removing [{keys_to_rename[-1][-1]}] from names[/]"
                )

            for old_key, new_key in keys_to_rename:
                out[new_key] = out.pop(old_key)

    elapsed_time = process_time() - t
    CONSOLE.info(f"[blue]Parsing time: {elapsed_time} s[/]")

    return out


def parse_all(
    file_path: str,
    filter_deriv: bool = True,
    exclude=None,
    scale_t: float = 1,
    dewrap: bool = False,
) -> dict:
    """parses all metrics from proxy

    Args:
        file_path (str): pass to proxy JSON file
        filter_deriv (bool, optional): Removes the metrics in case a similar metrics, which start with deriv is presented. Defaults to True.
        exclude (list,optional): list of metrics to exclude
        scale_t (float, optional): scale time unit (default 1). Default unit is "s"
        dewrap (bool, optional): reconstruct wall-clock bandwidth from size/time counter pairs (see dewrap_bandwidth)

    Returns:
        dict: parsed metrics with 2D numpy array
    """
    CONSOLE.info(f"\n[blue]Current file: {file_path}[/]")
    if scale_t != 1:
        CONSOLE.info(f"\n[yellow]Scaling time by: {scale_t}[/]")
    try:
        with open(file_path) as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(
            f"Error: Unable to decode JSON from file '{file_path}'. Check if the file is valid JSON."
        )
        return {}

    return filter_metrics(json_data, filter_deriv, exclude, scale_t, dewrap=dewrap)


def load_proxy_trace_stdin(deriv_and_not_deriv: bool = True, exclude=None):
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    else:
        return filter_metrics(data, deriv_and_not_deriv, exclude)


def clean_metrics(metrics: str):
    deriv_metrics = [metric for metric in metrics if metric.startswith("deriv")]
    non_deriv_metrics = [metric for metric in metrics if not metric.startswith("deriv")]
    cleaned_metrics = [
        metric
        for metric in metrics
        if not (metric in non_deriv_metrics and "deriv__" + metric in deriv_metrics)
    ]
    CONSOLE.info(
        f"[blue]Metrics reduced from {len(metrics)} to {len(cleaned_metrics)}[/]"
    )
    return cleaned_metrics


def get_all_metrics(job_id):
    mp = MetricProxy()
    metrics = {}
    all_metrics = mp.metric(job_id)
    for metric in all_metrics:
        value = mp.trace_metric(job_id, metric)
        t_out = np.array([item[0] for item in value])
        b_out = np.array([item[1] for item in value])
        # derive b
        b_out = numerical_derivative(t_out, b_out)
        metrics[metric] = [b_out, t_out]

    return metrics


# Calculate the numerical derivative using central differences
def numerical_derivative(t, f):
    n = len(t)
    df_dt = np.zeros(n)
    if n > 10:
        # Forward difference for the first point
        df_dt[0] = (f[1] - f[0]) / (t[1] - t[0])

        # Central differences for the interior points
        for i in range(1, n - 1):
            df_dt[i] = (f[i + 1] - f[i - 1]) / (t[i + 1] - t[i - 1])

        # Backward difference for the last point
        df_dt[-1] = (f[-1] - f[-2]) / (t[-1] - t[-2])
    return df_dt
