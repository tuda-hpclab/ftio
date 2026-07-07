"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Jul 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse
import difflib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count

import numpy as np

from ftio.api.metric_proxy.helper import create_process_bar, extract_data

# from rich.progress import Progress
from ftio.api.metric_proxy.parse_proxy import (
    filter_metrics,
    parse_all,
)
from ftio.api.metric_proxy.plot_proxy import (
    density_heatmap,
    heatmap,
    heatmap_2,
    scatter,
    scatter2D,
)
from ftio.api.metric_proxy.proxy_analysis import phases, phases_and_timeseries
from ftio.api.metric_proxy.proxy_cluster import dbscan, optics
from ftio.api.metric_proxy.req import MetricProxy
from ftio.freq.helper import MyConsole
from ftio.prediction.helper import print_data
from ftio.prediction.tasks import ftio_metric_task, ftio_metric_task_save


def parse_args():
    # file = '/d/github/FTIO/ftio/api/metric_proxy/traces/alberto_unito/bench_8x144.json'
    # file = '/d/github/FTIO/ftio/api/metric_proxy/traces/jb_traces/WACOM_PROCESS_BASED_json/wacommplusplus.36procs.trace.json'
    # file = '/d/sim/metric_proxy/traces/Mixed_1x8_5.json'file = /d/github/FTIO/ftio/api/metric_proxy/new_traces/imbio.json'
    # file = '/d/github/FTIO/ftio/api/metric_proxy/new_traces/imbio.json'
    # file = '/d/github/FTIO/ftio/api/metric_proxy/new_traces/wacom.json'
    file = "/d/github/FTIO/ftio/api/metric_proxy/new_traces/wacoml.json"
    # file = '/d/github/FTIO/ftio/api/metric_proxy/new_traces/lulesh_8_procs.json'
    # file = '/d/github/FTIO/ftio/api/metric_proxy/new_traces/lulesh_27_procs.json'

    parser = argparse.ArgumentParser(
        description="Executes FTIO in parallel on a JSON file from the proxy or by directly communicating with the proxy"
    )
    # settings in case the proxy is not running an an JSON file has been created
    parser.add_argument(
        "--file",
        type=str,
        nargs="?",  # '*' allows zero or more filenames
        default=file,
        help="The paths to the JSON file from the proxy",
    )

    # Settings in case the Proxy is running
    parser.add_argument(
        "--proxy",
        action="store_true",
        default=False,
        help="Weather to use the proxy HTTP endpoints or not",
    )
    parser.add_argument(
        "-j",
        "--job_id",
        type=str,
        default="",
        help="Job ID to use with the proxy",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="",
        help="applies FTIO to a specific metric",
    )

    # Other settings
    parser.add_argument(
        "--dewrap",
        action="store_true",
        default=False,
        help="reconstruct wall-clock bandwidth from size/time counter pairs "
        "(shifts each burst back over its estimated duration) instead of "
        "derivating the cumulative size counter",
    )
    parser.add_argument(
        "--disable_parallel",
        action="store_true",
        default=False,
        help="parallel or not",
    )
    parser.add_argument(
        "-S",
        "--sample_freq_proxy",
        type=float,
        default=1,
        help="Sample rate used in proxy",
    )
    parsed_args, unknown = parser.parse_known_args()
    parsed_args.ftio_args = unknown
    return parsed_args


def main(args: argparse.Namespace = parse_args()) -> None:
    start = time.time()
    console = MyConsole()
    console.set(True)
    ftio_args = args.ftio_args
    show = False  # shows the results from FTIO
    pools = False  # pools or future
    if args.metric:
        exclude = ["time", "hits", "proxy"]
    elif args.dewrap:
        # size metrics carry the dewrapped signal, so keep them
        exclude = ["time", "hits", "proxy"]
    else:
        exclude = ["size", "hits", "proxy"]

    if ftio_args and "-e" not in ftio_args:
        if args.metric:
            ftio_args.extend(["-e", "plotly"])
        else:
            ftio_args.extend(["-e", "no"])

    # pass -n <N> to use multi-frequency wave reconstruction (default: dominant freq only)

    console.print(f"FTIO args: {ftio_args}")
    if args.proxy:
        mp = MetricProxy()
        if not args.job_id:
            jobs = mp.jobs()
            job_id = jobs[0]["jobid"]
        else:
            job_id = args.job_id

        jsonl = mp.trace(job_id)
        metrics = filter_metrics(
            jsonl,
            filter_deriv=False,
            exclude=[],
            scale_t=1 / args.sample_freq_proxy,
            dewrap=args.dewrap,
        )

    else:
        metrics = parse_all(
            args.file,
            filter_deriv=True,
            exclude=exclude,
            scale_t=1 / args.sample_freq_proxy,
            dewrap=args.dewrap,
        )  # 'mpi'
        # metrics = parse_all(args.file , filter_deriv=True,exclude=['size','hits','proxy'], scale_t=1/args.sample_freq_proxy) # 'mpi'

    console.print(
        "[blue]\nSettings:\n---------[/]\n"
        f"[blue]- parallel: {not args.disable_parallel}[/]\n"
        f"[blue]- future: {not pools}[/]\n"
        f"[blue]- proxy: {args.proxy}[/]\n\n"
    )
    ranks = 32

    # plot the metrics if needed
    # _ = plot_timeseries_metrics(metrics, 2000,500)

    if args.metric:
        try:
            array = metrics[args.metric]

        except KeyError:
            avail_metrics = ""
            console.print(
                "[red]Error, unknown metric provided[/red]\nAvailable metrics are:"
            )
            for metric, _ in metrics.items():
                avail_metrics += f"{metric},"
            console.print(avail_metrics[:-1])
            # Find closest 3 matches
            close_matches = difflib.get_close_matches(args.metric, metrics.keys(), n=3)
            if close_matches:
                suggestions = ", ".join(close_matches)
                console.print(
                    f"\nDid you mean one of these?\n [green]{suggestions}[/green]"
                )

            exit(0)
        ftio_metric_task(args.metric, array, ftio_args, ranks, True)
    else:
        if not args.disable_parallel:
            data = execute_parallel(metrics, ftio_args, ranks, show, pools)
        else:
            data = execute(metrics, ftio_args, ranks, show)

        post(data, metrics, ftio_args)

    if "-v" in ftio_args:
        console.print(f"[cyan]Total elapsed time:[/] {time.time()-start:.3f} s\n")


def execute_parallel(
    metrics: dict,
    argv: list,
    ranks: int,
    show: bool = False,
    pools: bool = False,
):
    # parallel
    manager = Manager()
    data = manager.list()
    counter = 0
    total_metrics = len(metrics)
    progress = create_process_bar(total_metrics)
    task = progress.add_task("[green]Processing metrics", total=total_metrics)
    with progress:
        try:
            if pools:
                for metric, arrays in metrics.items():
                    with ProcessPoolExecutor() as executor:
                        _ = executor.submit(
                            ftio_metric_task_save,
                            data,
                            metric,
                            arrays,
                            argv,
                            ranks,
                            show,
                        )
                        counter += 1
                        progress.update(task, completed=counter)
            else:  # use futures
                with ProcessPoolExecutor(max_workers=cpu_count() - 2) as executor:
                    futures = {
                        executor.submit(
                            ftio_metric_task_save,
                            data,
                            metric,
                            arrays,
                            argv,
                            ranks,
                            show,
                        ): metric
                        for metric, arrays in metrics.items()
                    }
                    for _future in as_completed(futures):
                        counter += 1
                        progress.update(task, completed=counter)
        except KeyboardInterrupt:
            print_data(data)
            print("-- done -- ")
            exit()

    result = list(data)  # copy out before manager shuts down on scope exit
    manager.shutdown()
    return result


def execute(metrics: dict, argv: list, ranks: int, show: bool):
    data = []
    save = True
    check = True
    error_counter = 0
    counter = 0
    total_files = len(metrics)
    progress = create_process_bar(total_files)

    with progress:
        task = progress.add_task("[green]Processing metrics", total=total_files)
        for metric, arrays in metrics.items():
            if check:
                if len(arrays[1]) < 2:
                    error_counter += 1
                    console = MyConsole()
                    console.set(True)
                    console.print(
                        f"[bold red]- {error_counter}. Error in {metric}: [bold yellow]too few samples[/]"
                    )
                    continue
                increasing_order = np.all(arrays[1][-1] >= arrays[1][0])
                # negative = np.all(arrays[0] <= 0)
                if not increasing_order:  # or not  negative:
                    error_counter += 1
                    # err = '[bold red] Negative metric' if not negative else '[bold yellow]time not increasing'
                    err = "[bold yellow]time not increasing"
                    console = MyConsole()
                    console.set(True)
                    console.print(
                        f"[bold red]- {error_counter}. Error in {metric}:{err}[/]"
                    )
                    continue
            if save:
                # save stuff in queue, data is non empty
                ftio_metric_task_save(data, metric, arrays, argv, ranks, show)
            else:
                # skip saving
                ftio_metric_task(metric, arrays, argv, ranks, show)
            counter += 1
            progress.update(task, completed=counter)
    return data


def post(data, metrics, argv):
    if data:
        # print_data(data)
        phases(data, argv)
        phases_and_timeseries(metrics, data, argv)
        df = extract_data(data)
        heatmap(data)
        scatter2D(df)
        scatter(
            df,
            x="Phi",
            y="Dominant Frequency",
            color="Confidence",
            symbol="Metric",
        )
        _ = optics(df, "Phi", "Dominant Frequency")
        _ = dbscan(df, "Phi", "Dominant Frequency", 0.1)
        density_heatmap(data)
        heatmap_2(data)
    else:
        console = MyConsole()
        console.set(True)
        console.print("[bold red] No data[/]")


if __name__ == "__main__":
    args = parse_args()
    main(args)
