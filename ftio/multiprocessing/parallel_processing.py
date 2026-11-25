"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mär 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def submit_tasks(worker_func, args_list, num_procs, use_futures):
    """
    Submit tasks to the parallel pool or ProcessPoolExecutor.

    :param worker_func: Function to be executed in parallel.
    :param args_list: List of tuples containing arguments for worker_func.
    :param num_procs: Number of processes to use.
    :param use_futures: If True, use ProcessPoolExecutor; otherwise, use multiprocessing.Pool.
    :return: A collection of futures or async results.
    """
    if use_futures:
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            return {executor.submit(worker_func, *args): args for args in args_list}
    else:
        with Pool(processes=num_procs) as pool:
            return [pool.apply_async(worker_func, args) for args in args_list]


def receive_results(task_results, total_tasks, progress, task, use_futures):
    """
    Receive results as tasks are completed and update progress.

    :param task_results: Futures or async results.
    :param total_tasks: Total number of tasks to process.
    :param progress: The Progress object used for tracking.
    :param task: The task being tracked in the progress.
    :param use_futures: A flag to indicate whether to use futures or results.
    :return: Two separate lists: processed_results and failed_results.
    """
    processed_results = []
    failed_results = []
    counter = 0

    if use_futures:
        # Handle futures (ProcessPoolExecutor)
        for future in as_completed(task_results):
            args = task_results[future]
            try:
                result = future.result(timeout=120)
            except TimeoutError:
                result = (None, *args, "[red bold] Timeout reached")

            if result[0]:
                processed_results.append(result)
            else:
                failed_results.append(result)

            counter += 1
            progress.console.print(f"Processed ({counter}/{total_tasks})")
            progress.update(task, completed=counter)
    else:
        # Handle async results (multiprocessing.Pool)
        for async_result in task_results:
            result = async_result.get()

            if result[0]:
                processed_results.append(result)
            else:
                failed_results.append(result)

            counter += 1
            progress.console.print(f"Processed ({counter}/{total_tasks})")
            progress.update(task, completed=counter)

    return processed_results, failed_results


def parallel_processing(worker_func, args_list, num_procs=-1, use_futures=True):
    """
    Process tasks in parallel using either multiprocessing.Pool or ProcessPoolExecutor.

    :param worker_func: Function to be executed in parallel.
    :param args_list: List of tuples containing arguments for worker_func.
    :param num_procs: Number of processes to use. Defaults to half of available CPUs.
    :param use_futures: If True, use ProcessPoolExecutor; otherwise, use multiprocessing.Pool.
    :return: Processed results and failed results.
    """
    if num_procs == -1:
        num_procs = max(1, cpu_count() // 2)

    total_tasks = len(args_list)
    processed_results = []
    failed_results = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description} ({task.completed}/{task.total})"
        ),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        "[yellow]-- runtime",
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Processing tasks", total=total_tasks)

        # Step 1: Submit tasks
        task_results = submit_tasks(worker_func, args_list, num_procs, use_futures)

        # Step 2: Receive results as they are completed
        processed_results, failed_results = receive_results(
            task_results, total_tasks, progress, task, use_futures
        )

    return processed_results, failed_results
