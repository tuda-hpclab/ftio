"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ftio.api.trace_analysis.trace_ftio_v2 import main as trace_ftio

console = Console()


def main(argv=sys.argv[1:]) -> None:
    verbose = False
    name = "plafrim"
    json_flag = False

    # specify the name with -n
    if "-n" in argv:
        index = argv.index("-n")
        name = str(argv[index + 1])
        argv.pop(index)
        argv.pop(index)
    if "-v" in argv:
        index = argv.index("-v")
        verbose = bool(argv[index + 1])
        argv.pop(index)
        argv.pop(index)
    if "-j" in argv:
        index = argv.index("-j")
        argv.pop(index)
        json_flag = True
    # print(argv)
    start_time = time.time()
    # pattern = "_signal_plafrim.csv"
    pattern = f"_signal_{name}.csv"
    df = pd.DataFrame()

    # Handle command-line arguments properly
    if len(argv) > 0:
        folder_path = argv[0]
        argv.pop(0)
    else:
        # folder_path = os.getcwd()
        folder_path = "/d/github/FTIO/ftio/api/trace_analysis/plafrim"

    folder_path = os.path.abspath(folder_path)
    console.print(f"[bold green]Path is: {folder_path}[/]")

    # Create a list to store all matching csv files
    csv_files = []

    # Walk through all subdirectories and find matching files
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(pattern):
                csv_files.append(os.path.join(dirpath, filename))

    if not csv_files:
        console.print(f"[bold red]No files matched the pattern: {pattern}![/]")
        return

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        "[yellow]--  elapsed time",
        TimeElapsedColumn(),
    )

    try:
        with progress:
            task = progress.add_task("[green]Processing files...", total=len(csv_files))

            # Iterate over each csv file
            for file_path in csv_files:
                # Display the file being processed
                progress.console.print(
                    f"Processing ({csv_files.index(file_path)}/{len(csv_files)}): {file_path}"
                )

                # Run the trace_ftio function
                res = trace_ftio([file_path] + argv, verbose, json_flag)

                # Create the new file name by replacing the pattern
                base_name = os.path.basename(file_path)
                json_name = base_name.replace(f"_signal_{name}.csv", f"_freq_{name}.json")
                json_path = os.path.join(os.path.dirname(file_path), json_name)
                data_converted = convert_dict(res)

                if json_name.endswith("json"):
                    with open(json_path, "w") as file:
                        json.dump(data_converted, file, indent=4)
                else:
                    console.print(f"[bold red]Cannot dump Json file in {json_path}[/]")

                flat_res = flatten_dict(res)
                try:
                    flat_res["job_id"] = base_name.split("_")[0]
                    flat_res["file"] = file_path
                except Exception:
                    flat_res["job_id"] = "??"
                    flat_res["file"] = "??"
                    console.print("[bold red]Unable to extract job id/path[/]")

                new_row_df = pd.DataFrame([flat_res])
                # Append the new row DataFrame to the existing DataFrame
                df = pd.concat([df, new_row_df], ignore_index=True)

                # Update the progress bar
                progress.advance(task)

        progress.console.print("[bold green]All files processed successfully![/]\n")
        console.print(
            f"[blue]FTIO total time:[/] {time.time()  - start_time:.4f} seconds\n"
            f"[blue]Location:[/] {folder_path}\n"
            f"[blue]Pattern:[/] {pattern}\n"
        )
        statistics(df)
    except KeyboardInterrupt:
        progress.console.print("[bold red]Keyboard interrupt![/]\n")
        statistics(df)
        sys.exit()
    console.print(f"[blue]Execution time:[/] {time.time()  - start_time:.4f} seconds")


def convert_dict(data):
    """Convert NumPy arrays and sets to lists in the dictionary."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, set):
                data[key] = list(value)
            elif isinstance(value, dict):
                convert_dict(value)
    return data


def flatten_dict(d):
    """Flatten the dictionary for DataFrame insertion."""
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    # Convert numpy arrays to lists
                    flat[f"{key}_{sub_key}"] = sub_value.tolist()
                elif isinstance(sub_value, set):
                    # Convert sets to lists
                    flat[f"{key}_{sub_key}"] = list(sub_value)
                else:
                    flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


def statistics(df, elapsed_time="", settings=None) -> None:
    # print(df)
    if settings is None:
        settings = {}
    df_dom = reduce_to_max_conf(df)
    prefixes = relevant_prefix(df)
    color = ["purple4", "gold3", "deep_sky_blue1"]
    if settings:
        path = settings["res_path"]
    else:
        path = "."

    content = ""
    with open(f"{path}/ftio_output.txt", "w") as file:
        for prefix in prefixes:
            s = ""
            s += periodic_apps(df, prefix)
            s += compute_metrics(df_dom, prefix, "conf")
            s += compute_metrics(df_dom, prefix, "dominant_freq", "Hz")
            s += time_app(df, prefix)
            console.print(
                Panel.fit(
                    s,
                    title=prefix.capitalize(),
                    border_style=color[prefixes.index(prefix)],
                    title_align="left",
                )
            )
            console.print("\n")
            content += cleaned_text(
                f"{prefix.capitalize()}" + "\n----------------\n" + s + "\n\n"
            )
        file.write(content + cleaned_text(elapsed_time))
        if settings:
            for _, field in enumerate(settings):
                file.write(f"\n{field}: {settings[field]}")

    df.to_csv(f"{path}/ftio.csv", index=False)
    df_dom.to_csv(f"{path}/ftio_flat.csv", index=False)
    # print(dom_df)


def periodic_apps(df, prefix) -> str:
    all_freq = len(df[f"{prefix}_dominant_freq"])
    # n = df[f'{prefix}_dominant_freq'].apply(lambda x: len(x)>0).sum()
    n = df[f"{prefix}_dominant_freq"].apply(lambda x: not np.isnan(x)).sum()
    # out = f"[blue]Periodic {prefix.capitalize()}: {n:.3e}/{all_freq:.3e} ({n/all_freq*100:.3e}%)[/]"
    out = f"[blue]Periodic I/O:[/]\n - {n:.0f}/{all_freq:.0f} ({n/all_freq*100:.2f}%)\n\n"
    return out


def compute_metrics(df: pd.DataFrame, prefix, suffix="conf", unit="%", title="") -> str:
    min = np.nan
    max = np.nan
    mean = np.nan
    median = np.nan
    nanmean = np.nan
    nanmedian = np.nan
    if not title:
        title = suffix.capitalize()

    conf_col = f"{prefix}_{suffix}"
    if not df[conf_col].isna().all():
        min = np.min(df[conf_col])
        max = np.max(df[conf_col])
        mean = np.mean(df[conf_col])
        median = np.median(df[conf_col])
        nanmean = np.nanmean(df[conf_col])
        nanmedian = np.nanmedian(df[conf_col])

    scale = 100 if "conf" in suffix else 1
    # out = f"[green]{prefix.capitalize()} {title}:\n - range: [{min*scale:.3e},{max*scale:.3e}] {unit}\n - mean: {mean*scale:.3e} {unit}\n - nanmean: {nanmean*scale:.3e} {unit}\n - median: {median*scale:.3e} {unit}\n - nanmedian: {nanmedian*scale:.3e} {unit}\n[/]"
    out = f"[gray][green]{title}:[/]\n - range: [{min*scale:.3f},{max*scale:.3f}] {unit}\n - mean: {mean*scale:.3f} {unit}\n - nanmean: {nanmean*scale:.3f} {unit}\n - median: {median*scale:.3f} {unit}\n - nanmedian: {nanmedian*scale:.3f} {unit}\n\n[/]"
    return out


def time_app(df, prefix):
    df[f"{prefix}_time"] = df[f"{prefix}_t_end"] - df[f"{prefix}_t_start"]
    out = compute_metrics(df, prefix, "time", "sec", "I/O Time")
    return out


def reduce_to_max_conf(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = relevant_prefix(df)
    print(prefixes)
    # Iterate over each row
    for i, row in df.iterrows():
        for prefix in prefixes:
            conf_col = f"{prefix}_conf"
            freq_col = f"{prefix}_dominant_freq"
            amp_col = f"{prefix}_amp"
            phi_col = f"{prefix}_phi"
            freq = np.nan
            conf = np.nan
            amp = np.nan
            phi = np.nan
            if isinstance(row[conf_col], list) and len(row[conf_col]) > 0:
                dominant_index = np.argmax(row[conf_col])
                freq = row[freq_col][dominant_index]
                conf = row[conf_col][dominant_index]
                amp = row[amp_col][dominant_index]
                phi = row[phi_col][dominant_index]
            df.at[i, freq_col] = freq
            df.at[i, conf_col] = conf
            df.at[i, amp_col] = amp
            df.at[i, phi_col] = phi

    return df


def relevant_prefix(df):
    prefixes = ["read", "write", "both"]
    res = []
    for mode in prefixes:
        matching_columns = df.columns[df.columns.str.contains(mode)]
        if len(matching_columns) > 0:
            res.append(mode)
    return res


def cleaned_text(text: str) -> str:
    # Remove color tags and placeholders for rich console
    text = re.sub(r"\[\w+\]", "", text)  # Remove [color] tags
    text = re.sub(r"\[\/\]", "", text)  # Remove [/] tags
    return text


if __name__ == "__main__":
    main(sys.argv[1:])
